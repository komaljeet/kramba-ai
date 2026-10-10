"""EdgeHVN-native agentic tool-calling session (issue #110 / #97 Phase C).

The pre-existing ``hvn.tools.ToolCallingSession`` is bound to a
``model.generate_incremental(ids, ...) -> ids`` contract that doesn't exist
on ``EdgeHVN`` and isn't even a good fit for it (a full re-encode of the
whole transcript every turn, never touching the carried recurrent/KV-cache
state ``chat_step`` maintains). This module is a from-scratch, EdgeHVN-native
session built directly on ``EdgeHVN.forward_step`` -- true incremental
decode, matching how the model was actually fine-tuned to use tools
(``finetune_edge_conversational.py``'s ``ToolUseDataset``).

Reuses ``hvn.tools``'s ``ToolCallParser``/``ToolExecutor`` unchanged -- they
are pure text/JSON processing with no model coupling, already correct.

Loop (mirrors how ``ToolUseDataset`` renders training examples, so
inference matches training exactly):
  1. Feed the formatted prompt (``<|user|>...\\n<|assistant|>\\n``) through
     ``forward_step`` token by token, seeding the recurrent/KV-cache state.
  2. Sample tokens one at a time. After each token, check whether the
     decoded generated-so-far text now contains a NEW complete
     ``[TOOL_CALL]...[/TOOL_CALL]`` block (``ToolCallParser.parse_tool_calls``
     only matches balanced pairs -- an in-progress, unclosed call is
     correctly not matched yet).
  3. When a new call completes: execute it via ``ToolExecutor`` (real tool
     execution, real result), feed the resulting
     ``[TOOL_RESULT]...[/TOOL_RESULT]`` block through ``forward_step`` as
     CONTEXT (never sampled -- the model was never trained to produce this
     text, only to read it; see ``ToolUseDataset``'s masking), then resume
     sampling for the next tool call or the final answer.
  4. Stop on EOS, ``max_tool_calls`` exceeded (safety cap), or
     ``max_new_tokens`` exhausted.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..tools import ToolCallParser, ToolExecutor
from ..vocab import GPT2_EOS_ID, check_vocab_match, resolve_eos_id
from .chat import EdgeConversationFormatter

# Legacy alias. Kept so existing importers keep working, but nothing in this
# module reads it any more -- EOS is resolved per session from the model and
# tokenizer (#136). See hvn/vocab.py for why a module-level EOS is a bug.
EOS_ID = GPT2_EOS_ID


class EdgeToolCallingSession:
    """One-shot (not multi-turn-conversation) agentic tool-calling run
    against an EdgeHVN model: ``session.run(query)`` -> final answer text.

    Parameters
    ----------
    model : EdgeHVN
    tokenizer : object with .encode()/.decode() (GPT2Tokenizer in practice)
    executor : hvn.tools.ToolExecutor
        Already bound to a ToolRegistry (e.g. ``create_coding_registry``).
    device : torch.device
    max_new_tokens : int
        Total token budget across the whole run (prompt excluded), covering
        every tool-call segment plus the final answer.
    max_tool_calls : int
        Safety cap -- stop forcing more tool calls after this many (a
        genuinely runaway model calling tools forever should not hang).
    temperature, top_k, repetition_penalty : sampling params, same meaning
        as ``EdgeHVN.chat_step`` (this session implements the same
        sampling primitives directly rather than calling chat_step, since
        chat_step has no hook to pause generation mid-turn for tool
        execution).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        executor: ToolExecutor,
        device: Optional[torch.device] = None,
        max_new_tokens: int = 300,
        max_tool_calls: int = 5,
        temperature: float = 0.7,
        top_k: int = 10,
        repetition_penalty: float = 1.2,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.executor = executor
        self.device = device or next(model.parameters()).device
        self.max_new_tokens = max_new_tokens
        self.max_tool_calls = max_tool_calls
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.formatter = EdgeConversationFormatter()
        # #136: resolved per session, never read from a module constant. A 32K
        # model's EOS is 0; with 50256 it could never stop, so every run would
        # burn max_new_tokens while looking like a model that rambles.
        check_vocab_match(model, tokenizer, where="EdgeToolCallingSession")
        self.eos_id = resolve_eos_id(model, tokenizer)

    @torch.no_grad()
    def run(self, query: str) -> dict:
        """Run one query through the agentic tool-calling loop.

        Returns
        -------
        dict with keys:
          - ``text``: the full generated text (tool-call/result segments +
            final answer), for logging/debugging.
          - ``answer``: best-effort final answer -- the generated text with
            every ``[TOOL_CALL]...[/TOOL_RESULT]`` segment stripped.
          - ``tool_calls``: list of parsed tool calls actually executed, in
            order; each carries its ``result`` block.
          - ``num_tool_calls``: len(tool_calls), for the agentic gate (C6).
        """
        model = self.model
        device = self.device
        dtype = next(model.parameters()).dtype

        prompt_text = (self.formatter.format_user(query) + self.formatter.format_assistant_open())
        prompt_ids = self.tokenizer.encode(prompt_text)

        conv_state = model.init_conversation_state(device=device, dtype=dtype)
        states = conv_state["hamiltonian_states"]
        pos = conv_state["position"]

        cur = torch.empty(1, 1, dtype=torch.long, device=device)
        logits_step = None
        for tid in prompt_ids:
            cur.fill_(tid)
            logits_step, states = model.forward_step(cur, states, position_idx=pos)
            pos += 1

        generated_ids: list[int] = []
        seen_ids: set[int] = set(prompt_ids)
        tool_calls: list[dict] = []
        next_logits = logits_step[0, -1, :] / self.temperature
        in_call = False

        while len(generated_ids) < self.max_new_tokens:
            # #167: no penalty inside an unclosed [TOOL_CALL] block. Arguments
            # legitimately repeat tokens -- a filename copied from the query,
            # the 2 in "2+2" -- and the penalty pushed them to a near miss.
            penalty = 1.0 if in_call else self.repetition_penalty
            logits = model._apply_repetition_penalty(next_logits, seen_ids, penalty)
            logits = model._filter_top_k(logits, self.top_k)
            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

            if next_id == self.eos_id:
                break

            generated_ids.append(next_id)
            seen_ids.add(next_id)
            cur.fill_(next_id)
            logits_step, states = model.forward_step(cur, states, position_idx=pos)
            pos += 1
            next_logits = logits_step[0, -1, :] / self.temperature

            decoded_so_far = self.tokenizer.decode(generated_ids)
            in_call = ToolCallParser.has_tool_call_start(decoded_so_far)
            calls = ToolCallParser.parse_tool_calls(decoded_so_far)
            if len(calls) > len(tool_calls):
                if len(tool_calls) >= self.max_tool_calls:
                    break  # safety cap -- stop forcing more tool calls
                new_call = calls[len(tool_calls)]
                tool_calls.append(new_call)

                result_block = self.executor.execute(
                    {"name": new_call["name"], "arguments": new_call["arguments"]}
                )
                # Kept so a caller can tell a call that ran from one that
                # returned "Error: file not found" (#166).
                new_call["result"] = result_block
                result_ids = self.tokenizer.encode("\n" + result_block + "\n")
                # Feed the result as CONTEXT via forward_step -- never
                # sampled, matching ToolUseDataset's masking (the model was
                # never trained to generate this text).
                for rid in result_ids:
                    cur.fill_(rid)
                    logits_step, states = model.forward_step(cur, states, position_idx=pos)
                    pos += 1
                generated_ids.extend(result_ids)
                seen_ids.update(result_ids)
                next_logits = logits_step[0, -1, :] / self.temperature

        conv_state["hamiltonian_states"] = states
        conv_state["position"] = pos
        conv_state["history_ids"].extend(prompt_ids)
        conv_state["history_ids"].extend(generated_ids)

        text = self.tokenizer.decode(generated_ids)
        answer = self.formatter.parse_response(ToolCallParser.strip_tool_blocks(text))
        return {
            "text": text,
            "answer": answer,
            "tool_calls": tool_calls,
            "num_tool_calls": len(tool_calls),
        }
