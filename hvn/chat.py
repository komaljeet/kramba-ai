"""
Conversation Memory for HVN
==========================

Multi-turn chat with persistent Hamiltonian state — the HVN equivalent of a
chat interface.  The Hamiltonian memory carries context across turns at O(1)
per-token cost (no KV cache, no context reprocessing).

This module implements:
  - ``CHAT_TOKENS``           — special tokens for chat formatting
  - ``ConversationFormatter`` — formats conversation turns with special tokens
  - ``ChatSession``           — multi-turn chat with persistent Hamiltonian state
  - ``StateCheckpoint``       — save / load / fork conversation state to disk

Special tokens (added to the tokenizer):
  - ``[SYSTEM]``    / ``[/SYSTEM]``    — system prompt delimiters
  - ``[USER]``      / ``[/USER]``      — user message delimiters
  - ``[ASSISTANT]`` / ``[/ASSISTANT]`` — assistant response delimiters

Usage (standalone):
    from hvn.chat import ChatSession, ConversationFormatter

    session = ChatSession(model, tokenizer, device=device, system_prompt="You are helpful.")
    response = session.user_turn("Hello!")
    print(response)

    # Save and resume
    session.save("conversation.json")
    session2 = ChatSession.load("conversation.json", model, tokenizer, device)
    response2 = session2.user_turn("What did I just say?")

Design principles:
  - O(1) inference per token — Hamiltonian state is fixed d×d, constant in
    conversation length.
  - No KV cache — the Hamiltonian *is* the memory.
  - Context window management — old turns are truncated when the token
    history exceeds ``max_context_tokens``.
  - State checkpointing — save / load / fork conversations to disk.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Special chat tokens
# ---------------------------------------------------------------------------

CHAT_TOKENS: Dict[str, str] = {
    "system_start": "[SYSTEM]",
    "system_end": "[/SYSTEM]",
    "user_start": "[USER]",
    "user_end": "[/USER]",
    "assistant_start": "[ASSISTANT]",
    "assistant_end": "[/ASSISTANT]",
}

# All special token strings (for tokenizer registration)
CHAT_SPECIAL_TOKEN_STRINGS: List[str] = list(CHAT_TOKENS.values())


# ---------------------------------------------------------------------------
# Conversation Formatter
# ---------------------------------------------------------------------------

class ConversationFormatter:
    """Format conversation turns with special tokens for the model.

    Each turn is wrapped in role-specific delimiters so the model can
    distinguish user input from its own responses:

    .. code-block:: text

        [SYSTEM]You are a helpful assistant.[/SYSTEM]
        [USER]What is 2+2?[/USER]
        [ASSISTANT]2+2 equals 4.[/ASSISTANT]
        [USER]Thank you![/USER]
        [ASSISTANT]

    The trailing ``[ASSISTANT]`` (without closing tag) signals the model
    to generate a response.

    Parameters
    ----------
    system_prompt : str, optional
        Default system prompt prepended to every conversation.
    add_newlines : bool
        If True, add newlines between turns for readability.
    """

    def __init__(
        self,
        system_prompt: str = "",
        add_newlines: bool = True,
    ):
        self.system_prompt = system_prompt
        self.add_newlines = add_newlines

    def format_system(self, content: str) -> str:
        """Format a system message."""
        s = f"{CHAT_TOKENS['system_start']}{content}{CHAT_TOKENS['system_end']}"
        if self.add_newlines:
            s += "\n"
        return s

    def format_user(self, content: str) -> str:
        """Format a user message."""
        s = f"{CHAT_TOKENS['user_start']}{content}{CHAT_TOKENS['user_end']}"
        if self.add_newlines:
            s += "\n"
        return s

    def format_assistant(self, content: str) -> str:
        """Format an assistant response."""
        s = f"{CHAT_TOKENS['assistant_start']}{content}{CHAT_TOKENS['assistant_end']}"
        if self.add_newlines:
            s += "\n"
        return s

    def format_assistant_open(self) -> str:
        """Format the opening assistant tag (for generation prompt).

        Returns just ``[ASSISTANT]`` without a closing tag — this signals
        the model to start generating a response.
        """
        return CHAT_TOKENS["assistant_start"]

    def format_turn(
        self,
        role: str,
        content: str,
    ) -> str:
        """Format a single turn by role.

        Parameters
        ----------
        role : str
            One of "system", "user", "assistant".
        content : str
            The message content.

        Returns
        -------
        str
            Formatted turn string.
        """
        if role == "system":
            return self.format_system(content)
        elif role == "user":
            return self.format_user(content)
        elif role == "assistant":
            return self.format_assistant(content)
        else:
            raise ValueError(f"Unknown role: {role!r}. Expected 'system', 'user', or 'assistant'.")

    def format_conversation(
        self,
        turns: List[Dict[str, str]],
        include_system: bool = True,
    ) -> str:
        """Format a full conversation history.

        Parameters
        ----------
        turns : list of dict
            Each dict has "role" and "content" keys.
        include_system : bool
            If True and ``self.system_prompt`` is non-empty, prepend it.

        Returns
        -------
        str
            Full formatted conversation string.
        """
        parts: List[str] = []
        if include_system and self.system_prompt:
            parts.append(self.format_system(self.system_prompt))
        for turn in turns:
            parts.append(self.format_turn(turn["role"], turn["content"]))
        return "".join(parts)

    def format_for_generation(
        self,
        turns: List[Dict[str, str]],
        include_system: bool = True,
    ) -> str:
        """Format conversation for generation (with trailing [ASSISTANT] tag).

        Like ``format_conversation`` but appends ``[ASSISTANT]`` at the end
        to prompt the model to generate a response.

        Returns
        -------
        str
            Formatted conversation + ``[ASSISTANT]`` (no closing tag).
        """
        conv = self.format_conversation(turns, include_system=include_system)
        return conv + self.format_assistant_open()

    def strip_assistant_tags(self, text: str) -> str:
        """Remove [ASSISTANT] and [/ASSISTANT] tags from text."""
        text = text.replace(CHAT_TOKENS["assistant_start"], "")
        text = text.replace(CHAT_TOKENS["assistant_end"], "")
        return text.strip()

    def parse_response(self, text: str) -> str:
        """Extract the assistant response from generated text.

        The model generates text after the ``[ASSISTANT]`` tag.  This method
        extracts everything up to the ``[/ASSISTANT]`` tag (or the end of
        text if no closing tag is found).

        Parameters
        ----------
        text : str
            Generated text (may contain ``[/ASSISTANT]``).

        Returns
        -------
        str
            The cleaned assistant response.
        """
        # Remove any leading [ASSISTANT] tag (in case it's duplicated)
        text = text.lstrip()
        if text.startswith(CHAT_TOKENS["assistant_start"]):
            text = text[len(CHAT_TOKENS["assistant_start"]):]

        # Truncate at [/ASSISTANT] if present
        end_tag = CHAT_TOKENS["assistant_end"]
        if end_tag in text:
            text = text[:text.index(end_tag)]

        return text.strip()


# ---------------------------------------------------------------------------
# State Checkpoint
# ---------------------------------------------------------------------------

# Tag used to mark a serialized torch tensor inside the (otherwise
# JSON-native) conversation-state tree. Chosen not to collide with any state
# dict key the blocks use ("S"/"conv"/…).
_TENSOR_TAG = "__hvn_tensor__"


def _encode_state_tree(obj: Any) -> Any:
    """Recursively encode a conversation-state tree to JSON-native values.

    The per-block state shape differs by block type — the default HVN block
    stores ``list[head] -> list[Tensor]`` while the Gated-DeltaNet block
    stores ``{"S": Tensor, "conv": Tensor}`` — so serialization must be
    structure-agnostic rather than assuming a fixed nesting.  Tensors become
    ``{_TENSOR_TAG: <nested list>}``; dicts/lists recurse; scalars pass through.
    """
    if torch.is_tensor(obj):
        return {_TENSOR_TAG: obj.detach().cpu().tolist()}
    if isinstance(obj, dict):
        return {k: _encode_state_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode_state_tree(v) for v in obj]
    return obj


def _decode_state_tree(obj: Any, device: torch.device, dtype: torch.dtype) -> Any:
    """Inverse of :func:`_encode_state_tree` — rebuild tensors on ``device``."""
    if isinstance(obj, dict):
        if _TENSOR_TAG in obj:
            return torch.tensor(obj[_TENSOR_TAG], dtype=dtype, device=device)
        return {k: _decode_state_tree(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_state_tree(v, device, dtype) for v in obj]
    return obj


class StateCheckpoint:
    """Save / load / fork conversation state (Hamiltonian matrices) to disk.

    The Hamiltonian state is the core memory of the HVN model — a set of
    fixed-size d×d matrices (one per block, per head).  This class serializes
    them to a JSON file along with conversation metadata.

    Both block families are supported: the default HVN block (nested
    per-head tensor lists) and the Gated-DeltaNet block (per-block ``{"S",
    "conv"}`` dicts).  The on-disk format is versioned — v2 uses a generic,
    structure-agnostic tensor-tagged tree; v1 (legacy, default-block only) is
    still readable.

    Parameters
    ----------
    path : str
        File path for the checkpoint (``.json``).
    """

    def __init__(self, path: str):
        self.path = Path(path)

    def save(
        self,
        conv_state: dict,
        turns: List[Dict[str, str]],
        system_prompt: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save conversation state to disk.

        Parameters
        ----------
        conv_state : dict
            Conversation state from ``ChatSession.conv_state`` (contains
            Hamiltonian states, position, history_ids).
        turns : list of dict
            Conversation turns (role + content).
        system_prompt : str
            System prompt used for this conversation.
        metadata : dict, optional
            Additional metadata to save (e.g. model config, timestamp).
        """
        # Serialize the Hamiltonian state tree structure-agnostically so both
        # the default block (nested per-head tensor lists) and the DeltaNet
        # block (per-block {"S","conv"} dicts) round-trip — see
        # _encode_state_tree.
        hamiltonian_data = _encode_state_tree(conv_state["hamiltonian_states"])

        checkpoint = {
            "version": 2,
            "timestamp": time.time(),
            "system_prompt": system_prompt,
            "turns": turns,
            "conv_state": {
                "hamiltonian_states": hamiltonian_data,
                "position": conv_state["position"],
                "history_ids": conv_state["history_ids"],
            },
            "metadata": metadata or {},
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    def load(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[dict, List[Dict[str, str]], str, Dict[str, Any]]:
        """Load conversation state from disk.

        Parameters
        ----------
        device : torch.device
            Device to load Hamiltonian tensors onto.
        dtype : torch.dtype
            Data type for Hamiltonian tensors.

        Returns
        -------
        conv_state : dict
            Conversation state with Hamiltonian tensors on the given device.
        turns : list of dict
            Conversation turns.
        system_prompt : str
            System prompt from the checkpoint.
        metadata : dict
            Additional metadata from the checkpoint.
        """
        with open(self.path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        raw_states = checkpoint["conv_state"]["hamiltonian_states"]
        if checkpoint.get("version", 1) >= 2:
            # v2: generic tensor-tagged tree (any block type).
            hamiltonian_states = _decode_state_tree(raw_states, device, dtype)
        else:
            # v1 legacy: list of blocks -> list of heads -> list[Tensor]
            # (default HVN block only, no tensor tags).
            hamiltonian_states = []
            for block_data in raw_states:
                block_states = []
                for head_data in block_data:
                    head_states = [
                        torch.tensor(h_list, dtype=dtype, device=device)
                        for h_list in head_data
                    ]
                    block_states.append(head_states)
                hamiltonian_states.append(block_states)

        conv_state = {
            "hamiltonian_states": hamiltonian_states,
            "position": checkpoint["conv_state"]["position"],
            "history_ids": checkpoint["conv_state"]["history_ids"],
        }

        return (
            conv_state,
            checkpoint["turns"],
            checkpoint["system_prompt"],
            checkpoint.get("metadata", {}),
        )

    def fork(
        self,
        fork_path: str,
        after_turn: int,
    ) -> str:
        """Fork a conversation from a specific turn.

        Creates a new checkpoint containing only the first ``after_turn``
        turns (0-indexed: turns 0..after_turn-1 are kept).  The Hamiltonian
        state is re-derived by replaying the truncated conversation through
        the model.

        .. note::

           This method only copies the checkpoint file and truncates the
           turn history.  The Hamiltonian state is **not** recomputed here —
           use ``ChatSession.fork()`` for a full fork with correct state.

        Parameters
        ----------
        fork_path : str
            Path for the forked checkpoint.
        after_turn : int
            Keep turns 0..after_turn-1 (exclusive upper bound).

        Returns
        -------
        str
            Path to the forked checkpoint.
        """
        with open(self.path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        # Truncate turns
        checkpoint["turns"] = checkpoint["turns"][:after_turn]
        checkpoint["metadata"]["forked_from"] = str(self.path)
        checkpoint["metadata"]["fork_turn"] = after_turn

        fork_path_obj = Path(fork_path)
        fork_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(fork_path_obj, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        return str(fork_path_obj)


# ---------------------------------------------------------------------------
# Chat Session
# ---------------------------------------------------------------------------

class ChatSession:
    """Multi-turn chat with persistent Hamiltonian state.

    The Hamiltonian memory carries context across turns — O(1) per token,
    no KV cache.  Each user turn is fed through ``model.chat_step()``, which
    processes the input tokens incrementally (updating the Hamiltonian) and
    then generates a response.

    Key features:
      - **Persistent state**: Hamiltonian matrices are maintained across turns.
      - **Conversation formatting**: Special tokens for [USER]/[ASSISTANT]/[SYSTEM].
      - **Context window management**: Old turns are truncated when the token
        history exceeds ``max_context_tokens``.
      - **State checkpointing**: Save / load / fork conversations to disk.
      - **Interactive mode**: Terminal-based chat loop.

    Parameters
    ----------
    model : HamiltonianVortexNetwork
        The trained HVN model.
    tokenizer : object
        Tokenizer with ``encode()`` and ``decode()`` methods.
    device : torch.device
        Device to run on (GPU recommended).
    system_prompt : str
        System prompt prepended to the conversation.
    max_new_tokens : int
        Maximum tokens to generate per assistant turn.
    temperature : float
        Sampling temperature.
    top_k : int
        Top-k sampling (0 = disabled).
    max_context_tokens : int
        Maximum number of tokens to keep in history.  When exceeded, old
        turns are truncated (the Hamiltonian state is preserved — only the
        text history is trimmed for formatting purposes).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device | None = None,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 10,
        max_context_tokens: int = 2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens

        # Conversation formatter
        self.formatter = ConversationFormatter(system_prompt=system_prompt)

        # Conversation state
        self.conv_state: Optional[dict] = None
        self.turns: List[Dict[str, str]] = []
        self._init_state()

    def _init_state(self) -> None:
        """Initialize fresh conversation state."""
        self.conv_state = self.model.init_conversation_state(
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )
        self.turns = []

        # Feed system prompt through the model to initialize context
        if self.formatter.system_prompt:
            system_text = self.formatter.format_system(self.formatter.system_prompt)
            system_ids = self.tokenizer.encode(system_text)
            if system_ids:
                self._feed_tokens(system_ids)

    def _feed_tokens(self, token_ids: list[int]) -> None:
        """Feed token IDs through the model (updating Hamiltonian state).

        This does not generate — it only processes the tokens to update
        the Hamiltonian memory.  Used for system prompts and context
        management.

        Parameters
        ----------
        token_ids : list[int]
            Token IDs to process.
        """
        if not token_ids:
            return

        states = self.conv_state["hamiltonian_states"]
        pos = self.conv_state["position"]

        for tok_id in token_ids:
            tok = torch.tensor([[tok_id]], dtype=torch.long, device=self.device)
            logits_step, states = self.model.forward_step(tok, states, position_idx=pos)
            pos += 1

        self.conv_state["hamiltonian_states"] = states
        self.conv_state["position"] = pos
        self.conv_state["history_ids"].extend(token_ids)

    def _manage_context_window(self) -> None:
        """Truncate old turns when token history exceeds max_context_tokens.

        The Hamiltonian state is **not** reset — it retains all context.
        Only the text-level turn history is trimmed (for formatting and
        display purposes).  This prevents unbounded growth of the history
        list while preserving the O(1) Hamiltonian memory.
        """
        total_tokens = len(self.conv_state["history_ids"])
        if total_tokens <= self.max_context_tokens:
            return

        # Remove oldest turns until we're under the limit
        while len(self.turns) > 1 and len(self.conv_state["history_ids"]) > self.max_context_tokens:
            old_turn = self.turns.pop(0)
            # Estimate tokens for this turn (rough: chars ≈ tokens for char tokenizer)
            old_text = self.formatter.format_turn(old_turn["role"], old_turn["content"])
            old_ids = self.tokenizer.encode(old_text)
            # We can't remove from history_ids precisely (tokens are interleaved),
            # but we track the count for the limit check
            # The Hamiltonian state retains everything — this is just for display
            _ = old_ids  # tokens are already in the Hamiltonian

        # Note: history_ids keeps growing (it's a record), but the Hamiltonian
        # state is fixed-size.  We cap history_ids to prevent unbounded list growth.
        if len(self.conv_state["history_ids"]) > self.max_context_tokens * 2:
            # Keep only the recent half
            keep_from = len(self.conv_state["history_ids"]) - self.max_context_tokens
            self.conv_state["history_ids"] = self.conv_state["history_ids"][keep_from:]

    def user_turn(self, user_message: str) -> str:
        """Process a user message and generate an assistant response.

        This is the main chat method.  It:
        1. Formats the user message with [USER]...[/USER] tags.
        2. Feeds the formatted message through the model (updating Hamiltonian).
        3. Feeds the [ASSISTANT] opening tag (prompting generation).
        4. Generates the response using ``model.chat_step()``.
        5. Records the turn in history.
        6. Manages the context window.

        Parameters
        ----------
        user_message : str
            The user's input message.

        Returns
        -------
        str
            The assistant's response (cleaned, without tags).
        """
        # Format user message + [ASSISTANT] opening tag
        user_formatted = self.formatter.format_user(user_message)
        user_ids = self.tokenizer.encode(user_formatted)

        assistant_open = self.formatter.format_assistant_open()
        assistant_open_ids = self.tokenizer.encode(assistant_open)

        # Pass all input tokens (user message + assistant open tag) to chat_step.
        # chat_step feeds them through forward_step (updating Hamiltonian state)
        # and then generates the response.
        all_input_ids = user_ids + assistant_open_ids
        response_ids, self.conv_state = self.model.chat_step(
            input_ids=all_input_ids,
            conv_state=self.conv_state,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            device=self.device,
        )

        # Decode response
        response_text = self.tokenizer.decode(response_ids)
        response_clean = self.formatter.parse_response(response_text)

        # Record turns
        self.turns.append({"role": "user", "content": user_message})
        self.turns.append({"role": "assistant", "content": response_clean})

        # Manage context window
        self._manage_context_window()

        return response_clean

    def user_turn_streaming(self, user_message: str):
        """Generator variant of ``user_turn()`` — yields incremental text
        deltas as the assistant response is generated, token by token, then
        updates ``conv_state``/``turns`` exactly like ``user_turn()`` once
        generation completes (Issue #30's WebSocket streaming needs this;
        ``user_turn()`` only returns the finished response).

        Decodes the full generated-id sequence at each step and yields only
        the new suffix, the same incremental-decode technique
        ``stream_text_chunks`` in ``hvn/server.py`` uses — this avoids
        emitting malformed partial-BPE-token text.

        Yields
        ------
        str
            Non-empty text deltas, in order.
        """
        user_formatted = self.formatter.format_user(user_message)
        user_ids = self.tokenizer.encode(user_formatted)
        assistant_open = self.formatter.format_assistant_open()
        assistant_open_ids = self.tokenizer.encode(assistant_open)
        all_input_ids = user_ids + assistant_open_ids

        with torch.no_grad():
            states = self.conv_state["hamiltonian_states"]
            pos = self.conv_state["position"]

            logits_step = None
            for tok_id in all_input_ids:
                tok = torch.tensor([[tok_id]], dtype=torch.long, device=self.device)
                logits_step, states = self.model.forward_step(tok, states, position_idx=pos)
                pos += 1

            temp = max(self.temperature, 1e-5)
            next_logits = logits_step[0, -1, :] / temp

            response_ids: list[int] = []
            prev_text = ""
            for _ in range(self.max_new_tokens):
                if self.top_k > 0:
                    k = min(self.top_k, next_logits.size(-1))
                    values, _ = torch.topk(next_logits, k)
                    next_logits = next_logits.clone()
                    next_logits[next_logits < values[-1]] = float("-inf")

                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
                response_ids.append(next_id)

                full_text = self.tokenizer.decode(response_ids)
                delta = full_text[len(prev_text):]
                prev_text = full_text
                if delta:
                    yield delta

                tok = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                logits_step, states = self.model.forward_step(tok, states, position_idx=pos)
                pos += 1
                next_logits = logits_step[0, -1, :] / temp

            self.conv_state["hamiltonian_states"] = states
            self.conv_state["position"] = pos
            self.conv_state["history_ids"].extend(all_input_ids)
            self.conv_state["history_ids"].extend(response_ids)

        response_clean = self.formatter.parse_response(prev_text)
        self.turns.append({"role": "user", "content": user_message})
        self.turns.append({"role": "assistant", "content": response_clean})
        self._manage_context_window()

    def add_system_message(self, system_message: str) -> None:
        """Add a system message to the conversation.

        System messages are fed through the model (updating the Hamiltonian)
        but are not added to the turn history for display purposes.

        Parameters
        ----------
        system_message : str
            The system message content.
        """
        system_formatted = self.formatter.format_system(system_message)
        system_ids = self.tokenizer.encode(system_formatted)
        self._feed_tokens(system_ids)

    def get_history(self) -> List[Dict[str, str]]:
        """Return the conversation history (list of turns).

        Returns
        -------
        list of dict
            Each dict has "role" and "content" keys.
        """
        return list(self.turns)

    def get_formatted_history(self) -> str:
        """Return the formatted conversation history as a string.

        Returns
        -------
        str
            Full formatted conversation (with special tokens).
        """
        return self.formatter.format_conversation(self.turns)

    def reset(self) -> None:
        """Reset the conversation — clear state and history."""
        self._init_state()

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save conversation state to disk.

        Parameters
        ----------
        path : str
            File path (``.json``).
        metadata : dict, optional
            Additional metadata (e.g. model config, label).
        """
        ckpt = StateCheckpoint(path)
        ckpt.save(
            conv_state=self.conv_state,
            turns=self.turns,
            system_prompt=self.formatter.system_prompt,
            metadata=metadata,
        )

    @classmethod
    def load(
        cls,
        path: str,
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> "ChatSession":
        """Load a conversation from disk and resume.

        Parameters
        ----------
        path : str
            Checkpoint file path (``.json``).
        model : HamiltonianVortexNetwork
            The trained HVN model.
        tokenizer : object
            Tokenizer with ``encode()``/``decode()`` methods.
        device : torch.device
            Device to load tensors onto.
        **kwargs
            Additional arguments for ChatSession (max_new_tokens, etc.).

        Returns
        -------
        ChatSession
            A ChatSession with the loaded conversation state.
        """
        dev = device or next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        ckpt = StateCheckpoint(path)
        conv_state, turns, system_prompt, metadata = ckpt.load(dev, dtype)

        session = cls(
            model=model,
            tokenizer=tokenizer,
            device=dev,
            system_prompt=system_prompt,
            **kwargs,
        )
        # Replace the freshly-initialized state with the loaded state
        session.conv_state = conv_state
        session.turns = turns
        return session

    def fork(self, path: str, after_turn: int) -> "ChatSession":
        """Fork the conversation from a specific turn.

        Creates a new ChatSession containing only the first ``after_turn``
        turns.  The Hamiltonian state is recomputed by replaying the
        truncated conversation through the model.

        Parameters
        ----------
        path : str
            Path to save the forked checkpoint.
        after_turn : int
            Keep turns 0..after_turn-1 (exclusive upper bound).
            Must be <= len(self.turns).

        Returns
        -------
        ChatSession
            A new ChatSession with the forked state.
        """
        if after_turn > len(self.turns):
            raise ValueError(
                f"after_turn ({after_turn}) exceeds number of turns ({len(self.turns)})"
            )

        # Create a new session with the same config
        forked = ChatSession(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            system_prompt=self.formatter.system_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            max_context_tokens=self.max_context_tokens,
        )

        # Replay the truncated turns through the model
        for turn in self.turns[:after_turn]:
            if turn["role"] == "user":
                # Format and feed user message
                user_formatted = forked.formatter.format_user(turn["content"])
                user_ids = forked.tokenizer.encode(user_formatted)
                forked._feed_tokens(user_ids)

                # Feed [ASSISTANT] tag
                assistant_open = forked.formatter.format_assistant_open()
                assistant_open_ids = forked.tokenizer.encode(assistant_open)
                forked._feed_tokens(assistant_open_ids)

                # Feed assistant response
                assistant_formatted = forked.formatter.format_assistant(turn["content"])
                assistant_ids = forked.tokenizer.encode(assistant_formatted)
                forked._feed_tokens(assistant_ids)
            elif turn["role"] == "assistant":
                # In case turns are not paired (e.g. after_turn is odd)
                assistant_formatted = forked.formatter.format_assistant(turn["content"])
                assistant_ids = forked.tokenizer.encode(assistant_formatted)
                forked._feed_tokens(assistant_ids)

        forked.turns = list(self.turns[:after_turn])

        # Save the forked state
        forked.save(path, metadata={"forked_from": "in-memory", "fork_turn": after_turn})

        return forked

    def interactive(self, prompt_str: str = "You> ") -> None:
        """Run an interactive chat loop in the terminal.

        Reads user input from stdin, generates responses, and prints them.
        Type 'quit', 'exit', or '/bye' to end the conversation.
        Type '/save <path>' to save the conversation.
        Type '/reset' to reset the conversation.
        Type '/history' to print the conversation history.

        Parameters
        ----------
        prompt_str : str
            The prompt string to show for user input.
        """
        print("=" * 60)
        print("HVN Chat Session (Issue #28)")
        print("Type 'quit' or 'exit' to end. Commands: /save, /load, /reset, /history, /fork")
        print("=" * 60)

        if self.formatter.system_prompt:
            print(f"[System: {self.formatter.system_prompt}]")
            print("-" * 60)

        while True:
            try:
                user_input = input(prompt_str).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Commands
            if user_input.lower() in ("quit", "exit", "/bye"):
                print("Goodbye!")
                break
            elif user_input.startswith("/save"):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: /save <path>")
                    continue
                self.save(parts[1])
                print(f"✓ Conversation saved to {parts[1]}")
                continue
            elif user_input.startswith("/load"):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: /load <path>")
                    continue
                try:
                    loaded = ChatSession.load(
                        parts[1], self.model, self.tokenizer, self.device,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        max_context_tokens=self.max_context_tokens,
                    )
                    self.conv_state = loaded.conv_state
                    self.turns = loaded.turns
                    self.formatter.system_prompt = loaded.formatter.system_prompt
                    print(f"✓ Conversation loaded from {parts[1]}")
                except Exception as e:
                    print(f"✗ Load failed: {e}")
                continue
            elif user_input == "/reset":
                self.reset()
                print("✓ Conversation reset")
                continue
            elif user_input == "/history":
                print("-" * 60)
                print(self.get_formatted_history())
                print("-" * 60)
                continue
            elif user_input.startswith("/fork"):
                parts = user_input.split()
                if len(parts) < 3:
                    print("Usage: /fork <path> <after_turn>")
                    continue
                try:
                    fork_path = parts[1]
                    after_turn = int(parts[2])
                    forked = self.fork(fork_path, after_turn)
                    print(f"✓ Forked conversation (turns 0..{after_turn-1}) saved to {fork_path}")
                except Exception as e:
                    print(f"✗ Fork failed: {e}")
                continue

            # Normal conversation turn
            print("Assistant> ", end="", flush=True)
            t0 = time.time()
            response = self.user_turn(user_input)
            elapsed = time.time() - t0
            print(response)
            print(f"  [{elapsed:.2f}s]", flush=True)


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def generate_chat_training_data(
    num_examples: int = 50,
    system_prompts: Optional[List[str]] = None,
) -> List[str]:
    """Generate synthetic chat-formatted training examples.

    Produces conversations with [SYSTEM]/[USER]/[ASSISTANT] tags so the
    model learns the chat protocol during training.

    Parameters
    ----------
    num_examples : int
        Number of training examples to generate.
    system_prompts : list of str, optional
        System prompts to use.  If None, uses defaults.

    Returns
    -------
    list of str
        Formatted training examples (each a multi-turn conversation string).
    """
    if system_prompts is None:
        system_prompts = [
            "You are a helpful assistant.",
            "You are a knowledgeable tutor.",
            "You are a creative writer.",
            "You are a concise technical advisor.",
        ]

    # Simple Q&A pairs for synthetic training data
    qa_pairs = [
        ("What is the Hamiltonian Vortex Network?", "The HVN is a novel architecture that replaces attention with a fixed-size Hamiltonian matrix."),
        ("How does O(1) inference work?", "The Hamiltonian state is a fixed d×d matrix that summarizes all past tokens, so each new token costs O(d²) regardless of sequence length."),
        ("What is the Biot-Savart law in HVN?", "It computes the velocity field induced by the Hamiltonian on a new token's vorticity vector: v_t = Gamma * (H * omega_t)."),
        ("What is entanglement renormalization?", "It updates the Hamiltonian by absorbing new vortex vectors while truncating noise modes via SVD."),
        ("What are the advantages of HVN?", "HVN uses 43% fewer parameters than Transformers and maintains O(1) inference memory regardless of sequence length."),
        ("What is the bond dimension?", "The bond dimension is the SVD truncation rank that controls how many entangled modes are preserved in the Hamiltonian."),
        ("How does multi-thread memory work?", "Multiple independent Hamiltonians per head provide K different views of the sequence, improving expressiveness."),
        ("What is RoPE in HVN?", "Rotary positional embeddings are applied in the vortex injection layer to encode token positions."),
        ("Can HVN be quantized?", "Yes, the Hamiltonian matrices can be quantized to INT8 or NF4 4-bit with minimal quality loss."),
        ("What is the scaling exponent of HVN?", "HVN's PPL scales as params^-0.007, which is nearly flat — the main open research challenge."),
    ]

    import random

    examples = []
    for i in range(num_examples):
        system = random.choice(system_prompts)
        formatter = ConversationFormatter(system_prompt=system)

        # 1-3 turn conversations
        n_turns = random.randint(1, 3)
        turns = []
        parts = [formatter.format_system(system)]

        for t in range(n_turns):
            q, a = random.choice(qa_pairs)
            # Vary the questions slightly
            if t > 0:
                # Add follow-up context
                q = f"Following up: {q}"
            turns.append({"role": "user", "content": q})
            turns.append({"role": "assistant", "content": a})
            parts.append(formatter.format_user(q))
            parts.append(formatter.format_assistant(a))

        examples.append("".join(parts))

    return examples