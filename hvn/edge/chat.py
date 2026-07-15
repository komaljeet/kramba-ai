"""
EdgeHVN conversation formatter (issue #97 Phase B)
===================================================

``hvn.chat.ChatSession`` is architecture-agnostic (see its ``formatter``
constructor param) but its default ``ConversationFormatter`` speaks
``HamiltonianVortexNetwork``'s trained protocol: bracket tags
(``[SYSTEM]``/``[USER]``/``[ASSISTANT]``), closing-tag-terminated turns, a
trained system-prompt convention.

EdgeHVN was never trained on that format. Its actual trained protocol
(``finetune_edge_conversational.py``'s ``InstructionDataset`` /
``GroundedQADataset``) is::

    <|user|>
    {user text}
    <|assistant|>
    {assistant text}<|endoftext|>

EOS-terminated turns, no closing tag, and **no system-prompt convention was
ever trained** — the model never saw one. Rather than force-fitting the
mismatched bracket formatter (which would feed the model out-of-distribution
text it never learned to interpret), this module provides a small sibling
class duck-typed to the same interface ``ChatSession`` calls, so it plugs in
via ``ChatSession(model, tokenizer, formatter=EdgeConversationFormatter())``
with zero changes to ``ChatSession`` itself.

The three format constants below must match
``finetune_edge_conversational.py``'s ``USER_PREFIX``/``ASSISTANT_PREFIX``/
``EOS_ID`` exactly (duplicated rather than imported — the ``hvn`` package
doesn't depend on root-level training scripts; this mirrors how
``OPEN_BOOK_INSTRUCTION`` is already duplicated between ``continue_ft_hybrid.py``
and ``eval/synthetic_recall_probe.py`` in this codebase).
"""

from __future__ import annotations

from typing import Dict, List

# Must match finetune_edge_conversational.py exactly.
EOS_ID = 50256
USER_PREFIX = "<|user|>\n"
ASSISTANT_PREFIX = "<|assistant|>\n"


class EdgeConversationFormatter:
    """Format conversation turns in EdgeHVN's actual trained chat format.

    Duck-typed to the same interface as ``hvn.chat.ConversationFormatter``
    (``.system_prompt``, ``format_system``, ``format_user``,
    ``format_assistant``, ``format_assistant_open``, ``format_turn``,
    ``parse_response``, ``format_conversation``) so it's a drop-in swap via
    ``ChatSession``'s ``formatter`` param.
    """

    def __init__(self, system_prompt: str = "", add_newlines: bool = True):
        # Stored for interface parity with ConversationFormatter (ChatSession
        # reads .system_prompt directly in a few places), but NOT rendered
        # into the prompt -- see format_system below.
        self.system_prompt = system_prompt
        self.add_newlines = add_newlines

    def format_system(self, content: str) -> str:
        """No-op: EdgeHVN was never trained with a system-prompt convention.

        Rendering a fabricated system-turn format now would be untested,
        out-of-distribution input, not a real capability -- so a system
        prompt (if set) is simply not injected into the conversation.
        """
        return ""

    def format_user(self, content: str) -> str:
        s = f"{USER_PREFIX}{content}"
        if self.add_newlines:
            s += "\n"
        return s

    def format_assistant(self, content: str) -> str:
        s = f"{ASSISTANT_PREFIX}{content}"
        # EOS terminates the turn (matches training); no closing tag.
        return s

    def format_assistant_open(self) -> str:
        """The prefix that signals the model to start generating a response."""
        return ASSISTANT_PREFIX

    def format_turn(self, role: str, content: str) -> str:
        if role == "system":
            return self.format_system(content)
        elif role == "user":
            return self.format_user(content)
        elif role == "assistant":
            return self.format_assistant(content)
        else:
            raise ValueError(f"Unknown role: {role!r}. Expected 'system', 'user', or 'assistant'.")

    def parse_response(self, text: str) -> str:
        """Extract the assistant response from generated text.

        No closing tag to trim (EOS-terminated turns) -- EdgeHVN.chat_step
        already stops generation at EOS_ID before it reaches decoded text,
        so this is just a whitespace strip.
        """
        return text.strip()

    def format_conversation(self, turns: List[Dict[str, str]], include_system: bool = True) -> str:
        """Format a full conversation history. System prompt is never
        rendered (see format_system) regardless of `include_system`."""
        parts: List[str] = [self.format_turn(t["role"], t["content"]) for t in turns]
        return "".join(parts)
