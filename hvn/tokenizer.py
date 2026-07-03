"""
BPE Tokenizer for HVN
=====================

A wrapper around HuggingFace's `tokenizers` library that provides
Byte-Pair Encoding (BPE) tokenization as a drop-in replacement for
CharTokenizer.

BPE gives two major advantages over character-level tokenization:
1. Better PPL — predicting subwords is easier than predicting characters
2. Faster training — fewer tokens = fewer forward passes

Usage:
    from hvn.tokenizer import BPETokenizer

    # Train on text
    tokenizer = BPETokenizer.train(text, vocab_size=1000)

    # Or load from file
    tokenizer = BPETokenizer.load("tokenizer.json")

    # Encode/decode
    ids = tokenizer.encode("The cat sat")
    text = tokenizer.decode(ids)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# Special tool-calling tokens (Issue #24)
from .tools import SPECIAL_TOKEN_STRINGS as TOOL_SPECIAL_TOKENS

# Special chat tokens (Issue #28)
from .chat import CHAT_SPECIAL_TOKEN_STRINGS as CHAT_SPECIAL_TOKENS


class BPETokenizer:
    """BPE tokenizer wrapper with CharTokenizer-compatible interface.

    Provides the same encode/decode/save/load API as CharTokenizer
    so it can be used as a drop-in replacement.
    """

    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 1000,
        use_byte_level: bool = True,
    ) -> "BPETokenizer":
        """Train a BPE tokenizer on the given text.

        Parameters
        ----------
        text : str
            Training text to learn BPE merges from.
        vocab_size : int
            Target vocabulary size (including special tokens).
        use_byte_level : bool
            If True, use byte-level BPE (handles any character, no <unk> needed).
            If False, use whitespace pre-tokenization (simpler but may produce <unk>).

        Returns
        -------
        tokenizer : BPETokenizer
        """
        if use_byte_level:
            # Byte-level BPE: handles any character without <unk>
            tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            tokenizer.decoder = ByteLevelDecoder()
            # Include tool-calling special tokens (Issue #24) + chat tokens (Issue #28)
            special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"] + TOOL_SPECIAL_TOKENS + CHAT_SPECIAL_TOKENS
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                show_progress=False,
                initial_alphabet=ByteLevel.alphabet(),
            )
        else:
            tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            tokenizer.pre_tokenizer = Whitespace()
            special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"] + TOOL_SPECIAL_TOKENS + CHAT_SPECIAL_TOKENS
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                show_progress=False,
            )

        # Train on the text
        tokenizer.train_from_iterator([text], trainer)
        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token IDs."""
        encoding = self._tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to text."""
        return self._tokenizer.decode(ids)

    def save(self, path: str):
        """Save the tokenizer to a JSON file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(save_path))

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load a tokenizer from a JSON file."""
        tokenizer = Tokenizer.from_file(path)
        return cls(tokenizer)


__all__ = ["BPETokenizer"]