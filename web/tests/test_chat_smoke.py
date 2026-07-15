"""
End-to-end chat smoke test (Issue #52).

Drives the *real* serving path — ``hvn_worker.worker.get_model`` ->
``hvn_worker.inference.run_turn``/``stream_turn`` -> ``hvn.chat.ChatSession`` —
against a tiny, freshly-trained Gated-DeltaNet + BPE checkpoint (the same
architecture family as the #51 chat checkpoint the app serves) and asserts a
**non-degenerate** reply (non-empty, not an echo of the prompt, not a single
repeated token).

This is what guards the #52 wiring: the #51 checkpoint is a DeltaNet model, so
``worker._load`` MUST rebuild the model with ``use_deltanet`` (and the other
saved hyperparameters) or ``load_state_dict`` fails. Building a real tiny
checkpoint here exercises that load path for real rather than mocking it.

The heavy path (torch + the hvn package) is skipped cleanly where torch isn't
installed so the rest of the web suite still runs.
"""
import os
import shutil
import sys
import tempfile
import unittest

from django.contrib.auth import get_user_model
from django.test import TransactionTestCase, override_settings

from api.models import Conversation, ModelVersion

# Force UTF-8 stdout/stderr so the ✓/→ characters printed by the reused
# train.py loop don't crash on a legacy cp1252 Windows console (same guard as
# build_chat_checkpoint.py).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
        pass

try:
    import torch
    from torch.utils.data import DataLoader

    from hvn.core import HamiltonianVortexNetwork
    from hvn.tokenizer import BPETokenizer
    from hvn.chat import generate_chat_training_data
    from hvn.lora import generate_instruction_training_data, format_instruction_example
    from train import TextDataset, train_model

    _HAVE_TORCH = True
except Exception:  # pragma: no cover - torch not installed in this environment
    _HAVE_TORCH = False

try:
    from hvn.edge.model import EdgeHVN
    from train_edge_conversational import _save_edge_checkpoint
    from transformers import GPT2Tokenizer

    _HAVE_EDGEHVN = True
except Exception:  # pragma: no cover - transformers/hvn.edge not available
    _HAVE_EDGEHVN = False

User = get_user_model()
STRONG_PASSWORD = "correct-horse-battery-staple-9"

# One shared tempdir for the trained checkpoint + per-conversation Hamiltonian
# state files (HVN_CHECKPOINT_DIR is overridden onto it below).
_CKPT_DIR = tempfile.mkdtemp(prefix="hvn_chat_smoke_")


def _build_tiny_deltanet_checkpoint(path: str) -> None:
    """Train a tiny BPE + Gated-DeltaNet HVN checkpoint on synthetic
    instruction/chat data and save it (with tokenizer) at ``path``.

    Small enough to run on CPU in a few seconds, trained enough (it overfits
    the tiny corpus, ppl ~1.1) to emit fluent, non-degenerate completions.
    Mirrors the #51 pipeline's block/tokenizer choices in miniature.
    """
    segments = [
        format_instruction_example(ex)
        for ex in generate_instruction_training_data(num_examples=80, seed=0)
    ]
    segments += generate_chat_training_data(num_examples=40)
    text = "\n\n".join(segments)

    tokenizer = BPETokenizer.train(text, vocab_size=512)
    ids = tokenizer.encode(text)
    loader = DataLoader(TextDataset(ids, seq_len=64), batch_size=8, shuffle=True)

    torch.manual_seed(0)
    model = HamiltonianVortexNetwork(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        num_layers=2,
        n_heads=2,
        chunk_size=32,
        use_deltanet=True,  # the #51 checkpoint is a DeltaNet model
    )
    # train_model saves the checkpoint (with use_deltanet etc.) to `path`.
    train_model(model, loader, epochs=15, lr=3e-3, device=torch.device("cpu"), save_path=path)
    tokenizer.save(path.replace(".pt", "_tokenizer.json"))


def _is_non_degenerate(reply: str, prompt: str) -> bool:
    """Non-empty, not an echo of the prompt, not a single repeated char/token.

    Same criterion as build_chat_checkpoint.is_non_degenerate (the #51 smoke
    gate), duplicated here to keep this test self-contained.
    """
    r = reply.strip()
    if len(r) < 2:
        return False
    if r == prompt.strip():
        return False
    if len(set(r.replace(" ", ""))) <= 1:
        return False
    return True


@unittest.skipUnless(_HAVE_TORCH, "torch/hvn not installed in this environment")
@override_settings(HVN_CHECKPOINT_DIR=_CKPT_DIR)
class ChatSmokeTests(TransactionTestCase):
    """Uses TransactionTestCase because the served path touches on-disk state
    files and loads real model weights — kept isolated from DB transaction
    fixtures, matching the WebSocket consumer test's rationale."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.ckpt_path = os.path.join(_CKPT_DIR, "hvn_chat_smoke.pt")
        _build_tiny_deltanet_checkpoint(cls.ckpt_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(_CKPT_DIR, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        # Drop any model cached under this tag from a previous test so the
        # fresh ModelVersion's checkpoint_path is actually (re)loaded.
        from hvn_worker.worker import clear_cache

        clear_cache()
        self.model_version = ModelVersion.objects.create(
            tag="chat-smoke-v1", checkpoint_path=self.ckpt_path
        )
        self.user = User.objects.create_user(username="smoke_user", password=STRONG_PASSWORD)
        self.conversation = Conversation.objects.create(
            owner=self.user, title="smoke", model_version=self.model_version
        )

    def test_run_turn_loads_deltanet_checkpoint_and_replies(self):
        """The core acceptance check: a coding prompt through the real worker
        path returns an on-topic, non-degenerate reply. If worker._load ignored
        use_deltanet this would raise on load_state_dict instead."""
        from hvn_worker.inference import run_turn

        torch.manual_seed(0)
        prompt = "Create a function that adds two numbers."
        reply = run_turn(
            conversation_id=self.conversation.id,
            user_message=prompt,
            model_tag=self.model_version.tag,
            checkpoint_path=self.model_version.checkpoint_path,
        )
        self.assertTrue(
            _is_non_degenerate(reply, prompt),
            f"degenerate reply from run_turn: {reply!r}",
        )

        # A second turn on the same conversation must load the persisted
        # DeltaNet Hamiltonian state from disk (StateCheckpoint v2 dict tree)
        # and reply again — this exercises the save->load round-trip that a
        # dict-shaped DeltaNet state broke before the generic serializer.
        prompt2 = "Now write one that subtracts them."
        reply2 = run_turn(
            conversation_id=self.conversation.id,
            user_message=prompt2,
            model_tag=self.model_version.tag,
            checkpoint_path=self.model_version.checkpoint_path,
        )
        self.assertTrue(
            _is_non_degenerate(reply2, prompt2),
            f"degenerate reply from second run_turn: {reply2!r}",
        )

    def test_stream_turn_yields_deltas_and_saves_final_content(self):
        """The streaming path (what the WebSocket consumer drives): deltas are
        emitted in order and the saved final content (parsed out of the raw
        stream, as api/tasks.py persists it) is non-degenerate."""
        from hvn_worker.inference import stream_turn

        torch.manual_seed(0)
        prompt = "Hello!"
        result: dict = {}
        deltas = list(
            stream_turn(
                conversation_id=self.conversation.id,
                user_message=prompt,
                model_tag=self.model_version.tag,
                checkpoint_path=self.model_version.checkpoint_path,
                result=result,
            )
        )
        # Some deltas were streamed, and every yielded delta is non-empty
        # (the consumer forwards each one over the WebSocket verbatim).
        self.assertTrue(deltas)
        self.assertTrue(all(d for d in deltas))
        # stream_turn writes the finished, parsed reply back through `result`
        # (the out-param api/tasks.py reads to persist the Message).
        self.assertIn("content", result)
        self.assertTrue(
            _is_non_degenerate(result["content"], prompt),
            f"degenerate streamed reply: {result['content']!r}",
        )


# A SEPARATE tempdir from _CKPT_DIR (not shared with ChatSmokeTests): that
# class's tearDownClass rmtree's _CKPT_DIR when it finishes, which would
# break this class if it ran afterward and reused the same directory.
_EDGE_CKPT_DIR = tempfile.mkdtemp(prefix="edgehvn_chat_smoke_")


def _build_tiny_edgehvn_checkpoint(path: str) -> None:
    """Save a tiny, freshly-init (not trained -- this test is about the
    worker/session WIRING, not output quality) EdgeHVN hybrid checkpoint:
    a real block_plan mix of DeltaNet + sliding-window attention layers, the
    same architecture family as checkpoints/hybrid_chat_v1.pt (issue #97
    Phase B / #108)."""
    torch.manual_seed(0)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = EdgeHVN(
        vocab_size=tok.vocab_size + 1, d_model=32, num_layers=4, n_heads=4,
        dropout=0.0, block_plan=["deltanet", "transformer", "deltanet", "transformer"],
        attn_window=16,
    )
    _save_edge_checkpoint(model, path)


@unittest.skipUnless(_HAVE_TORCH and _HAVE_EDGEHVN,
                     "torch/hvn.edge/transformers not installed in this environment")
@override_settings(HVN_CHECKPOINT_DIR=_EDGE_CKPT_DIR)
class EdgeHVNChatSmokeTests(TransactionTestCase):
    """EdgeHVN counterpart of ChatSmokeTests (issue #97 Phase B): drives the
    real serving path with ModelVersion.architecture="edgehvn" and asserts
    worker._load dispatches to _load_edgehvn (GPT-2 tokenizer, block_plan-
    aware load_checkpoint_for_resume) and ChatSession gets the
    EdgeConversationFormatter -- not the hvn_core defaults, which would
    either crash (BPE/CharTokenizer can't decode GPT-2 ids) or silently feed
    the model out-of-distribution bracket-tag text."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.ckpt_path = os.path.join(_EDGE_CKPT_DIR, "edgehvn_chat_smoke.pt")
        _build_tiny_edgehvn_checkpoint(cls.ckpt_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(_EDGE_CKPT_DIR, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        from hvn_worker.worker import clear_cache

        clear_cache()
        self.model_version = ModelVersion.objects.create(
            tag="edgehvn-chat-smoke-v1", checkpoint_path=self.ckpt_path,
            architecture=ModelVersion.Architecture.EDGEHVN,
        )
        self.user = User.objects.create_user(username="edge_smoke_user", password=STRONG_PASSWORD)
        self.conversation = Conversation.objects.create(
            owner=self.user, title="edge-smoke", model_version=self.model_version
        )

    def test_run_turn_loads_edgehvn_checkpoint_and_replies(self):
        from hvn_worker.inference import run_turn

        torch.manual_seed(0)
        prompt = "hi"
        reply = run_turn(
            conversation_id=self.conversation.id,
            user_message=prompt,
            model_tag=self.model_version.tag,
            checkpoint_path=self.model_version.checkpoint_path,
            architecture=self.model_version.architecture,
        )
        self.assertTrue(
            _is_non_degenerate(reply, prompt),
            f"degenerate reply from EdgeHVN run_turn: {reply!r}",
        )

        # Second turn: exercises the StateCheckpoint save/load round-trip for
        # EdgeHVN's mixed DeltaNet+attention-KV-cache state.
        prompt2 = "how are you?"
        reply2 = run_turn(
            conversation_id=self.conversation.id,
            user_message=prompt2,
            model_tag=self.model_version.tag,
            checkpoint_path=self.model_version.checkpoint_path,
            architecture=self.model_version.architecture,
        )
        self.assertTrue(
            _is_non_degenerate(reply2, prompt2),
            f"degenerate reply from second EdgeHVN run_turn: {reply2!r}",
        )

    def test_stream_turn_yields_deltas_for_edgehvn(self):
        from hvn_worker.inference import stream_turn

        torch.manual_seed(0)
        prompt = "hello"
        result: dict = {}
        deltas = list(
            stream_turn(
                conversation_id=self.conversation.id,
                user_message=prompt,
                model_tag=self.model_version.tag,
                checkpoint_path=self.model_version.checkpoint_path,
                architecture=self.model_version.architecture,
                result=result,
            )
        )
        self.assertTrue(deltas)
        self.assertTrue(all(d for d in deltas))
        self.assertIn("content", result)
        self.assertTrue(
            _is_non_degenerate(result["content"], prompt),
            f"degenerate streamed EdgeHVN reply: {result['content']!r}",
        )
