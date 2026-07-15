"""
Loads an HVN checkpoint once per Celery worker process and caches it by
ModelVersion tag, so repeated tasks against the same conversation don't pay
the model-load cost every time. Reuses train.py's CharTokenizer (vendored
into the image alongside hvn/ - see web/Dockerfile) rather than duplicating
a tokenizer class here.

Two architectures are supported (ModelVersion.architecture, issue #97 Phase
B): "hvn_core" (the original HamiltonianVortexNetwork, BPE/Char tokenizer)
and "edgehvn" (hvn/edge/model.py's EdgeHVN, GPT-2 tokenizer). `_load`
branches on it; everything downstream (ChatSession, StateCheckpoint) is
already architecture-agnostic (see hvn/chat.py).
"""
import threading

import torch

from hvn.core import HamiltonianVortexNetwork
from train import CharTokenizer

_lock = threading.Lock()
_cache: dict[str, tuple] = {}


def _load_hvn_core(checkpoint_path: str, device: torch.device, ckpt: dict):
    # Rebuild with the SAME hyperparameters the checkpoint was trained with -
    # in particular use_deltanet/deltanet_conv_kernel (Issue #34) and the
    # weightless projections (Issue #40). The #51 chat checkpoint is a
    # Gated-DeltaNet model, so omitting these flags builds the wrong block and
    # load_state_dict fails on missing/unexpected keys. Mirrors generate.py's
    # load_model so the worker serves exactly what generate.py / the #51
    # pipeline produced.
    model = HamiltonianVortexNetwork(
        vocab_size=ckpt["vocab_size"],
        d_model=ckpt["d_model"],
        num_layers=ckpt["num_layers"],
        n_heads=ckpt.get("n_heads", 4),
        chunk_size=ckpt.get("chunk_size", 64),
        use_deltanet=ckpt.get("use_deltanet", False),
        deltanet_conv_kernel=ckpt.get("deltanet_conv_kernel", 4),
        weightless=ckpt.get("weightless", False),
        weightless_seed=ckpt.get("weightless_seed", 1234),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Same auto-detect convention as generate.py: try BPE first, fall back
    # to CharTokenizer, both loaded from `<checkpoint>_tokenizer.json`.
    tokenizer_path = checkpoint_path.replace(".pt", "_tokenizer.json")
    try:
        from hvn.tokenizer import BPETokenizer

        tokenizer = BPETokenizer.load(tokenizer_path)
    except Exception:
        tokenizer = CharTokenizer.load(tokenizer_path)

    return model, tokenizer


def _load_edgehvn(checkpoint_path: str, device: torch.device):
    # load_checkpoint_for_resume is block_plan-aware -- rebuilds the hybrid
    # architecture (DeltaNet + sliding-window attention layers) exactly as
    # trained and loads weights strict=False with a matched/missing/
    # unexpected key report, the same path finetune_edge_conversational.py
    # uses to resume a hybrid checkpoint for further training.
    from train_edge_conversational import load_checkpoint_for_resume

    model, _step, _meta = load_checkpoint_for_resume(checkpoint_path, device)
    model.eval()

    # GPT-2 tokenizer, not BPE/CharTokenizer -- what EdgeHVN was actually
    # trained/evaluated with throughout (continue_ft_hybrid.py,
    # finetune_edge_conversational.py, eval/synthetic_recall_probe.py).
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _load(checkpoint_path: str, device: torch.device, architecture: str = "hvn_core"):
    if architecture == "edgehvn":
        return _load_edgehvn(checkpoint_path, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return _load_hvn_core(checkpoint_path, device, ckpt)


def get_model(tag: str, checkpoint_path: str, architecture: str = "hvn_core"):
    """Returns (model, tokenizer, device) for the given ModelVersion tag,
    loading and caching it on first use in this worker process."""
    if tag in _cache:
        return _cache[tag]
    with _lock:
        if tag not in _cache:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _cache[tag] = (*_load(checkpoint_path, device, architecture), device)
    return _cache[tag]


def clear_cache() -> None:
    """Test-only: drop cached models between test runs that register
    different ModelVersion fixtures under the same tag."""
    with _lock:
        _cache.clear()
