"""
Loads an HVN checkpoint once per Celery worker process and caches it by
ModelVersion tag, so repeated tasks against the same conversation don't pay
the model-load cost every time. Reuses train.py's CharTokenizer (vendored
into the image alongside hvn/ - see web/Dockerfile) rather than duplicating
a tokenizer class here.
"""
import threading

import torch

from hvn.core import HamiltonianVortexNetwork
from train import CharTokenizer

_lock = threading.Lock()
_cache: dict[str, tuple] = {}


def _load(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = HamiltonianVortexNetwork(
        vocab_size=ckpt["vocab_size"],
        d_model=ckpt["d_model"],
        num_layers=ckpt["num_layers"],
        n_heads=ckpt.get("n_heads", 4),
        chunk_size=ckpt.get("chunk_size", 64),
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


def get_model(tag: str, checkpoint_path: str):
    """Returns (model, tokenizer, device) for the given ModelVersion tag,
    loading and caching it on first use in this worker process."""
    if tag in _cache:
        return _cache[tag]
    with _lock:
        if tag not in _cache:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _cache[tag] = (*_load(checkpoint_path, device), device)
    return _cache[tag]


def clear_cache() -> None:
    """Test-only: drop cached models between test runs that register
    different ModelVersion fixtures under the same tag."""
    with _lock:
        _cache.clear()
