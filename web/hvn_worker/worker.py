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
