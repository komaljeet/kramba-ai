"""The megakernel backward must not synchronise the host with the GPU.

Author: Rambarun Komaljeet

``_MegaScan.backward`` used to build ``torch.tensor([nblocks], device="cuda")``
to hand the block count to its kernels. That is a pageable host->device copy,
and PyTorch follows a blocking copy with ``cudaStreamSynchronize`` -- so every
DeltaNet block's backward drained the launch queue. The count is now a plain
int kernel argument.

The failure this guards against is silent: a reintroduced sync changes no
number, it only costs throughput. So the test asks CUDA itself, via
``torch.cuda.set_sync_debug_mode("error")``, which raises on any synchronising
call. ``test_sync_detector_is_live`` proves the detector fires on the exact
pattern that was removed -- without it, a PyTorch version that stopped flagging
that copy would turn the main test into one that cannot fail.
"""

import pytest
import torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _inputs(N=2, S=64, D=32, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(N, S, D, device="cuda", generator=g) * 0.1
    k = torch.nn.functional.normalize(
        torch.randn(N, S, D, device="cuda", generator=g), dim=-1)
    v = torch.randn(N, S, D, device="cuda", generator=g)
    a = torch.sigmoid(torch.randn(N, S, D, device="cuda", generator=g) + 3)
    b = torch.sigmoid(torch.randn(N, S, 1, device="cuda", generator=g))
    return [t.requires_grad_(True) for t in (q, k, v, a, b)]


@cuda_only
def test_sync_detector_is_live():
    """The removed line must still trip the detector, or the next test is vacuous."""
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError):
            torch.tensor([16], dtype=torch.int32, device="cuda")
    finally:
        torch.cuda.set_sync_debug_mode("default")


@cuda_only
def test_megakernel_backward_does_not_sync():
    from hvn.core.deltanet_triton import megakernel_scan
    ts = _inputs()
    o = megakernel_scan(*ts)
    do = torch.randn_like(o)
    megakernel_scan(*_inputs(seed=1)).backward(torch.randn_like(o))  # compile/warm
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        o.backward(do)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in ts)


@cuda_only
def test_megakernel_backward_matches_sequential_scan():
    """The int argument must carry the same block count the tensor did: a wrong
    count leaves ``torch.empty`` gradients unwritten, which this would catch."""
    from hvn.core.deltanet_triton import megakernel_scan, gated_delta_scan
    t1 = _inputs(N=3, S=256, D=32, seed=2)
    t2 = [t.detach().clone().requires_grad_(True) for t in t1]
    do = torch.randn(3, 256, 32, device="cuda")
    megakernel_scan(*t1).backward(do)
    gated_delta_scan(*t2).backward(do)
    for a, b in zip(t1, t2):
        rel = (a.grad - b.grad).abs().max() / b.grad.abs().max()
        assert rel < 1e-5, f"megakernel grad drifted from the sequential scan: {rel:.2e}"
