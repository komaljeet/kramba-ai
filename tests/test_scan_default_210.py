"""#210: the trainer's default DeltaNet scan kernel is "t"; "v1" is the opt-out.

Author: Rambarun Komaljeet

Claims and the test for each:
* The real trainer parser defaults --scan-kernel to "t"; "--scan-kernel v1" still
  parses to "v1"; an unknown name is refused.
* With the parsed DEFAULT args applied the way main() applies them, a real
  DeltaNetBlock (megakernel strided-IO path, d_head 32) under activation
  checkpointing runs the "t" kernels: SCAN_T_STATS["fused_ckpt"] goes up and the
  t autograd Function is the one called.
* Must-fail partner: the same block with "--scan-kernel v1" leaves fused_ckpt
  unchanged and calls the v1 Function, so the counter assertion above is not
  vacuous (it would fail if the default were still v1).
"""
import pytest
import torch

import train_edge_conversational as T

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _parse(argv):
    return T.build_arg_parser().parse_args(argv)


def test_default_parses_to_t():
    assert _parse([]).scan_kernel == "t"


def test_opt_out_parses_to_v1():
    assert _parse(["--scan-kernel", "v1"]).scan_kernel == "v1"
    assert _parse(["--scan-kernel", "t"]).scan_kernel == "t"


def test_unknown_kernel_refused():
    with pytest.raises(SystemExit):
        _parse(["--scan-kernel", "v2"])


def _run_block(argv, monkeypatch):
    """Apply the parsed args like main() does, then fwd+bwd one checkpointed
    DeltaNetBlock. Returns (fused_ckpt delta, calls per Function)."""
    from hvn.core import deltanet_triton as dt
    from hvn.core.deltanet import DeltaNetBlock

    monkeypatch.setattr(dt, "SCAN_KERNEL", list(dt.SCAN_KERNEL))   # restored after test
    calls = {"v1": 0, "t": 0}
    for name, cls in (("v1", dt._MegaScanBSHD), ("t", dt._MegaScanBSHD_T)):
        orig = cls.apply

        def counted(*a, _orig=orig, _name=name):
            calls[_name] += 1
            return _orig(*a)
        monkeypatch.setattr(cls, "apply", counted)

    T.apply_scan_kernel_args(_parse(argv))
    torch.manual_seed(0)
    # d_head 32, as at 1B; the training fast path (--fast turns both scan flags on).
    blk = DeltaNetBlock(d_model=128, n_heads=4, dropout=0.0, use_triton_scan=True,
                        use_megakernel_scan=True).cuda()
    x = torch.randn(2, 256, 128, device="cuda", requires_grad=True)
    f0 = dt.SCAN_T_STATS["fused_ckpt"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
    y.float().square().mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    return dt.SCAN_T_STATS["fused_ckpt"] - f0, calls


@cuda_only
def test_default_runs_t_kernel_on_real_block(monkeypatch):
    d_fused, calls = _run_block([], monkeypatch)
    assert d_fused > 0, "default did not run the t fused-checkpoint kernel"
    assert calls["t"] > 0 and calls["v1"] == 0, calls


@cuda_only
def test_must_fail_partner_v1_opt_out(monkeypatch):
    """The assertion above must be able to fail: under the v1 opt-out the same
    block runs, but fused_ckpt does not move and only the v1 Function is called."""
    d_fused, calls = _run_block(["--scan-kernel", "v1"], monkeypatch)
    assert d_fused == 0
    assert calls["v1"] > 0 and calls["t"] == 0, calls
