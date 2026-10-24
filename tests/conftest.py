"""Suite-wide import-order guard.

Author: Rambarun Komaljeet

Loads pyarrow (and pandas, which pulls it in) BEFORE any test module is
collected. On this stack (torch 2.12.1+cu126, pyarrow 24, pandas 3, Windows 11)
loading pyarrow's native code after torch has initialised the CUDA driver
corrupts the process heap and kills pytest with 0xC0000374 -- no traceback,
no Python exception. Measured, each line a fresh process:

    import torch; torch.cuda.is_available(); import pandas     -> 0xC0000374
    import torch; torch.cuda.is_available(); import pyarrow.lib -> 0xC0000374
    import pyarrow; import torch; torch.cuda.is_available();
        torch.zeros(1, device="cuda"); import pandas, datasets  -> fine

So any test file that touches CUDA at module scope (e.g. a
``skipif(not torch.cuda.is_available())`` marker) followed, in the same
process, by any file that imports pandas/pyarrow/datasets -- directly or via
``train_edge_conversational`` -- crashed collection. ``pytest tests`` died this
way. Same root cause as the ordering note at the top of
``train_edge_conversational.py``; this applies that fix suite-wide.
"""

import pyarrow  # noqa: F401  (ordering is load-bearing -- see above)
import pandas  # noqa: F401

import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_tf32_flags():
    """Suite-wide TF32 guard (#177).

    ``apply_fast_stack(tf32=True)`` and several tests flip TF32 process-wide.
    A leaked True silently loosens every later fp32 matmul in the run -- #170
    nearly flipped test_bcd_balanced_offload's loss check that way. The
    per-file fixtures in test_fp8_fused.py / test_fused_ternary_in_loop.py
    only cover those two files; this covers every test, including any new
    one. tests/test_tf32_guard.py fails if this fixture stops working.
    """
    saved = (torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32)
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved
