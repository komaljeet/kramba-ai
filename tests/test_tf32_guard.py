"""Vacuity guard for the suite-wide TF32 restore fixture in conftest.py (#177).

The first test flips both TF32 flags and does NOT restore them; the second
asserts they came back. If the conftest fixture is removed or broken, the
second test fails. Relies on pytest running tests in file order (it does;
no reordering plugin is installed).
"""

import torch

_AT_IMPORT = (torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32)


def test_a_leaks_tf32_on_purpose():
    torch.backends.cuda.matmul.allow_tf32 = not _AT_IMPORT[0]
    torch.backends.cudnn.allow_tf32 = not _AT_IMPORT[1]
    assert (torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32) != _AT_IMPORT


def test_b_conftest_put_tf32_back():
    assert (torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32) == _AT_IMPORT, (
        "TF32 flags leaked out of the previous test -- the autouse "
        "_restore_tf32_flags fixture in tests/conftest.py is not working")
