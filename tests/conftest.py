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
