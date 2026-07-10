"""Verify _WindowDataset + MixedLocalDataset are picklable across a Windows
DataLoader spawn pipe (the fix for the step-2000 first-val crash on the
Chinchilla-floor pretrain).

numpy.memmap objects raise `OSError [Errno 22]` when pickled through the spawn
pipe; the fix stores backing file paths and re-mmaps in __setstate__. This
probe round-trips both datasets through pickle and confirms __getitem__ works
after unpickle. Uses the EXISTING tokenized bins (cache-hit), so no tokenize.

Run from repo root:  python scripts/probe_dataset_pickle.py
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import numpy as np
import torch
from transformers import GPT2Tokenizer

import train_edge_conversational as T

enc = GPT2Tokenizer.from_pretrained("gpt2")
enc.model_max_length = 10 ** 9

print("Constructing MixedLocalDataset (cache-hit on existing bins, mix 0.85/0.0/0.15)...")
ds = T.MixedLocalDataset(
    tokenizer=enc, seq_len=512, cache_dir="data_cache",
    slimpajama_dir=None, slimpajama_max_bytes=5 * 1024 ** 3,
    slimpajama_token_bytes=6 * 1024 ** 3, max_repeat=3,
    mix=(0.85, 0.0, 0.15), val_fraction=0.05,
)
print(f"  sources={len(ds.sources)} names={ds.source_names} "
      f"paths={[os.path.basename(p) for p in ds._source_paths]}")
n_src = len(ds.sources)
tok_before = [len(a) for a in ds.sources]

# --- Round-trip the TRAIN dataset ---
blob = pickle.dumps(ds)
ds2 = pickle.loads(blob)
assert not hasattr(ds2, "_just_loaded"), "noop"
assert len(ds2.sources) == n_src, f"sources count after unpickle: {len(ds2.sources)} != {n_src}"
tok_after = [len(a) for a in ds2.sources]
assert tok_before == tok_after, f"memmap lengths changed: {tok_before} -> {tok_after}"
assert all(isinstance(a, np.memmap) for a in ds2.sources), "unpickled sources not memmap"
x, y, sid = ds2[0]
assert x.shape == (512,) and y.shape == (512,), (x.shape, y.shape)
assert x.dtype == torch.int64 and y.dtype == torch.int64
print(f"  TRAIN pickle round-trip OK: sources re-mmaped, __getitem__ -> x{x.shape} y{y.shape} sid={sid}")

# --- Round-trip the VAL dataset ---
val = ds.val_dataset()
assert isinstance(val, T._WindowDataset)
vblob = pickle.dumps(val)
val2 = pickle.loads(vblob)
assert len(val2.sources) == n_src, f"val sources after unpickle: {len(val2.sources)}"
assert all(isinstance(a, np.memmap) for a in val2.sources)
assert [len(a) for a in val2.sources] == tok_before
vx, vy, vsid = val2[0]
assert vx.shape == (512,) and vy.shape == (512,)
print(f"  VAL   pickle round-trip OK: sources re-mmaped, __getitem__ -> x{vx.shape} y{vy.shape} sid={vsid}")

# --- Confirm the OLD failure mode is gone: memmaps are NOT in the pickle stream ---
# numpy.memmap pickling raises on Windows spawn; the fix is to ensure 'sources'
# is absent from the getstate payload (dropped) and the file paths are kept.
st = ds.__getstate__()
assert "sources" not in st, "sources not dropped from train getstate!"
stv = val.__getstate__()
assert "sources" not in stv, "sources not dropped from val getstate!"
assert st["_source_paths"] == ds._source_paths
print("  getstate drops 'sources' + keeps _source_paths: OK")

print("\nALL PICKLE PROBES PASSED — DataLoader worker spawn will not crash on memmap.")