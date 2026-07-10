"""Quick check that --pretrain-mix excludes zero-fraction sources and the
total_chunks math is right. Uses the EXISTING 129M slimpajama.bin (cache-hit),
so no full re-tokenize. Run from repo root."""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from transformers import GPT2Tokenizer
import train_edge_conversational as T

enc = GPT2Tokenizer.from_pretrained("gpt2")
enc.model_max_length = 10 ** 9

NAMES = {0: "SlimPajama", 1: "OpenAssistant", 2: "WikiText-103"}

for mix, label in [
    (None, "default 0.50/0.35/0.15 (OASST binds ~135M)"),
    ((0.85, 0.0, 0.15), "knowledge-heavy 0.85/0.0/0.15 (OASST dropped)"),
]:
    print(f"\n=== {label}  mix={mix} ===")
    ds = T.MixedLocalDataset(
        tokenizer=enc, seq_len=512, cache_dir="data_cache",
        slimpajama_dir=None, slimpajama_max_bytes=5 * 1024**3,
        slimpajama_token_bytes=500 * 1024**2, max_repeat=3, mix=mix, val_fraction=0.0,
    )
    src_names = [os.path.basename(a.filename) if a.filename else "?" for a in ds.sources]
    print(f"  sources included : {len(ds.sources)}  files={src_names}")
    print(f"  total_tokens disk : {ds.total_tokens:,}")
    cnt = collections.Counter(int(s) for s in ds.src_ids)
    for sid in sorted(cnt):
        print(f"    sid {sid} ({NAMES.get(sid,'?'):12s}): {cnt[sid]:>9,} chunks  "
              f"({cnt[sid]*512:>12,} tok)  share={cnt[sid]/len(ds):.3f}")
    print(f"  num_samples(train): {len(ds):,}  ({len(ds)*512:,} tokens in train index)")