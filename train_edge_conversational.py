"""
EdgeHVN Conversational Training Pipeline (Issue #72)
=====================================================

A complete training pipeline that builds a base conversational AI model
using the existing EdgeHVN + DeltaNet + ternary BitNet b1.58 implementation.

Steps (executed in order — each step stops and reports if it fails):
  1. Hardware scan         — nvidia-smi, RAM, torch device checks → optimal settings
  2. Dataset pipeline      — SlimPajama (50%) + OpenAssistant (35%) + WikiText-103 (15%)
  3. Training configuration — DeltaNet, bf16, grad accum, cosine LR, checkpointing
  4. Pre-training audit     — shape check, forward pass, checkpoint cycle
  5. Start training         — param count, compression ratio, loss/PPL logging

Usage:
    python train_edge_conversational.py
    python train_edge_conversational.py --smoke          # tiny budget, fast validation
    python train_edge_conversational.py --max-steps 1000  # cap training steps

O(1) inference is preserved: this only trains the model; the served model
is exactly what EdgeHVN produces (no KV cache, fixed-size Hamiltonian).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

# Force UTF-8 stdout/stderr so the ✓/→ characters don't crash on cp1252 Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from hvn.edge.model import EdgeHVN
from hvn.edge.ternary import apply_ternary_to_model, TernaryLinear


def _save_edge_checkpoint(model: EdgeHVN, save_path: str, global_step: int = 0):
    """Save an EdgeHVN checkpoint with full architecture metadata.

    Unlike train._save_checkpoint, this correctly stores n_heads (which
    EdgeHVN doesn't expose as a model-level attribute, so the generic
    getattr fallback would default to 4 instead of the real value).
    """
    # Find the first block to extract architecture details
    if hasattr(model, 'shared_block') and model.shared_block is not None:
        first_block = model.shared_block.block
    elif hasattr(model, 'blocks') and len(model.blocks) > 0:
        first_block = model.blocks[0]
    else:
        first_block = None

    # Get n_heads from the block (where it's actually stored)
    n_heads = getattr(first_block, 'n_heads', 4) if first_block is not None else 4

    ckpt = {
        "model_state_dict": model.state_dict(),
        "vocab_size": model.vocab_size,
        "d_model": model.d_model,
        "num_layers": model.num_layers,
        "n_heads": n_heads,
        "n_threads": getattr(model, 'n_threads', 1),
        "chunk_size": getattr(first_block, 'chunk_size', 64) if first_block is not None else 64,
        "use_deltanet": getattr(model, 'use_deltanet', False),
        "deltanet_conv_kernel": getattr(first_block, 'conv_kernel', 4) if first_block is not None else 4,
        "weightless": getattr(model, 'weightless', False),
        "weightless_seed": getattr(model, 'weightless_seed', 1234),
        "is_edge": True,
        "shared_weights": getattr(model, 'shared_weights', False),
        "global_step": global_step,
    }
    torch.save(ckpt, save_path)

# ---------------------------------------------------------------------------
# Step 1 — Hardware Scan
# ---------------------------------------------------------------------------

def hardware_scan() -> dict:
    """Run nvidia-smi, check RAM, and torch device capabilities.

    Returns a dict of findings and chosen training settings.
    """
    print("=" * 70)
    print("STEP 1 — HARDWARE SCAN")
    print("=" * 70)

    # --- nvidia-smi ---
    print("\n--- nvidia-smi ---")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        print(result.stdout)
    except Exception as e:
        print(f"  nvidia-smi failed: {e}")

    # --- System RAM (Windows: use WMI/CIM; Linux: free -h) ---
    print("--- System RAM ---")
    if sys.platform == "win32":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_ram = stat.ullTotalPhys / (1024**3)
            free_ram = stat.ullAvailPhys / (1024**3)
            print(f"  Total RAM: {total_ram:.2f} GB")
            print(f"  Free RAM:  {free_ram:.2f} GB")
        except Exception as e:
            print(f"  RAM check failed: {e}")
            total_ram = 0
            free_ram = 0
    else:
        try:
            result = subprocess.run(["free", "-h"], capture_output=True, text=True, timeout=10)
            print(result.stdout)
            # Parse total/free from output
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if line.startswith("Mem:"):
                    parts = line.split()
                    total_ram = float(parts[1].rstrip("G")) if "G" in parts[1] else 0
                    free_ram = float(parts[3].rstrip("G")) if "G" in parts[3] else 0
                    break
        except Exception as e:
            print(f"  free -h failed: {e}")
            total_ram = 0
            free_ram = 0

    # --- torch device checks ---
    print("\n--- Torch Device Checks ---")
    if not torch.cuda.is_available():
        print("  ❌ CUDA not available — cannot train on GPU. Aborting.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_vram_bytes, _ = torch.cuda.mem_get_info(0)
    free_vram = free_vram_bytes / (1024**3)
    bf16_supported = torch.cuda.is_bf16_supported()
    torch_version = torch.__version__

    print(f"  GPU:              {gpu_name}")
    print(f"  VRAM total:       {vram_total:.2f} GB")
    print(f"  VRAM free:        {free_vram:.2f} GB")
    print(f"  bf16 supported:   {bf16_supported}")
    print(f"  Torch version:    {torch_version}")

    # --- Determine optimal settings ---
    # WDDM warning: on Windows, WDDM silently spills past VRAM into host RAM
    # (4-15× slower). Keep peak_reserved < ~4.5 GB on a 6 GB card.
    # From VRAM probing (Issue #72 Step 1):
    #   d=512, L=12, h=8, B=4, S=512 → peak ~2.6 GB (forward+backward, ternary+bf16)
    #   + optimizer state (fp32 AdamW: 2× params) ~0.66 GB
    #   Total est. ~3.3 GB — safe within 4.5 GB envelope.

    if free_vram >= 4.0:
        # Comfortable: use the reference architecture
        settings = {
            "d_model": 512,
            "num_layers": 12,
            "n_heads": 8,
            "batch_size": 4,
            "seq_len": 512,
            "grad_accum": 8,       # effective batch = 4 × 8 = 32
            "precision": "bf16",
        }
    elif free_vram >= 2.5:
        # Tighter: smaller model
        settings = {
            "d_model": 256,
            "num_layers": 6,
            "n_heads": 4,
            "batch_size": 4,
            "seq_len": 256,
            "grad_accum": 8,       # effective batch = 32
            "precision": "bf16",
        }
    else:
        # Very constrained: minimal model
        settings = {
            "d_model": 128,
            "num_layers": 3,
            "n_heads": 4,
            "batch_size": 2,
            "seq_len": 128,
            "grad_accum": 16,      # effective batch = 32
            "precision": "bf16",
        }

    effective_batch = settings["batch_size"] * settings["grad_accum"]
    estimated_peak = 3.3 if settings["d_model"] == 512 else 1.5

    print(f"\n--- Chosen Settings (with reasoning) ---")
    print(f"  d_model:       {settings['d_model']}  (reference dimension from README)")
    print(f"  num_layers:    {settings['num_layers']}  (depth for language understanding)")
    print(f"  n_heads:       {settings['n_heads']}  (standard head count for d={settings['d_model']})")
    print(f"  batch_size:    {settings['batch_size']}  (fits in VRAM with ternary compression)")
    print(f"  seq_len:       {settings['seq_len']}  (good context for conversational training)")
    print(f"  grad_accum:    {settings['grad_accum']}  (effective batch = {effective_batch})")
    print(f"  precision:     {settings['precision']}  (RTX 4050 supports bf16)")
    print(f"  Est. peak VRAM: ~{estimated_peak} GB (safe within {free_vram:.1f} GB free)")
    print(f"  Effective batch size: {effective_batch}")

    findings = {
        "gpu_name": gpu_name,
        "vram_total_gb": vram_total,
        "vram_free_gb": free_vram,
        "total_ram_gb": total_ram,
        "free_ram_gb": free_ram,
        "bf16_supported": bf16_supported,
        "torch_version": torch_version,
        "settings": settings,
        "effective_batch_size": effective_batch,
        "estimated_peak_vram_gb": estimated_peak,
    }
    print(f"\n✓ Step 1 complete.")
    return findings


# ---------------------------------------------------------------------------
# Step 2 — Dataset Pipeline
# ---------------------------------------------------------------------------

# GPT-2 vocab size (50257) — used as the tokenizer baseline
GPT2_VOCAB_SIZE = 50257


def _download_language_foundation(max_bytes: int, cache_dir: str, slimpajama_dir: str | None = None) -> str:
    """Prepare the SlimPajama-627B language foundation corpus.

    Uses local parquet files downloaded via `hf download` CLI (not Python
    streaming, which crashes on Windows). Reads the parquet files directly
    with pandas/pyarrow and extracts text to a flat .txt file.
    Returns the path to the extracted text file.

    Parameters
    ----------
    max_bytes : int
        Cap on total bytes extracted.
    cache_dir : str
        Directory for the output .txt file.
    slimpajama_dir : str or None
        Directory containing SlimPajama parquet files. If None, defaults
        to <cache_dir>/slimpajama/data.
    """
    out_path = os.path.join(cache_dir, "slimpajama_en.txt")
    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        print(f"    SlimPajama cached: {out_path} ({size / 1024**2:.1f} MB)")
        return out_path

    # Look for locally downloaded parquet files (from `hf download`)
    if slimpajama_dir:
        parquet_dir = slimpajama_dir
    else:
        parquet_dir = os.path.join(cache_dir, "slimpajama", "data")
    if not os.path.exists(parquet_dir):
        raise FileNotFoundError(
            f"SlimPajama parquet files not found at {parquet_dir}.\n"
            "Download them first with:\n"
            "  hf download gmongaras/SlimPajama-627B_Reupload --repo-type dataset "
            "--local-dir data_cache/slimpajama "
            "--include 'data/train-00000-of-00250.parquet' "
            "--include 'data/train-00001-of-00250.parquet' "
            "--include 'data/train-00002-of-00250.parquet'"
        )

    import pandas as pd
    parquet_files = sorted([
        os.path.join(parquet_dir, f)
        for f in os.listdir(parquet_dir)
        if f.startswith("train-") and f.endswith(".parquet")
    ])
    print(f"    Found {len(parquet_files)} SlimPajama parquet files in {parquet_dir}")
    bytes_seen = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for pf in parquet_files:
            if bytes_seen >= max_bytes:
                break
            print(f"    Reading {os.path.basename(pf)}...")
            df = pd.read_parquet(pf, columns=["text"])
            for text in df["text"]:
                if bytes_seen >= max_bytes:
                    break
                if not text or not text.strip():
                    continue
                f.write(text)
                f.write("\n")
                bytes_seen += len(text.encode("utf-8"))
            print(f"      {bytes_seen / 1024**3:.2f} GB extracted so far...")
    print(f"    SlimPajama saved: {out_path} ({os.path.getsize(out_path) / 1024**2:.1f} MB)")
    return out_path


def _download_oasst(cache_dir: str) -> str:
    """Download OpenAssistant Conversations (oasst1) to a local text file.

    Returns the path to the downloaded text file.
    """
    out_path = os.path.join(cache_dir, "oasst1.txt")
    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        print(f"    OpenAssistant cached: {out_path} ({size / 1024**2:.1f} MB)")
        return out_path

    from datasets import load_dataset
    print(f"    Downloading OpenAssistant (oasst1)...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    with open(out_path, "w", encoding="utf-8") as f:
        for example in ds:
            text = example.get("text", "")
            if text:
                f.write(text)
                f.write("\n")
    print(f"    OpenAssistant saved: {out_path} ({os.path.getsize(out_path) / 1024**2:.1f} MB)")
    return out_path


def _tokenize_text_to_bin(
    text_path: str,
    bin_path: str,
    tokenizer,
    max_bytes: int | None = None,
    block_chars: int = 4_000_000,
) -> int:
    """Stream-tokenize a UTF-8 text file into an on-disk uint16 .bin of token ids.

    Reads the text in fixed-size character blocks, encodes each block with the
    (fast) GPT-2 tokenizer, and appends the ids to `bin_path` as native-endian
    uint16 (GPT-2 vocab 50257 < 65536, so 2 bytes/token suffices). Only one block
    is held in RAM at a time, so memory stays flat regardless of corpus size —
    this is what lets multi-GB SlimPajama tokenize on a 16 GB laptop.

    The result is cached: if `bin_path` already exists non-empty it is reused.
    `max_bytes` caps how much source text is consumed (by UTF-8 byte count).
    Returns the total number of tokens written.
    """
    if os.path.exists(bin_path) and os.path.getsize(bin_path) > 0:
        n = os.path.getsize(bin_path) // 2
        print(f"    Tokenized cache hit: {bin_path} ({n:,} tokens)")
        return n

    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    tmp_path = bin_path + ".tmp"
    total_tokens = 0
    bytes_read = 0
    blocks = 0
    t0 = time.time()
    with open(text_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(tmp_path, "wb") as fout:
        while True:
            if max_bytes is not None and bytes_read >= max_bytes:
                break
            block = fin.read(block_chars)
            if not block:
                break
            bytes_read += len(block.encode("utf-8"))
            ids = tokenizer.encode(block)
            np.asarray(ids, dtype=np.uint16).tofile(fout)
            total_tokens += len(ids)
            blocks += 1
            if blocks % 20 == 0:
                print(f"      {os.path.basename(bin_path)}: {total_tokens:,} tokens "
                      f"({bytes_read / 1024**2:.0f} MB, {time.time() - t0:.0f}s)", flush=True)
    os.replace(tmp_path, bin_path)
    print(f"    Tokenized {os.path.basename(text_path)} -> {bin_path} "
          f"({total_tokens:,} tokens, {time.time() - t0:.0f}s)")
    return total_tokens


class MixedLocalDataset(Dataset):
    """Map-style dataset mixing three local text files at a fixed ratio.

    Mix ratio: 50% SlimPajama, 35% OpenAssistant, 15% WikiText-103.
    All sources are downloaded to local files first (no streaming threads),
    then tokenized and concatenated with source tags for the mix ratio.
    This avoids the HuggingFace streaming thread crash on Windows.

    Parameters
    ----------
    tokenizer : GPT2Tokenizer
        HuggingFace GPT-2 BPE tokenizer.
    seq_len : int
        Sequence length per training sample.
    cache_dir : str
        Directory for downloaded dataset files.
    slimpajama_max_bytes : int
        Cap on SlimPajama download size (default 5 GB).
    max_chars_per_source : int or None
        If set, cap characters read from each source (for smoke mode).
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int = 512,
        cache_dir: str = "data_cache",
        slimpajama_dir: str | None = None,
        slimpajama_max_bytes: int = 5 * 1024**3,
        slimpajama_token_bytes: int = 500 * 1024**2,
        max_repeat: int = 3,
    ):
        """Build the mixed dataset from on-disk memory-mapped token files.

        Each source is tokenized once to a cached uint16 .bin (streaming — flat
        RAM) and memory-mapped read-only. Samples are seq_len windows referenced
        by (source_id, start_offset); only the windows actually read are paged in
        from disk, so RAM stays flat no matter how large the corpus is.

        `slimpajama_token_bytes` caps how much SlimPajama text is tokenized (the
        full .txt may be much larger; we only need a slice). `max_repeat` bounds
        how many times any single source may repeat to satisfy the 50/35/15 mix,
        preventing tiny OpenAssistant (~12M tokens) from being duplicated ~40x
        against multi-GB SlimPajama.
        """
        self.seq_len = seq_len
        os.makedirs(cache_dir, exist_ok=True)
        tok_dir = os.path.join(cache_dir, "tokenized")

        # Use a fast tokenizer for the one-time encode pass: identical BPE ids to
        # the slow GPT2Tokenizer but far faster on large corpora. Fall back to the
        # provided tokenizer if the fast one can't be loaded.
        enc = tokenizer
        try:
            from transformers import GPT2TokenizerFast
            enc = GPT2TokenizerFast.from_pretrained("gpt2")
            enc.model_max_length = 10 ** 9  # we chunk ourselves; silence length warnings
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"    (fast tokenizer unavailable, using provided tokenizer: {e})")

        print("\n  Preparing datasets (tokenize once -> memmap, flat RAM)...")

        # 1. SlimPajama-627B (50%) — language foundation
        lang_path = _download_language_foundation(slimpajama_max_bytes, cache_dir, slimpajama_dir)
        print(f"    Tokenizing SlimPajama (cap {slimpajama_token_bytes / 1024**2:.0f} MB)...")
        slim_bin = os.path.join(tok_dir, "slimpajama.bin")
        _tokenize_text_to_bin(lang_path, slim_bin, enc, max_bytes=slimpajama_token_bytes)

        # 2. OpenAssistant (35%)
        oasst_path = _download_oasst(cache_dir)
        print(f"    Tokenizing OpenAssistant...")
        oasst_bin = os.path.join(tok_dir, "oasst.bin")
        _tokenize_text_to_bin(oasst_path, oasst_bin, enc)

        # 3. WikiText-103 (15%) — local files already in the codebase
        local_wt103 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wikitext-103-v1")
        wt103_path = os.path.join(local_wt103, "wiki.train.tokens")
        wt_bin = None
        if os.path.exists(wt103_path):
            print(f"    Tokenizing WikiText-103 from {wt103_path}...")
            wt_bin = os.path.join(tok_dir, "wikitext.bin")
            _tokenize_text_to_bin(wt103_path, wt_bin, enc)
        else:
            print(f"    WikiText-103 not found locally, skipping")

        # --- Memory-map each tokenized source (read-only, paged on demand) ---
        # Source IDs: 0=SlimPajama, 1=OpenAssistant, 2=WikiText-103
        source_paths = [slim_bin, oasst_bin]
        ratios = [0.50, 0.35]
        if wt_bin is not None:
            source_paths.append(wt_bin)
            ratios.append(0.15)
        ratios = [r / sum(ratios) for r in ratios]  # renormalise if wikitext missing

        self.sources: list[np.memmap] = []
        avail = []  # number of non-overlapping seq_len windows per source
        for p in source_paths:
            arr = np.memmap(p, dtype=np.uint16, mode="r")
            self.sources.append(arr)
            n_win = max(0, (len(arr) - seq_len - 1) // seq_len + 1) if len(arr) >= seq_len + 1 else 0
            avail.append(n_win)
        self.total_tokens = int(sum(len(a) for a in self.sources))

        # --- Mix at the chunk level with a repetition cap ---
        # Pick the largest total window count such that no source must repeat
        # more than `max_repeat` times to hit its target share.
        total_cap = min(
            (max_repeat * n) / r
            for n, r in zip(avail, ratios) if n > 0 and r > 0
        )
        total_chunks = int(min(sum(avail), total_cap))

        starts_parts, src_parts = [], []
        for sid, (n_avail, r) in enumerate(zip(avail, ratios)):
            if n_avail == 0:
                continue
            target = int(total_chunks * r)
            if target <= 0:
                continue
            offs = np.arange(n_avail, dtype=np.int64) * seq_len
            if target <= n_avail:
                sel = offs[:target]
            else:
                sel = np.tile(offs, target // n_avail + 1)[:target]
            starts_parts.append(sel)
            src_parts.append(np.full(sel.shape, sid, dtype=np.uint8))

        self.starts = np.concatenate(starts_parts)
        self.src_ids = np.concatenate(src_parts)

        # Deterministic shuffle of the (source, offset) index.
        perm = np.random.RandomState(42).permutation(len(self.starts))
        self.starts = self.starts[perm]
        self.src_ids = self.src_ids[perm]

        names = ["SlimPajama", "OpenAssistant", "WikiText-103"]
        counts = [int((self.src_ids == s).sum()) for s in range(len(self.sources))]
        tot = max(1, len(self.src_ids))
        print(f"\n    Mixed dataset: {len(self.starts):,} samples (seq_len={seq_len})")
        print(f"    Unique tokens on disk: {self.total_tokens:,}  |  windows/source: {avail}")
        print("    Per-source: " + "  ".join(
            f"{names[s]}={counts[s]:,} ({100 * counts[s] / tot:.1f}%)"
            for s in range(len(self.sources))))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        src_id = int(self.src_ids[idx])
        start = int(self.starts[idx])
        arr = self.sources[src_id]
        chunk = np.asarray(arr[start:start + self.seq_len + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y, src_id


class WikiText103LocalDataset(Dataset):
    """Map-style dataset reading WikiText-103 from local files.

    Used as a fallback when streaming datasets are unavailable.
    """

    def __init__(self, tokenizer, seq_len: int = 512, max_chars: int | None = None):
        self.seq_len = seq_len
        local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wikitext-103-v1")
        train_path = os.path.join(local_dir, "wiki.train.tokens")

        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"WikiText-103 not found at {train_path}. "
                "Download it or ensure streaming datasets are available."
            )

        print(f"  Reading WikiText-103 from {train_path}...")
        with open(train_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(max_chars) if max_chars else f.read()

        print(f"  Tokenizing {len(text)} characters...")
        self.token_ids = tokenizer.encode(text)
        print(f"  Total tokens: {len(self.token_ids)}")

    def __len__(self):
        return max(1, (len(self.token_ids) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.token_ids[start:end]
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [0] * (self.seq_len + 1 - len(chunk))
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def build_dataset_pipeline(
    tokenizer,
    seq_len: int,
    batch_size: int,
    smoke: bool = False,
    data_dir: str = "data_cache",
    slimpajama_dir: str | None = None,
) -> tuple[DataLoader, dict]:
    """Build the mixed-source DataLoader and report dataset stats.

    Parameters
    ----------
    data_dir : str
        Directory for cached dataset files (oasst1.txt, slimpajama_en.txt).
    slimpajama_dir : str or None
        Directory containing SlimPajama parquet files. If None, defaults
        to <data_dir>/slimpajama/data.

    Returns (dataloader, stats_dict).
    """
    print("\n" + "=" * 70)
    print("STEP 2 — DATASET PIPELINE")
    print("=" * 70)

    mix_ratio = {"SlimPajama": 0.50, "OpenAssistant": 0.35, "WikiText-103": 0.15}
    print(f"\nMix ratio: {mix_ratio}")
    print(f"Sequence length: {seq_len}")
    print(f"Batch size: {batch_size}")

    stats = {"mix_ratio": mix_ratio, "seq_len": seq_len, "batch_size": batch_size}

    if smoke:
        # Smoke mode: use local WikiText-103 only, tiny budget
        print("\n  [SMOKE MODE] Using local WikiText-103 only (tiny budget)")
        max_chars = 200_000  # ~200K chars for smoke
        dataset = WikiText103LocalDataset(tokenizer, seq_len=seq_len, max_chars=max_chars)
        stats["source"] = "wikitext103_local_smoke"
        stats["total_tokens"] = len(dataset.token_ids)
        stats["num_samples"] = len(dataset)
        print(f"  Tokens: {stats['total_tokens']}")
        print(f"  Samples: {stats['num_samples']}")
    else:
        # Full mode: download all three sources to local files, then mix
        print(f"\n  Loading datasets from {data_dir}...")
        dataset = MixedLocalDataset(
            tokenizer=tokenizer,
            seq_len=seq_len,
            cache_dir=data_dir,
            slimpajama_dir=slimpajama_dir,
            slimpajama_max_bytes=5 * 1024**3,
        )
        stats["source"] = "mixed_local"
        stats["total_tokens"] = dataset.total_tokens
        stats["num_samples"] = len(dataset)
        print(f"  Tokens: {stats['total_tokens']}")
        print(f"  Samples: {stats['num_samples']}")

    # Build DataLoader (always map-style now — no IterableDataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    stats["num_batches"] = len(dataloader)
    print(f"\n  DataLoader ready: batch_size={batch_size}")
    print(f"✓ Step 2 complete.")
    return dataloader, stats


# ---------------------------------------------------------------------------
# Step 3 — Training Configuration
# ---------------------------------------------------------------------------

def build_model_and_config(
    settings: dict,
    tokenizer,
    device: torch.device,
    checkpoint_dir: str,
) -> tuple[EdgeHVN, dict]:
    """Build the EdgeHVN model with DeltaNet + ternary weights and training config.

    Returns (model, training_config).
    """
    print("\n" + "=" * 70)
    print("STEP 3 — TRAINING CONFIGURATION")
    print("=" * 70)

    config = {
        "use_deltanet": True,
        "bf16": True,
        "batch_size": settings["batch_size"],
        "grad_accum": settings["grad_accum"],
        "seq_len": settings["seq_len"],
        "d_model": settings["d_model"],
        "num_layers": settings["num_layers"],
        "n_heads": settings["n_heads"],
        "optimizer": "AdamW",
        "lr": 3e-4,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "cosine_schedule": True,
        "save_every_steps": 500,
        "log_every_steps": 50,
        "checkpoint_dir": checkpoint_dir,
        "log_file": os.path.join(checkpoint_dir, "training_log.csv"),
    }

    print(f"\n  use_deltanet:      {config['use_deltanet']}")
    print(f"  precision:         bf16")
    print(f"  batch_size:        {config['batch_size']}")
    print(f"  grad_accum:        {config['grad_accum']} (effective batch = {config['batch_size'] * config['grad_accum']})")
    print(f"  seq_len:           {config['seq_len']}")
    print(f"  d_model:           {config['d_model']}")
    print(f"  num_layers:        {config['num_layers']}")
    print(f"  n_heads:           {config['n_heads']}")
    print(f"  optimizer:         {config['optimizer']} (lr={config['lr']}, wd={config['weight_decay']})")
    print(f"  schedule:          cosine LR, warmup {config['warmup_steps']} steps")
    print(f"  checkpoint_every:  {config['save_every_steps']} steps → {checkpoint_dir}/")
    print(f"  log_every:         {config['log_every_steps']} steps (console + CSV)")
    print(f"  resume:            auto-resume from {checkpoint_dir}/ if checkpoint exists")

    # Build the model
    print(f"\n  Building EdgeHVN model...")
    model = EdgeHVN(
        vocab_size=GPT2_VOCAB_SIZE,
        d_model=settings["d_model"],
        num_layers=settings["num_layers"],
        n_heads=settings["n_heads"],
        use_deltanet=True,
        deltanet_conv_kernel=4,
    )

    # Apply ternary BitNet b1.58 weights
    print(f"  Applying ternary BitNet b1.58 weight quantization...")
    all_linears = {name: mod for name, mod in model.named_modules()
                   if isinstance(mod, nn.Linear)}
    exclude_patterns = ("beta_proj", "alpha_proj", "decoder")
    target_names = [name for name in all_linears
                    if not any(p in name for p in exclude_patterns)]
    replaced = apply_ternary_to_model(model, target_names=target_names)
    print(f"  Ternary: replaced {len(replaced)} nn.Linear layers with TernaryLinear (STE)")
    print(f"  Excluded {len(all_linears) - len(replaced)} gate/decoder layers")

    # Move to device
    model = model.to(device)

    # Report compression
    dense_bytes = sum(p.numel() * 4 for p in model.parameters())
    ternary_bytes = 0
    accounted = set()
    for name, mod in model.named_modules():
        if isinstance(mod, TernaryLinear):
            ternary_bytes += mod.memory_bytes()
            accounted.add(f"{name}.weight")
            if mod.bias is not None:
                accounted.add(f"{name}.bias")
    for name, p in model.named_parameters():
        if name not in accounted:
            ternary_bytes += p.numel() * 4

    compression_ratio = dense_bytes / max(ternary_bytes, 1)
    config["dense_bytes"] = dense_bytes
    config["compressed_bytes"] = ternary_bytes
    config["compression_ratio"] = compression_ratio

    print(f"\n  Weight storage (dense fp32):  {dense_bytes / 1024**2:.2f} MB")
    print(f"  Weight storage (ternary):     {ternary_bytes / 1024**2:.2f} MB")
    print(f"  Compression ratio:            {compression_ratio:.2f}x")

    print(f"\n✓ Step 3 complete.")
    return model, config


# ---------------------------------------------------------------------------
# Step 4 — Pre-Training Audit
# ---------------------------------------------------------------------------

def pre_training_audit(
    model: EdgeHVN,
    dataloader: DataLoader,
    config: dict,
    device: torch.device,
    checkpoint_dir: str,
) -> bool:
    """Verify the training pipeline before starting the training loop.

    Checks:
      1. DataLoader produces correct token shapes
      2. EdgeHVN + DeltaNet + ternary forward pass runs without error
      3. Checkpoint save and load cycle works correctly

    Returns True if all checks pass, False otherwise.
    """
    print("\n" + "=" * 70)
    print("STEP 4 — PRE-TRAINING AUDIT")
    print("=" * 70)

    all_passed = True

    # --- Check 1: DataLoader produces correct token shapes ---
    print("\n  [Check 1] DataLoader token shapes...")
    try:
        batch = next(iter(dataloader))
        # MixedLocalDataset returns (x, y, src_id); WikiText103LocalDataset returns (x, y)
        if len(batch) == 3:
            batch_x, batch_y, batch_src = batch
            print(f"    batch_src (source IDs): {batch_src}")
        else:
            batch_x, batch_y = batch
            batch_src = None

        expected_shape = (config["batch_size"], config["seq_len"])
        print(f"    batch_x shape: {batch_x.shape} (expected {expected_shape})")
        print(f"    batch_y shape: {batch_y.shape} (expected {expected_shape})")
        print(f"    batch_x dtype: {batch_x.dtype} (expected torch.long)")
        print(f"    batch_x range: [{batch_x.min().item()}, {batch_x.max().item()}]")

        if batch_x.shape != expected_shape:
            if batch_x.shape[0] <= config["batch_size"] and batch_x.shape[1] == config["seq_len"]:
                print(f"    ✓ Shapes OK (partial batch: {batch_x.shape[0]} ≤ {config['batch_size']})")
            else:
                print(f"    ❌ Shape mismatch: got {batch_x.shape}, expected {expected_shape}")
                all_passed = False
        else:
            print(f"    ✓ Shapes correct")
        if batch_x.dtype != torch.long:
            print(f"    ❌ dtype mismatch: got {batch_x.dtype}, expected torch.long")
            all_passed = False
    except Exception as e:
        print(f"    ❌ DataLoader check failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # --- Check 2: Forward pass runs without error ---
    print("\n  [Check 2] Forward pass (EdgeHVN + DeltaNet + ternary)...")
    try:
        model.eval()
        x = batch_x.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        print(f"    Input shape:  {x.shape}")
        print(f"    Output shape: {logits.shape} (expected [{x.shape[0]}, {x.shape[1]}, {GPT2_VOCAB_SIZE}])")
        if logits.shape[0] != x.shape[0] or logits.shape[1] != x.shape[1]:
            print(f"    ❌ Output shape mismatch!")
            all_passed = False
        elif logits.shape[2] != GPT2_VOCAB_SIZE:
            print(f"    ❌ Vocab dimension mismatch: got {logits.shape[2]}, expected {GPT2_VOCAB_SIZE}")
            all_passed = False
        else:
            print(f"    ✓ Forward pass correct")

        # Also test backward pass
        model.train()
        y = batch_y.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = nn.CrossEntropyLoss()(
                logits.reshape(-1, GPT2_VOCAB_SIZE),
                y.reshape(-1),
            )
        loss.backward()
        print(f"    ✓ Backward pass correct (loss = {loss.item():.4f})")

        # Check gradients exist on ternary weights
        has_grad = False
        for name, mod in model.named_modules():
            if isinstance(mod, TernaryLinear) and mod.weight.grad is not None:
                has_grad = True
                break
        if has_grad:
            print(f"    ✓ Ternary weight gradients present (STE working)")
        else:
            print(f"    ⚠ No ternary weight gradients found (may be normal if all grads are None)")

        # Zero gradients for training
        model.zero_grad()
    except Exception as e:
        print(f"    ❌ Forward/backward check failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # --- Check 3: Checkpoint save and load cycle ---
    print("\n  [Check 3] Checkpoint save/load cycle...")
    test_ckpt_path = os.path.join(checkpoint_dir, "audit_test_checkpoint.pt")
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save using our EdgeHVN-aware checkpoint function
        _save_edge_checkpoint(model, test_ckpt_path, global_step=0)
        print(f"    ✓ Save: {test_ckpt_path}")

        # Load and verify
        ckpt = torch.load(test_ckpt_path, map_location=device, weights_only=False)
        print(f"    ✓ Load: keys = {list(ckpt.keys())}")
        print(f"    ✓ n_heads in checkpoint: {ckpt.get('n_heads', 'MISSING')}")

        # Rebuild model from checkpoint metadata and verify state_dict loads
        rebuilt = EdgeHVN(
            vocab_size=ckpt["vocab_size"],
            d_model=ckpt["d_model"],
            num_layers=ckpt["num_layers"],
            n_heads=ckpt.get("n_heads", 4),
            use_deltanet=ckpt.get("use_deltanet", False),
            deltanet_conv_kernel=ckpt.get("deltanet_conv_kernel", 4),
        )
        # Apply ternary to rebuilt model (same layers as original)
        all_linears = {name: mod for name, mod in rebuilt.named_modules()
                       if isinstance(mod, nn.Linear)}
        exclude_patterns = ("beta_proj", "alpha_proj", "decoder")
        target_names = [name for name in all_linears
                        if not any(p in name for p in exclude_patterns)]
        apply_ternary_to_model(rebuilt, target_names=target_names)
        rebuilt = rebuilt.to(device)
        rebuilt.load_state_dict(ckpt["model_state_dict"])
        print(f"    ✓ State dict loads into rebuilt model")

        # Clean up test checkpoint
        os.remove(test_ckpt_path)
        print(f"    ✓ Cleaned up test checkpoint")
    except Exception as e:
        print(f"    ❌ Checkpoint cycle failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
        # Clean up
        if os.path.exists(test_ckpt_path):
            try:
                os.remove(test_ckpt_path)
            except Exception:
                pass

    # --- Summary ---
    print(f"\n  {'✓ All audit checks passed!' if all_passed else '❌ Some audit checks failed!'}")
    print(f"✓ Step 4 complete.")
    return all_passed


# ---------------------------------------------------------------------------
# Step 5 — Start Training
# ---------------------------------------------------------------------------

def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the latest step checkpoint in the checkpoint directory.

    Looks for files matching the pattern *_step*.pt, and also the base
    checkpoint name. Returns the path to the latest one, or None.
    """
    if not os.path.exists(checkpoint_dir):
        return None

    step_ckpts = []
    for f in os.listdir(checkpoint_dir):
        if f.endswith(".pt") and "_step" in f:
            # Extract step number
            try:
                step_str = f.split("_step")[-1].replace(".pt", "")
                step_num = int(step_str)
                step_ckpts.append((step_num, os.path.join(checkpoint_dir, f)))
            except ValueError:
                continue

    if step_ckpts:
        step_ckpts.sort(key=lambda x: x[0], reverse=True)
        return step_ckpts[0][1]

    # Also check for a base checkpoint
    base_path = os.path.join(checkpoint_dir, "edge_conversational.pt")
    if os.path.exists(base_path):
        return base_path

    return None


def load_checkpoint_for_resume(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[EdgeHVN, int, dict]:
    """Load a checkpoint and return (model, global_step, ckpt_dict).

    Rebuilds the EdgeHVN with the same architecture metadata, applies
    ternary weights, and loads the state dict.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = EdgeHVN(
        vocab_size=ckpt["vocab_size"],
        d_model=ckpt["d_model"],
        num_layers=ckpt["num_layers"],
        n_heads=ckpt.get("n_heads", 4),
        use_deltanet=ckpt.get("use_deltanet", False),
        deltanet_conv_kernel=ckpt.get("deltanet_conv_kernel", 4),
    )

    # Apply ternary to match the saved model
    all_linears = {name: mod for name, mod in model.named_modules()
                   if isinstance(mod, nn.Linear)}
    exclude_patterns = ("beta_proj", "alpha_proj", "decoder")
    target_names = [name for name in all_linears
                    if not any(p in name for p in exclude_patterns)]
    apply_ternary_to_model(model, target_names=target_names)

    model = model.to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    global_step = ckpt.get("global_step", 0)
    return model, global_step, ckpt


def train(
    model: EdgeHVN,
    dataloader: DataLoader,
    config: dict,
    device: torch.device,
    checkpoint_dir: str,
    max_steps: int | None = None,
    resume_step: int = 0,
) -> None:
    """Run the training loop with all configured settings.

    - bf16 mixed precision
    - Gradient accumulation to effective batch ≥ 32
    - AdamW with cosine LR schedule + warmup
    - Checkpoint every 500 steps
    - Log loss, PPL, LR every 50 steps (console + CSV)
    - Auto-resume from checkpoint
    """
    print("\n" + "=" * 70)
    print("STEP 5 — START TRAINING")
    print("=" * 70)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Model stats ---
    param_count = sum(p.numel() for p in model.parameters())
    dense_bytes = config.get("dense_bytes", param_count * 4)
    compressed_bytes = config.get("compressed_bytes", 0)
    compression_ratio = config.get("compression_ratio", 0)

    print(f"\n  Model parameter count:     {param_count:,}")
    print(f"  Ternary compression ratio: {compression_ratio:.2f}x")
    print(f"  Dense fp32 weight size:     {dense_bytes / 1024**2:.2f} MB")
    print(f"  Ternary weight size:        {compressed_bytes / 1024**2:.2f} MB")

    # --- Optimizer ---
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    warmup_steps = config["warmup_steps"]
    grad_accum = config["grad_accum"]

    use_fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        fused=use_fused,
    )
    if use_fused:
        print(f"  Optimizer: fused AdamW (CUDA)")

    # --- Cosine LR schedule with warmup ---
    total_steps = max_steps if max_steps is not None else 10_000
    # When resuming, the cosine schedule is reset — T_max should be the
    # remaining steps so the decay reaches ~0 at the end of training.
    scheduler_t_max = total_steps - resume_step if resume_step > 0 else total_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_t_max)

    # Warmup: linearly increase LR for the first warmup_steps, then let the
    # cosine scheduler take over. During warmup we override the LR manually;
    # after warmup we call scheduler.step() every optimizer step so the cosine
    # decay actually happens.
    warmup_done = resume_step >= warmup_steps
    def get_lr_scale(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    # --- Resume from checkpoint ---
    global_step = resume_step
    if global_step > 0:
        print(f"  Resuming from step {global_step}")
        # When resuming with a potentially different LR, reset the cosine
        # schedule to start fresh from the checkpoint (don't advance the
        # scheduler to the old step position). The model weights are carried
        # forward; the optimizer and schedule are fresh.
        print(f"  Resetting cosine schedule (fresh from checkpoint)")
        warmup_done = True  # skip warmup on resume — model is already trained
    else:
        warmup_done = False

    # --- CSV logging ---
    log_file = config["log_file"]
    file_exists = os.path.exists(log_file)
    csv_file = open(log_file, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not file_exists:
        csv_writer.writerow(["step", "epoch", "loss", "ppl", "lr", "timestamp"])
    csv_file.flush()

    # --- Training loop ---
    criterion = nn.CrossEntropyLoss()
    model.train()

    print(f"\n  Training for up to {total_steps} steps (effective batch = {config['batch_size'] * grad_accum})")
    print(f"  Logging every {config['log_every_steps']} steps | Checkpoint every {config['save_every_steps']} steps")
    print("-" * 70)

    epoch = 0
    start_time = time.time()
    step_start_time = start_time
    optimizer.zero_grad()
    accumulated_loss = 0.0

    # Per-source loss tracking (for diagnosing dataset quality variance)
    # Source IDs: 0=TinyStories, 1=OpenAssistant, 2=WikiText-103
    source_names = ["SlimPajama", "OpenAssistant", "WikiText-103"]
    source_loss_accum = [0.0, 0.0, 0.0]  # accumulated loss per source
    source_count = [0, 0, 0]  # sample count per source

    # For iterable datasets, we need to cycle through epochs manually
    data_iter = iter(dataloader)
    micro_step = 0  # micro-batches within a grad-accum window

    while True:
        # Stop at total_steps (== max_steps when given, else the 10k default that
        # sizes the LR schedule). Previously this checked `max_steps is not None`,
        # so an unset --max-steps left the loop unbounded — it ran past the LR
        # schedule's end forever at LR~=0. Tie the stop to the schedule horizon.
        if global_step >= total_steps:
            break

        # Get next batch — handle epoch boundaries
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(dataloader)
            try:
                batch = next(data_iter)
            except StopIteration:
                print(f"\n  Dataset exhausted after {epoch} epochs.")
                break

        # MixedLocalDataset returns (x, y, src_id); WikiText103LocalDataset returns (x, y)
        if len(batch) == 3:
            batch_x, batch_y, batch_src = batch
        else:
            batch_x, batch_y = batch
            batch_src = None

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        # Forward pass with bf16 autocast
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch_x)
            loss = criterion(
                logits.reshape(-1, GPT2_VOCAB_SIZE),
                batch_y.reshape(-1),
            )
            loss = loss / grad_accum

        # Backward pass
        loss.backward()
        accumulated_loss += loss.item()

        # Track per-source loss (for diagnosing dataset quality variance)
        if batch_src is not None:
            with torch.no_grad():
                per_sample_loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, GPT2_VOCAB_SIZE),
                    batch_y.reshape(-1),
                    reduction="none",
                ).reshape(batch_x.shape[0], -1).mean(dim=1)  # [B] per-sample loss
                for i in range(batch_x.shape[0]):
                    src = batch_src[i].item() if isinstance(batch_src[i], torch.Tensor) else batch_src[i]
                    source_loss_accum[src] += per_sample_loss[i].item()
                    source_count[src] += 1

        micro_step += 1

        # Optimizer step every grad_accum micro-batches
        if micro_step >= grad_accum:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Apply warmup LR scaling (during warmup only)
            if not warmup_done:
                lr_scale = get_lr_scale(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * lr_scale
                if global_step + 1 >= warmup_steps:
                    warmup_done = True

            optimizer.step()
            optimizer.zero_grad()
            micro_step = 0
            global_step += 1

            # After warmup, advance the cosine scheduler every step
            if warmup_done:
                scheduler.step()

            # Logging every log_every_steps
            if global_step % config["log_every_steps"] == 0:
                # accumulated_loss is the sum of grad_accum micro-batch losses,
                # each already divided by grad_accum — so the average loss is
                # accumulated_loss (not accumulated_loss * grad_accum).
                step_loss = accumulated_loss
                step_ppl = math.exp(min(step_loss, 20))
                current_lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - step_start_time
                steps_per_sec = config["log_every_steps"] / max(elapsed, 0.001)
                step_start_time = time.time()

                print(f"  Step {global_step:>6d} (epoch {epoch})  |  "
                      f"Loss: {step_loss:.4f}  |  PPL: {step_ppl:.2f}  |  "
                      f"LR: {current_lr:.6f}  |  "
                      f"{steps_per_sec:.1f} steps/s")

                # Per-source loss reporting
                if any(source_count):
                    src_str = "  |  ".join(
                        f"{source_names[s]}: {source_loss_accum[s]/max(1,source_count[s]):.4f}"
                        f" (n={source_count[s]})"
                        for s in range(3) if source_count[s] > 0
                    )
                    print(f"    Per-source: {src_str}")

                csv_writer.writerow([
                    global_step, epoch, f"{step_loss:.6f}", f"{step_ppl:.4f}",
                    f"{current_lr:.8f}", time.strftime("%Y-%m-%d %H:%M:%S"),
                ])
                csv_file.flush()

                # Reset per-source accumulators after logging
                source_loss_accum = [0.0, 0.0, 0.0]
                source_count = [0, 0, 0]

                # Estimate time per epoch based on first 10 steps
                if global_step == config["log_every_steps"]:
                    total_elapsed = time.time() - start_time
                    est_per_step = total_elapsed / global_step
                    print(f"  ⏱ Estimated {est_per_step:.2f}s/step | "
                          f"~{est_per_step * config['save_every_steps']:.0f}s per checkpoint interval")

            # Checkpoint every save_every_steps
            if global_step % config["save_every_steps"] == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"edge_conversational_step{global_step}.pt")
                _save_edge_checkpoint(model, ckpt_path, global_step=global_step)
                print(f"  ✓ Checkpoint saved: {ckpt_path}")

            accumulated_loss = 0.0

    # --- Final checkpoint ---
    final_ckpt_path = os.path.join(checkpoint_dir, "edge_conversational.pt")
    _save_edge_checkpoint(model, final_ckpt_path, global_step=global_step)

    csv_file.close()

    total_elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total steps:     {global_step}")
    print(f"  Total time:       {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Avg time/step:    {total_elapsed / max(global_step, 1):.3f}s")
    print(f"  Final checkpoint: {final_ckpt_path}")
    print(f"  Training log:     {log_file}")
    print(f"✓ Step 5 complete.")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EdgeHVN Conversational Training Pipeline (Issue #72)"
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: tiny budget, local WikiText-103 only, fast validation")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Maximum number of optimizer steps (default: unlimited)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from hardware scan")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Override sequence length from hardware scan")
    parser.add_argument("--d-model", type=int, default=None,
                        help="Override d_model from hardware scan")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Override number of layers from hardware scan")
    parser.add_argument("--n-heads", type=int, default=None,
                        help="Override number of heads from hardware scan")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Override gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default 3e-4)")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="checkpoints/edge_conversational",
                        help="Checkpoint directory")
    parser.add_argument("--data-dir", type=str,
                        default="data_cache",
                        help="Directory for cached dataset files (oasst1.txt, slimpajama_en.txt, etc.)")
    parser.add_argument("--slimpajama-dir", type=str,
                        default=None,
                        help="Directory containing SlimPajama parquet files (downloaded via hf CLI). "
                             "If not set, defaults to <data-dir>/slimpajama/data")
    parser.add_argument("--hf-cache-dir", type=str,
                        default=None,
                        help="HuggingFace cache directory for tokenizer/dataset downloads. "
                             "If not set, uses the default HF cache (~/.cache/huggingface).")
    args = parser.parse_args()

    # Force UTF-8 for Windows console
    if sys.platform == "win32":
        os.system("chcp 65001 >nul 2>&1")

    # === Step 1: Hardware Scan ===
    hw = hardware_scan()
    settings = hw["settings"]

    # Apply user overrides
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    if args.seq_len is not None:
        settings["seq_len"] = args.seq_len
    if args.d_model is not None:
        settings["d_model"] = args.d_model
    if args.num_layers is not None:
        settings["num_layers"] = args.num_layers
    if args.n_heads is not None:
        settings["n_heads"] = args.n_heads
    if args.grad_accum is not None:
        settings["grad_accum"] = args.grad_accum

    device = torch.device("cuda")

    # === Step 2: Dataset Pipeline ===
    # Use GPT-2 tokenizer as the baseline BPE tokenizer
    from transformers import GPT2Tokenizer
    print("\n  Loading GPT-2 tokenizer...")
    tokenizer_kwargs = {}
    if args.hf_cache_dir:
        tokenizer_kwargs["cache_dir"] = args.hf_cache_dir
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", **tokenizer_kwargs)
    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  GPT-2 tokenizer loaded: vocab_size = {tokenizer.vocab_size}")

    dataloader, dataset_stats = build_dataset_pipeline(
        tokenizer=tokenizer,
        seq_len=settings["seq_len"],
        batch_size=settings["batch_size"],
        smoke=args.smoke,
        data_dir=args.data_dir,
        slimpajama_dir=args.slimpajama_dir,
    )

    # === Step 3: Training Configuration ===
    model, config = build_model_and_config(
        settings=settings,
        tokenizer=tokenizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )
    config["lr"] = args.lr

    # === Step 4: Pre-Training Audit ===
    audit_passed = pre_training_audit(
        model=model,
        dataloader=dataloader,
        config=config,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    if not audit_passed:
        print("\n❌ Pre-training audit FAILED. Stopping before training.")
        print("   Fix the issues above and re-run.")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    # === Step 5: Start Training ===
    # Check for existing checkpoint to resume from
    resume_ckpt = find_latest_checkpoint(args.checkpoint_dir)
    resume_step = 0
    if resume_ckpt:
        print(f"\n  Found existing checkpoint: {resume_ckpt}")
        print(f"  Resuming training from this checkpoint...")
        model, resume_step, _ = load_checkpoint_for_resume(resume_ckpt, device)
        print(f"  Resumed at step {resume_step}")

    train(
        model=model,
        dataloader=dataloader,
        config=config,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        max_steps=args.max_steps,
        resume_step=resume_step,
    )

    # HuggingFace `datasets` streaming leaves background download threads that
    # cause "Fatal Python error: PyInterpreterState_Delete: remaining threads"
    # during normal Python shutdown. Force-exit to skip the broken thread join.
    # This is a well-known HuggingFace streaming issue on Windows.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        # HuggingFace `datasets` leaves background threads that crash the
        # interpreter on normal shutdown (Fatal Python error:
        # PyInterpreterState_Delete: remaining threads). Force-exit to
        # skip the broken thread join. This is a well-known HF issue on Windows.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)