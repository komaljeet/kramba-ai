"""
EdgeHVN Conversational Training Pipeline (Issue #72)
=====================================================

A complete training pipeline that builds a base conversational AI model
using the existing EdgeHVN + DeltaNet + ternary BitNet b1.58 implementation.

Steps (executed in order — each step stops and reports if it fails):
  1. Hardware scan         — nvidia-smi, RAM, torch device checks → optimal settings
  2. Dataset pipeline      — SlimPajama (50%) + OpenAssistant (35%) + WikiText-103 (15%)
  3. Training configuration — DeltaNet, precision (fp32/bf16/fp16, Issue #80), grad accum, cosine LR, checkpointing
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
import contextlib
import copy
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

# Force UTF-8 for the process — torch.compile's inductor codecache writes
# generated Python source to temp files, and on Windows cp1252 this crashes
# on Unicode characters in docstrings/comments (e.g. α, β in deltanet.py).
# PYTHONUTF8=1 makes all file I/O default to UTF-8, fixing the inductor crash.
os.environ.setdefault("PYTHONUTF8", "1")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from hvn.edge.model import EdgeHVN
from hvn.edge.ternary import apply_ternary_to_model, TernaryLinear
from hvn.optimizers import build_optimizer


def _save_edge_checkpoint(model: EdgeHVN, save_path: str, global_step: int = 0):
    """Save an EdgeHVN checkpoint with full architecture metadata.

    Unlike train._save_checkpoint, this correctly stores n_heads (which
    EdgeHVN doesn't expose as a model-level attribute, so the generic
    getattr fallback would default to 4 instead of the real value).
    """
    # Unwrap torch.compile's OptimizedModule so state_dict keys and attribute
    # access refer to the original EdgeHVN (no "_orig_mod." key prefix).
    model = getattr(model, "_orig_mod", model)
    # Find the first block to extract architecture details
    if hasattr(model, 'shared_block') and model.shared_block is not None:
        first_block = model.shared_block.block
    elif hasattr(model, 'blocks') and len(model.blocks) > 0:
        first_block = model.blocks[0]
        # #155: under per-type sharing each entry is a SharedBlockRef wrapper
        # holding the real block. n_heads and the other architecture fields live
        # on the block, and getattr(wrapper, 'n_heads', 4) would silently record
        # 4 -- a checkpoint that rebuilds as a different model with no error.
        first_block = getattr(first_block, 'block', first_block)
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
        "block_type": getattr(model, 'block_type', None),
        # Per-layer block plan + sliding window (hybrid token-recall). Old pure-HVN
        # checkpoints have neither field → load_* builds block_plan=None on resume.
        "block_plan": getattr(model, 'block_plan', None),
        "attn_window": getattr(model, 'attn_window', None),
        "ternary": any(isinstance(m, TernaryLinear) for m in model.modules()),
        "weightless": getattr(model, 'weightless', False),
        "weightless_seed": getattr(model, 'weightless_seed', 1234),
        "is_edge": True,
        "shared_weights": getattr(model, 'shared_weights', False),
        "tie_embeddings": getattr(model, 'tie_embeddings', False),
        # Issue #139: persisted so a resumed run rebuilds the SAME vocab. Read
        # from the model rather than a constant, and defaulted on load, so every
        # checkpoint written before this key existed still loads as 50257.
        "vocab_size": getattr(model, 'vocab_size', GPT2_VOCAB_SIZE),
        "eos_token_id": getattr(model, 'eos_token_id', 50256),
        "grad_checkpoint": getattr(model, 'grad_checkpoint', False),
        "global_step": global_step,
    }
    torch.save(ckpt, save_path)


# Rolling resume-state file (Issue #112). Deliberately NOT part of the model
# checkpoint: optimizer state is ~1.4 GiB for Muon at 355M (~2.8 GiB for AdamW),
# and archiving that alongside every `--save-every` checkpoint would fill the disk
# over a multi-week run. One file per checkpoint_dir, overwritten each save, holding
# only what is needed to continue the *optimization* rather than reproduce history.
RESUME_STATE_FILENAME = "resume_state.pt"


def _save_resume_state(checkpoint_dir: str, global_step: int, optimizer,
                       bcd_scheduler=None, lr_scheduler=None) -> None:
    """Persist optimizer + BCD-cycle + LR-schedule state so a restart continues.

    Written atomically (temp file + os.replace) because this file is overwritten in
    place — a crash mid-write would otherwise corrupt the only copy, and surviving
    crashes is the entire point of it.

    ``lr_scheduler`` state is saved as of #165. Without it the resume path
    reconstructed the schedule by replaying ``scheduler.step()``, which anchored
    the cosine on the LR that ``optimizer.load_state_dict`` had just restored —
    the already-decayed value — so the entire cosine curve was applied again on
    every resume. It compounded multiplicatively: two resumes in the 48M code run
    left the LR at 41% of intended by step 20000, and the training curve showed
    nothing. Saving the scheduler's own ``last_epoch`` removes the replay
    entirely.
    """
    state = {
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict(),
        "bcd_state_dict": bcd_scheduler.state_dict() if bcd_scheduler is not None else None,
        "sched_state_dict": (lr_scheduler.state_dict()
                             if lr_scheduler is not None else None),
    }
    path = os.path.join(checkpoint_dir, RESUME_STATE_FILENAME)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def build_cosine_scheduler(optimizer, total_steps: int, resume_step: int = 0,
                           schedule: str = "cosine"):
    """Cosine schedule that CONTINUES across a resume instead of restarting (Issue #112).

    The old behaviour built a fresh cosine over the *remaining* steps, so every
    restart snapped the LR back to peak — a sawtooth for any run restarted more than
    once, and the 355M pretrain is expected to restart many times.

    Advancing by calling ``step()`` ``resume_step`` times looks crude next to passing
    ``last_epoch=resume_step-1``, and that was the first attempt — it does not work.
    ``CosineAnnealingLR.get_lr()`` is *recursive*: it derives the next LR from the
    group's current LR rather than closed-form from ``last_epoch``, so constructing
    with ``last_epoch`` leaves the LR sitting at its peak (caught by
    ``tests/test_resume_state.py``). Replaying the recursion is what actually
    reproduces the uninterrupted trajectory, and ~100K cheap iterations is nothing
    against a multi-week run.
    """
    if schedule == "constant":
        # #159: a scaling-ratio study reads held-out loss off checkpoints taken
        # DURING a run. Under a cosine fitted to the run's end, the checkpoint at
        # 20 tokens/param is not "a model trained on a 20x budget" -- it is one
        # partway through a 40x schedule, still at a high LR and mid-descent,
        # so every intermediate point is biased against the properly-annealed
        # model it is standing in for, and the small budgets are biased most.
        # Hoffmann et al. hit the same trap and had to re-run with matched
        # schedules. A constant LR makes every checkpoint a valid "trained to X
        # tokens" model at the cost of slightly worse absolute loss.
        # factor=1.0 leaves the LR untouched; step() stays a no-op.
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0,
                                                   total_iters=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    if resume_step > 0:
        import warnings
        with warnings.catch_warnings():
            # Replaying the schedule before any optimizer.step() trips PyTorch's
            # ordering warning; it is spurious here — we are fast-forwarding, not
            # training.
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(min(resume_step, total_steps)):
                scheduler.step()
    return scheduler


def _load_resume_state(checkpoint_dir: str, expected_step: int) -> dict | None:
    """Load the rolling resume state, but only if it matches the checkpoint.

    Returns None (explaining why) when the file is absent, stale, or unreadable.
    A mismatch is normal — e.g. resuming from an older archived checkpoint than the
    last rolling save — and must degrade to a fresh optimizer rather than silently
    pair one step's weights with another step's momentum.
    """
    path = os.path.join(checkpoint_dir, RESUME_STATE_FILENAME)
    if not os.path.exists(path):
        print(f"  [resume] no {RESUME_STATE_FILENAME} — optimizer/BCD state starts fresh")
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [resume] {RESUME_STATE_FILENAME} unreadable ({e}) — optimizer starts fresh")
        return None
    saved_step = state.get("global_step", -1)
    if saved_step != expected_step:
        print(f"  [resume] {RESUME_STATE_FILENAME} is at step {saved_step} but the checkpoint "
              f"is at {expected_step} — ignoring it, optimizer starts fresh (pairing "
              f"mismatched weights and momentum would be worse than restarting)")
        return None
    return state


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

# Module-level torch.profiler handle; set by main() when --profile is passed so
# the training loop can call ._PROF.step() once per optimizer step (see train()).
_PROF = None


def _fmt_profile_table(prof, row_limit: int = 25) -> str:
    """Render torch.profiler key_averages as a self-device-time-sorted table.

    Robust across torch versions: EventList.table()'s kwargs change between
    releases (torch 2.12 rejects ``sort=``), so we read the Event attributes
    directly and sort by self device (GPU) time — the metric that attributes the
    bottleneck to the op whose own kernels actually burn the GPU.
    """
    ka = prof.key_averages()

    def _self_dev(e):
        for a in ("self_device_time_total", "self_cuda_time_total",
                  "device_time_total", "cuda_time_total"):
            v = getattr(e, a, None)
            if v:
                return float(v)
        return 0.0

    def _cpu(e):
        return float(getattr(e, "self_cpu_time_total", 0.0) or getattr(e, "cpu_time_total", 0.0) or 0.0)

    def _cnt(e):
        return int(getattr(e, "count", 0) or 0)

    rows = sorted(ka, key=_self_dev, reverse=True)[:row_limit]
    tot = sum(_self_dev(e) for e in ka) or 1.0
    out = [f"{'self GPU us':>14} {'%GPU':>6} {'cpu us':>12} {'count':>8}  key",
           "-" * 80]
    for e in rows:
        d = _self_dev(e)
        out.append(f"{d:>14.0f} {100 * d / tot:>5.1f}% {_cpu(e):>12.0f} {_cnt(e):>8d}  {e.key}")
    return "\n".join(out)


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


def _load_fineweb_edu(
    cache_dir: str,
    fineweb_edu_dir: str | None = None,
    max_bytes: int | None = None,
) -> str | None:
    """Extract text from locally-downloaded FineWeb-Edu parquet files
    (Issue #120) into a flat .txt file, mirroring
    `_download_language_foundation`'s pattern for SlimPajama.

    Does NOT download anything itself -- acquisition is a separate,
    explicit step (`scripts/download_fineweb_edu.py`, resumable, chosen
    over inline download here because a 100-300+ file pull can take
    hours and shouldn't silently block a training script that's supposed
    to start training). Returns `None` (not an error) if no parquet
    files are found, so callers can treat FineWeb-Edu as an optional
    source -- e.g. a `--pretrain-mix` config that includes a nonzero
    FineWeb-Edu share on a machine that hasn't run the download script
    yet should fail with a clear error at the mix-building step, not an
    obscure one here.

    Why a SEPARATE source/bin from SlimPajama, not just more SlimPajama
    text: `_tokenize_text_to_bin`'s cache is keyed only on whether the
    output .bin already exists -- appending more raw text to SlimPajama's
    existing 1.32B-token cache wouldn't be picked up without also
    invalidating that cache (discarding already-tokenized work). A
    distinct fineweb_edu.bin avoids that risk entirely and keeps
    FineWeb-Edu's higher curation quality separately identifiable in the
    mix (see docs/training.md's Issue #120 section for the quality
    rationale), rather than blending two differently-sourced corpora
    silently.

    Parameters
    ----------
    max_bytes : int or None
        Cap on total bytes extracted (None = use everything downloaded).
    fineweb_edu_dir : str or None
        Directory to scan recursively for `*.parquet` files. Defaults to
        `<cache_dir>/fineweb_edu` (where `scripts/download_fineweb_edu.py`
        places them, under `sample/<config>/`).
    """
    out_path = os.path.join(cache_dir, "fineweb_edu_en.txt")
    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        print(f"    FineWeb-Edu cached: {out_path} ({size / 1024**2:.1f} MB)")
        return out_path

    parquet_dir = fineweb_edu_dir or os.path.join(cache_dir, "fineweb_edu")
    if not os.path.exists(parquet_dir):
        print(f"    FineWeb-Edu not found at {parquet_dir} -- skipping "
              f"(run scripts/download_fineweb_edu.py first to use this source)")
        return None

    parquet_files = sorted(
        os.path.join(root, f)
        for root, _dirs, files in os.walk(parquet_dir)
        for f in files
        if f.endswith(".parquet")
    )
    if not parquet_files:
        print(f"    FineWeb-Edu directory {parquet_dir} exists but has no "
              f".parquet files -- skipping")
        return None

    import pandas as pd
    print(f"    Found {len(parquet_files)} FineWeb-Edu parquet files in {parquet_dir}")
    bytes_seen = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for pf in parquet_files:
            if max_bytes is not None and bytes_seen >= max_bytes:
                break
            print(f"    Reading {os.path.basename(pf)}...")
            df = pd.read_parquet(pf, columns=["text"])
            for text in df["text"]:
                if max_bytes is not None and bytes_seen >= max_bytes:
                    break
                if not text or not text.strip():
                    continue
                f.write(text)
                f.write("\n")
                bytes_seen += len(text.encode("utf-8"))
            print(f"      {bytes_seen / 1024**3:.2f} GB extracted so far...")
    print(f"    FineWeb-Edu saved: {out_path} ({os.path.getsize(out_path) / 1024**2:.1f} MB)")
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


from hvn.vocab import GPT2_EOS_ID, resolve_eos_id  # noqa: E402


def _tokenizer_signature(tokenizer) -> str:
    """A short, stable identity for whichever tokenizer produced a .bin.

    Name plus vocab size: enough to catch every realistic mix-up (GPT-2 vs a
    32K custom vocab) without depending on a hash of the merge table, which
    would churn on cosmetic retrains.
    """
    name = getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__
    # Normalise path separators before comparing. `name_or_path` is a filesystem
    # path for a local tokenizer, and on Windows the same directory reaches this
    # function as both "data_cache/tokenizer_32k_fim" and
    # "data_cache\tokenizer_32k_fim" depending on how it was typed. Comparing the
    # raw strings then rejects a perfectly valid cache -- observed refusing a
    # 121M-token FIM tokenization whose ONLY difference was a backslash. The
    # guard is meant to catch a different vocabulary, not a different way of
    # spelling the same directory.
    name = str(name).replace("\\", "/").rstrip("/")
    size = getattr(tokenizer, "vocab_size", None)
    if size is None and hasattr(tokenizer, "get_vocab_size"):
        size = tokenizer.get_vocab_size()
    return f"{name}/{size}"


def _fim_signature(fim: dict | None) -> str:
    """Cache-signature fragment for the FIM configuration (#136).

    Returns "" when FIM is off, so every pre-#136 .bin keeps its existing
    signature and is not needlessly invalidated. When FIM is on the fragment is
    appended, which means a cache built WITHOUT the transform can never be
    silently reused for a FIM run -- the same failure #139 closed for the
    tokenizer, in the same place, for the same reason: the ids look perfectly
    valid, they are just the wrong data.
    """
    if not fim:
        return ""
    # ``v2`` is the gap-density fix (#164): short, mostly line-aligned middles,
    # one gap per ~2000-character unit instead of one per document. A v1 cache
    # has the SAME rate/spm/seed and is silently the old shape -- the very
    # substitution this signature exists to prevent -- so the version is part of
    # the key, and must be bumped by anything that changes the transform.
    from hvn.fim import FIM_TRANSFORM_VERSION
    return (f"|fim{fim.get('rate', 0.5)}"
            f"-spm{fim.get('spm_rate', 0.5)}"
            f"-seed{fim.get('seed', 0)}"
            f"-{FIM_TRANSFORM_VERSION}")


def _fim_blockwise(text: str, fim: dict, rng) -> str:
    """Apply FIM per DOCUMENT within a block of concatenated documents.

    Documents are separated by the corpus separator (``<|endoftext|>``). The
    transform is meaningless applied to a 4 MB slab of many files -- it would
    teach the model to relocate whole files, not to fill a gap in one.

    Each document is further cut into ~1200-character UNITS, each gapped
    independently (#164). One gap per file gave one jump-back token per ~4,800
    corpus tokens, too sparse for the model to learn from -- see
    ``hvn.fim.split_document``. Units are joined with the plain separator so the
    byte content of the corpus is unchanged; only where the gaps fall differs.
    """
    from hvn.fim import apply_fim, iter_fim_units
    sep = fim.get("separator", "<|endoftext|>")
    parts = text.split(sep)
    out = []
    for part in parts:
        if not part:
            out.append(part)
            continue
        units = []
        for unit in iter_fim_units(part, rng):
            if rng.random() < fim.get("rate", 0.5):
                units.append(apply_fim(
                    unit, rng, spm=rng.random() < fim.get("spm_rate", 0.5)))
            else:
                units.append(unit)
        out.append("".join(units))
    return sep.join(out)


def _encode_parallel(tokenizer, text: str, n_chunks: int) -> list:
    r"""Encode via the Rust backend's multi-core ``encode_batch`` (Issue #139).

    ``tokenizer.encode`` is one Python call into a single-threaded path, so a
    multi-GB re-tokenize pegs one core and leaves the rest idle. The Rust
    backend parallelises ``encode_batch`` across cores.

    **Split points are chosen so the result is identical to encoding the whole
    string.** Chunks begin AT a newline rather than after one, because BPE
    merges of the form ``"\n    "`` — a newline plus an indent — are exactly what
    a code-trained vocabulary learns. Splitting after the newline would break
    that merge at every boundary and silently inflate the token count on
    precisely the corpus this vocabulary exists for.

    Falls back to the plain path for slow tokenizers, which have no backend.
    """
    if n_chunks <= 1 or len(text) < 2 * n_chunks:
        return tokenizer.encode(text)
    backend = (getattr(tokenizer, "backend_tokenizer", None)
               or getattr(tokenizer, "_tokenizer", None))
    if backend is None or not hasattr(backend, "encode_batch"):
        return tokenizer.encode(text)

    target = len(text) // n_chunks
    parts, start = [], 0
    for _ in range(n_chunks - 1):
        cut = text.find("\n", start + target)
        if cut == -1 or cut <= start:
            break
        parts.append(text[start:cut])
        start = cut                      # next chunk BEGINS with the newline
    parts.append(text[start:])
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return tokenizer.encode(text)

    ids = []
    for enc in backend.encode_batch(parts, add_special_tokens=False):
        ids.extend(enc.ids)
    return ids


def _tokenize_text_to_bin(
    text_path: str,
    bin_path: str,
    tokenizer,
    max_bytes: int | None = None,
    block_chars: int = 4_000_000,
    encode_workers: int | None = None,
    fim: dict | None = None,
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
    # Issue #139: the cache is keyed on the .bin existing, and NOTHING recorded
    # which tokenizer produced it. Switching vocab would therefore reuse ids
    # from the old vocabulary in silence — training on token 45,000 in a 32K
    # model, or worse, on ids that are merely WRONG rather than out of range.
    # A sidecar records the tokenizer's identity, and a mismatch refuses to
    # proceed rather than guessing which one is correct.
    # Issue #157: the signature must also identify the SOURCE CORPUS. The bin
    # name comes from `_bin("code")` -- the literal string "code" -- so pointing
    # --code-corpus at a different file reused whatever was cached first, in
    # silence. Observed: a run launched against stack_code.txt (11.20 GB)
    # announced that path and then loaded a 121,122,231-token cache built the
    # previous day from a different file -- 4% of the intended data, which at
    # the planned step count would have been 2.7 passes over it (a memorisation
    # regime) rather than one pass over fresh tokens. The tokenizer guard below
    # was already loud about the wrong VOCABULARY; nothing checked the wrong
    # TEXT. Path plus size plus mtime is enough to catch a swapped or rebuilt
    # corpus without hashing multiple GB on every start.
    try:
        _st = os.stat(text_path)
        src_sig = (f"|src={os.path.basename(text_path)}"
                   f":{_st.st_size}:{int(_st.st_mtime)}")
    except OSError:
        src_sig = f"|src={os.path.basename(text_path)}"
    sig = _tokenizer_signature(tokenizer) + _fim_signature(fim) + src_sig
    sig_path = bin_path + ".tokenizer"
    if os.path.exists(bin_path) and os.path.getsize(bin_path) > 0:
        prev = None
        if os.path.exists(sig_path):
            with open(sig_path, encoding="utf-8") as fh:
                prev = fh.read().strip()
        # A sidecar written before the corpus fingerprint existed carries only
        # the tokenizer/FIM part. Comparing it against the full signature marks
        # every pre-#164 cache as a corpus mismatch and forces a multi-hour
        # re-tokenize of data that is perfectly valid -- slimpajama.bin (2.6 GB)
        # and wikitext.bin among them. The fingerprint cannot be checked for
        # those: nothing recorded which file produced them. So compare only the
        # part the sidecar actually has, and upgrade it in place so the check
        # does apply from here on.
        if prev is not None and "|src=" not in prev and "|src=" in sig:
            if prev == sig.split("|src=")[0]:
                with open(sig_path, "w", encoding="utf-8") as fh:
                    fh.write(sig)
                print(f"    (sidecar for {os.path.basename(bin_path)} predates "
                      f"corpus fingerprinting; adopted and upgraded)")
                prev = sig

        if prev is not None and prev != sig:
            same_tok = prev.split("|src=")[0] == sig.split("|src=")[0]
            what = ("a DIFFERENT SOURCE CORPUS" if same_tok
                    else "a different tokenizer/FIM setting")
            raise RuntimeError(
                f"{bin_path} was built from {prev!r} but this run uses "
                f"{sig!r} -- {what}. Reusing it would train on the wrong data "
                f"silently: the run would announce your corpus and then consume "
                f"whatever was cached first. Delete the .bin (and its "
                f".tokenizer sidecar) to re-tokenize, or point at the corpus "
                f"that produced the cache."
            )
        if prev is None:
            # Pre-#139 cache with no sidecar. It can only be GPT-2 — that is the
            # only tokenizer this pipeline ever used — so adopt that rather than
            # forcing a needless multi-hour re-tokenize, but say so.
            print(f"    (unlabelled cache {os.path.basename(bin_path)} assumed "
                  f"GPT-2; writing sidecar)")
            # Compare only the TOKENIZER part. #164 appended a corpus
            # fingerprint (|src=name:size:mtime) and a FIM fragment to the
            # signature; a pre-#139 cache has neither, so comparing the whole
            # string would reject every legacy cache and force a multi-hour
            # re-tokenize that this branch exists specifically to avoid. The
            # fingerprint cannot be checked here in any case -- there is no
            # record of which corpus produced an unlabelled .bin.
            tok_part = sig.split("|", 1)[0]
            with open(sig_path, "w", encoding="utf-8") as fh:
                fh.write(sig)
            if tok_part != "gpt2/50257":
                raise RuntimeError(
                    f"{bin_path} predates tokenizer tracking and is GPT-2, but "
                    f"this run uses {sig!r}. Delete it to re-tokenize.")
        n = os.path.getsize(bin_path) // 2
        print(f"    Tokenized cache hit: {bin_path} ({n:,} tokens)")
        return n

    if encode_workers is None:
        encode_workers = max(1, os.cpu_count() or 4)
    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    tmp_path = bin_path + ".tmp"
    total_tokens = 0
    bytes_read = 0
    blocks = 0
    t0 = time.time()
    # FIM is per-DOCUMENT, but blocks are a fixed number of CHARACTERS, so a
    # document routinely straddles a block boundary. `pending` carries the
    # trailing partial document into the next block. Without it, the tail of one
    # file and the head of the next would be transformed as a single document --
    # teaching the model to fill gaps that span two unrelated files, which is
    # both wrong and completely invisible in the output.
    import random as _random
    fim_rng = _random.Random(fim.get("seed", 0)) if fim else None
    fim_sep = (fim or {}).get("separator", "<|endoftext|>")
    pending = ""

    with open(text_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(tmp_path, "wb") as fout:
        while True:
            if max_bytes is not None and bytes_read >= max_bytes:
                break
            block = fin.read(block_chars)
            if not block:
                if fim and pending:          # final partial document
                    block = _fim_blockwise(pending, fim, fim_rng)
                    pending = ""
                    bytes_read += len(block.encode("utf-8"))
                    ids = _encode_parallel(tokenizer, block, encode_workers)
                    np.asarray(ids, dtype=np.uint16).tofile(fout)
                    total_tokens += len(ids)
                break
            bytes_read += len(block.encode("utf-8"))
            if fim:
                buf = pending + block
                head, sep_found, tail = buf.rpartition(fim_sep)
                if sep_found:
                    # Everything before the LAST separator is whole documents.
                    pending = tail
                    block = _fim_blockwise(head, fim, fim_rng) + fim_sep
                else:
                    # No separator in the buffer yet -- keep accumulating rather
                    # than transform a fragment as if it were a document.
                    pending = buf
                    continue
            ids = _encode_parallel(tokenizer, block, encode_workers)
            np.asarray(ids, dtype=np.uint16).tofile(fout)
            total_tokens += len(ids)
            blocks += 1
            if blocks % 20 == 0:
                print(f"      {os.path.basename(bin_path)}: {total_tokens:,} tokens "
                      f"({bytes_read / 1024**2:.0f} MB, {time.time() - t0:.0f}s)", flush=True)
    os.replace(tmp_path, bin_path)
    with open(sig_path, "w", encoding="utf-8") as fh:
        fh.write(sig)
    print(f"    Tokenized {os.path.basename(text_path)} -> {bin_path} "
          f"({total_tokens:,} tokens, {time.time() - t0:.0f}s)")
    return total_tokens


class _WindowDataset(Dataset):
    """Minimal (source_id, offset) window dataset over shared memmap sources.

    Yields (x, y, src_id) for each seq_len window, matching MixedLocalDataset's
    item format. Used for the token-disjoint held-out validation split.
    """

    def __init__(self, source_paths, src_ids, starts, seq_len, dtype=np.uint16):
        # Store backing file paths (absolute) + dtype, NOT the live memmap
        # handles, so the dataset is picklable across a Windows DataLoader spawn
        # pipe. numpy.memmap objects raise `OSError [Errno 22] Invalid argument`
        # when serialized through the spawn pipe (and truncate it, killing the
        # child with `_pickle.UnpicklingError: pickle data was truncated`); each
        # worker re-mmaps from disk in __setstate__ instead. (This is the fix for
        # the step-2000 first-val crash on the Chinchilla-floor pretrain.)
        self.source_paths = [os.path.abspath(p) for p in source_paths]
        self.dtype = np.dtype(dtype)
        self.src_ids = src_ids
        self.starts = starts
        self.seq_len = seq_len
        self.sources = [np.memmap(p, dtype=self.dtype, mode="r")
                        for p in self.source_paths]

    def __getstate__(self):
        # Drop the live memmap handles (un-picklable on the Windows spawn pipe);
        # the worker re-opens them from source_paths in __setstate__.
        state = self.__dict__.copy()
        state.pop("sources", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.sources = [np.memmap(p, dtype=self.dtype, mode="r")
                        for p in self.source_paths]

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
        fineweb_edu_dir: str | None = None,
        fineweb_edu_token_bytes: int | None = None,
        max_repeat: int = 3,
        mix: tuple[float, float, float, float] | None = None,
        val_fraction: float = 0.0,
        code_corpus: str | None = None,
        fim: dict | None = None,
    ):
        """Build the mixed dataset from on-disk memory-mapped token files.

        Each source is tokenized once to a cached uint16 .bin (streaming — flat
        RAM) and memory-mapped read-only. Samples are seq_len windows referenced
        by (source_id, start_offset); only the windows actually read are paged in
        from disk, so RAM stays flat no matter how large the corpus is.

        `slimpajama_token_bytes` caps how much SlimPajama text is tokenized (the
        full .txt may be much larger; we only need a slice). `fineweb_edu_token_bytes`
        does the same for FineWeb-Edu (Issue #120) — `None` means use everything
        that's been downloaded via `scripts/download_fineweb_edu.py`. `max_repeat`
        bounds how many times any single source may repeat to satisfy the mix,
        preventing a small source (e.g. OASST) from being duplicated dozens of
        times against a multi-GB source — see Issue #119: this doesn't just
        affect that one source, it caps the WHOLE dataset's token budget, so
        `build_dataset`/`MixedLocalDataset` prints a clear warning identifying
        whether the actual constraint is a mix-ratio bind or a real data-volume
        limit.
        """
        self.seq_len = seq_len
        # Pretrain source mix (SlimPajama, OpenAssistant, WikiText-103,
        # FineWeb-Edu — Issue #120 added the 4th). Default is the original
        # conversational mix (0.50/0.35/0.15/0.0, i.e. FineWeb-Edu off unless
        # explicitly requested — it requires running
        # scripts/download_fineweb_edu.py first). A knowledge-heavy mix e.g.
        # (0.85/0.0/0.15/0.0) DROPS OpenAssistant from PRETRAIN (it belongs in
        # the chat fine-tune stage) so the tiny ~16M-token OASST source stops
        # capping the dataset via the max_repeat bind (see total_cap below,
        # and Issue #119's warning). Zero-fraction sources are excluded from
        # the train index entirely.
        self.mix = mix if mix is not None else (0.50, 0.35, 0.15, 0.0)
        os.makedirs(cache_dir, exist_ok=True)
        tok_dir = os.path.join(cache_dir, "tokenized")

        # Use a fast tokenizer for the one-time encode pass: identical BPE ids to
        # the slow GPT2Tokenizer but far faster on large corpora. Fall back to the
        # provided tokenizer if the fast one can't be loaded.
        enc = tokenizer
        # Issue #139: only swap in fast GPT-2 when the caller actually GAVE us
        # GPT-2. The previous version overrode unconditionally, so passing a
        # custom tokenizer silently tokenized the entire corpus with GPT-2
        # anyway — the model would have been trained on ids from a vocabulary
        # it does not have, and nothing would have raised.
        _is_gpt2 = type(tokenizer).__name__.startswith("GPT2") or tokenizer is None
        if _is_gpt2:
            try:
                from transformers import GPT2TokenizerFast
                enc = GPT2TokenizerFast.from_pretrained("gpt2")
                enc.model_max_length = 10 ** 9  # we chunk ourselves; silence warnings
            except Exception as e:  # pragma: no cover - environment dependent
                print(f"    (fast tokenizer unavailable, using provided tokenizer: {e})")
        else:
            print(f"    Using caller's tokenizer: {type(tokenizer).__name__} "
                  f"(vocab {getattr(tokenizer, 'vocab_size', '?')})")

        # Issue #139: give non-GPT-2 vocabularies their own .bin files. The
        # signature sidecar already REFUSES to reuse a mismatched cache, which
        # is the safety property — but with a single filename per source, a
        # 32K run and a GPT-2 run cannot coexist, and an A/B between them would
        # mean re-tokenizing the whole corpus on every arm switch. GPT-2 keeps
        # the bare name so every existing cache on disk is still found.
        _tok_suffix = ""
        if not (type(enc).__name__.startswith("GPT2")
                and getattr(enc, "vocab_size", None) == GPT2_VOCAB_SIZE):
            _tok_suffix = f".v{getattr(enc, 'vocab_size', 'x')}"

        def _bin(name: str) -> str:
            return os.path.join(tok_dir, f"{name}{_tok_suffix}.bin")

        if _tok_suffix:
            print(f"  Tokenized caches suffixed '{_tok_suffix}' "
                  f"(custom tokenizer; GPT-2 caches untouched)")

        print("\n  Preparing datasets (tokenize once -> memmap, flat RAM)...")

        # Issue #136: a code model trains on code ALONE, so `code_corpus`
        # REPLACES the four-source mix rather than becoming a fifth share.
        # Short-circuiting here matters: falling through would download and
        # tokenize SlimPajama, OASST and WikiText for a run that uses none of
        # them -- hours of work whose only effect is to fill the disk.
        if code_corpus is not None:
            if not os.path.exists(code_corpus):
                raise FileNotFoundError(
                    f"code corpus {code_corpus!r} not found. Build it with "
                    f"scripts/download_the_stack.py then "
                    f"scripts/extract_the_stack.py.")
            print(f"    Code-only corpus: {code_corpus} "
                  f"({os.path.getsize(code_corpus) / 1024**3:.2f} GB)")
            if fim:
                print(f"    FIM: rate={fim.get('rate')} "
                      f"spm_rate={fim.get('spm_rate')} seed={fim.get('seed')}")
            else:
                print("    FIM: OFF")
            code_bin = _bin("code")
            _tokenize_text_to_bin(code_corpus, code_bin, enc, fim=fim)
            self._source_paths = [os.path.abspath(code_bin)]
            self._source_dtype = np.uint16
            self.source_names = ["Code"]
            self.sources = [np.memmap(code_bin, dtype=np.uint16, mode="r")]
            self.total_tokens = int(len(self.sources[0]))
            self.mix = (1.0,)
            self._build_code_index(seq_len, val_fraction)
            return

        # 1. SlimPajama-627B (50%) — language foundation
        lang_path = _download_language_foundation(slimpajama_max_bytes, cache_dir, slimpajama_dir)
        print(f"    Tokenizing SlimPajama (cap {slimpajama_token_bytes / 1024**2:.0f} MB)...")
        slim_bin = _bin("slimpajama")
        _tokenize_text_to_bin(lang_path, slim_bin, enc, max_bytes=slimpajama_token_bytes)

        # 2. OpenAssistant (35%)
        oasst_path = _download_oasst(cache_dir)
        print(f"    Tokenizing OpenAssistant...")
        oasst_bin = _bin("oasst")
        _tokenize_text_to_bin(oasst_path, oasst_bin, enc)

        # 3. WikiText-103 (15%) — local files already in the codebase
        local_wt103 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wikitext-103-v1")
        wt103_path = os.path.join(local_wt103, "wiki.train.tokens")
        wt_bin = None
        if os.path.exists(wt103_path):
            print(f"    Tokenizing WikiText-103 from {wt103_path}...")
            wt_bin = _bin("wikitext")
            _tokenize_text_to_bin(wt103_path, wt_bin, enc)
        else:
            print(f"    WikiText-103 not found locally, skipping")

        # 4. FineWeb-Edu (Issue #120) — curated, quality-filtered web corpus,
        # replacing/supplementing SlimPajama as the primary knowledge source
        # for larger (4B+) pretrain runs. Acquisition is a separate, explicit
        # step (scripts/download_fineweb_edu.py) since a full target pull can
        # take hours; _load_fineweb_edu gracefully returns None (not an
        # error) if nothing's been downloaded yet, same pattern as WikiText's
        # local-file check above. A DISTINCT .bin from SlimPajama's — see
        # _load_fineweb_edu's docstring for why merging into SlimPajama's
        # cache would silently discard it (cache is keyed on the .bin
        # existing, not on how much source text backs it).
        # Unlike the other three sources (all well under ~5 GB, so preparing
        # them unconditionally regardless of mix share is cheap and matches
        # the codebase's existing pattern), FineWeb-Edu can be hundreds of
        # GB once fully downloaded -- scanning/extracting/tokenizing it
        # unconditionally would make every dataset build pay that cost even
        # when its mix share is 0. Only attempt it when actually wanted.
        slim_r, oasst_r, wt_r, fineweb_r = self.mix
        fineweb_bin = None
        if fineweb_r > 0:
            fineweb_path = _load_fineweb_edu(cache_dir, fineweb_edu_dir, max_bytes=fineweb_edu_token_bytes)
            if fineweb_path is not None:
                print(f"    Tokenizing FineWeb-Edu...")
                fineweb_bin = _bin("fineweb_edu")
                _tokenize_text_to_bin(fineweb_path, fineweb_bin, enc, max_bytes=fineweb_edu_token_bytes)

        # --- Memory-map each tokenized source (read-only, paged on demand) ---
        # Source mix (SlimPajama, OpenAssistant, WikiText-103, FineWeb-Edu).
        # Zero-fraction sources are excluded from the train index, so dropping
        # OASST (0.0) stops the tiny ~16M-token source from capping the
        # dataset via the max_repeat bind. Source IDs are reassigned densely
        # over the included sources only.
        candidates = [("SlimPajama", slim_bin, slim_r),
                      ("OpenAssistant", oasst_bin, oasst_r),
                      ("WikiText-103", wt_bin, wt_r),
                      ("FineWeb-Edu", fineweb_bin, fineweb_r)]
        source_paths, ratios, self.source_names = [], [], []
        for _name, _path, _r in candidates:
            if _r > 0 and _path is None:
                # A nonzero share was requested but no data exists for this
                # source -- fail loudly and specifically here, not silently
                # (which would renormalise the OTHER sources to cover the
                # gap, quietly changing the mix the caller actually asked
                # for). WikiText's local-file check is exempt: it has always
                # silently skipped when absent (pre-existing behavior, not
                # changed here), but FineWeb-Edu is new and its most likely
                # failure mode (forgot to run the download script) deserves
                # a clear error, per Issue #120.
                if _name == "FineWeb-Edu":
                    raise FileNotFoundError(
                        f"--pretrain-mix requested a nonzero FineWeb-Edu share "
                        f"({_r}) but no FineWeb-Edu data has been downloaded. "
                        f"Run scripts/download_fineweb_edu.py first, or set "
                        f"FineWeb-Edu's share to 0."
                    )
            if _r > 0 and _path is not None:
                source_paths.append(_path)
                ratios.append(_r)
                self.source_names.append(_name)
        ratios = [r / sum(ratios) for r in ratios]  # renormalise over included sources

        # Store absolute backing file paths + dtype so the dataset is picklable
        # across a Windows DataLoader spawn pipe (numpy.memmap itself is not —
        # see _WindowDataset.__getstate__).
        self._source_paths = [os.path.abspath(p) for p in source_paths]
        self._source_dtype = np.uint16

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

        # Issue #119: the max_repeat bind on whichever mix source is smallest
        # (relative to its target share) silently caps the ENTIRE dataset's
        # total token budget -- this already happened once (OASST capped the
        # old 83M pretrain at ~135M tokens, undetected for a while) and would
        # recur with WikiText-103 (a fixed ~119M-token corpus, nothing more
        # to download) the moment a run targets Chinchilla-optimal volume for
        # a bigger model (e.g. ~80-120B tokens for 4-6B params) -- the cap
        # lands around ~2.4B tokens with WikiText still in the mix at 15%,
        # ~2-3% of that floor, and training would look normal (loss
        # decreasing) while silently never reaching enough fresh data. Print
        # this loudly, always, rather than let it stay silent until someone
        # does the arithmetic by hand after the fact.
        self.total_chunks = total_chunks  # exposed for callers/tests, e.g. #119's cap warning
        effective_tokens = total_chunks * self.seq_len
        equiv_params_m = effective_tokens / 20 / 1e6  # 20 tokens/param, this project's Chinchilla convention
        if total_cap < sum(avail):
            # A source's max_repeat bind is the ACTIVE constraint: there is
            # MORE raw data available than this mix could ever use at its
            # current share, because the smallest-relative-to-share source
            # would have to repeat past max_repeat to keep up. This is the
            # #119/OASST failure mode -- fixable by changing the MIX (drop/
            # reduce that source's share, or raise --max-repeat).
            binding_idx = min(
                range(len(avail)),
                key=lambda i: (max_repeat * avail[i]) / ratios[i] if avail[i] > 0 and ratios[i] > 0 else float("inf"),
            )
            print(
                f"[data mix] total pretrain budget capped at {effective_tokens:,} tokens "
                f"({effective_tokens / 1e9:.2f}B) by a MIX-RATIO bind -- binding source: "
                f"{self.source_names[binding_idx]!r} (share={ratios[binding_idx]:.2f}, "
                f"max_repeat={max_repeat}) would need to repeat past max_repeat to keep "
                f"up with the other source(s), even though more of THEM is available. "
                f"At the 20 tokens/param Chinchilla convention this project uses, that's "
                f"enough for a ~{equiv_params_m:.0f}M-param model -- if you're targeting "
                f"something bigger, THIS MIX will silently under-serve it regardless of "
                f"how much more data you download. Fix: drop/reduce the binding source's "
                f"share, or raise --max-repeat (weighing the overfitting risk of "
                f"repeating a small source that many times)."
            )
        else:
            # No source's repeat cap is actually binding -- every available
            # window from every source is being used, and the limit is
            # simply TOTAL DATA VOLUME across all sources combined, not the
            # mix ratio. Downloading more of any source (proportionate to
            # its share) raises this ceiling directly -- changing shares
            # will NOT help until raw volume grows past the point where one
            # source's max_repeat bind becomes the active constraint instead
            # (see the branch above).
            print(
                f"[data mix] total pretrain budget capped at {effective_tokens:,} tokens "
                f"({effective_tokens / 1e9:.2f}B) by DATA VOLUME, not the mix ratio -- every "
                f"available window from every source is already being used "
                f"({sum(avail):,} total), none of them hit their max_repeat={max_repeat} "
                f"limit. At the 20 tokens/param Chinchilla convention this project uses, "
                f"that's enough for a ~{equiv_params_m:.0f}M-param model -- if you're "
                f"targeting something bigger, download more data (proportionate to the "
                f"current mix shares); adjusting the mix ratio alone will not raise this "
                f"ceiling until raw volume grows enough for a source's repeat limit to "
                f"become the active constraint instead."
            )

        # Hold out the last `val_fraction` of each source's UNIQUE windows as a
        # token-disjoint validation set: those offsets never enter the train
        # index, so there is no leakage (whole seq_len windows are disjoint).
        starts_parts, src_parts = [], []
        val_starts_parts, val_src_parts = [], []
        for sid, (n_avail, r) in enumerate(zip(avail, ratios)):
            if n_avail == 0:
                continue
            offs = np.arange(n_avail, dtype=np.int64) * seq_len
            n_val = int(n_avail * val_fraction)
            train_offs = offs[: n_avail - n_val] if n_val > 0 else offs
            val_offs = offs[n_avail - n_val:] if n_val > 0 else offs[:0]
            n_train = len(train_offs)

            target = int(total_chunks * r)
            if target > 0 and n_train > 0:
                if target <= n_train:
                    sel = train_offs[:target]
                else:  # repeat (wrapping) to hit the mix target
                    sel = np.tile(train_offs, target // n_train + 1)[:target]
                starts_parts.append(sel)
                src_parts.append(np.full(sel.shape, sid, dtype=np.uint8))
            if n_val > 0:
                val_starts_parts.append(val_offs)
                val_src_parts.append(np.full(val_offs.shape, sid, dtype=np.uint8))

        self.starts = np.concatenate(starts_parts)
        self.src_ids = np.concatenate(src_parts)

        # Deterministic shuffle of the (source, offset) train index.
        perm = np.random.RandomState(42).permutation(len(self.starts))
        self.starts = self.starts[perm]
        self.src_ids = self.src_ids[perm]

        # Validation index — deterministically shuffled (fixed seed) so a capped
        # eval samples a representative per-source mix rather than one source.
        if val_starts_parts:
            self.val_starts = np.concatenate(val_starts_parts)
            self.val_src_ids = np.concatenate(val_src_parts)
            vperm = np.random.RandomState(1234).permutation(len(self.val_starts))
            self.val_starts = self.val_starts[vperm]
            self.val_src_ids = self.val_src_ids[vperm]
        else:
            self.val_starts = np.zeros(0, dtype=np.int64)
            self.val_src_ids = np.zeros(0, dtype=np.uint8)

        names = self.source_names
        counts = [int((self.src_ids == s).sum()) for s in range(len(self.sources))]
        tot = max(1, len(self.src_ids))
        print(f"\n    Mixed dataset: {len(self.starts):,} train samples "
              f"({len(self.val_starts):,} held-out val), seq_len={seq_len}")
        print(f"    Unique tokens on disk: {self.total_tokens:,}  |  windows/source: {avail}")
        print("    Per-source (train): " + "  ".join(
            f"{names[s]}={counts[s]:,} ({100 * counts[s] / tot:.1f}%)"
            for s in range(len(self.sources))))

    def _build_code_index(self, seq_len: int, val_fraction: float) -> None:
        """Window index for the single-source code corpus (#136).

        The held-out split is the CONTIGUOUS TAIL of the corpus, not a random
        sample of windows, and one window is dropped between train and val as a
        gap. Both choices matter:

        - A random split would put windows from the same source FILE on both
          sides. Code repeats itself heavily within a file (imports, boilerplate,
          the same helper called twenty times), so a random split measures
          memorisation of files the model has already read. That is the shape of
          the mistake retracted in #132.
        - Without the gap, the last train window and the first val window share
          no tokens but sit adjacent, and a model that has seen the train window
          has effectively seen the val window's immediate context.

        The corpus is written in language order, so the tail is SQL-heavy rather
        than representative. Recorded as a known limit: this split is for
        watching the loss curve, not for a language-stratified quality claim.
        """
        arr = self.sources[0]
        n_win = (len(arr) - seq_len - 1) // seq_len + 1 if len(arr) >= seq_len + 1 else 0
        if n_win <= 0:
            raise ValueError(
                f"code corpus has {len(arr):,} tokens, too few for seq_len="
                f"{seq_len}")
        starts = np.arange(n_win, dtype=np.int64) * seq_len

        n_val = int(n_win * val_fraction)
        if n_val > 0 and n_win - n_val - 1 > 0:
            train_starts = starts[: n_win - n_val - 1]   # -1 = the gap window
            val_starts = starts[n_win - n_val:]
        else:
            train_starts = starts
            val_starts = np.zeros(0, dtype=np.int64)

        perm = np.random.RandomState(42).permutation(len(train_starts))
        self.starts = train_starts[perm]
        self.src_ids = np.zeros(len(self.starts), dtype=np.uint8)

        if len(val_starts):
            vperm = np.random.RandomState(1234).permutation(len(val_starts))
            self.val_starts = val_starts[vperm]
            self.val_src_ids = np.zeros(len(self.val_starts), dtype=np.uint8)
        else:
            self.val_starts = np.zeros(0, dtype=np.int64)
            self.val_src_ids = np.zeros(0, dtype=np.uint8)

        self.total_chunks = len(self.starts)
        print()
        print(f"    Code dataset: {len(self.starts):,} train windows "
              f"({len(self.val_starts):,} held-out), seq_len={seq_len}, "
              f"{self.total_tokens:,} tokens "
              f"({self.total_tokens / 1e9:.2f}B)")

    def val_dataset(self) -> "_WindowDataset":
        """Return a Dataset over the held-out validation windows (token-disjoint
        from train), sharing the same memory-mapped sources."""
        return _WindowDataset(self._source_paths, self.val_src_ids, self.val_starts,
                              self.seq_len, self._source_dtype)

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

    def __getstate__(self):
        # Drop the live memmap handles (un-picklable on a Windows spawn pipe);
        # workers re-mmap from _source_paths in __setstate__.
        state = self.__dict__.copy()
        state.pop("sources", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.sources = [np.memmap(p, dtype=self._source_dtype, mode="r")
                        for p in self._source_paths]


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


def _check_fim_tokenizer(tokenizer) -> None:
    """Refuse --fim unless the tokenizer carries the sentinels as single tokens.

    Without them each sentinel encodes to EIGHT tokens. Training would proceed
    perfectly happily -- no error, a normal-looking loss curve -- while every FIM
    document carried 24 tokens of overhead and the model tried to learn that a
    particular 8-token sequence is structural punctuation. The failure is only
    visible days later as a model that is oddly bad at completion.

    Returns None so the caller can use it inline; raises on failure.
    """
    from hvn.fim import FIM_TOKENS
    bad = []
    for t in FIM_TOKENS:
        try:
            n = len(tokenizer.encode(t))
        except Exception:                      # pragma: no cover - defensive
            n = -1
        if n != 1:
            bad.append(f"{t} -> {n} tokens")
    if bad:
        raise SystemExit(
            "--fim requires a tokenizer with the FIM sentinels as single "
            "tokens, but: " + "; ".join(bad) + ".\n"
            "Build one with: python scripts/build_fim_tokenizer.py\n"
            "then pass: --tokenizer data_cache/tokenizer_32k_fim")
    return None


def build_dataset_pipeline(
    tokenizer,
    seq_len: int,
    batch_size: int,
    smoke: bool = False,
    data_dir: str = "data_cache",
    slimpajama_dir: str | None = None,
    fineweb_edu_dir: str | None = None,
    fineweb_edu_token_bytes: int | None = None,
    val_fraction: float = 0.0,
    num_workers: int = 0,
    mix: tuple[float, float, float, float] | None = None,
    slimpajama_token_bytes: int = 500 * 1024**2,
    max_repeat: int = 3,
    code_corpus: str | None = None,
    fim: dict | None = None,
) -> tuple[DataLoader, "DataLoader | None", dict]:
    """Build the mixed-source train (and held-out val) DataLoaders + stats.

    Returns (train_loader, val_loader, stats). ``val_loader`` is None when
    ``val_fraction`` is 0 or in smoke mode.

    Parameters
    ----------
    data_dir : str
        Directory for cached dataset files (oasst1.txt, slimpajama_en.txt).
    slimpajama_dir : str or None
        Directory containing SlimPajama parquet files. If None, defaults
        to <data_dir>/slimpajama/data.
    fineweb_edu_dir : str or None
        Directory containing FineWeb-Edu parquet files (Issue #120). If
        None, defaults to <data_dir>/fineweb_edu.

    Returns (dataloader, stats_dict).
    """
    print("\n" + "=" * 70)
    print("STEP 2 — DATASET PIPELINE")
    print("=" * 70)

    eff_mix = mix if mix is not None else (0.50, 0.35, 0.15, 0.0)
    mix_ratio = {"SlimPajama": eff_mix[0], "OpenAssistant": eff_mix[1],
                 "WikiText-103": eff_mix[2], "FineWeb-Edu": eff_mix[3]}
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
            slimpajama_token_bytes=slimpajama_token_bytes,
            fineweb_edu_dir=fineweb_edu_dir,
            fineweb_edu_token_bytes=fineweb_edu_token_bytes,
            max_repeat=max_repeat,
            mix=mix,
            val_fraction=val_fraction,
            code_corpus=code_corpus,
            fim=fim,
        )
        stats["source"] = "code_only" if code_corpus else "mixed_local"
        stats["total_tokens"] = dataset.total_tokens
        stats["num_samples"] = len(dataset)
        print(f"  Tokens: {stats['total_tokens']}")
        print(f"  Samples: {stats['num_samples']}")

    # Build DataLoader (always map-style now — no IterableDataset)
    # Issue #91: num_workers > 0 enables async batch preparation overlapping
    # with GPU compute. persistent_workers avoids per-epoch spawn overhead.
    # prefetch_factor=2 (default) keeps 2 batches ready per worker.
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        # Issue #105: a smaller final batch would be a fresh dynamic shape
        # torch.compile hasn't seen, forcing a recompile/guard-fail once per
        # epoch instead of reusing the steady-state compiled graph. The val
        # loader below isn't compiled, so it keeps drop_last=False (no data
        # loss there).
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        loader_kwargs.update(
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=2,
        )
    dataloader = DataLoader(dataset, **loader_kwargs)

    # Held-out validation loader (full mode only; deterministic order, no shuffle).
    val_loader = None
    val_starts = getattr(dataset, "val_starts", None)
    if not smoke and val_starts is not None and len(val_starts) > 0:
        # Validation runs infrequently (every --val-every steps) on a small
        # token-disjoint split, so async worker overlap buys little. Run it
        # single-process (num_workers=0): no spawn -> no memmap pickling across
        # the Windows spawn pipe -> the first-val crash that killed the
        # Chinchilla-floor pretrain at step 2000 cannot recur here. (The train
        # dataset's own pickle-safety covers its persistent workers.)
        val_loader = DataLoader(
            dataset.val_dataset(),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=0,
            persistent_workers=False,
        )
        stats["val_samples"] = int(len(val_starts))
        print(f"  Held-out val: {stats['val_samples']:,} samples")

    stats["num_batches"] = len(dataloader)
    print(f"\n  DataLoader ready: batch_size={batch_size}")
    print(f"✓ Step 2 complete.")
    return dataloader, val_loader, stats


# ---------------------------------------------------------------------------
# Step 3 — Training Configuration
# ---------------------------------------------------------------------------

def load_tokenizer(spec: str = "gpt2", **kwargs):
    """Load GPT-2, or a locally-trained BPE from a directory (Issue #139).

    ``spec`` is either ``"gpt2"`` or a path to a directory holding ``vocab.json``
    and ``merges.txt`` (what ``scripts/train_tokenizer.py`` writes). The custom
    tokenizer is wrapped in ``PreTrainedTokenizerFast`` so every caller keeps the
    same interface -- ``.encode``, ``.vocab_size``, ``.eos_token_id`` -- and
    nothing downstream needs to know which one it got.

    Returning a real tokenizer object rather than a vocab-size integer is what
    lets the rest of the pipeline stay honest: ``_tokenize_text_to_bin`` reads
    its identity for the cache signature, and the model reads ``eos_token_id``
    off it instead of assuming GPT-2's 50256.
    """
    if spec in ("gpt2", "", None):
        # Imported here, not at module scope: this module is imported by scripts
        # that never touch a tokenizer, and the transformers import is slow.
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained("gpt2", **kwargs)
        tok.pad_token = tok.eos_token
        return tok

    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    # #136: a directory may also carry tokenizer.json, the fast format, which
    # is how the FIM sentinels are persisted -- they are ADDED tokens and
    # cannot be expressed in vocab.json/merges.txt alone (the ByteLevel
    # pre-tokenizer would split them into bytes before BPE ever saw them).
    # Preferred when present, so a FIM-trained model always loads the
    # vocabulary it was trained on.
    fast_json = os.path.join(spec, "tokenizer.json")
    if os.path.isfile(fast_json):
        from tokenizers import Tokenizer as _FastTokenizer
        tok = PreTrainedTokenizerFast(
            tokenizer_object=_FastTokenizer.from_file(fast_json),
            eos_token="<|endoftext|>", bos_token="<|endoftext|>",
            unk_token="<|endoftext|>", pad_token="<|endoftext|>",
        )
        tok.name_or_path = spec
        tok.model_max_length = 10 ** 9
        return tok

    vocab = os.path.join(spec, "vocab.json")
    merges = os.path.join(spec, "merges.txt")
    if not (os.path.exists(vocab) and os.path.exists(merges)):
        raise FileNotFoundError(
            f"--tokenizer {spec!r} is not 'gpt2' and does not contain "
            f"vocab.json + merges.txt. Train one with "
            f"scripts/train_tokenizer.py.")
    backend = ByteLevelBPETokenizer(vocab, merges)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=getattr(backend, "_tokenizer", backend),
        eos_token="<|endoftext|>", bos_token="<|endoftext|>",
        unk_token="<|endoftext|>", pad_token="<|endoftext|>",
    )
    tok.name_or_path = spec
    tok.model_max_length = 10 ** 9      # we chunk ourselves
    return tok


def apply_fast_stack(model, triton_scan: bool = True,
                     megakernel_scan: bool = False, tf32: bool = True,
                     compile_model: bool = True, verbose: bool = True):
    """Put a built model into the execution mode training actually uses (#136).

    **The single implementation of what ``--fast`` means.** ``--fast`` is
    ``default=True`` on this script, but it was applied inline in ``main()``, so
    any other caller -- every measurement harness -- silently got an uncompiled,
    un-Tritonised model and reported throughput no real run would ever see.

    That is not hypothetical. `scripts/size_sweep.py` measured eager and
    published "#136's config spills at 105% of VRAM and needs 30.6 days". Run
    properly it fits at 82.6% and needs 9.0 -- a 3.4x error, in a number that was
    about to size a 9-day training run. The retraction is in docs/state.md.

    Note this canNOT be folded into ``build_model_and_config``: ``torch.compile``
    must be applied AFTER checkpoint resume, so that loading sees a plain module
    and ``_save_edge_checkpoint`` unwraps a known shape. Construction and
    execution mode are genuinely separate steps; the fix is to have one function
    for the second, not to merge it into the first.

    Returns the model -- possibly wrapped in an OptimizedModule, so call it after
    resume and before the training loop.
    """
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def _blocks(m):
        if getattr(m, "blocks", None) is not None:
            return m.blocks
        if getattr(m, "shared_block", None) is not None:
            return [m.shared_block.block]
        return []

    if triton_scan:
        n = 0
        for blk in _blocks(model):
            if hasattr(blk, "use_triton_scan"):
                blk.use_triton_scan = True
                n += 1
        if verbose:
            print(f"\n  Fused Triton scan enabled on {n} DeltaNet block(s) (#83).")

    if megakernel_scan and triton_scan:
        n = 0
        for blk in _blocks(model):
            if hasattr(blk, "use_megakernel_scan"):
                blk.use_megakernel_scan = True
                n += 1
        if verbose:
            print(f"\n  Design B blockwise-recompute backward enabled on {n} "
                  f"DeltaNet block(s) (#96) -- ~1.05x faster step under --fast.")
    elif megakernel_scan and not triton_scan and verbose:
        print("\n  WARNING: megakernel_scan requires triton_scan; ignoring "
              "(the sequential forward must already be the scan).")

    if compile_model:
        # Issue #158: torch.compile is a NET LOSS on this architecture. Measured
        # in the REAL training loop (seq 4096, grad_accum 8, grad-checkpoint on):
        #
        #     eager + triton scan + megakernel   8.66 s/step   GPU 98%
        #     per-block compile + those kernels 20.72 s/step   GPU 35%
        #     per-block compile, kernels OFF    21.42 s/step   GPU 36%
        #
        # 2.4x SLOWER, and removing the fused kernels does not help -- so it is
        # not the graph-break interaction #129 documented, it is compile itself.
        # The GPU signature is the tell: 36% mean but 96% MAX, i.e. it bursts to
        # full and then stalls. That is a gap BETWEEN kernels, not slow kernels.
        # Unverified hypothesis: recomputation under `--grad-checkpoint` forces a
        # sync per checkpointed segment, which at 16 blocks x 8 micro-batches is
        # 128 stalls per step -- invisible at 1 micro-batch, dominant at 8.
        #
        # A 4-step, 1-micro-batch microbenchmark said per-block compile was
        # +18%. It measured a shape training never runs. The real loop inverted
        # it. This comment exists so the microbenchmark is not trusted again.
        #
        # Whole-model compile is separately impossible here: Inductor's
        # `compute_ancestors` is O(N^2) in graph nodes and MemoryErrors at 16
        # layers (3 reproductions, seq 4096 and 2048, 6.9-9 GB free).
        #
        # This is the FOURTH independent time compile has lost on this project:
        # #129 (megakernel breaks fusion, 2 graph breaks), state.md:558
        # (--block-coordinate conflict, 6.73 vs 4.41 s/step at 355M), the OOM
        # above, and this. `--fast` therefore keeps bf16 + TF32 + Triton scan +
        # megakernel and SKIPS torch.compile.
        if verbose:
            print("\n  torch.compile SKIPPED (#158): measured 2.4x slower in "
                  "the real loop (20.72 vs 8.66 s/step). bf16 + TF32 + Triton "
                  "scan + megakernel are still on.")
    return model


def check_head_dim(d_model: int, n_heads: int, triton_scan: bool = True) -> None:
    """Fail NOW if the Triton scan cannot compile for this shape (#136).

    ``hvn/core/deltanet_triton.py`` does ``tl.arange(0, D)`` over the head
    dimension, and Triton requires that range to be a power of 2. The rule is
    documented in ``hvn/kernels/hamiltonian_kernel.py:293`` and was enforced
    nowhere, so a bad shape did not warn -- it raised an InductorError partway
    into the run, naming ``tl.arange`` and saying nothing about d_model or
    n_heads. On a 9-day pretrain that is a very expensive way to find out.

    d_model=768 with the common n_heads=8 gives head_dim 96 and dies. It wants
    12 heads. This turns that into one line at startup.
    """
    if not triton_scan:
        return
    if n_heads <= 0 or d_model % n_heads != 0:
        raise ValueError(
            f"d_model={d_model} is not divisible by n_heads={n_heads}")
    head_dim = d_model // n_heads
    if head_dim & (head_dim - 1):
        # Candidates whose head_dim is a power of 2 in the range the kernel is
        # happy with, preferring head_dim closest to 64 -- the conventional
        # choice and the one the #136 sweep measured. Returning a technically
        # valid but unusual shape (768 -> 6 heads of 128) is worse advice than
        # returning the standard one.
        cands = [h for h in range(1, d_model + 1)
                 if d_model % h == 0
                 and 16 <= d_model // h <= 128
                 and not ((d_model // h) & (d_model // h - 1))]
        cands.sort(key=lambda h: (abs((d_model // h) - 64), h))
        hint = (f" Try n_heads={cands[0]} (head_dim="
                f"{d_model // cands[0]})." if cands else "")
        raise ValueError(
            f"head_dim = d_model/n_heads = {d_model}/{n_heads} = {head_dim}, "
            f"which is not a power of 2. The Triton scan kernel does "
            f"tl.arange(0, {head_dim}) and will fail to compile under --fast."
            f"{hint}")


def build_model_and_config(
    settings: dict,
    tokenizer,
    device: torch.device,
    checkpoint_dir: str,
    arch: str = "hvn",
    apply_ternary: bool = True,
    attn_layers: list[int] | None = None,
    attn_window: int | None = None,
    grad_checkpoint: bool = False,
    tie_embeddings: bool = False,
    shared_weights: bool = False,
    lr_schedule: str = "cosine",
    chunked_loss: int = 0,
    optimizer: str = "muon",
    precision: str = "bf16",
    lr: float = 3e-4,
    vocab_size: int = GPT2_VOCAB_SIZE,
    eos_token_id: int | None = None,
) -> tuple[EdgeHVN, dict]:
    """Build the EdgeHVN model with DeltaNet (or transformer baseline) + config.

    Parameters
    ----------
    arch : {"hvn", "transformer", "hvn-hybrid"}
        "hvn" uses the DeltaNet block (default); "transformer" swaps in the
        matched causal-MHA baseline for the A/B (#73); "hvn-hybrid" builds a
        per-layer block_plan (DeltaNet trunk + sliding-window attention at
        ``attn_layers``) for token-recall. Only the block mix differs.
    apply_ternary : bool
        If False, keep fp32 weights (used for the fp32 A/B — no ternary either
        side). Default True preserves the original EdgeHVN behaviour.
    attn_layers : list[int] | None
        0-indexed layer positions for attention blocks (hvn-hybrid only).
    attn_window : int | None
        Sliding-window width for the attention blocks (hvn-hybrid only).

    Returns (model, training_config).
    """
    print("\n" + "=" * 70)
    print("STEP 3 — TRAINING CONFIGURATION")
    print("=" * 70)

    # #136: fail here rather than partway into the run. The Triton scan needs a
    # power-of-2 head_dim and nothing used to check it.
    check_head_dim(settings["d_model"], settings["n_heads"])

    config = {
        "arch": arch,
        "ternary": apply_ternary,
        "vocab_size": vocab_size,
        "use_deltanet": arch == "hvn",
        # Real values from the CLI. These used to be hardcoded here and only
        # overwritten by main() AFTER this function printed its banner, so the
        # banner reported "bf16" and "3e-4" whatever was actually passed --
        # the same defect that made the optimizer line always say "AdamW".
        "precision": precision,
        "batch_size": settings["batch_size"],
        "grad_accum": settings["grad_accum"],
        "seq_len": settings["seq_len"],
        "d_model": settings["d_model"],
        "num_layers": settings["num_layers"],
        "n_heads": settings["n_heads"],
        # Real optimizer name, passed in from the CLI. Was hardcoded "AdamW"
        # while main() only wrote args.optimizer into config AFTER this
        # function printed its banner -- so every log this project has ever
        # produced reported "AdamW" no matter what actually ran. That stale
        # banner caused a real misreading of the chinchilla20k/83m logs
        # (both of which in fact ran Muon) during #129.
        "optimizer": optimizer,
        "lr": lr,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "cosine_schedule": True,
        "lr_schedule": lr_schedule,
        "chunked_loss": chunked_loss,
        "save_every_steps": 500,
        "log_every_steps": 50,
        "checkpoint_dir": checkpoint_dir,
        "log_file": os.path.join(checkpoint_dir, "training_log.csv"),
    }

    print(f"\n  use_deltanet:      {config['use_deltanet']}")
    print(f"  precision:         {config['precision']}")
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
    print(f"\n  Building EdgeHVN model (arch={arch})...")
    block_plan = None
    if arch in ("hvn-hybrid", "hvn-original-hybrid"):
        if not attn_layers:
            raise SystemExit(f"--arch {arch} requires --attn-layers (e.g. '3,10')")
        attn_set = sorted(set(int(x) for x in attn_layers))
        for li in attn_set:
            if not 0 <= li < settings["num_layers"]:
                raise SystemExit(
                    f"--attn-layers position {li} out of range for "
                    f"num_layers={settings['num_layers']}"
                )
        # #166: hvn-original-hybrid swaps ONLY the recurrent block -- the
        # original HVNBlock instead of the published gated-DeltaNet one -- so a
        # comparison against hvn-hybrid isolates that single variable rather
        # than also changing whether attention layers are present.
        recurrent = "hvn" if arch == "hvn-original-hybrid" else "deltanet"
        block_plan = [
            "transformer" if i in attn_set else recurrent
            for i in range(settings["num_layers"])
        ]
        config["attn_layers"] = attn_set
        config["attn_window"] = attn_window
        print(f"  Hybrid block_plan: {block_plan} (window={attn_window})")
    model = EdgeHVN(
        # Issue #139: no longer hardcoded. At d_model=512 the two vocab tables
        # are 62% of an "83M" model, so this number is a first-class size knob,
        # not a constant. Defaults to GPT-2's 50257 so every existing caller and
        # every checkpoint on disk is unchanged.
        vocab_size=vocab_size,
        d_model=settings["d_model"],
        num_layers=settings["num_layers"],
        n_heads=settings["n_heads"],
        use_deltanet=(arch == "hvn"),
        deltanet_conv_kernel=4,
        block_type=("transformer" if arch == "transformer" else None),
        block_plan=block_plan,
        attn_window=attn_window,
        grad_checkpoint=grad_checkpoint,
        tie_embeddings=tie_embeddings,
        shared_weights=shared_weights,
        # #139 gave EdgeHVN an eos_token_id and main() has passed one ever since,
        # but this function never accepted the argument -- so every call through
        # main() raised TypeError at model construction. It went unnoticed
        # because every A/B harness since then (ab_vocab_on_code, ab_vocab_mixed,
        # size_sweep) calls build_model_and_config directly and never reaches
        # main(). The measurement scripts bypassing main() is exactly why main()
        # rotted. Resolved rather than defaulted: a 32K model's EOS is 0, and 0
        # is falsy, so `or` here would silently substitute GPT-2's 50256.
        eos_token_id=(eos_token_id if eos_token_id is not None
                      else resolve_eos_id(tokenizer=tokenizer)),
    )
    param_count = sum(p.numel() for p in model.parameters())
    config["param_count"] = param_count
    print(f"  Architecture: {arch}  |  total params: {param_count:,}")

    # Apply ternary BitNet b1.58 weights (skipped for the fp32 A/B)
    if apply_ternary:
        print(f"  Applying ternary BitNet b1.58 weight quantization...")
        all_linears = {name: mod for name, mod in model.named_modules()
                       if isinstance(mod, nn.Linear)}
        exclude_patterns = ("beta_proj", "alpha_proj", "decoder")
        target_names = [name for name in all_linears
                        if not any(p in name for p in exclude_patterns)]
        replaced = apply_ternary_to_model(model, target_names=target_names)
        print(f"  Ternary: replaced {len(replaced)} nn.Linear layers with TernaryLinear (STE)")
        print(f"  Excluded {len(all_linears) - len(replaced)} gate/decoder layers")
    else:
        print(f"  Ternary: DISABLED — fp32 weights (A/B mode)")

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
    # Runs on a THROWAWAY deepcopy of the model. The real `model` is about to be
    # torch.compile'd by the caller; an eager forward+backward on it BEFORE
    # compile perturbs the compiled graph (autograd/allocator state) and makes
    # training ~0.1 s/step slower (measured: 0.7 vs 0.81 steps/s, #95). Running
    # the audit on an identical copy leaves the real model pristine for compile
    # while still validating that the architecture + ternary STE forward/backward
    # works. The copy is freed immediately after.
    print("\n  [Check 2] Forward pass (EdgeHVN + DeltaNet + ternary)...")
    if config.get("block_coordinate", False):
        print("    ⚠ Skipped under --block-coordinate: this check deepcopies the whole")
        print("      model onto GPU and runs an UN-FROZEN backward pass across every")
        print("      param, before BlockCoordinateScheduler exists to bound memory --")
        print("      at large model sizes that alone can OOM a 6 GB card (Issue #112).")
        print("      BCD's own per-block freeze already bounds real-training risk; any")
        print("      real forward/backward issue will surface immediately on step 1.")
    else:
        audit_model = None
        try:
            audit_model = copy.deepcopy(model)
            x = batch_x.to(device)
            precision = config.get("precision", "bf16")
            use_amp = precision in ("fp16", "bf16")
            amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

            audit_model.eval()
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = audit_model(x)
            # Issue #139: compare against the MODEL's vocab, not a constant.
            # Hardcoding 50257 here made the audit reject every non-GPT-2 vocab
            # outright — the audit would have blocked a 32K run before step 1,
            # which is exactly the failure a pre-training audit is supposed to
            # prevent, not cause.
            expected_vocab = getattr(audit_model, "vocab_size", GPT2_VOCAB_SIZE)
            print(f"    Input shape:  {x.shape}")
            print(f"    Output shape: {logits.shape} (expected [{x.shape[0]}, {x.shape[1]}, {expected_vocab}])")
            if logits.shape[0] != x.shape[0] or logits.shape[1] != x.shape[1]:
                print(f"    ❌ Output shape mismatch!")
                all_passed = False
            elif logits.shape[2] != expected_vocab:
                print(f"    ❌ Vocab dimension mismatch: got {logits.shape[2]}, expected {expected_vocab}")
                all_passed = False
            else:
                print(f"    ✓ Forward pass correct")

            # Also test backward pass
            audit_model.train()
            y = batch_y.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = audit_model(x)
                loss = nn.CrossEntropyLoss()(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
            loss.backward()
            print(f"    ✓ Backward pass correct (loss = {loss.item():.4f})")

            # Check gradients exist on ternary weights
            has_grad = False
            for name, mod in audit_model.named_modules():
                if isinstance(mod, TernaryLinear) and mod.weight.grad is not None:
                    has_grad = True
                    break
            if has_grad:
                print(f"    ✓ Ternary weight gradients present (STE working)")
            else:
                print(f"    ⚠ No ternary weight gradients found (may be normal if all grads are None)")

            # Zero gradients (on the throwaway copy; the real model was never touched).
            audit_model.zero_grad()
        except Exception as e:
            print(f"    ❌ Forward/backward check failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
        finally:
            # Free the throwaway copy's VRAM before training allocates activations.
            del audit_model
            torch.cuda.empty_cache()

    # --- Check 3: Checkpoint save and load cycle ---
    print("\n  [Check 3] Checkpoint save/load cycle...")
    if config.get("block_coordinate", False):
        print("    ⚠ Skipped under --block-coordinate: this check rebuilds a SECOND full")
        print("      EdgeHVN and moves it onto GPU alongside the real model (no BCD/freeze")
        print("      applies here either) -- two full-precision weight copies resident at")
        print("      once can OOM a 6 GB card at large sizes, same class of issue as Check 2")
        print("      (Issue #112). The real training loop's own checkpoint saves/resumes")
        print("      exercise this same code path, so a real problem still surfaces there.")
    else:
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
            # Every field the checkpoint records must be replayed here, or the
            # audit rebuilds a DIFFERENT model and its own check fails.
            # block_plan/attn_window/tie_embeddings were saved but never passed:
            # invisible for arch=hvn (block_plan is None there), fatal for the
            # hybrid, which is the architecture this project actually trains.
            # The audit could not have passed on a hybrid since block_plan was
            # added -- and nothing noticed, because the audit only ever ran on
            # the plain arch.
            rebuilt = EdgeHVN(
                vocab_size=ckpt["vocab_size"],
                d_model=ckpt["d_model"],
                num_layers=ckpt["num_layers"],
                n_heads=ckpt.get("n_heads", 4),
                use_deltanet=ckpt.get("use_deltanet", False),
                deltanet_conv_kernel=ckpt.get("deltanet_conv_kernel", 4),
                block_type=ckpt.get("block_type", None),
                block_plan=ckpt.get("block_plan", None),
                attn_window=ckpt.get("attn_window", None),
                tie_embeddings=ckpt.get("tie_embeddings", False),
                eos_token_id=ckpt.get("eos_token_id", GPT2_EOS_ID),
                # #155: the fourth field to be forgotten here. A shared model's
                # parameters live at blocks.N.block.* plus two per-layer scalars,
                # so rebuilding dense makes load_state_dict report every key as
                # missing AND unexpected -- the audit failing on its own rebuild
                # rather than on the model.
                shared_weights=ckpt.get("shared_weights", False),
            )
            # Apply ternary to rebuilt model (same layers as original) — only if the
            # checkpoint was ternary (fp32 A/B runs skip it).
            if ckpt.get("ternary", True):
                all_linears = {name: mod for name, mod in rebuilt.named_modules()
                               if isinstance(mod, nn.Linear)}
                exclude_patterns = ("beta_proj", "alpha_proj", "decoder")
                target_names = [name for name in all_linears
                                if not any(p in name for p in exclude_patterns)]
                apply_ternary_to_model(rebuilt, target_names=target_names)
            rebuilt = rebuilt.to(device)
            rebuilt.load_state_dict(ckpt["model_state_dict"])
            print(f"    ✓ State dict loads into rebuilt model")

            # #155: a strict load CANNOT verify sharing. A model whose wrappers
            # each hold their OWN block has identical key names and an identical
            # state_dict size -- it loads perfectly clean while being a dense
            # model wearing shared clothing (measured: 184,020 unique params vs
            # 100,044 correct, both reporting 200,404 in the state dict). The
            # #117/#164 shape exactly. Counting DISTINCT block objects against
            # distinct plan types is what actually separates them; the
            # unique-vs-total parameter inequality is too weak and passes on the
            # broken model.
            if ckpt.get("shared_weights", False) and rebuilt.blocks is not None:
                plan = ckpt.get("block_plan") or []
                n_types = len(set(plan))
                n_objs = len({id(getattr(b, "block", b)) for b in rebuilt.blocks})
                if n_types and n_objs != n_types:
                    raise AssertionError(
                        f"shared_weights=True but {n_objs} distinct block objects "
                        f"exist for {n_types} block type(s) -- sharing did not "
                        f"engage; this model would train dense while reporting "
                        f"as shared")
                print(f"    ✓ Sharing verified: {n_objs} block object(s) for "
                      f"{n_types} type(s) across {len(plan)} layers")

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


def _check_resume_matches_request(requested, ckpt: dict, path: str, args) -> None:
    """Refuse to resume a checkpoint whose architecture is not what was asked for.

    Resuming replaces the model built from the CLI flags with one rebuilt from
    the checkpoint. That is correct when continuing the same run and catastrophic
    otherwise: a stale checkpoint in the default directory silently overrides
    --d-model, --num-layers, --arch and the vocabulary, and the only visible
    symptom is a parameter count nobody is reading closely.

    Observed on 2026-08-10: a request for d=896/16L (157M) trained a 28M model
    resumed from an unrelated 83M-line checkpoint, at step 100 of 120, with no
    warning. It also silently invalidates the pre-training audit, which ran
    against the discarded model.

    This is the mirror of #139's tokenizer-cache guard: same failure (stale
    artifact quietly preferred over the request), same fix (compare and refuse).
    """
    def _want(attr, default=None):
        v = getattr(args, attr, None)
        return default if v is None else v

    mismatches = []
    checks = [
        ("d_model", ckpt.get("d_model"), _want("d_model")),
        ("num_layers", ckpt.get("num_layers"), _want("num_layers")),
        ("n_heads", ckpt.get("n_heads"), _want("n_heads")),
        ("vocab_size", ckpt.get("vocab_size"),
         getattr(requested, "vocab_size", None)),
        ("tie_embeddings", bool(ckpt.get("tie_embeddings", False)),
         bool(_want("tie_embeddings", False))),
    ]
    for name, have, want in checks:
        if want is not None and have is not None and have != want:
            mismatches.append(f"{name}: checkpoint={have}, requested={want}")

    ckpt_plan = ckpt.get("block_plan")
    want_hybrid = getattr(args, "arch", None) == "hvn-hybrid"
    if want_hybrid and not ckpt_plan:
        mismatches.append("arch: checkpoint is not hybrid, requested hvn-hybrid")
    if ckpt_plan and not want_hybrid:
        mismatches.append(f"arch: checkpoint is hybrid, requested "
                          f"{getattr(args, 'arch', '?')}")

    if mismatches:
        raise SystemExit(
            f"\nRefusing to resume: {path}\n"
            f"  its architecture does not match what this run asked for:\n"
            + "".join(f"    - {m}\n" for m in mismatches)
            + "\n  Resuming would silently train the CHECKPOINT's model and "
              "discard your\n  flags -- and the pre-training audit you just "
              "passed was run against\n  the model being discarded.\n\n"
              "  Either point --checkpoint-dir at a fresh directory for this "
              "run, or\n  delete/move the checkpoint if you meant to start "
              "over.\n")


def load_checkpoint_for_resume(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[EdgeHVN, int, dict]:
    """Load a checkpoint and return (model, global_step, ckpt_dict).

    Rebuilds the EdgeHVN with the same architecture metadata (including the
    per-layer block_plan for hybrid checkpoints), applies ternary weights,
    and loads the state dict.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    block_plan = ckpt.get("block_plan", None)
    attn_window = ckpt.get("attn_window", None)
    model = EdgeHVN(
        vocab_size=ckpt["vocab_size"],
        d_model=ckpt["d_model"],
        num_layers=ckpt["num_layers"],
        n_heads=ckpt.get("n_heads", 4),
        use_deltanet=ckpt.get("use_deltanet", False),
        deltanet_conv_kernel=ckpt.get("deltanet_conv_kernel", 4),
        block_type=ckpt.get("block_type", None),
        block_plan=block_plan,
        attn_window=attn_window,
        tie_embeddings=ckpt.get("tie_embeddings", False),
        eos_token_id=ckpt.get("eos_token_id", 50256),
        grad_checkpoint=ckpt.get("grad_checkpoint", False),
        # #155: a shared checkpoint rebuilt as dense (or the reverse) loads
        # with a shape mismatch at best and trains a different architecture at
        # worst, so it is read back from the checkpoint like every other
        # structural field rather than taken from the current flags.
        shared_weights=ckpt.get("shared_weights", False),
    )

    # Apply ternary to match the saved model — only if it was ternary. Older
    # checkpoints predate the "ternary" flag, so default True for back-compat.
    if ckpt.get("ternary", True):
        all_linears = {name: mod for name, mod in model.named_modules()
                       if isinstance(mod, nn.Linear)}
        exclude_patterns = ("beta_proj", "alpha_proj", "decoder")
        target_names = [name for name in all_linears
                        if not any(p in name for p in exclude_patterns)]
        apply_ternary_to_model(model, target_names=target_names)

    model = model.to(device)
    # strict=False + report so a hybrid checkpoint resumes cleanly (the state
    # dict matches the rebuilt block_plan) and any drift is surfaced, not fatal.
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"  [resume] missing={len(missing)} unexpected={len(unexpected)} keys")
        if missing:
            print(f"    missing (first 8): {missing[:8]}")
        if unexpected:
            print(f"    unexpected (first 8): {unexpected[:8]}")

    global_step = ckpt.get("global_step", 0)
    return model, global_step, ckpt


def load_hybrid_trunk_into_model(
    model: EdgeHVN,
    trunk_ckpt_path: str,
    device: torch.device,
) -> int:
    """Warm-start a hybrid EdgeHVN's DeltaNet trunk from a pure-HVN checkpoint.

    Copies all matching trunk keys (token_embedding, final_norm, decoder, and the
    DeltaNet blocks at non-attention positions), leaves the new attention layers
    at fresh init, and skips the orphaned DeltaNet params at the attention
    positions (they no longer exist in the hybrid). Returns the trunk's step.
    """
    ckpt = torch.load(trunk_ckpt_path, map_location=device, weights_only=False)
    trunk_sd = ckpt["model_state_dict"]
    hybrid_sd = model.state_dict()

    matched, copied = [], 0
    skipped_attn_positions = set()
    for i, t in enumerate(model.block_plan):
        if t == "transformer":
            skipped_attn_positions.add(i)

    for key, tensor in trunk_sd.items():
        # Skip blocks at attention positions entirely — the hybrid has a fresh
        # TransformerBlock there, not the trunk's DeltaNet params.
        if key.startswith("blocks."):
            try:
                li = int(key.split(".")[1])
            except (IndexError, ValueError):
                li = -1
            if li in skipped_attn_positions:
                continue
        if key in hybrid_sd and hybrid_sd[key].shape == tensor.shape:
            hybrid_sd[key] = tensor.to(hybrid_sd[key].dtype)
            matched.append(key)
            copied += 1

    missing = [k for k in hybrid_sd if k not in matched]
    unexpected = [k for k in trunk_sd if k not in matched]
    model.load_state_dict(hybrid_sd, strict=True)

    print(f"  [hybrid-trunk-init] copied {copied} trunk keys from {trunk_ckpt_path}")
    print(f"    matched/copied: {copied}")
    print(f"    missing (fresh-init, mostly attention): {len(missing)} (first 8: {missing[:8]})")
    print(f"    unexpected (skipped, orphaned DeltaNet at attn positions): "
          f"{len(unexpected)} (first 8: {unexpected[:8]})")
    return ckpt.get("global_step", 0)


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int = 100,
             precision: str = "bf16"):
    """Held-out validation: mean loss overall + per-source.

    Returns (overall_loss, [slim, oasst, wiki] losses) or None if no val set.
    Deterministic (val_loader is unshuffled); capped at ``max_batches`` for speed.

    Parameters
    ----------
    precision : str
        Mixed-precision mode for eval (Issue #80): "fp32" (no autocast),
        "bf16", or "fp16". Should match training precision.
    """
    if val_loader is None:
        return None
    was_training = model.training
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="none")
    # GPU-side accumulators (deferred host sync): accumulate per-sample loss
    # and per-source sums/counts on-device, then sync ONCE at the end. The old
    # per-sample .item() loop forced a GPU->CPU sync per sample (B syncs /
    # batch); this mirrors the training loop's scatter_add pattern and drops
    # to a single host sync for the whole eval. Values are identical to the
    # per-sample sums, just synced less often.
    tot_loss_t = torch.zeros((), device=device)
    tot_n_t = torch.zeros((), device=device, dtype=torch.long)
    src_loss_t = torch.zeros(3, device=device)
    src_count_t = torch.zeros(3, device=device, dtype=torch.long)
    # Precision-aware autocast (Issue #80): fp32 = no autocast, bf16/fp16 = autocast.
    use_amp = precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y, src = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            logits = model(x)
            per_sample = crit(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            ).reshape(x.shape[0], -1).mean(dim=1)
        B = x.shape[0]
        tot_loss_t += per_sample.detach().sum()
        tot_n_t += B
        src_ids = (src.to(device=device, dtype=torch.long)
                   if isinstance(src, torch.Tensor)
                   else torch.tensor(src, device=device, dtype=torch.long))
        # Clamp out-of-range source IDs (defensive; matches the old `0 <= s < 3`
        # guard) so scatter_add never indexes off the end of the 3-bucket tensors.
        valid = (src_ids >= 0) & (src_ids < 3)
        if valid.any():
            v_ids = src_ids[valid]
            v_loss = per_sample.detach()[valid].to(src_loss_t.dtype)
            src_loss_t.scatter_add_(0, v_ids, v_loss)
            src_count_t.scatter_add_(0, v_ids, torch.ones_like(v_ids, dtype=torch.long))
    if was_training:
        model.train()
    # Single host sync for the whole eval (was B * num_batches syncs).
    tot_n = int(tot_n_t.item())
    overall = float(tot_loss_t.item()) / max(1, tot_n)
    src_n = src_count_t.tolist()
    src_loss = src_loss_t.tolist()
    per_src = [src_loss[s] / src_n[s] if src_n[s] else float("nan") for s in range(3)]
    return overall, per_src


def train(
    model: EdgeHVN,
    dataloader: DataLoader,
    config: dict,
    device: torch.device,
    checkpoint_dir: str,
    max_steps: int | None = None,
    resume_step: int = 0,
    val_loader=None,
    resume_state: dict | None = None,
) -> None:
    """Run the training loop with all configured settings.

    - Mixed precision (Issue #80): fp32 (no autocast), bf16, or fp16 (with GradScaler)
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
    # #158: 0/absent = the original single-shot loss, so nothing changes for any
    # existing caller or config.
    chunked_loss_size = int(config.get("chunked_loss", 0) or 0)
    grad_accum = config["grad_accum"]

    # Issue #86: pluggable optimizer (adamw | muon | lion). Training-time only —
    # the HVN forward pass is untouched, so a model trained with any of these is
    # the *same* HVN, judged on convergence efficiency + final val-loss parity.
    opt_name = config.get("optimizer", "adamw").lower()
    use_fused = device.type == "cuda" and opt_name == "adamw"
    optimizer, opt_desc = build_optimizer(
        model, opt_name,
        lr=lr,
        weight_decay=weight_decay,
        fused=use_fused,
        muon_lr=config.get("muon_lr"),
        galore_rank=config.get("galore_rank", 128),
        galore_update_gap=config.get("galore_update_gap", 200),
        galore_scale=config.get("galore_scale", 0.25),
    )
    print(f"  Optimizer: {opt_desc}")

    # Capture each group's base LR before warmup/cosine mutate it, so multi-group
    # optimizers (Muon: high-LR matrices + low-LR AdamW aux) keep their LR ratio
    # through warmup. For single-group AdamW this is exactly the old behaviour.
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    # Issue #99: BAdam-style block-coordinate-descent. Training-time only --
    # freezes every layer-block except one, cycling round-robin, so only the
    # active block's Adam moments are ever VRAM-resident. The HVN forward pass
    # is untouched; every block gets trained over a full cycle, just not
    # simultaneously. Composes with --optimizer (validated with adamw; other
    # optimizers should work via the same requires_grad/optimizer.state
    # mechanism but are unverified here).
    bcd_scheduler = None
    if config.get("block_coordinate", False):
        from hvn.core.block_coordinate import BlockCoordinateScheduler
        bcd_scheduler = BlockCoordinateScheduler(
            model, optimizer, steps_per_block=config.get("bcd_steps_per_block", 100),
        )
        print(f"  Block-coordinate-descent: {bcd_scheduler.num_blocks} blocks, "
              f"{bcd_scheduler.steps_per_block} steps/block, "
              f"active={bcd_scheduler.active_block_name!r}")

    # --- Restore optimizer + BCD cycle position (Issue #112) ---
    # Without this, every restart of a long run threw away Muon/AdamW momentum AND
    # rewound the BCD round-robin to block 0. Over a multi-week run with many
    # restarts that systematically undertrains the later blocks: block 0 gets
    # retrained after every crash while block 13 only ever advances in the
    # uninterrupted stretches. BCD is mandatory at 355M (docs/training.md), so this
    # is a correctness issue for the run, not a convenience.
    if resume_state is not None:
        try:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            print(f"  [resume] optimizer state restored ({type(optimizer).__name__})")
        except Exception as e:
            # A changed --optimizer or param-group layout lands here. Continuing with
            # a fresh optimizer is correct-but-slower; silently proceeding as if the
            # state had loaded would not be.
            print(f"  [resume] WARNING optimizer state NOT restored ({e}) — "
                  f"continuing with fresh optimizer state")
        bcd_sd = resume_state.get("bcd_state_dict")
        if bcd_sd is not None and bcd_scheduler is not None:
            if bcd_sd.get("steps_per_block") != bcd_scheduler.steps_per_block:
                print(f"  [resume] WARNING --bcd-steps-per-block changed "
                      f"({bcd_sd.get('steps_per_block')} -> {bcd_scheduler.steps_per_block}); "
                      f"restoring the saved cycle position anyway, which is what keeps "
                      f"block coverage even")
            bcd_scheduler.load_state_dict(bcd_sd)
            print(f"  [resume] BCD cycle restored: block {bcd_scheduler._cycle_pos}, "
                  f"{bcd_scheduler._step_in_block} steps into it, "
                  f"active={bcd_scheduler.active_block_name!r}")
        elif bcd_sd is not None and bcd_scheduler is None:
            print("  [resume] saved BCD state ignored — this run has no --block-coordinate")
        elif bcd_sd is None and bcd_scheduler is not None:
            print("  [resume] no saved BCD state — cycle starts at block 0")

    # --- LR schedule (cosine by default) with warmup ---
    _lr_schedule = config.get("lr_schedule", "cosine")
    total_steps = max_steps if max_steps is not None else 10_000
    sched_sd = resume_state.get("sched_state_dict") if resume_state else None
    if resume_step > 0 and sched_sd is not None:
        # #165: restore the schedule's OWN position. optimizer.load_state_dict
        # above has already put the exact save-time LR back into param_groups,
        # and last_epoch says where on the cosine that value sits, so the
        # recursion continues from the right anchor with no replay.
        scheduler = build_cosine_scheduler(optimizer, total_steps, 0,
                                           schedule=_lr_schedule)
        scheduler.load_state_dict(sched_sd)
        print(f"  [resume] LR schedule restored at step {resume_step}/{total_steps} "
              f"(lr {optimizer.param_groups[0]['lr']:.3e}, "
              f"cosine position {scheduler.last_epoch})")
    else:
        if resume_step > 0:
            # Legacy checkpoint written before #165 -- no saved schedule. Replaying
            # is only correct from the PEAK LR, so reset the groups first;
            # otherwise the cosine anchors on the decayed value and is applied
            # twice (the bug itself). Replay excludes the warmup steps, during
            # which the original run never advanced the cosine.
            for pg, base in zip(optimizer.param_groups, base_lrs):
                pg["lr"] = base
            replay = max(0, resume_step - warmup_steps + 1)
            print(f"  [resume] no saved LR schedule (pre-#165 checkpoint) — "
                  f"rebuilding from peak, replaying {replay} steps")
        else:
            replay = 0
        scheduler = build_cosine_scheduler(optimizer, total_steps, replay,
                                           schedule=_lr_schedule)
        if resume_step > 0:
            print(f"  [resume] cosine schedule continued at step "
                  f"{resume_step}/{total_steps} "
                  f"(lr {optimizer.param_groups[0]['lr']:.3e}), not restarted")

    # --- Dynamic phase-aware LR multiplier (Issue #133) ---
    # A GLOBAL multiplier layered on top of the warmup+cosine LR above, chosen
    # from the shape of the live loss curve. Muon's per-parameter logic is not
    # touched: the multiplier scales every param group by the same factor, so
    # Muon's matrix-group / aux-group LR ratio is preserved exactly.
    #
    # It is applied around optimizer.step() and removed again before
    # scheduler.step(), because CosineAnnealingLR.get_lr() is recursive (see
    # build_cosine_scheduler) — a multiplier left in param_group["lr"] across a
    # scheduler.step() would be compounded into the cosine itself.
    dyn_lr = None
    if config.get("dynamic_lr", False):
        from ghost.dynamic_lr import DynamicLRScheduler
        dyn_lr = DynamicLRScheduler(
            log_path=config.get("dynamic_lr_log", "results/dynamic_lr/lr_log.json"),
            run_name=os.path.basename(checkpoint_dir),
        )
        print(f"  Dynamic LR: ON — phase detection every {dyn_lr.cfg.check_every} "
              f"steps over a {dyn_lr.cfg.window}-step window, multiplier in "
              f"[{dyn_lr.cfg.min_mult}, {dyn_lr.cfg.max_mult}]")
        print(f"              log -> {dyn_lr.log_path}")

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
        # Warmup is skipped on resume: the model is already trained, and (since
        # Issue #112) the optimizer's momentum is restored too, so there is no
        # cold-start to ease into. The cosine schedule is CONTINUED, not reset —
        # see the scheduler construction above.
        warmup_done = True
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

    # --- Held-out validation logging (A/B primary metric) ---
    val_every = config.get("val_every_steps", config["save_every_steps"])
    val_log = os.path.join(checkpoint_dir, "val_log.csv")
    val_exists = os.path.exists(val_log)
    val_file = open(val_log, "a", newline="", encoding="utf-8") if val_loader is not None else None
    val_writer = csv.writer(val_file) if val_file is not None else None
    if val_writer is not None and not val_exists:
        val_writer.writerow(["step", "val_loss", "val_ppl",
                             "val_slim", "val_oasst", "val_wiki", "timestamp"])
        val_file.flush()

    # --- Training loop ---
    model.train()

    # --- Mixed-precision setup (Issue #80) ---
    precision = config.get("precision", "bf16")
    use_amp = precision in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if (use_amp and precision == "fp16") else None
    if use_amp:
        print(f"  Mixed-precision: {precision.upper()} (Issue #80)")
    else:
        print(f"  Mixed-precision: fp32 (no autocast)")

    print(f"\n  Training for up to {total_steps} steps (effective batch = {config['batch_size'] * grad_accum})")
    print(f"  Logging every {config['log_every_steps']} steps | Checkpoint every {config['save_every_steps']} steps")
    print("-" * 70)

    epoch = 0
    start_time = time.time()
    step_start_time = start_time
    optimizer.zero_grad()

    # Per-source loss tracking (for diagnosing dataset quality variance).
    # Source names come from the dataset (dense over the INCLUDED sources only),
    # so a knowledge-heavy mix that drops OASST labels sid 1 as WikiText-103, not
    # OpenAssistant. Fall back to the legacy 3-source names if unavailable.
    try:
        source_names = list(dataloader.dataset.source_names)
    except AttributeError:
        source_names = ["SlimPajama", "OpenAssistant", "WikiText-103"]
    n_sources = len(source_names)
    source_loss_accum = [0.0] * n_sources  # accumulated loss per source
    source_count = [0] * n_sources  # sample count per source

    # GPU-side accumulators (deferred host sync): accumulating loss/per-source
    # stats on-device and syncing only at log time avoids a GPU->CPU stall on
    # every micro-batch, which stalls the async H2D/compute pipeline. Values are
    # exact — identical to the per-step .item() sums, just synced less often.
    loss_accum_t = torch.zeros((), device=device)      # current step's summed micro-losses
    src_loss_t = torch.zeros(3, device=device)         # per-source summed per-sample loss
    src_count_t = torch.zeros(3, device=device)        # per-source sample count

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

        # Forward pass with mixed-precision autocast (Issue #80). Compute the
        # per-token cross-entropy ONCE (reduction="none"); the scalar training
        # loss is its mean, and the per-source diagnostics reuse the same tensor
        # — this drops the redundant second full-vocab CE pass the per-source
        # tracking used to run every micro-batch.
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            if chunked_loss_size:
                # #158: the [B, S, vocab] logits and their gradient are ~1.5 GB at
                # 4096x32003 -- three times the weights, and the single largest
                # consumer on the card. Ask the model for hidden states instead
                # (4 MB) and project+score a slice of positions at a time, with
                # each slice's logits recomputed in backward rather than kept.
                # Exact, not approximate: tests/test_chunked_loss.py asserts
                # bit-identical per-token losses and gradients.
                from hvn.edge.chunked_loss import chunked_cross_entropy
                hidden = model(batch_x, return_hidden=True)
                tok_loss = chunked_cross_entropy(
                    hidden, batch_y,
                    getattr(model, "_orig_mod", model).decoder,
                    chunk=chunked_loss_size,
                )
            else:
                logits = model(batch_x)
                tok_loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch_y.reshape(-1),
                    reduction="none",
                )
            loss = tok_loss.mean() / grad_accum

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # Accumulate on-device (no host sync); .item() happens once at log time.
        loss_accum_t += loss.detach()

        # Track per-source loss (for diagnosing dataset quality variance).
        # Accumulate per-source sums/counts on the GPU via scatter_add — no
        # per-sample .item() loop, so no host sync in the hot path.
        if batch_src is not None:
            with torch.no_grad():
                per_sample = tok_loss.detach().reshape(batch_x.shape[0], -1).mean(dim=1)  # [B]
                src_ids = (batch_src.to(device=device, dtype=torch.long)
                           if isinstance(batch_src, torch.Tensor)
                           else torch.tensor(batch_src, device=device, dtype=torch.long))
                src_loss_t.scatter_add_(0, src_ids, per_sample.to(src_loss_t.dtype))
                src_count_t.scatter_add_(0, src_ids,
                                         torch.ones_like(src_ids, dtype=src_count_t.dtype))

        micro_step += 1

        # Optimizer step every grad_accum micro-batches
        if micro_step >= grad_accum:
            # Clip gradients (unscale first if using fp16 GradScaler)
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Apply warmup LR scaling (during warmup only). Scale each group from
            # its own base LR so multi-group optimizers keep their LR ratio.
            if not warmup_done:
                lr_scale = get_lr_scale(global_step)
                for pg, base in zip(optimizer.param_groups, base_lrs):
                    pg["lr"] = base * lr_scale
                if global_step + 1 >= warmup_steps:
                    warmup_done = True

            # Issue #133: feed this step's mean loss to the dynamic scheduler and
            # apply its multiplier for the duration of this one optimizer step.
            # loss_accum_t is the sum of grad_accum micro-losses each already
            # divided by grad_accum, and it is zeroed at the end of every step,
            # so it IS this step's mean loss. The .item() is a host sync, which
            # is why it is inside the flag guard: with --dynamic-lr off, this
            # path is byte-identical to before.
            if dyn_lr is not None:
                dyn_lr.observe(global_step, loss_accum_t.item())

            with contextlib.ExitStack() as _lr_stack:
                if dyn_lr is not None:
                    _lr_stack.enter_context(dyn_lr.applied(optimizer))
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            optimizer.zero_grad()
            micro_step = 0
            global_step += 1

            # Issue #99: advance the block-coordinate cycle (may switch the
            # active block + clear the outgoing block's optimizer state).
            if bcd_scheduler is not None:
                bcd_scheduler.step()

            # torch.profiler step boundary (only active under --profile)
            if _PROF is not None:
                _PROF.step()

            # After warmup, advance the cosine scheduler every step
            if warmup_done:
                scheduler.step()

            # Logging every log_every_steps
            if global_step % config["log_every_steps"] == 0:
                # loss_accum_t is the sum of grad_accum micro-batch losses, each
                # already divided by grad_accum — so it IS the step's average
                # loss. Sync the deferred GPU accumulators to host once, here.
                step_loss = loss_accum_t.item()
                source_loss_accum = src_loss_t.tolist()
                source_count = [int(c) for c in src_count_t.tolist()]
                step_ppl = math.exp(min(step_loss, 20))
                current_lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - step_start_time
                steps_per_sec = config["log_every_steps"] / max(elapsed, 0.001)
                step_start_time = time.time()

                # With --dynamic-lr the LR that was actually used is the
                # schedule's value times the multiplier; report that, not the
                # schedule value alone, or the log misrepresents the run.
                dyn_str = ""
                if dyn_lr is not None:
                    dyn_str = (f"  |  xLR: {dyn_lr.multiplier:.2f} "
                               f"(eff {current_lr * dyn_lr.multiplier:.6f})")
                print(f"  Step {global_step:>6d} (epoch {epoch})  |  "
                      f"Loss: {step_loss:.4f}  |  PPL: {step_ppl:.2f}  |  "
                      f"LR: {current_lr:.6f}{dyn_str}  |  "
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

                # Reset per-source accumulators after logging (on device)
                src_loss_t.zero_()
                src_count_t.zero_()

                # Estimate time per epoch based on first 10 steps
                if global_step == config["log_every_steps"]:
                    total_elapsed = time.time() - start_time
                    est_per_step = total_elapsed / global_step
                    print(f"  ⏱ Estimated {est_per_step:.2f}s/step | "
                          f"~{est_per_step * config['save_every_steps']:.0f}s per checkpoint interval")

            # Held-out validation eval (A/B primary metric)
            if val_writer is not None and global_step % val_every == 0:
                v_overall, v_src = evaluate(model, val_loader, device,
                                             precision=precision)
                v_ppl = math.exp(min(v_overall, 20))
                print(f"    ↳ VAL step {global_step}: loss {v_overall:.4f} | ppl {v_ppl:.2f} | "
                      f"Slim {v_src[0]:.4f}  OA {v_src[1]:.4f}  Wiki {v_src[2]:.4f}", flush=True)
                val_writer.writerow([
                    global_step, f"{v_overall:.6f}", f"{v_ppl:.4f}",
                    f"{v_src[0]:.6f}", f"{v_src[1]:.6f}", f"{v_src[2]:.6f}",
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                ])
                val_file.flush()

            # Checkpoint every save_every_steps
            if global_step % config["save_every_steps"] == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"edge_conversational_step{global_step}.pt")
                _save_edge_checkpoint(model, ckpt_path, global_step=global_step)
                _save_resume_state(checkpoint_dir, global_step, optimizer, bcd_scheduler,
                       lr_scheduler=scheduler)
                print(f"  ✓ Checkpoint saved: {ckpt_path}")

            loss_accum_t.zero_()

    # --- Final checkpoint ---
    final_ckpt_path = os.path.join(checkpoint_dir, "edge_conversational.pt")
    _save_edge_checkpoint(model, final_ckpt_path, global_step=global_step)
    _save_resume_state(checkpoint_dir, global_step, optimizer, bcd_scheduler,
                       lr_scheduler=scheduler)

    if dyn_lr is not None:
        dyn_lr.close()
        print(f"  Dynamic LR: {len(dyn_lr.adjustments)} adjustments, "
              f"final multiplier {dyn_lr.multiplier:.3f}, "
              f"phases {dyn_lr._phase_counts} -> {dyn_lr.log_path}")

    csv_file.close()
    if val_file is not None:
        val_file.close()

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
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed torch/numpy/random before model build for "
                             "reproducible init + data shuffling. Use for fair "
                             "optimizer/arch A/Bs (#86) so both arms start identical.")
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
    parser.add_argument("--fineweb-edu-dir", type=str, default=None,
                        help="Directory containing FineWeb-Edu parquet files (Issue #120, "
                             "downloaded via scripts/download_fineweb_edu.py). If not set, "
                             "defaults to <data-dir>/fineweb_edu")
    parser.add_argument("--fineweb-edu-token-bytes", type=int, default=None,
                        help="Cap on FineWeb-Edu text tokenized, in UTF-8 bytes. Default "
                             "None = use everything that's been downloaded.")
    # --- Issue #136: code-only pretrain + Fill-in-the-Middle ---
    parser.add_argument("--code-corpus", type=str, default=None,
                        help="Path to a code corpus .txt (built by "
                             "scripts/extract_the_stack.py). REPLACES the "
                             "four-source mix entirely -- a code model trains on "
                             "code alone -- so SlimPajama/OASST/WikiText are "
                             "never downloaded or tokenized for such a run.")
    parser.add_argument("--fim", action="store_true", default=False,
                        help="Fill-in-the-Middle (Bavarian et al. 2022). A "
                             "TRAINING-DATA transform, so it must be chosen "
                             "before the pretrain -- it cannot be added to a "
                             "finished model. Requires a tokenizer carrying the "
                             "three sentinels: build one with "
                             "scripts/build_fim_tokenizer.py and pass "
                             "--tokenizer data_cache/tokenizer_32k_fim.")
    parser.add_argument("--fim-rate", type=float, default=0.9,
                        help="Fraction of UNITS transformed (a unit is ~1200 "
                             "characters, not a whole file). Below 1.0 on "
                             "purpose: the model must still generate forward "
                             "from a cursor with nothing after it. Raised from "
                             "0.5 to 0.9 in #164 -- Bavarian et al. report up to "
                             "0.9 is harmless to left-to-right ability, and at "
                             "0.5 half the units carried no jump-back at all.")
    parser.add_argument("--fim-spm-rate", type=float, default=0.5,
                        help="Fraction of FIM documents using SPM (suffix "
                             "first) rather than PSM.")
    parser.add_argument("--fim-seed", type=int, default=0,
                        help="Seed for cut points. Part of the .bin cache "
                             "signature, so changing it re-tokenizes rather "
                             "than silently reusing differently-cut data.")
    parser.add_argument("--fim-separator", type=str, default="<|endoftext|>",
                        help="Document separator in the corpus file.")
    parser.add_argument("--pretrain-mix", type=str, default=None,
                        help="Pretrain source mix as 'slim,oasst,wiki,fineweb_edu' fractions "
                             "(e.g. '0.85,0.0,0.15,0.0' = knowledge-heavy mix that DROPS "
                             "OpenAssistant from pretrain; '0.0,0.0,0.0,1.0' = FineWeb-Edu "
                             "only, Issue #120's recommended mix for 4B+ runs, sidesteps "
                             "WikiText's Issue #119 mix-ratio cap by construction). Default "
                             "None = 0.50,0.35,0.15,0.0 (the original conversational mix, "
                             "FineWeb-Edu off unless explicitly requested). Zero-fraction "
                             "sources are excluded from the train index, so dropping the "
                             "tiny ~16M-token OASST source stops it capping the dataset "
                             "via the max_repeat bind (see build_dataset, and the #119/#120 "
                             "mix-cap warning it prints). OASST still goes into the separate "
                             "chat fine-tune stage. A nonzero FineWeb-Edu share requires "
                             "having run scripts/download_fineweb_edu.py first.")
    parser.add_argument("--slimpajama-token-bytes", type=int, default=500 * 1024**2,
                        help="Cap on bytes of SlimPajama text to tokenize (default "
                             "500MB ~ 129M tokens). Set large (e.g. 6e9) to tokenize "
                             "the full ~5.4GB slice (~1.35B tokens). Cached: reuses an "
                             "existing non-empty slimpajama.bin regardless of this cap, "
                             "so delete the old bin to force a full re-tokenize.")
    parser.add_argument("--max-repeat", type=int, default=3,
                        help="Max times any source may repeat to hit its mix share. "
                             "Binds dataset size on the smallest source at its share. "
                             "Default 3.")
    parser.add_argument("--hf-cache-dir", type=str,
                        default=None,
                        help="HuggingFace cache directory for tokenizer/dataset downloads. "
                             "If not set, uses the default HF cache (~/.cache/huggingface).")
    parser.add_argument("--arch", type=str, default="hvn",
                        choices=["hvn", "transformer", "hvn-hybrid",
                                 "hvn-original-hybrid"],
                        help="Block architecture: 'hvn' (DeltaNet, default), "
                             "'transformer' (matched causal-MHA baseline for the A/B, #73), "
                             "or 'hvn-hybrid' (DeltaNet trunk + a few sliding-window "
                             "attention layers for token-recall; requires --attn-layers).")
    parser.add_argument("--attn-layers", type=str, default=None,
                        help="Comma-separated 0-indexed layer positions for sliding-window "
                             "attention blocks when --arch hvn-hybrid (e.g. '3,10'). "
                             "Non-listed layers are DeltaNet. Required with hvn-hybrid.")
    parser.add_argument("--attn-window", type=int, default=4096,
                        help="Sliding-window width for hybrid attention layers (tokens). "
                             "KV cache is capped at W keys/layer (~16MB at W=4096, batch=1 "
                             "bf16). Defaults to 4096 = 8x training seq_len (RoPE "
                             "generalizes cleanly, no long-ctx FT needed for v1).")
    parser.add_argument("--hybrid-trunk-init", type=str, default=None,
                        help="Path to a pure-HVN/DeltaNet checkpoint to warm-start the "
                             "hybrid's DeltaNet trunk from (copy matching keys, leave the "
                             "new attention layers at fresh init). Used by the continue-FT.")
    parser.add_argument("--no-ternary", action="store_true",
                        help="Disable ternary quantization — train fp32 weights. "
                             "Use for the fp32 A/B (no ternary either side).")
    parser.add_argument("--val-fraction", type=float, default=0.0,
                        help="Fraction of each source's unique windows held out as a "
                             "token-disjoint validation set (e.g. 0.02). 0 disables.")
    parser.add_argument("--val-every", type=int, default=None,
                        help="Run held-out val eval every N steps (default: checkpoint interval).")
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16", "bf16"],
                        help="Mixed-precision training mode (Issue #80). "
                             "fp32 (default) keeps A/B fairness — both arms fp32. "
                             "bf16 enables autocast bf16 (~1.5-2x compute speedup, "
                             "RTX 40-series+). fp16 uses GradScaler (older GPUs).")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap the model in torch.compile for training (#83). "
                             "Identity-preserving (same math/loss); ~2.4x faster + ~15%% less "
                             "VRAM on the DeltaNet path. Checkpoints are saved unwrapped.")
    parser.add_argument("--triton-scan", action="store_true",
                        help="Use the fused Triton gated-delta scan on DeltaNet blocks (#83). "
                             "Identity-preserving (grads match autograd ~1e-7); ~2.6x faster "
                             "full-model + less VRAM, stable at all decay. HVN arch only. Can "
                             "combine with --compile (compile fuses the non-scan parts).")
    parser.add_argument("--megakernel-scan", action="store_true",
                        help="Route the DeltaNet scan through the Design B fused "
                             "blockwise-recompute backward (#96) when --triton-scan is on: "
                             "keeps the optimal sequential forward, replaces the 2-kernel "
                             "backward (which dumps a 268 MB [N,S,D,D] states tensor to HBM) "
                             "with one SRAM-recompute kernel + block-boundary checkpoints. "
                             "Identity-preserving (grads match the sequential backward "
                             "~5e-8); measured ~1.05x faster step end-to-end under --fast + "
                             "~4x less transient backward VRAM. HVN arch only; requires "
                             "--triton-scan.")
    parser.add_argument("--adaptive-chunk", action="store_true",
                        help="Grow the DeltaNet chunk to the largest size that stays "
                             "numerically EXACT for the current decay (#95): fast at "
                             "trained decay (α~0.95 → chunk 256, ~4x scan), safe at hard "
                             "decay. Identity-preserving (fwd+bwd parity ~1e-7 at all "
                             "decay) and fixes a latent bug (fixed chunk-32 is wrong at "
                             "α<0.31). Beats --compile alone (196 vs 232 ms/microbatch); "
                             "does NOT combine with --compile (data-dependent chunk). "
                             "HVN arch only; mutually exclusive with --triton-scan.")
    parser.add_argument("--profile", action="store_true",
                        help="Profile steady-state training with torch.profiler (skips the "
                             "first ~12 compile-warmup steps, records the next ~30, then "
                             "prints a kernel-level breakdown and exits). Diagnostic only; "
                             "default behavior is unchanged when this flag is absent.")
    parser.add_argument("--tf32", action="store_true",
                        help="Enable TF32 for fp32 matmuls (Issue #81). TF32 uses 19-bit "
                             "precision (10 mantissa bits) for the internal matmul accumulate, "
                             "giving ~1.2-1.5x speedup on Ampere+ (RTX 30/40-series, A100). "
                             "Only affects fp32 matmuls — bf16/fp16 autocast already uses "
                             "tensor cores. Identity-preserving (same forward math, negligible "
                             "numerical difference). Default off for fp32 A/B fairness.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of DataLoader worker processes for async batch "
                             "preparation (Issue #91). 0 = main-thread (default, A/B-fair). "
                             "2-4 recommended for real training — overlaps data loading "
                             "with GPU compute. Identity-preserving (same data, same order "
                             "with fixed seed).")
    parser.add_argument("--optimizer", type=str, default="muon",
                        choices=["adamw", "muon", "lion", "galore", "adafactor",
                                 "prodigy", "muon-prodigy"],
                        help="Optimizer. DEFAULT 'muon' since #129: with ternary "
                             "ON (the config this project actually trains with) "
                             "and --fast, Muon reaches AdamW's 1000-step val loss "
                             "in ~570 steps (1.75x fewer) and keeps a ~0.4 nat "
                             "lead throughout; it also passes 6.0, which AdamW "
                             "never reaches in 1000 steps. Muon's steps are ~12 pct "
                             "slower, so the honest wall-clock win is ~1.6x. "
                             "CAVEAT: measured over 1000 from-scratch steps; no "
                             "run in this project has taken Muon past that, so "
                             "watch early val on a long pretrain and fall back "
                             "with --optimizer adamw if the curve misbehaves. "
                             "'muon' orthogonalises 2-D "
                             "weight-matrix updates (faster convergence); 'lion' "
                             "is sign-momentum; 'galore' (Issue #84) projects "
                             "gradients into a low-rank subspace so Adam's "
                             "optimizer state is O(rank) instead of O(m*n) for "
                             "large weight matrices -- weights stay full-rank, "
                             "'adafactor' (Issue #85) factors the second moment "
                             "into O(m+n) instead of O(m*n) -- no bitsandbytes "
                             "dependency (unreliable on native Windows), native "
                             "torch.optim.Adafactor. "
                             "only the optimizer's view of the gradient is "
                             "compressed. Training-time only -- HVN forward "
                             "is untouched. Default: adamw.")
    parser.add_argument("--galore-rank", type=int, default=128,
                        help="GaLore projection rank (only used with "
                             "--optimizer galore). Default: 128.")
    parser.add_argument("--galore-update-gap", type=int, default=200,
                        help="Steps between GaLore projection-matrix SVD "
                             "refreshes (only used with --optimizer galore). "
                             "Default: 200 (paper's default).")
    parser.add_argument("--galore-scale", type=float, default=0.25,
                        help="GaLore update scale factor alpha (only used "
                             "with --optimizer galore). Default: 0.25 (paper's "
                             "default, does not depend on rank).")
    parser.add_argument("--muon-lr", type=float, default=0.02,
                        help="Muon LR for 2-D hidden matrices (default 0.02; "
                             "only used with --optimizer muon).")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                        help="Issue #139: 'gpt2' (default, 50257) or a directory "
                             "holding vocab.json + merges.txt from "
                             "scripts/train_tokenizer.py. A 32K mixed "
                             "English+code vocab measured 24 pct cheaper per "
                             "document AND produced 11 pct SHORTER sequences. "
                             "Changing this is a one-way door: a model cannot be "
                             "fine-tuned onto a different tokenizer.")
    parser.add_argument("--dynamic-lr", action="store_true",
                        help="Issue #133: phase-aware GLOBAL LR multiplier layered "
                             "on top of the warmup+cosine schedule. Watches the "
                             "live loss curve and raises the LR on plateaus and "
                             "phase transitions, cuts it on spikes. Multiplier is "
                             "clamped to [0.1, 3.0] and to 0.5-1.5x per adjustment. "
                             "Does NOT change the optimizer: every param group is "
                             "scaled equally, so Muon's group LR ratio is preserved. "
                             "Default OFF — the flag costs one host sync per "
                             "optimizer step to read the loss.")
    parser.add_argument("--dynamic-lr-log", default="results/dynamic_lr/lr_log.json",
                        help="Where --dynamic-lr writes its adjustment log "
                             "(default results/dynamic_lr/lr_log.json).")
    parser.add_argument("--block-coordinate", action="store_true",
                        help="BAdam-style block-coordinate-descent (Issue #99): train "
                             "one layer-block at a time (round-robin), explicitly "
                             "clearing the outgoing block's optimizer state on every "
                             "switch, so only one block's Adam moments are ever VRAM-"
                             "resident regardless of total model size. Training-time "
                             "only -- the HVN forward pass is untouched, every block "
                             "gets trained over a full cycle, just not simultaneously. "
                             "Default: off (vanilla full-model training).")
    parser.add_argument("--bcd-steps-per-block", type=int, default=100,
                        help="Optimizer steps per active block before switching to the "
                             "next (only used with --block-coordinate). Default: 100.")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Exact activation checkpointing (Issue #100, mirrors the "
                             "core model's #49): recompute each block's activations in "
                             "the backward pass instead of storing them. Bit-identical "
                             "result, ~20-30%% more compute, large activation-VRAM cut. "
                             "Composes with --block-coordinate (that shrinks optimizer-"
                             "state memory; this shrinks activation memory -- "
                             "independent axes). Default: off.")
    parser.add_argument("--chunked-loss", type=int, nargs="?", const=512, default=0,
                        metavar="POSITIONS",
                        help="Compute the loss POSITIONS at a time (default 512 "
                             "when given without a value; 0 = off) and recompute "
                             "each slice in backward (#158). The cross-entropy "
                             "over the vocabulary is ~1.5 GB at seq 4096 x 32K "
                             "vocab -- three times the weights and the largest "
                             "single consumer on a 6 GB card. Measured: -1436 MB "
                             "reserved at seq 4096, 1.07x step time. Exact in "
                             "fp32 (tests assert bit-identical losses AND "
                             "gradients); under bf16 autocast the value differs "
                             "in the 4th decimal from accumulation order alone "
                             "-- 3.8e-03 on a loss of 10.5, i.e. 3.6e-04 "
                             "relative -- because slices reduce separately.")
    parser.add_argument("--lr-schedule", choices=["cosine", "constant"],
                        default="cosine",
                        help="LR shape after warmup. 'constant' exists for #159: "
                             "a study that reads held-out loss off MID-RUN "
                             "checkpoints cannot use a cosine fitted to the "
                             "run's end, because those checkpoints are partway "
                             "through a schedule rather than finished models. "
                             "Use cosine for anything you intend to keep.")
    parser.add_argument("--shared-weights", action="store_true", default=False,
                        help="ALBERT-style weight sharing: all layers of the same "
                             "block type reuse ONE block's weights (#155). On a "
                             "12-layer hybrid this stores 2 blocks instead of 12 "
                             "-- measured 5.98x fewer body params at d=512. The "
                             "block plan is preserved: attention layers still "
                             "share only with attention layers. Each virtual "
                             "layer keeps its own RoPE offset and residual scale. "
                             "Sharing costs ~29%% PPL at FIXED width (measured "
                             "2026-08-02), so it only wins if the saved "
                             "parameters are spent on a WIDER model -- compare "
                             "at equal params, never at equal width.")
    parser.add_argument("--tie-embeddings", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Tie token_embedding and the decoder output projection "
                             "(Issue #106). DEFAULT ON since #139, which measured it "
                             "rather than assuming it: 1,500 from-scratch steps, "
                             "identical batches/seed/--deterministic, only this flag "
                             "differing -> held-out 5.50681 tied vs 5.56450 untied, "
                             "i.e. 0.058 nats BETTER with 31 pct FEWER parameters "
                             "(83.08M -> 57.35M), 12.5 sigma against the #133 noise "
                             "floor. Not a memory-for-quality trade -- strictly better "
                             "AND smaller, most likely because the two tables were "
                             "learning nearly the same mapping and one of them was "
                             "free capacity to memorise with. Costs ~7 pct per step "
                             "(the shared table takes gradients from both the lookup "
                             "and the head, serialising an accumulation). "
                             "--no-tie-embeddings restores the old behaviour; existing "
                             "checkpoints are unaffected either way, since "
                             "load_checkpoint_for_resume reads the setting from the "
                             "checkpoint.")
    fast_group = parser.add_mutually_exclusive_group()
    fast_group.add_argument("--no-fast", dest="fast", action="store_false",
                            help="Disable the --fast preset and train from the bare "
                                 "defaults (fp32, no compile, no Triton scan, "
                                 "num_workers=0). Use for a controlled fp32 A/B or "
                                 "parity test, where both arms must be bit-comparable "
                                 "-- bf16/TF32 change precision, so --fast is not "
                                 "appropriate there. This is the escape hatch for the "
                                 "#129 default flip; before that, bare defaults were "
                                 "what you got by saying nothing.")
    fast_group.add_argument("--fast", dest="fast", action="store_true", default=True,
                        help="Meta-flag, ON BY DEFAULT since #129: enable the full recommended fast-training "
                             "stack in one switch — bf16 + TF32 + torch.compile + "
                             "Triton scan + async DataLoader (4 workers). The bit-"
                             "identical levers (compile/triton/workers) never change "
                             "the model; bf16/TF32 change precision negligibly. Use "
                             "for real training runs. NOT for the fp32 A/B — that "
                             "needs the bare fp32 default. Any lever passed explicitly "
                             "overrides the preset (e.g. --fast --precision fp32). "
                             "Triton scan is skipped under --arch transformer.")
    args = parser.parse_args()

    # --- Issue #86: expand the --fast meta-flag into the recommended stack ---
    # One switch instead of five flags. Any lever the user passed explicitly on
    # the command line wins over the preset (mirrors train.py's --full). The
    # bit-identical levers (compile/triton/workers) leave the model unchanged;
    # bf16/TF32 change precision negligibly — so --fast is for real training,
    # not the fp32 A/B (which needs the bare default).
    if args.fast:
        explicit = {tok.split("=")[0] for tok in sys.argv[1:] if tok.startswith("--")}
        fast_enabled = []
        if "--precision" not in explicit:
            args.precision = "bf16"; fast_enabled.append("precision=bf16")
        if "--tf32" not in explicit:
            args.tf32 = True; fast_enabled.append("tf32")
        if "--compile" not in explicit:
            args.compile = True; fast_enabled.append("compile")
        # Triton scan targets the DeltaNet/HVN block — skip it for the transformer
        # A/B arm, where it's a no-op at best and a shape mismatch at worst.
        if "--triton-scan" not in explicit and args.arch == "hvn":
            args.triton_scan = True; fast_enabled.append("triton-scan")
        if "--num-workers" not in explicit and args.num_workers == 0:
            args.num_workers = 4; fast_enabled.append("num-workers=4")
        print(f"[--fast] Enabled recommended training stack: {', '.join(fast_enabled)}")

    # --- Issue #93/#120: parse --pretrain-mix 'slim,oasst,wiki,fineweb_edu' -> float tuple (or None) ---
    if args.pretrain_mix is not None:
        parts = [float(x) for x in args.pretrain_mix.split(",")]
        if len(parts) != 4:
            parser.error(
                "--pretrain-mix must be four comma-separated fractions "
                "'slim,oasst,wiki,fineweb_edu' (Issue #120 added the 4th; a "
                "3-value mix from before this change needs a trailing ',0.0' added)"
            )
        if sum(parts) <= 0:
            parser.error("--pretrain-mix fractions must sum to > 0")
        args.pretrain_mix = tuple(parts)

    # Force UTF-8 for Windows console
    if sys.platform == "win32":
        os.system("chcp 65001 >nul 2>&1")

    # --- Issue #81: TF32 matmul acceleration (Ampere+) ---
    # TF32 uses 19-bit precision for fp32 matmul internals — a free speedup on
    # RTX 30/40-series and A100. Default off for fp32 A/B fairness; opt in with
    # --tf32. Only affects fp32 matmuls (bf16/fp16 autocast already uses tensor
    # cores). Identity-preserving: same forward math, negligible numerical diff.
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 matmul enabled (Issue #81): fp32 matmuls use TF32 on Ampere+ tensor cores.")
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Issue #104: cudnn.benchmark lets cuDNN autotune and cache the fastest conv
    # algorithm for a given input shape, instead of always using a fixed
    # heuristic. seq_len/batch_size are fixed for the whole run (Issue #105
    # made drop_last=True, so this holds for every step including the last),
    # so autotuning pays for itself after a few warmup steps. Same math, no
    # numerical or A/B-fairness concern -- unlike TF32 above, this is always
    # safe to enable unconditionally.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Reproducible init for fair A/B comparisons (#86): seed all RNGs before the
    # model is built and data is shuffled, so two runs differing only in
    # --optimizer start from identical weights and see the same data order.
    if args.seed is not None:
        import random as _random
        _random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"Seeded all RNGs with {args.seed} (reproducible init + shuffling).")

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
    tokenizer = load_tokenizer(args.tokenizer, **tokenizer_kwargs)
    # `len(tokenizer)`, NOT `tokenizer.vocab_size`. HuggingFace's `vocab_size`
    # deliberately EXCLUDES added special tokens, so a FIM tokenizer reports
    # 32000 while its three sentinels occupy ids 32000-32002. Sizing the
    # embedding table from `vocab_size` therefore builds a model three rows too
    # small for the data it is about to be trained on, and the first FIM
    # sentinel in the stream indexes out of bounds -- surfacing as a bare CUDA
    # "device-side assert triggered" inside the pre-training audit, with no hint
    # of the cause. This is why --fim had never completed a run.
    #
    # Correct for both cases: GPT-2 has no added tokens (vocab_size == len ==
    # 50257), so this changes nothing for every existing recipe.
    vocab_size = len(tokenizer)
    n_added = vocab_size - getattr(tokenizer, "vocab_size", vocab_size)
    eos_token_id = (tokenizer.eos_token_id
                    if tokenizer.eos_token_id is not None else 50256)
    print(f"  Tokenizer: {args.tokenizer} | vocab_size = {vocab_size}"
          + (f" (base {tokenizer.vocab_size} + {n_added} added)" if n_added else "")
          + f" | eos_token_id = {eos_token_id}")

    dataloader, val_loader, dataset_stats = build_dataset_pipeline(
        tokenizer=tokenizer,
        seq_len=settings["seq_len"],
        batch_size=settings["batch_size"],
        smoke=args.smoke,
        data_dir=args.data_dir,
        slimpajama_dir=args.slimpajama_dir,
        fineweb_edu_dir=args.fineweb_edu_dir,
        fineweb_edu_token_bytes=args.fineweb_edu_token_bytes,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        mix=args.pretrain_mix,
        slimpajama_token_bytes=args.slimpajama_token_bytes,
        max_repeat=args.max_repeat,
        code_corpus=args.code_corpus,
        fim=(_check_fim_tokenizer(tokenizer) or
             {"rate": args.fim_rate, "spm_rate": args.fim_spm_rate,
              "seed": args.fim_seed, "separator": args.fim_separator}
             if args.fim else None),
    )

    # === Step 3: Training Configuration ===
    attn_layers = None
    if args.attn_layers:
        attn_layers = [int(x) for x in args.attn_layers.split(",") if x.strip() != ""]
    model, config = build_model_and_config(
        settings=settings,
        tokenizer=tokenizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        arch=args.arch,
        apply_ternary=not args.no_ternary,
        attn_layers=attn_layers,
        attn_window=args.attn_window,
        grad_checkpoint=args.grad_checkpoint,
        tie_embeddings=args.tie_embeddings,
        shared_weights=args.shared_weights,
        lr_schedule=args.lr_schedule,
        chunked_loss=args.chunked_loss,
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
        optimizer=args.optimizer,
        precision=args.precision,
        lr=args.lr,
    )
    config["lr"] = args.lr
    config["precision"] = args.precision
    config["optimizer"] = args.optimizer
    config["muon_lr"] = args.muon_lr
    config["galore_rank"] = args.galore_rank
    config["galore_update_gap"] = args.galore_update_gap
    config["galore_scale"] = args.galore_scale
    config["block_coordinate"] = args.block_coordinate
    config["bcd_steps_per_block"] = args.bcd_steps_per_block
    config["dynamic_lr"] = args.dynamic_lr
    config["dynamic_lr_log"] = args.dynamic_lr_log
    if args.val_every is not None:
        config["val_every_steps"] = args.val_every

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
    resume_state = None
    if resume_ckpt:
        print(f"\n  Found existing checkpoint: {resume_ckpt}")
        print(f"  Resuming training from this checkpoint...")
        _resumed, resume_step, _ckpt = load_checkpoint_for_resume(resume_ckpt, device)
        # Resuming REPLACES the model built from the CLI flags. Without this
        # check that swap is silent: ask for d=896/16L, leave an unrelated
        # checkpoint in the default directory, and you train a 28M model while
        # the log says nothing. It also makes the pre-training audit decorative,
        # since the audit validated the model that just got thrown away.
        _check_resume_matches_request(model, _ckpt, resume_ckpt, args)
        model = _resumed
        print(f"  Resumed at step {resume_step}")
        # Optimizer momentum + BCD cycle position (Issue #112) live in a separate
        # rolling file, not in the archived checkpoint — see _save_resume_state.
        resume_state = _load_resume_state(args.checkpoint_dir, resume_step)
    elif args.hybrid_trunk_init:
        # First run of a hybrid continue-FT: warm-start the DeltaNet trunk from a
        # pure-HVN checkpoint, leaving the fresh attention layers at init. Later
        # resumes hit the branch above (the hybrid checkpoint in checkpoint_dir).
        if args.arch != "hvn-hybrid":
            raise SystemExit("--hybrid-trunk-init requires --arch hvn-hybrid")
        print(f"\n  Warm-starting hybrid trunk from {args.hybrid_trunk_init}...")
        trunk_step = load_hybrid_trunk_into_model(model, args.hybrid_trunk_init, device)
        print(f"  Trunk warm-started (trunk was at step {trunk_step}); "
              f"attention layers at fresh init. resume_step=0.")

    # Fused Triton scan (#83): identity-preserving ~2.6x training speedup + less
    # VRAM, numerically stable at all decay. Enabled per-DeltaNet-block. Set
    # before compile so torch.compile fuses the non-scan parts around it.
    # #136: the Triton scan / megakernel / compile stack now lives in ONE
    # function, apply_fast_stack, so measurement harnesses can put a model into
    # the same mode instead of each re-implementing a subset of it. This block
    # used to be the only implementation, which is why every script that called
    # build_model_and_config directly measured an uncompiled model.
    # Applied after resume so checkpoint loading saw a plain module.
    model = apply_fast_stack(
        model,
        triton_scan=args.triton_scan,
        megakernel_scan=args.megakernel_scan,
        tf32=args.tf32,
        compile_model=args.compile,
    )

    # Adaptive-exact chunk (#95): pick the largest numerically-exact chunk per
    # forward given the decay. Faster than fixed chunk32 AND than --compile alone
    # at trained decay; do not combine with --compile (data-dependent trip count).
    if args.adaptive_chunk:
        blocks = model.blocks if getattr(model, "blocks", None) is not None else \
            ([model.shared_block.block] if getattr(model, "shared_block", None) is not None else [])
        n_set = 0
        for blk in blocks:
            if hasattr(blk, "adaptive_chunk"):
                blk.adaptive_chunk = True
                n_set += 1
        print(f"\n  Adaptive-exact chunk enabled on {n_set} DeltaNet block(s) (#95).")

    # torch.compile is applied by apply_fast_stack above (#136), after resume so
    # checkpoint loading saw a plain module and _save_edge_checkpoint unwraps a
    # known shape.

    if args.profile:
        # Steady-state profiling: skip the first ~12 compile-warmup optimizer steps,
        # warm up 2, record the next 30, then stop. _PROF.step() (called once per
        # optimizer step in train()) marks the step boundaries the schedule counts.
        import torch.profiler as _tp
        _prof = _tp.profile(
            activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA],
            schedule=_tp.schedule(wait=12, warmup=2, active=30, repeat=1),
        )
        _prof.__enter__()
        globals()["_PROF"] = _prof
        print("\n  [profile] torch.profiler armed — skipping first ~12 compile-warmup steps.")
        try:
            train(
                model=model,
                dataloader=dataloader,
                config=config,
                device=device,
                checkpoint_dir=args.checkpoint_dir,
                max_steps=args.max_steps,
                resume_step=resume_step,
                val_loader=val_loader,
                resume_state=resume_state,
            )
        finally:
            # Render BEFORE __exit__: the schedule clears events at cycle end, so
            # key_averages() is most reliable while the profiler context is live.
            print("\n" + "=" * 78)
            print("[profile] steady-state kernel breakdown (self GPU time, top 25)")
            print("=" * 78)
            try:
                print(_fmt_profile_table(_prof, row_limit=25))
            except Exception as _e:
                print(f"[profile] could not render table: {_e!r}")
            _prof.__exit__(None, None, None)
    else:
        train(
            model=model,
            dataloader=dataloader,
            config=config,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            max_steps=args.max_steps,
            resume_step=resume_step,
            val_loader=val_loader,
            resume_state=resume_state,
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