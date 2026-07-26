"""Test for Issue #120: FineWeb-Edu integration into the pretrain data
pipeline. Uses a real (small, 541 MB) already-downloaded FineWeb-Edu
parquet file in an isolated directory -- not the live, growing
data_cache/fineweb_edu directory this session's background downloads are
still writing to, and not a synthetic/mocked source.

Run:  python -m pytest tests/test_issue120_fineweb_edu.py -q
  or: python tests/test_issue120_fineweb_edu.py
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import GPT2TokenizerFast

from train_edge_conversational import MixedLocalDataset, _load_fineweb_edu

_SMALL_FINEWEB_DIR = os.path.join(_REPO_ROOT, "data_cache", "fineweb_edu_test")
_HAS_SMALL_FINEWEB = os.path.exists(
    os.path.join(_SMALL_FINEWEB_DIR, "sample", "10BT", "013_00000.parquet")
)
_TOK_DIR = os.path.join(_REPO_ROOT, "data_cache", "tokenized")
_HAS_CACHED_DATA = all(
    os.path.exists(os.path.join(_TOK_DIR, f))
    for f in ("slimpajama.bin", "oasst.bin", "wikitext.bin")
)


def test_load_fineweb_edu_returns_none_when_nothing_downloaded():
    """Graceful, not an error: a config that hasn't run the download
    script yet should report 'nothing here' rather than crash."""
    with tempfile.TemporaryDirectory() as empty_dir:
        result = _load_fineweb_edu(cache_dir=empty_dir, fineweb_edu_dir=os.path.join(empty_dir, "nope"))
        assert result is None


def test_load_fineweb_edu_extracts_real_parquet_to_text():
    """Real extraction from a real downloaded parquet file (not mocked)."""
    if not _HAS_SMALL_FINEWEB:
        print("  [skip] small FineWeb-Edu test file not present locally")
        return
    with tempfile.TemporaryDirectory() as tmp_cache:
        path = _load_fineweb_edu(
            cache_dir=tmp_cache, fineweb_edu_dir=_SMALL_FINEWEB_DIR,
            max_bytes=10_000_000,  # 10MB cap, keep the test fast
        )
        assert path is not None
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        with open(path, encoding="utf-8") as f:
            sample = f.read(500)
        assert len(sample.strip()) > 0, "extracted text file should have real content"


def test_nonzero_fineweb_edu_share_without_download_raises_clear_error():
    """Issue #120's promised behavior: a nonzero share with no data
    downloaded fails LOUDLY at mix-building time, not silently
    renormalizing the other sources to quietly cover the gap.

    Test-isolation note: shares the real data_cache/ (to reuse the
    already-cached SlimPajama/OASST/WikiText .bin files, which are
    unconditionally prepared regardless of FineWeb-Edu's share -- forcing
    a fully isolated cache_dir would make SlimPajama/OASST re-download
    and re-tokenize from scratch, slow and network-dependent for what
    should be a fast unit test). But data_cache/fineweb_edu_en.txt is
    genuinely reusable cache another test in this file may have already
    populated (test_mixed_dataset_with_real_fineweb_edu_source runs
    first alphabetically) -- explicitly clear just that one file so this
    test's "nothing downloaded yet" premise holds regardless of test
    execution order.
    """
    if not _HAS_CACHED_DATA:
        print("  [skip] tokenized data_cache not present locally")
        return
    stale_cache = os.path.join(_REPO_ROOT, "data_cache", "fineweb_edu_en.txt")
    if os.path.exists(stale_cache):
        os.remove(stale_cache)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    with tempfile.TemporaryDirectory() as empty_fineweb_dir:
        try:
            MixedLocalDataset(
                tokenizer, seq_len=512, cache_dir=os.path.join(_REPO_ROOT, "data_cache"),
                fineweb_edu_dir=empty_fineweb_dir,
                mix=(0.5, 0.0, 0.0, 0.5),  # nonzero FineWeb-Edu share
            )
            raise AssertionError("expected FileNotFoundError for missing FineWeb-Edu data")
        except FileNotFoundError as e:
            assert "FineWeb-Edu" in str(e)
            assert "download_fineweb_edu.py" in str(e)


def test_zero_fineweb_edu_share_never_touches_the_download_dir():
    """Performance/correctness guard: FineWeb-Edu's share=0 (the default)
    must not scan/extract/tokenize anything, even if a nonexistent or
    huge directory is configured -- unlike the other 3 sources (all
    small, always prepared unconditionally), FineWeb-Edu can be hundreds
    of GB, so this must be skip-by-default, not skip-if-share-and-happens
    -to-be-fast."""
    if not _HAS_CACHED_DATA:
        print("  [skip] tokenized data_cache not present locally")
        return
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # Point at a directory that would error if actually scanned (doesn't
    # exist) -- share=0 must mean this is never touched.
    ds = MixedLocalDataset(
        tokenizer, seq_len=512, cache_dir=os.path.join(_REPO_ROOT, "data_cache"),
        fineweb_edu_dir="/definitely/does/not/exist/at/all",
        mix=(0.85, 0.0, 0.15, 0.0),
    )
    assert "FineWeb-Edu" not in ds.source_names


def test_mixed_dataset_with_real_fineweb_edu_source():
    """End-to-end: build a real MixedLocalDataset with FineWeb-Edu as the
    ONLY source, using the real small downloaded parquet file, and
    confirm samples are actually produced."""
    if not (_HAS_SMALL_FINEWEB and _HAS_CACHED_DATA):
        print("  [skip] small FineWeb-Edu test file or tokenized data_cache not present")
        return
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    buf = io.StringIO()
    # Issue #120 real bug hit while writing this test: the FIRST draft
    # used the shared data_cache/ while an earlier (pre-fix) STALE
    # background process was still unconditionally re-extracting all of
    # FineWeb-Edu, and both processes' os.replace() calls collided on the
    # same data_cache/tokenized/fineweb_edu.bin.tmp path -- a genuine
    # PermissionError, not flaky. That stale process (running old,
    # pre-share-gated code) has been stopped and its incomplete output
    # deleted; SlimPajama/OASST/WikiText caches are safe to share (only
    # ever read, never re-tokenized once cached) and nothing else writes
    # to fineweb_edu.bin.tmp now that loading is share-gated. Using the
    # shared cache_dir here (not an isolated tempdir) so this test reuses
    # the already-cached SlimPajama/OASST/WikiText .bin files instead of
    # re-extracting multi-GB corpora from scratch every run.
    # Issue #121: this test writes REAL artifacts into the shared
    # data_cache/ (fineweb_edu_en.txt, tokenized/fineweb_edu.bin), capped
    # tiny (5MB) for test speed. Without cleanup, a real training run
    # using the same shared data_cache/ later silently hit
    # _tokenize_text_to_bin's cache-hit path and reused this test's
    # leftover 5MB-derived bin instead of tokenizing its own much larger
    # real extraction -- caught only by manually reading a training run's
    # log closely enough to notice an unexpected "cache hit" -- not
    # caught by any automated check. try/finally so cleanup happens even
    # if an assertion below fails.
    fineweb_txt_cache = os.path.join(_REPO_ROOT, "data_cache", "fineweb_edu_en.txt")
    fineweb_bin_cache = os.path.join(_REPO_ROOT, "data_cache", "tokenized", "fineweb_edu.bin")
    try:
        with redirect_stdout(buf):
            ds = MixedLocalDataset(
                tokenizer, seq_len=512, cache_dir=os.path.join(_REPO_ROOT, "data_cache"),
                fineweb_edu_dir=_SMALL_FINEWEB_DIR,
                fineweb_edu_token_bytes=5_000_000,  # keep it fast
                mix=(0.0, 0.0, 0.0, 1.0),  # FineWeb-Edu only
            )
        out = buf.getvalue()
        assert ds.source_names == ["FineWeb-Edu"]
        assert len(ds) > 0
        x, y, src_id = ds[0]
        assert x.shape == (512,)
        assert y.shape == (512,)
        assert src_id == 0
        assert "[data mix]" in out
    finally:
        # ds.sources holds numpy.memmap handles onto fineweb_bin_cache --
        # Windows refuses to delete a file that's still memory-mapped by
        # this (or any) process, even in a finally block, unless the
        # memmap is explicitly released first (a bare `del ds` isn't
        # reliably enough -- Python's GC timing isn't guaranteed to
        # release the underlying OS handle before the next line runs).
        if "ds" in locals():
            ds.sources.clear()
            del ds
        import gc
        gc.collect()
        for path in (fineweb_txt_cache, fineweb_bin_cache):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except PermissionError:
                    print(f"  [warn] could not remove {path} (still locked) -- "
                          f"a leftover from this run, harmless but worth a manual cleanup")


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa
            failed += 1
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
