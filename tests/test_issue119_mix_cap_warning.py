"""Test for Issue #119: build_dataset must print a clear, CORRECT warning
about what's actually limiting the total pretrain token budget --
distinguishing two genuinely different cases:

  1. A MIX-RATIO bind: some source's max_repeat cap is the active
     constraint (there's more of another source available than this mix
     could ever use at its current share). Fixable by changing the mix
     (drop/reduce that source's share, or raise max_repeat). This is the
     mechanism that capped the old 83M pretrain at ~135M tokens via OASST.
  2. A DATA-VOLUME limit: every available window from every source is
     already used, no one's repeat cap is binding. Fixable only by
     downloading more data -- changing mix ratios does nothing here.

An earlier version of this warning conflated these two cases (always
blaming whichever source's repeat-cap term was numerically smallest,
even when that term wasn't actually the active constraint) -- caught by
running it against this project's real cached data_cache, where the
current default mix + max_repeat=3 turns out to be DATA-VOLUME-limited
(only ~1.32B SlimPajama tokens cached), not MIX-RATIO-limited, even
though WikiText IS documented (in the #119 issue) to become the binding
mix-ratio constraint once more data is downloaded.

Run:  python -m pytest tests/test_issue119_mix_cap_warning.py -q
  or: python tests/test_issue119_mix_cap_warning.py
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import GPT2TokenizerFast

from train_edge_conversational import MixedLocalDataset

_TOK_DIR = os.path.join(_REPO_ROOT, "data_cache", "tokenized")
_HAS_CACHED_DATA = all(
    os.path.exists(os.path.join(_TOK_DIR, f))
    for f in ("slimpajama.bin", "oasst.bin", "wikitext.bin")
)


def _build(mix, max_repeat=3):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    buf = io.StringIO()
    with redirect_stdout(buf):
        ds = MixedLocalDataset(
            tokenizer, seq_len=512, cache_dir=os.path.join(_REPO_ROOT, "data_cache"),
            mix=mix, max_repeat=max_repeat,
        )
    return ds, buf.getvalue()


def test_current_cached_volume_is_data_limited_not_mix_limited():
    """With only ~1.32B SlimPajama tokens currently cached, the default
    Chinchilla-floor mix (0.85/0.0/0.15) at max_repeat=3 is limited by
    DATA VOLUME, not by WikiText's mix share -- every available window is
    already used (total_chunks == sum(avail)). Confirms the warning
    correctly does NOT blame the mix ratio when it isn't actually the
    active constraint."""
    if not _HAS_CACHED_DATA:
        print("  [skip] tokenized data_cache not present locally")
        return
    ds, out = _build(mix=(0.85, 0.0, 0.15))
    assert "[data mix] total pretrain budget capped at" in out
    assert "by DATA VOLUME, not the mix ratio" in out, (
        "expected a data-volume-limited message at current cached data; got:\n"
        + [l for l in out.splitlines() if "[data mix]" in l][0]
    )
    assert "by a MIX-RATIO bind" not in out


def test_lower_max_repeat_exposes_wikitext_as_the_real_mix_ratio_bind():
    """Lowering max_repeat (less tolerance for repeating the smaller
    source) against the SAME real cached data flips this into the
    MIX-RATIO-bind case, and WikiText-103 is correctly named -- this is
    the actual #119 failure mode, reproduced directly against real data,
    not simulated."""
    if not _HAS_CACHED_DATA:
        print("  [skip] tokenized data_cache not present locally")
        return
    ds, out = _build(mix=(0.85, 0.0, 0.15), max_repeat=1)
    assert "by a MIX-RATIO bind" in out, (
        "expected max_repeat=1 to expose a mix-ratio bind against real cached data; got:\n"
        + [l for l in out.splitlines() if "[data mix]" in l][0]
    )
    assert "WikiText-103" in [l for l in out.splitlines() if "binding source" in l][0]


def test_dropping_wikitext_removes_it_from_the_mix_entirely():
    """SlimPajama-only mix (WikiText share=0) -- WikiText no longer
    appears at all (not a source, not a binding candidate); the cap
    becomes purely a function of cached SlimPajama volume."""
    if not _HAS_CACHED_DATA:
        print("  [skip] tokenized data_cache not present locally")
        return
    ds, out = _build(mix=(1.0, 0.0, 0.0))
    mix_line = [l for l in out.splitlines() if "[data mix]" in l][0]
    assert "WikiText-103" not in mix_line, (
        "WikiText excluded (share=0) should not appear in the mix-cap warning "
        f"itself, even though its raw tokenization is still logged separately: {mix_line}"
    )
    assert "by DATA VOLUME, not the mix ratio" in mix_line


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
