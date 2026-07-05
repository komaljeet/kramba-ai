"""
Test suite for Issue #41 — real web search tool.

Replaces the old canned-dict mock ``web_search`` (tools.py) with a live search
backend (keyless DuckDuckGo Instant Answer API by default, Tavily when a key is
present).  These tests mock the HTTP layer so they run offline and verify:

  * DuckDuckGo JSON is parsed into ranked (title, snippet, url) result lines,
  * results flow through ``ToolExecutor`` and honour ``max_result_len`` truncation,
  * network / parse errors degrade to a clear error string (never a crash),
  * provider selection (env override + Tavily auto-select on key) works,
  * empty queries are rejected and the registry still wires ``web_search``.

Pure tool-layer feature — no GPU / model needed:  python test_issue41.py
"""

import json

# --- repo-root path bootstrap (added during reorg, issue #48) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path bootstrap ---
import hvn.tools as tools
from hvn.tools import (
    web_search,
    ToolExecutor,
    create_demo_registry,
    DEFAULT_SEARCH_RESULTS,
)


# --- fixtures -------------------------------------------------------------

_DDG_JSON = json.dumps({
    "Heading": "Python (programming language)",
    "AbstractText": "Python is a high-level, general-purpose programming language.",
    "AbstractURL": "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "Answer": "",
    "Definition": "",
    "RelatedTopics": [
        {"Text": "Guido van Rossum created Python in 1991.",
         "FirstURL": "https://example.com/guido"},
        {"Topics": [
            {"Text": "CPython is the reference implementation.",
             "FirstURL": "https://example.com/cpython"},
        ]},
    ],
})

_TAVILY_JSON = json.dumps({
    "answer": "Python is a popular high-level language.",
    "results": [
        {"title": "Python.org", "content": "The official home of Python.",
         "url": "https://python.org"},
        {"title": "Wikipedia", "content": "Python programming language.",
         "url": "https://en.wikipedia.org/wiki/Python"},
    ],
})


class _EnvGuard:
    """Context manager that restores search-related env vars afterwards."""

    _KEYS = ("HVN_SEARCH_PROVIDER", "TAVILY_API_KEY")

    def __init__(self, **overrides):
        self.overrides = overrides

    def __enter__(self):
        import os
        self._saved = {k: os.environ.get(k) for k in self._KEYS}
        for k in self._KEYS:
            os.environ.pop(k, None)
        for k, v in self.overrides.items():
            os.environ[k] = v
        return self

    def __exit__(self, *exc):
        import os
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _patch(monkey_get=None, monkey_post=None):
    """Swap the HTTP helpers; returns a restore() callable."""
    orig_get, orig_post = tools._http_get, tools._http_post_json
    if monkey_get is not None:
        tools._http_get = monkey_get
    if monkey_post is not None:
        tools._http_post_json = monkey_post

    def restore():
        tools._http_get = orig_get
        tools._http_post_json = orig_post
    return restore


# --- tests ----------------------------------------------------------------

def test_duckduckgo_parse():
    """DDG JSON → ranked result lines with title/snippet/url, abstract first."""
    captured = {}

    def fake_get(url, timeout, headers=None):
        captured["url"] = url
        return _DDG_JSON

    restore = _patch(monkey_get=fake_get)
    try:
        with _EnvGuard():  # no key -> duckduckgo default
            out = web_search("python programming")
    finally:
        restore()

    assert "api.duckduckgo.com" in captured["url"]
    assert "Search results for 'python programming':" in out
    # Abstract (instant answer) is ranked first with its heading + url.
    assert "1. Python (programming language):" in out
    assert "high-level" in out
    assert "en.wikipedia.org" in out
    # Related topics (flat + nested) are flattened in.
    assert "Guido van Rossum" in out
    assert "CPython is the reference implementation." in out
    print("[ok] test_duckduckgo_parse")


def test_max_results_cap():
    """max_results caps the number of entries emitted."""
    def fake_get(url, timeout, headers=None):
        return _DDG_JSON

    restore = _patch(monkey_get=fake_get)
    try:
        with _EnvGuard():
            out = web_search("python", max_results=2)
    finally:
        restore()

    # Header line + at most 2 numbered results.
    numbered = [ln for ln in out.splitlines() if ln[:2] in ("1.", "2.", "3.")]
    assert len(numbered) == 2, f"expected 2 results, got {len(numbered)}"
    assert "3." not in out
    print("[ok] test_max_results_cap")


def test_executor_truncation():
    """Long results are truncated by ToolExecutor.max_result_len."""
    long_json = json.dumps({
        "Heading": "Big",
        "AbstractText": "x " * 2000,  # ~4000 chars
        "AbstractURL": "",
        "RelatedTopics": [],
    })

    def fake_get(url, timeout, headers=None):
        return long_json

    restore = _patch(monkey_get=fake_get)
    try:
        registry = create_demo_registry()
        executor = ToolExecutor(registry, max_result_len=120)
        with _EnvGuard():
            block = executor.execute({"name": "web_search", "arguments": {"query": "big"}})
    finally:
        restore()

    assert "[truncated]" in block
    # The formatted-and-truncated payload must respect the cap (+ suffix).
    body = block  # execute() returns a [TOOL_RESULT] wrapped string
    assert "...[truncated]" in body
    print("[ok] test_executor_truncation")


def test_network_error_graceful():
    """A network error returns a clear error string, not an exception."""
    def boom(url, timeout, headers=None):
        raise OSError("connection refused")

    restore = _patch(monkey_get=boom)
    try:
        with _EnvGuard():
            out = web_search("anything")
    finally:
        restore()

    assert out.startswith("Error: web search failed")
    assert "OSError" in out
    print("[ok] test_network_error_graceful")


def test_empty_query_rejected():
    """Empty / whitespace queries are rejected before any network call."""
    called = {"n": 0}

    def fake_get(url, timeout, headers=None):
        called["n"] += 1
        return _DDG_JSON

    restore = _patch(monkey_get=fake_get)
    try:
        with _EnvGuard():
            out = web_search("   ")
    finally:
        restore()

    assert out.startswith("Error: empty search query")
    assert called["n"] == 0, "no network call should be made for an empty query"
    print("[ok] test_empty_query_rejected")


def test_provider_selection_and_tavily():
    """Tavily is auto-selected on key; env override forces a provider."""
    def fake_post(url, payload, timeout, headers=None):
        assert "tavily" in url
        assert payload["query"] == "python"
        return _TAVILY_JSON

    def fake_get(url, timeout, headers=None):
        return _DDG_JSON

    restore = _patch(monkey_get=fake_get, monkey_post=fake_post)
    try:
        # Auto-select Tavily when the key is present.
        with _EnvGuard(TAVILY_API_KEY="secret"):
            assert tools._select_search_provider() == "tavily"
            out = web_search("python")
        assert "Answer: Python is a popular high-level language." in out
        assert "python.org" in out

        # Explicit override wins even with a key set.
        with _EnvGuard(HVN_SEARCH_PROVIDER="duckduckgo", TAVILY_API_KEY="secret"):
            assert tools._select_search_provider() == "duckduckgo"
            out2 = web_search("python")
        assert "Python (programming language)" in out2

        # Unknown provider -> clear error, no crash.
        with _EnvGuard(HVN_SEARCH_PROVIDER="bogus"):
            out3 = web_search("python")
        assert out3.startswith("Error: unknown search provider 'bogus'")
    finally:
        restore()
    print("[ok] test_provider_selection_and_tavily")


def test_registry_wires_web_search():
    """The demo registry still exposes web_search with its schema unchanged."""
    registry = create_demo_registry()
    assert registry.exists("web_search")
    schema = registry.get("web_search")["schema"]
    assert schema["properties"]["query"]["type"] == "string"
    assert "query" in schema["required"]
    assert DEFAULT_SEARCH_RESULTS == 5
    print("[ok] test_registry_wires_web_search")


def main():
    tests = [
        test_duckduckgo_parse,
        test_max_results_cap,
        test_executor_truncation,
        test_network_error_graceful,
        test_empty_query_rejected,
        test_provider_selection_and_tavily,
        test_registry_wires_web_search,
    ]
    passed = 0
    for t in tests:
        t()
        passed += 1
    print(f"\n{passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
