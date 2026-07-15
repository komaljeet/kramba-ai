"""Tests for hvn/tools.py's coding tools (issue #110 / #97 Phase C):
read_file / edit_file / run_tests, sandboxed via create_coding_registry.

Covers: normal read/edit/run_tests round-trip, path-traversal rejection
(both relative ``..`` and absolute paths), and that ``run_tests`` fails
safely (not silently, not by executing anything) when the model passes it
an unexpected argument -- the security property the module docstring
describes (ToolExecutor calls ``func(**args)`` with whatever the model
produced; a zero-arg function makes any injected argument a TypeError,
which ToolExecutor's existing try/except turns into an error string).
"""

from __future__ import annotations

import os
import tempfile

import pytest

from hvn.tools import ToolExecutor, create_coding_registry


@pytest.fixture
def sandbox():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "add.py"), "w", encoding="utf-8") as f:
            f.write("def add(a, b):\n    return a - b\n")
        with open(os.path.join(tmp, "test_add.py"), "w", encoding="utf-8") as f:
            f.write("from add import add\ndef test_add():\n    assert add(2, 3) == 5\n")
        yield tmp


@pytest.fixture
def executor(sandbox):
    return ToolExecutor(create_coding_registry(sandbox))


def test_read_file_returns_contents(executor):
    result = executor.execute({"name": "read_file", "arguments": {"path": "add.py"}})
    assert "return a - b" in result
    assert "[TOOL_RESULT]" in result and "[/TOOL_RESULT]" in result


def test_read_file_missing_file_errors(executor):
    result = executor.execute({"name": "read_file", "arguments": {"path": "nope.py"}})
    assert "Error" in result


@pytest.mark.parametrize("bad_path", ["../../etc/passwd", "..\\..\\windows\\win.ini",
                                      "/etc/passwd", "sub/../../escape.py"])
def test_read_file_rejects_path_traversal(executor, bad_path):
    result = executor.execute({"name": "read_file", "arguments": {"path": bad_path}})
    assert "Error" in result and "escapes the sandbox root" in result


def test_edit_file_replaces_first_occurrence(executor, sandbox):
    result = executor.execute({
        "name": "edit_file",
        "arguments": {"path": "add.py", "old_text": "return a - b", "new_text": "return a + b"},
    })
    assert "OK" in result
    with open(os.path.join(sandbox, "add.py"), encoding="utf-8") as f:
        assert "return a + b" in f.read()


def test_edit_file_missing_old_text_errors(executor):
    result = executor.execute({
        "name": "edit_file",
        "arguments": {"path": "add.py", "old_text": "not present anywhere", "new_text": "x"},
    })
    assert "Error" in result and "old_text not found" in result


def test_edit_file_rejects_path_traversal(executor):
    result = executor.execute({
        "name": "edit_file",
        "arguments": {"path": "../outside.py", "old_text": "a", "new_text": "b"},
    })
    assert "Error" in result and "escapes the sandbox root" in result


def test_run_tests_reports_failure_then_passes_after_fix(executor, sandbox):
    before = executor.execute({"name": "run_tests", "arguments": {}})
    assert "FAILED" in before

    executor.execute({
        "name": "edit_file",
        "arguments": {"path": "add.py", "old_text": "return a - b", "new_text": "return a + b"},
    })

    after = executor.execute({"name": "run_tests", "arguments": {}})
    assert "PASSED" in after


def test_run_tests_rejects_injected_argument(executor):
    """A model (or an adversarial prompt) passing run_tests an argument
    must fail safely -- not execute anything, not silently ignore it and
    run pytest anyway with unexpected behavior."""
    result = executor.execute({"name": "run_tests", "arguments": {"command": "echo pwned"}})
    assert "Error executing 'run_tests'" in result
    assert "unexpected keyword argument" in result
