"""
Verification tests for the OpenAI-compatible API server (Issue #31).

Starts hvn/server.py in a subprocess against the bundled checkpoint, then
exercises every documented success criterion: health, models list, chat
completions (streaming + non-streaming), text completions, and tool-call
exposure. Run with the checkpoint already trained (hvn_checkpoint.pt).

    python test_server.py
"""

import json
import subprocess
import sys
import time
import urllib.request

PORT = 8199
BASE = f"http://127.0.0.1:{PORT}"


def _post(path, payload, stream=False):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=30)
    if stream:
        return resp
    return json.loads(resp.read().decode("utf-8"))


def _get(path):
    resp = urllib.request.urlopen(f"{BASE}{path}", timeout=10)
    return json.loads(resp.read().decode("utf-8"))


def run_tests():
    passed = 0
    total = 0

    def check(name, cond):
        nonlocal passed, total
        total += 1
        status = "PASS" if cond else "FAIL"
        if cond:
            passed += 1
        print(f"[{status}] {name}")

    health = _get("/v1/health")
    check("GET /v1/health returns ok", health.get("status") == "ok" and health.get("model_loaded"))

    models = _get("/v1/models")
    check("GET /v1/models lists the served model", len(models.get("data", [])) >= 1)

    comp = _post("/v1/completions", {"model": "hvn", "prompt": "The ", "max_tokens": 10})
    check("POST /v1/completions returns text", isinstance(comp["choices"][0]["text"], str) and len(comp["choices"][0]["text"]) > 0)

    chat = _post("/v1/chat/completions", {"model": "hvn", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10})
    check("POST /v1/chat/completions returns assistant message",
          chat["choices"][0]["message"]["role"] == "assistant" and chat["choices"][0]["finish_reason"] == "stop")

    stream_resp = _post(
        "/v1/chat/completions",
        {"model": "hvn", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8, "stream": True},
        stream=True,
    )
    chunks = 0
    saw_done = False
    for line in stream_resp:
        line = line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            saw_done = True
            break
        chunks += 1
    check("Streaming chat completions yields chunks + [DONE]", chunks > 0 and saw_done)

    tool_call_text = _post(
        "/v1/chat/completions",
        {
            "model": "hvn",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1,
            "tools": [{"type": "function", "function": {"name": "calculator", "description": "adds", "parameters": {}}}],
        },
    )
    check("Tool-enabled request completes without error", "choices" in tool_call_text)

    print(f"\n{passed}/{total} checks passed")
    return passed == total


def main():
    proc = subprocess.Popen(
        [sys.executable, "-m", "hvn.server", "--checkpoint", "hvn_checkpoint.pt", "--port", str(PORT)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        # Wait for server to become ready
        for _ in range(60):
            try:
                _get("/v1/health")
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("Server did not start in time")

        ok = run_tests()
        sys.exit(0 if ok else 1)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
