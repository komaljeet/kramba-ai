"""Minimal OpenAI-compatible HTTP server for EdgeHVN (Issue #137 follow-on).

Written by: Rambarun Komaljeet

Exposes ``POST /v1/chat/completions`` and ``GET /v1/models`` over an EdgeHVN
checkpoint, so any client that speaks the OpenAI API — VS Code's Continue
extension, curl, an SDK — can talk to a locally-trained model. Standard library
``http.server`` only: **no FastAPI, no uvicorn, no new dependency.**

This exists because the repo's only serving surface was the Docker/Django chat
app, which speaks its own REST protocol and needs postgres + redis + celery
running. That is the right thing for a chat product and the wrong thing for
"point my editor at it".

**Optional retrieval (#137).** With ``--rag-corpus``, each user message is used
as a BM25 query and the single best passage is prepended as context. One passage,
not three: #137 measured `top-k=1` beating `top-k=3` by 16 points on this model,
because an 83M model has little capacity to ignore an irrelevant passage and the
context budget is only ~512 tokens.

Not implemented, deliberately, and listed so nobody assumes otherwise: streaming
(``stream: true`` is rejected rather than silently ignored), authentication,
concurrency (single-threaded — one request at a time), and function calling.
This is a local development bridge, not a production endpoint; do not expose it
beyond localhost.

Usage
-----
    python serve_openai.py --checkpoint checkpoints/hybrid_chat_v1.pt
    python serve_openai.py --checkpoint ... --rag-corpus data/facts.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Overwritten from the checkpoint at load time (#139): EOS belongs to the
# tokenizer, and a 32K-vocab model's is 0, not GPT-2's 50256. Serving with the
# wrong one means the stop token is never sampled and every reply runs to the
# token limit.
EOS = 50256
USER_PREFIX = "<|user|>\n"
ASST_PREFIX = "<|assistant|>\n"

_STATE: dict = {}


@torch.no_grad()
def generate(prompt: str, max_new: int, temperature: float, top_p: float,
             repetition_penalty: float) -> str:
    """Sampling loop over the full-forward path.

    Uses ``model(x)`` rather than ``forward_step``: this server re-forwards the
    whole prompt each token, which is O(N) per token instead of O(1) and throws
    away the architecture's entire inference advantage. It is correct and simple,
    which is what a bridge needs; wiring the incremental path (#97) through here
    is the obvious follow-up and is gated on #138's ring-buffer work landing.
    """
    model, tok, device = _STATE["model"], _STATE["tok"], _STATE["device"]
    out, _stopped = generate_ids(model, tok.encode(prompt), eos=EOS,
                                 device=device, max_new=max_new,
                                 temperature=temperature, top_p=top_p,
                                 repetition_penalty=repetition_penalty,
                                 decode=tok.decode)
    return tok.decode(out).strip()


@torch.no_grad()
def generate_ids(model, ids: list[int], *, eos: int, device, max_new: int,
                 temperature: float, top_p: float,
                 repetition_penalty: float,
                 decode=None) -> tuple[list[int], bool]:
    """Model-agnostic core of :func:`generate`. Returns ``(new_ids, hit_eos)``.

    ``hit_eos`` is exposed for ``eval/chat_coherence.py``, which scores whether
    a reply terminates on its own or runs to the token limit. Also accepts a
    Hugging Face causal LM (output has ``.logits``) so a reference model is
    decoded by the *same* loop, not a parallel one.

    ``decode`` (ids -> str, e.g. ``tok.decode``): when given, the repetition
    penalty is switched OFF while generation is inside an unclosed
    ``[TOOL_CALL]`` block (#167). Arguments legitimately repeat tokens --
    ``"2+2"`` -- and at 1.3 the second ``2`` lost to ``22``. Plain text
    outside the block keeps the full penalty.
    """
    import torch.nn.functional as F
    from hvn.tools import ToolCallParser
    x = torch.tensor([ids], device=device)
    out: list[int] = []
    for _ in range(max_new):
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"),
                                dtype=torch.bfloat16):
            res = model(x)
        logits = getattr(res, "logits", res)[0, -1].float()
        in_call = (decode is not None and bool(out)
                   and ToolCallParser.has_tool_call_start(decode(out)))
        for t in (() if in_call else set(out)):
            # Sign-aware (CTRL), as EdgeHVN.chat_step does. Plain division
            # RAISED a negative logit toward 0, i.e. favoured repeating it.
            logits[t] = (logits[t] / repetition_penalty if logits[t] > 0
                         else logits[t] * repetition_penalty)
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, dim=-1)
            sp[cum - sp > top_p] = 0
            sp = sp / sp.sum()
            nxt = int(si[torch.multinomial(sp, 1)])
        if nxt == eos:
            return out, True
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return out, False


def build_prompt(messages) -> str:
    """Render OpenAI-style messages into EdgeHVN's ACTUAL trained format.

    The model was fine-tuned on ``<|user|>`` / ``<|assistant|>`` turns terminated
    by EOS. #122 recorded a fully retracted result caused by probing this model
    outside its trained template, so the mapping is explicit rather than
    improvised: system messages are folded into the first user turn, because
    EdgeHVN was never trained on a system role and inventing a marker for one
    would put it off-distribution.
    """
    index = _STATE.get("index")
    system = " ".join(m["content"] for m in messages if m.get("role") == "system")
    parts = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "assistant":
            # Trained format ends an assistant turn with EOS and no newline
            # (finetune_edge_conversational.py); GPT-2's tokenizer maps this
            # literal to id 50256. A bare "\n" here put every multi-turn
            # prompt off-distribution.
            parts.append(ASST_PREFIX + m["content"] + "<|endoftext|>")
        else:
            content = m["content"]
            if index is not None:
                got = index.retrieve(content, k=1)     # ONE passage — see #137
                if got:
                    content = (f"Answer the question using only the context "
                               f"below.\n\nContext: {got[0]}\n\n"
                               f"Question: {content}")
            parts.append(USER_PREFIX + content + "\n")
    if system and parts:
        parts[0] = USER_PREFIX + system + "\n\n" + parts[0][len(USER_PREFIX):]
    return "".join(parts) + ASST_PREFIX


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):          # quieter default logging
        sys.stderr.write("[serve] " + fmt % args + "\n")

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send(200, {"object": "list", "data": [
                {"id": _STATE["model_name"], "object": "model",
                 "owned_by": "kramba-ai"}]})
        else:
            self._send(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._send(404, {"error": {"message": f"no route {self.path}"}})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            self._send(400, {"error": {"message": f"bad JSON: {e}"}})
            return

        if req.get("stream"):
            # Rejected rather than silently ignored: a client that asked for
            # streaming and got one blob would look like a hang.
            self._send(400, {"error": {
                "message": "streaming is not implemented by this server; "
                           "set \"stream\": false"}})
            return

        messages = req.get("messages") or []
        if not messages:
            self._send(400, {"error": {"message": "messages[] is required"}})
            return

        prompt = build_prompt(messages)
        t0 = time.time()
        text = generate(
            prompt,
            max_new=int(req.get("max_tokens") or _STATE["max_tokens"]),
            temperature=float(req.get("temperature", 0.7)),
            top_p=float(req.get("top_p", 0.9)),
            repetition_penalty=_STATE["repetition_penalty"],
        )
        print(f"[serve] {len(prompt)} chars in -> {len(text)} chars out "
              f"in {time.time() - t0:.2f}s", flush=True)

        self._send(200, {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _STATE["model_name"],
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            # Token counts are omitted rather than faked: clients that budget on
            # them should not be handed a made-up number.
            "usage": {},
        })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/hybrid_chat_v1.pt")
    ap.add_argument("--host", default="127.0.0.1",
                    help="localhost by design — this server has no auth")
    ap.add_argument("--port", type=int, default=811)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--repetition-penalty", type=float, default=1.3,
                    help="1.3 is the value the Docker serving path uses; this "
                         "checkpoint loops without it (see docs/state.md)")
    ap.add_argument("--rag-corpus", default=None,
                    help="text file, one passage per line. Enables #137 "
                         "retrieval: the best SINGLE passage is prepended.")
    ap.add_argument("--model-name", default="edgehvn")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[serve] WARNING: no CUDA — this will be slow.")

    from transformers import GPT2TokenizerFast
    from train_edge_conversational import load_checkpoint_for_resume

    print(f"[serve] loading {args.checkpoint}")
    model, step, _ck = load_checkpoint_for_resume(args.checkpoint, device)
    model.eval()
    global EOS
    EOS = getattr(model, "eos_token_id", 50256)
    print(f"[serve] eos_token_id={EOS} (from checkpoint)")

    _STATE.update(model=model, device=device,
                  tok=GPT2TokenizerFast.from_pretrained("gpt2"),
                  model_name=args.model_name, max_tokens=args.max_tokens,
                  repetition_penalty=args.repetition_penalty, index=None)

    if args.rag_corpus:
        from hvn.retrieval import BM25Index
        with open(args.rag_corpus, encoding="utf-8") as fh:
            docs = [ln.strip() for ln in fh if ln.strip()]
        _STATE["index"] = BM25Index(docs)
        print(f"[serve] RAG on: {len(docs)} passages indexed, top-k=1 (#137)")

    print(f"[serve] checkpoint step {step} on {device}")
    print(f"[serve] listening on http://{args.host}:{args.port}/v1")
    HTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
