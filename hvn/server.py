"""
OpenAI-Compatible API Server for HVN (Issue #31)
=================================================

Exposes a trained Hamiltonian Vortex Network over an OpenAI-compatible REST
API so it can be used from LangChain, OpenWebUI, LibreChat, the OpenAI
Python SDK, curl, etc. — anything that speaks the ``/v1/chat/completions``
protocol.

Ollama is not an option here (llama.cpp/GGUF only supports Transformer-family
architectures), so this is a small FastAPI server built directly on HVN's
existing incremental-generation code (``forward_step`` / Hamiltonian state
cache) — no KV cache, O(1) per-token inference is preserved end to end.

Endpoints
---------
- ``POST /v1/chat/completions`` — chat completions (streaming + non-streaming)
- ``POST /v1/completions``      — text completions (streaming + non-streaming)
- ``GET  /v1/models``           — list the loaded model
- ``GET  /v1/health``           — health check

Usage
-----
    python -m hvn.server --checkpoint hvn_checkpoint.pt --port 8000
    python -m hvn.server --checkpoint hvn_checkpoint.pt --lora hvn_checkpoint_lora.pt

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="hvn")
    response = client.chat.completions.create(
        model="hvn",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Scope note: a single model is loaded per server process (no dynamic
multi-model switching) — ``/v1/models`` lists that one model, which
satisfies the OpenAI models-list contract without the added complexity of
hot-swapping checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from hvn.core import HamiltonianVortexNetwork
from hvn.tools import TOOL_TOKENS, ToolCallParser


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, tokenizer_path: str, device: torch.device):
    """Load a trained HVN model + tokenizer from a checkpoint (see generate.py)."""
    from train import CharTokenizer

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = HamiltonianVortexNetwork(
        vocab_size=ckpt["vocab_size"],
        d_model=ckpt["d_model"],
        num_layers=ckpt["num_layers"],
        n_heads=ckpt.get("n_heads", 4),
        chunk_size=ckpt.get("chunk_size", 64),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    try:
        from hvn.tokenizer import BPETokenizer
        tokenizer = BPETokenizer.load(tokenizer_path)
    except Exception:
        tokenizer = CharTokenizer.load(tokenizer_path)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Server state (single model per process — see module docstring scope note)
# ---------------------------------------------------------------------------

class ServerState:
    model: Any = None
    tokenizer: Any = None
    device: Optional[torch.device] = None
    served_model_name: str = "hvn"
    max_new_tokens_default: int = 200
    created: int = int(time.time())


STATE = ServerState()


def _require_model():
    if STATE.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return STATE.model, STATE.tokenizer, STATE.device


# ---------------------------------------------------------------------------
# O(1) incremental token generator (mirrors model.generate_incremental, but
# yields token IDs one at a time so we can stream them over SSE)
# ---------------------------------------------------------------------------

@torch.no_grad()
def stream_token_ids(
    model,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
):
    if not prompt_ids:
        prompt_ids = [0]

    dtype = next(model.parameters()).dtype
    states = model.init_generation_state(1, device=device, dtype=dtype)

    logits_step = None
    for t, tok_id in enumerate(prompt_ids):
        tok = torch.tensor([[tok_id]], dtype=torch.long, device=device)
        logits_step, states = model.forward_step(tok, states, position_idx=t)

    temp = max(temperature, 1e-5)
    next_logits = logits_step[0, -1, :] / temp

    for step in range(max_new_tokens):
        if top_k > 0:
            k = min(top_k, next_logits.size(-1))
            values, _ = torch.topk(next_logits, k)
            next_logits = next_logits.clone()
            next_logits[next_logits < values[-1]] = float("-inf")

        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        yield next_id

        tok = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits_step, states = model.forward_step(tok, states, position_idx=len(prompt_ids) + step)
        next_logits = logits_step[0, -1, :] / temp


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
    prompt_ids = tokenizer.encode(prompt) or [0]
    ids = list(prompt_ids)
    for tok_id in stream_token_ids(model, prompt_ids, max_new_tokens, temperature, top_k, device):
        ids.append(tok_id)
    return tokenizer.decode(ids[len(prompt_ids):])


def stream_text_chunks(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
):
    """Yield incremental text deltas as tokens are generated.

    Decodes the full generated-id sequence at each step and yields only the
    new suffix — this avoids emitting malformed partial-BPE-token text.
    """
    prompt_ids = tokenizer.encode(prompt) or [0]
    ids = list(prompt_ids)
    prev_text = ""
    for tok_id in stream_token_ids(model, prompt_ids, max_new_tokens, temperature, top_k, device):
        ids.append(tok_id)
        full_text = tokenizer.decode(ids[len(prompt_ids):])
        delta = full_text[len(prev_text):]
        prev_text = full_text
        if delta:
            yield delta


# ---------------------------------------------------------------------------
# Chat formatting (reuses hvn.chat's [SYSTEM]/[USER]/[ASSISTANT] protocol)
# ---------------------------------------------------------------------------

def _messages_to_prompt(messages: List["ChatMessage"], tools: Optional[List[Dict[str, Any]]] = None) -> str:
    from hvn.chat import ConversationFormatter

    system_prompt = ""
    turns: List[Dict[str, str]] = []
    for m in messages:
        if m.role == "system" and not system_prompt:
            system_prompt = m.content
        else:
            turns.append({"role": m.role, "content": m.content or ""})

    formatter = ConversationFormatter(system_prompt=system_prompt)
    prompt = formatter.format_for_generation(turns)

    if tools:
        tool_lines = ["Available tools:"]
        for t in tools:
            fn = t.get("function", t)
            name = fn.get("name", "unknown")
            desc = fn.get("description", "")
            schema = json.dumps(fn.get("parameters", {}), separators=(",", ":"))
            tool_lines.append(f"  - {name}: {desc} Schema: {schema}")
        tool_lines.append(
            f"Use {TOOL_TOKENS['tool_call_start']}{{\"name\":...,\"arguments\":{{...}}}}"
            f"{TOOL_TOKENS['tool_call_end']} to call a tool."
        )
        prompt = "\n".join(tool_lines) + "\n\n" + prompt

    return prompt


def _parse_tool_calls_to_openai(text: str) -> Optional[List[Dict[str, Any]]]:
    calls = ToolCallParser.parse_tool_calls(text)
    if not calls:
        return None
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": c["name"],
                "arguments": json.dumps(c["arguments"], separators=(",", ":")),
            },
        }
        for c in calls
    ]


# ---------------------------------------------------------------------------
# OpenAI-compatible request/response schemas (subset)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = ""


class ChatCompletionRequest(BaseModel):
    model: str = "hvn"
    messages: List[ChatMessage]
    temperature: float = 0.8
    top_p: Optional[float] = None
    top_k: int = 10
    max_tokens: int = 200
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    functions: Optional[List[Dict[str, Any]]] = None


class CompletionRequest(BaseModel):
    model: str = "hvn"
    prompt: str
    temperature: float = 0.8
    top_k: int = 10
    max_tokens: int = 200
    stream: bool = False


def _usage(prompt_tokens: int, completion_tokens: int) -> Dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="HVN OpenAI-Compatible API Server")


@app.get("/v1/health")
def health():
    return {"status": "ok", "model_loaded": STATE.model is not None}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": STATE.served_model_name,
                "object": "model",
                "created": STATE.created,
                "owned_by": "hvn",
            }
        ],
    }


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    model, tokenizer, device = _require_model()
    prompt_ids = tokenizer.encode(req.prompt) or [0]
    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"

    if req.stream:
        def event_stream():
            for delta in stream_text_chunks(
                model, tokenizer, req.prompt, req.max_tokens, req.temperature, req.top_k, device
            ):
                chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": STATE.served_model_name,
                    "choices": [{"text": delta, "index": 0, "logprobs": None, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            final = {
                "id": completion_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": STATE.served_model_name,
                "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    text = generate_text(model, tokenizer, req.prompt, req.max_tokens, req.temperature, req.top_k, device)
    completion_ids = tokenizer.encode(text) or []
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": STATE.served_model_name,
        "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": "stop"}],
        "usage": _usage(len(prompt_ids), len(completion_ids)),
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    model, tokenizer, device = _require_model()
    tools = req.tools or req.functions
    prompt = _messages_to_prompt(req.messages, tools=tools)
    prompt_ids = tokenizer.encode(prompt) or [0]
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if req.stream:
        def event_stream():
            first = True
            for delta in stream_text_chunks(
                model, tokenizer, prompt, req.max_tokens, req.temperature, req.top_k, device
            ):
                delta_obj = {"content": delta}
                if first:
                    delta_obj = {"role": "assistant", "content": delta}
                    first = False
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": STATE.served_model_name,
                    "choices": [{"index": 0, "delta": delta_obj, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": STATE.served_model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    raw_text = generate_text(model, tokenizer, prompt, req.max_tokens, req.temperature, req.top_k, device)
    tool_calls = _parse_tool_calls_to_openai(raw_text) if tools else None
    content = ToolCallParser.strip_tool_blocks(raw_text) if tool_calls else raw_text
    completion_ids = tokenizer.encode(raw_text) or []

    message: Dict[str, Any] = {"role": "assistant", "content": content or None}
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": STATE.served_model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": _usage(len(prompt_ids), len(completion_ids)),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible API server for HVN")
    parser.add_argument("--checkpoint", type=str, default="hvn_checkpoint.pt")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer JSON (default: auto-detect from checkpoint name)")
    parser.add_argument("--lora", type=str, default=None,
                        help="Path to a LoRA checkpoint to apply on top of the base model at startup")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--served_model_name", type=str, default="hvn")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_path = args.tokenizer or args.checkpoint.replace(".pt", "_tokenizer.json")

    model, tokenizer = load_model(args.checkpoint, tok_path, device)

    if args.lora:
        from hvn.lora import apply_lora, load_lora_checkpoint
        apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        load_lora_checkpoint(model, args.lora)
        model.to(device)
        model.eval()

    STATE.model = model
    STATE.tokenizer = tokenizer
    STATE.device = device
    STATE.served_model_name = args.served_model_name

    print(f"✓ Loaded model: {args.checkpoint} (served as '{args.served_model_name}')")
    print(f"✓ Device: {device}")
    print(f"✓ Serving OpenAI-compatible API at http://{args.host}:{args.port}/v1")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
