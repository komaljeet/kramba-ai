# Using EdgeHVN from VS Code

> **Written by: Rambarun Komaljeet**
>
> Point VS Code at a locally-trained EdgeHVN checkpoint over an OpenAI-compatible
> endpoint. No cloud, no API key, no data leaving the machine.

## Read this first — what to expect

This is an **83M-parameter** model. GPT-4 is roughly 20,000x larger. Setting the
expectation honestly here saves an hour of thinking the setup is broken when it
is working exactly as measured:

| you ask it to | expect |
|---|---|
| answer a short factual question | often correct — *"The capital of France is Paris."* |
| chat, greet, acknowledge | coherent |
| emit a tool call / JSON in a learned format | **reliable** — this is its strongest measured skill (#110) |
| answer from a passage you give it | ~22% exact ([state.md](state.md) #137) |
| write a working function from a comment | **no** |
| reason about your codebase | **no** |

The measured gap is contextual reasoning, and `docs/state.md` concludes that is
parameter-count-bound at this scale, not fixable by data or tuning. Treat this as
a local model you *built*, not a Copilot replacement. Issue #136 scopes the code
model that would be a genuine autocomplete.

## 1. Start the server

```bash
python serve_openai.py --checkpoint checkpoints/hybrid_chat_v1.pt --port 8811
```

Standard-library HTTP only — no FastAPI, no uvicorn, nothing new to install. It
binds `127.0.0.1` by design: **there is no authentication, so do not expose it.**

Check it:

```bash
curl -s http://127.0.0.1:8811/v1/models
curl -s -X POST http://127.0.0.1:8811/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":40,"temperature":0.3}'
```

## 2. Point Continue at it

[Continue](https://continue.dev) is the VS Code extension that accepts any
OpenAI-compatible endpoint. Install it, then in `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "EdgeHVN (local 83M)",
      "provider": "openai",
      "model": "edgehvn",
      "apiBase": "http://127.0.0.1:8811/v1",
      "apiKey": "unused",
      "completionOptions": { "maxTokens": 120, "temperature": 0.7 }
    }
  ]
}
```

Pick "EdgeHVN (local 83M)" in Continue's model dropdown. `Ctrl+L` opens the chat.

**Turn Continue's tab-autocomplete OFF for this model.** It is configured
separately (`tabAutocompleteModel`) and this checkpoint was never trained on
code — leaving it on produces noise on every keystroke. See #136.

Any other OpenAI-compatible client works the same way: set the base URL to
`http://127.0.0.1:8811/v1` and the model to `edgehvn`.

## 3. Optional: give it knowledge it does not have (#137)

The model holds almost no facts — that is its measured weakness. Retrieval fixes
that without training. Put one passage per line in a text file:

```
The deploy key lives in infra/secrets/deploy.pem.
The staging database is reachable at db-staging.internal:5432.
```

```bash
python serve_openai.py --checkpoint checkpoints/hybrid_chat_v1.pt \
    --rag-corpus notes.txt --port 8811
```

Each user message becomes a BM25 query and the **single** best passage is
prepended. One, not three, deliberately: #137 measured `top-k=1` beating
`top-k=3` by 16 points, because an 83M model has little capacity to ignore an
irrelevant passage. Measured effect on facts it cannot otherwise know:
**0% → 22.5%**.

**22.5% means it fails roughly three times in four.** A real exchange from this
setup, kept here rather than a flattering one:

> **Q:** Where does the deploy key live?
> **A:** *"The transfer key is a reference to the transfer key, which can be used
> in the command line. The transfer key will be the transfer key for..."*

Retrieval found the right passage; the model then ignored it and looped. That is
the expected majority case at this size, and no retriever setting fixes it —
#137's oracle arm proved the ceiling is the model, not the search.

**Use `hybrid_a4_step8000.pt` rather than `hybrid_chat_v1.pt` if retrieval
matters more to you than chat fluency.** The chat fine-tune measurably eroded
reading: near-window recall 35.5% → 28.3% (#108). The two are a trade, and the
chat checkpoint is on the wrong side of it for RAG.

## Known limits

| limit | why |
|---|---|
| **No streaming** | `"stream": true` returns a 400 rather than being silently ignored — a client expecting a stream and getting one blob looks like a hang. Continue handles the non-streaming path fine; replies appear all at once. |
| **One request at a time** | Single-threaded. It is a development bridge. |
| **No auth** | Hence localhost only. |
| **No function calling** | The model *can* emit tool-call format (#110), but this server does not implement OpenAI's `tools` API. |
| **Slower than it should be** | Every token re-forwards the whole prompt instead of using the incremental decode path (#97), so cost is O(N) per token rather than O(1) — throwing away the architecture's main advantage. Fixing it is gated on #138's ring-buffer work. |
| **No system-prompt role** | EdgeHVN was never trained with one. System messages are folded into the first user turn rather than given an invented marker, which would put the model off-distribution (the failure mode retracted in #122). |

## Troubleshooting

**`Could not refresh your Hugging Face access token ... CERTIFICATE_VERIFY_FAILED`**
— harmless. Avast's HTTPS-scanning root certificate is malformed by strict X.509
rules; the tokenizer is already cached locally. Documented in
[training.md](training.md) (#120).

**Replies repeat themselves.** Lower `temperature`, or raise
`--repetition-penalty` above the 1.3 default. This checkpoint loops badly without
one; the same value is used by the Docker serving path.

**Replies ignore the question.** Expected at this scale — see the table at the
top. If the answer is *available in a document*, use `--rag-corpus`; that is the
one case where this is fixable today.

## The other serving path

`web/` has a full Django + Celery + Channels chat app with a browser UI, verified
end-to-end against a real EdgeHVN checkpoint ([state.md](state.md)). Use that for
a chat product; use `serve_openai.py` when you want an editor or a script to talk
to the model without standing up postgres and redis.
