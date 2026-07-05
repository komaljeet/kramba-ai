# HVN Chat App (Django + Channels + Celery)

WebSocket chat UI/API backed by a real HVN checkpoint. The Django `app-web`
role serves the REST API + WebSocket; the `app-worker` role (Celery) runs the
GPU/CPU inference against a checkpoint on the shared `checkpoints` volume,
persisting each conversation's Hamiltonian state to disk (O(1), no KV cache).

The served model is the **Issue #51 chat/base checkpoint** — a BPE +
Gated-DeltaNet HVN model. (Issue #52 wired it in: the worker rebuilds the model
with the checkpoint's `use_deltanet`/`weightless` flags, and per-conversation
state persistence handles the DeltaNet dict-shaped state.)

## Run it end-to-end (dev)

All commands are from the **repo root** (the compose build context is `..` so
the image can vendor `../hvn`).

1. **Build the #51 checkpoint** on the host (writes `hvn_chat_checkpoint.pt` +
   `hvn_chat_checkpoint_tokenizer.json`, both gitignored). docker-compose
   bind-mounts these into the worker at `/data/checkpoints/`:

   ```bash
   python build_chat_checkpoint.py --out hvn_chat_checkpoint.pt
   # (CI/loop-validation only, not a quality model:)
   # python build_chat_checkpoint.py --smoke --out hvn_chat_checkpoint.pt
   ```

2. **Configure env** — copy the example and fill in secrets:

   ```bash
   cp web/.env.example web/.env
   # edit DJANGO_SECRET_KEY, POSTGRES_PASSWORD, etc.
   ```

3. **Start the stack:**

   ```bash
   docker compose -f web/docker-compose.yml up --build
   ```

   `app-web` runs migrations + `collectstatic` then serves on
   `http://127.0.0.1:8000`. `app-worker` loads the checkpoint on first request.

4. **Register the checkpoint as the active model** (staff user required). Get a
   JWT via `POST /api/auth/login/`, then:

   ```bash
   curl -X POST http://127.0.0.1:8000/api/models/ \
     -H "Authorization: Bearer $ADMIN_JWT" -H "Content-Type: application/json" \
     -d '{"tag": "chat-v1",
          "checkpoint_path": "/data/checkpoints/hvn_chat_checkpoint.pt",
          "is_active": true}'
   ```

   Checkpoint bytes are never uploaded through the API — only a path inside
   `HVN_CHECKPOINT_DIR` that already exists on the volume is accepted.

5. **Chat.** Create a conversation pinned to that model version, open the
   WebSocket, and post a prompt:

   ```bash
   # Create a conversation (returns its UUID):
   curl -X POST http://127.0.0.1:8000/api/conversations/ \
     -H "Authorization: Bearer $JWT" -H "Content-Type: application/json" \
     -d '{"title": "hello", "model_version": "<model-version-uuid>"}'

   # Open ws://127.0.0.1:8000/ws/chat/<conversation-uuid>/?token=<JWT>
   # then POST a message to kick off generation:
   curl -X POST http://127.0.0.1:8000/api/conversations/<uuid>/messages/ \
     -H "Authorization: Bearer $JWT" -H "Content-Type: application/json" \
     -d '{"content": "Create a function that adds two numbers."}'
   ```

   Generated tokens stream over the WebSocket as `{"type":"token","delta":...}`
   messages, ending with `{"type":"done","message_id":...}`; the finished reply
   is also stored as a `Message`. A frontend at `http://localhost:3000` (the
   default `CORS_ALLOWED_ORIGINS`) can drive the same endpoints.

## Tests

The `web/tests/` suite runs under Django's test runner (Postgres + Redis, as in
`docker-compose.yml`). The end-to-end chat smoke test
(`tests/test_chat_smoke.py`) trains a tiny BPE + Gated-DeltaNet checkpoint and
drives the **real** serving path (`hvn_worker` → `ChatSession.user_turn` /
streaming), asserting a non-degenerate reply and a working save→load state
round-trip; it skips cleanly where torch isn't installed.

```bash
# inside the app image / a torch-enabled env with Postgres+Redis available:
python web/manage.py test
```
