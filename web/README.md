# HVN Chat App (Django + Channels + Celery)

WebSocket chat UI/API backed by a real HVN checkpoint. The Django `app-web`
role serves the REST API + WebSocket; the `app-worker` role (Celery) runs the
GPU/CPU inference against a checkpoint on the shared `checkpoints` volume,
persisting each conversation's Hamiltonian state to disk (O(1), no KV cache).
A separate `app-train-worker` role runs in-app training (Issue #57) on its own
Celery queue, so a GPU-saturating curriculum run never starves chat latency.

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

5. **Chat in the browser.** Open `http://127.0.0.1:8000/` — the built-in
   single-page chat UI (Issue #56, no separate build step). Log in, pick the
   registered model, create a conversation, and send a prompt; tokens stream in
   token-by-token over the WebSocket. Per-message **Thinking** and **Web search**
   toggles are sent with each prompt, and **deleting** a chat frees both its DB
   rows and its on-disk Hamiltonian state file.

   To drive the same endpoints by hand instead, create a conversation pinned to
   that model version, open the WebSocket, and post a prompt:

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
   is also stored as a `Message`. The bundled UI at `/` (step 5) drives exactly
   these endpoints; a separate SPA origin (e.g. the `CORS_ALLOWED_ORIGINS`
   default) could drive them too.

## In-app training (Issue #57)

A **staff** user can kick off a curriculum/validation run from the browser
instead of the CLI. Open `/`, log in as staff, and click **⚙ Start Training**
(the button is hidden for non-staff — the endpoints are staff-only). Pick a
registered model version to continue from, choose curriculum stages + budget,
and **Start**; per-stage progress (loss/stage/PPL) streams live over a
WebSocket, and on success the produced checkpoint is registered as a new
`ModelVersion` you can immediately pick for a new conversation.

Under the hood this reuses the CLI pieces:

- `POST /api/training/` (staff-only) validates the config and enqueues a Celery
  task onto the dedicated **`training`** queue — routed there by
  `CELERY_TASK_ROUTES` so it runs on `app-train-worker`, never the chat
  inference worker. **One job at a time** (a second request while one is active
  returns `409`) so concurrent runs can't OOM the single GPU.
- The task runs `auto_train.run_curriculum` (Issue #54): it continues from the
  chosen checkpoint, trains the requested stages in order, and writes each stage
  checkpoint under `HVN_CHECKPOINT_DIR/training_<job_id>/`. The model
  architecture (d_model, DeltaNet, …) is fixed by the base checkpoint — the run
  carries its weights forward.
- Progress streams over `ws/training/<job_id>/?token=<JWT>` as
  `{"type":"status|progress|done", ...}` frames (`TrainingConsumer`, staff-only).
- `POST /api/training/<id>/cancel/` stops a queued or running job — cooperatively
  for a running one (it stops after the current stage's checkpoint is committed).

> ⚠️ This box (RTX 4050/4060) is the "wind tunnel" (README §1.1): it validates
> scaling behavior, it **cannot** train a frontier model. The button is for
> validation/curriculum runs on a rented GPU box, not frontier training.

```bash
# Enqueue a smoke (tiny/offline) validation run continuing from a model version:
curl -X POST http://127.0.0.1:8000/api/training/ \
  -H "Authorization: Bearer $ADMIN_JWT" -H "Content-Type: application/json" \
  -d '{"from_model_version": "<model-version-uuid>",
       "stages": ["html", "css"], "epochs_per_stage": 1, "smoke": true}'
# Poll status:  GET /api/training/<job-id>/   |   Cancel:  POST /api/training/<job-id>/cancel/
```

## Tests

`web/settings/test.py` runs the whole suite locally with **no external
services** — sqlite in-memory DB, in-memory Channels layer, in-memory Celery
broker — while inheriting the real DRF/JWT/security config from `base.py`:

```bash
cd web
python manage.py test --settings=web.settings.test
```

The end-to-end chat smoke test (`tests/test_chat_smoke.py`) trains a tiny BPE +
Gated-DeltaNet checkpoint and drives the **real** serving path (`hvn_worker` →
`ChatSession.user_turn` / streaming), asserting a non-degenerate reply and a
working save→load state round-trip; it skips cleanly where torch isn't
installed. (Running against the compose Postgres/Redis instead is also fine —
just point `DJANGO_SETTINGS_MODULE` at `web.settings.dev` with a `.env`.)

`tests/test_training.py` covers the Issue #57 training flow — staff-only access,
config validation, the one-job-at-a-time guard, cancellation, and the task body
(progress streaming + `ModelVersion` registration) — with the heavy
`auto_train.run_curriculum` stubbed, so it needs neither a GPU nor torch.

> Note: these suites import the repo-root `hvn`/`eval`/`auto_train` packages, so
> run them with the repo root on `PYTHONPATH` (the Docker image sets
> `PYTHONPATH=/srv`; locally: `cd web && PYTHONPATH=.. python manage.py test
> --settings=web.settings.test`).
