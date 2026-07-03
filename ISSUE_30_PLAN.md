# Plan: Issue #30 — Multi-User Web Interface (Django + DRF + Celery)

> Created: 2026-07-03
> GitLab: https://gitlab.com/komalbarun/kramba-ai/-/issues/30
> Status: ✅ Phases 0-8 done, MR !5 open (`issue-30-web-interface`, closes #30 on merge). Phase 8 (public-launch readiness) was added after the user confirmed intent to open public registration.
> Branch: `issue-30-web-interface` (pushed, MR !5 open — closes #30 on merge)

This plan exists so work can resume from a fresh chat session with full context.
See the issue body (`glab issue view 30 --repo komalbarun/kramba-ai`) for the
original requirements; this file is the execution plan derived from it plus
session decisions on Docker/async/deployment.

---

## Session Decisions (locked in, don't re-litigate)

1. **Docker for everything**, two compose files:
   - `docker-compose.yml` — local development (hot reload, exposed debug ports on `127.0.0.1` only)
   - `docker-prod.yml` — production
2. **Async is mandatory**: HVN generation takes 0.5–5+ seconds and must never block
   an HTTP worker. Two layers of async:
   - **Celery** for the actual generation task (already required by the issue)
   - **ASGI** (not WSGI) for the Django app itself, so HTTP + WebSocket (Channels)
     share one async-capable server process (Daphne or Uvicorn+Channels)
3. **"Single image" = one image for the app, not one image for the whole stack.**
   Django + DRF + Celery + Channels are all the same codebase → one Dockerfile,
   reused for multiple roles via command override:
   - `app` image → run as `web` (ASGI server), `worker` (Celery), `beat` (Celery beat, if needed)
   - Postgres and Redis stay as their own official images (separate software,
     stateful, need independent upgrade/backup lifecycles — bundling a DB into
     an app image is an anti-pattern, not what "single image" was asking for)
4. **Every Docker-published port is bound to `127.0.0.1` explicitly**
   (`127.0.0.1:8000:8000`, never bare `8000:8000` which defaults to `0.0.0.0`).
   This applies in both `docker-compose.yml` and `docker-prod.yml`, with **no
   exception** — including whatever serves HTTP/WebSocket to the outside world.
5. **Public exposure is out of Docker's hands.** A reverse proxy running on the
   *host* (existing nginx/Caddy, or a cloud load balancer) terminates TLS and
   forwards to the loopback-bound app port. Nothing in `docker-prod.yml` binds
   to `0.0.0.0`. This keeps Docker's attack surface at "reachable only from
   the host," matching the issue's "security is the deciding factor" framing.
   - Deploying this stack therefore requires a host-level proxy config
     (nginx/Caddy vhost forwarding `443 → 127.0.0.1:8000` and
     `wss:// → 127.0.0.1:8000/ws/`) — **not** created by this plan, since it's
     host/environment-specific, but documented as a deploy prerequisite.

---

## Architecture

```
                    ┌─────────────────────────────┐
  Internet ───────▶ │ Host-level reverse proxy     │   (nginx/Caddy, OUTSIDE docker-prod.yml,
                    │ TLS termination, 0.0.0.0:443 │    or wherever it already lives)
                    └──────────────┬──────────────┘
                                   │ forwards to 127.0.0.1 only
                                   ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │  docker-prod.yml (all ports 127.0.0.1-bound)                      │
   │                                                                     │
   │  ┌──────────────┐        ┌──────────────────┐      ┌────────────┐ │
   │  │ app (web)     │◀──────▶│ PostgreSQL 16     │      │ Redis 7    │ │
   │  │ ASGI: Daphne/ │        │ users, convs, msgs│      │ broker,    │ │
   │  │ Uvicorn+      │        │ audit log         │      │ cache,     │ │
   │  │ Channels      │        └──────────────────┘      │ pub/sub    │ │
   │  │ 127.0.0.1:8000│                                    └─────┬──────┘ │
   │  └──────┬───────┘                                          │        │
   │         │ submits task                                     │        │
   │         ▼                                                  │        │
   │  ┌──────────────┐   result via Redis pub/sub  ◀─────────────┘        │
   │  │ app (worker)  │────────────────────────────▶ Channels layer       │
   │  │ Celery, loads │   (consumed by `app (web)` process, pushed to    │
   │  │ HVN model     │    client over WebSocket)                        │
   │  │ once, GPU     │                                                  │
   │  └──────────────┘                                                  │
   └───────────────────────────────────────────────────────────────────┘
```

Same image (`app`) runs as both `web` and `worker` — only the container
command differs.

---

## Directory Layout (new, under `web/`)

```
web/
├── Dockerfile                      # single image: Django + DRF + Channels + Celery
├── docker-compose.yml              # dev: hot reload, debug ports, all on 127.0.0.1
├── docker-prod.yml                 # prod: gunicorn/daphne + celery worker + beat, 127.0.0.1 only
├── docker-entrypoint.sh            # dispatches on $ROLE: web | worker | beat | migrate
├── .env.example                    # DB/Redis creds, secret key, allowed hosts, JWT settings
├── manage.py
├── requirements.txt                # Django, DRF, Celery, Channels, channels-redis, daphne/uvicorn,
│                                    # djangorestframework-simplejwt, psycopg, django-ratelimit
├── web/                            # Django project package
│   ├── settings/
│   │   ├── base.py
│   │   ├── dev.py
│   │   ├── prod.py                 # SECURE_SSL_REDIRECT, HSTS, secure cookies, strict CORS
│   │   └── security.py             # CSP, CORS allowlist, rate-limit config
│   ├── urls.py
│   ├── asgi.py                     # ProtocolTypeRouter: http -> Django, websocket -> Channels
│   ├── celery.py
│   └── middleware/
│       ├── audit_log.py
│       └── rate_limit.py
├── api/                            # DRF app
│   ├── models.py                   # User(auth), Conversation, Message, AuditLog, ModelVersion
│   ├── serializers.py
│   ├── views.py                    # ViewSets: conversations, messages, models (admin), auth
│   ├── permissions.py              # IsConversationOwner, IsAdminOrReadOnly
│   ├── throttles.py
│   ├── urls.py
│   └── tasks.py                    # Celery: generate_response(conversation_id, message, model_version)
├── chat/                           # Channels app
│   ├── consumers.py                # WebSocket consumer, streams tokens from Redis pub/sub
│   ├── routing.py
│   └── auth.py                     # JWT auth on WS connect (query param or header)
├── hvn_worker/                     # bridges Celery <-> existing hvn/ package
│   ├── worker.py                   # loads HamiltonianVortexNetwork / ChatSession once per process
│   ├── inference.py                # wraps ChatSession.step(), yields tokens
│   └── state_manager.py            # Hamiltonian state cache, Redis-backed (StateCheckpoint from #28)
└── tests/
    ├── test_auth.py
    ├── test_security.py            # CSRF, XSS, SQLi, prompt-injection guardrails
    ├── test_conversations.py
    ├── test_generation.py
    └── test_websocket.py
```

This reuses the existing `hvn/` package as-is (in particular `hvn/chat.py`'s
`ChatSession`/`ConversationFormatter`/`StateCheckpoint` from #28, and
`hvn/server.py`'s streaming pattern from #31) — `hvn_worker/` is a thin adapter,
not a reimplementation.

---

## Phased Execution

### Phase 0: Scaffolding — ✅ Done (2026-07-03)
- `web/` Django project, `requirements.txt`, base/dev/prod settings split
- `Dockerfile` + `docker-entrypoint.sh` (role dispatch), `.env.example`
- `docker-compose.yml` (dev): app-web, app-worker, postgres, redis — all ports `127.0.0.1`-bound (confirmed via `docker port`, not just declared in YAML)
- Added a root `.dockerignore` that wasn't in the original plan — build context is repo root, so without it `.venv/` (multi-GB) would've been sent to the daemon
- Fixed a real Linux-build bug: root `requirements.txt` pins `triton-windows` (Windows-only); added `web/hvn-requirements-docker.txt` with the Linux-appropriate subset (numpy, tokenizers — torch/triton already ship in the `pytorch/pytorch` CUDA base image)
- Added dev bind mounts (`.:/srv/web`, `../hvn:/srv/hvn`) on `app-web`/`app-worker` for hot-reload and so generated migrations land on the host, not just inside the ephemeral container layer
- **Verified**: `docker compose build` succeeds for both roles; `docker compose up` — postgres/redis pass health checks, migrations apply clean (28 migrations), `curl 127.0.0.1:8000/api/health/` → `{"status": "ok"}`

### Phase 1: Auth & Data Model — ✅ Done (2026-07-03)
- `ModelVersion`, `Conversation`, `Message`, `AuditLog` models (`web/api/models.py`) + `web/api/migrations/0001_initial.py`; Django's built-in `User` used as-is
- JWT auth (`djangorestframework-simplejwt`): `POST /api/auth/register/`, `/auth/login/`, `/auth/refresh/`, `/auth/logout/` (blacklists the refresh token)
- Password policy enforced via Django's `AUTH_PASSWORD_VALIDATORS` (min 12 chars + complexity, wired into `RegisterSerializer`)
- Per-user isolation: `ConversationViewSet.get_queryset()` scopes to `request.user` (so a non-owner's request 404s rather than 403s — doesn't confirm another user's conversation ID exists) plus `IsConversationOwner` object-permission as defense in depth
- Minimal `ConversationViewSet` (list/create/retrieve/update/delete) — enough to prove isolation; Phase 2 adds message history and generation
- `web/api/admin.py` registered for all four models (not originally scoped to Phase 1, but essentially free)
- **Verified**: `web/tests/test_auth.py`, 11/11 pass in the `app-web` container against real postgres — register (incl. weak-password rejection), login (incl. wrong password), refresh, logout/blacklist, owner access, cross-user 404, list-excludes-others, unauthenticated 401
- **Bug caught and fixed during verification**: DRF's default throttle classes are cache-backed, and Django's cache isn't part of a `TestCase`'s per-test transaction rollback — anonymous request counts accumulated across test methods in the same run and later tests got 429 instead of the status under test. Fixed with a `ThrottleResetMixin` that clears the cache in `setUp`. Worth remembering for every future test module that hits an endpoint with `AnonRateThrottle`/`UserRateThrottle`.

### Phase 2: Conversation CRUD — absorbed into Phase 1
`ConversationViewSet` (list/create/retrieve/update/delete, owner-scoped) was
already needed in Phase 1 to prove per-user isolation, so there was no
separate Phase 2 left to do. `IsConversationOwner` permission done.
Formal `test_conversations.py` (dedicated CRUD edge cases beyond isolation)
still outstanding but low priority — isolation is already covered by
`test_auth.py`.

### Phase 3: Async Generation (Celery) — ✅ Done (2026-07-03)
- `web/hvn_worker/worker.py` — lazily loads an HVN checkpoint once per Celery
  worker process, cached by `ModelVersion.tag` (thread-safe double-checked
  lock). Reuses `train.py`'s `CharTokenizer` (vendored into the image
  alongside `hvn/` — see Dockerfile) rather than duplicating a tokenizer class.
- `web/hvn_worker/inference.py` — `run_turn()` wraps `hvn.chat.ChatSession`:
  loads the conversation's persisted Hamiltonian state from the shared
  `checkpoints` volume (`ChatSession.load`/`.save`, JSON), runs one user turn,
  saves state back. Disk (not in-memory) is the source of truth, so it's
  correct even if a different worker process picks up the next turn.
- `web/api/tasks.py` — `generate_response` Celery task: takes the already-created
  user `Message` id (not raw text, so DB ordering stays correct even if the
  task queues for a while), runs `hvn_worker.inference.run_turn`, creates the
  assistant `Message`, writes an `AuditLog` row (prompt hash, not raw prompt).
- `POST /api/conversations/{id}/messages/` (added as a `ConversationViewSet`
  `@action`) creates the user message, enqueues the task, returns 202 +
  `task_id` immediately. `GET` on the same path returns message history.
  A separate `GenerationRateThrottle` (scope `"generation"`, 10/min) applies
  only to the POST — the general `UserRateThrottle` (60/min) would be too
  loose for a GPU-bound endpoint, matching the issue's stated target.
- `MessageCreateSerializer` enforces the issue's "max 10K chars per user
  message" requirement.
- `web/hvn-requirements-docker.txt` used for the worker's Python deps (see
  Phase 0) — `train.py` also vendored into the image (`COPY train.py
  /srv/train.py`, `PYTHONPATH=/srv`) purely so `CharTokenizer` can be reused
  without duplicating it.
- **Verified end-to-end against the real GPU and the real bundled
  `hvn_checkpoint.pt`** (registered as `ModelVersion` tag `dev-v1`, mounted
  into the `checkpoints` volume the same way a production rsync/scp deploy
  would land it): login → create conversation → `POST .../messages/` (202 +
  task_id) → Celery log shows the task received and succeeded in ~6.9s on
  GPU → `GET .../messages/` shows both the user message and a persisted
  assistant reply. Output text is low-quality gibberish, as expected — that's
  the small char-level checkpoint from early training runs, not a bug in this
  pipeline; per the earlier "build infra now, train later" decision, a better
  checkpoint swaps in later via the `ModelVersion` registry with no code change.
- **Two real bugs caught and fixed during verification**:
  1. `web/web/celery.py` was missing `django.setup()`. The standalone
     `celery -A web worker` process never goes through `manage.py`/`asgi.py`,
     so nothing else initializes the Django app registry for it — any task
     touching the ORM would have failed on import in a real deployment (Phase
     0's verification never exercised Celery, so this wasn't caught until now).
  2. The `checkpoints` named volume is created root-owned by Docker on first
     use, but the container runs as non-root `appuser` (uid 10001) — writes
     to it (`conversations/<id>.json` state files) failed with
     `PermissionError`. Fixed by creating and chown'ing `/data/checkpoints` in
     the Dockerfile *before* the `USER appuser` switch, so a fresh volume
     inherits the right ownership when Docker auto-populates it. (The volume
     already created during this session's testing needed a one-off manual
     `chown` via a throwaway container since Docker doesn't retroactively fix
     ownership on existing volumes.)
- Also added, not originally itemized in this plan: dev bind mounts for the
  bundled `hvn_checkpoint.pt`/`hvn_checkpoint_tokenizer.json` onto the
  `checkpoints` volume in `docker-compose.yml`, as a local stand-in for the
  rsync/scp-to-server workflow discussed earlier in the session.

### Phase 4: WebSocket Streaming (Channels) — ✅ Done (2026-07-03)
- **New in `hvn/chat.py`** (not web-project-specific — genuinely reusable core
  functionality, same reasoning as `hvn/server.py`'s `stream_text_chunks` for
  Issue #31's SSE): `ChatSession.user_turn_streaming()`, a generator variant of
  `user_turn()` that yields incremental text deltas token-by-token (same
  incremental-decode-suffix technique as `stream_text_chunks`, avoiding
  malformed partial-BPE-token output) and updates `conv_state`/`turns`
  exactly like `user_turn()` once exhausted.
- `web/hvn_worker/inference.py` — `stream_turn()`: generator wrapping
  `user_turn_streaming()` with the same load/save-state bracketing as
  `run_turn()`. Takes a `result: dict` out-param for the final cleaned text,
  since a plain `for delta in gen():` loop can't see a generator's `return`
  value.
- `web/api/tasks.py` — `generate_response` now streams: for each delta it
  `group_send`s `{"type": "chat.token", "delta": ...}` to the Channels group
  `conversation_<id>` (Redis-backed channel layer, already configured in
  Phase 0), then a final `chat.done` event after the assistant `Message` is saved.
- `web/chat/consumers.py` — `ChatConsumer`: on connect, rejects unauthenticated
  (`close(4401)`) and non-owners (`close(4403)`, same 404-style "don't confirm
  the conversation exists" reasoning as `IsConversationOwner`) via a
  `database_sync_to_async` ownership check; otherwise joins the conversation's
  group and relays `chat.token`/`chat.done` events to the client as JSON.
- `web/chat/routing.py` — `ws/chat/<uuid:conversation_id>/` → `ChatConsumer`.
- `web/chat/auth.py` (already built in Phase 0) — JWT via `?token=` query param.
- **Verified**: `web/tests/test_websocket.py`, 3/3 pass (`TransactionTestCase`,
  required because the consumer's `database_sync_to_async` ownership check
  runs on a different thread than a plain `TestCase`'s transaction) —
  unauthenticated rejected, non-owner rejected, owner receives relayed
  `chat.token`/`chat.done` events in order.
- **Also verified manually against real GPU generation** (not just the mocked
  `group_send` in the unit test): a WebSocket client connected, a `POST
  .../messages/` was fired concurrently, and the client received the actual
  per-character stream as the model generated it — confirmed the concatenated
  stream exactly matches the `Message.content` persisted to the DB afterward.
- 14/14 total tests passing across `test_auth.py` + `test_websocket.py`.

### Phase 5: Security Hardening Pass — ✅ Done (2026-07-03)
- **Prompt-injection guard** — `web/api/security.py`'s `contains_injection_markers()`
  rejects (400, via `MessageCreateSerializer.validate_content`) any user
  message containing HVN's literal `[SYSTEM]`/`[USER]`/`[ASSISTANT]` (and
  closing) tag strings. This is HVN-specific, not a generic keyword denylist:
  `ConversationFormatter` (hvn/chat.py) delimits conversation turns with these
  literal strings in the raw text fed to the tokenizer, so without this guard
  a user message could forge fake system/assistant turns the model would see
  as real conversation structure.
- **Output sanitization** — `api/security.py`'s `sanitize_output()` (via
  `bleach.clean(..., strip=True)`) strips all HTML/script tags from the
  finished assistant reply before it's persisted (`api/tasks.py`). Known,
  accepted limitation: the live per-delta WebSocket stream is sent
  unsanitized as it's generated (HTML tags can split across delta
  boundaries, so streaming-safe sanitization isn't reliably possible) —
  only the final persisted `Message.content` is guaranteed clean. Low risk
  in practice since the attacker doesn't control model output directly, only
  the prompt (which the injection guard above already constrains).
- **Max conversation length** — `settings.MAX_MESSAGES_PER_CONVERSATION`
  (default 500, `api/views.py`) hard-caps turns per conversation; exceeding
  it returns 400 rather than letting a conversation's Hamiltonian state file
  grow unbounded.
- **Rate limiting** — kept the DRF throttling from Phase 3 (`UserRateThrottle`
  60/min general, `AnonRateThrottle` 10/min, `GenerationRateThrottle` 10/min)
  and deliberately did **not** add `django-ratelimit` on top: DRF's
  `AnonRateThrottle` already covers per-IP for unauthenticated requests
  (login/register), and `UserRateThrottle` covers per-user for everything
  else — two overlapping rate-limit systems would add complexity without
  covering a gap the first doesn't already close. Removed the
  `django-ratelimit` line that had been added to `requirements.txt` in Phase
  0 but never used.
- **CSP headers** — `web/web/middleware/security_headers.py`'s
  `ContentSecurityPolicyMiddleware` (hand-rolled, one header, not worth a
  dependency) sets a `default-src 'self'` policy; matters mainly for the
  Django admin and DRF's browsable API since the rest of this service
  returns JSON.
- **Request body size** — `DATA_UPLOAD_MAX_MEMORY_SIZE` (1MB) moved from
  `security.py` (prod-only) to `base.py` — this is a correctness/security
  property that should hold in dev and tests too, not just production.
- HSTS, secure cookies, session timeout, CORS strict allowlist: already done
  in Phase 0's `web/web/settings/security.py`, unchanged here.
- **Verified**: `web/tests/test_security.py`, 9/9 pass — oversized message
  (>10K chars) rejected, exact-limit message accepted, prompt-injection
  marker rejected, SQL-injection-shaped content stored as inert literal text
  (proves ORM parametrization, not string interpolation), conversation
  message-limit enforcement (via `override_settings`), CSP header present,
  oversized request body (2MB) rejected before JSON parsing, JWT-authenticated
  POST works without a CSRF token (JWT travels via an explicit header, not an
  ambient cookie, so it isn't CSRF-vulnerable the way session auth is), plus
  2 unit tests directly on `sanitize_output`/`contains_injection_markers`.
- 24/24 total tests passing across all three test modules.

### Phase 6: Model Registry & Admin — ✅ Done (2026-07-03)
- `ModelVersion` model (path, hash, tag, active flag) — already existed from
  Phase 1; this phase added the actual API surface around it.
- `api/permissions.py` — `IsAdminOrReadOnly` (any authenticated user can
  GET/list, only `is_staff` can write) — matches the issue body's own
  proposed permission name.
- `api/serializers.py` — `ModelVersionSerializer.validate_checkpoint_path`
  enforces two things beyond "does this look like a path": the file must
  actually exist (`os.path.isfile`) and must resolve to somewhere *under*
  `settings.HVN_CHECKPOINT_DIR` (`os.path.commonpath` check) — rejects both a
  typo'd path and a path-traversal attempt like registering `/etc/passwd` as
  a "checkpoint". Still never accepts uploaded bytes — only a path to
  something already placed on the server's volume via rsync/scp.
- `GET/POST /api/models/` wired into `api/urls.py` via `ModelVersionViewSet`.
- Django admin panel for `ModelVersion`/`AuditLog` — already done in Phase 1.
- **Verified**: `web/tests/test_model_registry.py`, 5/5 pass — regular user
  can list but not create (403), admin can register a real file under
  `HVN_CHECKPOINT_DIR` (201, `created_by` set), registration rejects both a
  nonexistent path and a path outside the checkpoint directory (400 each).
- 29/29 total tests passing across all four test modules.

### Phase 7: Production Compose + Deploy Hardening — ✅ Done (2026-07-03)
- `web/docker-prod.yml`: `app-web`, `app-worker` (same image as dev, `web`
  vs `worker` command), `postgres`, `redis` — every port `127.0.0.1`-bound,
  no dev bind-mounts (image is self-contained; code changes mean a new image,
  not a live-edited volume), given a distinct Compose project name
  (`name: kramba-prod`) so it can never collide with the dev stack if both
  happen to be up on the same host. Documented (in-file comment) that a real
  deployment should build in CI and swap `build:` for `image:
  registry/...:tag` rather than building on the prod host itself.
- `whitenoise` added (`web/requirements.txt`, `MIDDLEWARE`,
  `STORAGES["staticfiles"]` in `base.py`) so the app serves its own collected
  static files (admin/DRF-browsable-API assets) directly — no separate
  nginx/static server needed, keeping with "only a host-level proxy is
  internet-facing."
- `prod.py`/`security.py` (`SECURE_SSL_REDIRECT`, HSTS, secure cookies,
  strict CORS) already existed from Phase 0 — unchanged here, just exercised
  for real for the first time.
- **Verified end-to-end**: built and booted `docker-prod.yml` for real
  (distinct `kramba-prod-*` container names confirmed no collision with the
  dev stack) —
  - migrations apply clean, whitenoise post-processes 772 static files, Daphne listens
  - plain HTTP request → **301 redirect to `https://`** (proves
    `SECURE_SSL_REDIRECT` is actually active, not just set)
  - same request with `X-Forwarded-Proto: https` (simulating the real
    reverse proxy) → `200 {"status": "ok"}` — confirms
    `SECURE_PROXY_SSL_HEADER` trust chain works end to end
  - response headers include CSP, HSTS (`max-age=31536000; includeSubDomains;
    preload`), `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`
  - a 404 renders Django's plain production error page, not the debug
    traceback page — confirms `DEBUG=False` is actually in effect, not just
    configured
  - `docker port kramba-prod-app-web-1` → `127.0.0.1:8000` only

---

## Phase 8: Public-Launch Readiness (added 2026-07-04) — ✅ Done

Triggered by confirming the deployment intent is public registration, not
just private/team use — at that point several items left unchecked on the
issue stop being optional. Not a full rewrite of the earlier "not needed"
judgment call, just the subset that only matters once strangers can sign up:
content filtering, GDPR export/deletion, and PII auto-purge become real
requirements; 2FA and breach-checking were already recommended regardless of
scale (admin-account compromise is a risk either way).

Explicitly still out of scope even for public launch (infra-level, not
app-level — see the earlier discussion): DDoS protection beyond request-size
limits (belongs at the reverse-proxy/CDN layer), and DB/Redis
encryption-at-rest (revisit once there's real user data at meaningful scale).

- **8.1 2FA/TOTP for admin accounts** — `django-otp` (`TOTPDevice`).
  `POST /api/auth/2fa/setup/` (staff only) creates/returns an unconfirmed
  device's provisioning URI; `POST /api/auth/2fa/confirm/` activates it after
  the admin proves they can generate a valid code. Login (`TokenObtainPairView`
  replaced with a custom view) requires a valid `otp_token` in the same
  request for any user with a confirmed device — staff accounts in practice,
  since only admins can create one via the setup endpoint.
- **8.2 Password breach checking** — HaveIBeenPwned's k-anonymity range API
  (only a SHA-1 prefix ever leaves the server, not the password) via a custom
  `AUTH_PASSWORD_VALIDATORS` entry. Stdlib `urllib.request`, not a new HTTP
  client dependency. Fails open (allows registration, logs a warning) on
  network error — a third-party API outage must not be able to block
  registration entirely.
- **8.3 Content filtering on model output** — `better-profanity` (lightweight,
  pure Python, no ML dependency) applied alongside `sanitize_output()` in
  `api/tasks.py`. Given HVN's actual output quality (often letter-salad, not
  real words — see `docs/TRAINING_GUIDE.md`'s honest-assessment section), a
  word-list filter will rarely trigger today, but it's the requested baseline
  and is swappable for a real classifier later without changing the call site.
- **8.4 GDPR data export** — `GET /api/account/export/` returns the
  requesting user's account + all conversations + all messages as JSON.
- **8.5 GDPR data deletion** — `DELETE /api/account/` deletes the account
  (cascades to conversations/messages via the existing FK). Requires
  re-entering the password in the request body — irreversible action, and a
  15-minute-lived access token alone shouldn't be sufcient to trigger it.
- **8.6 PII auto-purge job** — Celery beat periodic task, static
  `CELERY_BEAT_SCHEDULE` (no `django-celery-beat` needed for one fixed daily
  job), deletes conversations whose `updated_at` is older than
  `CONVERSATION_RETENTION_DAYS`. New `app-beat` service in both compose files
  (`docker-entrypoint.sh` already supports a `beat` role from Phase 0/7, just
  unused until now).

**Verified**: `web/tests/test_2fa.py`, `test_account.py`, `test_purge.py`,
plus additions to `test_security.py` — 22 new tests, 51/51 total passing.
Notable cases: unconfirmed TOTP device does NOT gate login (only a confirmed
one does — otherwise calling setup without following through would lock an
account out); breach validator fails open on a mocked network error;
purge task removes both the DB row and its on-disk Hamiltonian state file,
and leaves conversations within the retention window untouched; deleted
account's still-live access token is rejected on the very next request
(no separate blacklisting needed — JWT auth re-fetches the user from the DB
every request). Also manually booted `app-beat` in the dev stack and
confirmed it picks up the `purge-expired-conversations` schedule cleanly.

---

## Explicit Non-Goals (deferred per the issue itself)

- Full agentic mode (#25, still open) — API should be structured so it slots in later
- Model file upload-over-HTTP — deliberately excluded (see Phase 6); large
  checkpoints move via rsync/scp per the earlier session discussion
- Multi-model hot-swapping within a single running worker — `ModelVersion` is
  selectable per conversation, but a worker process loads one model at a time
  (same scope note as #31's server)

---

## Open Questions to Confirm Before/During Implementation

- Which HVN checkpoint backs the first deployed `ModelVersion`? (Can start with
  the existing small bundled one per the earlier "build infra now, train later" decision.)
- Retention/purge policy specifics for conversation data (issue lists it as a
  requirement but doesn't specify a duration) — needs a number before Phase 5's
  auto-purge is implemented.
- Whether 2FA/TOTP for admin accounts (listed in the issue) is in scope for v1
  or a fast-follow — affects Phase 1 scope/timeline.
