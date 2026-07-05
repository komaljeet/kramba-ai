#!/bin/sh
# Role dispatch for the single app image - see ISSUE_30_PLAN.md.
# Usage: docker run <image> [web|worker|train-worker|beat|migrate]
set -e

ROLE="${1:-web}"

case "$ROLE" in
  web)
    python manage.py migrate --noinput
    # collectstatic can fail on a Windows Docker bind mount: WhiteNoise's gzip
    # post-processing calls os.utime() on STATIC_ROOT, which the mounted
    # filesystem rejects ("Operation not permitted"). Don't let that abort
    # startup - with DEBUG=True Django's staticfiles app serves assets directly.
    python manage.py collectstatic --noinput \
      || echo "WARN: collectstatic failed (continuing; dev bind-mount utime issue?)" >&2
    exec daphne -b 0.0.0.0 -p 8000 web.asgi:application
    ;;
  worker)
    # Chat inference worker: consume the default `celery` queue ONLY, never the
    # `training` queue (Issue #57) - keeps GPU-saturating training runs off the
    # latency-sensitive chat path.
    exec celery -A web worker --loglevel=info -Q celery
    ;;
  train-worker)
    # Dedicated GPU training worker (Issue #57): consumes only the `training`
    # queue, one job at a time (concurrency 1), so a long curriculum run never
    # starves chat inference and can't fight another run for GPU memory.
    exec celery -A web worker --loglevel=info -Q training --concurrency 1
    ;;
  beat)
    exec celery -A web beat --loglevel=info
    ;;
  migrate)
    exec python manage.py migrate --noinput
    ;;
  *)
    echo "Unknown role: $ROLE (expected web|worker|train-worker|beat|migrate)" >&2
    exec "$@"
    ;;
esac
