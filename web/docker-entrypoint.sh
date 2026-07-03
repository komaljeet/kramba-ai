#!/bin/sh
# Role dispatch for the single app image - see ISSUE_30_PLAN.md.
# Usage: docker run <image> [web|worker|beat|migrate]
set -e

ROLE="${1:-web}"

case "$ROLE" in
  web)
    python manage.py migrate --noinput
    python manage.py collectstatic --noinput
    exec daphne -b 0.0.0.0 -p 8000 web.asgi:application
    ;;
  worker)
    exec celery -A web worker --loglevel=info
    ;;
  beat)
    exec celery -A web beat --loglevel=info
    ;;
  migrate)
    exec python manage.py migrate --noinput
    ;;
  *)
    echo "Unknown role: $ROLE (expected web|worker|beat|migrate)" >&2
    exec "$@"
    ;;
esac
