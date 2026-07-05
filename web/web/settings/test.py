"""Test settings: the real dev/base config (DRF auth, JWT, security guards,
middleware — all inherited) but with **no external services**, so the suite
runs locally with just `python manage.py test`, no Docker/Postgres/Redis.

  - sqlite in-memory DB instead of Postgres
  - in-memory Channels layer instead of channels_redis
  - local-memory cache instead of Redis
  - eager Celery (`generate_response.delay(...)` runs inline in the caller)

Placeholder secrets/DB env are set here so importing `.base` (which reads them
via os.environ[...]) doesn't require a .env file.

    python manage.py test --settings=web.settings.test
"""
import os

os.environ.setdefault("DJANGO_SECRET_KEY", "test-" + "x" * 50)
os.environ.setdefault("POSTGRES_DB", "test")
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("DJANGO_ALLOWED_HOSTS", "testserver,localhost,127.0.0.1")

from .base import *  # noqa: F401,F403

DEBUG = False

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

# No Redis in the test environment.
CHANNEL_LAYERS = {"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
CACHES = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}

# In-memory Celery broker so `generate_response.delay(...)` enqueues without a
# running Redis/worker (the task is never consumed, matching the real app where
# the web process only dispatches — the worker process runs inference). NOT
# eager: eager would run real GPU inference inline against test fixtures.
CELERY_BROKER_URL = "memory://"
CELERY_RESULT_BACKEND = "cache+memory://"
CELERY_TASK_ALWAYS_EAGER = False
