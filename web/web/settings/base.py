"""
Shared settings. Never import this directly — use dev.py or prod.py, which
both start with `from .base import *`.
"""
import os
from datetime import timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_KEY = os.environ["DJANGO_SECRET_KEY"]

ALLOWED_HOSTS = [h.strip() for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "").split(",") if h.strip()]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "rest_framework_simplejwt.token_blacklist",
    "corsheaders",
    "channels",
    "django_otp",
    "django_otp.plugins.otp_totp",
    "api",
    "chat",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    # Serves collected static files (admin CSS/JS, DRF browsable API assets)
    # directly from the app process - no separate nginx/static-file server
    # needed, which matters here because the host-level reverse proxy is the
    # only thing allowed to be internet-facing (see docker-prod.yml).
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "web.middleware.security_headers.ContentSecurityPolicyMiddleware",
    "web.middleware.audit_log.AuditLogMiddleware",
]

ROOT_URLCONF = "web.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

ASGI_APPLICATION = "web.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ["POSTGRES_DB"],
        "USER": os.environ["POSTGRES_USER"],
        "PASSWORD": os.environ["POSTGRES_PASSWORD"],
        "HOST": os.environ.get("POSTGRES_HOST", "postgres"),
        "PORT": os.environ.get("POSTGRES_PORT", "5432"),
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
        "OPTIONS": {"min_length": 12},
    },
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
    {"NAME": "web.validators.BreachedPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage"},
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Redis (Celery broker + Channels layer + cache) ---
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {"hosts": [REDIS_URL]},
    }
}

CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TASK_TRACK_STARTED = True
# HVN generation is GPU-bound and slow (0.5-5s+) - one task at a time per
# worker process avoids fighting over the same model/GPU memory.
CELERY_WORKER_CONCURRENCY = int(os.environ.get("CELERY_WORKER_CONCURRENCY", "1"))
CELERY_WORKER_PREFETCH_MULTIPLIER = 1

# Route the in-app training task (Issue #57) onto its own `training` queue so it
# runs on a SEPARATE worker (docker role `train-worker`, `celery -A web worker
# -Q training`) from chat inference. An hours-long GPU-saturating curriculum run
# must never sit in the same queue as chat turns and starve their latency. Chat
# inference stays on the default `celery` queue; the inference worker consumes
# `-Q celery` only, so it never picks up a training job.
CELERY_TASK_ROUTES = {
    "api.tasks.run_training": {"queue": "training"},
}

# One fixed daily job - a static schedule is enough, no need for
# django-celery-beat's DB-backed dynamic scheduling for a single task.
CELERY_BEAT_SCHEDULE = {
    "purge-expired-conversations": {
        "task": "api.tasks.purge_expired_conversations",
        "schedule": timedelta(hours=24),
    },
}

# --- DRF ---
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ),
    "DEFAULT_PERMISSION_CLASSES": (
        "rest_framework.permissions.IsAuthenticated",
    ),
    "DEFAULT_THROTTLE_CLASSES": (
        "rest_framework.throttling.UserRateThrottle",
        "rest_framework.throttling.AnonRateThrottle",
    ),
    "DEFAULT_THROTTLE_RATES": {
        "user": "60/min",
        "anon": "10/min",
        "generation": "10/min",
    },
}

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=15),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
    "UPDATE_LAST_LOGIN": True,
}

# --- Model checkpoints (never uploaded over HTTP - see ISSUE_30_PLAN.md Phase 6) ---
HVN_CHECKPOINT_DIR = os.environ.get("HVN_CHECKPOINT_DIR", "/data/checkpoints")

# --- Conversation retention (Phase 5 - value TBD, see ISSUE_30_PLAN.md open questions) ---
CONVERSATION_RETENTION_DAYS = int(os.environ.get("CONVERSATION_RETENTION_DAYS", "90"))

# --- Security (Issue #30 "Input/Output Security" checklist) ---
# Applies in dev too, not just prod - a request-body-size guard against
# memory-exhaustion is a correctness/security property, not an
# environment-specific hardening step. Model checkpoints deliberately never
# go through this endpoint (see Phase 6), so this can stay small.
DATA_UPLOAD_MAX_MEMORY_SIZE = 1 * 1024 * 1024  # 1 MB

# "Max conversation length" requirement: hard cap on turns per conversation
# (see api/views.py ConversationViewSet.messages).
MAX_MESSAGES_PER_CONVERSATION = int(os.environ.get("MAX_MESSAGES_PER_CONVERSATION", "500"))
