from .base import *  # noqa: F401,F403

DEBUG = True

ALLOWED_HOSTS = ALLOWED_HOSTS or ["localhost", "127.0.0.1"]

CORS_ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()
]

# Relaxed only for local development - prod.py overrides all of these.
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False
SECURE_SSL_REDIRECT = False
