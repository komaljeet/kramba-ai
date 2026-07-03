"""
Production-only hardening. Imported by prod.py after base.py so these values
take precedence. Kept in a separate file so the security posture is auditable
in one place rather than scattered through prod.py.
"""
import os

# --- Transport security ---
# The container itself is only ever reachable via 127.0.0.1 (see docker-prod.yml);
# a host-level reverse proxy terminates TLS and forwards here. SECURE_PROXY_SSL_HEADER
# tells Django to trust that proxy's signal that the original request was HTTPS.
SECURE_SSL_REDIRECT = True
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

# --- Cookies ---
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_AGE = 30 * 60  # 30 min idle timeout

# --- Clickjacking / MIME sniffing ---
X_FRAME_OPTIONS = "DENY"
SECURE_CONTENT_TYPE_NOSNIFF = True

# --- CORS: strict allowlist, never a wildcard ---
CORS_ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()
]
CORS_ALLOW_CREDENTIALS = True

# Note: DATA_UPLOAD_MAX_MEMORY_SIZE and MAX_MESSAGES_PER_CONVERSATION live in
# base.py, not here - they're security properties that should hold in dev
# too, not just production hardening.
