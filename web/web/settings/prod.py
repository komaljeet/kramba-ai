from .base import *  # noqa: F401,F403
from .security import *  # noqa: F401,F403

DEBUG = False

if not ALLOWED_HOSTS:
    raise RuntimeError("DJANGO_ALLOWED_HOSTS must be set in production")
