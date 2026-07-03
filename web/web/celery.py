import os

import django
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.settings.dev")
# The standalone `celery -A web worker` process never goes through
# manage.py/asgi.py, so nothing else calls django.setup() for it - without
# this, task modules that touch the ORM (api/tasks.py) fail on import.
django.setup()

app = Celery("web")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
