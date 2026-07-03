import os

from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.settings.dev")

# get_asgi_application() must run before importing anything that touches
# Django models/apps (chat.routing imports consumers, which import models).
django_asgi_app = get_asgi_application()

from chat.auth import JWTAuthMiddleware  # noqa: E402
from chat.routing import websocket_urlpatterns  # noqa: E402

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": JWTAuthMiddleware(URLRouter(websocket_urlpatterns)),
    }
)
