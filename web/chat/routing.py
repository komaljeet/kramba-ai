from django.urls import path

from .consumers import ChatConsumer, TrainingConsumer

websocket_urlpatterns = [
    path("ws/chat/<uuid:conversation_id>/", ChatConsumer.as_asgi()),
    path("ws/training/<uuid:job_id>/", TrainingConsumer.as_asgi()),
]
