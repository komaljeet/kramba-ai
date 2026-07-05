from django.urls import path

from .views import ChatUIView

urlpatterns = [
    path("", ChatUIView.as_view(), name="chat-ui"),
]
