import logging
import os

from rest_framework import generics, permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.throttling import UserRateThrottle
from rest_framework.views import APIView
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.tokens import RefreshToken

from django.conf import settings
from django.contrib.auth import get_user_model

logger = logging.getLogger(__name__)

from .models import Conversation, Message, ModelVersion
from .permissions import IsAdminOrReadOnly, IsConversationOwner
from .serializers import (
    ConversationSerializer,
    MessageCreateSerializer,
    MessageSerializer,
    ModelVersionSerializer,
    RegisterSerializer,
)
from .tasks import generate_response

User = get_user_model()


class GenerationRateThrottle(UserRateThrottle):
    # Separate scope from the general "user" rate (60/min) - generation is
    # GPU-bound and much more expensive per request than ordinary CRUD, so
    # it gets its own, stricter budget (see DEFAULT_THROTTLE_RATES["generation"]
    # in web/settings/base.py - 10/min, matching the issue's stated target).
    scope = "generation"


class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer
    permission_classes = [permissions.AllowAny]


class LogoutView(APIView):
    """Blacklists the given refresh token so it can no longer mint new
    access tokens (requires rest_framework_simplejwt.token_blacklist,
    already in INSTALLED_APPS)."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        refresh = request.data.get("refresh")
        if not refresh:
            return Response({"detail": "refresh token required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            RefreshToken(refresh).blacklist()
        except TokenError:
            return Response({"detail": "invalid token"}, status=status.HTTP_400_BAD_REQUEST)
        return Response(status=status.HTTP_205_RESET_CONTENT)


class ModelVersionViewSet(viewsets.ModelViewSet):
    """Registers a checkpoint already placed on the server's `checkpoints`
    volume (rsync/scp), by path - never accepts uploaded checkpoint bytes
    (see ISSUE_30_PLAN.md Phase 6). Any authenticated user can list/retrieve
    (needed to pick a model_version when creating a conversation); only
    staff can register/update/delete (IsAdminOrReadOnly)."""

    queryset = ModelVersion.objects.all()
    serializer_class = ModelVersionSerializer
    permission_classes = [permissions.IsAuthenticated, IsAdminOrReadOnly]

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)


class ConversationViewSet(viewsets.ModelViewSet):
    """Phase 1 scope: bare CRUD, enough to prove auth + per-user isolation.
    Phase 2 adds message history / fork / save-checkpoint endpoints on top."""

    serializer_class = ConversationSerializer
    permission_classes = [permissions.IsAuthenticated, IsConversationOwner]

    def get_queryset(self):
        # Scoping list/retrieve to the requesting user, not just the
        # object-permission check, so a user can't even discover another
        # user's conversation IDs exist (list would silently omit them,
        # retrieve on a guessed ID 404s instead of leaking a 403).
        return Conversation.objects.filter(owner=self.request.user)

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)

    def perform_destroy(self, instance):
        # Deleting the Conversation row cascades to its Messages (Postgres FK),
        # but the per-conversation Hamiltonian state file on the checkpoints
        # volume is NOT a DB row, so it would otherwise be orphaned until the
        # 90-day purge (which can no longer find it once the row is gone -
        # Issue #56). Remove it here so a user-deleted chat reclaims both DB
        # rows and disk/memory. Imported from the torch-free helper so the web
        # role never pulls the GPU stack in just to locate a file.
        from hvn_worker.state import state_path

        path = state_path(instance.id)
        super().perform_destroy(instance)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass  # no state file yet (conversation had no completed turn)
        except OSError:
            logger.warning("Failed to remove state file %s on conversation delete", path)

    def get_throttles(self):
        if self.action == "messages" and self.request.method == "POST":
            return [GenerationRateThrottle()]
        return super().get_throttles()

    @action(detail=True, methods=["get", "post"], url_path="messages")
    def messages(self, request, pk=None):
        conversation = self.get_object()

        if request.method == "POST":
            # Issue #30 "Max conversation length" requirement: enforce a hard
            # cap on turns rather than letting a conversation (and its
            # unbounded context-window-truncated-but-still-growing
            # history_ids list in the Hamiltonian state file) grow forever.
            if conversation.messages.count() >= settings.MAX_MESSAGES_PER_CONVERSATION:
                return Response(
                    {"detail": "This conversation has reached its message limit. Start a new conversation."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            serializer = MessageCreateSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            user_message = Message.objects.create(
                conversation=conversation,
                role=Message.Role.USER,
                content=serializer.validated_data["content"],
            )
            # Forward the per-message toggles (Issue #56) to the GPU worker.
            task = generate_response.delay(
                str(conversation.id),
                str(user_message.id),
                thinking=serializer.validated_data["thinking"],
                web_search=serializer.validated_data["web_search"],
            )
            return Response(
                {"message": MessageSerializer(user_message).data, "task_id": task.id},
                status=status.HTTP_202_ACCEPTED,
            )

        history = conversation.messages.all()
        return Response(MessageSerializer(history, many=True).data)
