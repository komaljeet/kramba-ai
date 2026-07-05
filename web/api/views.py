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
from django.core.cache import cache
from django.utils import timezone

logger = logging.getLogger(__name__)

from .models import Conversation, Message, ModelVersion, TrainingJob
from .permissions import IsAdminOrReadOnly, IsConversationOwner
from .serializers import (
    ConversationSerializer,
    MessageCreateSerializer,
    MessageSerializer,
    ModelVersionSerializer,
    RegisterSerializer,
    TrainingJobCreateSerializer,
    TrainingJobSerializer,
)
from .tasks import cancel_key, generate_response, run_training

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


class TrainingJobViewSet(viewsets.ModelViewSet):
    """Staff-only in-app training (Issue #57).

    POST /api/training/ validates a launch config and enqueues a Celery
    training task onto the dedicated GPU training queue (never the chat
    inference queue - see CELERY_TASK_ROUTES). Only one job may be active at a
    time so concurrent runs can't OOM the single GPU. GET list/retrieve exposes
    live status/progress (also streamed over WebSocket by TrainingConsumer);
    POST /api/training/{id}/cancel/ stops a queued or running job.
    """

    queryset = TrainingJob.objects.all()
    serializer_class = TrainingJobSerializer
    # Training saturates the GPU for hours and registers new model versions -
    # strictly staff, read included (unlike ModelVersion's IsAdminOrReadOnly,
    # which lets any user list models to pick one for a conversation).
    permission_classes = [permissions.IsAdminUser]
    http_method_names = ["get", "post", "head", "options"]  # no PUT/PATCH/DELETE

    def create(self, request, *args, **kwargs):
        # One job at a time: an hours-long GPU-saturating run must not stack
        # with another (OOM / GPU contention). Reject rather than queue a second.
        active = TrainingJob.objects.filter(status__in=TrainingJob.ACTIVE_STATUSES).first()
        if active is not None:
            return Response(
                {"detail": f"A training job is already {active.status} (id {active.id}). "
                           "Cancel or wait for it to finish before starting another."},
                status=status.HTTP_409_CONFLICT,
            )

        create_serializer = TrainingJobCreateSerializer(data=request.data)
        create_serializer.is_valid(raise_exception=True)

        job = TrainingJob.objects.create(
            created_by=request.user,
            config=create_serializer.to_config(),
            stages_total=len(create_serializer.validated_data["stages"]),
        )
        # Store the task id up front so cancel can revoke a still-queued job
        # (the task only overwrites it with the same id once it starts).
        task = run_training.delay(str(job.id))
        job.celery_task_id = task.id
        job.save(update_fields=["celery_task_id"])

        return Response(TrainingJobSerializer(job).data, status=status.HTTP_202_ACCEPTED)

    @action(detail=True, methods=["post"])
    def cancel(self, request, pk=None):
        job = self.get_object()
        if job.status not in TrainingJob.ACTIVE_STATUSES:
            return Response(
                {"detail": f"Job is {job.status}; nothing to cancel."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Revoke the queued/running task. terminate=False: never SIGKILL a
        # running stage mid-write - the task cooperates via its progress
        # callback (checks the CANCELLING status at each stage boundary and
        # stops after the current stage checkpoint is safely committed).
        run_training.AsyncResult(job.celery_task_id).revoke(terminate=False)

        if job.started_at is None:
            # Never started (still queued) - revoke prevents it from running, so
            # finalize immediately rather than leaving it stuck in CANCELLING.
            job.status = TrainingJob.Status.CANCELLED
            job.finished_at = timezone.now()
            job.detail = "Cancelled before start."
            job.save(update_fields=["status", "finished_at", "detail"])
        else:
            job.status = TrainingJob.Status.CANCELLING
            job.detail = "Cancelling after current stage…"
            job.save(update_fields=["status", "detail"])

        return Response(TrainingJobSerializer(job).data, status=status.HTTP_200_OK)


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

    @action(detail=True, methods=["post"])
    def stop(self, request, pk=None):
        # Issue #61: cooperatively stop an in-flight generation. Sets a shared
        # Redis flag the generate_response task polls between tokens; it closes
        # the generator at the next token and emits a terminal "done" frame.
        # No-op (still 202) if nothing is generating - the flag simply expires.
        conversation = self.get_object()  # enforces ownership
        cache.set(cancel_key(str(conversation.id)), "1", timeout=300)
        return Response(status=status.HTTP_202_ACCEPTED)

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
