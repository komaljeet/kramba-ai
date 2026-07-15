import uuid

from django.conf import settings
from django.db import models


class ModelVersion(models.Model):
    """A trained HVN checkpoint registered for serving.

    Bytes are never uploaded through this API (see ISSUE_30_PLAN.md Phase 6
    and the earlier rsync/scp discussion) - checkpoint_path points at a file
    already placed on the server's volume; this table only tracks metadata
    and which version is active. Conversations pin to one version at creation
    time so a checkpoint swap never changes an in-flight conversation's
    behavior underneath it.
    """

    class Architecture(models.TextChoices):
        # Original core model (hvn/core), BPE tokenizer, bracket-tag chat
        # protocol -- every ModelVersion registered before issue #97 Phase B
        # is this kind, hence the default below.
        HVN_CORE = "hvn_core"
        # EdgeHVN (hvn/edge/model.py), GPT-2 tokenizer, <|user|>/<|assistant|>
        # EOS-terminated chat protocol (issue #97 Phase B).
        EDGEHVN = "edgehvn"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tag = models.SlugField(max_length=100, unique=True)
    checkpoint_path = models.CharField(max_length=500)
    checkpoint_sha256 = models.CharField(max_length=64, blank=True)
    architecture = models.CharField(
        max_length=16, choices=Architecture.choices, default=Architecture.HVN_CORE,
    )
    is_active = models.BooleanField(default=False)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name="+"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.tag


class Conversation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="conversations"
    )
    title = models.CharField(max_length=200, blank=True)
    model_version = models.ForeignKey(
        ModelVersion, on_delete=models.PROTECT, related_name="conversations"
    )
    is_archived = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.title or self.id} ({self.owner})"


class Message(models.Model):
    class Role(models.TextChoices):
        SYSTEM = "system"
        USER = "user"
        ASSISTANT = "assistant"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=16, choices=Role.choices)
    content = models.TextField()
    token_count = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.role}: {self.content[:40]}"


class TrainingJob(models.Model):
    """An in-app training run (Issue #57): a staff user kicks off a
    curriculum/validation run on the dedicated GPU training worker, progress
    streams to the UI over Channels, and on success the resulting checkpoint is
    registered as a new ModelVersion.

    The job body reuses auto_train.run_curriculum (Issue #54) - this row only
    tracks lifecycle, the requested config, live progress, and the produced
    ModelVersion. It runs on a SEPARATE Celery queue/worker from chat inference
    (CELERY_TASK_ROUTES in settings) so an hours-long GPU-saturating run never
    starves chat latency, and only one job is allowed active at a time (guarded
    in the view) so concurrent runs can't OOM the single GPU.
    """

    class Status(models.TextChoices):
        QUEUED = "queued"
        RUNNING = "running"
        SUCCEEDED = "succeeded"
        FAILED = "failed"
        CANCELLING = "cancelling"
        CANCELLED = "cancelled"

    # Statuses in which the job still occupies the GPU worker / queue - used by
    # the view's one-job-at-a-time guard and to decide whether cancel applies.
    ACTIVE_STATUSES = (Status.QUEUED, Status.RUNNING, Status.CANCELLING)

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name="+"
    )
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    # The validated config the run was launched with (stages, epochs, precision,
    # d_model, smoke, from-checkpoint) - stored verbatim so a completed run is
    # self-describing and reproducible.
    config = models.JSONField(default=dict)
    celery_task_id = models.CharField(max_length=255, blank=True)

    # Live progress, updated by the task as it streams to the UI.
    current_stage = models.CharField(max_length=32, blank=True)
    stages_done = models.PositiveSmallIntegerField(default=0)
    stages_total = models.PositiveSmallIntegerField(default=0)
    # Last per-stage metrics row (ppl / bits_per_char / coding_score) from
    # auto_train, plus a short human-readable status line.
    last_metrics = models.JSONField(null=True, blank=True)
    detail = models.TextField(blank=True)
    error = models.TextField(blank=True)

    result_model_version = models.ForeignKey(
        ModelVersion, on_delete=models.SET_NULL, null=True, blank=True, related_name="+"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"TrainingJob {self.id} ({self.status})"


class AuditLog(models.Model):
    """One row per generation-relevant request. Populated by
    web.middleware.audit_log.AuditLogMiddleware (Phase 5) and/or directly by
    api/tasks.py when a generation completes.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name="+"
    )
    action = models.CharField(max_length=100)
    path = models.CharField(max_length=500)
    method = models.CharField(max_length=10)
    status_code = models.PositiveSmallIntegerField(null=True, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    prompt_hash = models.CharField(max_length=64, blank=True)
    model_version = models.ForeignKey(
        ModelVersion, on_delete=models.SET_NULL, null=True, blank=True, related_name="+"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.action} by {self.user} @ {self.created_at}"
