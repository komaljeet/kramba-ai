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

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tag = models.SlugField(max_length=100, unique=True)
    checkpoint_path = models.CharField(max_length=500)
    checkpoint_sha256 = models.CharField(max_length=64, blank=True)
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
