import hashlib
import logging
import os

from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import get_channel_layer

from django.conf import settings
from django.utils import timezone

from .models import AuditLog, Conversation, Message
from .security import filter_profanity, sanitize_output

logger = logging.getLogger(__name__)


def conversation_group_name(conversation_id) -> str:
    return f"conversation_{conversation_id}"


@shared_task
def generate_response(
    conversation_id: str,
    user_message_id: str,
    thinking: bool = True,
    web_search: bool = False,
) -> str:
    """Runs one HVN chat turn for `conversation_id`, streaming each text
    delta to that conversation's Channels group as it's generated (consumed
    by chat.consumers.ChatConsumer over WebSocket - see ISSUE_30_PLAN.md
    Phase 4), then stores the finished reply as a Message.

    Takes the already-created user Message's id (not the raw text) rather
    than generating it here - the view records what the user actually said
    before scheduling this task, so message ordering in the DB stays correct
    even if the task sits in the queue for a while before running.
    """
    # Local import: torch/hvn are heavy and GPU-bound: keep them off the
    # web process's import path entirely, only the worker role needs them.
    from hvn_worker.inference import stream_turn

    conversation = Conversation.objects.select_related("model_version").get(id=conversation_id)
    user_message = Message.objects.get(id=user_message_id, conversation_id=conversation_id)

    channel_layer = get_channel_layer()
    group = conversation_group_name(conversation_id)

    result: dict = {}
    for delta in stream_turn(
        conversation_id=conversation.id,
        user_message=user_message.content,
        model_tag=conversation.model_version.tag,
        checkpoint_path=conversation.model_version.checkpoint_path,
        result=result,
        thinking=thinking,
        web_search=web_search,
    ):
        async_to_sync(channel_layer.group_send)(group, {"type": "chat.token", "delta": delta})

    # Sanitizes the finished, persisted message. The live per-delta stream
    # already sent over WebSocket above is NOT re-sanitized retroactively -
    # cleaning HTML tags that may be split across delta boundaries mid-stream
    # isn't reliably possible without risking corrupting legitimate text, and
    # the attacker doesn't control model output directly (only the prompt,
    # which is itself guarded by MessageCreateSerializer's injection check).
    # Accepted as a known limitation rather than a gap silently left in.
    clean_content = filter_profanity(sanitize_output(result["content"]))
    assistant_message = Message.objects.create(
        conversation=conversation, role=Message.Role.ASSISTANT, content=clean_content
    )

    async_to_sync(channel_layer.group_send)(
        group, {"type": "chat.done", "message_id": str(assistant_message.id)}
    )

    AuditLog.objects.create(
        user_id=conversation.owner_id,
        action="generate",
        path=f"/api/conversations/{conversation.id}/messages/",
        method="POST",
        status_code=201,
        prompt_hash=hashlib.sha256(user_message.content.encode()).hexdigest(),
        model_version=conversation.model_version,
    )

    return str(assistant_message.id)


@shared_task
def purge_expired_conversations() -> int:
    """Deletes conversations (and their messages, via the FK cascade) whose
    `updated_at` is older than settings.CONVERSATION_RETENTION_DAYS, plus
    their on-disk Hamiltonian state files - Issue #30 Phase 8.6, PII
    auto-purge, run daily via Celery beat (see CELERY_BEAT_SCHEDULE in
    web/settings/base.py). Returns the number of conversations purged.
    """
    # torch-free helper (same file location as inference._state_path) so the
    # beat/worker process running the purge doesn't load the GPU stack just to
    # delete files.
    from hvn_worker.state import state_path as _state_path

    cutoff = timezone.now() - timezone.timedelta(days=settings.CONVERSATION_RETENTION_DAYS)
    expired = Conversation.objects.filter(updated_at__lt=cutoff)

    count = 0
    for conversation in expired:
        state_path = _state_path(conversation.id)
        if os.path.exists(state_path):
            try:
                os.remove(state_path)
            except OSError:
                logger.warning("Failed to remove state file %s during purge", state_path)
        conversation.delete()
        count += 1

    return count
