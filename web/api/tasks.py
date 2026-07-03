import hashlib

from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import get_channel_layer

from .models import AuditLog, Conversation, Message
from .security import sanitize_output


def conversation_group_name(conversation_id) -> str:
    return f"conversation_{conversation_id}"


@shared_task
def generate_response(conversation_id: str, user_message_id: str) -> str:
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
    ):
        async_to_sync(channel_layer.group_send)(group, {"type": "chat.token", "delta": delta})

    # Sanitizes the finished, persisted message. The live per-delta stream
    # already sent over WebSocket above is NOT re-sanitized retroactively -
    # cleaning HTML tags that may be split across delta boundaries mid-stream
    # isn't reliably possible without risking corrupting legitimate text, and
    # the attacker doesn't control model output directly (only the prompt,
    # which is itself guarded by MessageCreateSerializer's injection check).
    # Accepted as a known limitation rather than a gap silently left in.
    assistant_message = Message.objects.create(
        conversation=conversation, role=Message.Role.ASSISTANT, content=sanitize_output(result["content"])
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
