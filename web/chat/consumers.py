"""
Streams a conversation's in-progress assistant reply to its owner over
WebSocket. Tokens are produced by api/tasks.py's Celery task (running on the
worker role, GPU-bound) and relayed here via a Channels group named
`conversation_<id>` - the task and this consumer never talk to each other
directly, only through the channel layer (Redis-backed - see
CHANNEL_LAYERS in web/settings/base.py).
"""
import json

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from api.models import Conversation


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        user = self.scope["user"]
        conversation_id = self.scope["url_route"]["kwargs"]["conversation_id"]

        if not user.is_authenticated:
            # chat/auth.py's JWTAuthMiddleware already resolved scope["user"];
            # missing/invalid token means AnonymousUser here.
            await self.close(code=4401)
            return

        owns = await self._user_owns_conversation(user.id, conversation_id)
        if not owns:
            # Same 404-not-403 reasoning as api/permissions.py's
            # IsConversationOwner: don't confirm another user's conversation
            # ID exists. WebSocket close codes can't carry that nuance the
            # way an HTTP status can, so this just refuses the connection.
            await self.close(code=4403)
            return

        self.group_name = f"conversation_{conversation_id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        if hasattr(self, "group_name"):
            await self.channel_layer.group_discard(self.group_name, self.channel_name)

    # --- Channels group event handlers (dispatched by `type` - see
    # api/tasks.py's group_send calls: "chat.token" -> chat_token, etc.) ---

    async def chat_token(self, event):
        await self.send(text_data=json.dumps({"type": "token", "delta": event["delta"]}))

    async def chat_done(self, event):
        await self.send(text_data=json.dumps({
            "type": "done",
            "message_id": event.get("message_id", ""),
            "stopped": event.get("stopped", False),
        }))

    async def chat_error(self, event):
        # Terminal frame for a failed generation (Issue #60) so the client
        # stops waiting on "Generating…" and can send again.
        await self.send(text_data=json.dumps({
            "type": "error",
            "detail": event.get("detail", "Generation failed."),
        }))

    @database_sync_to_async
    def _user_owns_conversation(self, user_id, conversation_id) -> bool:
        return Conversation.objects.filter(id=conversation_id, owner_id=user_id).exists()


class TrainingConsumer(AsyncWebsocketConsumer):
    """Streams an in-app training job's live progress to staff over WebSocket
    (Issue #57). Events are produced by api/tasks.py's run_training task
    (running on the dedicated GPU training worker) and relayed here via the
    Channels group `training_<job_id>` - same fan-out pattern as ChatConsumer,
    but staff-only (training is a privileged operation)."""

    async def connect(self):
        user = self.scope["user"]
        job_id = self.scope["url_route"]["kwargs"]["job_id"]

        if not user.is_authenticated:
            await self.close(code=4401)
            return
        # Training is staff-only across the board (TrainingJobViewSet uses
        # IsAdminUser) - mirror that on the socket rather than leaking progress.
        if not user.is_staff:
            await self.close(code=4403)
            return

        self.group_name = f"training_{job_id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        if hasattr(self, "group_name"):
            await self.channel_layer.group_discard(self.group_name, self.channel_name)

    # --- Channels group event handlers (dispatched by `type` - see
    # api/tasks.py run_training's stream() calls) ---

    async def training_status(self, event):
        await self.send(text_data=json.dumps({"type": "status", **_strip_type(event)}))

    async def training_progress(self, event):
        await self.send(text_data=json.dumps({"type": "progress", **_strip_type(event)}))

    async def training_done(self, event):
        await self.send(text_data=json.dumps({"type": "done", **_strip_type(event)}))


def _strip_type(event: dict) -> dict:
    # The Channels dispatch key `type` (e.g. "training.progress") is internal
    # routing - re-key the client-facing frame's own `type` in the handler.
    return {k: v for k, v in event.items() if k != "type"}
