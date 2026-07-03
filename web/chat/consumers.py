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
        await self.send(text_data=json.dumps({"type": "done", "message_id": event["message_id"]}))

    @database_sync_to_async
    def _user_owns_conversation(self, user_id, conversation_id) -> bool:
        return Conversation.objects.filter(id=conversation_id, owner_id=user_id).exists()
