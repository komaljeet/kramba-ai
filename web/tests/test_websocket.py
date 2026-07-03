import json

from channels.layers import get_channel_layer
from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator

from django.contrib.auth import get_user_model
from django.test import TransactionTestCase
from rest_framework_simplejwt.tokens import RefreshToken

from api.models import Conversation, ModelVersion
from api.tasks import conversation_group_name
from chat.auth import JWTAuthMiddleware
from chat.routing import websocket_urlpatterns

User = get_user_model()
STRONG_PASSWORD = "correct-horse-battery-staple-9"


class ChatConsumerTests(TransactionTestCase):
    """Uses TransactionTestCase (not TestCase) because the consumer's
    database_sync_to_async ownership check runs on a separate thread, which
    doesn't see TestCase's per-test transaction - see Channels' own testing
    docs on this exact gotcha."""

    def setUp(self):
        self.model_version = ModelVersion.objects.create(
            tag="ws-test-v1", checkpoint_path="/dev/null"
        )
        self.owner = User.objects.create_user(username="ws_owner", password=STRONG_PASSWORD)
        self.other = User.objects.create_user(username="ws_other", password=STRONG_PASSWORD)
        self.conversation = Conversation.objects.create(
            owner=self.owner, title="ws test", model_version=self.model_version
        )
        self.application = JWTAuthMiddleware(URLRouter(websocket_urlpatterns))
        # Generated here (sync setUp), not inside the async test methods -
        # RefreshToken.for_user() writes an OutstandingToken row (the
        # token_blacklist app is installed), and Django forbids ORM writes
        # from an async context without an explicit sync_to_async wrapper.
        self.owner_token = str(RefreshToken.for_user(self.owner).access_token)
        self.other_token = str(RefreshToken.for_user(self.other).access_token)

    def _ws_path(self, token=None):
        path = f"/ws/chat/{self.conversation.id}/"
        return f"{path}?token={token}" if token else path

    async def test_unauthenticated_connection_rejected(self):
        communicator = WebsocketCommunicator(self.application, self._ws_path())
        connected, _ = await communicator.connect()
        self.assertFalse(connected)
        await communicator.disconnect()

    async def test_other_user_cannot_connect(self):
        communicator = WebsocketCommunicator(self.application, self._ws_path(self.other_token))
        connected, _ = await communicator.connect()
        self.assertFalse(connected)
        await communicator.disconnect()

    async def test_owner_can_connect_and_receives_streamed_tokens(self):
        communicator = WebsocketCommunicator(self.application, self._ws_path(self.owner_token))
        connected, _ = await communicator.connect()
        self.assertTrue(connected)

        # Simulates what api/tasks.py's Celery task does per generated token -
        # this test doesn't run real GPU inference, just proves the
        # task -> channel layer -> consumer -> client relay works.
        channel_layer = get_channel_layer()
        group = conversation_group_name(self.conversation.id)
        await channel_layer.group_send(group, {"type": "chat.token", "delta": "Hel"})
        await channel_layer.group_send(group, {"type": "chat.token", "delta": "lo"})
        await channel_layer.group_send(group, {"type": "chat.done", "message_id": "abc-123"})

        first = json.loads(await communicator.receive_from())
        second = json.loads(await communicator.receive_from())
        third = json.loads(await communicator.receive_from())

        self.assertEqual(first, {"type": "token", "delta": "Hel"})
        self.assertEqual(second, {"type": "token", "delta": "lo"})
        self.assertEqual(third, {"type": "done", "message_id": "abc-123"})

        await communicator.disconnect()
