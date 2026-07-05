"""
Issue #56: the per-message web-search / thinking toggles must be accepted by
the message endpoint and forwarded to the GPU worker task.

Torch-free — the Celery task is mocked, so these assert the request-side
plumbing (serializer defaults + view -> generate_response.delay kwargs) without
loading a model. The worker-side effect of `thinking` (stripping [THINK] blocks
from the persisted reply) is exercised end-to-end by the #52 chat smoke test
when torch is available.
"""
from unittest import mock

from rest_framework.test import APITestCase

from django.contrib.auth import get_user_model

from api.models import Conversation, ModelVersion
from api.serializers import MessageCreateSerializer

User = get_user_model()
STRONG_PASSWORD = "correct-horse-battery-staple-9"


class MessageToggleSerializerTests(APITestCase):
    def test_defaults(self):
        s = MessageCreateSerializer(data={"content": "hi"})
        self.assertTrue(s.is_valid(), s.errors)
        self.assertTrue(s.validated_data["thinking"])
        self.assertFalse(s.validated_data["web_search"])

    def test_explicit_values(self):
        s = MessageCreateSerializer(data={"content": "hi", "thinking": False, "web_search": True})
        self.assertTrue(s.is_valid(), s.errors)
        self.assertFalse(s.validated_data["thinking"])
        self.assertTrue(s.validated_data["web_search"])


class MessageToggleViewTests(APITestCase):
    def setUp(self):
        self.model_version = ModelVersion.objects.create(tag="tg-test-v1", checkpoint_path="/dev/null")
        self.user = User.objects.create_user(username="tg_user", password=STRONG_PASSWORD)
        self.conversation = Conversation.objects.create(
            owner=self.user, title="toggles", model_version=self.model_version
        )
        self.client.force_authenticate(user=self.user)

    @mock.patch("api.views.generate_response")
    def test_toggles_forwarded_to_worker(self, mock_task):
        mock_task.delay.return_value = mock.Mock(id="task-123")

        res = self.client.post(
            f"/api/conversations/{self.conversation.id}/messages/",
            {"content": "search the web please", "thinking": False, "web_search": True},
            format="json",
        )

        self.assertEqual(res.status_code, 202)
        mock_task.delay.assert_called_once()
        _, kwargs = mock_task.delay.call_args
        self.assertEqual(kwargs["thinking"], False)
        self.assertEqual(kwargs["web_search"], True)

    @mock.patch("api.views.generate_response")
    def test_toggle_defaults_when_omitted(self, mock_task):
        mock_task.delay.return_value = mock.Mock(id="task-456")

        res = self.client.post(
            f"/api/conversations/{self.conversation.id}/messages/",
            {"content": "hello"},
            format="json",
        )

        self.assertEqual(res.status_code, 202)
        _, kwargs = mock_task.delay.call_args
        self.assertEqual(kwargs["thinking"], True)
        self.assertEqual(kwargs["web_search"], False)
