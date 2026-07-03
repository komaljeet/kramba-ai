from rest_framework import status
from rest_framework.test import APITestCase

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.urls import reverse

from api.models import Conversation, Message, ModelVersion

User = get_user_model()
STRONG_PASSWORD = "correct-horse-battery-staple-9"


class AccountExportDeleteTests(APITestCase):
    def setUp(self):
        cache.clear()
        self.model_version = ModelVersion.objects.create(tag="acct-test-v1", checkpoint_path="/dev/null")
        self.user = User.objects.create_user(username="acct_user", password=STRONG_PASSWORD)
        self.conversation = Conversation.objects.create(
            owner=self.user, title="export test", model_version=self.model_version
        )
        Message.objects.create(conversation=self.conversation, role=Message.Role.USER, content="hi")
        login = self.client.post(
            reverse("token_obtain_pair"), {"username": "acct_user", "password": STRONG_PASSWORD}
        )
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {login.data['access']}")

    def test_export_requires_auth(self):
        self.client.credentials()
        resp = self.client.get(reverse("account_export"))
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_export_returns_account_and_conversations_with_messages(self):
        resp = self.client.get(reverse("account_export"))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data["account"]["username"], "acct_user")
        self.assertEqual(len(resp.data["conversations"]), 1)
        self.assertEqual(len(resp.data["conversations"][0]["messages"]), 1)
        self.assertEqual(resp.data["conversations"][0]["messages"][0]["content"], "hi")

    def test_delete_rejects_wrong_password(self):
        resp = self.client.delete(reverse("account_delete"), {"password": "wrong-password-entirely"})
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertTrue(User.objects.filter(username="acct_user").exists())

    def test_delete_with_correct_password_cascades(self):
        conversation_id = self.conversation.id
        resp = self.client.delete(reverse("account_delete"), {"password": STRONG_PASSWORD})
        self.assertEqual(resp.status_code, status.HTTP_204_NO_CONTENT)
        self.assertFalse(User.objects.filter(username="acct_user").exists())
        self.assertFalse(Conversation.objects.filter(id=conversation_id).exists())

    def test_token_unusable_immediately_after_deletion(self):
        self.client.delete(reverse("account_delete"), {"password": STRONG_PASSWORD})
        # Same (still time-valid) access token, now for a deleted user.
        resp = self.client.get(reverse("conversation-list"))
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)
