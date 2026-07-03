from rest_framework import status
from rest_framework.test import APITestCase

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.urls import reverse

from api.models import Conversation, ModelVersion

User = get_user_model()

STRONG_PASSWORD = "correct-horse-battery-staple-9"  # 12+ chars, not all-numeric


class ThrottleResetMixin:
    """DRF's default throttles are cache-backed and cache isn't part of
    Django's per-test transaction rollback, so anonymous request counts would
    otherwise accumulate across every test in this module and start
    returning 429 instead of the status the test actually means to check."""

    def setUp(self):
        cache.clear()
        super().setUp()


class RegistrationTests(ThrottleResetMixin, APITestCase):
    def test_register_creates_user(self):
        resp = self.client.post(
            reverse("register"),
            {"username": "alice", "email": "alice@example.com", "password": STRONG_PASSWORD},
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        self.assertTrue(User.objects.filter(username="alice").exists())
        self.assertNotIn("password", resp.data)

    def test_register_rejects_short_password(self):
        resp = self.client.post(
            reverse("register"),
            {"username": "bob", "email": "bob@example.com", "password": "short1"},
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertFalse(User.objects.filter(username="bob").exists())

    def test_register_rejects_fully_numeric_password(self):
        resp = self.client.post(
            reverse("register"),
            {"username": "carol", "email": "carol@example.com", "password": "123456789012"},
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)


class LoginRefreshLogoutTests(ThrottleResetMixin, APITestCase):
    def setUp(self):
        super().setUp()
        self.user = User.objects.create_user(username="dave", password=STRONG_PASSWORD)

    def test_login_returns_access_and_refresh_tokens(self):
        resp = self.client.post(
            reverse("token_obtain_pair"), {"username": "dave", "password": STRONG_PASSWORD}
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("access", resp.data)
        self.assertIn("refresh", resp.data)

    def test_login_rejects_wrong_password(self):
        resp = self.client.post(
            reverse("token_obtain_pair"), {"username": "dave", "password": "wrong-password-entirely"}
        )
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_refresh_returns_new_access_token(self):
        login = self.client.post(
            reverse("token_obtain_pair"), {"username": "dave", "password": STRONG_PASSWORD}
        )
        resp = self.client.post(reverse("token_refresh"), {"refresh": login.data["refresh"]})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("access", resp.data)

    def test_logout_blacklists_refresh_token(self):
        login = self.client.post(
            reverse("token_obtain_pair"), {"username": "dave", "password": STRONG_PASSWORD}
        )
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {login.data['access']}")
        logout_resp = self.client.post(reverse("logout"), {"refresh": login.data["refresh"]})
        self.assertEqual(logout_resp.status_code, status.HTTP_205_RESET_CONTENT)

        # The blacklisted refresh token must no longer mint access tokens.
        refresh_resp = self.client.post(reverse("token_refresh"), {"refresh": login.data["refresh"]})
        self.assertEqual(refresh_resp.status_code, status.HTTP_401_UNAUTHORIZED)


class ConversationIsolationTests(ThrottleResetMixin, APITestCase):
    def setUp(self):
        super().setUp()
        self.model_version = ModelVersion.objects.create(
            tag="test-v1", checkpoint_path="/data/checkpoints/test.pt"
        )
        self.user_a = User.objects.create_user(username="user_a", password=STRONG_PASSWORD)
        self.user_b = User.objects.create_user(username="user_b", password=STRONG_PASSWORD)
        self.conversation = Conversation.objects.create(
            owner=self.user_a, title="A's conversation", model_version=self.model_version
        )

    def _login(self, user):
        resp = self.client.post(
            reverse("token_obtain_pair"), {"username": user.username, "password": STRONG_PASSWORD}
        )
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {resp.data['access']}")

    def test_owner_can_retrieve_own_conversation(self):
        self._login(self.user_a)
        resp = self.client.get(reverse("conversation-detail", args=[self.conversation.id]))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_other_user_cannot_retrieve_conversation(self):
        self._login(self.user_b)
        resp = self.client.get(reverse("conversation-detail", args=[self.conversation.id]))
        # Queryset is scoped to the requesting user (api/views.py
        # ConversationViewSet.get_queryset), so a non-owner's request never
        # matches a row to run object-permission checks against - it 404s
        # rather than 403s, which avoids confirming the ID even exists.
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_other_user_conversation_list_excludes_it(self):
        self._login(self.user_b)
        resp = self.client.get(reverse("conversation-list"))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        ids = [c["id"] for c in resp.data["results"]] if "results" in resp.data else [c["id"] for c in resp.data]
        self.assertNotIn(str(self.conversation.id), ids)

    def test_unauthenticated_request_rejected(self):
        resp = self.client.get(reverse("conversation-list"))
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)
