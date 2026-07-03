import os

from rest_framework.test import APITestCase

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import override_settings
from django.utils import timezone

from api.models import Conversation, ModelVersion
from api.tasks import purge_expired_conversations
from hvn_worker.inference import _state_path

User = get_user_model()


@override_settings(CONVERSATION_RETENTION_DAYS=30)
class PurgeExpiredConversationsTests(APITestCase):
    def setUp(self):
        self.model_version = ModelVersion.objects.create(tag="purge-test-v1", checkpoint_path="/dev/null")
        self.user = User.objects.create_user(username="purge_user", password="correct-horse-battery-staple-9")

    def _make_conversation(self, days_old: int) -> Conversation:
        conversation = Conversation.objects.create(
            owner=self.user, title=f"{days_old}d old", model_version=self.model_version
        )
        # updated_at has auto_now=True, so a normal .save() would just reset
        # it to "now" again - .update() bypasses model-level auto_now logic.
        Conversation.objects.filter(pk=conversation.pk).update(
            updated_at=timezone.now() - timezone.timedelta(days=days_old)
        )
        conversation.refresh_from_db()
        return conversation

    def test_purges_conversations_older_than_retention(self):
        old = self._make_conversation(days_old=45)
        recent = self._make_conversation(days_old=5)

        purged_count = purge_expired_conversations()

        self.assertEqual(purged_count, 1)
        self.assertFalse(Conversation.objects.filter(pk=old.pk).exists())
        self.assertTrue(Conversation.objects.filter(pk=recent.pk).exists())

    def test_purges_associated_state_file(self):
        old = self._make_conversation(days_old=45)
        state_path = _state_path(old.id)
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "w") as f:
            f.write("{}")

        purge_expired_conversations()

        self.assertFalse(os.path.exists(state_path))

    def test_no_op_when_nothing_expired(self):
        self._make_conversation(days_old=1)
        purged_count = purge_expired_conversations()
        self.assertEqual(purged_count, 0)
