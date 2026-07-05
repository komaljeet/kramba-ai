"""
Issue #56: deleting a conversation must free its on-disk Hamiltonian state
file, not just its Postgres rows.

Before the fix, ConversationViewSet had no custom destroy, so a user-deleted
chat orphaned `<checkpoints>/conversations/<id>.json` forever (the 90-day purge
can't find it once the row is gone). These tests drive the real DELETE endpoint
and assert the file is gone. Torch-free — uses hvn_worker.state.state_path, so
it runs without the GPU stack installed.
"""
import os
import tempfile

from rest_framework.test import APITestCase

from django.contrib.auth import get_user_model
from django.test import override_settings

from api.models import Conversation, ModelVersion
from hvn_worker.state import state_path

User = get_user_model()
STRONG_PASSWORD = "correct-horse-battery-staple-9"
_CKPT_DIR = tempfile.mkdtemp(prefix="hvn_delete_state_")


@override_settings(HVN_CHECKPOINT_DIR=_CKPT_DIR)
class DeleteFreesStateTests(APITestCase):
    def setUp(self):
        self.model_version = ModelVersion.objects.create(tag="del-test-v1", checkpoint_path="/dev/null")
        self.user = User.objects.create_user(username="del_user", password=STRONG_PASSWORD)
        self.other = User.objects.create_user(username="other_user", password=STRONG_PASSWORD)
        self.conversation = Conversation.objects.create(
            owner=self.user, title="to delete", model_version=self.model_version
        )
        self.client.force_authenticate(user=self.user)

    def _write_state_file(self, conversation_id) -> str:
        path = state_path(conversation_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("{}")
        return path

    def test_delete_removes_state_file_and_rows(self):
        path = self._write_state_file(self.conversation.id)
        self.assertTrue(os.path.exists(path))

        res = self.client.delete(f"/api/conversations/{self.conversation.id}/")

        self.assertEqual(res.status_code, 204)
        self.assertFalse(Conversation.objects.filter(pk=self.conversation.pk).exists())
        self.assertFalse(os.path.exists(path), "state file orphaned after delete")

    def test_delete_without_state_file_still_succeeds(self):
        # A conversation deleted before any completed turn has no state file;
        # the destroy path must not raise on the missing file.
        path = state_path(self.conversation.id)
        self.assertFalse(os.path.exists(path))

        res = self.client.delete(f"/api/conversations/{self.conversation.id}/")

        self.assertEqual(res.status_code, 204)
        self.assertFalse(Conversation.objects.filter(pk=self.conversation.pk).exists())

    def test_cannot_delete_another_users_conversation(self):
        # Isolation guard: a foreign conversation 404s (not 403) and its state
        # file is left untouched.
        foreign = Conversation.objects.create(
            owner=self.other, title="not yours", model_version=self.model_version
        )
        path = self._write_state_file(foreign.id)

        res = self.client.delete(f"/api/conversations/{foreign.id}/")

        self.assertEqual(res.status_code, 404)
        self.assertTrue(Conversation.objects.filter(pk=foreign.pk).exists())
        self.assertTrue(os.path.exists(path))
