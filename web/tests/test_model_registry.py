import os

from rest_framework import status
from rest_framework.test import APITestCase

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.urls import reverse

from api.models import ModelVersion

User = get_user_model()
STRONG_PASSWORD = "correct-horse-battery-staple-9"


class ModelRegistryTests(APITestCase):
    def setUp(self):
        cache.clear()
        self.regular_user = User.objects.create_user(username="mv_user", password=STRONG_PASSWORD)
        self.admin_user = User.objects.create_user(
            username="mv_admin", password=STRONG_PASSWORD, is_staff=True
        )
        # A real file under HVN_CHECKPOINT_DIR - both existence and
        # directory-containment are enforced by ModelVersionSerializer, so a
        # fixture like "/dev/null" (used elsewhere for ORM-level ModelVersion
        # fixtures that skip the serializer) wouldn't pass here.
        os.makedirs(settings.HVN_CHECKPOINT_DIR, exist_ok=True)
        self.valid_checkpoint_path = os.path.join(settings.HVN_CHECKPOINT_DIR, "test_registry_ckpt.pt")
        with open(self.valid_checkpoint_path, "wb") as f:
            f.write(b"not a real checkpoint, just needs to exist")

    def tearDown(self):
        if os.path.exists(self.valid_checkpoint_path):
            os.remove(self.valid_checkpoint_path)

    def _login(self, user):
        resp = self.client.post(
            reverse("token_obtain_pair"), {"username": user.username, "password": STRONG_PASSWORD}
        )
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {resp.data['access']}")

    def test_regular_user_can_list_models(self):
        ModelVersion.objects.create(tag="mv-list-v1", checkpoint_path="/dev/null")
        self._login(self.regular_user)
        resp = self.client.get(reverse("modelversion-list"))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_regular_user_cannot_create_model(self):
        self._login(self.regular_user)
        resp = self.client.post(
            reverse("modelversion-list"),
            {"tag": "mv-forbidden", "checkpoint_path": self.valid_checkpoint_path},
        )
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)

    def test_admin_can_register_valid_checkpoint(self):
        self._login(self.admin_user)
        resp = self.client.post(
            reverse("modelversion-list"),
            {"tag": "mv-admin-v1", "checkpoint_path": self.valid_checkpoint_path},
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        created = ModelVersion.objects.get(tag="mv-admin-v1")
        self.assertEqual(created.created_by, self.admin_user)

    def test_admin_registration_rejects_nonexistent_path(self):
        self._login(self.admin_user)
        missing_path = os.path.join(settings.HVN_CHECKPOINT_DIR, "does_not_exist.pt")
        resp = self.client.post(
            reverse("modelversion-list"), {"tag": "mv-missing", "checkpoint_path": missing_path}
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_admin_registration_rejects_path_outside_checkpoint_dir(self):
        self._login(self.admin_user)
        resp = self.client.post(
            reverse("modelversion-list"), {"tag": "mv-traversal", "checkpoint_path": "/etc/hostname"}
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
