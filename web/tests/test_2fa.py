from django_otp.oath import totp
from django_otp.plugins.otp_totp.models import TOTPDevice
from rest_framework import status
from rest_framework.test import APITestCase

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.urls import reverse

User = get_user_model()
STRONG_PASSWORD = "correct-horse-battery-staple-9"


def current_totp_token(device: TOTPDevice) -> str:
    return f"{totp(device.bin_key):06d}"


class TwoFactorSetupTests(APITestCase):
    def setUp(self):
        cache.clear()
        self.admin = User.objects.create_user(username="2fa_admin", password=STRONG_PASSWORD, is_staff=True)
        self.regular = User.objects.create_user(username="2fa_regular", password=STRONG_PASSWORD)

    def _login(self, user):
        resp = self.client.post(
            reverse("token_obtain_pair"), {"username": user.username, "password": STRONG_PASSWORD}
        )
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {resp.data['access']}")

    def test_setup_requires_staff(self):
        self._login(self.regular)
        resp = self.client.post(reverse("2fa_setup"))
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)

    def test_setup_creates_unconfirmed_device(self):
        self._login(self.admin)
        resp = self.client.post(reverse("2fa_setup"))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("config_url", resp.data)
        device = TOTPDevice.objects.get(user=self.admin, confirmed=False)
        self.assertIn(device.config_url[:15], resp.data["config_url"][:15])  # both start "otpauth://totp/"

    def test_confirm_with_valid_token_activates_device(self):
        self._login(self.admin)
        self.client.post(reverse("2fa_setup"))
        device = TOTPDevice.objects.get(user=self.admin, confirmed=False)

        resp = self.client.post(reverse("2fa_confirm"), {"token": current_totp_token(device)})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        device.refresh_from_db()
        self.assertTrue(device.confirmed)

    def test_confirm_with_invalid_token_rejected(self):
        self._login(self.admin)
        self.client.post(reverse("2fa_setup"))
        resp = self.client.post(reverse("2fa_confirm"), {"token": "000000"})
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)


class TwoFactorLoginTests(APITestCase):
    def setUp(self):
        cache.clear()
        self.admin = User.objects.create_user(username="2fa_login_admin", password=STRONG_PASSWORD, is_staff=True)

    def test_login_without_2fa_device_works_normally(self):
        resp = self.client.post(
            reverse("token_obtain_pair"), {"username": self.admin.username, "password": STRONG_PASSWORD}
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("access", resp.data)

    def test_unconfirmed_device_does_not_gate_login(self):
        TOTPDevice.objects.create(user=self.admin, confirmed=False, name="pending")
        resp = self.client.post(
            reverse("token_obtain_pair"), {"username": self.admin.username, "password": STRONG_PASSWORD}
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_confirmed_device_requires_otp_token(self):
        TOTPDevice.objects.create(user=self.admin, confirmed=True, name="default")
        resp = self.client.post(
            reverse("token_obtain_pair"), {"username": self.admin.username, "password": STRONG_PASSWORD}
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_confirmed_device_with_valid_otp_token_succeeds(self):
        device = TOTPDevice.objects.create(user=self.admin, confirmed=True, name="default")
        resp = self.client.post(
            reverse("token_obtain_pair"),
            {
                "username": self.admin.username,
                "password": STRONG_PASSWORD,
                "otp_token": current_totp_token(device),
            },
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("access", resp.data)

    def test_confirmed_device_with_wrong_otp_token_rejected(self):
        TOTPDevice.objects.create(user=self.admin, confirmed=True, name="default")
        resp = self.client.post(
            reverse("token_obtain_pair"),
            {"username": self.admin.username, "password": STRONG_PASSWORD, "otp_token": "000000"},
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
