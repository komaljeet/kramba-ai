"""
Self-service account endpoints: 2FA (admin accounts), GDPR export/deletion.

Split out from views.py (which stays focused on the Conversation/ModelVersion
resource ViewSets) because these act on the requesting user's own account
rather than on a resource with its own ownership model.
"""
from django_otp import match_token
from django_otp.plugins.otp_totp.models import TOTPDevice
from rest_framework import permissions, serializers, status
from rest_framework.exceptions import PermissionDenied
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.views import TokenObtainPairView

from django.contrib.auth import get_user_model

from .models import Conversation
from .serializers import ConversationSerializer, MessageSerializer

User = get_user_model()


# --- 2FA / TOTP (Issue #30 Phase 8.1 - admin accounts) ---------------------


class TwoFactorSetupView(APIView):
    """Creates (or returns the existing) unconfirmed TOTP device for the
    requesting admin and returns its otpauth:// provisioning URI. The client
    renders this as a QR code (e.g. any JS QR library) - no image generation
    happens server-side, keeping this dependency-light."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        if not request.user.is_staff:
            raise PermissionDenied("2FA is currently limited to admin accounts.")
        device, _ = TOTPDevice.objects.get_or_create(
            user=request.user, confirmed=False, defaults={"name": "default"}
        )
        return Response({"config_url": device.config_url})


class TwoFactorConfirmView(APIView):
    """Activates the pending device once the admin proves they can generate
    a valid code from it - a device that was merely created (setup) but
    never confirmed does NOT gate login (see TwoFactorTokenObtainPairSerializer),
    otherwise a setup call with no follow-through would lock the account out."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        token = request.data.get("token", "")
        device = TOTPDevice.objects.filter(user=request.user, confirmed=False).first()
        if device is None:
            return Response(
                {"detail": "No pending 2FA device - call /auth/2fa/setup/ first."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not device.verify_token(token):
            return Response({"detail": "Invalid token."}, status=status.HTTP_400_BAD_REQUEST)
        device.confirmed = True
        device.save(update_fields=["confirmed"])
        return Response({"detail": "2FA enabled."})


class TwoFactorTokenObtainPairSerializer(TokenObtainPairSerializer):
    otp_token = serializers.CharField(required=False, allow_blank=True, write_only=True)

    def validate(self, attrs):
        otp_token = attrs.pop("otp_token", "")
        data = super().validate(attrs)
        # self.user is set by the parent's validate() above.
        if TOTPDevice.objects.filter(user=self.user, confirmed=True).exists():
            if not otp_token or match_token(self.user, otp_token) is None:
                raise serializers.ValidationError(
                    {"otp_token": "A valid 2FA code is required for this account."}
                )
        return data


class TwoFactorTokenObtainPairView(TokenObtainPairView):
    serializer_class = TwoFactorTokenObtainPairSerializer


# --- GDPR export / deletion (Issue #30 Phase 8.4/8.5) -----------------------


class AccountExportView(APIView):
    """Returns everything this account owns as JSON - account fields plus
    every conversation and message. Password hash is never included (it's
    not a serialized field on any of these)."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        user = request.user
        conversations = Conversation.objects.filter(owner=user).prefetch_related("messages")
        return Response(
            {
                "account": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "date_joined": user.date_joined,
                },
                "conversations": [
                    {
                        **ConversationSerializer(c).data,
                        "messages": MessageSerializer(c.messages.all(), many=True).data,
                    }
                    for c in conversations
                ],
            }
        )


class AccountDeleteView(APIView):
    """Deletes the account and, via the Conversation.owner FK's CASCADE,
    every conversation/message it owns. Requires re-entering the password in
    the request body - this is irreversible, and a leaked/short-lived access
    token alone shouldn't be enough to trigger it."""

    permission_classes = [permissions.IsAuthenticated]

    def delete(self, request):
        password = request.data.get("password", "")
        if not request.user.check_password(password):
            return Response({"detail": "Incorrect password."}, status=status.HTTP_400_BAD_REQUEST)
        request.user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
