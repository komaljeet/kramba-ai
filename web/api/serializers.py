import os

from rest_framework import serializers

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password

from .models import Conversation, Message, ModelVersion
from .security import contains_injection_markers

User = get_user_model()

# Issue #30 security requirement: "Prompt length limits (max 10K chars per
# user message)".
MAX_MESSAGE_LENGTH = 10_000


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, validators=[validate_password])

    class Meta:
        model = User
        fields = ("id", "username", "email", "password")
        read_only_fields = ("id",)

    def create(self, validated_data):
        return User.objects.create_user(
            username=validated_data["username"],
            email=validated_data.get("email", ""),
            password=validated_data["password"],
        )


class ConversationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Conversation
        fields = ("id", "title", "model_version", "is_archived", "created_at", "updated_at")
        read_only_fields = ("id", "created_at", "updated_at")


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ("id", "role", "content", "token_count", "created_at")
        read_only_fields = fields


class ModelVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelVersion
        fields = ("id", "tag", "checkpoint_path", "checkpoint_sha256", "is_active", "created_at")
        read_only_fields = ("id", "created_at")

    def validate_checkpoint_path(self, value):
        # This registers a checkpoint already placed on the server's volume
        # (rsync/scp - see ISSUE_30_PLAN.md Phase 6 and the earlier session
        # discussion on why model files never go through an HTTP upload
        # endpoint), so it must already exist - and must live under the
        # configured checkpoint directory, not an arbitrary server path an
        # admin might mistype (e.g. "/etc/passwd").
        checkpoint_dir = os.path.realpath(settings.HVN_CHECKPOINT_DIR)
        resolved = os.path.realpath(value)
        if os.path.commonpath([checkpoint_dir, resolved]) != checkpoint_dir:
            raise serializers.ValidationError(
                f"checkpoint_path must be under {settings.HVN_CHECKPOINT_DIR}"
            )
        if not os.path.isfile(resolved):
            raise serializers.ValidationError(f"No file found at {value}")
        return value


class MessageCreateSerializer(serializers.Serializer):
    content = serializers.CharField(max_length=MAX_MESSAGE_LENGTH, allow_blank=False, trim_whitespace=False)

    def validate_content(self, value):
        if contains_injection_markers(value):
            # See api/security.py - these are HVN's literal conversation-turn
            # delimiters, not just arbitrary "suspicious" text.
            raise serializers.ValidationError(
                "Message contains reserved conversation control tokens and cannot be sent."
            )
        return value
