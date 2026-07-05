import os

from rest_framework import serializers

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password

from .models import Conversation, Message, ModelVersion, TrainingJob
from .security import contains_injection_markers

# The coding-curriculum stage names (Issue #53/#54). eval.curriculum is the
# source of truth and torch-free, so importing it doesn't pull the GPU stack
# into the web process - but `eval` only sits on the path when the repo root is
# importable (the Docker image sets PYTHONPATH=/srv; a bare `cd web && manage.py
# test` does not). Fall back to the literal list so app/test startup never hinges
# on repo-root layout; the two are kept in sync (the ChoiceField below rejects
# anything else anyway).
try:
    from eval.curriculum import CURRICULUM_STAGES
except Exception:  # pragma: no cover - repo root not on sys.path (local test run)
    CURRICULUM_STAGES = ["html", "css", "js", "php", "python", "devops"]

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


class TrainingJobSerializer(serializers.ModelSerializer):
    """Read view of a training run (Issue #57): status, config, live progress,
    and the ModelVersion it produced on success."""

    result_model_version_tag = serializers.CharField(
        source="result_model_version.tag", read_only=True, default=None
    )

    class Meta:
        model = TrainingJob
        fields = (
            "id", "status", "config", "current_stage", "stages_done", "stages_total",
            "last_metrics", "detail", "error", "result_model_version",
            "result_model_version_tag", "created_at", "started_at", "finished_at",
        )
        read_only_fields = fields


class TrainingJobCreateSerializer(serializers.Serializer):
    """Validates the launch config for an in-app training run.

    The model architecture (d_model, deltanet, etc.) is fixed by the base
    checkpoint the run continues from (auto_train.run_curriculum carries the
    weights forward) - this only chooses which registered ModelVersion to
    continue from, which curriculum stages to train, and the training budget.
    """

    from_model_version = serializers.PrimaryKeyRelatedField(queryset=ModelVersion.objects.all())
    stages = serializers.ListField(
        child=serializers.ChoiceField(choices=CURRICULUM_STAGES),
        allow_empty=False,
        default=list(CURRICULUM_STAGES),
    )
    epochs_per_stage = serializers.IntegerField(min_value=1, max_value=100, default=2)
    seq_len = serializers.IntegerField(min_value=16, max_value=2048, default=128)
    batch_size = serializers.IntegerField(min_value=1, max_value=256, default=16)
    lr = serializers.FloatField(min_value=1e-6, max_value=1.0, default=1e-4)
    precision = serializers.ChoiceField(choices=["fp32", "fp16", "bf16"], default="bf16")
    download = serializers.BooleanField(default=False)
    # Fast offline loop-validation (tiny budget/epochs) - the app's equivalent
    # of `auto_train.py --smoke`, for verifying the wiring on the 6 GB box.
    smoke = serializers.BooleanField(default=False)

    def validate_stages(self, value):
        # Preserve the canonical curriculum order regardless of the order the
        # client sent them in, and drop duplicates (stages train sequentially).
        seen = set()
        ordered = [s for s in CURRICULUM_STAGES if s in value and not (s in seen or seen.add(s))]
        return ordered

    def to_config(self) -> dict:
        """Flattens the validated data into the JSON config stored on the
        TrainingJob and consumed by api.tasks.run_training."""
        data = self.validated_data
        model_version = data["from_model_version"]
        cfg = {
            "from_model_version_id": str(model_version.id),
            "from_model_version_tag": model_version.tag,
            "base_checkpoint_path": model_version.checkpoint_path,
            "stages": data["stages"],
            "epochs_per_stage": data["epochs_per_stage"],
            "seq_len": data["seq_len"],
            "batch_size": data["batch_size"],
            "lr": data["lr"],
            "precision": data["precision"],
            "download": data["download"],
            "smoke": data["smoke"],
        }
        if data["smoke"]:
            # Mirror auto_train.main()'s --smoke preset: tiny everything, offline.
            cfg.update(
                epochs_per_stage=1,
                seq_len=min(data["seq_len"], 64),
                max_train_chars=60_000,
                max_eval_chars=6_000,
                download=False,
            )
        return cfg


class MessageCreateSerializer(serializers.Serializer):
    content = serializers.CharField(max_length=MAX_MESSAGE_LENGTH, allow_blank=False, trim_whitespace=False)
    # Per-message toggles (Issue #56), plumbed through to the worker
    # (api/tasks.py -> hvn_worker.inference.stream_turn):
    #   thinking   - keep the model's [THINK] chain-of-thought (#42) in the reply
    #   web_search - offer the web_search tool (#41) for this turn
    thinking = serializers.BooleanField(required=False, default=True)
    web_search = serializers.BooleanField(required=False, default=False)

    def validate_content(self, value):
        if contains_injection_markers(value):
            # See api/security.py - these are HVN's literal conversation-turn
            # delimiters, not just arbitrary "suspicious" text.
            raise serializers.ValidationError(
                "Message contains reserved conversation control tokens and cannot be sent."
            )
        return value
