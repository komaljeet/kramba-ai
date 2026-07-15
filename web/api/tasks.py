import hashlib
import logging
import os

from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import get_channel_layer

from django.conf import settings
from django.core.cache import cache
from django.utils import timezone

from .models import AuditLog, Conversation, Message, ModelVersion, TrainingJob
from .security import filter_profanity, sanitize_output

logger = logging.getLogger(__name__)


def conversation_group_name(conversation_id) -> str:
    return f"conversation_{conversation_id}"


def cancel_key(conversation_id) -> str:
    """Redis cache key for the cross-process "stop generation" flag (Issue #61).

    The web role (ConversationViewSet.stop) sets it; the generate_response task
    running on the GPU worker polls it between tokens. Must live on the shared
    Redis cache (not LocMem) since those are separate processes - see CACHES in
    web/settings/base.py."""
    return f"chat_cancel:{conversation_id}"


def training_group_name(job_id) -> str:
    return f"training_{job_id}"


@shared_task
def generate_response(
    conversation_id: str,
    user_message_id: str,
    thinking: bool = True,
    web_search: bool = False,
) -> str:
    """Runs one HVN chat turn for `conversation_id`, streaming each text
    delta to that conversation's Channels group as it's generated (consumed
    by chat.consumers.ChatConsumer over WebSocket - see ISSUE_30_PLAN.md
    Phase 4), then stores the finished reply as a Message.

    Takes the already-created user Message's id (not the raw text) rather
    than generating it here - the view records what the user actually said
    before scheduling this task, so message ordering in the DB stays correct
    even if the task sits in the queue for a while before running.
    """
    # Local import: torch/hvn are heavy and GPU-bound: keep them off the
    # web process's import path entirely, only the worker role needs them.
    from hvn_worker.inference import stream_turn

    conversation = Conversation.objects.select_related("model_version").get(id=conversation_id)
    user_message = Message.objects.get(id=user_message_id, conversation_id=conversation_id)

    channel_layer = get_channel_layer()
    group = conversation_group_name(conversation_id)
    key = cancel_key(conversation_id)
    # NB: deliberately do NOT clear the flag here. The Stop button is only shown
    # while a turn is generating, so the flag is always set against *this* running
    # task and consumed below; a start-of-task delete would instead race with (and
    # silently swallow) a stop issued in the enqueue→start window. The `finally`
    # clears it after every turn, so it can't leak into the next one.
    result: dict = {}
    gen = stream_turn(
        conversation_id=conversation.id,
        user_message=user_message.content,
        model_tag=conversation.model_version.tag,
        checkpoint_path=conversation.model_version.checkpoint_path,
        architecture=conversation.model_version.architecture,
        result=result,
        thinking=thinking,
        web_search=web_search,
    )
    try:
        stopped = False
        for delta in gen:
            if cache.get(key):
                # Stop requested (Issue #61). Close the generator: its post-yield
                # state-save (hvn.chat.user_turn_streaming's finalize) never runs,
                # so this turn is discarded cleanly - the Hamiltonian state file
                # keeps its pre-turn value rather than being left half-updated.
                stopped = True
                gen.close()
                break
            async_to_sync(channel_layer.group_send)(group, {"type": "chat.token", "delta": delta})

        if stopped:
            async_to_sync(channel_layer.group_send)(
                group, {"type": "chat.done", "message_id": "", "stopped": True}
            )
            return "stopped"

        # Sanitizes the finished, persisted message. The live per-delta stream
        # already sent over WebSocket above is NOT re-sanitized retroactively -
        # cleaning HTML tags that may be split across delta boundaries mid-stream
        # isn't reliably possible without risking corrupting legitimate text, and
        # the attacker doesn't control model output directly (only the prompt,
        # which is itself guarded by MessageCreateSerializer's injection check).
        # Accepted as a known limitation rather than a gap silently left in.
        clean_content = filter_profanity(sanitize_output(result["content"]))
        assistant_message = Message.objects.create(
            conversation=conversation, role=Message.Role.ASSISTANT, content=clean_content
        )

        async_to_sync(channel_layer.group_send)(
            group, {"type": "chat.done", "message_id": str(assistant_message.id)}
        )

        AuditLog.objects.create(
            user_id=conversation.owner_id,
            action="generate",
            path=f"/api/conversations/{conversation.id}/messages/",
            method="POST",
            status_code=201,
            prompt_hash=hashlib.sha256(user_message.content.encode()).hexdigest(),
            model_version=conversation.model_version,
        )
        return str(assistant_message.id)
    except Exception:  # noqa: BLE001 - any generation failure must reach the UI,
        # not hang it. Without this the task would die silently and the client
        # would sit on "Generating…" forever with no terminal frame (Issue #60).
        logger.exception("generate_response failed for conversation %s", conversation_id)
        async_to_sync(channel_layer.group_send)(
            group, {"type": "chat.error", "detail": "Generation failed. Please try again."}
        )
        return "error"
    finally:
        cache.delete(key)


class _TrainingCancelled(Exception):
    """Raised out of the progress callback when a staff user cancels a running
    job, so run_curriculum unwinds cleanly after the current stage is committed
    (its checkpoint/resume-state are already on disk) rather than being killed
    mid-write."""


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@shared_task(bind=True)
def run_training(self, job_id: str) -> str:
    """Runs one in-app training job (Issue #57) on the dedicated GPU training
    worker: reuses auto_train.run_curriculum (Issue #54) to train the requested
    curriculum stages from a base checkpoint, streams per-stage progress to the
    job's Channels group (consumed by chat.consumers.TrainingConsumer over
    WebSocket), and on success registers the produced checkpoint as a new
    ModelVersion.

    Routed to the ``training`` queue (CELERY_TASK_ROUTES) so it runs on a
    separate worker from chat inference - an hours-long GPU-saturating run must
    never starve chat latency. Cancellation is cooperative: the view sets the
    job to ``cancelling`` and the progress callback (checked at each stage
    boundary) raises so the run stops after the current stage's checkpoint is
    safely written.
    """
    from django.utils import timezone

    # torch/hvn are heavy and GPU-bound - keep them off the import path until a
    # job actually runs (same discipline as generate_response above).
    from auto_train import run_curriculum

    job = TrainingJob.objects.get(id=job_id)
    channel_layer = get_channel_layer()
    group = training_group_name(job_id)

    def stream(msg_type: str, **data) -> None:
        async_to_sync(channel_layer.group_send)(group, {"type": msg_type, **data})

    job.status = TrainingJob.Status.RUNNING
    job.started_at = timezone.now()
    job.celery_task_id = self.request.id or ""
    job.detail = "Starting…"
    job.save(update_fields=["status", "started_at", "celery_task_id", "detail"])
    stream("training.status", status=job.status, detail=job.detail)

    cfg = job.config
    stages = cfg["stages"]

    def progress_cb(event: dict) -> None:
        # Re-read only the status column so a concurrent cancel is seen at the
        # next stage boundary; raising here aborts run_curriculum cooperatively.
        job.refresh_from_db(fields=["status"])
        if job.status == TrainingJob.Status.CANCELLING:
            raise _TrainingCancelled()

        ev = event.get("event")
        if ev == "run_start":
            job.stages_total = len(event.get("stages", stages))
            job.stages_done = len(event.get("done", []))
            job.detail = f"Running {len(event.get('todo', []))} stage(s)."
            job.save(update_fields=["stages_total", "stages_done", "detail"])
        elif ev == "stage_start":
            job.current_stage = event["stage"]
            job.detail = f"Training stage '{event['stage']}'…"
            job.save(update_fields=["current_stage", "detail"])
        elif ev == "stage_complete":
            job.stages_done += 1
            job.last_metrics = event.get("metrics")
            job.detail = f"Stage '{event['stage']}' done."
            job.save(update_fields=["stages_done", "last_metrics", "detail"])
        elif ev == "run_complete":
            job.current_stage = ""
            job.save(update_fields=["current_stage"])
        stream("training.progress", event=event)

    try:
        base_ckpt = cfg["base_checkpoint_path"]
        out_dir = os.path.join(settings.HVN_CHECKPOINT_DIR, f"training_{job_id}")
        state = run_curriculum(
            base_checkpoint=base_ckpt,
            stages=stages,
            out_dir=out_dir,
            epochs_per_stage=cfg["epochs_per_stage"],
            seq_len=cfg["seq_len"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            precision=cfg["precision"],
            max_train_chars=cfg.get("max_train_chars"),
            max_eval_chars=cfg.get("max_eval_chars"),
            local_dir=cfg.get("local_dir"),
            allow_download=cfg.get("download", False),
            resume=False,  # each job is a fresh run in its own out_dir
            progress_cb=progress_cb,
        )
    except _TrainingCancelled:
        job.refresh_from_db()
        job.status = TrainingJob.Status.CANCELLED
        job.finished_at = timezone.now()
        job.detail = "Cancelled."
        job.save(update_fields=["status", "finished_at", "detail"])
        stream("training.done", status=job.status, detail=job.detail)
        return job.status
    except Exception as exc:  # noqa: BLE001 - report any failure to the UI, don't crash silently
        logger.exception("Training job %s failed", job_id)
        job.refresh_from_db()
        job.status = TrainingJob.Status.FAILED
        job.finished_at = timezone.now()
        job.error = f"{type(exc).__name__}: {exc}"
        job.detail = "Failed."
        job.save(update_fields=["status", "finished_at", "error", "detail"])
        stream("training.done", status=job.status, detail=job.detail, error=job.error)
        return job.status

    # Register the produced checkpoint as a new ModelVersion (if any stage ran).
    final_ckpt = state.get("current_checkpoint")
    model_version = None
    if final_ckpt and os.path.isfile(final_ckpt) and os.path.realpath(final_ckpt) != os.path.realpath(base_ckpt):
        tag = f"train-{str(job_id)[:8]}-{state['completed'][-1]['stage']}" if state.get("completed") else f"train-{str(job_id)[:8]}"
        model_version = ModelVersion.objects.create(
            tag=tag,
            checkpoint_path=final_ckpt,
            checkpoint_sha256=_sha256_file(final_ckpt),
            created_by=job.created_by,
        )

    job.refresh_from_db()
    job.status = TrainingJob.Status.SUCCEEDED
    job.finished_at = timezone.now()
    job.result_model_version = model_version
    job.detail = "Complete."
    job.save(update_fields=["status", "finished_at", "result_model_version", "detail"])
    stream(
        "training.done",
        status=job.status,
        detail=job.detail,
        model_version_id=str(model_version.id) if model_version else None,
        model_version_tag=model_version.tag if model_version else None,
    )
    return job.status


@shared_task
def purge_expired_conversations() -> int:
    """Deletes conversations (and their messages, via the FK cascade) whose
    `updated_at` is older than settings.CONVERSATION_RETENTION_DAYS, plus
    their on-disk Hamiltonian state files - Issue #30 Phase 8.6, PII
    auto-purge, run daily via Celery beat (see CELERY_BEAT_SCHEDULE in
    web/settings/base.py). Returns the number of conversations purged.
    """
    # torch-free helper (same file location as inference._state_path) so the
    # beat/worker process running the purge doesn't load the GPU stack just to
    # delete files.
    from hvn_worker.state import state_path as _state_path

    cutoff = timezone.now() - timezone.timedelta(days=settings.CONVERSATION_RETENTION_DAYS)
    expired = Conversation.objects.filter(updated_at__lt=cutoff)

    count = 0
    for conversation in expired:
        state_path = _state_path(conversation.id)
        if os.path.exists(state_path):
            try:
                os.remove(state_path)
            except OSError:
                logger.warning("Failed to remove state file %s during purge", state_path)
        conversation.delete()
        count += 1

    return count
