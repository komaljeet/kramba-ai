"""
In-app training tests (Issue #57).

Covers the staff-only launch endpoint, its config validation, the
one-job-at-a-time concurrency guard, cancellation, and the Celery task body
(progress streaming + ModelVersion registration) - all without a GPU or torch:
the heavy auto_train.run_curriculum is stubbed via sys.modules so the
orchestration/wiring is exercised for real while the GPU work is faked.
"""
import os
import sys
import tempfile
import types
from unittest import mock

from rest_framework import status
from rest_framework.test import APITestCase

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import override_settings
from django.urls import reverse

from api.models import ModelVersion, TrainingJob

User = get_user_model()
STRONG_PASSWORD = "correct-horse-battery-staple-9"

_CKPT_DIR = tempfile.mkdtemp(prefix="hvn_training_test_")


@override_settings(HVN_CHECKPOINT_DIR=_CKPT_DIR)
class TrainingEndpointTests(APITestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(username="tr_user", password=STRONG_PASSWORD)
        self.admin = User.objects.create_user(
            username="tr_admin", password=STRONG_PASSWORD, is_staff=True
        )
        # A base checkpoint file to continue from, registered as a ModelVersion.
        os.makedirs(_CKPT_DIR, exist_ok=True)
        self.base_path = os.path.join(_CKPT_DIR, "base_ckpt.pt")
        with open(self.base_path, "wb") as f:
            f.write(b"base checkpoint bytes")
        self.model_version = ModelVersion.objects.create(
            tag="base-v1", checkpoint_path=self.base_path
        )

    def _login(self, user):
        resp = self.client.post(
            reverse("token_obtain_pair"), {"username": user.username, "password": STRONG_PASSWORD}
        )
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {resp.data['access']}")

    def _valid_body(self, **over):
        body = {
            "from_model_version": str(self.model_version.id),
            "stages": ["python"],
            "epochs_per_stage": 2,
            "precision": "bf16",
        }
        body.update(over)
        return body

    # --- Authorization -----------------------------------------------------

    def test_regular_user_forbidden_list(self):
        self._login(self.user)
        resp = self.client.get(reverse("training-list"))
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)

    def test_regular_user_forbidden_create(self):
        self._login(self.user)
        with mock.patch("api.views.run_training.delay") as delay:
            resp = self.client.post(reverse("training-list"), self._valid_body(), format="json")
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)
        delay.assert_not_called()

    # --- Create / enqueue --------------------------------------------------

    def test_admin_create_enqueues_and_stores_config(self):
        self._login(self.admin)
        with mock.patch("api.views.run_training.delay") as delay:
            delay.return_value = types.SimpleNamespace(id="task-123")
            resp = self.client.post(
                reverse("training-list"),
                self._valid_body(stages=["python", "html"], smoke=True),
                format="json",
            )
        self.assertEqual(resp.status_code, status.HTTP_202_ACCEPTED)
        job = TrainingJob.objects.get(id=resp.data["id"])
        self.assertEqual(job.created_by, self.admin)
        self.assertEqual(job.celery_task_id, "task-123")
        # Stages reordered into canonical curriculum order (html before python).
        self.assertEqual(job.config["stages"], ["html", "python"])
        self.assertEqual(job.config["base_checkpoint_path"], self.base_path)
        # smoke preset applied.
        self.assertEqual(job.config["epochs_per_stage"], 1)
        self.assertEqual(job.config["max_train_chars"], 60_000)
        delay.assert_called_once_with(str(job.id))

    def test_create_rejects_unknown_stage(self):
        self._login(self.admin)
        with mock.patch("api.views.run_training.delay"):
            resp = self.client.post(
                reverse("training-list"), self._valid_body(stages=["rust"]), format="json"
            )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_create_rejects_bad_precision(self):
        self._login(self.admin)
        with mock.patch("api.views.run_training.delay"):
            resp = self.client.post(
                reverse("training-list"), self._valid_body(precision="int4"), format="json"
            )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_only_one_active_job_at_a_time(self):
        self._login(self.admin)
        TrainingJob.objects.create(
            created_by=self.admin, status=TrainingJob.Status.RUNNING, config={}
        )
        with mock.patch("api.views.run_training.delay") as delay:
            resp = self.client.post(reverse("training-list"), self._valid_body(), format="json")
        self.assertEqual(resp.status_code, status.HTTP_409_CONFLICT)
        delay.assert_not_called()

    # --- Cancel ------------------------------------------------------------

    def test_cancel_queued_job_marks_cancelled(self):
        self._login(self.admin)
        job = TrainingJob.objects.create(
            created_by=self.admin, status=TrainingJob.Status.QUEUED,
            celery_task_id="t1", config={},
        )
        with mock.patch("api.views.run_training.AsyncResult") as ar:
            resp = self.client.post(reverse("training-cancel", args=[job.id]))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        job.refresh_from_db()
        self.assertEqual(job.status, TrainingJob.Status.CANCELLED)
        ar.return_value.revoke.assert_called_once()

    def test_cancel_running_job_marks_cancelling(self):
        from django.utils import timezone

        self._login(self.admin)
        job = TrainingJob.objects.create(
            created_by=self.admin, status=TrainingJob.Status.RUNNING,
            celery_task_id="t2", config={}, started_at=timezone.now(),
        )
        with mock.patch("api.views.run_training.AsyncResult"):
            resp = self.client.post(reverse("training-cancel", args=[job.id]))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        job.refresh_from_db()
        self.assertEqual(job.status, TrainingJob.Status.CANCELLING)

    def test_cancel_finished_job_rejected(self):
        self._login(self.admin)
        job = TrainingJob.objects.create(
            created_by=self.admin, status=TrainingJob.Status.SUCCEEDED, config={}
        )
        with mock.patch("api.views.run_training.AsyncResult"):
            resp = self.client.post(reverse("training-cancel", args=[job.id]))
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)


def _fake_auto_train_module():
    """A torch-free stand-in for the auto_train module: run_curriculum drives
    the same progress_cb events the real one emits and writes a fake stage
    checkpoint, so the task's streaming + ModelVersion registration run for real."""
    mod = types.ModuleType("auto_train")

    def run_curriculum(**kwargs):
        out_dir = kwargs["out_dir"]
        os.makedirs(out_dir, exist_ok=True)
        cb = kwargs.get("progress_cb")
        stages = kwargs["stages"]
        if cb:
            cb({"event": "run_start", "stages": stages, "done": [], "todo": stages,
                "base": kwargs["base_checkpoint"]})
        ckpt, completed = None, []
        for i, s in enumerate(stages):
            if cb:
                cb({"event": "stage_start", "stage": s, "index": i, "total": len(stages),
                    "from": "carry"})
            ckpt = os.path.join(out_dir, f"checkpoint_stage_{s}.pt")
            with open(ckpt, "wb") as f:
                f.write(b"stage checkpoint bytes")
            completed.append({"stage": s})
            if cb:
                cb({"event": "stage_complete", "stage": s, "index": i, "total": len(stages),
                    "checkpoint": ckpt, "metrics": {"ppl": 1.0, "bits_per_char": 0.5},
                    "seconds": 0.1})
        if cb:
            cb({"event": "run_complete", "completed": len(completed), "total": len(stages),
                "final_checkpoint": ckpt})
        return {"current_checkpoint": ckpt, "completed": completed}

    mod.run_curriculum = run_curriculum
    return mod


@override_settings(HVN_CHECKPOINT_DIR=_CKPT_DIR)
class TrainingTaskTests(APITestCase):
    def setUp(self):
        cache.clear()
        os.makedirs(_CKPT_DIR, exist_ok=True)
        self.admin = User.objects.create_user(
            username="tt_admin", password=STRONG_PASSWORD, is_staff=True
        )
        self.base_path = os.path.join(_CKPT_DIR, "task_base.pt")
        with open(self.base_path, "wb") as f:
            f.write(b"base")

    def _make_job(self, **cfg):
        config = {
            "base_checkpoint_path": self.base_path,
            "stages": ["python"],
            "epochs_per_stage": 1,
            "seq_len": 64,
            "batch_size": 4,
            "lr": 1e-4,
            "precision": "bf16",
            "download": False,
        }
        config.update(cfg)
        return TrainingJob.objects.create(created_by=self.admin, config=config, stages_total=len(config["stages"]))

    def test_task_registers_model_version_on_success(self):
        from api.tasks import run_training

        job = self._make_job()
        with mock.patch.dict(sys.modules, {"auto_train": _fake_auto_train_module()}):
            run_training.apply(args=[str(job.id)]).get()

        job.refresh_from_db()
        self.assertEqual(job.status, TrainingJob.Status.SUCCEEDED)
        self.assertIsNotNone(job.result_model_version)
        self.assertEqual(job.stages_done, 1)
        # The produced checkpoint is registered and its hash recorded.
        mv = job.result_model_version
        self.assertTrue(os.path.isfile(mv.checkpoint_path))
        self.assertEqual(len(mv.checkpoint_sha256), 64)
        self.assertTrue(ModelVersion.objects.filter(id=mv.id).exists())

    def test_task_cooperative_cancel(self):
        from api.tasks import run_training

        # Simulate a cancel arriving mid-run: the fake orchestrator flips the
        # running job to CANCELLING after run_start, so the progress callback
        # sees it at the first stage boundary and aborts (matching the view
        # setting CANCELLING while the task is already RUNNING).
        mod = types.ModuleType("auto_train")

        def run_curriculum(**kwargs):
            cb = kwargs["progress_cb"]
            cb({"event": "run_start", "stages": kwargs["stages"], "done": [],
                "todo": kwargs["stages"], "base": kwargs["base_checkpoint"]})
            TrainingJob.objects.filter(status=TrainingJob.Status.RUNNING).update(
                status=TrainingJob.Status.CANCELLING
            )
            # This callback re-reads status -> CANCELLING -> raises internally.
            cb({"event": "stage_start", "stage": kwargs["stages"][0], "index": 0,
                "total": len(kwargs["stages"]), "from": "carry"})
            raise AssertionError("run_curriculum should have been aborted before this")

        mod.run_curriculum = run_curriculum

        job = self._make_job()
        with mock.patch.dict(sys.modules, {"auto_train": mod}):
            run_training.apply(args=[str(job.id)]).get()

        job.refresh_from_db()
        self.assertEqual(job.status, TrainingJob.Status.CANCELLED)
        self.assertIsNone(job.result_model_version)

    def test_task_reports_failure(self):
        from api.tasks import run_training

        broken = types.ModuleType("auto_train")

        def boom(**kwargs):
            raise RuntimeError("CUDA out of memory")

        broken.run_curriculum = boom
        job = self._make_job()
        with mock.patch.dict(sys.modules, {"auto_train": broken}):
            run_training.apply(args=[str(job.id)]).get()

        job.refresh_from_db()
        self.assertEqual(job.status, TrainingJob.Status.FAILED)
        self.assertIn("CUDA out of memory", job.error)
