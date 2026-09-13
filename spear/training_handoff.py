"""Single-GPU inference handoff and filesystem resource ownership."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from inference_service import (GPUWorkloadOwner, InferenceServiceController,
                                InferenceServiceError)


class TrainingResourceError(RuntimeError):
    pass


class TrainingResourceLease:
    """An exclusive on-disk claim on the assigned GPU, held by one job."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    def acquire(self, job_id: str, gpu_uuid: str) -> None:
        if not job_id or not gpu_uuid:
            raise TrainingResourceError("lease identity incomplete")

        payload = json.dumps({"job_id": job_id, "gpu_uuid": gpu_uuid}, sort_keys=True) + "\n"

        # O_EXCL is the whole mechanism: the create either wins or fails, so
        # two jobs cannot both believe they hold the card.

        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise TrainingResourceError("assigned GPU lease is already owned") from exc

        try:
            os.write(fd, payload.encode())
            os.fsync(fd)
        finally:
            os.close(fd)

    def release(self, job_id: str) -> None:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingResourceError("resource lease is corrupt") from exc

        # Only the holder may release: dropping someone else's lease would let
        # a second job take a GPU that is still in use.

        if value.get("job_id") != job_id:
            raise TrainingResourceError("lease owner mismatch")

        self.path.unlink()

    def owner(self) -> Mapping[str, object] | None:
        """Who holds the lease, or None if it is free or unreadable."""

        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None


@dataclass(frozen=True)
class HandoffResult:
    ok: bool
    was_running: bool
    stopped_by_job: bool
    restart_required: bool
    message: str


class TrainingResourceHandoff:
    """Take the GPU from the inference service for a training run, and give it back."""

    def __init__(self, controller: InferenceServiceController,
                 lease: TrainingResourceLease | None = None) -> None:
        self.controller = controller
        self.lease = lease

    def acquire(self, job_id: str, gpu_uuid: str) -> HandoffResult:
        """Stop the managed inference service, or refuse and change nothing."""

        result = self.controller.status()

        # Two refusals before anything is touched: a workload that is not ours
        # is never displaced, and neither is one we cannot positively identify.

        if (getattr(self.controller, "owner", GPUWorkloadOwner.BUSY_BY_MANAGED_INFERENCE)
                == GPUWorkloadOwner.BUSY_BY_OTHER_WORKLOAD):
            return HandoffResult(False, False, False, False,
                                 "assigned GPU is busy by another workload")

        if result.status == "UNKNOWN" or not result.identity_verified:
            return HandoffResult(False, False, False, False,
                                 "inference service identity is not verified")

        if self.lease:
            self.lease.acquire(job_id, gpu_uuid)

        # Already stopped: the lease is held, but there is nothing to restart
        # afterwards, so restart_required stays false.

        if result.status == "STOPPED":
            return HandoffResult(True, False, False, False, "inference was already stopped")

        try:
            stopped = self.controller.stop()

            if stopped.status != "STOPPED":
                raise InferenceServiceError("inference did not stop")
        except Exception:
            # The service is still up, so the lease must not be left held: it
            # would block every later job from a card nothing handed over.

            if self.lease:
                try:
                    self.lease.release(job_id)
                except Exception:
                    pass

            return HandoffResult(False, True, False, False, "inference stop failed")

        return HandoffResult(True, True, True, True,
                             "inference stopped and GPU handoff acquired")

    def restore(self, job_id: str, handoff: Mapping[str, object]) -> Mapping[str, object]:
        """Give the GPU back, recording what happened rather than raising.

        Called from job cleanup paths that must complete: a failed restart is
        reported in the returned metadata so the job still closes cleanly.
        """

        # Nothing to undo unless this job is the one that stopped the service.

        if not handoff.get("restart_required") or not handoff.get("stopped_by_job"):
            return {**dict(handoff), "restart_attempted": False,
                    "restart_result": "NOT_REQUIRED"}

        # Marked as attempted before the attempt, so a crash mid-restart is not
        # mistaken later for a restart that was never tried.

        updated = {**dict(handoff), "restart_attempted": True}

        try:
            result = self.controller.start()
            updated["restart_result"] = "READY" if result.health == "READY" else "FAILED"
        except Exception as exc:
            updated["restart_result"] = "FAILED"
            updated["restart_error"] = type(exc).__name__

        # The lease is released whatever the restart did: this job is finished
        # with the card either way.

        if self.lease:
            try:
                self.lease.release(job_id)
            except Exception:
                updated["lease_release"] = "FAILED"

        return updated
