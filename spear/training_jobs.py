"""Durable execution records for operator-started training experiments."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Mapping


JOB_SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 1
_SAFE_JOB_ID = re.compile(r"ftjob_[0-9a-f]{24}\Z")


class TrainingJobError(RuntimeError):
    pass


class TrainingMethod(StrEnum):
    SFT = "SFT"
    KTO = "KTO"
    DPO = "DPO"


class TrainingJobStatus(StrEnum):
    PREPARED = "PREPARED"
    PREFLIGHT = "PREFLIGHT"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    LOST = "LOST"


_TRANSITIONS = {
    TrainingJobStatus.PREPARED: {TrainingJobStatus.PREFLIGHT, TrainingJobStatus.FAILED},
    TrainingJobStatus.PREFLIGHT: {TrainingJobStatus.STARTING, TrainingJobStatus.FAILED},
    TrainingJobStatus.STARTING: {TrainingJobStatus.RUNNING, TrainingJobStatus.FAILED},
    TrainingJobStatus.RUNNING: {
        TrainingJobStatus.COMPLETED, TrainingJobStatus.FAILED,
        TrainingJobStatus.STOPPING, TrainingJobStatus.LOST,
    },
    TrainingJobStatus.STOPPING: {
        TrainingJobStatus.STOPPED, TrainingJobStatus.FAILED, TrainingJobStatus.LOST,
    },
    TrainingJobStatus.COMPLETED: set(), TrainingJobStatus.FAILED: set(),
    TrainingJobStatus.STOPPED: set(), TrainingJobStatus.LOST: set(),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrainingJob:
    job_id: str
    method: str
    bundle_id: str
    bundle_path: str
    bundle_checksum: str
    training_config_checksum: str
    source_model: str
    source_model_revision: str | None
    dataset_checksums: Mapping[str, str]
    launcher_type: str
    target_host: str
    remote_job_directory: str
    schema_version: int = JOB_SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now)
    status: str = TrainingJobStatus.PREPARED.value
    remote_process_identity: Mapping[str, object] = field(default_factory=dict)
    start_time: str | None = None
    completion_time: str | None = None
    exit_code: int | None = None
    output_reference: str | None = None
    artifact_references: tuple[str, ...] = ()
    logs_reference: str | None = None
    failure_summary: str | None = None
    training_library_metadata: Mapping[str, object] = field(default_factory=dict)
    hardware_metadata: Mapping[str, object] = field(default_factory=dict)
    progress_metrics: Mapping[str, object] = field(default_factory=dict)
    readiness_override: bool = False

    # FT4D handoff metadata is optional for backward-compatible v1 jobs.

    inference_handoff: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        TrainingJobStore.validate_job_id(self.job_id)

        if self.schema_version != JOB_SCHEMA_VERSION:
            raise TrainingJobError("unsupported training job schema")

        if self.method not in {item.value for item in TrainingMethod}:
            raise TrainingJobError("invalid training method")

        if self.status not in {item.value for item in TrainingJobStatus}:
            raise TrainingJobError("invalid training job status")

        if not self.bundle_id or not self.bundle_checksum or not self.training_config_checksum:
            raise TrainingJobError("training job lineage is incomplete")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TrainingJob":
        fields = {name for name in cls.__dataclass_fields__}
        unknown = set(value) - fields

        if unknown:
            raise TrainingJobError(f"unknown training job fields: {sorted(unknown)}")

        payload = dict(value)
        payload["artifact_references"] = tuple(payload.get("artifact_references", ()))

        try:
            return cls(**payload)
        except (TypeError, ValueError) as exc:
            raise TrainingJobError(f"malformed training job: {exc}") from exc


class TrainingJobStore:
    """Filesystem job state with atomic snapshots and a recovery journal."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    @staticmethod
    def validate_job_id(job_id: str) -> str:
        if not isinstance(job_id, str) or not _SAFE_JOB_ID.fullmatch(job_id):
            raise TrainingJobError("unsafe training job id")

        return job_id

    @staticmethod
    def new_job_id(seed: str | None = None) -> str:
        material = os.urandom(32) if seed is None else seed.encode("utf-8")
        return "ftjob_" + hashlib.sha256(material).hexdigest()[:24]

    def _directory(self, job_id: str) -> Path:
        job_id = self.validate_job_id(job_id)
        path = (self.root / job_id).resolve()

        if path.parent != self.root:
            raise TrainingJobError("training job path escaped store")

        return path

    @staticmethod
    def _serialize(value: object) -> bytes:
        return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)

        try:
            os.fchmod(descriptor, 0o600)

            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())

            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def create(self, job: TrainingJob) -> TrainingJob:
        directory = self._directory(job.job_id)
        directory.mkdir(mode=0o700)
        self._atomic_write(directory / "job.json", self._serialize(job.to_dict()))
        self.append_event(job, "finetune_job_prepared")

        return job

    def append_event(self, job: TrainingJob, event: str,
                     details: Mapping[str, object] | None = None) -> None:
        """Append to the journal, carrying a whole snapshot of the job.

        The snapshot is what makes the journal a recovery source: `_recover`
        can rebuild job.json from the last complete record without replaying
        anything.
        """

        directory = self._directory(job.job_id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        record = {"schema_version": EVENT_SCHEMA_VERSION, "timestamp": utc_now(),
                  "event": event, "details": dict(details or {}),
                  "job_snapshot": job.to_dict()}
        lock_fd = os.open(directory / ".events.lock", os.O_RDWR | os.O_CREAT, 0o600)

        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            fd = os.open(directory / "events.jsonl",
                         os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)

            try:
                os.write(fd, self._serialize(record))
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def save(self, job: TrainingJob, *, event: str = "finetune_job_updated",
             details: Mapping[str, object] | None = None) -> TrainingJob:
        directory = self._directory(job.job_id)

        if not directory.is_dir():
            raise TrainingJobError("training job does not exist")

        self._atomic_write(directory / "job.json", self._serialize(job.to_dict()))
        self.append_event(job, event, details)

        return job

    def transition(self, job: TrainingJob, status: TrainingJobStatus,
                   *, event: str | None = None, **updates: object) -> TrainingJob:
        """Move a job to a new status, refusing a transition the model forbids."""

        # Re-entering the same status is allowed: a refresh that only updates
        # progress metrics is not a transition.

        current = TrainingJobStatus(job.status)

        if status != current and status not in _TRANSITIONS[current]:
            raise TrainingJobError(f"invalid training job transition: {current} -> {status}")

        payload = job.to_dict()
        payload.update(updates)
        payload["status"] = status.value

        # Round-tripped rather than mutated in place, so an invalid update is
        # refused by the dataclass before it can be written.

        changed = TrainingJob.from_dict(payload)

        return self.save(changed, event=event or f"finetune_job_{status.value.lower()}")

    def _recover(self, directory: Path) -> TrainingJob:
        """Rebuild job.json from the last usable journal record."""

        # Every line is tried and the LAST valid one wins, so a crash-torn tail
        # costs the most recent event rather than the whole job.

        journal = directory / "events.jsonl"
        latest = None

        try:
            for line in journal.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)

                    if item.get("schema_version") == EVENT_SCHEMA_VERSION:
                        latest = TrainingJob.from_dict(item["job_snapshot"])
                except (json.JSONDecodeError, KeyError, TrainingJobError):
                    continue
        except OSError as exc:
            raise TrainingJobError(f"training job is unrecoverable: {exc}") from exc

        if latest is None:
            raise TrainingJobError("training job is unrecoverable")

        self._atomic_write(directory / "job.json", self._serialize(latest.to_dict()))

        return latest

    def load(self, job_id: str) -> TrainingJob:
        directory = self._directory(job_id)

        # job.json is the fast path; the journal is the fallback whenever it is
        # missing, truncated or no longer parses.

        try:
            return TrainingJob.from_dict(json.loads(
                (directory / "job.json").read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TrainingJobError):
            return self._recover(directory)

    def list(self) -> list[TrainingJob]:
        result = []

        if not self.root.exists():
            return result

        # One unrecoverable job must not make every other job unlistable.

        for directory in sorted(self.root.iterdir()):
            if directory.is_dir() and _SAFE_JOB_ID.fullmatch(directory.name):
                try:
                    result.append(self.load(directory.name))
                except TrainingJobError:
                    continue

        return sorted(result, key=lambda item: (item.created_at, item.job_id), reverse=True)

    def write_remote(self, job_id: str, value: Mapping[str, object]) -> None:
        self._atomic_write(self._directory(job_id) / "remote.json", self._serialize(value))

    def cache_logs(self, job_id: str, content: str, *, maximum_bytes: int = 65536) -> None:
        # The tail, because that is where a training run says what went wrong.

        encoded = content.encode("utf-8", errors="replace")[-maximum_bytes:]
        self._atomic_write(self._directory(job_id) / "logs-cache.txt", encoded)
