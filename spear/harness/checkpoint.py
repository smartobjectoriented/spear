"""Task-owned, Git-independent filesystem checkpoints.

Checkpoint storage is deliberately outside the workspace.  A manager can
only capture and restore paths beneath its configured workspace, and rollback
uses conservative all-or-nothing conflict preflight by default.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence


CHECKPOINT_SCHEMA_VERSION = 1
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{7,95}$")


class CheckpointError(RuntimeError):
    pass


class CheckpointConflict(CheckpointError):
    pass


class CheckpointStatus(StrEnum):
    ACTIVE = "active"
    FINALIZED = "finalized"
    ROLLED_BACK = "rolled_back"
    CONFLICT = "conflict"


class MutationType(StrEnum):
    MODIFIED = "modified"
    CREATED = "created"
    DELETED = "deleted"


class PathRollbackStatus(StrEnum):
    RESTORED = "restored"
    REMOVED_CREATED_FILE = "removed_created_file"
    ALREADY_ORIGINAL = "already_original"
    CONFLICT = "conflict"
    MISSING = "missing"
    ERROR = "error"


class RollbackStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    CONFLICT = "conflict"
    FAILED = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise CheckpointError(f"invalid {label}")

    return value


@dataclass
class CheckpointPath:
    relative_path: str
    original_exists: bool
    original_hash: str | None
    blob_name: str | None
    original_mode: int | None
    post_mutation_hash: str | None = None
    mutation_type: MutationType | None = None
    action_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "original_exists": self.original_exists,
            "original_hash": self.original_hash,
            "blob_name": self.blob_name,
            "original_mode": self.original_mode,
            "post_mutation_hash": self.post_mutation_hash,
            "mutation_type": self.mutation_type.value if self.mutation_type else None,
            "action_id": self.action_id,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CheckpointPath":
        try:
            mutation = raw.get("mutation_type")
            return cls(
                str(raw["relative_path"]), bool(raw["original_exists"]),
                raw.get("original_hash"), raw.get("blob_name"),
                raw.get("original_mode"), raw.get("post_mutation_hash"),
                MutationType(mutation) if mutation else None,
                raw.get("action_id"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(f"malformed checkpoint path: {exc}") from exc


@dataclass
class Checkpoint:
    checkpoint_id: str
    task_id: str
    session_id: str
    created_at: str
    status: CheckpointStatus = CheckpointStatus.ACTIVE
    paths: dict[str, CheckpointPath] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_id": self.checkpoint_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "status": self.status.value,
            "paths": {key: value.to_dict() for key, value in self.paths.items()},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Checkpoint":
        if raw.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError("unsupported checkpoint schema")

        try:
            paths = raw.get("paths", {})

            if not isinstance(paths, Mapping):
                raise TypeError("paths must be an object")

            result = cls(
                _validate_id(raw["checkpoint_id"], "checkpoint_id"),
                _validate_id(raw["task_id"], "task_id"),
                _validate_id(raw["session_id"], "session_id"),
                str(raw["created_at"]), CheckpointStatus(raw["status"]),
                {str(key): CheckpointPath.from_dict(value)
                 for key, value in paths.items()},
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(f"malformed checkpoint metadata: {exc}") from exc

        for key, value in result.paths.items():
            if key != value.relative_path:
                raise CheckpointError("checkpoint path key mismatch")

        return result


@dataclass(frozen=True)
class PathRollbackResult:
    path: str
    status: PathRollbackStatus
    summary: str | None = None


@dataclass(frozen=True)
class RollbackResult:
    checkpoint_id: str
    status: RollbackStatus
    paths: tuple[PathRollbackResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "status": self.status.value,
            "paths": [{"path": item.path, "status": item.status.value,
                       "summary": item.summary} for item in self.paths],
        }


class CheckpointManager:
    """Filesystem-backed checkpoints scoped to one authorized workspace."""

    def __init__(self, storage_root: str | os.PathLike[str],
                 workspace: str | os.PathLike[str],
                 extra_roots: "Sequence[str | os.PathLike[str]]" = ()) -> None:
        self.storage_root = Path(storage_root).resolve()
        self.workspace = Path(workspace).resolve(strict=True)

        if not self.workspace.is_dir():
            raise CheckpointError("workspace must be a directory")

        # The declared corpora are writable roots too, and a mutation there
        # deserves rollback like any other. Without them every edit_file in a
        # secondary root died on "checkpoint target escapes workspace" -- which
        # is why a model kept abandoning edit_file for shell splices in a tree
        # the harness had told it was writable.

        roots = []

        for item in extra_roots:
            try:
                resolved = Path(item).resolve(strict=True)
            except OSError:
                continue                      # a corpus whose tree has gone

            if resolved.is_dir() and resolved != self.workspace:
                roots.append(resolved)

        self.extra_roots = tuple(roots)
        self.storage_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.storage_root, 0o700)

    def begin_checkpoint(self, task_id: str, session_id: str) -> Checkpoint:
        _validate_id(task_id, "task_id")
        _validate_id(session_id, "session_id")

        checkpoint = Checkpoint(
            "checkpoint_" + uuid.uuid4().hex, task_id, session_id, _now(),
        )
        directory = self._directory(checkpoint, create=True)
        (directory / "blobs").mkdir(mode=0o700)
        self._save(checkpoint)

        return checkpoint

    def load(self, checkpoint_id: str, *, task_id: str | None = None,
             session_id: str | None = None) -> Checkpoint:
        _validate_id(checkpoint_id, "checkpoint_id")

        # Checkpoints live under their session, which the caller need not know,
        # so the id is searched for. Two matches would make it ambiguous.

        matches = list(self.storage_root.glob(f"*/{checkpoint_id}/metadata.json"))

        if len(matches) != 1:
            raise CheckpointError("checkpoint not found")

        try:
            checkpoint = Checkpoint.from_dict(json.loads(matches[0].read_text("utf-8")))
        except json.JSONDecodeError as exc:
            raise CheckpointError(f"malformed checkpoint metadata: {exc}") from exc

        if task_id is not None and checkpoint.task_id != task_id:
            raise CheckpointError("checkpoint belongs to another task")

        if session_id is not None and checkpoint.session_id != session_id:
            raise CheckpointError("checkpoint belongs to another session")

        return checkpoint

    def capture(self, checkpoint: Checkpoint, target: str | os.PathLike[str]) -> CheckpointPath:
        """Record a file's pre-task content so it can be restored later."""

        self._require_active(checkpoint)
        path, relative = self._target(target)

        # The FIRST capture is the original: capturing again after an edit
        # would quietly replace the state rollback restores.

        if relative in checkpoint.paths:
            return checkpoint.paths[relative]

        # Only regular files: a symlink or device would make restoring it mean
        # something other than putting the bytes back.

        exists = path.exists()

        if exists and (path.is_symlink() or not path.is_file()):
            raise CheckpointError("only regular workspace files may be captured")

        # Blobs are content-addressed, so capturing identical files costs one
        # copy between them.

        data = path.read_bytes() if exists else b""
        blob_name = None

        if exists:
            blob_name = _digest(data) + ".blob"
            blob = self._directory(checkpoint) / "blobs" / blob_name

            if not blob.exists():
                self._atomic_write(blob, data)

        item = CheckpointPath(
            relative, exists, _digest(data) if exists else None, blob_name,
            (path.stat().st_mode & 0o7777) if exists else None,
        )
        checkpoint.paths[relative] = item
        self._save(checkpoint)

        return item

    def record_mutation(self, checkpoint: Checkpoint, target: str | os.PathLike[str],
                        action_id: str, mutation_type: MutationType | None = None) -> None:
        """Record what a mutation left behind, so rollback can detect interference."""

        self._require_active(checkpoint)
        path, relative = self._target(target)

        item = checkpoint.paths.get(relative)

        # Without a capture there is no original to restore, which is why the
        # tool layer captures before it writes.

        if item is None:
            raise CheckpointError("path was not captured before mutation")

        exists = path.exists()

        if exists and (path.is_symlink() or not path.is_file()):
            raise CheckpointError("mutated path is not a regular file")

        post_hash = _digest(path.read_bytes()) if exists else None

        # Inferred from what the file was and what it now is, unless the caller
        # already knows better.

        if mutation_type is None:
            if not item.original_exists and exists:
                mutation_type = MutationType.CREATED
            elif item.original_exists and not exists:
                mutation_type = MutationType.DELETED
            else:
                mutation_type = MutationType.MODIFIED

        item.post_mutation_hash = post_hash
        item.mutation_type = mutation_type
        item.action_id = action_id
        self._save(checkpoint)

    def finalize(self, checkpoint: Checkpoint) -> None:
        self._require_active(checkpoint)
        checkpoint.status = CheckpointStatus.FINALIZED
        self._save(checkpoint)

    def rollback(self, checkpoint: Checkpoint, *, all_or_nothing: bool = True) -> RollbackResult:
        """Put every mutated file back, all together or not at all by default."""

        # A finalized or already rolled-back checkpoint has nothing to undo. A
        # conflicted one may be retried once the operator has looked at it.

        if checkpoint.status not in {CheckpointStatus.ACTIVE, CheckpointStatus.CONFLICT}:
            raise CheckpointError("checkpoint is not rollback-eligible")

        # Everything is checked before anything is written, so all_or_nothing
        # can refuse the whole rollback without having half-applied it.

        preflight = [self._preflight(checkpoint, item)
                     for item in checkpoint.paths.values() if item.mutation_type]
        conflicts = [item for item in preflight if item.status in {
            PathRollbackStatus.CONFLICT, PathRollbackStatus.MISSING,
            PathRollbackStatus.ERROR,
        }]

        if conflicts and all_or_nothing:
            checkpoint.status = CheckpointStatus.CONFLICT
            self._save(checkpoint)

            return RollbackResult(checkpoint.checkpoint_id, RollbackStatus.CONFLICT,
                                  tuple(preflight))

        results: list[PathRollbackResult] = []

        for item, checked in zip(
            [p for p in checkpoint.paths.values() if p.mutation_type], preflight,
        ):
            if checked.status in {PathRollbackStatus.CONFLICT,
                                  PathRollbackStatus.MISSING,
                                  PathRollbackStatus.ERROR}:
                results.append(checked)
                continue

            try:
                results.append(self._restore(checkpoint, item, checked))
            except (OSError, CheckpointError) as exc:
                results.append(PathRollbackResult(
                    item.relative_path, PathRollbackStatus.ERROR, str(exc)[:240],
                ))

        # SUCCESS demands that every path came back; anything less is reported
        # as the specific way it fell short, never rounded up.

        if all(item.status in {PathRollbackStatus.RESTORED,
                               PathRollbackStatus.REMOVED_CREATED_FILE,
                               PathRollbackStatus.ALREADY_ORIGINAL}
               for item in results):
            overall = RollbackStatus.SUCCESS
            checkpoint.status = CheckpointStatus.ROLLED_BACK
        elif any(item.status == PathRollbackStatus.CONFLICT for item in results):
            overall = RollbackStatus.CONFLICT
            checkpoint.status = CheckpointStatus.CONFLICT
        else:
            overall = RollbackStatus.PARTIAL if results else RollbackStatus.FAILED

        self._save(checkpoint)

        return RollbackResult(checkpoint.checkpoint_id, overall, tuple(results))

    def inspect(self, checkpoint: Checkpoint) -> Mapping[str, Any]:
        return checkpoint.to_dict()

    def original_bytes(self, checkpoint: Checkpoint, relative_path: str) -> bytes | None:
        """Read captured pre-task bytes without exposing checkpoint paths."""

        self._validate_identity(checkpoint)
        item = checkpoint.paths.get(relative_path)

        if item is None:
            raise CheckpointError("path is not captured by checkpoint")

        if not item.original_exists:
            return None

        if not item.blob_name or not re.fullmatch(r"[0-9a-f]{64}\.blob", item.blob_name):
            raise CheckpointError("invalid checkpoint blob reference")

        blob = self._directory(checkpoint) / "blobs" / item.blob_name
        payload = blob.read_bytes()

        if _digest(payload) != item.original_hash:
            raise CheckpointError("checkpoint blob hash mismatch")

        return payload

    def current_bytes(self, checkpoint: Checkpoint, relative_path: str) -> bytes | None:
        """Read current workspace bytes through the manager's scoped path gate."""

        self._validate_identity(checkpoint)

        if relative_path not in checkpoint.paths:
            raise CheckpointError("path is not captured by checkpoint")

        path, resolved = self._target(relative_path)

        if resolved != relative_path:
            raise CheckpointError("checkpoint path mismatch")

        if not path.exists():
            return None

        if path.is_symlink() or not path.is_file():
            raise CheckpointError("current checkpoint target is not a regular file")

        return path.read_bytes()

    def _validate_identity(self, checkpoint: Checkpoint) -> None:
        """Prove the in-memory checkpoint is the one still on disk."""

        loaded = self.load(
            checkpoint.checkpoint_id, task_id=checkpoint.task_id,
            session_id=checkpoint.session_id,
        )

        if loaded.created_at != checkpoint.created_at:
            raise CheckpointError("checkpoint identity mismatch")

    def _preflight(self, checkpoint: Checkpoint,
                   item: CheckpointPath) -> PathRollbackResult:
        """Decide whether one path can be safely restored, without touching it.

        The rule is that rollback undoes THIS task's mutation and nothing else:
        a file that no longer matches what the task left behind was changed by
        someone else, and is reported as a conflict rather than overwritten.
        """

        path, relative = self._target(item.relative_path)

        if relative != item.relative_path:
            return PathRollbackResult(relative, PathRollbackStatus.CONFLICT,
                                      "path identity changed")

        if path.is_symlink():
            return PathRollbackResult(relative, PathRollbackStatus.CONFLICT,
                                      "target became a symlink")

        exists = path.exists()
        current_hash = _digest(path.read_bytes()) if exists and path.is_file() else None

        # Already back where it started, by either route: the file matches its
        # original, or it never existed and still does not.

        if item.original_exists and current_hash == item.original_hash:
            return PathRollbackResult(relative, PathRollbackStatus.ALREADY_ORIGINAL)

        if not item.original_exists and not exists:
            return PathRollbackResult(relative, PathRollbackStatus.ALREADY_ORIGINAL)

        # The task deleted it and it is still gone, which is exactly the state
        # the task left: restorable, not missing.

        if (item.mutation_type == MutationType.DELETED and item.original_exists
                and not exists and item.post_mutation_hash is None):
            return PathRollbackResult(relative, PathRollbackStatus.RESTORED)

        if not exists:
            return PathRollbackResult(relative, PathRollbackStatus.MISSING,
                                      "task-owned post-state is missing")

        if not path.is_file() or current_hash != item.post_mutation_hash:
            return PathRollbackResult(relative, PathRollbackStatus.CONFLICT,
                                      "current file differs from task-owned post-state")

        return PathRollbackResult(relative, PathRollbackStatus.RESTORED)

    def _restore(self, checkpoint: Checkpoint, item: CheckpointPath,
                 checked: PathRollbackResult) -> PathRollbackResult:
        path, _ = self._target(item.relative_path)

        if checked.status == PathRollbackStatus.ALREADY_ORIGINAL:
            return checked

        if not item.original_exists:
            path.unlink()
            return PathRollbackResult(item.relative_path,
                                      PathRollbackStatus.REMOVED_CREATED_FILE)

        if not item.blob_name or not re.fullmatch(r"[0-9a-f]{64}\.blob", item.blob_name):
            raise CheckpointError("invalid checkpoint blob reference")

        blob = self._directory(checkpoint) / "blobs" / item.blob_name
        data = blob.read_bytes()

        if _digest(data) != item.original_hash:
            raise CheckpointError("checkpoint blob hash mismatch")

        self._atomic_write(path, data, mode=item.original_mode)

        return PathRollbackResult(item.relative_path, PathRollbackStatus.RESTORED)

    def _root_for(self, lexical: Path) -> Path:
        """The writable root containing a path; the primary workspace wins ties."""

        for root in (self.workspace, *self.extra_roots):
            try:
                lexical.relative_to(root)
            except ValueError:
                continue

            return root

        raise CheckpointError("checkpoint target escapes workspace")

    def _target(self, target: str | os.PathLike[str]) -> tuple[Path, str]:
        """Resolve a mutation target to (path, key).

        The key is relative for the primary workspace, as it has always been,
        and ABSOLUTE for a declared secondary root -- there is no single base
        the two share, and _target is applied to the stored key again on
        rollback, so an absolute key round-trips through this same function.
        """
        raw = Path(target)
        candidate = raw if raw.is_absolute() else self.workspace / raw

        # Lexical, not resolved: the containing root has to be decided from the
        # path as written, before any symlink could redirect it elsewhere.

        lexical = Path(os.path.abspath(candidate))
        root = self._root_for(lexical)

        # No component between the target and its root may be a symlink, or a
        # capture could be made to read, and a rollback to write, outside it.

        probe = lexical

        while probe != root:
            if probe.is_symlink():
                raise CheckpointError("checkpoint target uses a symlink")

            probe = probe.parent

        resolved = candidate.resolve(strict=False)

        # The checkpoint store holds the originals; letting a task mutate it
        # would let a task destroy its own means of being undone.

        if resolved == self.storage_root or self.storage_root in resolved.parents:
            raise CheckpointError("checkpoint storage is not a mutation target")

        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise CheckpointError("checkpoint target escapes workspace") from exc

        if relative in {"", "."} or relative.startswith("../"):
            raise CheckpointError("invalid checkpoint target")

        return resolved, (relative if root == self.workspace
                          else resolved.as_posix())

    def _directory(self, checkpoint: Checkpoint, create: bool = False) -> Path:
        _validate_id(checkpoint.session_id, "session_id")
        _validate_id(checkpoint.checkpoint_id, "checkpoint_id")
        directory = self.storage_root / checkpoint.session_id / checkpoint.checkpoint_id

        if create:
            directory.mkdir(parents=True, exist_ok=False, mode=0o700)
            os.chmod(directory.parent, 0o700)
            os.chmod(directory, 0o700)

        return directory

    def _save(self, checkpoint: Checkpoint) -> None:
        self._atomic_write(
            self._directory(checkpoint) / "metadata.json",
            json.dumps(checkpoint.to_dict(), sort_keys=True).encode("utf-8"),
        )

    @staticmethod
    def _atomic_write(path: Path, data: bytes, mode: int | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temporary = tempfile.mkstemp(prefix="checkpoint-", dir=path.parent)

        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())

            os.chmod(temporary, mode if mode is not None else 0o600)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _require_active(checkpoint: Checkpoint) -> None:
        if checkpoint.status != CheckpointStatus.ACTIVE:
            raise CheckpointError("checkpoint is not active")
