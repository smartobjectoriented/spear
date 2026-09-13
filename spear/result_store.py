"""Small filesystem-backed store for tool results kept out of model context."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_REFERENCE_RE = re.compile(r"^result_[0-9a-f]{64}$")


class ResultStoreError(ValueError):
    pass


@dataclass(frozen=True)
class ResultReference:
    reference: str
    byte_count: int
    sha256: str
    task_id: str | None = None


class ResultStore:
    """Stores bytes under opaque validated IDs with sidecar metadata."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(
        self, content: str | bytes, *, task_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ResultReference:
        payload = content.encode("utf-8") if isinstance(content, str) else bytes(content)

        # The reference is content-addressed, so storing the same output twice
        # costs nothing. The task id is mixed in so two tasks producing
        # identical output still get references of their own.

        digest = hashlib.sha256(
            (task_id or "").encode("utf-8") + b"\0" + payload
        ).hexdigest()
        reference = "result_" + digest
        record = {
            "schema_version": 1,
            "reference": reference,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
            "task_id": task_id,
            "metadata": _json_safe(metadata or {}),
        }
        self._atomic_write(self._path(reference, ".bin"), payload)
        self._atomic_write(
            self._path(reference, ".json"),
            json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        )

        return ResultReference(reference, len(payload), record["sha256"], task_id)

    def get(self, reference: str, *, task_id: str | None = None) -> bytes:
        metadata = self.metadata(reference)

        # A reference is opaque but guessable in principle, so a caller naming
        # its task may only read results belonging to it or to no task.

        if task_id is not None and metadata.get("task_id") not in (None, task_id):
            raise ResultStoreError("result reference belongs to another task")

        try:
            payload = self._path(reference, ".bin").read_bytes()
        except FileNotFoundError as exc:
            raise KeyError(reference) from exc

        # The stored bytes are re-hashed on every read: a result cited as
        # evidence must be the result that was actually produced.

        if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
            raise ResultStoreError("stored result checksum mismatch")

        return payload

    def get_text(self, reference: str, *, task_id: str | None = None) -> str:
        return self.get(reference, task_id=task_id).decode("utf-8", errors="replace")

    def metadata(self, reference: str) -> dict[str, object]:
        try:
            data = json.loads(self._path(reference, ".json").read_text("utf-8"))
        except FileNotFoundError as exc:
            raise KeyError(reference) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ResultStoreError("malformed result metadata") from exc

        if not isinstance(data, dict) or data.get("reference") != reference:
            raise ResultStoreError("malformed result metadata")

        return data

    def exists(self, reference: str) -> bool:
        try:
            return (self._path(reference, ".bin").is_file()
                    and self._path(reference, ".json").is_file())
        except ResultStoreError:
            return False

    def _path(self, reference: str, suffix: str) -> Path:
        """The file a reference names, refusing anything that is not one."""

        if not isinstance(reference, str) or not _REFERENCE_RE.fullmatch(reference):
            raise ResultStoreError("malformed result reference")

        # Belt and braces: the pattern already excludes separators, but the
        # resolved path is confined to the store as well.

        path = (self.root / (reference + suffix)).resolve()

        if path.parent != self.root:
            raise ResultStoreError("result reference escapes store")

        return path

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        fd, temporary = tempfile.mkstemp(prefix=".result-", dir=path.parent)

        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())

            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _json_safe(value: object) -> object:
    """Metadata coerced to JSON; anything else keeps a bounded repr."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]

    return repr(value)[:240]
