"""Append-only resumable task sessions with atomic derived snapshots.

Session records are authoritative for continuation, not for task truth:
WorkingState remains the reducer-owned authority embedded in each snapshot.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from compaction import (
    CompactionArtifact, ConversationSummary, StructuredCompactionState,
    validate_compaction,
)
from budgets import BudgetManager
from context_engine import ContextItem, ContextLayer, Freshness
from model_backend import ConversationMessage, TextBlock, ToolResultBlock, ToolUseBlock
from tracing import safe_value
from working_state import (
    ActionKind, StateEvent, StateEventType, StateSource, TerminalStatus,
    WorkingState,
)


SESSION_SCHEMA_VERSION = 1
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{7,95}$")


class SessionError(RuntimeError):
    pass


class SessionCompatibilityError(SessionError):
    pass


class SessionEventType(StrEnum):
    SESSION_STARTED = "session_started"
    SESSION_RESUMED = "session_resumed"
    TASK_STARTED = "task_started"
    WORKING_STATE_TRANSITION = "working_state_transition"
    MODEL_TURN_STARTED = "model_turn_started"
    MODEL_TURN_COMPLETED = "model_turn_completed"
    TOOL_ACTION_STARTED = "tool_action_started"
    TOOL_ACTION_COMPLETED = "tool_action_completed"
    COMPACTION_COMPLETED = "compaction_completed"
    VERIFICATION_RECORDED = "verification_recorded"
    CHECKPOINT_RECORDED = "checkpoint_recorded"
    ROLLBACK_RECORDED = "rollback_recorded"
    EXPLORATION_COMPLETED = "exploration_completed"
    REVIEW_COMPLETED = "review_completed"
    PLAN_RECORDED = "plan_recorded"
    DELEGATION_COMPLETED = "delegation_completed"
    BUDGET_UPDATED = "budget_updated"
    TASK_INTERRUPTED = "task_interrupted"
    TASK_FAILED = "task_failed"
    TASK_COMPLETED = "task_completed"
    SESSION_CLOSED = "session_closed"
    STANDARD_BINDING_SET = "standard_binding_set"
    STANDARD_BINDING_CLEARED = "standard_binding_cleared"
    STANDARD_SEARCH_COMPLETED = "standard_search_completed"
    STANDARD_SOURCE_FETCHED = "standard_source_fetched"
    STANDARD_INDEX_CHANGED = "standard_index_changed"
    STANDARD_RETRIEVAL_CHANGED = "standard_retrieval_changed"
    STANDARD_BINDING_MISMATCH = "standard_binding_mismatch"


def new_session_id() -> str:
    return "session_" + uuid.uuid4().hex


def _validate_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise SessionError(f"invalid {name}")

    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise SessionError(f"session value is not JSON-compatible: {exc}") from exc


# Exact key names, not a substring match: "token" is redacted, but the token
# ACCOUNTING fields ("input_tokens", "token_estimate") must survive, and a
# substring rule would silently blank the harness's own budget numbers.

_SESSION_SECRET_KEYS = {
    "api_key", "apikey", "authorization", "cookie", "credential",
    "passwd", "password", "secret", "token", "access_token", "refresh_token",
}


def _session_safe(value: Any) -> Any:
    """Redact credentials without mistaking token accounting for a secret."""

    if isinstance(value, str):
        return safe_value(value)

    if isinstance(value, Mapping):
        result = {}

        for raw_key, item in value.items():
            key = str(raw_key)
            normal = key.lower().replace("-", "_")
            result[key] = ("[REDACTED]" if normal in _SESSION_SECRET_KEYS
                           else _session_safe(item))

        return result

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_session_safe(item) for item in value]

    return value


@dataclass(frozen=True)
class SessionConfiguration:
    workspace: str
    project: str
    execution_mode: str
    context_policy_version: str = "context-v1"
    tool_registry_version: str = "tools-v1"
    provider: str | None = None
    model: str | None = None

    def compatible_with(self, current: "SessionConfiguration") -> bool:
        """Provider/model are informational and intentionally resume-compatible."""
        return all(getattr(self, name) == getattr(current, name) for name in (
            "workspace", "project", "execution_mode",
            "context_policy_version", "tool_registry_version",
        ))


@dataclass(frozen=True)
class InFlightOperation:
    kind: str
    operation_id: str
    name: str
    mutating: bool = False


@dataclass(frozen=True)
class SessionEvent:
    event_type: SessionEventType
    session_id: str
    task_id: str
    sequence: int
    payload: Mapping[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: "sessionevt_" + uuid.uuid4().hex)
    timestamp: str = field(default_factory=_now)
    schema_version: int = SESSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_id(self.session_id, "session_id")
        _validate_id(self.task_id, "task_id")

        if self.sequence < 1:
            raise SessionError("event sequence must be positive")

        if not isinstance(self.event_type, SessionEventType):
            raise SessionError("invalid session event type")

        object.__setattr__(self, "payload", _json_copy(_session_safe(dict(self.payload))))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version, "event_id": self.event_id,
            "event_type": self.event_type.value, "session_id": self.session_id,
            "task_id": self.task_id, "sequence": self.sequence,
            "timestamp": self.timestamp, "payload": _json_copy(self.payload),
        }
        return _json_copy(_session_safe(result))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SessionEvent":
        if not isinstance(raw, Mapping) or raw.get("schema_version") != SESSION_SCHEMA_VERSION:
            raise SessionError("unsupported or malformed session event")

        try:
            return cls(
                SessionEventType(raw["event_type"]), raw["session_id"], raw["task_id"],
                raw["sequence"], raw.get("payload", {}), raw["event_id"],
                raw["timestamp"], raw["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionError(f"malformed session event: {exc}") from exc


def _block_to_dict(block: object) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}

    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name,
                "arguments": _json_copy(block.arguments)}

    if isinstance(block, ToolResultBlock):
        return {"type": "tool_result", "tool_call_id": block.tool_call_id,
                "content": block.content, "is_error": block.is_error}

    raise SessionError(f"unsupported conversation block: {type(block).__name__}")


def _message_from_dict(raw: Mapping[str, Any]) -> ConversationMessage:
    blocks = []

    for item in raw.get("content", []):
        if not isinstance(item, Mapping):
            raise SessionError("malformed conversation block")

        kind = item.get("type")

        if kind == "text":
            blocks.append(TextBlock(str(item.get("text", ""))))
        elif kind == "tool_use":
            args = item.get("arguments", {})

            if not isinstance(args, Mapping):
                raise SessionError("tool arguments must be an object")

            blocks.append(ToolUseBlock(str(item["id"]), str(item["name"]), args))
        elif kind == "tool_result":
            blocks.append(ToolResultBlock(str(item["tool_call_id"]),
                                          str(item.get("content", "")),
                                          bool(item.get("is_error", False))))
        else:
            raise SessionError("unknown conversation block type")

    try:
        return ConversationMessage(
            raw["role"], tuple(blocks),
            authored_by=raw.get("authored_by", "operator"))
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionError(f"malformed conversation message: {exc}") from exc


def _artifact_to_dict(artifact: CompactionArtifact | None) -> dict[str, Any] | None:
    if artifact is None:
        return None

    return {
        "structured_state": artifact.structured_state.to_dict(),
        "conversation_summary": asdict(artifact.conversation_summary),
        "compacted_item_ids": sorted(artifact.compacted_item_ids),
        "messages_compacted": artifact.messages_compacted,
        "tool_evidence_compacted": artifact.tool_evidence_compacted,
        "tokens_before": artifact.tokens_before, "tokens_after": artifact.tokens_after,
        "trigger_reason": artifact.trigger_reason, "retries": artifact.retries,
    }


def _artifact_from_dict(raw: Mapping[str, Any] | None, task_id: str) -> CompactionArtifact | None:
    if raw is None:
        return None

    if not isinstance(raw, Mapping):
        raise SessionError("compaction artifact must be an object")

    try:
        structured = StructuredCompactionState.from_dict(raw["structured_state"])
        summary_raw = raw["conversation_summary"]
        summary = ConversationSummary(
            int(summary_raw["generation"]), str(summary_raw["content"]),
            int(summary_raw["token_estimate"]), bool(summary_raw.get("summary_derived", True)),
        )
        artifact = CompactionArtifact(
            structured, summary, frozenset(raw["compacted_item_ids"]),
            int(raw["messages_compacted"]), int(raw["tool_evidence_compacted"]),
            int(raw["tokens_before"]), int(raw["tokens_after"]),
            str(raw["trigger_reason"]), int(raw["retries"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionError(f"malformed compaction artifact: {exc}") from exc

    if artifact.structured_state.task_id != task_id:
        raise SessionError("compaction artifact belongs to another task")

    return artifact


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    task_id: str
    sequence: int
    configuration: SessionConfiguration
    working_state: WorkingState
    conversation: tuple[ConversationMessage, ...]
    context_items: tuple[ContextItem, ...] = ()
    compaction_artifact: CompactionArtifact | None = None
    result_references: tuple[str, ...] = ()
    in_flight: InFlightOperation | None = None
    tool_log: tuple[str, ...] = ()
    trajectory: tuple[Mapping[str, object], ...] = ()
    progress_state: Mapping[str, object] = field(default_factory=dict)
    checkpoint_id: str | None = None
    checkpoint_status: str | None = None
    checkpoint_paths: tuple[str, ...] = ()
    exploration_reports: tuple[Mapping[str, object], ...] = ()
    review_results: tuple[Mapping[str, object], ...] = ()
    budget_state: Mapping[str, object] = field(default_factory=dict)
    delegation_results: tuple[Mapping[str, object], ...] = ()
    tool_exposure: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = SESSION_SCHEMA_VERSION
    standard_binding: Mapping[str, object] | None = None
    standard_source_ids_used: tuple[str, ...] = ()
    standard_retrieval_cache_fingerprint: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.session_id, "session_id")
        _validate_id(self.task_id, "task_id")

        if self.working_state.task_id != self.task_id:
            raise SessionError("snapshot WorkingState belongs to another task")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version, "session_id": self.session_id,
            "task_id": self.task_id, "sequence": self.sequence,
            "configuration": asdict(self.configuration),
            "working_state": self.working_state.to_dict(),
            "conversation": [{"role": msg.role,
                              "authored_by": msg.authored_by,
                              "content": [_block_to_dict(b) for b in msg.content]}
                             for msg in self.conversation],
            "context_items": [{
                "item_id": item.item_id, "layer": item.layer.value,
                "source": item.source, "content": item.content,
                "priority": item.priority, "freshness": item.freshness.value,
                "protected": item.protected, "token_estimate": item.token_estimate,
                "token_overhead": item.token_overhead,
                "inclusion_reason": item.inclusion_reason,
                "eviction_group": item.eviction_group, "truncatable": item.truncatable,
            } for item in self.context_items if item.layer != ContextLayer.TASK_WORKING_STATE],
            "compaction_artifact": _artifact_to_dict(self.compaction_artifact),
            "result_references": list(self.result_references),
            "in_flight": asdict(self.in_flight) if self.in_flight else None,
            "tool_log": list(self.tool_log), "trajectory": _json_copy(self.trajectory),
            "progress_state": _json_copy(self.progress_state),
            "checkpoint": ({"checkpoint_id": self.checkpoint_id,
                            "status": self.checkpoint_status,
                            "affected_paths": list(self.checkpoint_paths)}
                           if self.checkpoint_id else None),
            "exploration_reports": _json_copy(self.exploration_reports),
            "review_results": _json_copy(self.review_results),
            "budget_state": _json_copy(self.budget_state),
            "delegation_results": _json_copy(self.delegation_results),
            "tool_exposure": _json_copy(self.tool_exposure),
            "standard_binding": _json_copy(self.standard_binding),
            "standard_source_ids_used": list(self.standard_source_ids_used),
            "standard_retrieval_cache_fingerprint": (
                self.standard_retrieval_cache_fingerprint),
        }
        return _json_copy(_session_safe(result))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SessionSnapshot":
        if not isinstance(raw, Mapping) or raw.get("schema_version") != SESSION_SCHEMA_VERSION:
            raise SessionError("unsupported or malformed session snapshot")

        # Every field is re-validated: a snapshot is read back after a crash or
        # a version change, and continuing from a malformed one is worse than
        # refusing to resume at all.

        try:
            task_id = raw["task_id"]
            configuration = SessionConfiguration(**raw["configuration"])
            state = WorkingState.from_dict(raw["working_state"])
            conversation = tuple(_message_from_dict(item) for item in raw["conversation"])
            items = tuple(ContextItem(
                item["item_id"], ContextLayer(item["layer"]), item["source"],
                item["content"], item["priority"], Freshness(item["freshness"]),
                item["protected"], item.get("token_estimate"), item.get("token_overhead", 0),
                item["inclusion_reason"], None, item.get("eviction_group"),
                item.get("truncatable", True),
            ) for item in raw.get("context_items", []))
            in_flight_raw = raw.get("in_flight")
            in_flight = InFlightOperation(**in_flight_raw) if in_flight_raw else None

            try:
                artifact = _artifact_from_dict(raw.get("compaction_artifact"), task_id)

                if artifact is not None and not validate_compaction(
                    state, artifact.structured_state,
                    artifact.conversation_summary.content,
                ).valid:
                    artifact = None
            except SessionError:
                # A stale/malformed summary is expendable. WorkingState and raw
                # bounded conversation remain valid continuation sources.

                artifact = None

            checkpoint = raw.get("checkpoint") or {}

            if not isinstance(checkpoint, Mapping):
                raise SessionError("malformed checkpoint reference")

            reports = raw.get("exploration_reports", ())

            if not isinstance(reports, (list, tuple)) or not all(
                isinstance(report, Mapping) for report in reports
            ):
                raise SessionError("malformed exploration reports")

            reviews = raw.get("review_results", ())

            if not isinstance(reviews, (list, tuple)) or not all(
                isinstance(review, Mapping) for review in reviews
            ):
                raise SessionError("malformed review results")

            budget_state = raw.get("budget_state", {})
            delegations = raw.get("delegation_results", ())
            tool_exposure = raw.get("tool_exposure", {})

            if not isinstance(budget_state, Mapping):
                raise SessionError("malformed budget state")

            if budget_state:
                try:
                    BudgetManager.from_dict(budget_state)
                except Exception as exc:
                    raise SessionError(f"malformed budget state: {exc}") from exc

            if not isinstance(delegations, (list, tuple)) or not all(
                isinstance(item, Mapping) for item in delegations
            ):
                raise SessionError("malformed delegation results")

            if not isinstance(tool_exposure, Mapping):
                raise SessionError("malformed tool exposure")

            standard_binding = raw.get("standard_binding")

            if standard_binding is not None:
                if not isinstance(standard_binding, Mapping):
                    raise SessionError("malformed standard binding")

                from standard_schema import StandardBinding
                standard_binding = StandardBinding.from_dict(standard_binding).to_dict()

            standard_source_ids = tuple(raw.get("standard_source_ids_used", ()))

            if not all(isinstance(item, str) for item in standard_source_ids):
                raise SessionError("malformed standard source provenance")

            return cls(
                raw["session_id"], task_id, int(raw["sequence"]), configuration,
                state, conversation, items,
                artifact,
                tuple(raw.get("result_references", ())), in_flight,
                tuple(raw.get("tool_log", ())), tuple(raw.get("trajectory", ())),
                raw.get("progress_state", {}), checkpoint.get("checkpoint_id"),
                checkpoint.get("status"), tuple(checkpoint.get("affected_paths", ())),
                tuple(_json_copy(report) for report in reports),
                tuple(_json_copy(review) for review in reviews),
                _json_copy(budget_state),
                tuple(_json_copy(item) for item in delegations),
                _json_copy(tool_exposure),
                standard_binding=standard_binding,
                standard_source_ids_used=standard_source_ids,
                standard_retrieval_cache_fingerprint=raw.get(
                    "standard_retrieval_cache_fingerprint"),
            )
        except SessionError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionError(f"malformed session snapshot: {exc}") from exc


class SessionStore(Protocol):
    def append(self, event: SessionEvent) -> None: ...

    def events(self, session_id: str, *, after_sequence: int = 0) -> tuple[SessionEvent, ...]: ...

    def save_snapshot(self, snapshot: SessionSnapshot) -> None: ...

    def load_snapshot(self, session_id: str) -> SessionSnapshot: ...


class FileSessionStore:
    """One private directory per opaque session; events JSONL + atomic JSON snapshot."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        self._last_sequence: dict[str, int] = {}

    def _dir(self, session_id: str, *, create: bool = False) -> Path:
        _validate_id(session_id, "session_id")
        path = self.root / session_id

        if create:
            path.mkdir(mode=0o700, parents=False, exist_ok=True)

        return path

    def append(self, event: SessionEvent) -> None:
        """Append one event, repairing a crash-torn tail first."""

        directory = self._dir(event.session_id, create=True)
        path = directory / "events.jsonl"
        data = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"

        with self._lock:
            if path.exists():
                raw = path.read_bytes()

                if raw and not raw.endswith(b"\n"):
                    # Discard only the crash-torn final record before a new
                    # append; complete earlier records were already parsed.

                    with open(path, "r+b") as recovery:
                        recovery.truncate(raw.rfind(b"\n") + 1)

            # The sequence is cached after the first append, so the log is
            # re-read once per process rather than on every event.

            previous = self._last_sequence.get(event.session_id)

            if previous is None:
                existing = self.events(event.session_id)
                previous = existing[-1].sequence if existing else 0

            # A non-increasing sequence would make replay order ambiguous.

            if event.sequence <= previous:
                raise SessionError("session event sequence must increase")

            with open(path, "a", encoding="utf-8") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())

            self._last_sequence[event.session_id] = event.sequence

        os.chmod(path, 0o600)

    def events(self, session_id: str, *, after_sequence: int = 0) -> tuple[SessionEvent, ...]:
        """Read the log, tolerating only a torn final line."""

        path = self._dir(session_id) / "events.jsonl"

        if not path.exists():
            return ()

        raw_lines = path.read_bytes().splitlines(keepends=True)
        result = []
        last_sequence = 0

        for index, raw in enumerate(raw_lines):
            try:
                event = SessionEvent.from_dict(json.loads(raw))
            except (json.JSONDecodeError, UnicodeDecodeError, SessionError):
                # An unterminated LAST line is an interrupted append and is
                # dropped. Damage anywhere else means the log cannot be trusted.

                if index == len(raw_lines) - 1 and not raw.endswith(b"\n"):
                    break

                raise SessionError(f"malformed session event at line {index + 1}")

            if event.session_id != session_id:
                raise SessionError("cross-session event detected")

            if event.sequence <= last_sequence:
                raise SessionError("session event sequence is not increasing")

            last_sequence = event.sequence

            if event.sequence > after_sequence:
                result.append(event)

        return tuple(result)

    def save_snapshot(self, snapshot: SessionSnapshot) -> None:
        """Replace the snapshot atomically; a resume must never read a partial one."""

        directory = self._dir(snapshot.session_id, create=True)
        target = directory / "snapshot.json"
        data = json.dumps(snapshot.to_dict(), ensure_ascii=False, sort_keys=True)

        with self._lock:
            fd, temporary = tempfile.mkstemp(prefix="snapshot-", suffix=".tmp", dir=directory)

            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())

                os.chmod(temporary, 0o600)
                os.replace(temporary, target)

                # The directory is fsynced too, or a crash can lose the rename
                # itself and leave the previous snapshot in place.

                directory_fd = os.open(directory, os.O_RDONLY)

                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    def load_snapshot(self, session_id: str) -> SessionSnapshot:
        path = self._dir(session_id) / "snapshot.json"

        try:
            snapshot = SessionSnapshot.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError as exc:
            raise SessionError("session snapshot not found") from exc
        except json.JSONDecodeError as exc:
            raise SessionError(f"malformed session snapshot: {exc}") from exc

        if snapshot.session_id != session_id:
            raise SessionError("snapshot session ID mismatch")

        return snapshot


@dataclass
class SessionHandle:
    store: SessionStore
    session_id: str
    configuration: SessionConfiguration
    sequence: int = 0

    def append(
        self, event_type: SessionEventType, task_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> SessionEvent:
        self.sequence += 1
        event = SessionEvent(event_type, self.session_id, task_id, self.sequence,
                             payload or {})

        try:
            self.store.append(event)
        except Exception as exc:
            self.sequence -= 1

            if isinstance(exc, SessionError):
                raise

            raise SessionError(f"could not append session event: {exc}") from exc

        return event

    def save(self, snapshot: SessionSnapshot) -> None:
        if snapshot.session_id != self.session_id or snapshot.sequence != self.sequence:
            raise SessionError("snapshot does not match session handle")

        try:
            self.store.save_snapshot(snapshot)
        except Exception as exc:
            if isinstance(exc, SessionError):
                raise

            raise SessionError(f"could not save session snapshot: {exc}") from exc

    def assert_compatible(self, stored: SessionConfiguration) -> None:
        if not stored.compatible_with(self.configuration):
            raise SessionCompatibilityError("workspace/project/runtime configuration mismatch")


def restore_session(
    store: SessionStore, session_id: str, current: SessionConfiguration, *,
    standard_store=None,
) -> tuple[SessionHandle, SessionSnapshot]:
    """Restore a snapshot and replay durable reducer events after it.

    An unmatched tool-start is converted into a cancelled grounded action and
    a synthetic error result.  It is deliberately never re-executed.
    """
    stored = store.load_snapshot(session_id)

    if not stored.configuration.compatible_with(current):
        raise SessionCompatibilityError("workspace/project/runtime configuration mismatch")

    standard_binding = stored.standard_binding
    index_changed = False
    old_index_fingerprint = None
    old_retrieval_fingerprint = None

    if standard_binding is not None and standard_store is not None:
        from standard_schema import StandardBinding
        persisted = StandardBinding.from_dict(standard_binding)

        try:
            available = standard_store.binding(
                persisted.standard_id, persisted.revision,
                bound_at=persisted.bound_at,
            )
        except Exception as exc:
            raise SessionCompatibilityError(
                "Resume blocked: bound canonical normative source is unavailable\n"
                f"Bound standard: {persisted.standard_id}\n"
                f"revision {persisted.revision}\n"
                f"corpus {persisted.corpus_manifest_sha256}\n"
                f"Current standard: unavailable ({exc})") from exc

        if (persisted.pdf_sha256 != available.pdf_sha256
                or persisted.corpus_manifest_sha256 != available.corpus_manifest_sha256
                or persisted.revision != available.revision):
            raise SessionCompatibilityError(
                "Bound standard:\n"
                f"  {persisted.standard_id}\n  revision {persisted.revision}\n"
                f"  corpus {persisted.corpus_manifest_sha256}\n"
                "Current standard:\n"
                f"  {available.standard_id}\n  revision {available.revision}\n"
                f"  corpus {available.corpus_manifest_sha256}\n"
                "Resume blocked: canonical normative source changed")

        if ((persisted.retrieval_fingerprint or persisted.index_fingerprint)
                != (available.retrieval_fingerprint or available.index_fingerprint)):
            index_changed = True
            old_index_fingerprint = persisted.index_fingerprint
            old_retrieval_fingerprint = (
                persisted.retrieval_fingerprint or persisted.index_fingerprint)
            standard_binding = available.to_dict()

    state = WorkingState.from_dict(stored.working_state.to_dict())
    conversation = list(stored.conversation)
    in_flight = stored.in_flight
    sequence = stored.sequence

    for event in store.events(session_id, after_sequence=stored.sequence):
        sequence = event.sequence

        if event.event_type == SessionEventType.WORKING_STATE_TRANSITION:
            raw = event.payload.get("state_event")

            if isinstance(raw, Mapping):
                transition = StateEvent.from_dict(raw)

                if transition.event_id not in state.applied_event_ids:
                    state.apply(transition)
        elif event.event_type == SessionEventType.TOOL_ACTION_STARTED:
            in_flight = InFlightOperation(
                "tool", str(event.payload.get("tool_call_id", "unknown")),
                str(event.payload.get("tool_name", "unknown")),
                bool(event.payload.get("mutating", False)),
            )
        elif event.event_type == SessionEventType.MODEL_TURN_STARTED:
            in_flight = InFlightOperation(
                "model", str(event.event_id), "model_complete", False,
            )
        elif event.event_type in {
            SessionEventType.TOOL_ACTION_COMPLETED,
            SessionEventType.MODEL_TURN_COMPLETED,
        }:
            in_flight = None

    if state.terminal_status == TerminalStatus.INTERRUPTED:
        state.apply(StateEvent.create(
            StateEventType.TASK_RESUMED, state.task_id, StateSource.HARNESS,
        ))

    if in_flight is not None and in_flight.kind == "tool":
        action_id = "resume_interrupted_" + re.sub(
            r"[^A-Za-z0-9_-]", "_", in_flight.operation_id,
        )

        if not any(item.action_id == action_id for item in state.actions):
            state.apply(StateEvent.create(
                StateEventType.ACTION_CANCELLED, state.task_id, StateSource.HARNESS,
                action_id=action_id, kind=ActionKind.TOOL.value,
                name=in_flight.name, observed_status="cancelled",
                category="interrupted",
                summary="tool outcome was not grounded before interruption",
                round_number=state.current_round,
            ))

        conversation.append(ConversationMessage("user", (ToolResultBlock(
            in_flight.operation_id,
            "ERROR: the prior tool operation was interrupted; its outcome is unknown and "
            "it was not automatically repeated.", True,
        ),)))
        in_flight = None

    # A crash between calls in a multi-tool assistant turn may leave later
    # calls without results. Explicitly cancel those calls; never replay them.

    outstanding: list[ToolUseBlock] = []
    answered: set[str] = set()

    for message in conversation:
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                outstanding.append(block)
            elif isinstance(block, ToolResultBlock):
                answered.add(block.tool_call_id)

    missing = [block for block in outstanding if block.id not in answered]

    if missing:
        conversation.append(ConversationMessage("user", tuple(
            ToolResultBlock(
                block.id,
                "ERROR: this tool operation was interrupted; it was not automatically repeated.",
                True,
            ) for block in missing
        )))

    restored = SessionSnapshot(
        session_id, stored.task_id, sequence, current, state, tuple(conversation),
        stored.context_items, stored.compaction_artifact, stored.result_references,
        in_flight, stored.tool_log, stored.trajectory, stored.progress_state,
        stored.checkpoint_id, stored.checkpoint_status, stored.checkpoint_paths,
        stored.exploration_reports, stored.review_results, stored.budget_state,
        stored.delegation_results, stored.tool_exposure,
        standard_binding=standard_binding,
        standard_source_ids_used=stored.standard_source_ids_used,
        standard_retrieval_cache_fingerprint=(
            standard_binding.get("retrieval_fingerprint")
            or standard_binding.get("index_fingerprint")
            if index_changed and isinstance(standard_binding, Mapping)
            else stored.standard_retrieval_cache_fingerprint),
    )
    handle = SessionHandle(store, session_id, current, sequence)

    if index_changed:
        handle.append(SessionEventType.STANDARD_RETRIEVAL_CHANGED, state.task_id, {
            "old_retrieval_fingerprint": old_retrieval_fingerprint,
            "new_retrieval_fingerprint": (
                standard_binding.get("retrieval_fingerprint")
                or standard_binding.get("index_fingerprint")),
            "components": {
                "lexical": standard_binding.get("index_fingerprint"),
                "vector": standard_binding.get("vector_index_fingerprint"),
                "cross_references": standard_binding.get(
                    "cross_reference_index_fingerprint"),
            },
            "canonical_source_unchanged": True,
            "cached_retrieval_invalidated": True,
        })

        if old_index_fingerprint != standard_binding.get("index_fingerprint"):
            handle.append(SessionEventType.STANDARD_INDEX_CHANGED, state.task_id, {
                "old_index_fingerprint": old_index_fingerprint,
                "new_index_fingerprint": standard_binding.get("index_fingerprint"),
                "canonical_source_unchanged": True,
                "cached_retrieval_invalidated": True,
            })

    handle.append(SessionEventType.SESSION_RESUMED, state.task_id,
                  {"from_sequence": stored.sequence})
    restored = SessionSnapshot(
        session_id, stored.task_id, handle.sequence, current, state,
        restored.conversation, restored.context_items, restored.compaction_artifact,
        restored.result_references, restored.in_flight, restored.tool_log,
        restored.trajectory, restored.progress_state, restored.checkpoint_id,
        restored.checkpoint_status, restored.checkpoint_paths,
        restored.exploration_reports, restored.review_results, restored.budget_state,
        restored.delegation_results, restored.tool_exposure,
        standard_binding=standard_binding,
        standard_source_ids_used=restored.standard_source_ids_used,
        standard_retrieval_cache_fingerprint=(
            restored.standard_retrieval_cache_fingerprint),
    )
    handle.save(restored)

    return handle, restored
