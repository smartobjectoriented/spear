"""Lightweight, provider-neutral runtime tracing for the SPEAR harness.

Tracing is deliberately observational: recorders are append-only, receive no
raw prompts by default, and isolate every serialization or I/O failure from
the agent loop.  The event vocabulary is broader than the Phase 0 emission
sites so later harness components can add events without changing the format.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Protocol


class EventType(StrEnum):
    TASK_STARTED = "task_started"
    TASK_FINISHED = "task_finished"
    TASK_INTERRUPTED = "task_interrupted"
    MODEL_CALL_STARTED = "model_call_started"
    MODEL_CALL_FINISHED = "model_call_finished"
    MODEL_CALL_FAILED = "model_call_failed"
    CONTEXT_COMPOSED = "context_composed"
    COMPACTION_STARTED = "compaction_started"
    COMPACTION_FINISHED = "compaction_finished"
    COMPACTION_FAILED = "compaction_failed"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_FINISHED = "tool_call_finished"
    TOOL_CALL_FAILED = "tool_call_failed"
    FILE_READ = "file_read"
    FILE_MODIFIED = "file_modified"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_FINISHED = "verification_finished"
    VERIFICATION_CLASSIFIED = "verification_classified"
    VERIFICATION_INVALIDATED = "verification_invalidated"
    VERIFICATION_PASSED = "verification_passed"
    VERIFICATION_FAILED = "verification_failed"
    CHECKPOINT_STARTED = "checkpoint_started"
    CHECKPOINT_PATH_CAPTURED = "checkpoint_path_captured"
    CHECKPOINT_FINALIZED = "checkpoint_finalized"
    ROLLBACK_STARTED = "rollback_started"
    ROLLBACK_FINISHED = "rollback_finished"
    ROLLBACK_CONFLICT = "rollback_conflict"
    RETRY = "retry"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REPEATED_ACTION_DETECTED = "repeated_action_detected"
    READ_ONLY_VIOLATION = "read_only_violation"
    WORKING_STATE_UPDATED = "working_state_updated"
    SESSION_STARTED = "session_started"
    SESSION_RESUMED = "session_resumed"
    SESSION_SNAPSHOT = "session_snapshot"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLATION_OBSERVED = "cancellation_observed"
    RETRY_DECISION = "retry_decision"
    STALLED_DETECTED = "stalled_detected"
    EXPLORATION_STARTED = "exploration_started"
    EXPLORATION_FINISHED = "exploration_finished"
    EXPLORATION_FAILED = "exploration_failed"
    REVIEW_STARTED = "review_started"
    REVIEW_FINISHED = "review_finished"
    REVIEW_FAILED = "review_failed"
    REVIEW_REPAIR_REQUESTED = "review_repair_requested"
    REVIEW_ACCEPTED = "review_accepted"
    REVIEW_BLOCKED = "review_blocked"
    PLAN_CREATED = "plan_created"
    PLAN_REVISED = "plan_revised"
    PLAN_STEP_STARTED = "plan_step_started"
    PLAN_STEP_COMPLETED = "plan_step_completed"
    DELEGATION_STARTED = "delegation_started"
    DELEGATION_FINISHED = "delegation_finished"
    BUDGET_ALLOCATED = "budget_allocated"
    TOOL_VIEW_SELECTED = "tool_view_selected"


class EventStatus(StrEnum):
    STARTED = "started"
    OK = "ok"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXHAUSTED = "exhausted"
    DETECTED = "detected"


_SECRET_KEY = re.compile(
    r"(?:^|[_-])(api[_-]?key|authorization|cookie|credential|passwd|password|secret|token)(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|authorization|cookie|password|secret|token)\s*[:=]\s*([^\s,;]+)"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+={0,2}")
_MAX_DEPTH = 8


def new_task_id() -> str:
    """Return an opaque task identifier unrelated to prompt contents."""

    return f"task_{uuid.uuid4().hex}"


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def new_action_id(prefix: str = "action") -> str:
    """An action id carrying a readable prefix, sanitized so it stays one token."""

    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", prefix)[:32] or "action"

    return f"{safe_prefix}_{uuid.uuid4().hex}"


def _redact_string(value: str) -> str:
    value = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
    return _BEARER.sub("Bearer [REDACTED]", value)


def safe_value(value: Any, *, _depth: int = 0, _seen: set[int] | None = None) -> Any:
    """Convert arbitrary metadata to bounded JSON-safe, redacted values."""

    # Depth and cycle limits: metadata comes from callers all over the harness,
    # and a trace must never hang or blow the stack on a structure it is given.

    if _depth > _MAX_DEPTH:
        return "[MAX_DEPTH]"

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return _redact_string(value)

    if isinstance(value, bytes):
        return f"[bytes:{len(value)}]"

    if isinstance(value, Path):
        return str(value)

    seen = _seen if _seen is not None else set()
    identity = id(value)

    if identity in seen:
        return "[CYCLE]"

    if isinstance(value, Mapping):
        seen.add(identity)
        result = {}

        for raw_key, item in value.items():
            try:
                key = str(raw_key)
            except Exception:
                key = f"[{type(raw_key).__name__}]"

            # A key that names a credential drops its value entirely; the
            # string redaction above only catches ones spelled out inline.

            result[key] = (
                "[REDACTED]" if _SECRET_KEY.search(key)
                else safe_value(item, _depth=_depth + 1, _seen=seen)
            )

        seen.discard(identity)

        return result

    if isinstance(value, (list, tuple, set, frozenset)):
        seen.add(identity)
        result = [safe_value(item, _depth=_depth + 1, _seen=seen) for item in value]
        seen.discard(identity)

        return result

    # Anything else is named but not rendered: an unknown object's repr could
    # carry the very content tracing is supposed to stay out of.

    return f"[{type(value).__name__}]"


@dataclass(frozen=True)
class TraceEvent:
    event_type: EventType
    task_id: str
    event_id: str = field(default_factory=new_event_id)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )
    monotonic_ns: int = field(default_factory=time.monotonic_ns)
    session_id: str | None = None
    parent_event_id: str | None = None
    status: EventStatus | None = None
    provider: str | None = None
    model: str | None = None
    tool_name: str | None = None
    action_id: str | None = None
    duration_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    context_tokens_estimate: int | None = None
    error_category: str | None = None
    error_summary: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "event_id": self.event_id,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "monotonic_ns": self.monotonic_ns,
            "event_type": self.event_type.value,
        }

        # Absent fields are omitted rather than written as null, which keeps
        # the JSONL narrow and makes an event's shape say what it carried.

        for name in (
            "session_id", "parent_event_id", "provider", "model", "tool_name",
            "action_id", "duration_ms", "input_tokens", "output_tokens",
            "context_tokens_estimate", "error_category", "error_summary",
        ):
            value = getattr(self, name)

            if value is not None:
                result[name] = safe_value(value)

        if self.status is not None:
            result["status"] = self.status.value

        if self.metadata:
            result["metadata"] = safe_value(self.metadata)

        return result


class TraceRecorder(Protocol):
    def record(self, event: TraceEvent) -> None: ...


class NullTraceRecorder:
    def record(self, event: TraceEvent) -> None:
        return None


class JsonlTraceRecorder:
    """Append one event per line; recording failures are retained, not raised."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.failure_count = 0
        self.last_error: str | None = None
        self._lock = threading.Lock()

    def record(self, event: TraceEvent) -> None:
        # Serialized before the lock is taken, so a slow or failing conversion
        # never holds up another thread's write.

        try:
            line = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"))

            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)

                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
        except Exception as exc:
            # Counted, not raised: losing a trace line must never cost the user
            # their task. The counter is what makes the loss visible.

            self.failure_count += 1
            self.last_error = f"{type(exc).__name__}: {exc}"[:240]


class TraceEmitter:
    """Failure-isolating facade used by production integration points."""

    def __init__(self, recorder: TraceRecorder | None = None):
        self.recorder = recorder or NullTraceRecorder()
        self.enabled = not isinstance(self.recorder, NullTraceRecorder)

    def emit(self, event_type: EventType, task_id: str, **fields: Any) -> TraceEvent | None:
        if not self.enabled:
            return None

        try:
            event = TraceEvent(event_type=event_type, task_id=task_id, **fields)
            self.recorder.record(event)

            return event
        except Exception:
            # A custom recorder or malformed caller metadata is never allowed
            # to affect model calls, tools, or user-visible completion.

            return TraceEvent(event_type=event_type, task_id=task_id)

    def start_span(
        self,
        started_type: EventType,
        finished_type: EventType,
        failed_type: EventType,
        task_id: str,
        **fields: Any,
    ) -> "TraceSpan":
        """Open a started/finished-or-failed pair sharing one action id."""

        if not self.enabled:
            return NullTraceSpan()

        action_id = fields.pop("action_id", None) or new_action_id(started_type.value)
        started = self.emit(
            started_type, task_id, action_id=action_id,
            status=EventStatus.STARTED, **fields,
        )

        return TraceSpan(self, started, finished_type, failed_type)


class NullTraceSpan:
    def finish(self, **fields: Any) -> None:
        return None

    def fail(self, error: BaseException | str, **fields: Any) -> None:
        return None


class TraceSpan:
    def __init__(
        self,
        emitter: TraceEmitter,
        started: TraceEvent,
        finished_type: EventType,
        failed_type: EventType,
    ):
        self.emitter = emitter
        self.started = started
        self.finished_type = finished_type
        self.failed_type = failed_type
        self._started_ns = time.monotonic_ns()
        self._closed = False

    def finish(self, **fields: Any) -> TraceEvent | None:
        # A span closes once: a second finish or fail would report the same
        # work twice and double its duration in any aggregate.

        if self._closed:
            return None

        self._closed = True

        return self.emitter.emit(
            self.finished_type,
            self.started.task_id,
            parent_event_id=self.started.event_id,
            action_id=self.started.action_id,
            duration_ms=(time.monotonic_ns() - self._started_ns) / 1_000_000,
            status=fields.pop("status", EventStatus.OK),
            **fields,
        )

    def fail(self, error: BaseException | str, **fields: Any) -> TraceEvent | None:
        if self._closed:
            return None

        self._closed = True
        category = type(error).__name__ if isinstance(error, BaseException) else "error"

        return self.emitter.emit(
            self.failed_type,
            self.started.task_id,
            parent_event_id=self.started.event_id,
            action_id=self.started.action_id,
            duration_ms=(time.monotonic_ns() - self._started_ns) / 1_000_000,
            status=fields.pop("status", EventStatus.FAILED),
            error_category=fields.pop("error_category", category),
            error_summary=str(error)[:240],
            **fields,
        )


def tracing_enabled(environment: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environment is None else environment
    return env.get("SPEAR_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}


def create_trace_emitter(
    default_path: str | Path, environment: Mapping[str, str] | None = None
) -> TraceEmitter:
    env = os.environ if environment is None else environment

    if not tracing_enabled(env):
        return TraceEmitter(NullTraceRecorder())

    path = env.get("SPEAR_TRACE_FILE") or str(default_path)

    return TraceEmitter(JsonlTraceRecorder(path))
