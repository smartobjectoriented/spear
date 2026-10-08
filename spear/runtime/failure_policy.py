"""Structured failure classification and conservative bounded retry policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


class FailureKind(StrEnum):
    MODEL_ERROR = "model_error"
    INVALID_MODEL_TURN = "invalid_model_turn"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    TOOL_ERROR = "tool_error"
    COMMAND_FAILED = "command_failed"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"
    CONTEXT_OVERFLOW = "context_overflow"
    COMPACTION_FAILED = "compaction_failed"
    VERIFICATION_FAILED = "verification_failed"
    INTERRUPTED = "interrupted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STALLED = "stalled"
    RUNTIME_ERROR = "runtime_error"
    CHECKPOINT_CONFLICT = "checkpoint_conflict"


class RetryAction(StrEnum):
    RETRY = "retry"
    REPROMPT = "reprompt"
    DO_NOT_RETRY = "do_not_retry"
    TERMINATE = "terminate"


@dataclass(frozen=True)
class Failure:
    kind: FailureKind
    summary: str
    action_id: str | None = None
    retryable: bool | None = None


@dataclass(frozen=True)
class RetryDecision:
    action: RetryAction
    attempt: int
    maximum_attempts: int
    reason: str

    @property
    def should_retry(self) -> bool:
        return self.action in {RetryAction.RETRY, RetryAction.REPROMPT}


@dataclass(frozen=True)
class RetryPolicy:
    """Central policy; defaults add no blind tool/permission retries."""

    maximum_attempts: Mapping[FailureKind, int] = field(default_factory=lambda: {
        # Preserve the historical default (surface provider failures). A caller
        # may explicitly opt into a bounded transient-model retry.
        FailureKind.MODEL_ERROR: 0,
        FailureKind.INVALID_MODEL_TURN: 2,
    })

    def decide(self, failure: Failure, prior_attempts: int) -> RetryDecision:
        if prior_attempts < 0:
            raise ValueError("prior_attempts cannot be negative")

        maximum = int(self.maximum_attempts.get(failure.kind, 0))

        if failure.retryable is False:
            maximum = 0

        if prior_attempts < maximum:
            action = (RetryAction.REPROMPT
                      if failure.kind == FailureKind.INVALID_MODEL_TURN
                      else RetryAction.RETRY)
            return RetryDecision(action, prior_attempts + 1, maximum, failure.kind.value)

        terminal = failure.kind in {
            FailureKind.INTERRUPTED, FailureKind.BUDGET_EXHAUSTED,
            FailureKind.STALLED, FailureKind.RUNTIME_ERROR,
        }

        return RetryDecision(
            RetryAction.TERMINATE if terminal else RetryAction.DO_NOT_RETRY,
            prior_attempts, maximum, "retry limit reached" if maximum else "not retryable",
        )


def classify_tool_failure(status: str, category: str | None, exit_code: int | None) -> FailureKind:
    value = (category or status or "").lower()

    if "permission" in value or status == "denied":
        return FailureKind.PERMISSION_DENIED

    if status == "unknown_tool":
        return FailureKind.UNKNOWN_TOOL

    if status == "invalid_arguments":
        return FailureKind.INVALID_TOOL_ARGUMENTS

    if status == "timeout" or "timeout" in value:
        return FailureKind.TIMEOUT

    if exit_code not in (None, 0):
        return FailureKind.COMMAND_FAILED

    return FailureKind.TOOL_ERROR
