"""Small provider-neutral cooperative cancellation primitives."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum


class CancellationScope(StrEnum):
    OPERATION = "operation"
    TASK = "task"


class OperationCancelled(RuntimeError):
    def __init__(self, reason: str, scope: CancellationScope) -> None:
        super().__init__(reason)
        self.reason = reason
        self.scope = scope


@dataclass
class _CancellationState:
    event: threading.Event
    lock: threading.Lock
    reason: str | None = None
    scope: CancellationScope | None = None


class CancellationToken:
    """Read-only view of cancellation state shared with runtime components."""

    def __init__(self, state: _CancellationState | None = None) -> None:
        self._state = state or _CancellationState(threading.Event(), threading.Lock())

    @property
    def is_cancelled(self) -> bool:
        return self._state.event.is_set()

    @property
    def reason(self) -> str | None:
        return self._state.reason

    @property
    def scope(self) -> CancellationScope | None:
        return self._state.scope

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise OperationCancelled(
                self.reason or "operation cancelled",
                self.scope or CancellationScope.TASK,
            )


class CancellationSource:
    def __init__(self) -> None:
        self._state = _CancellationState(threading.Event(), threading.Lock())
        self.token = CancellationToken(self._state)

    def cancel(
        self, reason: str = "cancellation requested",
        scope: CancellationScope = CancellationScope.TASK,
    ) -> bool:
        """Request cancellation once; return true only for the first request."""

        if not isinstance(scope, CancellationScope):
            raise ValueError("scope must be a CancellationScope")

        with self._state.lock:
            if self._state.event.is_set():
                return False

            self._state.reason = str(reason).strip() or "cancellation requested"
            self._state.scope = scope
            self._state.event.set()

            return True


class LinkedCancellationToken(CancellationToken):
    """A child token cancelled by either its parent or its private source.

    Deliberately does not call ``super().__init__``: it holds no state of its
    own, and every property below answers from whichever token is cancelled.
    The link is one-way, so cancelling a child never touches its parent.
    """

    def __init__(self, parent: CancellationToken, local: CancellationToken) -> None:
        self._parent = parent
        self._local = local

    @property
    def is_cancelled(self) -> bool:
        return self._parent.is_cancelled or self._local.is_cancelled

    @property
    def reason(self) -> str | None:
        # The parent wins when both are cancelled: its reason is the one that
        # explains the whole tree stopping.

        token = self._parent if self._parent.is_cancelled else self._local

        return token.reason

    @property
    def scope(self) -> CancellationScope | None:
        token = self._parent if self._parent.is_cancelled else self._local

        return token.scope


def linked_cancellation(
    parent: CancellationToken,
) -> tuple[CancellationSource, LinkedCancellationToken]:
    """Create child-local cancellation linked one-way to a parent token."""

    source = CancellationSource()

    return source, LinkedCancellationToken(parent, source.token)


NEVER_CANCELLED = CancellationToken()
