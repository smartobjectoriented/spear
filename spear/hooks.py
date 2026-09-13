"""Failure-isolated, observer-only lifecycle hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


class ToolLifecycleHook(Protocol):
    """Observer contract: return values are ignored and cannot block execution."""

    def before_tool(self, event: "ToolHookEvent") -> None: ...

    def after_tool(self, event: "ToolHookEvent") -> None: ...

    def on_tool_failure(self, event: "ToolHookEvent") -> None: ...


@dataclass(frozen=True)
class ToolHookEvent:
    phase: str
    task_id: str
    action_id: str
    tool_name: str
    category: str
    argument_keys: tuple[str, ...] = ()
    success: bool | None = None
    status: str | None = None
    duration_seconds: float | None = None
    metadata: Mapping[str, object] | None = None


class HookManager:
    """Runs observer hooks in registration order; hook failures are isolated."""

    def __init__(self, hooks: Sequence[ToolLifecycleHook] = ()) -> None:
        self._hooks = tuple(hooks)
        self.failures: list[tuple[str, str, str]] = []

    def before_tool(self, event: ToolHookEvent) -> None:
        self._notify("before_tool", event)

    def after_tool(self, event: ToolHookEvent) -> None:
        self._notify("after_tool", event)

    def on_tool_failure(self, event: ToolHookEvent) -> None:
        self._notify("on_tool_failure", event)

    def _notify(self, method: str, event: ToolHookEvent) -> None:
        # A hook is an observer: one that raises is recorded and stepped over,
        # so a faulty hook can never stop a tool from running or reporting.

        for hook in self._hooks:
            try:
                getattr(hook, method)(event)
            except Exception as exc:
                self.failures.append((method, type(exc).__name__, str(exc)[:160]))
