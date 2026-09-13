"""Provider-neutral hierarchical runtime budget accounting."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


class BudgetKind(StrEnum):
    PRIMARY_MODEL_CALLS = "primary_model_calls"
    AUXILIARY_MODEL_CALLS = "auxiliary_model_calls"
    MODEL_TURNS = "model_turns"
    TOOL_CALLS = "tool_calls"
    COMMAND_EXECUTIONS = "command_executions"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    CHILD_AGENT_CALLS = "child_agent_calls"
    CHILD_MODEL_TURNS = "child_model_turns"
    WALL_TIME_MS = "wall_time_ms"


class BudgetError(RuntimeError):
    pass


class BudgetExceeded(BudgetError):
    def __init__(self, kind: BudgetKind, component: str):
        self.kind = kind
        self.component = component
        super().__init__(f"{component} {kind.value} budget exhausted")


@dataclass(frozen=True)
class BudgetLimit:
    amount: int
    approximate: bool = False

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError("budget limit cannot be negative")


@dataclass
class BudgetManager:
    """Small counter set with an optional parent allocation.

    Token counters may be estimates; that fact is retained in snapshots. Child
    managers debit both their allocation and the parent task budget.
    """

    component: str = "main"
    limits: dict[BudgetKind, BudgetLimit] = field(default_factory=dict)
    consumed: dict[BudgetKind, int] = field(default_factory=dict)
    approximate: set[BudgetKind] = field(default_factory=set)
    children: dict[str, "BudgetManager"] = field(default_factory=dict)
    started_monotonic: float = field(default_factory=time.monotonic, repr=False)
    _parent: "BudgetManager | None" = field(default=None, repr=False, compare=False)
    _wall_offset_ms: int = field(default=0, repr=False, compare=False)

    @classmethod
    def from_legacy(
        cls, *, component: str, model_turns: int, tool_calls: int,
        context_tokens: int | None = None,
    ) -> "BudgetManager":
        """Build a manager from the old two-number round/tool limits.

        The four extra turns cover the calls the runtime makes around the loop
        -- the conclusion turn, a re-prompt -- which the legacy round count
        never included.
        """

        limits = {
            BudgetKind.MODEL_TURNS: BudgetLimit(model_turns + 4),
            BudgetKind.PRIMARY_MODEL_CALLS: BudgetLimit(model_turns + 4),
            BudgetKind.TOOL_CALLS: BudgetLimit(tool_calls),
        }

        if context_tokens:
            limits[BudgetKind.INPUT_TOKENS] = BudgetLimit(context_tokens, True)

        return cls(component=component, limits=limits)

    def consume(
        self, kind: BudgetKind, amount: int = 1, *, approximate: bool = False,
    ) -> None:
        if amount < 0:
            raise ValueError("budget consumption cannot be negative")

        self.check_wall_time()
        current = self.consumed.get(kind, 0)
        limit = self.limits.get(kind)

        if limit is not None and current + amount > limit.amount:
            raise BudgetExceeded(kind, self.component)

        # The parent is debited first, and may itself refuse. Recording our own
        # consumption last keeps the two ledgers from disagreeing on a refusal.
        # A child's model turns count against the parent's CHILD_MODEL_TURNS,
        # not its own turns: delegated work is a separate allowance.

        if self._parent is not None:
            parent_kind = {
                BudgetKind.MODEL_TURNS: BudgetKind.CHILD_MODEL_TURNS,
                BudgetKind.PRIMARY_MODEL_CALLS: BudgetKind.PRIMARY_MODEL_CALLS,
                BudgetKind.AUXILIARY_MODEL_CALLS: BudgetKind.AUXILIARY_MODEL_CALLS,
                BudgetKind.TOOL_CALLS: BudgetKind.TOOL_CALLS,
                BudgetKind.COMMAND_EXECUTIONS: BudgetKind.COMMAND_EXECUTIONS,
                BudgetKind.INPUT_TOKENS: BudgetKind.INPUT_TOKENS,
                BudgetKind.OUTPUT_TOKENS: BudgetKind.OUTPUT_TOKENS,
            }.get(kind)

            if parent_kind is not None:
                self._parent.consume(parent_kind, amount, approximate=approximate)

        self.consumed[kind] = current + amount

        # Once anything estimated has been counted, the whole counter is an
        # estimate, and every snapshot of it says so.

        if approximate or (limit is not None and limit.approximate):
            self.approximate.add(kind)

    def refund(self, kind: BudgetKind, amount: int = 1) -> None:
        """Give back a unit the caller consumed and got nothing for.

        A model call that came back malformed and was retried cost the turn
        a unit of its budget and gave it nothing to work with. Charging for
        it means a turn can run out of room without ever having used it --
        which is how a turn died five rounds past a ceiling it was told it
        had. Only the caller knows an attempt produced nothing, so only the
        caller can refund it; the counter never goes below zero, and a
        refund on a budget with no limit is a no-op like the charge was.
        """
        if kind not in self.limits:
            return

        self.consumed[kind] = max(0, self.consumed.get(kind, 0) - max(0, amount))

        if self._parent is not None:
            self._parent.refund(kind, amount)

    def ensure_available(self, kind: BudgetKind, amount: int = 1) -> None:
        """Fail before an operation without consuming its eventual outcome."""

        if amount < 0:
            raise ValueError("budget check cannot be negative")

        self.check_wall_time()
        limit = self.limits.get(kind)

        if limit is not None and self.consumed.get(kind, 0) + amount > limit.amount:
            raise BudgetExceeded(kind, self.component)

        if self._parent is not None:
            parent_kind = {
                BudgetKind.MODEL_TURNS: BudgetKind.CHILD_MODEL_TURNS,
                BudgetKind.PRIMARY_MODEL_CALLS: BudgetKind.PRIMARY_MODEL_CALLS,
                BudgetKind.AUXILIARY_MODEL_CALLS: BudgetKind.AUXILIARY_MODEL_CALLS,
                BudgetKind.TOOL_CALLS: BudgetKind.TOOL_CALLS,
                BudgetKind.COMMAND_EXECUTIONS: BudgetKind.COMMAND_EXECUTIONS,
                BudgetKind.INPUT_TOKENS: BudgetKind.INPUT_TOKENS,
                BudgetKind.OUTPUT_TOKENS: BudgetKind.OUTPUT_TOKENS,
            }.get(kind)

            if parent_kind is not None:
                self._parent.ensure_available(parent_kind, amount)

    def remaining(self, kind: BudgetKind) -> int | None:
        limit = self.limits.get(kind)
        return None if limit is None else max(0, limit.amount - self.consumed.get(kind, 0))

    def allocate_child(
        self, allocation_id: str, *, component: str,
        limits: Mapping[BudgetKind, BudgetLimit | int],
    ) -> "BudgetManager":
        """Carve a child allowance out of this budget."""

        if not allocation_id or allocation_id in self.children:
            raise BudgetError("child allocation identifier must be unique")

        self.consume(BudgetKind.CHILD_AGENT_CALLS)

        normalized = {
            kind: value if isinstance(value, BudgetLimit) else BudgetLimit(value)
            for kind, value in limits.items()
        }

        # Refused up front rather than part-way through the child's run: an
        # allowance the parent cannot cover would fail at an arbitrary point.

        for child_kind, parent_kind in (
            (BudgetKind.MODEL_TURNS, BudgetKind.CHILD_MODEL_TURNS),
            (BudgetKind.PRIMARY_MODEL_CALLS, BudgetKind.PRIMARY_MODEL_CALLS),
            (BudgetKind.TOOL_CALLS, BudgetKind.TOOL_CALLS),
        ):
            requested = normalized.get(child_kind)
            remaining = self.remaining(parent_kind)

            if requested is not None and remaining is not None and requested.amount > remaining:
                raise BudgetExceeded(parent_kind, self.component)

        child = BudgetManager(component, normalized, _parent=self)
        self.children[allocation_id] = child

        return child

    def check_wall_time(self) -> None:
        limit = self.limits.get(BudgetKind.WALL_TIME_MS)

        if limit is None:
            return

        # The offset carries the time already elapsed before a resume, so a
        # restored budget does not restart the clock at zero.

        elapsed = self._wall_offset_ms + int(
            (time.monotonic() - self.started_monotonic) * 1000
        )
        self.consumed[BudgetKind.WALL_TIME_MS] = elapsed

        if elapsed > limit.amount:
            raise BudgetExceeded(BudgetKind.WALL_TIME_MS, self.component)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "component": self.component,
            "limits": {key.value: {"amount": value.amount,
                                    "approximate": value.approximate}
                       for key, value in self.limits.items()},
            "consumed": {key.value: value for key, value in self.consumed.items()},
            "approximate": sorted(key.value for key in self.approximate),
            "children": {key: value.to_dict() for key, value in self.children.items()},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "BudgetManager":
        if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
            raise BudgetError("unsupported or malformed budget snapshot")

        try:
            limits = {BudgetKind(key): BudgetLimit(
                int(value["amount"]), bool(value.get("approximate", False)),
            ) for key, value in dict(raw.get("limits", {})).items()}
            manager = cls(str(raw["component"]), limits)
            manager.consumed = {BudgetKind(key): int(value)
                                for key, value in dict(raw.get("consumed", {})).items()}
            manager.approximate = {BudgetKind(value)
                                   for value in raw.get("approximate", ())}
            manager._wall_offset_ms = manager.consumed.get(BudgetKind.WALL_TIME_MS, 0)

            for key, value in dict(raw.get("children", {})).items():
                child = cls.from_dict(value)
                child._parent = manager
                manager.children[str(key)] = child

            return manager
        except (KeyError, TypeError, ValueError) as exc:
            raise BudgetError(f"malformed budget snapshot: {exc}") from exc

    def metrics(self) -> dict[str, object]:
        return {
            kind.value: {
                "allocated": self.limits[kind].amount if kind in self.limits else None,
                "consumed": self.consumed.get(kind, 0),
                "remaining": self.remaining(kind),
                "approximate": kind in self.approximate,
            } for kind in BudgetKind
        }
