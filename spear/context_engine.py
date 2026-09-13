"""Provider-neutral layered context composition and accounting.

The engine selects bounded context items; it does not know how any provider
serializes messages.  Callers retain typed references and render the selected
items after composition.  Phase A2 performs priority/freshness eviction and
last-resort textual truncation only -- it does not summarize or compact.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Iterable, Mapping, Protocol

from working_state import PlanStepStatus, WorkingState


class ContextError(ValueError):
    pass


class ContextLayer(StrEnum):
    SYSTEM_RULES = "system_rules"
    PROJECT_RULES = "project_rules"
    DURABLE_MEMORY = "durable_memory"
    TASK_WORKING_STATE = "task_working_state"
    CONVERSATION_SUMMARY = "conversation_summary"
    RECENT_CONVERSATION = "recent_conversation"
    RETRIEVED_CONTEXT = "retrieved_context"
    TOOL_EVIDENCE = "tool_evidence"


class Freshness(StrEnum):
    STALE = "stale"
    RECENT = "recent"
    CURRENT = "current"


# Ascending, so a plain `min` over these picks the stalest item to evict.

_FRESHNESS_ORDER = {
    Freshness.STALE: 0,
    Freshness.RECENT: 1,
    Freshness.CURRENT: 2,
}


class TokenEstimator(Protocol):
    approximate: bool

    def estimate(self, content: str) -> int: ...


@dataclass(frozen=True)
class ApproximateTokenEstimator:
    """Roughly N UTF-8 characters per token.

    Four was the historical figure and it is right for prose. It is wrong for
    what an agentic turn actually accumulates: source code, JSON tool results
    and long absolute paths tokenise far worse. Two turns died on a 400 from
    the server -- 65950 and 67506 tokens against a 65536 window -- while this
    estimate still read them as comfortably under the compaction threshold.
    Measured on those two requests, the real count was about 1.5x the
    estimate, so the divisor is 3 and not 4.

    Overestimating costs an earlier summary. Underestimating costs the turn.
    """

    chars_per_token: int = 3
    approximate: bool = True

    def estimate(self, content: str) -> int:
        if not isinstance(content, str):
            raise ContextError("token estimation requires text content")

        if not content:
            return 0

        return len(content) // self.chars_per_token + 1


@dataclass(frozen=True)
class ContextItem:
    item_id: str
    layer: ContextLayer
    source: str
    content: str
    priority: int = 50
    freshness: Freshness = Freshness.RECENT
    protected: bool = False
    token_estimate: int | None = None
    token_overhead: int = 0
    inclusion_reason: str = "available context"
    reference: Any = field(default=None, compare=False, repr=False)
    eviction_group: str | None = None
    truncatable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id.strip():
            raise ContextError("context item id must be a non-empty string")

        if not isinstance(self.layer, ContextLayer):
            raise ContextError("context item layer must be a ContextLayer")

        if not isinstance(self.source, str) or not self.source.strip():
            raise ContextError("context item source must be a non-empty string")

        if not isinstance(self.content, str):
            raise ContextError("context item content must be text")

        if not self.content and self.reference is None:
            raise ContextError("context item needs content or a reference")

        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise ContextError("context item priority must be an integer")

        if not 0 <= self.priority <= 100:
            raise ContextError("context item priority must be between 0 and 100")

        if not isinstance(self.freshness, Freshness):
            raise ContextError("context item freshness must be a Freshness")

        if self.token_estimate is not None and (
            not isinstance(self.token_estimate, int)
            or isinstance(self.token_estimate, bool)
            or self.token_estimate < 0
        ):
            raise ContextError("token estimate must be null or non-negative")

        if (not isinstance(self.token_overhead, int)
                or isinstance(self.token_overhead, bool) or self.token_overhead < 0):
            raise ContextError("token overhead must be non-negative")

        if not isinstance(self.inclusion_reason, str) or not self.inclusion_reason:
            raise ContextError("inclusion reason must be a non-empty string")

        if self.eviction_group is not None and (
            not isinstance(self.eviction_group, str) or not self.eviction_group
        ):
            raise ContextError("eviction group must be null or non-empty text")


@dataclass(frozen=True)
class ContextRequest:
    items: tuple[ContextItem, ...] = ()
    context_limit: int = 32768
    output_reserve: int = 8192
    safety_margin: int = 768
    fixed_input_tokens: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise ContextError("context request items must be a tuple")

        for name in ("context_limit", "output_reserve", "safety_margin",
                     "fixed_input_tokens"):
            value = getattr(self, name)

            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ContextError(f"{name} must be a non-negative integer")

        if self.context_limit < 1:
            raise ContextError("context_limit must be positive")


@dataclass(frozen=True)
class ExcludedContextItem:
    item: ContextItem
    reason: str


@dataclass(frozen=True)
class TruncationDecision:
    item_id: str
    original_tokens: int
    selected_tokens: int
    reason: str


@dataclass(frozen=True)
class LayerStatistics:
    included_items: int = 0
    excluded_items: int = 0
    estimated_tokens: int = 0


@dataclass(frozen=True)
class ContextSnapshot:
    selected_items: tuple[ContextItem, ...]
    excluded_items: tuple[ExcludedContextItem, ...]
    estimated_input_tokens: int
    available_input_tokens: int
    fixed_input_tokens: int
    layer_statistics: Mapping[ContextLayer, LayerStatistics]
    truncation_decisions: tuple[TruncationDecision, ...]
    estimate_is_approximate: bool
    overflow_tokens: int = 0
    context_limit: int = 0
    output_reserve: int = 0
    safety_margin: int = 0

    @property
    def within_budget(self) -> bool:
        return self.overflow_tokens == 0


# What gets dropped first when the window is too small. Old conversation goes
# before tool evidence, which goes before retrieval, and the rules and task
# state that define the work go last -- they are what makes the rest legible.

_EVICTION_TIER = {
    ContextLayer.CONVERSATION_SUMMARY: 0,
    ContextLayer.RECENT_CONVERSATION: 0,
    ContextLayer.TOOL_EVIDENCE: 1,
    ContextLayer.RETRIEVED_CONTEXT: 2,
    ContextLayer.DURABLE_MEMORY: 3,
    ContextLayer.TASK_WORKING_STATE: 4,
    ContextLayer.PROJECT_RULES: 5,
    ContextLayer.SYSTEM_RULES: 6,
}


class ContextEngine:
    """Deterministically select context while preserving caller order."""

    _TRUNCATION_MARKER = "\n…[context truncated to fit budget]"

    def __init__(self, estimator: TokenEstimator | None = None):
        self.estimator = estimator or ApproximateTokenEstimator()

    def estimate_request_tokens(self, request: ContextRequest) -> int:
        """Estimate unselected input size before eviction or truncation."""

        return request.fixed_input_tokens + sum(
            item.token_estimate or 0 for item in self._normalize(request.items)
        )

    def compose(self, request: ContextRequest) -> ContextSnapshot:
        """Fit the request into the window, and say exactly what that cost."""

        normalized = self._normalize(request.items)

        # What is left for input once the reply and the margin are set aside.

        available = max(
            0, request.context_limit - request.output_reserve - request.safety_margin
        )

        # Keyed by original position, so the caller's order survives eviction.

        selected: dict[int, ContextItem] = dict(enumerate(normalized))
        excluded: dict[int, ExcludedContextItem] = {}
        truncations: list[TruncationDecision] = []

        def total() -> int:
            return request.fixed_input_tokens + sum(
                item.token_estimate or 0 for item in selected.values()
            )

        # First pass: drop whole groups, cheapest first, until it fits or
        # nothing is left that may be dropped.

        while total() > available:
            groups = self._evictable_groups(selected)

            if not groups:
                break

            _, indexes = min(groups, key=lambda entry: entry[0])

            for index in indexes:
                item = selected.pop(index)
                excluded[index] = ExcludedContextItem(
                    item, self._exclusion_reason(item.layer)
                )

        # Protected content survives as an item. If protected text alone is
        # oversized, reduce its text as a last resort rather than silently
        # dropping the rule/state source. Typed, non-truncatable references
        # remain intact so provider tool-call/result structure cannot break.

        while total() > available:
            candidates = [
                (index, item) for index, item in selected.items()
                if (item.truncatable and item.content
                    and (item.token_estimate or 0) > item.token_overhead + 1)
            ]

            if not candidates:
                break

            index, item = min(candidates, key=lambda pair: (
                pair[1].priority,
                _FRESHNESS_ORDER[pair[1].freshness],
                -(pair[1].token_estimate or 0),
                pair[0],
            ))
            excess = total() - available
            target = max(1, (item.token_estimate or 0) - excess)
            truncated = self._truncate(
                item.content, max(1, target - item.token_overhead)
            )
            estimate = self.estimator.estimate(truncated) + item.token_overhead

            # Truncation that saves nothing would loop forever; the request
            # is then reported as overflowing rather than silently mangled.

            if estimate >= (item.token_estimate or 0):
                break

            selected[index] = replace(item, content=truncated, token_estimate=estimate)
            truncations.append(TruncationDecision(
                item.item_id, item.token_estimate or 0, estimate,
                "protected_or_required_item_truncated",
            ))

        selected_items = tuple(item for _, item in sorted(selected.items()))
        excluded_items = tuple(item for _, item in sorted(excluded.items()))
        estimated = total()
        statistics = self._layer_statistics(selected_items, excluded_items)

        return ContextSnapshot(
            selected_items=selected_items,
            excluded_items=excluded_items,
            estimated_input_tokens=estimated,
            available_input_tokens=available,
            fixed_input_tokens=request.fixed_input_tokens,
            layer_statistics=statistics,
            truncation_decisions=tuple(truncations),
            estimate_is_approximate=bool(getattr(self.estimator, "approximate", True)),
            overflow_tokens=max(0, estimated - available),
            context_limit=request.context_limit,
            output_reserve=request.output_reserve,
            safety_margin=request.safety_margin,
        )

    def _normalize(self, items: Iterable[ContextItem]) -> tuple[ContextItem, ...]:
        """Validate the items and give each one a token estimate up front."""

        result, ids = [], set()

        for item in items:
            if not isinstance(item, ContextItem):
                raise ContextError("context requests may contain only ContextItem values")

            if item.item_id in ids:
                raise ContextError(f"duplicate context item id: {item.item_id}")

            ids.add(item.item_id)
            estimate = (item.token_estimate if item.token_estimate is not None
                        else self.estimator.estimate(item.content) + item.token_overhead)
            result.append(replace(item, token_estimate=estimate))

        return tuple(result)

    def _evictable_groups(
        self, selected: Mapping[int, ContextItem]
    ) -> list[tuple[tuple[int, int, int, int], tuple[int, ...]]]:
        """Groups that may be dropped, each with the key deciding the order.

        Items sharing an eviction group go together: a tool call and its result
        are only meaningful as a pair, and evicting half would leave the
        conversation malformed for the provider.
        """

        grouped: dict[str, list[tuple[int, ContextItem]]] = {}

        for index, item in selected.items():
            group = item.eviction_group or f"item:{index}"
            grouped.setdefault(group, []).append((index, item))

        result = []

        for members in grouped.values():
            # One protected member protects the whole group.

            if any(item.protected for _, item in members):
                continue

            # A group is as valuable as its most valuable member, so `min`
            # everywhere; the position breaks ties deterministically.

            tier = min(_EVICTION_TIER[item.layer] for _, item in members)
            priority = min(item.priority for _, item in members)
            freshness = min(_FRESHNESS_ORDER[item.freshness] for _, item in members)
            first = min(index for index, _ in members)
            result.append(((tier, priority, freshness, first),
                           tuple(index for index, _ in members)))

        return result

    @staticmethod
    def _exclusion_reason(layer: ContextLayer) -> str:
        if layer in {ContextLayer.RECENT_CONVERSATION,
                     ContextLayer.CONVERSATION_SUMMARY}:
            return "obsolete_conversation_evicted"

        if layer == ContextLayer.TOOL_EVIDENCE:
            return "redundant_tool_evidence_evicted"

        if layer == ContextLayer.RETRIEVED_CONTEXT:
            return "low_value_retrieval_evicted"

        return "lower_priority_item_evicted"

    def _truncate(self, content: str, target_tokens: int) -> str:
        chars_per_token = getattr(self.estimator, "chars_per_token", 4)

        # ApproximateTokenEstimator uses len // chars_per_token + 1, so the
        # largest string that fits N tokens is N*ratio - 1 characters.

        target_chars = max(1, target_tokens * chars_per_token - 1)
        marker = self._TRUNCATION_MARKER

        if target_chars <= len(marker):
            return marker[:target_chars]

        return content[:target_chars - len(marker)] + marker

    @staticmethod
    def _layer_statistics(
        selected: tuple[ContextItem, ...],
        excluded: tuple[ExcludedContextItem, ...],
    ) -> Mapping[ContextLayer, LayerStatistics]:
        values = {layer: LayerStatistics() for layer in ContextLayer}

        for item in selected:
            current = values[item.layer]
            values[item.layer] = LayerStatistics(
                current.included_items + 1,
                current.excluded_items,
                current.estimated_tokens + (item.token_estimate or 0),
            )

        for excluded_item in excluded:
            item = excluded_item.item
            current = values[item.layer]
            values[item.layer] = LayerStatistics(
                current.included_items,
                current.excluded_items + 1,
                current.estimated_tokens,
            )

        return values


def working_state_projection(state: WorkingState) -> str:
    """Return deterministic model-facing task truth, never the event log."""

    if not isinstance(state, WorkingState):
        raise ContextError("working state projection requires WorkingState")

    lines = ["\n\n## Current task state", "", f"Objective: {state.objective}"]

    if state.user_constraints:
        lines.extend(("", "Constraints:"))
        lines.extend(f"- {item}" for item in state.user_constraints)

    if state.acceptance_criteria:
        lines.extend(("", "Acceptance criteria:"))
        lines.extend(f"- {item}" for item in state.acceptance_criteria)

    active = state.current_step
    pending = [step for step in state.plan_steps.values()
               if step.status in {PlanStepStatus.PENDING, PlanStepStatus.BLOCKED}]

    if active is not None or pending:
        lines.extend(("", "Plan:"))

        if active is not None:
            criteria = (f"; done when: {', '.join(active.completion_criteria)}"
                        if active.completion_criteria else "")
            lines.append(f"- [active] {active.description}{criteria}")

        for step in pending:
            dependencies = (f"; depends on: {', '.join(step.dependencies)}"
                            if step.dependencies else "")
            lines.append(f"- [{step.status.value}] {step.description}{dependencies}")

    if state.unresolved_failures:
        lines.extend(("", "Unresolved failures:"))
        lines.extend(
            f"- {failure.category}: {failure.summary}"
            for failure in state.unresolved_failures
        )

    if state.discoveries:
        lines.extend(("", "Grounded discoveries:"))

        for discovery in state.discoveries:
            location = ""

            if discovery.file_path:
                location += f" ({discovery.file_path}"

                if discovery.symbol:
                    location += f": {discovery.symbol}"

                location += ")"

            lines.append(f"- {discovery.summary}{location}")

    if state.verifications:
        latest = state.verifications[-1]
        lines.extend(("", "Verification:"))
        lines.append(
            f"- {latest.kind}: {latest.outcome.value} "
            f"({latest.coverage} coverage, mutation generation "
            f"{latest.mutation_generation}/{state.mutation_generation})"
            + (f" — {latest.summary}" if latest.summary else "")
        )

    if state.checkpoint_id:
        lines.extend(("", "Recovery:"))
        lines.append(f"- checkpoint {state.checkpoint_status or 'unknown'}")

    return "\n".join(lines)


def working_state_context_item(state: WorkingState) -> ContextItem:
    return ContextItem(
        item_id=f"working-state:{state.task_id}",
        layer=ContextLayer.TASK_WORKING_STATE,
        source="working_state_projection",
        content=working_state_projection(state),
        priority=100,
        freshness=Freshness.CURRENT,
        protected=True,
        inclusion_reason="authoritative task truth",
    )
