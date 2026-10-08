"""One model call of an agent turn: its context, its compaction, its retries."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Mapping, Sequence

from runtime.cancellation import OperationCancelled
from runtime.failure_policy import Failure, FailureKind
from runtime.session_store import InFlightOperation, SessionEventType
from runtime.compaction import (
    CompactionRequest, CompactionService, ModelBackendSummarizer,
)
from context.context_engine import (
    ContextItem, ContextLayer, ContextRequest, ContextSnapshot, Freshness,
    working_state_context_item,
)
from models.model_backend import (
    ConversationMessage, ModelTurn, StopReason, TextBlock, ToolDefinition,
    ToolResultBlock, ToolUseBlock,
)
from runtime.tracing import EventStatus, EventType, new_action_id
from runtime.budgets import BudgetExceeded, BudgetKind
from runtime.working_state import ActionKind, StateEventType
from runtime.agent_context import AgentContext


_SYSTEM_CONTEXT_LAYERS = {
    ContextLayer.SYSTEM_RULES,
    ContextLayer.PROJECT_RULES,
    ContextLayer.DURABLE_MEMORY,
    ContextLayer.TASK_WORKING_STATE,
    ContextLayer.CONVERSATION_SUMMARY,
    ContextLayer.RETRIEVED_CONTEXT,
}


def _narrowed(tools, names):
    """The tools offered this round, when the round is only allowed one thing.

    Telling a model "fix it now, do not read anything else first" is a
    request, and requests have been measured ineffective here nine times
    over. A build that fails three times running, answered three times with
    another `sed`, is the same measurement again. What the harness controls
    is what it OFFERS: for the one round after a failed build, the only tools
    on the table are the ones that change a file. Reading is not forbidden by
    instruction -- it is simply not available, for that round, and it comes
    back immediately afterwards.
    """
    if not names:
        return tools

    wanted = set(names)
    kept = tuple(tool for tool in tools if tool.name in wanted)

    return kept or tools


# Rounds between two compactions, and the pressure that overrides the wait.
COMPACTION_COOLDOWN_ROUNDS = int(
    os.environ.get("SPEAR_COMPACTION_COOLDOWN", "4"))
COMPACTION_URGENT = float(os.environ.get("SPEAR_COMPACTION_URGENT", "0.92"))


def _message_context_content(message: ConversationMessage) -> tuple[str, bool]:
    parts = []
    text_only = True

    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ToolResultBlock):
            text_only = False
            parts.append(block.content)
        elif isinstance(block, ToolUseBlock):
            text_only = False
            parts.append(block.name + " " + json.dumps(
                dict(block.arguments), ensure_ascii=False, sort_keys=True,
            ))

    return "\n".join(parts), text_only


def _conversation_context_items(
    conversation: Sequence[ConversationMessage],
) -> tuple[ContextItem, ...]:
    items = []
    call_groups: dict[str, str] = {}
    plain_group = None
    count = len(conversation)

    for index, message in enumerate(conversation):
        content, text_only = _message_context_content(message)
        calls = [block for block in message.content if isinstance(block, ToolUseBlock)]
        results = [block for block in message.content if isinstance(block, ToolResultBlock)]

        # The eviction group is what keeps a call and its results together: drop
        # one without the other and the conversation stops being well-formed.

        if calls:
            group = "tool-exchange:" + ",".join(block.id for block in calls)

            for block in calls:
                call_groups[block.id] = group
        elif results:
            groups = sorted({call_groups.get(
                block.tool_call_id, f"tool-result:{block.tool_call_id}"
            ) for block in results})
            group = "+".join(groups)
        else:
            # A plain exchange is grouped from its user message through to the
            # assistant reply that closes it.

            if message.role == "user":
                plain_group = f"conversation-exchange:{index}"

            group = plain_group or f"conversation-message:{index}"

            if message.role == "assistant":
                plain_group = None

        distance = count - index
        freshness = (Freshness.CURRENT if distance <= 3 else
                     Freshness.RECENT if index >= count // 2 else Freshness.STALE)
        is_tool = bool(calls or results)

        items.append(ContextItem(
            item_id=f"conversation:{index}",
            layer=(ContextLayer.TOOL_EVIDENCE if is_tool
                   else ContextLayer.RECENT_CONVERSATION),
            source="tool_protocol" if is_tool else "conversation_history",
            content=content,
            priority=100 if index == count - 1 else min(85, 35 + index),
            freshness=freshness,
            protected=distance <= 3,
            token_overhead=4,
            inclusion_reason=("preserve tool exchange" if is_tool
                              else "recent conversational continuity"),
            reference=(message, content, text_only),
            eviction_group=group,
            truncatable=text_only,
        ))

    return tuple(items)


def _tool_result_ids(messages) -> frozenset[str]:
    """The tool calls whose results these messages carry."""
    return frozenset(
        block.tool_call_id for message in messages
        for block in getattr(message, "content", ()) or ()
        if isinstance(block, ToolResultBlock) and block.content)


def _render_context_snapshot(
    snapshot: ContextSnapshot,
) -> tuple[str, tuple[ConversationMessage, ...]]:
    system = "".join(
        item.content for item in snapshot.selected_items
        if item.layer in _SYSTEM_CONTEXT_LAYERS and item.reference is None
    )
    conversation: list[ConversationMessage] = []

    for item in snapshot.selected_items:
        if item.reference is None:
            continue

        message, original_content, text_only = item.reference

        if text_only and item.content != original_content:
            message = ConversationMessage(message.role, (TextBlock(item.content),),
                                          authored_by=message.authored_by)

        # Merged only within one author. Folding a harness nudge into the
        # operator's question would put the nudge's words back into what the
        # routing reads as the request, which is the whole thing authored_by
        # exists to prevent.
        if (conversation and conversation[-1].role == message.role
                and conversation[-1].authored_by == message.authored_by
                and all(isinstance(block, TextBlock)
                        for block in conversation[-1].content)
                and all(isinstance(block, TextBlock) for block in message.content)):
            prior = "".join(block.text for block in conversation[-1].content)
            current = "".join(block.text for block in message.content)
            conversation[-1] = ConversationMessage(
                message.role, (TextBlock(prior + "\n\n" + current),),
                authored_by=message.authored_by,
            )
        else:
            conversation.append(message)

    return system, tuple(conversation)



class ModelTurnMixin:
    """One model call: its composed context, its compaction, its retries."""

    def _complete_with_retry(self, context: AgentContext, **kwargs: Any) -> ModelTurn:
        # `only_tools` travels with the call rather than living on the context:
        # it applies to ONE round and must not leak into the next.
        attempts = 0

        while True:
            context.cancellation.raise_if_cancelled()

            self._persist_session(
                context, SessionEventType.MODEL_TURN_STARTED,
                {"attempt": attempts + 1},
                in_flight=InFlightOperation("model", new_action_id("model_call"),
                                            "model_complete"),
            )

            try:
                turn = self.complete_model_turn(context, **kwargs)
                context.cancellation.raise_if_cancelled()

                self._persist_session(
                    context, SessionEventType.MODEL_TURN_COMPLETED,
                    {"stop_reason": turn.stop_reason.value},
                )

                return turn
            except OperationCancelled:
                # A cancellation is the user's decision, not a model failure,
                # and must never be retried.

                raise
            except BudgetExceeded:
                # Nor is a spent budget. BudgetExceeded is a RuntimeError, so
                # the handler below took it for a model failure and retried
                # it -- against a budget that raises on every charge. The
                # refund added beside it then handed the exhausted unit back
                # each time, and a turn that should have ended at its wall
                # clock ran for an hour past it. A budget says stop; retrying
                # a stop is not a retry.

                raise
            except Exception as exc:
                decision = context.retry_policy.decide(
                    Failure(FailureKind.MODEL_ERROR, str(exc)[:240]), attempts,
                )

                context.trace.emit(
                    EventType.RETRY_DECISION, context.task_id,
                    session_id=context.session_id,
                    status=(EventStatus.DETECTED if decision.should_retry
                            else EventStatus.FAILED),
                    error_category=FailureKind.MODEL_ERROR.value,
                    metadata={"decision": decision.action.value,
                              "attempt": decision.attempt,
                              "maximum_attempts": decision.maximum_attempts},
                )

                if not decision.should_retry:
                    raise

                attempts += 1

                # The attempt that just failed was charged to the turn's
                # budget and gave it nothing: no assistant message, no tool
                # call, nothing the next round can build on. Charging for it
                # lets a turn run out of room without ever having used it.
                # Refunded here, where the caller is the only one that knows
                # the attempt produced nothing.

                manager = getattr(context, "budget_manager", None)

                if manager is not None:
                    for kind in (BudgetKind.PRIMARY_MODEL_CALLS,
                                 BudgetKind.MODEL_TURNS):
                        manager.refund(kind)

                context.apply_state_event(
                    StateEventType.RETRY_RECORDED,
                    reason=FailureKind.MODEL_ERROR.value,
                )

    def complete_model_turn(
        self, context: AgentContext, *, use_tools: bool,
        conversation: Sequence[ConversationMessage] | None = None,
        on_token: Callable[[], None] | None = None,
        only_tools: Sequence[str] | None = None,
        grace: bool = False,
    ) -> ModelTurn:
        conversation = tuple(conversation or context.conversation)

        snapshot, system, selected_conversation = self.compose_context(
            context, conversation, use_tools=use_tools,
        )

        # Input tokens are charged from the composed snapshot rather than from
        # the provider's own count, which only arrives after the call.

        # `grace` is the one call a spent budget still allows: the tool-less
        # question "what did you just do", asked so a turn that ran out of
        # room hands back its work instead of a blank. Charging it would be
        # charging the budget for reporting that the budget ended.

        if context.budget_manager is not None and not grace:
            context.budget_manager.consume(BudgetKind.PRIMARY_MODEL_CALLS)
            context.budget_manager.consume(BudgetKind.MODEL_TURNS)
            context.budget_manager.consume(
                BudgetKind.INPUT_TOKENS, snapshot.estimated_input_tokens,
                approximate=True,
            )

        self._trace_context(context, snapshot, system, selected_conversation,
                            use_tools)

        action_id = new_action_id("model")

        # Held in a local as well: a capture failure detaches the recorder from
        # the context, and the rest of this call must then skip it too.

        recorder = context.training_recorder

        if recorder is not None:
            try:
                recorder.begin_model_turn(
                    system=system, conversation=selected_conversation,
                    tools=context.tools, role=context.role,
                    purpose="primary_agent" if context.role == "main" else context.role,
                    mutation_generation=context.working_state.mutation_generation,
                    memory_provenance_ids=tuple(
                        item.item_id for item in snapshot.selected_items
                        if item.layer == ContextLayer.DURABLE_MEMORY
                    ),
                    external_context=any(
                        item.source in {"external_web", "search_internet", "web"}
                        for item in snapshot.selected_items
                    ),
                    tools_enabled=use_tools,
                )
            except Exception as exc:
                context.observer.notice("training_capture_error", {
                    "stage": "model_input", "error_category": type(exc).__name__,
                })
                context.training_recorder = None
                recorder = None

        span = context.trace.start_span(
            EventType.MODEL_CALL_STARTED,
            EventType.MODEL_CALL_FINISHED,
            EventType.MODEL_CALL_FAILED,
            context.task_id,
            provider=context.provider,
            model=context.model,
            context_tokens_estimate=snapshot.estimated_input_tokens,
            metadata={
                "purpose": "primary_agent",
                "conversation_message_count": len(selected_conversation),
                "tool_schema_count": len(context.tools) if use_tools else 0,
                "use_tools": use_tools,
            },
        )

        try:
            turn = context.backend.complete(
                system=system,
                conversation=selected_conversation,
                tools=_narrowed(context.tools, only_tools),
                use_tools=use_tools,
                on_token=on_token,
            )
        except Exception as exc:
            if recorder is not None:
                try:
                    recorder.fail_model_turn(
                        exc, model_action_id=action_id,
                        mutation_generation=context.working_state.mutation_generation,
                    )
                except Exception:
                    context.training_recorder = None

            span.fail(exc, provider=context.provider, model=context.model)

            context.apply_state_event(
                StateEventType.ACTION_FAILED,
                action_id=action_id,
                kind=ActionKind.MODEL.value,
                name="model_complete",
                observed_status="failed",
                category=type(exc).__name__,
                summary=str(exc)[:240] or type(exc).__name__,
                round_number=context.working_state.current_round,
            )

            raise

        usage = turn.usage or {}

        # The provider's own output count is preferred; without it the text is
        # charged at a rough four characters per token, flagged as approximate.

        if context.budget_manager is not None:
            output_tokens = _usage_token(usage, "output_tokens", "completion_tokens")

            context.budget_manager.consume(
                BudgetKind.OUTPUT_TOKENS,
                output_tokens if output_tokens is not None else len(turn.text) // 4 + 1,
                approximate=output_tokens is None,
            )

        fields = {
            "provider": context.provider,
            "model": context.model,
            "input_tokens": _usage_token(usage, "input_tokens", "prompt_tokens"),
            "output_tokens": _usage_token(usage, "output_tokens", "completion_tokens"),
            "context_tokens_estimate": snapshot.estimated_input_tokens,
            "metadata": {
                "purpose": "primary_agent",
                "stop_reason": str(turn.stop_reason),
                "tool_call_count": len(turn.tool_calls),
                "response_chars": len(turn.text),
            },
        }

        # A backend that reports an error in the turn rather than raising is
        # still a failed action, and is recorded as one.

        if turn.stop_reason == StopReason.ERROR:
            span.fail(turn.error or "model backend returned an error",
                      error_category="model_error", **fields)

            context.apply_state_event(
                StateEventType.ACTION_FAILED,
                action_id=action_id,
                kind=ActionKind.MODEL.value,
                name="model_complete",
                observed_status="failed",
                category="model_error",
                summary=(turn.error or "model backend returned an error")[:240],
                round_number=context.working_state.current_round,
            )
        else:
            span.finish(**fields)

            context.apply_state_event(
                StateEventType.ACTION_SUCCEEDED,
                action_id=action_id,
                kind=ActionKind.MODEL.value,
                name="model_complete",
                observed_status="ok",
                summary=None,
                round_number=context.working_state.current_round,
            )

        if recorder is not None:
            try:
                if turn.stop_reason == StopReason.ERROR:
                    recorder.fail_model_turn(
                        RuntimeError(turn.error or "model backend returned an error"),
                        model_action_id=action_id,
                        mutation_generation=context.working_state.mutation_generation,
                    )
                else:
                    recorder.complete_model_turn(
                        turn, model_action_id=action_id,
                        mutation_generation=context.working_state.mutation_generation,
                    )
            except Exception as exc:
                context.observer.notice("training_capture_error", {
                    "stage": "model_output", "error_category": type(exc).__name__,
                })
                context.training_recorder = None

        return turn

    def compose_context(
        self, context: AgentContext,
        conversation: Sequence[ConversationMessage], *, use_tools: bool,
    ) -> tuple[ContextSnapshot, str, tuple[ConversationMessage, ...]]:
        # The working-state projection is regenerated every turn, so the stale
        # copy carried in the context items is swapped out rather than added to.

        projection = working_state_context_item(context.working_state)

        items = []
        replaced = False

        for item in context.context_items:
            if item.layer == ContextLayer.TASK_WORKING_STATE:
                items.append(projection)
                replaced = True
            else:
                items.append(item)

        if not replaced:
            items.append(projection)

        items.extend(_conversation_context_items(conversation))

        # Room the reply will need, which the engine must not spend on input.

        output_reserve = context.output_reserve

        if output_reserve is None:
            value = getattr(context.backend, "max_tokens", None)
            output_reserve = value if isinstance(value, int) and value >= 0 else 8192

        request = ContextRequest(
            items=tuple(items),
            context_limit=context.context_limit,
            output_reserve=output_reserve,
            safety_margin=context.safety_margin,
            fixed_input_tokens=self._tool_schema_tokens(context.tools, use_tools) + 4,
        )

        request = self._compact_context(context, request)
        snapshot = context.context_engine.compose(request)
        system, selected = _render_context_snapshot(snapshot)
        context.dropped_tool_results = (
            _tool_result_ids(context.conversation) - _tool_result_ids(selected))

        return snapshot, system, selected

    def _compact_context(
        self, context: AgentContext, request: ContextRequest,
    ) -> ContextRequest:
        artifact = context.compaction_artifact
        service = CompactionService(
            engine=context.context_engine,
            summarizer=ModelBackendSummarizer(
                context.backend, trace=context.trace,
                provider=context.provider, model=context.model,
                budget_manager=context.budget_manager,
            ),
            policy=context.compaction_policy,
            trace=context.trace,
        )

        # The request with the existing summary already folded in: it is what
        # is returned if compaction fails, so a failure costs nothing gained.

        effective = service.apply_artifact(request, artifact) if artifact else request

        # A cooldown, because compaction has no hysteresis of its own. It
        # fires above a fraction of the window, and one round after a
        # successful compaction the context is back above it -- so a turn
        # that crosses the threshold once pays for a compaction EVERY round
        # for the rest of its life. Watched live: twenty seconds of
        # "Compacting context…" against twenty-five of work, round after
        # round, for nothing that had not just been summarised.
        #
        # Skipped only while there is room to skip it. Past the urgent mark
        # the call would be at risk of overflowing, and a compaction that
        # costs twenty seconds is cheaper than a turn that cannot continue.

        round_now = context.working_state.current_round
        since = round_now - getattr(context, "_last_compaction_round", -99)
        limit = max(1, getattr(effective, "context_limit", 0) or 1)
        pressure = context.context_engine.estimate_request_tokens(effective) / limit

        if since < COMPACTION_COOLDOWN_ROUNDS and pressure < COMPACTION_URGENT:
            return effective

        compaction_request = CompactionRequest(
            working_state=context.working_state,
            context_request=effective,
            existing_artifact=artifact,
            trigger_reason="automatic_context_pressure",
        )

        # The cooldown says compaction is ALLOWED now; it does not say the
        # context needs it. That distinction was missing, so the spinner was
        # opened first and the service declined inside it -- returning before
        # any model call, in a millisecond. `_last_compaction_round` starts at
        # -99, so round one always entered, then one round in four for the
        # rest of the turn: a session with 41 rounds and 0.42 of its window in
        # use showed "Compacting context…" about eleven times and compacted
        # nothing. The operator watched the interface alternate between
        # analysing and compacting for two minutes of work that was neither.
        #
        # Asked of the service, not re-derived here: `will_compact` is the
        # same question `compact` asks of the same numbers, so the label and
        # the work cannot disagree.

        if not service.will_compact(compaction_request):
            return effective

        context._last_compaction_round = round_now

        # Compaction is a model call -- up to three of them -- and it ran
        # behind no spinner and no notice at all. Under context pressure it
        # runs before nearly every round, so the turn showed a spinner, then
        # several silent seconds, then a fresh spinner starting again at 0s.
        # Work the operator waits for has to be work the operator can see.

        try:
            with context.observer.model_activity("Compacting context…"):
                result = service.compact(compaction_request)

            if result.succeeded:
                context.compaction_artifact = result.artifact

                self._persist_session(
                    context, SessionEventType.COMPACTION_COMPLETED,
                    {"generation": result.artifact.conversation_summary.generation,
                     "messages_compacted": result.artifact.messages_compacted},
                )

                return service.apply_artifact(request, result.artifact)
        except Exception as exc:
            context.trace.emit(
                EventType.COMPACTION_FAILED, context.task_id,
                status=EventStatus.FAILED,
                error_category=type(exc).__name__, error_summary=str(exc)[:240],
                metadata={"trigger_reason": "automatic_context_pressure",
                          "validation_result": "not_committed",
                          "raw_content_recorded": False},
            )

            # Said once per turn, not once per round. A summarizer that
            # cannot meet its bound fails on every attempt for the rest of
            # the turn -- one trace held twenty-one in a row -- and the
            # context it was called to relieve only grows. Announcing each
            # one would drown the turn; never announcing it hid the fact
            # that every later round was paying for it.

            if not getattr(context, "_compaction_failure_reported", False):
                context._compaction_failure_reported = True
                context.observer.notice(
                    "compaction_failed",
                    {"reason": str(exc)[:160], "trigger": "context_pressure"},
                )

        return effective

    @staticmethod
    def _tool_schema_tokens(
        tools: Sequence[ToolDefinition], use_tools: bool,
    ) -> int:
        if not use_tools:
            return 0

        chars = sum(
            len(tool.name) + len(tool.description)
            + len(json.dumps(tool.input_schema, ensure_ascii=False, sort_keys=True))
            for tool in tools
        )

        return chars // 4 + 1

    @staticmethod
    def _show_intermediate(
        context: AgentContext, transcript: list[str], text: str,
    ) -> None:
        if text:
            context.observer.intermediate_text(text)
            transcript.append(text)

    @staticmethod
    def _trace_context(
        context: AgentContext, snapshot: ContextSnapshot, system: str,
        conversation: Sequence[ConversationMessage], use_tools: bool,
    ) -> None:
        layer_tokens = {
            layer.value: stats.estimated_tokens
            for layer, stats in snapshot.layer_statistics.items()
        }
        layer_items = {
            layer.value: stats.included_items
            for layer, stats in snapshot.layer_statistics.items()
        }

        # What was dropped and what was cut is traced by reason and count, not
        # by item: the content itself never goes into a trace.

        exclusions: dict[str, int] = {}

        for excluded in snapshot.excluded_items:
            exclusions[excluded.reason] = exclusions.get(excluded.reason, 0) + 1

        truncations: dict[str, int] = {}

        for decision in snapshot.truncation_decisions:
            truncations[decision.reason] = truncations.get(decision.reason, 0) + 1

        context.trace.emit(
            EventType.CONTEXT_COMPOSED, context.task_id,
            provider=context.provider, model=context.model,
            context_tokens_estimate=snapshot.estimated_input_tokens,
            metadata={
                "composition_stage": "model_request",
                "system_prompt_chars": len(system),
                "conversation_message_count": len(conversation),
                "tool_schema_count": len(context.tools) if use_tools else 0,
                "configured_context_limit": snapshot.context_limit,
                "configured_output_reserve": snapshot.output_reserve,
                "safety_margin_tokens": snapshot.safety_margin,
                "available_input_tokens": snapshot.available_input_tokens,
                "fixed_input_tokens": snapshot.fixed_input_tokens,
                "tokens_by_layer": layer_tokens,
                "items_by_layer": layer_items,
                "included_item_count": len(snapshot.selected_items),
                "excluded_item_count": len(snapshot.excluded_items),
                "exclusion_reasons": exclusions,
                "truncation_count": len(snapshot.truncation_decisions),
                "truncation_reasons": truncations,
                "overflow_tokens": snapshot.overflow_tokens,
                "token_estimate_approximate": snapshot.estimate_is_approximate,
                "raw_content_recorded": False,
            },
        )


def _usage_token(usage: Mapping[str, int], *names: str) -> int | None:
    """The first of ``names`` the provider reported, since each spells it its own way."""

    for name in names:
        value = usage.get(name)

        if isinstance(value, int) and not isinstance(value, bool):
            return value

    return None
