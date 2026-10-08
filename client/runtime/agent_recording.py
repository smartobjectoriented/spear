"""What an agent turn records: session snapshots, state events, start and completion."""

from __future__ import annotations

from typing import Mapping, Sequence

from harness.checkpoint import CheckpointStatus
from runtime.failure_policy import classify_tool_failure
from runtime.session_store import InFlightOperation, SessionEventType, SessionSnapshot
from models.model_backend import ConversationMessage
from runtime.tracing import EventStatus, EventType
from harness.tool_router import ToolResultEnvelope, ToolResultStatus
from evidence.verification import CompletionVerificationStatus, VerificationEvidence
from runtime.working_state import (
    ActionKind, StateEventType, StateSource, TerminalStatus, VerificationOutcome,
)
from runtime.agent_context import AgentContext, AgentResult


class RecordingMixin:
    """Session snapshots, state events, and the task's start and completion."""

    @staticmethod
    def _persist_session(
        context: AgentContext, event_type: SessionEventType,
        payload: Mapping[str, object], *,
        in_flight: InFlightOperation | None = None,
        conversation: Sequence[ConversationMessage] | None = None,
    ) -> None:
        if context.session is None:
            return

        context.session.append(event_type, context.task_id, payload)

        # The snapshot is written after every event, so a resume can rebuild
        # the whole runtime from the last one alone.

        snapshot = SessionSnapshot(
            context.session.session_id, context.task_id, context.session.sequence,
            context.session.configuration, context.working_state,
            tuple(conversation or context.conversation), context.context_items,
            context.compaction_artifact, tuple(sorted(context.result_references)),
            in_flight, tuple(context.session_tool_log),
            tuple(context.session_trajectory), context.progress_monitor.to_dict(),
            context.checkpoint.checkpoint_id if context.checkpoint else None,
            context.checkpoint.status.value if context.checkpoint else None,
            tuple(sorted(context.checkpoint.paths)) if context.checkpoint else (),
            tuple(context.exploration_reports),
            tuple(context.review_results),
            context.budget_manager.to_dict() if context.budget_manager else {},
            tuple(context.delegation_results), dict(context.tool_exposure),
            standard_binding=(dict(context.standard_binding)
                              if context.standard_binding else None),
            standard_source_ids_used=tuple(sorted(context.standard_source_ids_used)),
            standard_retrieval_cache_fingerprint=(
                str(context.standard_binding.get("retrieval_fingerprint")
                    or context.standard_binding.get("index_fingerprint"))
                if context.standard_binding else None),
        )

        context.session.save(snapshot)

        context.trace.emit(
            EventType.SESSION_SNAPSHOT, context.task_id,
            session_id=context.session.session_id, status=EventStatus.OK,
            metadata={"sequence": context.session.sequence,
                      "in_flight": in_flight.kind if in_flight else None,
                      "raw_content_recorded": False},
        )

    @staticmethod
    def persist_session_event(
        context: AgentContext, event_type: SessionEventType,
        payload: Mapping[str, object],
    ) -> None:
        """Persist an orchestration boundary through the canonical snapshot path."""
        from runtime.agent_runtime import AgentRuntime

        AgentRuntime._persist_session(context, event_type, payload)

    @staticmethod
    def _record_tool_result(
        context: AgentContext, envelope: ToolResultEnvelope,
    ) -> None:
        generation_before = context.working_state.mutation_generation

        if envelope.status == ToolResultStatus.CANCELLED:
            event_type = StateEventType.ACTION_CANCELLED
        elif envelope.success:
            event_type = StateEventType.ACTION_SUCCEEDED
        else:
            event_type = StateEventType.ACTION_FAILED

        kind = (ActionKind.COMMAND if envelope.category == "command"
                else ActionKind.TOOL)
        failure_category = (None if envelope.success else classify_tool_failure(
            envelope.status.value, envelope.error_category, envelope.exit_code,
        ).value)

        context.apply_state_event(
            event_type,
            source=StateSource.TOOL_RUNTIME,
            action_id=envelope.action_id,
            kind=kind.value,
            name=envelope.tool_name,
            observed_status=("ok" if envelope.success else
                             "cancelled" if envelope.status == ToolResultStatus.CANCELLED
                             else "failed"),
            category=failure_category,
            summary=envelope.error_summary,
            round_number=context.working_state.current_round,
            output_chars=len(envelope.text),
            exit_code=envelope.exit_code,
        )

        if envelope.status == ToolResultStatus.CACHED:
            context.apply_state_event(
                StateEventType.REPEATED_ACTION_RECORDED,
                source=StateSource.TOOL_RUNTIME,
                action_id=envelope.action_id,
            )

        for path in envelope.read_paths:
            context.apply_state_event(
                StateEventType.FILE_READ,
                source=StateSource.TOOL_RUNTIME,
                path=path,
                action_id=envelope.action_id,
            )

        # The tool reports which of the paths it touched did not exist before,
        # which is the only thing distinguishing a creation from an edit here.

        created = frozenset(envelope.metadata.get("created_paths", ()))

        for path in envelope.affected_paths:
            context.apply_state_event(
                (StateEventType.FILE_CREATED if path in created
                 else StateEventType.FILE_MODIFIED),
                source=StateSource.TOOL_RUNTIME,
                path=path,
                action_id=envelope.action_id,
            )

        # A new generation means every verification recorded so far describes
        # code that has just changed, so it is announced as invalidated.

        if context.working_state.mutation_generation > generation_before:
            context.trace.emit(
                EventType.VERIFICATION_INVALIDATED, context.task_id,
                session_id=context.session_id, status=EventStatus.DETECTED,
                action_id=envelope.action_id,
                metadata={"previous_generation": generation_before,
                          "mutation_generation": context.working_state.mutation_generation,
                          "affected_path_count": len(envelope.affected_paths)},
            )

    @staticmethod
    def _record_verification(
        context: AgentContext, evidence: VerificationEvidence,
    ) -> None:
        from runtime.agent_runtime import AgentRuntime

        context.apply_state_event(
            StateEventType.VERIFICATION_RECORDED,
            source=StateSource.TOOL_RUNTIME,
            verification_id=evidence.verification_id,
            kind=evidence.category.value, category=evidence.category.value,
            coverage=evidence.coverage.value, executed=evidence.executed,
            outcome=evidence.outcome.value, action_id=evidence.action_id,
            summary=evidence.summary,
            mutation_generation=evidence.mutation_generation,
            result_reference=evidence.result_reference,
            command=evidence.command, project_bench=evidence.project_bench,
        )

        # Only evidence that says something about the change may be traced as
        # passed or failed. A command nothing could classify was emitting
        # `verification_passed` with category=unknown and coverage=unknown
        # purely because it exited zero -- the completion gate ignored it, but
        # every reader of the trace counted it, and the benchmark counted it
        # as a verification the turn had performed. It is classified instead,
        # which is what it is.

        if not evidence.proves_change:
            event_type = EventType.VERIFICATION_CLASSIFIED
            status = EventStatus.DETECTED
        elif evidence.outcome == VerificationOutcome.PASSED:
            event_type = EventType.VERIFICATION_PASSED
            status = EventStatus.OK
        elif evidence.outcome == VerificationOutcome.FAILED:
            event_type = EventType.VERIFICATION_FAILED
            status = EventStatus.FAILED
        else:
            event_type = EventType.VERIFICATION_CLASSIFIED
            status = EventStatus.DETECTED

        context.trace.emit(
            event_type, context.task_id, status=status,
            action_id=evidence.action_id, session_id=context.session_id,
            metadata={"category": evidence.category.value,
                      "coverage": evidence.coverage.value,
                      "outcome": evidence.outcome.value,
                      "proves_change": evidence.proves_change,
                      "mutation_generation": evidence.mutation_generation,
                      "project_bench": evidence.project_bench,
                      "command_recorded": evidence.command is not None},
        )

        AgentRuntime._persist_session(
            context, SessionEventType.VERIFICATION_RECORDED,
            {"verification_id": evidence.verification_id,
             "category": evidence.category.value,
             "coverage": evidence.coverage.value,
             "outcome": evidence.outcome.value,
             "mutation_generation": evidence.mutation_generation,
             "result_reference": evidence.result_reference},
        )

    def complete(self, context: AgentContext, result: AgentResult) -> AgentResult:
        if result.task_id != context.task_id:
            raise ValueError("AgentResult does not belong to AgentContext")

        evaluation = context.verification_policy.evaluate_completion(
            context.working_state,
        )

        if context.review_required and (
            evaluation.status != CompletionVerificationStatus.VERIFIED
            or not context.working_state.current_review_accepted
        ):
            # Review is an additional current-generation gate.  Keep the task
            # and checkpoint open; the controller may repair, resume, or roll
            # back through its explicit boundaries.

            result.terminal_status = context.working_state.terminal_status
            result.verification_outcome = context.working_state.verification_outcome
            result.completion_deferred = True

            return result

        # A checkpoint is only finalized behind a verified completion; anything
        # less leaves it active so the work can still be rolled back.

        if (context.checkpoint is not None
                and context.checkpoint_manager is not None
                and context.checkpoint.status == CheckpointStatus.ACTIVE
                and evaluation.status == CompletionVerificationStatus.VERIFIED):
            context.checkpoint_manager.finalize(context.checkpoint)

            context.apply_state_event(
                StateEventType.CHECKPOINT_STATUS_UPDATED,
                checkpoint_id=context.checkpoint.checkpoint_id,
                status=context.checkpoint.status.value,
            )
            context.trace.emit(
                EventType.CHECKPOINT_FINALIZED, context.task_id,
                session_id=context.session_id, status=EventStatus.OK,
                metadata={"checkpoint_id": context.checkpoint.checkpoint_id,
                          "mutation_generation": context.working_state.mutation_generation},
            )

        if context.working_state.terminal_status == TerminalStatus.RUNNING:
            context.apply_state_event(
                StateEventType.TASK_COMPLETED, summary="task turn completed",
            )

        # The result was built before these last transitions, so the counters
        # it carries are refreshed from the working state that now owns them.

        result.terminal_status = context.working_state.terminal_status
        result.verification_outcome = context.working_state.verification_outcome
        result.tool_actions = context.working_state.tool_actions
        result.model_calls = context.working_state.model_rounds
        result.completion_deferred = False

        context.trace.emit(
            EventType.TASK_FINISHED, context.task_id,
            status=EventStatus.OK,
            provider=context.provider, model=context.model,
            metadata={
                "tool_call_count": len(result.tool_log),
                "changed_file_count": len(result.changed_paths),
                "response_chars": len(result.final_response),
                "round_budget_exhausted": result.budget_exhausted,
            },
        )

        self._persist_session(
            context, SessionEventType.TASK_COMPLETED,
            {"terminal_reason": result.terminal_reason.value},
        )

        return result

    @staticmethod
    def _start_task(context: AgentContext) -> None:
        from runtime.agent_runtime import AgentRuntime

        if context.trace_started:
            return

        context.trace_started = True

        metadata = dict(context.task_trace_metadata)
        metadata.setdefault("raw_content_recorded", False)

        context.trace.emit(
            EventType.TASK_STARTED, context.task_id,
            session_id=context.session_id,
            status=EventStatus.STARTED,
            provider=context.provider, model=context.model,
            metadata=metadata,
        )

        if context.session is not None:
            resumed = bool(context.task_trace_metadata.get("resumed"))

            # Sequence zero means nothing has ever been appended, which is the
            # one thing that distinguishes a new session from a resumed one.

            if context.session.sequence == 0:
                context.session.append(
                    SessionEventType.SESSION_STARTED, context.task_id,
                    {"provider": context.provider, "model": context.model},
                )

            context.trace.emit(
                EventType.SESSION_RESUMED if resumed else EventType.SESSION_STARTED,
                context.task_id, session_id=context.session.session_id,
                status=EventStatus.OK if resumed else EventStatus.STARTED,
                metadata={"raw_content_recorded": False},
            )

            AgentRuntime._persist_session(
                context, SessionEventType.TASK_STARTED,
                {"resumed": resumed},
            )
