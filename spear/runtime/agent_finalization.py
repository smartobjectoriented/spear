"""How an agent turn ends: its tool window closing, and the result it hands back."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from runtime.failure_policy import FailureKind
from models.model_backend import (
    ConversationMessage, ModelTurn, TextBlock, ToolResultBlock, ToolUseBlock,
    textual_tool_call,
)
from runtime.tracing import EventStatus, EventType
from evidence import project_build
from runtime import work_order
from standard.standard_answer_policy import STOPPED_BY_BUDGET
from runtime.agent_context import AgentContext, AgentResult, RuntimeTerminalReason
from runtime.agent_notes import unsupported_change_claim
from runtime.agent_verification import (
    _record_project_verification, project_build_runs, requirement_matrix_note,
    turn_changed_files, unverified_write_note, work_order_gap, write_note,
)


def no_conclusion_note(tool_log) -> str:
    """What a turn that did work and said nothing hands back.

    One wording, used by the loop's own fallback and by the exit-point
    guarantee behind it, so the two cannot drift apart.
    """
    return (
        "I stopped after " + str(len(tool_log)) + " tool call(s) without "
        "reaching a conclusion. What was observed:\n\n"
        + "\n\n".join(tool_log[-3:])[:1500]
        + "\n\nTell me which part to dig into, or narrow the request."
    )


# Where a turn is with its tools. INVESTIGATING: the tools offered may run.
# FINALIZING: none is offered, and the model is told so -- a closed window
# nobody announced was answered with tool calls written out as text. DONE or
# FAILED: the turn has its answer, or has said plainly that it has none.
INVESTIGATING, FINALIZING, DONE, FAILED = "INVESTIGATING", "FINALIZING", "DONE", "FAILED"

#: A FINALIZING turn that writes a tool call instead of answering is asked
#: once more. A second one ends the turn: a third request would not differ.
FINALIZATION_RETRY_LIMIT = 1

FINALIZATION_DEMAND = (
    "No further tools are available for this turn. Give your final answer now, "
    "from the evidence already collected. Do not request or write tool calls: "
    "none can run. Cite only what you have already read; if it is not enough to "
    "answer, say so plainly instead of searching again.")

FINALIZATION_RETRY = (
    "That reply was a tool call, and no tool is available for this turn: it was "
    "not run, and none will be. Answer now from the evidence already collected, "
    "with no tool-call markup of any kind. If that evidence is not enough to "
    "answer, say so -- do not attempt another search.")


@dataclass
class Finalization:
    """One turn's tool window, as a state the runtime and its callers can read."""

    state: str = INVESTIGATING
    reason: str = ""
    retries: int = 0


def finalization_failed_note(context) -> str:
    """What a turn hands back when it never produced an answer."""
    subject = "NORMATIVE ANSWER" if getattr(context, "standard_binding", None) else "ANSWER"

    return (f"{subject} NOT COMPLETED: with no tool available, the model kept "
            "writing tool calls instead of an answer, after being told once more "
            "that none could run. Nothing it wrote was executed, and nothing it "
            "wrote is presented as an answer.")



class FinalizationMixin:
    """The end of a turn: its tool window, its repaired answer, its result."""

    def _inject_evidence_call(self, context: AgentContext, policy,
                              injected, cache, tool_log) -> bool:
        """Run one policy-issued evidence call and hand the result back.

        Appended to the conversation the way any tool result is, so the model
        resumes from a turn that looks exactly like one it made itself. The
        answer it was about to give never stands; the trace keeps it.
        """
        call_id = f"policy-{injected.origin.lower()}-{len(tool_log)}"

        try:
            envelope = context.tool_executor(
                context, call_id, injected.tool, dict(injected.arguments), cache,
            )
        except Exception as exc:                                # noqa: BLE001
            context.observer.notice(
                "standard_policy_call_failed",
                {"tool_name": injected.tool, "error_summary": str(exc)[:240]},
            )

            return False

        policy.observe_tool_result(injected.tool, envelope.model_content,
                                   origin=injected.origin)
        context.conversation.append(ConversationMessage(
            "assistant", (ToolUseBlock(call_id, injected.tool,
                                       dict(injected.arguments)),),
        ))
        context.conversation.append(ConversationMessage(
            "user", (ToolResultBlock(call_id, envelope.model_content,
                                     is_error=not envelope.success),),
        ))
        context.trace.emit(
            EventType.TOOL_CALL_FINISHED, context.task_id,
            session_id=context.session_id, status=EventStatus.OK,
            metadata={"component": "standard_answer_policy",
                      "origin": injected.origin, "tool_name": injected.tool},
        )

        # The MODEL is not told this happened — that is the design, and it
        # stands. The USER is: the answer about to be given cited the standard
        # having consulted nothing, and whether the citation that finally
        # appears is grounded or invented is the one thing a reader of the
        # transcript cannot otherwise tell.

        context.observer.notice(
            "standard_evidence_injected",
            {"tool_name": injected.tool, "origin": injected.origin},
        )

        return True

    def _normative_event(self, context: AgentContext, event_type, metadata) -> None:
        if getattr(context, "standard_binding", None):
            context.trace.emit(event_type, context.task_id, status=EventStatus.DETECTED,
                               metadata=dict(metadata))

    def _enter_finalizing(self, context: AgentContext, reason: str, *,
                          announce: bool = True) -> None:
        """Close the tool window, and say so when it closes."""
        window = context.finalization

        if window.state == FINALIZING:
            return

        window.state, window.reason = FINALIZING, reason

        if announce:
            context.conversation.append(ConversationMessage(
                "user", (TextBlock(FINALIZATION_DEMAND),), authored_by="harness"))

        self._normative_event(context, EventType.NORMATIVE_FINALIZATION_STARTED,
                              {"reason": reason})

    def _retry_finalization(self, context: AgentContext, turn: ModelTurn) -> ModelTurn | None:
        """The one more answer FINALIZING may ask for, or None once it is spent.

        The reply that wrote a tool call stays in the conversation as text --
        it was never a call, so it gets no result -- and the next request
        offers no tool either.
        """
        window = context.finalization

        if window.retries >= FINALIZATION_RETRY_LIMIT:
            return None

        window.retries += 1

        if (turn.text or "").strip():
            context.conversation.append(ConversationMessage(
                "assistant", (TextBlock(turn.text),)))

        context.conversation.append(ConversationMessage(
            "user", (TextBlock(FINALIZATION_RETRY),), authored_by="harness"))
        context.trace.emit(EventType.RETRY, context.task_id, status=EventStatus.DETECTED,
                           metadata={"reason": "tool_call_while_finalizing",
                                     "retry": window.retries})
        self._normative_event(context, EventType.NORMATIVE_FINALIZATION_RETRY,
                              {"retry": window.retries,
                               "structured_calls": len(turn.tool_calls),
                               "written_call": textual_tool_call(turn.text)})

        with context.observer.model_activity("Concluding…") as on_token:
            return self._complete_with_retry(context, use_tools=False, on_token=on_token)

    def _finalized(self, context: AgentContext, turn: ModelTurn) -> ModelTurn | None:
        """The turn's answer when FINALIZING got one: `turn` itself, a retry,
        or None when the model wrote a tool call twice."""
        while turn is not None and (turn.tool_calls or textual_tool_call(turn.text)):
            turn = self._retry_finalization(context, turn)

        if turn is None:
            context.finalization.state = FAILED
            context.finalization.reason = "a tool call written instead of an answer, twice"

        return turn

    def _repair_ask(self, context: AgentContext, text: str) -> str:
        """One tool-less question, for the normative repair path.

        Deliberately the narrowest call the runtime can make: no tools, one
        message, the composed system context the turn already had. A repair
        that could reach a tool would be a second turn wearing the first
        one's clothes, which is the constraint the repair contract is built
        on -- so it is enforced here by what is not passed, not by asking.
        """
        from models.model_backend import ConversationMessage, TextBlock

        try:
            turn = self.complete_model_turn(
                context, use_tools=False,
                conversation=(ConversationMessage("user", (TextBlock(text),)),),
                grace=True)
        except Exception:
            return ""        # a failed repair withholds, exactly as before

        return (turn.text or "") if turn is not None else ""

    @staticmethod
    def _result(
        context: AgentContext, reason: RuntimeTerminalReason, response: str,
        transcript: list[str], tool_log: list[str],
        trajectory: list[Mapping[str, object]], did_modify: bool,
        budget_exhausted: bool, verification_action_id: str | None,
        cache: dict[Any, Any], *, error_category: str | None = None,
        error_summary: str | None = None, completion_deferred: bool = False,
    ) -> AgentResult:
        failure_kind = {
            RuntimeTerminalReason.MODEL_FAILURE: FailureKind.MODEL_ERROR,
            RuntimeTerminalReason.INVALID_TURN: FailureKind.INVALID_MODEL_TURN,
            RuntimeTerminalReason.ROUND_BUDGET_EXHAUSTED: FailureKind.BUDGET_EXHAUSTED,
            RuntimeTerminalReason.TOOL_BUDGET_EXHAUSTED: FailureKind.BUDGET_EXHAUSTED,
            RuntimeTerminalReason.INTERRUPTED: FailureKind.INTERRUPTED,
            RuntimeTerminalReason.STALLED: FailureKind.STALLED,
            RuntimeTerminalReason.RUNTIME_FAILURE: FailureKind.RUNTIME_ERROR,
        }.get(reason)

        # The deterministic normative boundary, on the way out and on every
        # return path. Same reasoning as the claim check below: the evidence
        # lives here, so an answer the evidence does not support must not reach
        # history either. Non-standard turns have no policy and are untouched.

        # A turn that did work and says nothing. The loop has its own
        # fallback for this, and on a bound turn it has been observed not to
        # reach here: three separate runs handed back an assessment record
        # with a blank body above it, and the reader could not tell whether
        # the model had said nothing or the harness had lost it. This is the
        # single exit every path goes through, so it is the one place the
        # guarantee can actually be made.

        if not (response or "").strip() and tool_log:
            response = no_conclusion_note(tool_log)
            context.observer.notice(
                "empty_final_synthesis", {"tool_call_count": len(tool_log)},
            )

        policy = getattr(context, "standard_policy", None)

        if policy is not None:
            stopped = (STOPPED_BY_BUDGET
                       if reason in (RuntimeTerminalReason.ROUND_BUDGET_EXHAUSTED,
                                     RuntimeTerminalReason.TOOL_BUDGET_EXHAUSTED,
                                     RuntimeTerminalReason.BUDGET_EXHAUSTED)
                       else None)
            response = policy.finalize(
                response, rounds=context.working_state.current_round,
                stopped_by=stopped)
            context.trace.emit(
                EventType.TOOL_CALL_FINISHED, context.task_id,
                session_id=context.session_id, status=EventStatus.OK,
                metadata={"component": "standard_answer_policy",
                          **{key: value
                             for key, value in policy.trace().items()
                             if key in ("bootstrap_triggered",
                                        "empty_structure_recovery_triggered",
                                        "exhaustion_triggered",
                                        "guard_replaced",
                                        "provenance_guard_triggered",
                                        # Without this the guard is invisible
                                        # in the only trace a diagnosis reads:
                                        # a removed requirement had to be
                                        # recovered by re-running the turn
                                        # under --record.
                                        "normative_precedence_triggered",
                                        "conformance_guard_triggered",
                                        "claims_guard_triggered",
                                        "model_tool_calls", "stopped_by")}},
            )

        window = getattr(context, "finalization", None)

        if window is not None and window.state != FAILED:
            window.state = DONE

        # Checked here rather than at the rendering layer: the evidence lives
        # here, and a claim contradicted by it must not reach history either.

        # A coding turn's verdict is the agent core path's (completion.py):
        # decided once and written into the answer, not appended twice.
        core_verdict = context.core_verdict

        claim_note = ("" if core_verdict is not None else
                      unsupported_change_claim(response or "", tool_log, did_modify))

        if claim_note:
            response = (response or "") + claim_note

        # The last word on a turn that wrote. Deterministic, from the tool log,
        # and on every exit path -- the budget's included, which is the one
        # that handed back four edited files and a tree that did not compile.

        # The project's own verification first, because the note below has to
        # know whether it ran: it is cached per generation, so this is the
        # same single execution either way.

        runs = project_build_runs(context, tool_log)
        write_note = unverified_write_note(
            tool_log, runs, getattr(context, "project_commands", None))

        if write_note and core_verdict is None:
            response = (response or "") + write_note

        broken, output = next(((command, text) for command, status, text in runs
                               if status != "passed"), ("", ""))
        commands = getattr(context, "project_commands", None)

        # A verification that could not be performed leaves the turn UNJUDGED,
        # not failed: "not judged" and "judged and wrong" are different facts,
        # and a corpus that confuses them teaches the confusion.

        unjudged = any(status == "not_run" for _, status, _ in runs)
        build_ok = (None if commands is None or not turn_changed_files(context, tool_log)
                    or unjudged else not broken)

        # The harness just ran the project's own build and tests against the
        # code this turn wrote. That IS a verification of this generation, and
        # recording it is what stops the turn from demanding the model run the
        # same suite a second time to prove what the harness already observed.
        # Only a CONFIGURED command counts: a probed `make` is a guess about
        # how to build a tree, not a certificate the operator asked for.

        _record_project_verification(context, commands, runs)

        if broken:
            response = (response or "") + project_build.note(broken, output)

        # What this turn read, so the next one starts from it rather than
        # from whatever a fresh search happens to surface.

        if policy is not None:
            context.clauses_read = tuple(sorted(policy.clauses.sections))

        # The lifecycle's last two moves, and the one thing it owes the
        # reader: a plan item that nothing validated. It is reported rather
        # than enforced -- "the existing suite already covers it" is a
        # legitimate answer and a deterministic layer cannot tell it from a
        # requirement quietly dropped -- but a turn that planned five changes
        # and ran one test should not be the only party that knows.

        phase = getattr(context, "work_phase", None)

        if phase is not None and phase.engaged:
            phase.begin_review()
            unresolved = phase.unresolved_items()

            if unresolved:
                response = (response or "") + (
                    "\n\nPLANNED, NOT VALIDATED — no validation action this "
                    "turn claims to cover:\n"
                    + "\n".join(f"- {item.requirement}" for item in unresolved))

            # And the contract, closed or not. A turn that inherited a set of
            # requirements owes its reader a line per requirement, because
            # the failure this replaces is a requirement that stopped being
            # mentioned -- and silence reads exactly like success.

            response = (response or "") + requirement_matrix_note(
                phase, response or "")

            context.trace.emit(
                EventType.TOOL_CALL_FINISHED, context.task_id,
                session_id=context.session_id, status=EventStatus.OK,
                metadata={"component": "work_phase", **{
                    key: value for key, value in phase.to_dict().items()
                    if key in ("phase", "first_write_phase", "writes",
                               "write_refusals", "syntheses_forced")},
                    "plan_items": len(phase.items),
                    "rejected_items": len(phase.rejected),
                    "replans": len(phase.replans),
                    "unresolved_items": len(unresolved),
                    "requirements_carried": len(phase.requirements),
                    "requirements_open": len(phase.requirements.open_items())},
            )
            phase.finish()

        # And the tally of the work order's own sections, when the turn was
        # working from one. Deterministic, from the diff.

        order, missing = work_order_gap(context, tool_log, cache)

        if order:
            response = (response or "") + work_order.record(
                order, list(turn_changed_files(context, tool_log)), missing)

        return AgentResult(
            task_id=context.task_id,
            terminal_status=context.working_state.terminal_status,
            terminal_reason=reason,
            final_response=response,
            working_state=context.working_state,
            rounds_consumed=context.working_state.current_round,
            model_calls=context.working_state.model_rounds,
            tool_actions=context.working_state.tool_actions,
            verification_outcome=context.working_state.verification_outcome,
            conversation=tuple(context.conversation),
            transcript=tuple(transcript),
            tool_log=tuple(tool_log),
            trajectory=tuple(trajectory),
            did_modify=did_modify,
            budget_exhausted=budget_exhausted,
            verification_action_id=verification_action_id,
            error_category=error_category,
            error_summary=error_summary,
            failure_kind=failure_kind,
            execution_cache=cache,
            completion_deferred=completion_deferred,
            project_build_ok=build_ok,
            changed_paths=turn_changed_files(context, tool_log),
        )
