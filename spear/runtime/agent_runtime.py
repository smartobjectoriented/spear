"""Reusable provider-neutral execution core for one SPEAR agent task.

The runtime owns model/tool iteration and task-scoped context state.  It knows
nothing about terminal rendering, project selection, persistence, or concrete
tool dispatch.  The legacy dispatcher is injected through ToolExecutor until
the later ToolRegistry/ToolRouter phase.

AgentRuntime keeps the legacy loop (`run`) here; the rest of its methods come
from mixins, one module each: agent_core_turn, agent_finalization,
agent_model_turn and agent_recording.
"""

from __future__ import annotations

import json
import os
import traceback
from typing import Any

from runtime.cancellation import OperationCancelled
from runtime.failure_policy import FailureKind
from runtime.session_store import InFlightOperation, SessionError, SessionEventType
from models.model_backend import (
    ConversationMessage, TextBlock, ToolResultBlock, ToolUseBlock, textual_tool_call,
)
from runtime.tracing import EventStatus, EventType, new_action_id
from harness.tool_router import ToolResultEnvelope, ToolResultStatus
from harness.tool_registry import COMMAND_TOOLS
from evidence import conformance_mode
from evidence import project_build
from runtime import work_order
from evidence.verification import CompletionVerificationStatus
from runtime.budgets import BudgetExceeded, BudgetKind
from runtime.working_state import StateEventType, StateSource, TerminalStatus
from runtime.agent_context import AgentContext, AgentResult, RuntimeTerminalReason
from runtime.agent_core_turn import CodingCoreMixin
from runtime.agent_finalization import (
    FINALIZING, INVESTIGATING, Finalization, FinalizationMixin,
    finalization_failed_note, no_conclusion_note,
)
from runtime.agent_model_turn import ModelTurnMixin
from runtime.agent_recording import RecordingMixin

# is_write_request and unverified_change are not used here: cli/rag_chat.py
# imports them from this module.

from runtime.agent_notes import (
    _PLAN_TOOL, _REPEAT_NOTICE, _WRITE_TOOLS, _asked, _investigation_ceiling,
    _past_wall_fraction, _repeated_result, _round_was_all_refused,
    _tool_activity, _with_repeat_note, _write_round_tools,
    announced_but_unmade_change, carried_obligations, conclude_demand,
    is_write_request, looks_like_preamble, make_it_demand, may_demand_write,
    standard_policy_for, strip_fabrications, turn_evidence,
    validate_agent_turn, wants_write, write_demand,
)
from runtime.agent_verification import (
    _NEEDS_VERIFICATION, _record_project_verification,
    asked_to_write_and_did_not, changed_files, project_build_gap,
    project_build_runs, unverified_change, verify_demand, work_order_gap,
)


# How many times a turn may be sent back to an unfinished work order.
# Bounded because a redirect costs a round; more than one because a
# multi-section order is never finished by a single extra one.
ORDER_REDIRECT_LIMIT = int(os.environ.get("SPEAR_ORDER_REDIRECTS", "4"))

# How many times a turn may be sent back to a build it broke. A build
# failure is concrete and usually one edit away, so this is worth more
# rounds than a vaguer redirect -- but not unbounded: a model that cannot
# fix it in three will not fix it in ten.
BUILD_REPROMPT_LIMIT = int(os.environ.get("SPEAR_BUILD_RETRIES", "3"))

# How many times a turn may be sent back to a change it was asked for and has
# not made. One was the old cap, and one is measurably not enough: a turn was
# told "nothing was changed — make the edit", read for another twenty rounds,
# looped, and ended with an empty diff and nothing left to ask it. Bounded
# like the work-order redirect, and for the same reason: a redirect costs a
# round, and a redirect that changes nothing twice will not change anything
# on the third.
WRITE_REDIRECT_LIMIT = int(os.environ.get("SPEAR_WRITE_REDIRECTS", "3"))

# Rounds a turn gets to itself between two of those redirects. Raising the
# cap from one to three without this made the harness impatient rather than
# persistent: the model answered its first two rounds with "let me examine
# the source files", the preamble guard absorbed two, and the three
# redirects then fired in three consecutive rounds -- all of them spent
# before the turn had read anything at all. Asking again is only useful
# after the turn has had room to act on the last ask.
WRITE_REDIRECT_SPACING = int(os.environ.get("SPEAR_WRITE_REDIRECT_SPACING", "6"))

# How many times a turn may be sent back to the clauses its own session
# established and its diff does not account for.
CLAUSE_REDIRECT_LIMIT = int(os.environ.get("SPEAR_CLAUSE_REDIRECTS", "3"))


class AgentRuntime(CodingCoreMixin, FinalizationMixin, ModelTurnMixin,
                   RecordingMixin):
    """Reusable execution lifecycle. Instances contain no task state."""

    def run(self, context: AgentContext, *, defer_completion: bool = False) -> AgentResult:
        if context.execution_core == "coding":
            return self._run_core(context, defer_completion=defer_completion)

        transcript: list[str] = []
        tool_log = context.session_tool_log
        trajectory = context.session_trajectory
        cache: dict[Any, Any] = {}
        response = ""

        # One deterministic policy per standard-bound turn. None for every
        # other turn, which then takes exactly the path it took before.

        policy = standard_policy_for(
            context, repair_ask=lambda text: self._repair_ask(context, text))
        context.standard_policy = policy

        # And one lifecycle, for the turns that have two sources of truth to
        # reconcile. Engaged from two facts already established: the turn is
        # bound to an authoritative document, and the user asked for the code
        # to change. Neither is guessed here.

        phase = context.work_phase

        if phase is not None:
            phase.engage(
                authority_bound=policy is not None,
                write_requested=wants_write(context, _asked(context.conversation)),
                # What the SESSION established, not only what this turn has
                # re-read. The two-prompt shape is the whole point of
                # `prior_clauses`: the question was answered from the document
                # in the first turn, and the change is asked for in the
                # second. A gate that ignored them told a run its own answer
                # was unsupported -- it planned against the rule it had just
                # quoted, was refused, and spent nine minutes re-searching for
                # a clause it already had. The strict current-turn rule stays
                # where it belongs, on the answer: the citation and identifier
                # guards are untouched.
                prior_authority=context.prior_clauses,
                # What counts as "the implementation". A session may have
                # several corpora attached and read a great deal of code that
                # has nothing to do with the task; a gate that opened on that
                # would have been satisfied by the wrong tree.
                root=getattr(context, "project_root", ""),
                requirements=context.carried_requirements)

        investigate_rounds = preamble_reprompts = repeats = make_reprompts = 0
        scope_final = False
        context.finalization = Finalization()
        self._normative_event(context, EventType.NORMATIVE_INVESTIGATION_STARTED,
                              {"tools": len(context.tools or ())})
        order_reprompts = 0
        order_remaining = None
        build_reprompts = 0
        clause_reprompts = 0
        did_modify = nudged = verify_reprompts = budget_event_emitted = False
        only_tools: tuple[str, ...] | None = None
        wrap_up_warned = landed = False
        validation_demands = next_item_demands = 0
        seen_results: dict[str, list[str]] = {}
        last_write_redirect = -WRITE_REDIRECT_SPACING
        last_clause_redirect = -WRITE_REDIRECT_SPACING
        last_build_output = ""
        round_envelopes: list = []
        rounds_since_nudge = 0
        verification_action_id = None
        reason = RuntimeTerminalReason.COMPLETED

        # Which stage the loop is in, so an exception below can name what was
        # being done rather than reporting an undifferentiated runtime failure.

        failure_stage = "runtime"

        try:
            self._start_task(context)
            context.cancellation.raise_if_cancelled()

            # A bound turn reads the standard before it reads anything else.
            # The boundary bootstrap already made this call, but only once the
            # model had finished: a turn that spent twelve rounds in the C
            # sources got its one normative search delivered into a context
            # that already held twelve greps and a written answer, and
            # concluded the code compliant anyway. The same call placed here
            # is a source rather than a rebuttal.

            if policy is not None:
                opening = policy.opening()

                if opening is not None:
                    self._inject_evidence_call(context, policy, opening, cache,
                                               tool_log)

            # A resumed task continues from the round it reached, so the range
            # starts at the working state rather than at zero.

            # The bound is a variable, not a literal range: a turn that ends
            # with the tree not compiling has not finished, and the one thing
            # worth spending a round beyond the budget on is making it build
            # again. Raised at most BUILD_REPROMPT_LIMIT times, by two rounds
            # each -- enough to read an error and make one edit.

            round_limit = context.max_model_rounds
            round_index = context.working_state.current_round - 1

            while round_index + 1 < round_limit:
                round_index += 1
                context.apply_state_event(
                    StateEventType.ROUND_STARTED, round_number=round_index + 1,
                )

                # "Conclude now, with no further commands" was advisory, and a
                # model that ignored it kept investigating for another eight
                # tool calls. It now costs exactly one more round: enough to
                # make the single small edit the nudge also asks for, and not
                # enough to resume grepping. Anything else makes the promise in
                # that message false, which is worse than not making it.

                # `round_envelopes` still holds the PREVIOUS round here: it is
                # reset further down, after this check.
                if nudged and not _round_was_all_refused(round_envelopes):
                    rounds_since_nudge += 1

                # A round the sandbox refused outright is not a round the
                # turn was given. The nudge grants exactly one round with
                # tools before the window closes, and twice in a row a turn
                # spent it on a no-op the allowlist denied -- `true`, then
                # `echo skip` -- and was then asked to conclude with no
                # tools and nothing done. It answered with nothing at all.
                # Refusing a call and charging for it is the harness taking
                # back what it just offered.

                # Two different things, and conflating them cost a user their
                # answer: a BUDGET stop is a failure the controller reports as
                # such, while CONCLUDING is the normal end of a turn that was
                # told to wrap up. Both close the tool window; only the first
                # is an exhausted budget. Routing the nudge through the budget
                # branch marked the task BUDGET_EXHAUSTED, and the CLI then
                # dropped the model's conclusion -- after its edit had landed.

                # Against the live bound, not the original one: a turn
                # granted extra rounds to repair a build would otherwise
                # spend them with its tool window already closed.
                budget_final = (len(tool_log) >= context.max_tool_actions
                                or round_index == round_limit - 1)
                concluding = rounds_since_nudge > 1

                # The budget is about to close the tool window. If the turn
                # wrote and the tree no longer builds, that is not a finished
                # turn, and the boundary check below will never be reached:
                # a turn that runs out of rounds never stops making tool
                # calls, so it never passes through the place where the
                # answer is judged. Repair is decided here instead, where the
                # window actually closes.

                if (budget_final and build_reprompts < BUILD_REPROMPT_LIMIT
                        and may_demand_write(context)
                        and changed_files(tool_log)):
                    broken, output = project_build_gap(context, tool_log)

                    if broken:
                        build_reprompts += 1
                        round_limit = round_index + 3
                        budget_final = False

                        # The same command failing the same way for the third
                        # time is not a build to retry, it is a plan to
                        # reconsider: each retry so far has asked for another
                        # fix against the same understanding, and the
                        # understanding is what the evidence now contradicts.

                        demand = project_build.demand(
                            broken, output, last_build_output)

                        if (phase is not None and phase.engaged
                                and phase.validation_exhausted(broken)):
                            phase.invalidate(
                                reason="the project's own verification keeps "
                                       "failing the same way",
                                evidence=output.splitlines()[0][:200]
                                if output else broken)
                            demand += (
                                "\n\nThis command has now failed unchanged "
                                "more than once. Stop patching against the "
                                "current plan: read the failure, then record "
                                "what it actually shows with plan_change, "
                                "naming the item it replaces.")

                        context.conversation.append(ConversationMessage(
                            "user", (TextBlock(demand),),
                            authored_by="harness"))
                        last_build_output = output
                        only_tools = _write_round_tools(context)
                        context.observer.notice(
                            "project_build_failed",
                            {"command": broken, "retry": build_reprompts},
                        )

                # A second attempt to change a file the user said not to
                # change. The first refusal named the scope and said what to
                # do instead; a turn that goes looking for another syntax is
                # not going to be argued out of it by a third. The tool
                # window closes and the answer is asked for.

                if max((envelope.metadata.get("read_only_violations", 0)
                        for envelope in round_envelopes), default=0) >= 2:
                    scope_final = True

                    context.observer.notice(
                        "read_only_violation",
                        {"attempts": max(envelope.metadata.get(
                            "read_only_violations", 0)
                            for envelope in round_envelopes)},
                    )

                force_final = budget_final or concluding or scope_final

                if force_final:
                    self._enter_finalizing(
                        context, "budget" if budget_final else
                        "read-only scope" if scope_final else "concluding")
                elif context.finalization.state == FINALIZING:
                    context.finalization.state = INVESTIGATING

                if budget_final and not budget_event_emitted:
                    budget_event_emitted = True

                    budget_reason = ("command_budget" if
                                     len(tool_log) >= context.max_tool_actions
                                     else "round_budget")
                    reason = (RuntimeTerminalReason.TOOL_BUDGET_EXHAUSTED
                              if budget_reason == "command_budget" else
                              RuntimeTerminalReason.ROUND_BUDGET_EXHAUSTED)

                    context.apply_state_event(
                        StateEventType.BUDGET_REACHED, reason=budget_reason,
                    )
                    context.trace.emit(
                        EventType.BUDGET_EXHAUSTED, context.task_id,
                        status=EventStatus.EXHAUSTED,
                        metadata={
                            "round": round_index + 1,
                            "tool_call_count": len(tool_log),
                            "max_tool_rounds": context.max_model_rounds,
                            "max_commands": context.max_tool_actions,
                            "reason": budget_reason,
                        },
                    )

                # Said once, at four fifths of the wall-clock budget: stop
                # opening new ground and land what you have. A hard stop that
                # arrives with no warning cuts a turn mid-thought; a warning
                # gives it the chance to finish deliberately.

                if not wrap_up_warned and _past_wall_fraction(context, 0.8):
                    wrap_up_warned = True
                    context.conversation.append(ConversationMessage(
                        "user", (TextBlock(
                            "Time budget nearly spent. Stop starting new "
                            "investigation. Finish the change you are on, "
                            "verify it, and report -- or say plainly what is "
                            "left undone."),), authored_by="harness"))
                    context.observer.notice("wall_time_wrap_up", {})

                label = "Thinking…" if round_index == 0 else "Analyzing results…"

                failure_stage = "model"

                # Consumed here and cleared immediately: it narrows exactly
                # one round, the one that was just told its change broke the
                # build.
                narrowed, only_tools = only_tools, None

                with context.observer.model_activity(label) as on_token:
                    turn = self._complete_with_retry(
                        context, use_tools=not force_final, on_token=on_token,
                        only_tools=narrowed,
                    )

                context.cancellation.raise_if_cancelled()

                failure_stage = "turn_validation"

                action = validate_agent_turn(turn)
                text = strip_fabrications(turn.text).strip()
                calls = list(turn.tool_calls)

                if action == "refusal":
                    reason = RuntimeTerminalReason.REFUSAL
                    response = text or "The model declined this request."

                    break

                # A call is executable only when the API returned it as one
                # AND a tool was offered. Written out as text it is neither:
                # it never runs and is never the answer. Either kind, met
                # where no tool was offered or written at all, closes the
                # window and asks for the answer -- once.

                if (force_final and calls) or (not calls and textual_tool_call(turn.text)):
                    self._enter_finalizing(context, "tool call written as text",
                                           announce=False)
                    turn = self._finalized(context, turn)

                    if turn is None:
                        reason = RuntimeTerminalReason.STALLED
                        response = finalization_failed_note(context)

                        break

                    text = strip_fabrications(turn.text).strip()
                    calls = []
                    force_final = True

                if not calls:
                    # A turn that only narrates what it is about to do has not
                    # done it. Re-prompting is bounded, and never on the final
                    # round, where tools are closed and narration is all there is.

                    if (not force_final and preamble_reprompts < 2
                            and looks_like_preamble(text)):
                        preamble_reprompts += 1

                        context.apply_state_event(
                            StateEventType.RETRY_RECORDED,
                            reason="action_preamble_without_tool_call",
                        )
                        context.trace.emit(
                            EventType.RETRY, context.task_id,
                            status=EventStatus.DETECTED,
                            metadata={"reason": "action_preamble_without_tool_call",
                                      "retry": preamble_reprompts},
                        )
                        self._show_intermediate(context, transcript, text)

                        context.conversation.append(ConversationMessage(
                            "assistant", (TextBlock(turn.text),)))
                        context.conversation.append(ConversationMessage(
                            "user", (TextBlock(
                                "Do it now — actually CALL the tool (bash, "
                                "read_file, edit_file, write_file). Don't just say "
                                "what you will do."
                            ),), authored_by="harness"))

                        continue

                    # A turn that changed files and stopped without running
                    # anything gets exactly one demand to verify its own work.

                    verification_state = context.verification_policy.evaluate_completion(
                        context.working_state,
                    )
                    native_mutation = any(
                        item.name in _WRITE_TOOLS and item.status.value == "succeeded"
                        for item in context.working_state.actions
                    )

                    # The project's own suite runs BEFORE the demand that the
                    # model go and verify -- because it is the answer to the
                    # question that demand is about to ask. Asking first cost
                    # a model call and a retry to learn what the harness can
                    # observe for free: a run fixed main.py, ran a targeted
                    # test, was told to verify anyway, and only then did the
                    # deterministic build/full/passed evidence appear.
                    #
                    # It is cached per generation, so this is the same single
                    # execution the end of the turn would have performed, just
                    # at the moment its answer is worth something.

                    if (native_mutation
                            and verification_state.status in _NEEDS_VERIFICATION):
                        _record_project_verification(
                            context, getattr(context, "project_commands", None),
                            project_build_runs(context, tool_log))
                        verification_state = (
                            context.verification_policy.evaluate_completion(
                                context.working_state))

                    if (not force_final and not verify_reprompts
                            and native_mutation
                            and verification_state.status in {
                                CompletionVerificationStatus.UNVERIFIED,
                                CompletionVerificationStatus.PARTIALLY_VERIFIED,
                                CompletionVerificationStatus.FAILED,
                            }):
                        verify_reprompts += 1

                        context.apply_state_event(
                            StateEventType.VERIFICATION_REPROMPT_RECORDED,
                            reason="unverified_change",
                        )
                        context.apply_state_event(
                            StateEventType.RETRY_RECORDED,
                            reason="unverified_change",
                        )

                        verification_action_id = new_action_id("verification_nudge")

                        context.trace.emit(
                            EventType.VERIFICATION_STARTED, context.task_id,
                            status=EventStatus.STARTED,
                            action_id=verification_action_id,
                            metadata={"kind": "verification_nudge",
                                      "changed_file_count": len(changed_files(tool_log))},
                        )
                        context.trace.emit(
                            EventType.RETRY, context.task_id,
                            status=EventStatus.DETECTED,
                            action_id=verification_action_id,
                            metadata={"reason": "unverified_change", "retry": 1},
                        )
                        self._show_intermediate(context, transcript, text)

                        context.conversation.append(ConversationMessage(
                            "assistant", (TextBlock(turn.text),)))
                        context.conversation.append(ConversationMessage(
                            "user", (TextBlock(verify_demand(tool_log)),), authored_by="harness"))

                        context.observer.notice(
                            "verification_nudge",
                            {"changed_file_count": len(changed_files(tool_log))},
                        )

                        continue

                    # The model is about to answer a bound normative question
                    # having looked at nothing, or having read an empty
                    # structure registry and stopped there. One deterministic
                    # retrieval, once per turn, then it decides again. It is
                    # not asked to and never sees that this happened.

                    if policy is not None:
                        injected = policy.answer_boundary(text)

                        if injected is not None and self._inject_evidence_call(
                                context, policy, injected, cache, tool_log):
                            continue

                    # The project's own build and tests, on a turn that
                    # wrote. This needs no task file and no acceptance block:
                    # a tree states how it is built, and a change that does
                    # not survive that is not finished. Bounded, and never
                    # once the real budget has closed the tool window.

                    # Not gated on the budget, unlike every other redirect.
                    # A round budget says "stop investigating"; it does not
                    # say "leave the tree broken". Asked to change the code,
                    # a turn spent sixty rounds reading, wrote once, and
                    # handed back a tree that did not compile -- with this
                    # gate disarmed by the very budget that ended it.

                    if (build_reprompts < BUILD_REPROMPT_LIMIT
                            and may_demand_write(context)):
                        broken, output = project_build_gap(context, tool_log)

                        if broken:
                            build_reprompts += 1

                            if round_index + 1 >= round_limit:
                                round_limit = round_index + 3

                            self._show_intermediate(context, transcript, text)
                            context.conversation.append(ConversationMessage(
                                "assistant", (TextBlock(turn.text),)))
                            context.conversation.append(ConversationMessage(
                                "user", (TextBlock(project_build.demand(
                                    broken, output, last_build_output)),),
                                authored_by="harness"))
                            last_build_output = output
                            only_tools = _write_round_tools(context)
                            context.observer.notice(
                                "project_build_failed",
                                {"command": broken, "retry": build_reprompts},
                            )
                            context.trace.emit(
                                EventType.RETRY, context.task_id,
                                status=EventStatus.DETECTED,
                                metadata={"reason": "project_build_failed",
                                          "retry": build_reprompts,
                                          "arguments_recorded": False},
                            )

                            # Keep what it just said: if the extra
                            # round produces nothing, the turn still
                            # has a conclusion to hand back.
                            response = text

                            continue

                    # Every clause this session established, accounted for.
                    # The specification is carried into the turn and the turn
                    # is asked to say what the code does about each rule;
                    # nothing checked it, and four runs in a row answered
                    # about two clauses out of ten, changed one line, and
                    # stopped -- each of them building and passing its tests.

                    # Same two brakes as the write redirect had, and the
                    # same evidence against them. `concluding` suppressed
                    # this on the turn that most needs it -- a session
                    # established twelve clauses, the turn added one field
                    # to one struct, said it was done, and nothing asked for
                    # the other eleven. And once is not enough for a change
                    # that spans five files; spaced, like the write ask, so
                    # each one has room to be acted on.

                    # ...and only for a turn that was asked to change the
                    # code. The clauses a previous turn established are
                    # evidence for this one, not work it owes: read alone,
                    # they turned a normative question into an
                    # implementation audit of someone else's tree.

                    carried = list(carried_obligations(
                        context, _asked(context.conversation)))

                    if (clause_reprompts < CLAUSE_REDIRECT_LIMIT
                            and round_index - last_clause_redirect
                                >= WRITE_REDIRECT_SPACING
                            and carried):
                        skipped = conformance_mode.unaddressed_clauses(
                            text, carried)

                        if skipped:
                            clause_reprompts += 1
                            last_clause_redirect = round_index

                            # Everything except the standard. The clauses
                            # this asks about are already established and
                            # carried in the turn's own context; watched
                            # live, the model answered "let me work through
                            # each clause systematically" and spent the rest
                            # of the turn re-fetching sections it had
                            # already been given, at twenty seconds of
                            # compaction a round. What it needs here is the
                            # code, not the rulebook.

                            only_tools = tuple(
                                tool.name for tool in context.tools
                                if not tool.name.startswith("standard."))

                            if round_index + 1 >= round_limit:
                                round_limit = round_index + 3

                            self._show_intermediate(context, transcript, text)
                            context.conversation.append(ConversationMessage(
                                "assistant", (TextBlock(turn.text),)))
                            context.conversation.append(ConversationMessage(
                                "user", (TextBlock(
                                    conformance_mode.clause_demand(
                                        skipped, carried)),), authored_by="harness"))
                            context.observer.notice(
                                "clauses_unaddressed",
                                {"missing": skipped[:8],
                                 "carried": len(carried)},
                            )
                            response = text

                            continue

                    # A work order with sections the turn never wrote to. The
                    # model does the mechanical part -- the signature, until
                    # it compiles -- and reports the task done; the sections
                    # that need a decision are the ones it skips.

                    # Gated on the BUDGET, not on force_final. The nudge to
                    # conclude sets force_final two rounds after it fires, and
                    # a turn that had done six of seven sections was let go
                    # because of it: the tally named the seventh, and nothing
                    # had asked for it. A round budget is a real limit; being
                    # told to wrap up is not, and finishing the work order is
                    # exactly what wrapping up should mean.

                    # Redirected more than once, while it keeps working. A
                    # seven-section order is not finished by one extra round,
                    # and capping at one let a turn hand back six of seven
                    # with the seventh named in the tally and nobody asking
                    # for it. The stopping rule is the monitor's own: a
                    # redirect that does not shrink the failing set is a loop,
                    # and the next one would be too.

                    if not budget_final and order_reprompts < ORDER_REDIRECT_LIMIT:
                        order, missing = work_order_gap(
                            context, tool_log, cache)
                        labels = {item["label"] for item in missing}
                        progressed = (order_remaining is None
                                      or labels < order_remaining)

                        if missing and progressed:
                            order_remaining = labels
                            order_reprompts += 1

                            self._show_intermediate(context, transcript, text)
                            context.conversation.append(ConversationMessage(
                                "assistant", (TextBlock(turn.text),)))
                            context.conversation.append(ConversationMessage(
                                "user", (TextBlock(
                                    work_order.demand(missing)),), authored_by="harness"))
                            context.observer.notice(
                                "work_order_sections_skipped",
                                {"sections": [item["label"]
                                              for item in missing]},
                            )

                            # Keep what it just said: if the extra
                            # round produces nothing, the turn still
                            # has a conclusion to hand back.
                            response = text

                            continue

                    # An announced edit that was never made. Once per turn,
                    # and never when the budget has already closed the tool
                    # window -- asking for an edit that cannot be executed
                    # would spend the last round on a refusal.

                    # A writing request answered with an empty diff. Like
                    # the build gate, this outlives the round budget: "stop
                    # investigating" is not "hand back nothing".

                    # Past the round budget AND past the nudge. Concluding
                    # used to suppress this, on the reasoning that a model
                    # told to conclude, which concluded, has answered the
                    # instruction it was given. That is true of a question
                    # and false of a write: the nudge's own words on a write
                    # request are "this turn is not finished until a file
                    # changes", so a conclusion with an empty diff has not
                    # answered it. Two master runs ended exactly there --
                    # correct diagnosis, correct plan, the sentence "Proposed
                    # fix (not yet applied)", and nothing written in sixteen
                    # minutes. The gate is the request's own nature, which is
                    # what asked_to_write_and_did_not already reads; and the
                    # conclusion is kept either way (response = text below),
                    # so nothing is discarded by asking once more.

                    if (make_reprompts < WRITE_REDIRECT_LIMIT
                            and may_demand_write(context)
                            and round_index - last_write_redirect
                                >= WRITE_REDIRECT_SPACING
                            and asked_to_write_and_did_not(
                                _asked(context.conversation), tool_log,
                                did_modify,
                                is_write=wants_write(
                                    context, _asked(context.conversation)))):
                        make_reprompts += 1

                        # Reopen the tool window. force_final is
                        # `budget_final or concluding`, and use_tools is its
                        # negation, so a redirect issued while concluding
                        # asked for an edit in a round that offered no tools
                        # at all. The model answered, accurately, "I did not
                        # call edit_file this turn" -- it could not have. A
                        # demand the harness makes impossible is worse than
                        # no demand: it reads as the model refusing.

                        nudged = False
                        rounds_since_nudge = 0
                        investigate_rounds = 0

                        if round_index + 1 >= round_limit:
                            round_limit = round_index + 3

                        self._show_intermediate(context, transcript, text)
                        context.conversation.append(ConversationMessage(
                            "assistant", (TextBlock(turn.text),)))
                        context.conversation.append(ConversationMessage(
                            "user", (TextBlock(write_demand(
                                _asked(context.conversation))),), authored_by="harness"))
                        # Same lever as the build repair, for the same
                        # reason and on the same evidence: told three times
                        # "make the edit now, read nothing first", a turn
                        # read three more times and ended with an empty
                        # diff. The demand is a request; the tool list is
                        # not. For this one round the only tools are the
                        # ones that change a file.
                        only_tools = _write_round_tools(context)
                        last_write_redirect = round_index
                        context.observer.notice(
                            "write_request_unanswered",
                            {"tool_calls": len(tool_log)},
                        )

                        # Every relaunch of a completion says so in the trace,
                        # with its reason. A notice alone lives in the
                        # observer and nowhere a trace reader can see, which
                        # made the extra model calls of a turn impossible to
                        # account for afterwards.

                        context.trace.emit(
                            EventType.RETRY, context.task_id,
                            status=EventStatus.DETECTED,
                            metadata={"reason": "write_request_unanswered",
                                      "retry": make_reprompts,
                                      "tool_calls": len(tool_log)},
                        )

                        response = text

                        continue

                    # An announcement is the model's own words, not the
                    # user's, so this gate never consulted the request at
                    # all -- and a turn forbidden to write that said "I will
                    # fix this" was then told to go and do it.

                    if (not force_final and not make_reprompts
                            and may_demand_write(context)
                            and announced_but_unmade_change(text, tool_log,
                                                            did_modify)):
                        make_reprompts += 1

                        self._show_intermediate(context, transcript, text)
                        context.conversation.append(ConversationMessage(
                            "assistant", (TextBlock(turn.text),)))
                        context.conversation.append(ConversationMessage(
                            "user", (TextBlock(make_it_demand(text)),), authored_by="harness"))
                        context.observer.notice(
                            "announced_change_not_made",
                            {"tool_calls": len(tool_log)},
                        )
                        context.trace.emit(
                            EventType.RETRY, context.task_id,
                            status=EventStatus.DETECTED,
                            metadata={"reason": "announced_change_not_made",
                                      "retry": 1},
                        )

                        response = text

                        continue

                    response = text

                    break

                self._show_intermediate(context, transcript, text)

                context.conversation.append(ConversationMessage(
                    "assistant", (TextBlock(turn.text),) + tuple(
                        ToolUseBlock(call.id, call.name, call.arguments)
                        for call in calls
                    ),
                ))

                result_blocks = []
                round_envelopes = []

                failure_stage = "tool"

                for call_index, call in enumerate(calls):
                    arguments = dict(call.arguments)

                    # A cancellation mid-round still owes the provider one
                    # result block per call it was given, or the conversation
                    # is malformed for every backend that checks the pairing.

                    if context.cancellation.is_cancelled:
                        for pending in calls[call_index:]:
                            result_blocks.append(ToolResultBlock(
                                pending.id,
                                "ERROR: task cancellation prevented this tool from starting.",
                                is_error=True,
                            ))

                        context.conversation.append(ConversationMessage(
                            "user", tuple(result_blocks),
                        ))
                        context.cancellation.raise_if_cancelled()

                    # The command budget is only checked here; it is consumed
                    # after the call, so a cached result costs nothing.

                    if context.budget_manager is not None:
                        context.budget_manager.consume(BudgetKind.TOOL_CALLS)

                        if call.name in COMMAND_TOOLS:
                            context.budget_manager.ensure_available(
                                BudgetKind.COMMAND_EXECUTIONS
                            )

                    self._persist_session(
                        context, SessionEventType.TOOL_ACTION_STARTED,
                        {"tool_call_id": call.id, "tool_name": call.name,
                         "mutating": call.name in _WRITE_TOOLS},
                        in_flight=InFlightOperation(
                            "tool", call.id, call.name,
                            call.name in _WRITE_TOOLS,
                        ),
                    )

                    # A tool that raises instead of returning is still a result
                    # the model must see, so the exception becomes an envelope.
                    #
                    # Behind the same indicator as a model call, because the
                    # operator is waiting the same way. Only the model call
                    # had one, so when a tool ran the status line kept the
                    # finished call's last frame -- label, token count and
                    # all -- and stopped moving: a tool that prints nothing
                    # while it works, which is every standard.* tool, looked
                    # like a freeze. Measured on one turn: 85 of 145 seconds
                    # spent in tool calls, 25 of 30 of them over two seconds.
                    # The clock is the TURN's, so it carries straight on
                    # across the handover instead of restarting.

                    try:
                        with context.observer.model_activity(_tool_activity(call.name)):
                            envelope = context.tool_executor(
                                context, call.id, call.name, arguments, cache,
                            )
                    except Exception as exc:
                        text = f"ERROR: tool '{call.name}' failed: {exc}"
                        envelope = ToolResultEnvelope(
                            call.id, new_action_id(call.name), call.name, False,
                            (ToolResultStatus.CANCELLED
                             if isinstance(exc, OperationCancelled)
                             else ToolResultStatus.FAILED),
                            text, text, "other", 0.0,
                            len(text.encode("utf-8")),
                            error_category=(FailureKind.INTERRUPTED.value
                                            if isinstance(exc, OperationCancelled)
                                            else type(exc).__name__),
                            error_summary=str(exc)[:240] or type(exc).__name__,
                        )
                        context.observer.notice(
                            "tool_exception", {
                                "tool_name": call.name,
                                "error_summary": text,
                            },
                        )

                    training_generation_before = context.working_state.mutation_generation

                    self._record_tool_result(context, envelope)

                    if policy is not None and call.name in policy.STANDARD_TOOLS:
                        policy.observe_tool_result(call.name,
                                                   envelope.model_content)

                    # Every call, for the record of which source files the
                    # turn read. A conformance answer that read none says so.

                    if policy is not None:
                        policy.observe_code_read(call.name, call.arguments,
                                                 envelope.model_content)

                    # The lifecycle reads the same two ledgers the policy
                    # keeps, and derives nothing of its own: what the turn
                    # retrieved from the document, and which files a reading
                    # tool put in front of it. A call that added neither is
                    # counted as such, which is how repetition is noticed
                    # without anyone deciding what a "useful" call looks like.

                    # An accepted plan is progress of the same kind as a
                    # write, and its clock has to be reset the same way. A run
                    # was nudged to conclude four minutes in, recorded five
                    # evidence-backed plan items immediately afterwards, and
                    # then spent the rest of the turn with a closing tool
                    # window it had just earned the right to use.

                    if (call.name == _PLAN_TOOL and envelope.success
                            and phase is not None and phase.has_plan):
                        nudged = False
                        rounds_since_nudge = 0
                        investigate_rounds = 0

                    if phase is not None and phase.engaged:
                        phase.observe_call(
                            authority_keys=(
                                set(policy.clauses.sections)
                                | set(context.standard_source_ids_used)
                                if policy is not None else ()),
                            authority_units=(len(policy.claim_evidence.units)
                                             if policy is not None else 0),
                            # Three ledgers, because one file can be put in
                            # front of the model three ways: named in a call's
                            # arguments (`sed -n 1,80p a.c`), reported by the
                            # command boundary as read, or printed line by
                            # line by a recursive grep that named no file at
                            # all. Taking only the first would have left a
                            # turn that read the tree entirely through
                            # `grep -rn` with nothing recorded, and the gate
                            # would never have opened for it.
                            implementation_paths=(
                                (set(policy.code.files)
                                 | policy.code_evidence.files())
                                if policy is not None else set())
                            | set(envelope.read_paths))

                        if envelope.mutation:
                            phase.note_write(envelope.affected_paths)

                    if (context.budget_manager is not None and call.name in COMMAND_TOOLS
                            and envelope.status != ToolResultStatus.CACHED):
                        context.budget_manager.consume(BudgetKind.COMMAND_EXECUTIONS)

                    evidence = context.verification_policy.evidence_for_tool(
                        envelope, arguments, context.working_state,
                    )

                    if evidence is not None:
                        self._record_verification(context, evidence)

                    # Training capture is derived evidence: a failure there
                    # detaches the recorder and never disturbs the turn.

                    if context.training_recorder is not None:
                        try:
                            context.training_recorder.record_tool_result(
                                envelope,
                                mutation_generation_before=training_generation_before,
                                mutation_generation_after=(
                                    context.working_state.mutation_generation
                                ),
                                verification_generation=(
                                    evidence.mutation_generation if evidence else None
                                ),
                            )
                        except Exception as exc:
                            context.observer.notice("training_capture_error", {
                                "stage": "tool_result",
                                "error_category": type(exc).__name__,
                            })
                            context.training_recorder = None

                    round_envelopes.append(envelope)
                    result = envelope.text

                    if envelope.status == ToolResultStatus.CACHED:
                        repeats += 1

                    # Repetition judged by what came BACK, not by what was
                    # asked. The cache above compares arguments, and a turn
                    # ran twelve greps that differed only in their --include
                    # glob, each returning the same wall of matches, and
                    # nothing counted them as repeats. What tells a loop from
                    # progress is that the answer stopped changing.

                    same = _repeated_result(
                        seen_results, envelope,
                        lambda call_id: call_id not in context.dropped_tool_results)

                    if same >= _REPEAT_NOTICE:
                        repeats += 1
                        envelope = _with_repeat_note(envelope, same)

                        if same == _REPEAT_NOTICE:
                            context.observer.notice(
                                "identical_result", {"tool": call.name,
                                                     "count": same})

                    # The log is the short form the nudges quote back at the
                    # model; the trajectory keeps the fuller record.

                    tool_log.append(
                        f"{call.name} "
                        f"{json.dumps(arguments, ensure_ascii=False)[:200]}\n"
                        f"{result[:400]}"
                    )
                    trajectory.append({
                        "tool": call.name, "arguments": arguments,
                        "result": result[:4000],
                        "action_id": envelope.action_id,
                        "status": envelope.status.value,
                        "result_reference": envelope.result_reference,
                    })

                    if envelope.result_reference:
                        context.result_references.add(envelope.result_reference)

                    result_blocks.append(ToolResultBlock(
                        call.id, envelope.model_content,
                        is_error=not envelope.success,
                    ))

                    # A result already seen carries no evidence, whatever
                    # paths the call happens to name. The monitor judges
                    # progress by new evidence, and a fetch of a section the
                    # turn has already been given looked like progress every
                    # time because the source id was in its arguments. That
                    # is how a turn reached a hundred and twenty-seven calls
                    # with the stall detector never firing: each repeat
                    # presented itself as new ground.

                    observation = context.progress_monitor.observe_action(
                        call.name, arguments,
                        evidence=tuple(envelope.read_paths)
                        + tuple(envelope.affected_paths),
                        mutation=envelope.mutation,
                        fruitless=same >= _REPEAT_NOTICE,
                    )

                    # A cached result is already counted as a repeat above, so
                    # recording it again here would double-count the same call.

                    if (observation.repeated
                            and envelope.status != ToolResultStatus.CACHED):
                        context.apply_state_event(
                            StateEventType.REPEATED_ACTION_RECORDED,
                            source=StateSource.HARNESS,
                            action_id=envelope.action_id,
                        )

                    self._persist_session(
                        context, SessionEventType.TOOL_ACTION_COMPLETED,
                        {"tool_call_id": call.id, "action_id": envelope.action_id,
                         "status": envelope.status.value,
                         "result_reference": envelope.result_reference},
                        conversation=tuple(context.conversation) + (
                            ConversationMessage("user", tuple(result_blocks)),
                        ),
                    )

                    if context.cancellation.is_cancelled:
                        for pending in calls[call_index + 1:]:
                            result_blocks.append(ToolResultBlock(
                                pending.id,
                                "ERROR: task cancellation prevented this tool from starting.",
                                is_error=True,
                            ))

                        context.conversation.append(ConversationMessage(
                            "user", tuple(result_blocks),
                        ))
                        context.cancellation.raise_if_cancelled()

                    if observation.stalled:
                        reason = RuntimeTerminalReason.STALLED

                        context.trace.emit(
                            EventType.STALLED_DETECTED, context.task_id,
                            status=EventStatus.DETECTED,
                            action_id=envelope.action_id,
                            metadata={"repeat_count": observation.consecutive_without_progress,
                                      "arguments_recorded": False},
                        )

                    # An identical failing call the router has already stopped,
                    # asked for again. The tool result it just read said that
                    # repeating it changes nothing and named the tools that
                    # act on this workspace; asking once more after that is a
                    # loop. The turn ends here, with whatever it has, instead
                    # of spending the rest of its budget on the same refusal --
                    # a run made the same fetch_url call twelve times.

                    # The router says when a repetition has become terminal:
                    # it told the model once, in the tool result the model
                    # just read, and the model asked for the same thing
                    # again. Failed or merely uninformative, the answer is
                    # the same -- this turn is not going anywhere.

                    if envelope.metadata.get("terminal_repeat"):
                        suppressed = envelope.metadata.get("repeat_count", 0)
                        reason = RuntimeTerminalReason.STALLED

                        context.trace.emit(
                            EventType.STALLED_DETECTED, context.task_id,
                            status=EventStatus.DETECTED,
                            action_id=envelope.action_id,
                            metadata={"kind": envelope.metadata.get(
                                          "repeat_kind", "repeated_action"),
                                      "tool_name": envelope.tool_name,
                                      "repeat_count": suppressed,
                                      "arguments_recorded": False},
                        )

                context.conversation.append(ConversationMessage(
                    "user", tuple(result_blocks),
                ))

                # One round of retrieval is over: did it establish anything the
                # session had not already read?

                if policy is not None:
                    policy.close_round(round_index + 1)

                if reason == RuntimeTerminalReason.STALLED:
                    # A stall is a loop, and a loop on a task that is not
                    # finished is the moment to say what is missing rather
                    # than to give up: told to stop reading files, a turn
                    # looped on a retrieval tool instead and ended here
                    # having written nothing, five of seven sections
                    # untouched.

                    order, missing = work_order_gap(
                        context, tool_log, cache)

                    # A writing turn that loops is not a finished turn.
                    # The work-order redirect below needs an order to name
                    # sections from; without one -- the ordinary case, two
                    # prompts and no task file -- a stall simply dropped the
                    # turn wherever it stood, once with an empty diff and
                    # once seventeen calls into re-grepping the same header.

                    # A broken tree comes first, and it was reachable from
                    # neither of the other two gates: they run at the answer
                    # boundary and at the budget, and a stall passes through
                    # neither. Measured -- a turn changed a declaration in a
                    # header, left the definition in the .c, looped, and
                    # ended stalled. The tree no longer compiled, the turn
                    # was never told, and the operator found out from the
                    # bench afterwards.

                    if (build_reprompts < BUILD_REPROMPT_LIMIT
                            and may_demand_write(context)):
                        broken, output = project_build_gap(context, tool_log)

                        if broken:
                            build_reprompts += 1
                            reason = RuntimeTerminalReason.COMPLETED
                            context.progress_monitor.record_progress(
                                "build repair")
                            context.conversation.append(ConversationMessage(
                                "user", (TextBlock(project_build.demand(
                                    broken, output, last_build_output)),),
                                authored_by="harness"))
                            last_build_output = output
                            only_tools = _write_round_tools(context)
                            context.observer.notice(
                                "project_build_failed",
                                {"command": broken, "retry": build_reprompts,
                                 "after": "stall"},
                            )
                            context.trace.emit(
                                EventType.RETRY, context.task_id,
                                status=EventStatus.DETECTED,
                                metadata={"reason": "project_build_failed",
                                          "retry": build_reprompts,
                                          "after": "stall",
                                          "arguments_recorded": False},
                            )

                            continue

                    # ...and nothing was written. The other site checks
                    # that through asked_to_write_and_did_not; this one
                    # checked only what was asked, so a turn that HAD edited
                    # the file and then looped was told "nothing was
                    # changed" -- false, and it sends the model looking for
                    # work it has already done.

                    if (make_reprompts < WRITE_REDIRECT_LIMIT
                            and may_demand_write(context)
                            and not did_modify
                            and not changed_files(tool_log)
                            and wants_write(
                                context, _asked(context.conversation))):
                        make_reprompts += 1
                        reason = RuntimeTerminalReason.COMPLETED
                        context.progress_monitor.record_progress(
                            "write redirect")
                        context.conversation.append(ConversationMessage(
                            "user", (TextBlock(write_demand(
                                _asked(context.conversation))),), authored_by="harness"))
                        only_tools = _write_round_tools(context)
                        last_write_redirect = round_index
                        context.observer.notice(
                            "write_request_unanswered",
                            {"tool_calls": len(tool_log), "after": "stall"},
                        )
                        context.trace.emit(
                            EventType.RETRY, context.task_id,
                            status=EventStatus.DETECTED,
                            metadata={"reason": "write_request_unanswered",
                                      "retry": make_reprompts, "after": "stall",
                                      "tool_calls": len(tool_log)},
                        )

                        continue

                    # The clauses this session established, on the path
                    # that had no way of asking for them. A turn wrote the
                    # one field the change needs, stalled, and ended -- the
                    # tree compiled, so the build gate was silent; something
                    # had been written, so the write redirect was silent;
                    # and the clause redirect lives only at the answer
                    # boundary, which a stall never reaches. Fourth time a
                    # gate has turned out to be unreachable from one exit.

                    carried = list(carried_obligations(
                        context, _asked(context.conversation)))

                    if clause_reprompts < CLAUSE_REDIRECT_LIMIT and carried:
                        # Everything the turn has said so far, not the
                        # final answer -- at a stall there is no final answer
                        # yet, and judging against an empty one declares
                        # every clause unaddressed and fires on a turn that
                        # had accounted for all of them.

                        said = response or "\n".join(transcript)
                        skipped = conformance_mode.unaddressed_clauses(
                            said, carried)

                        if skipped:
                            clause_reprompts += 1
                            last_clause_redirect = round_index
                            reason = RuntimeTerminalReason.COMPLETED
                            context.progress_monitor.record_progress(
                                "clause redirect")
                            context.conversation.append(ConversationMessage(
                                "user", (TextBlock(
                                    conformance_mode.clause_demand(
                                        skipped, carried)),),
                                authored_by="harness"))
                            only_tools = tuple(
                                tool.name for tool in context.tools
                                if not tool.name.startswith("standard."))
                            context.observer.notice(
                                "clauses_unaddressed",
                                {"missing": skipped, "carried": len(carried),
                                 "after": "stall"},
                            )

                            continue

                    labels = {item["label"] for item in missing}
                    progressed = (order_remaining is None
                                  or labels < order_remaining)

                    if (missing and progressed
                            and order_reprompts < ORDER_REDIRECT_LIMIT):
                        order_remaining = labels
                        order_reprompts += 1
                        reason = RuntimeTerminalReason.COMPLETED

                        # The loop that produced the stall is over, and the
                        # turn has been given a new instruction. Leaving the
                        # counter where it is makes the very next action stall
                        # again, which turned the redirect into a formality.

                        context.progress_monitor.record_progress(
                            "work-order redirect")
                        context.conversation.append(ConversationMessage(
                            "user", (TextBlock(work_order.demand(missing)),), authored_by="harness"))
                        context.observer.notice(
                            "work_order_sections_skipped",
                            {"sections": [item["label"] for item in missing],
                             "after": "stall"},
                        )

                        continue

                    # Nothing above took the turn back. Say so, and say what
                    # was true when the decision was made: a turn that ends
                    # on a loop, having been offered three ways out and taken
                    # none, is exactly the case an operator cannot diagnose
                    # from "stalled" alone.

                    context.observer.notice("stalled_without_redirect", {
                        "tool_calls": len(tool_log),
                        "wrote": bool(changed_files(tool_log)) or did_modify,
                        "write_request": wants_write(
                            context, _asked(context.conversation)),
                        "redirects_spent": {"write": make_reprompts,
                                            "build": build_reprompts,
                                            "order": order_reprompts},
                        "asked": _asked(context.conversation)[:80],
                    })

                    break

                names = [call.name for call in calls]

                if any(envelope.mutation for envelope in round_envelopes):
                    did_modify = True

                    # A turn that is writing is not the failure this nudge
                    # exists to stop, and its clock must not keep running. The
                    # nudge's own words are "you can make more small edits
                    # after this one"; with the clock left running the window
                    # shut on the very next round, so a master run made the
                    # first edit of a five-file change -- adding the field the
                    # rest of the change needs -- and was cut off before
                    # threading it anywhere. The promise was false, and the
                    # tree was left half-changed.

                    nudged = False
                    rounds_since_nudge = 0
                    investigate_rounds = 0
                elif names:
                    investigate_rounds += 1

                # Exploration that has stopped paying. The nudge below waits
                # for a fraction of the whole round budget to go by, which is
                # the right ceiling for a turn that is working and much too
                # patient for one that is circling: five calls in a row that
                # add no clause and open no file are not a turn on its way to
                # finding something. This fires on the evidence state itself,
                # so it is as short as it can be without cutting off a turn
                # that is still learning -- three barren calls, and the turn
                # is asked to say what it has and either plan from it or say
                # what it still needs.

                if phase is not None and phase.engaged and phase.repeating:
                    reached = phase.force_synthesis()
                    context.conversation.append(ConversationMessage(
                        "user", (TextBlock(
                            "The last few calls added no requirement and "
                            "opened no file that was not already read. Stop "
                            "searching. State what you have established so "
                            "far on both sides -- what the source requires "
                            "and what the code does -- and then either record "
                            "the plan with plan_change, or name the one thing "
                            "you still need and go straight to it."
                            # Where the work is. A turn that has searched
                            # itself to a standstill has usually been
                            # searching somewhere else: a run spent twenty
                            # calls grepping a registered corpus that shares
                            # a subject with the task and never opened a file
                            # of the tree it was launched in.
                            + (f" The code this session is about is under "
                               f"{context.project_root}; a path outside it is "
                               f"not the implementation you were asked about."
                               if getattr(context, "project_root", "") else "")
                            + turn_evidence(tool_log)),),
                        authored_by="harness"))
                    context.trace.emit(
                        EventType.RETRY, context.task_id,
                        status=EventStatus.DETECTED,
                        metadata={"reason": "exploration_without_evidence",
                                  "phase": str(reached),
                                  "retry": phase.syntheses_forced},
                    )
                    context.observer.notice(
                        "exploration_without_evidence",
                        {"phase": str(reached),
                         "forced": phase.syntheses_forced},
                    )

                # Source changed, proof still owed. The turn is steered to
                # the validation it planned rather than left to remember: in
                # four measured runs the first source edit came at call 37,
                # 73, 101 and 33, and the first test edit at 125, 102, never
                # and 51. Test authoring was always an afterthought, and twice
                # it never arrived at all -- once leaving the tree not
                # building, because the signature change never reached the
                # callers in the test file nobody opened.

                if (phase is not None and phase.engaged
                        and not force_final):
                    owed = phase.awaiting_validation()
                    broken, _ = project_build_gap(context, tool_log)

                    if broken and owed and validation_demands < 2:
                        validation_demands += 1
                        context.conversation.append(ConversationMessage(
                            "user", (TextBlock(
                                f"The project does not build after this "
                                f"change, so nothing it was meant to prove is "
                                f"proved: `{broken[:90]}` fails. Go back to "
                                f"the work item that broke it and repair it "
                                f"before anything else -- "
                                + ", ".join(found.key for found in owed[:4])
                                + ". Do not open new ground while the build "
                                  "is down."),), authored_by="harness"))
                        only_tools = _write_round_tools(context)
                        context.observer.notice(
                            "build_broken_work_item",
                            {"command": broken[:60],
                             "requirements": len(owed)})
                    elif owed and not broken and validation_demands < 2:
                        validation_demands += 1
                        planned = [found.key for found in owed]
                        context.conversation.append(ConversationMessage(
                            "user", (TextBlock(
                                "The source is changed and the validation you "
                                "planned for it does not exist yet: "
                                + ", ".join(planned[:4])
                                + ". Write that test now, in the project's own "
                                  "test files, following the fixtures, helpers "
                                  "and registration the neighbouring tests "
                                  "use. Then run the project's build and "
                                  "tests. Do not start new investigation "
                                  "until the change you already made is "
                                  "proved."),), authored_by="harness"))
                        only_tools = _write_round_tools(context)
                        context.observer.notice(
                            "validation_owed", {"requirements": len(planned)})

                # One change is finished and others are still open. The
                # turn is pointed at the next piece of work rather than left
                # to rediscover that there is any: the gate is local now, so
                # nothing else was ever going to remind it.

                if (phase is not None and phase.engaged and not force_final
                        and not phase.contract_closed()
                        and phase.work_items
                        and not phase.open_work_items()
                        and next_item_demands < 2):
                    pending = phase.next_open_requirements()

                    if pending:
                        next_item_demands += 1
                        context.conversation.append(ConversationMessage(
                            "user", (TextBlock(
                                "That change is made, built and proved. It is "
                                "finished -- do not revisit it. "
                                + f"{len(pending)} requirement(s) carried into "
                                  "this turn are still open and nothing has "
                                  "been planned for them: "
                                + ", ".join(found.key for found in pending[:4])
                                + ". Take the next one: read what the code "
                                  "does about it, then plan that change on "
                                  "its own."),), authored_by="harness"))
                        context.observer.notice(
                            "next_work_item", {"open": len(pending)})

                # The contract is closed and every change it names is
                # validated. There is nothing left to establish, and a turn
                # that keeps looking is spending the operator's time proving
                # what it has already proved -- a real run re-dispositioned
                # rules it had closed, for minutes, after its own tests went
                # green.

                if (phase is not None and phase.engaged
                        and phase.contract_closed() and not landed
                        and not force_final):
                    landed = True
                    context.conversation.append(ConversationMessage(
                        "user", (TextBlock(
                            "Every requirement carried into this turn now has "
                            "a disposition, nothing is left open, and the "
                            "changes are covered by validation that ran. "
                            "There is nothing further to establish. Report "
                            "what was done and stop."),), authored_by="harness"))
                    context.observer.notice("contract_closed", {})

                # Both halves in hand, nothing planned, and the reading goes
                # on. Not a gate -- the write gate is the gate -- but the
                # moment INVESTIGATE is over should be visible from inside the
                # turn, and without something that says so "read a bit more"
                # is always the easier move.

                if phase is not None and phase.owes_a_plan():
                    context.conversation.append(ConversationMessage(
                        "user", (TextBlock(
                            "You have now read both the requirements and the "
                            "implementation. Before any file changes, record "
                            "the plan with plan_change: one entry per "
                            "requirement you intend to satisfy, each naming "
                            "the provision and where you read it, what the "
                            "code does now and in which file, the gap, the "
                            "change you intend, and the test that will prove "
                            "it. The tools that modify files stay refused "
                            "until one entry is accepted."),),
                        authored_by="harness"))
                    context.trace.emit(
                        EventType.RETRY, context.task_id,
                        status=EventStatus.DETECTED,
                        metadata={"reason": "plan_owed",
                                  "retry": phase.plan_demands},
                    )
                    context.observer.notice("plan_owed",
                                            {"demand": phase.plan_demands})

                    # The last time of asking. After it the round is narrowed
                    # to the one call that can move the turn on: a demand the
                    # model can ignore while still calling `bash` is a demand
                    # it does ignore -- measured on a run that was asked
                    # twice, read for another twenty minutes and never
                    # planned at all. Same lever, same reason, as the write
                    # redirect: the demand is a request, the tool list is not.

                    if phase.plan_demand_ignored:
                        only_tools = (_PLAN_TOOL,)

                # Reading forever without changing anything is the failure mode
                # this catches: a round that only investigates counts towards a
                # single nudge to conclude, which is issued at most once.
                #
                # The count is a fraction of the budget, not a constant. Twelve
                # was calibrated against a sixty-round turn; a write request now
                # gets twice that, and the flat twelve cut off a turn eleven
                # reads into a five-file change -- before its first edit, with
                # nine tenths of its budget unspent. A turn given more room to
                # work is allowed more room to look.

                if (not did_modify and not nudged and not force_final
                        and (repeats >= 2
                             or investigate_rounds >= _investigation_ceiling(
                                 round_limit))):
                    nudged = True

                    context.conversation.append(ConversationMessage(
                        "user", (TextBlock(
                            conclude_demand(
                                _asked(context.conversation),
                                wants_write(context, _asked(context.conversation)))
                            + turn_evidence(tool_log)
                        ),), authored_by="harness"))
                    context.trace.emit(
                        EventType.RETRY, context.task_id,
                        status=EventStatus.DETECTED,
                        metadata={"reason": "investigation_nudge", "retry": 1,
                                  "investigate_rounds": investigate_rounds},
                    )
                    context.observer.notice(
                        "investigation_nudge",
                        {"investigate_rounds": investigate_rounds,
                         "repeated_actions": repeats},
                    )

            # The loop can run out of rounds with work done but nothing said.
            # One tool-free turn turns that into an answer for the user.

            if not response and (transcript or tool_log):
                context.conversation.append(ConversationMessage(
                    "user", (TextBlock(
                        "Now summarize your findings and give the final answer / fix. "
                        "No tool is available for this turn: do not call or write one."
                        + turn_evidence(tool_log)
                    ),), authored_by="harness"))
                self._enter_finalizing(context, "rounds exhausted", announce=False)

                failure_stage = "model"

                with context.observer.model_activity("Concluding…") as on_token:
                    final_turn = self._finalized(context, self._complete_with_retry(
                        context, use_tools=False, on_token=on_token,
                    ))

                failure_stage = "turn_validation"

                if final_turn is None:
                    reason = RuntimeTerminalReason.STALLED
                    response = finalization_failed_note(context)
                else:
                    validate_agent_turn(final_turn)
                    response = strip_fabrications(final_turn.text).strip()

                # Even the conclusion can come back empty; the tool log is then
                # the only honest thing left to hand the user.

                if not response:
                    context.observer.notice(
                        "empty_final_synthesis", {"tool_call_count": len(tool_log)},
                    )
                    response = no_conclusion_note(tool_log)
        except BudgetExceeded as exc:
            budget_event_emitted = True

            if context.working_state.terminal_status == TerminalStatus.RUNNING:
                context.apply_state_event(
                    StateEventType.BUDGET_EXHAUSTED,
                    summary=str(exc), reason=exc.kind.value,
                )

            context.trace.emit(
                EventType.BUDGET_EXHAUSTED, context.task_id,
                session_id=context.session_id, status=EventStatus.EXHAUSTED,
                metadata={"component": exc.component, "reason": exc.kind.value,
                          "raw_content_recorded": False},
            )

            self._persist_session(
                context, SessionEventType.BUDGET_UPDATED,
                {"component": exc.component, "reason": exc.kind.value,
                 "exhausted": True},
            )

            # A tree left broken is worse than a budget overrun by one
            # round. The build gate hangs off the answer boundary and off
            # the stall, and a budget exit passes through neither: measured
            # today, a master run made the right edit at minute nineteen of
            # thirty, left a symbol undefined, and the budget ended the turn
            # with ten minutes to spare and nothing asking it to finish.
            #
            # So the repair gets the same courtesy the summary below gets --
            # one round, outside the budget, with only the writing tools on
            # the table. Once. A budget that has already ended the turn is
            # not an invitation to keep going.

            broken, output = project_build_gap(context, tool_log)

            if broken and build_reprompts < BUILD_REPROMPT_LIMIT:
                build_reprompts += 1
                context.conversation.append(ConversationMessage(
                    "user", (TextBlock(project_build.demand(
                        broken, output, last_build_output)),),
                    authored_by="harness"))
                context.observer.notice(
                    "project_build_failed",
                    {"command": broken, "retry": build_reprompts,
                     "after": "budget"},
                )

                try:
                    with context.observer.model_activity("Repairing the build…") as tick:
                        repair = self.complete_model_turn(
                            context, use_tools=True, on_token=tick,
                            only_tools=_write_round_tools(context), grace=True)

                    for call in repair.tool_calls:
                        envelope = context.tool_executor(
                            context, call.id, call.name, call.arguments, cache)
                        tool_log.append(f"{call.name} {json.dumps(call.arguments)[:200]}")

                        if envelope.mutation:
                            did_modify = True
                except Exception:
                    # The repair is a courtesy too; its failure must not
                    # replace the budget's own account of the turn.
                    pass

            # A budget that ends a turn must not also erase it. One
            # tool-less call, outside the budget, asking the model to say
            # what it did -- the alternative, measured today, is a turn that
            # edited five files and handed back "I stopped after 125 tool
            # calls without reaching a conclusion".

            if not (response or "").strip() and tool_log:
                context.conversation.append(ConversationMessage(
                    "user", (TextBlock(
                        "You have run out of room for this turn. Do not call "
                        "any tool. Say what you changed, what you verified, "
                        "and what remains -- briefly, and only what actually "
                        "happened." + turn_evidence(tool_log)),),
                    authored_by="harness"))

                self._enter_finalizing(context, "budget exhausted", announce=False)

                try:
                    with context.observer.model_activity("Wrapping up…") as tick:
                        summary = self.complete_model_turn(
                            context, use_tools=False, on_token=tick, grace=True)

                    # The grace call is outside the budget; a written tool
                    # call in it is a missing summary, not one to retry.
                    if not (summary.tool_calls or textual_tool_call(summary.text)):
                        response = strip_fabrications(summary.text).strip() or response
                except Exception:
                    # The grace call is a courtesy; its failure must not
                    # replace the budget's own account of the turn.
                    pass

            return self._result(
                context, RuntimeTerminalReason.BUDGET_EXHAUSTED, response,
                transcript, tool_log, trajectory, did_modify, True,
                verification_action_id, cache,
                error_category="budget_exhausted", error_summary=str(exc),
            )
        except (KeyboardInterrupt, OperationCancelled) as exc:
            summary = (exc.reason if isinstance(exc, OperationCancelled)
                       else "interrupted by user")

            try:
                context.apply_state_event(
                    StateEventType.TASK_INTERRUPTED, summary=summary,
                )
            except SessionError:
                # Cancellation remains truthful even if durable persistence is
                # unavailable; the failure is observable in the returned result.

                context.session = None

                if context.working_state.terminal_status == TerminalStatus.RUNNING:
                    context.apply_state_event(
                        StateEventType.TASK_INTERRUPTED, summary=summary,
                    )

            context.trace.emit(
                EventType.TASK_INTERRUPTED, context.task_id,
                status=EventStatus.CANCELLED,
                provider=context.provider, model=context.model,
                session_id=context.session_id,
                metadata={"tool_call_count": len(tool_log)},
            )
            context.trace.emit(
                EventType.CANCELLATION_OBSERVED, context.task_id,
                session_id=context.session_id, status=EventStatus.CANCELLED,
                error_category=FailureKind.INTERRUPTED.value,
                error_summary=summary,
                metadata={"scope": (exc.scope.value if isinstance(exc, OperationCancelled)
                                    else "task")},
            )

            self._persist_session(
                context, SessionEventType.TASK_INTERRUPTED,
                {"reason": summary},
            )

            return self._result(
                context, RuntimeTerminalReason.INTERRUPTED, response,
                transcript, tool_log, trajectory, did_modify,
                budget_event_emitted, verification_action_id, cache,
                error_category=FailureKind.INTERRUPTED.value,
                error_summary=summary,
            )
        except Exception as exc:
            # Persistence failures are different from tracing failures: they
            # are surfaced as runtime failure. Detach the failed handle so the
            # in-memory task can be terminated truthfully without recursively
            # attempting the same broken store operation.

            if isinstance(exc, SessionError):
                context.session = None

            # An exception with no message -- raise SomeError() -- used to
            # reach the user as the bare words "runtime failure": the class
            # was recorded, and the one line that showed it was not. Name the
            # class and where it was raised, so a failed turn is diagnosable
            # from the terminal alone. Tracing is off by default, and a
            # persistence failure is exactly the case that also loses the
            # session record, so this string is sometimes the only evidence.

            detail = str(exc).strip()

            if not detail:
                frames = traceback.extract_tb(exc.__traceback__)

                # rsplit rather than os.path.basename: this module imports no
                # os and touches no filesystem, and one file name is not a
                # reason to give it either.

                where = (f" at {frames[-1].filename.rsplit('/', 1)[-1]}:"
                         f"{frames[-1].lineno}" if frames else "")
                detail = f"{type(exc).__name__}{where}"

            detail = detail[:240]

            if context.working_state.terminal_status == TerminalStatus.RUNNING:
                context.apply_state_event(
                    StateEventType.TASK_FAILED, summary=detail,
                )

            # The stage recorded as the loop advanced says whether the model,
            # its answer, or the runtime itself is what actually broke.

            reason = (RuntimeTerminalReason.MODEL_FAILURE if failure_stage == "model"
                      else RuntimeTerminalReason.INVALID_TURN
                      if failure_stage == "turn_validation"
                      else RuntimeTerminalReason.RUNTIME_FAILURE)

            context.trace.emit(
                EventType.TASK_FINISHED, context.task_id,
                status=EventStatus.FAILED,
                provider=context.provider, model=context.model,
                error_category=type(exc).__name__,
                error_summary=detail,
                metadata={"tool_call_count": len(tool_log),
                          "failure_stage": failure_stage},
            )

            self._persist_session(
                context, SessionEventType.TASK_FAILED,
                {"failure_stage": failure_stage,
                 "error_category": type(exc).__name__},
            )

            return self._result(
                context, reason, response, transcript, tool_log, trajectory,
                did_modify, budget_event_emitted, verification_action_id, cache,
                error_category=type(exc).__name__, error_summary=detail,
            )

        # A stall leaves the loop normally, so the task is failed here rather
        # than in an except branch it never reached.

        if (reason == RuntimeTerminalReason.STALLED
                and context.working_state.terminal_status == TerminalStatus.RUNNING):
            context.apply_state_event(
                StateEventType.TASK_FAILED,
                summary="stalled after repeated actions without grounded progress",
            )
            self._persist_session(
                context, SessionEventType.TASK_FAILED,
                {"failure_kind": FailureKind.STALLED.value},
            )

        result = self._result(
            context, reason, response, transcript, tool_log, trajectory,
            did_modify, budget_event_emitted, verification_action_id, cache,
            completion_deferred=defer_completion,
        )

        # A deferred completion leaves the task and its checkpoint open for the
        # controller to review, repair, or roll back before closing them.

        if not defer_completion:
            self.complete(context, result)

        return result
