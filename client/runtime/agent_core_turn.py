"""Implementation turns, run by the agent core and recorded by SPEAR."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from runtime.cancellation import OperationCancelled
from runtime.failure_policy import FailureKind
from runtime.session_store import SessionEventType
from models.model_backend import TextBlock
from runtime.tracing import EventStatus, EventType, new_action_id
from harness.tool_router import ToolResultEnvelope, ToolResultStatus
from harness.tool_registry import COMMAND_TOOLS
from harness import refusal_breaker
from runtime.working_state import StateEventType, TerminalStatus
from runtime.agent_context import AgentContext, AgentResult, RuntimeTerminalReason
from runtime.agent_verification import project_build_runs


# The most one model response may generate. The core reserves far more for
# its context arithmetic, and a response that degenerates -- one tool call
# whose arguments repeat until the limit -- ran to that reservation: 65536
# tokens and eleven minutes for one call. The longest legitimate response
# measured, a whole-file write, was under 11000. A response cut here ends as
# truncated, which the core already retries and then reports.
RESPONSE_MAX_TOKENS = int(os.environ.get("SPEAR_RESPONSE_MAX_TOKENS", "16384"))


def _core_envelope(record) -> ToolResultEnvelope:
    """An agent-core call as SPEAR's working state and evidence record it."""
    status = (ToolResultStatus.DENIED if record.refused
              else ToolResultStatus.TIMEOUT if record.timed_out
              else ToolResultStatus.OK if record.ok else ToolResultStatus.FAILED)
    text = record.result or ""

    return ToolResultEnvelope(
        record.call_id, new_action_id(record.name), record.name, record.ok, status,
        text, text, "command" if record.name in COMMAND_TOOLS else "file",
        0.0, len(text.encode("utf-8", "replace")), exit_code=record.exit_code,
        mutation=bool(record.changed_paths), affected_paths=tuple(record.changed_paths),
        read_paths=tuple(record.read_paths),
        error_category=None if record.ok else status.value,
        error_summary=None if record.ok else text[:240])


def _core_history(conversation) -> tuple[list[dict], str]:
    """Earlier turns as plain {"role", "content"} messages, and the request."""
    turns = []

    for message in conversation:
        content = "".join(block.text for block in message.content
                          if isinstance(block, TextBlock)).strip()

        if content and message.role in ("user", "assistant"):
            turns.append({"role": message.role, "content": content})

    request = turns.pop()["content"] if turns and turns[-1]["role"] == "user" else ""

    return turns, request


def _generated_in(root):
    """Whether a changed path is a build's output rather than delivered source:
    the control plane's generated trees, or a path the project's git ignores.
    Tracked files are source whatever the ignore rules say."""
    import functools

    from evidence import completion

    @functools.lru_cache(maxsize=None)
    def generated(path):
        if completion.generated_path(path):
            return True

        try:
            done = subprocess.run(["git", "-C", root, "check-ignore", "-q", "--", path],
                                  capture_output=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return False

        return done.returncode == 0

    return generated


def _relative_to(context):
    root = getattr(context, "project_root", "") or os.getcwd()

    def label(path):
        path = str(path)
        return os.path.relpath(path, root) if path.startswith(root.rstrip("/") + "/") else path

    return label



class CodingCoreMixin:
    """Implementation turns, run by the agent core (agent/)."""

    # ------------------------------------------------------------------
    # The agent core (agent/), for implementation turns.
    #
    # The core runs the turn: its prompt, its tools, its loop. This method
    # only connects it to SPEAR: it supplies the control plane as the
    # core's host, records every call into the working state, the evidence
    # and the canonical tool log, and gives the answer SPEAR's verdict.

    def _run_core(self, context: AgentContext, *,
                  defer_completion: bool = False) -> AgentResult:
        from agent import loop as core_loop
        from agent import prompt as core_prompt
        from evidence import completion

        transcript: list[str] = []
        tool_log = context.session_tool_log
        trajectory = context.session_trajectory
        cache: dict[Any, Any] = {}
        did_modify = False

        if context.core_evidence is None:
            context.core_evidence = []

        breaker = refusal_breaker.RefusalBreaker()

        def record(item):
            nonlocal did_modify
            breaker.observe(item.name, dict(item.arguments), item.result,
                            item.refused or (not item.ok and "refused" in item.result[:300]))
            envelope = _core_envelope(item)
            self._record_tool_result(context, envelope)
            evidence = context.verification_policy.evidence_for_tool(
                envelope, dict(item.arguments), context.working_state)

            if evidence is not None:
                self._record_verification(context, evidence)

            if item.changed_paths:
                did_modify = True

            tool_log.append(completion.canonical_entry(item, _relative_to(context)))
            context.core_evidence.append(completion.evidence(item, _relative_to(context)))
            trajectory.append({"tool": item.name, "arguments": dict(item.arguments),
                               "result": item.result[:4000],
                               "action_id": envelope.action_id,
                               "status": envelope.status.value})
            self._persist_session(
                context, SessionEventType.TOOL_ACTION_COMPLETED,
                {"tool_call_id": item.call_id, "action_id": envelope.action_id,
                 "status": envelope.status.value})

        self._start_task(context)
        definitions = [{"type": "function", "function": {
            "name": tool.name, "description": tool.description,
            "parameters": dict(tool.input_schema)}} for tool in context.tools]
        names = [item["function"]["name"] for item in definitions]
        backend = context.backend
        system = core_prompt.build(
            cwd=getattr(context, "project_root", "") or os.getcwd(),
            tool_names=names, model=getattr(backend, "model", None) or context.model or "",
            project_rules=context.coding_context)
        history, request = _core_history(context.conversation)
        host = context.coding_host(context, cache, record)

        def model(messages, tools, max_tokens):
            label = "Thinking…" if len(messages) <= 2 + len(history) else "Analyzing results…"

            with context.observer.model_activity(label) as tick:
                return backend.complete_messages(
                    messages, tools, max_tokens=min(max_tokens, RESPONSE_MAX_TOKENS),
                    on_token=tick)

        # The core reserves MAX_TOKENS of output and stops for a summary once a
        # prompt passes half of what is left. Against a window that small
        # reservation does not fit -- the 32768-token default when a server
        # states none -- that threshold is negative, and every turn stopped
        # after its first reply. Such a window gets a quarter of itself.

        window = context.context_limit
        max_tokens = (core_loop.MAX_TOKENS if not window or window > 2 * core_loop.MAX_TOKENS
                      else window // 4)

        try:
            core = core_loop.run(
                model=model, host=host, system=system, history=history,
                request=request, tool_definitions=definitions,
                max_iterations=context.max_model_rounds,
                max_tool_calls=context.max_tool_actions,
                context_window=window, max_tokens=max_tokens,
                cancelled=lambda: (context.cancellation.is_cancelled
                                   or breaker.tripped is not None))
        except (KeyboardInterrupt, OperationCancelled) as exc:
            summary = (exc.reason if isinstance(exc, OperationCancelled)
                       else "interrupted by user")

            if context.working_state.terminal_status == TerminalStatus.RUNNING:
                context.apply_state_event(StateEventType.TASK_INTERRUPTED, summary=summary)

            return self._result(context, RuntimeTerminalReason.INTERRUPTED, "",
                                transcript, tool_log, trajectory, did_modify, False,
                                None, cache, error_category=FailureKind.INTERRUPTED.value,
                                error_summary=summary)

        if breaker.tripped is not None and core.stop == "cancelled" \
                and not context.cancellation.is_cancelled:
            core.stop, core.final = "halted", breaker.conclusion()
            name, target, refusal = breaker.tripped
            context.trace.emit(EventType.REPEATED_REFUSAL_STOPPED, context.task_id,
                               status=EventStatus.OK,
                               metadata={"tool": name, "target": target[:200],
                                         "refusal": refusal,
                                         "count": breaker.counts[breaker.tripped],
                                         "limit": breaker.limit})

        reason = {
            "completed": RuntimeTerminalReason.COMPLETED,
            "iterations": RuntimeTerminalReason.ROUND_BUDGET_EXHAUSTED,
            "tool_budget": RuntimeTerminalReason.TOOL_BUDGET_EXHAUSTED,
            "context": RuntimeTerminalReason.BUDGET_EXHAUSTED,
            "halted": RuntimeTerminalReason.STALLED,
            "truncated": RuntimeTerminalReason.INVALID_TURN,
            "model_error": RuntimeTerminalReason.MODEL_FAILURE,
            "cancelled": RuntimeTerminalReason.INTERRUPTED,
        }[core.stop]

        if reason in (RuntimeTerminalReason.STALLED, RuntimeTerminalReason.MODEL_FAILURE,
                      RuntimeTerminalReason.INVALID_TURN, RuntimeTerminalReason.INTERRUPTED) \
                and context.working_state.terminal_status == TerminalStatus.RUNNING:
            context.apply_state_event(StateEventType.TASK_FAILED,
                                      summary=core.error or core.stop)

        root = getattr(context, "project_root", "") or os.getcwd()
        commands = getattr(context, "project_commands", None)
        epoch = completion.timeline(context.core_evidence, _generated_in(root), root)[0]
        project = completion.project_evidence(
            project_build_runs(context, tool_log), commands, epoch)
        verdict = completion.decide(
            context.core_evidence, project_runs=project, project_commands=commands,
            answer=core.final, generated=_generated_in(root), root=root)
        context.core_verdict = verdict
        final = completion.qualify(core.final, verdict)

        result = self._result(
            context, reason, final, transcript, tool_log, trajectory, did_modify,
            reason in (RuntimeTerminalReason.ROUND_BUDGET_EXHAUSTED,
                       RuntimeTerminalReason.TOOL_BUDGET_EXHAUSTED,
                       RuntimeTerminalReason.BUDGET_EXHAUSTED),
            None, cache, error_category=core.error and "model_error",
            error_summary=core.error, completion_deferred=defer_completion)

        if not defer_completion:
            self.complete(context, result)

        return result
