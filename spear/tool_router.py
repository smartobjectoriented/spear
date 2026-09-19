"""Generic provider-neutral tool execution and structured result lifecycle."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Callable, Mapping

from cancellation import CancellationToken, NEVER_CANCELLED, OperationCancelled
from checkpoint import Checkpoint, CheckpointManager
from hooks import HookManager, ToolHookEvent
from result_store import ResultStore
from tool_registry import (
    ToolCategory, ToolMutability, ToolRegistry, ToolSpec)
from tracing import EventStatus, EventType, TraceEmitter, new_action_id


class ToolResultStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DENIED = "denied"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNKNOWN_TOOL = "unknown_tool"
    TIMEOUT = "timeout"
    CACHED = "cached"


@dataclass(frozen=True)
class ToolHandlerResult:
    """What a handler returns: the output plus what it observed doing."""

    text: str | bytes
    status: ToolResultStatus = ToolResultStatus.OK
    exit_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    mutation: bool = False
    affected_paths: tuple[str, ...] = ()
    read_paths: tuple[str, ...] = ()
    error_category: str | None = None
    error_summary: str | None = None
    #: What the MODEL is shown, when that differs from what is kept. `text`
    #: stays the whole structured result -- stored, traced, and the identity
    #: the repeat ledger hashes -- while a handler that can present its own
    #: result better than a blind size cap does so here. Left unset, the two
    #: are the same and nothing changes. Last in the field order on purpose:
    #: every existing construction is positional.
    model_text: str | None = None


@dataclass(frozen=True)
class ToolResultEnvelope:
    """A handler result once the router has traced, sized and stored it.

    ``text`` is the whole output; ``model_content`` is what fits in the model's
    context, which for a large result is a preview naming ``result_reference``.
    """

    tool_call_id: str
    action_id: str
    tool_name: str
    success: bool
    status: ToolResultStatus
    text: str
    model_content: str
    category: str
    duration_seconds: float
    output_bytes: int
    exit_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    truncated_for_model: bool = False
    mutation: bool = False
    affected_paths: tuple[str, ...] = ()
    read_paths: tuple[str, ...] = ()
    error_category: str | None = None
    error_summary: str | None = None
    result_reference: str | None = None


CommandExecutor = Callable[[str], ToolHandlerResult | str]

#: Outcomes where the tool never ran. Nothing was written, so nothing is
#: stale: a refused `sed -i` must not cost the turn every observation it had.

NOT_EXECUTED = frozenset({
    ToolResultStatus.DENIED,
    ToolResultStatus.CANCELLED,
    ToolResultStatus.INVALID_ARGUMENTS,
    ToolResultStatus.UNKNOWN_TOOL,
})


def invalidates_reads(status: ToolResultStatus, mutation: bool,
                      metadata: Mapping[str, object] | None = None) -> bool:
    """Whether this outcome makes earlier reads stale.

    ONE answer, for the router's generation counter and for the caller's read
    cache alike. Two rules drifted apart here once: the cache dropped its
    reads for any executed non-read-only command while the generation moved
    only on success, so a command that wrote half a file and exited 1 left
    the cache empty and the generation unchanged -- and the fresh read that
    followed carried the same registry key as the stale one it replaced.

    The source of truth is the mutation, not the exit code:

    * never executed -> nothing changed;
    * a mutation observed, or ``mutation_count`` above zero -> changed,
      whatever the exit code;
    * ``mutation_count`` present and zero -> measured, and nothing changed;
    * no measurement available -> ``potentially_mutating`` decides, and a
      command that could have written is treated as though it did.
    """

    metadata = metadata or {}

    if status in NOT_EXECUTED:
        return False

    count = metadata.get("mutation_count")

    if mutation or (isinstance(count, int) and count > 0):
        return True

    if isinstance(count, int):
        return False

    return bool(metadata.get("potentially_mutating"))

# How many times the same call may fail before the router stops running it.
# Twice: once is a mistake, twice confirms it, and a third is the model
# hoping the world changed. A run made the SAME failing call twelve times --
# fetch_url({"url": "https://example.com/main.py", "save_as": "main.py"}) --
# and nothing stopped it, because dedup existed only for bash commands and
# only for ones that had succeeded.

MAX_FAILED_REPEATS = 2

# And how many times the same call may be answered from cache. Once: the
# first repeat is allowed and recorded, because a model may genuinely have
# lost the output; the second re-injects nothing, because the answer has now
# been in front of it twice.
#
# What this saves is MODEL CALLS, not the bytes of the repeated result. A
# read-only turn ran `cat main.py` once, was handed it back from cache six
# more times, and finished at nineteen model calls -- and every one of those
# calls re-sent the whole fixed context. The 316k tokens that run reported are
# the cumulative total across those nineteen requests, which six small `cat`
# outputs do not begin to explain. Ending the loop earlier is what removes
# them.

MAX_CACHED_REPEATS = 1

#: Where the per-task tally lives inside the execution cache. A sentinel
#: object rather than a string, because the caller clears the cache's string
#: keys after a mutation -- and a local write says nothing about whether a
#: network call is still going to fail.
_REPEAT_KEY = object()


def action_signature(name: str, arguments: Mapping[str, object] | object,
                     generation: int = 0) -> str:
    """The fingerprint of one call: its tool, its arguments, its generation.

    Canonical JSON, so `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` are the same
    action. Anything unserializable falls back to its repr, which is still
    stable for the same call repeated.

    The generation is what makes a real write -- and only a real write --
    forget an observation: reading a file again after changing it is a
    different action from reading it before. Everything else the turn does in
    between is irrelevant, which is the whole point. A run alternated
    `cat`, `grep`, `sed -n`, `cat -n` and came back to `cat` four times over
    thirty-nine model calls; not one of those was new information.
    """

    try:
        rendered = json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        rendered = repr(arguments)

    return f"{generation}:{name}:{rendered}"


@dataclass
class ToolExecutionContext:
    task_id: str
    trace: TraceEmitter
    cache: dict[Any, Any]
    result_store: ResultStore | None = None
    command_executor: CommandExecutor | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    cancellation: CancellationToken = field(default_factory=lambda: NEVER_CANCELLED)
    checkpoint_manager: CheckpointManager | None = None
    checkpoint: Checkpoint | None = None
    action_id: str | None = None
    role: str = "main"
    #: Given the tool_call_id of an earlier result, whether that result is
    #: still in the conversation the next model request will carry. Supplied
    #: by the caller, because only it knows what compaction and the context
    #: composer have left in. Absent means "assume it is still there".
    evidence_available: Callable[[str], bool] | None = None
    #: The request forbade changing anything. The command boundary drops
    #: workspace write for the whole turn, so a prohibition the tool view
    #: honours cannot be routed around with `sed -i`, `cp` or `>`.
    read_only: bool = False

    #: Whether a mutating action may run yet, asked fresh at every call.
    #:
    #: Supplied by the caller because only it holds the turn's lifecycle. It
    #: returns an object with `allowed`, `reason` and `message`; anything
    #: falsy here means no gate, which is every turn the lifecycle does not
    #: govern. A callable rather than a flag on purpose: the answer changes
    #: DURING the turn, as evidence arrives and a plan is accepted, and a
    #: value sampled when the context was built would be the answer to a
    #: question asked before the turn started.
    write_gate: Callable[[], Any] | None = None

    #: The session's permission mode, as its own name: "safe", "ask", "auto".
    #: A plain string rather than the enum so the routing layer keeps no
    #: dependency on the command runtime that owns it; ExecutionMode is a
    #: StrEnum, so a caller may pass the member itself.
    #:
    #: Empty means "not stated", and a caller that does not state it gets
    #: every declaration honoured as before -- the gate below can only refuse
    #: a mode it was actually told about.
    execution_mode: str = ""


class ToolRouter:
    """Routes registered calls; concrete semantics remain in handlers."""

    def __init__(
        self, registry: ToolRegistry, *, hooks: HookManager | None = None,
    ) -> None:
        self.registry = registry
        self.hooks = hooks or HookManager()

    def execute(
        self, context: ToolExecutionContext, tool_call_id: str,
        name: str, arguments: Mapping[str, object],
    ) -> ToolResultEnvelope:
        action_id = new_action_id(name)
        started = time.monotonic()

        # The gates below all fail before the span opens, so a call that never
        # ran is never traced as one that did.

        if context.cancellation.is_cancelled:
            return self._early_failure(
                context, tool_call_id, action_id, name, started,
                ToolResultStatus.CANCELLED, "interrupted",
                "ERROR: tool execution cancelled before start", "other", (),
            )

        try:
            spec = self.registry.get(name)
        except KeyError:
            return self._early_failure(
                context, tool_call_id, action_id, name, started,
                ToolResultStatus.UNKNOWN_TOOL, "unknown_tool",
                f"ERROR: unknown tool: {name}", "other", (),
            )

        if context.role not in spec.roles:
            return self._early_failure(
                context, tool_call_id, action_id, name, started,
                ToolResultStatus.DENIED, "permission_denied",
                f"ERROR: tool '{name}' is unavailable to role '{context.role}'",
                spec.category.value, (),
            )

        # A turn the user told not to change anything cannot change anything,
        # whoever is asking. Withholding the mutating tools from the MODEL was
        # enough for as long as the model was the only caller; it is not. The
        # harness itself converts a printed file into a write, and that path
        # calls the router directly -- so a turn whose tool view held seven
        # read-only tools still executed a write_file nobody had offered.
        #
        # The refusal names the request, not the mode, and offers no other way
        # to write: there is none, and pointing at one is how a read-only turn
        # spends its remaining rounds.

        if context.read_only and spec.mutability == ToolMutability.MUTATING:
            return self._early_failure(
                context, tool_call_id, action_id, name, started,
                ToolResultStatus.DENIED, "read_only_task",
                f"ERROR: this task is read-only -- you were asked not to "
                f"change anything, so '{name}' is refused. Nothing in this "
                f"turn can write: answer from what you have read.",
                spec.category.value, (),
            )

        # And the other reason a write may not run yet: the turn has not
        # established what it is changing or why. The read-only gate above
        # answers "was this turn allowed to write at all?"; this one answers
        # "has it earned the right yet?", and the difference matters in the
        # refusal -- one is permanent and the other names the missing work.
        #
        # Refused rather than withheld. A tool that disappears from the view
        # reads to the model as a capability it does not have, and a turn that
        # believes it cannot write describes the change instead of making it.
        # A refusal that says what is missing is an instruction, and it
        # arrives through the channel the model is already reading.

        if context.write_gate is not None and spec.mutability == ToolMutability.MUTATING:
            decision = context.write_gate()

            if decision is not None and not getattr(decision, "allowed", True):
                return self._early_failure(
                    context, tool_call_id, action_id, name, started,
                    ToolResultStatus.DENIED,
                    getattr(decision, "reason", "") or "investigation_incomplete",
                    "ERROR: " + (getattr(decision, "message", "")
                                 or "investigation incomplete"),
                    spec.category.value, (),
                )

        # The mode the session runs in, against the modes this tool declares
        # it works in. One invariant, checked once, for every tool: the
        # alternative was a local check inside each mutating handler, which
        # is a rule a new tool can forget -- and `save_skill` already had,
        # relying on its own authorization call instead of its declaration.
        #
        # `execution_modes` is authoritative and is not read as a synonym for
        # mutability: a read-only tool may restrict itself, and a mutating
        # tool that declares nothing is not refused here (it is refused, as
        # it always was, when it asks for authorization). An empty tuple is
        # what every read-only spec carries today, so it keeps meaning
        # "unrestricted"; reading it as "denied everywhere" would disable
        # search and retrieval in every session.

        if (spec.execution_modes and context.execution_mode
                and context.execution_mode not in spec.execution_modes):
            return self._early_failure(
                context, tool_call_id, action_id, name, started,
                ToolResultStatus.DENIED, "execution_mode_denied",
                f"ERROR: '{name}' is unavailable in "
                f"{context.execution_mode} mode.",
                spec.category.value, (),
            )

        # Only the argument NAMES ever reach a trace or a hook; the values may
        # hold file contents or credentials and are never recorded.

        argument_keys = (tuple(sorted(str(key) for key in arguments))
                         if isinstance(arguments, Mapping) else ())

        validation_error = _validate_arguments(spec, arguments)

        if validation_error:
            return self._early_failure(
                context, tool_call_id, action_id, name, started,
                ToolResultStatus.INVALID_ARGUMENTS, "invalid_arguments",
                f"ERROR: invalid arguments for {name}: {validation_error}",
                spec.category.value, argument_keys,
            )

        # Before anything runs: has this exact call already failed twice, or
        # already been answered from cache? The fingerprint covers the
        # arguments, so a different URL or a different file is a different
        # action and only the identical repetition is stopped.

        ledger = context.cache.setdefault(
            _REPEAT_KEY,
            {"failed": {}, "cached": {}, "evidence": {}, "generation": 0})
        failures, cached = ledger["failed"], ledger["cached"]

        # Two keys, because the two questions are different. A read is only
        # stale once something was written, so its identity carries the
        # generation. A call that keeps failing keeps failing whatever else
        # happened, so its identity does not.

        fingerprint = action_signature(name, arguments, ledger["generation"])
        failure_key = action_signature(name, arguments)

        if failures.get(failure_key, 0) >= MAX_FAILED_REPEATS:
            failures[failure_key] = failures.get(failure_key, 0) + 1

            return self._suppressed_repeat(
                context, tool_call_id, action_id, name, started,
                spec.category.value, argument_keys,
                failures[failure_key], "failed_action_repeated",
                self._REPEAT_REFUSAL.format(count=failures[failure_key]),
                failures[failure_key] > MAX_FAILED_REPEATS + 1,
            )

        # Refused only while the observation really is where the refusal says
        # it is. Once compaction or the composer has dropped it, "already
        # present in the context" is false, and telling a model that about
        # evidence it can no longer see leaves it nothing to do but find
        # another spelling of the same read -- which is the guard rail
        # defeated, not respected.

        if (cached.get(fingerprint, 0) >= MAX_CACHED_REPEATS
                and self._evidence_visible(context, ledger, fingerprint)):
            cached[fingerprint] = cached.get(fingerprint, 0) + 1

            return self._suppressed_repeat(
                context, tool_call_id, action_id, name, started,
                spec.category.value, argument_keys,
                cached[fingerprint], "cached_action_repeated",
                self._CACHED_REFUSAL,
                cached[fingerprint] > MAX_CACHED_REPEATS + 1,
            )

        safe_metadata = {
            "category": spec.category.value,
            "mutability": spec.mutability.value,
            "argument_keys": argument_keys,
            "arguments_recorded": False,
        }

        span = context.trace.start_span(
            EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_FINISHED,
            EventType.TOOL_CALL_FAILED, context.task_id,
            action_id=action_id, tool_name=name, metadata=safe_metadata,
        )

        self.hooks.before_tool(ToolHookEvent(
            "before_tool", context.task_id, action_id, name,
            spec.category.value, argument_keys,
        ))

        # Nothing below raises: a handler failure becomes a result the model can
        # read, because a raised exception here would lose the span and the hooks.

        try:
            context.cancellation.raise_if_cancelled()

            if spec.category == ToolCategory.COMMAND:
                # Commands do not run through a handler: they cross a security
                # boundary the caller owns, and its absence is a denial.

                if context.command_executor is None:
                    raw = ToolHandlerResult(
                        "ERROR: command execution boundary unavailable",
                        ToolResultStatus.DENIED,
                        error_category="security_boundary_unavailable",
                        error_summary="command execution boundary unavailable",
                    )
                else:
                    raw = context.command_executor(str(arguments["command"]))
            else:
                handler = self.registry.handler(name)

                if handler is None:
                    raise RuntimeError("registered tool has no handler")

                raw = handler(replace(context, action_id=action_id), arguments)

            context.cancellation.raise_if_cancelled()

            normalized = _normalize_handler_result(raw)
        except OperationCancelled as exc:
            normalized = ToolHandlerResult(
                f"ERROR: {exc.reason}", ToolResultStatus.CANCELLED,
                error_category="interrupted", error_summary=exc.reason,
            )
        except Exception as exc:
            normalized = ToolHandlerResult(
                f"ERROR: tool '{name}' failed: {exc}", ToolResultStatus.FAILED,
                error_category=type(exc).__name__,
                error_summary=str(exc)[:240] or type(exc).__name__,
            )

        envelope = self._envelope(
            context, spec, tool_call_id, action_id, normalized,
            time.monotonic() - started,
        )

        # Consecutive identical failures are what the breaker counts; a call
        # that worked clears its own history, because the next identical one
        # is a repetition of something that succeeded, not of something that
        # did not.

        if envelope.status == ToolResultStatus.CACHED:
            # Served from cache: the call succeeded once, and this time it
            # carried nothing new. Counted FIRST, so restoring lost evidence
            # below cannot look like progress.

            cached[fingerprint] = cached.get(fingerprint, 0) + 1

            if not self._evidence_visible(context, ledger, fingerprint):
                envelope = self._rehydrated(envelope, ledger, fingerprint)
        elif envelope.success:
            failures.pop(failure_key, None)
            seen = fingerprint in ledger["evidence"]

            ledger["evidence"][fingerprint] = {
                "tool_call_id": tool_call_id,
                "content": envelope.model_content,
                "result_reference": envelope.result_reference,
            }

            # The same observation, taken again, and nothing has been written
            # since: whether the handler served it from a cache or ran it
            # afresh is the handler's business, not a difference in what the
            # turn learned. Counting only the handler's cache hits made the
            # whole guard rail depend on a cache the caller drops for its own
            # reasons -- which is how four identical `cat main.py` calls all
            # came back `ok` and none of them counted.

            if seen:
                cached[fingerprint] = cached.get(fingerprint, 0) + 1

                context.trace.emit(
                    EventType.REPEATED_ACTION_DETECTED, context.task_id,
                    status=EventStatus.DETECTED, tool_name=name,
                    action_id=action_id,
                    metadata={"kind": "observation_repeated",
                              "repeat_count": cached[fingerprint],
                              "arguments_recorded": False},
                )
        else:
            failures[failure_key] = failures.get(failure_key, 0) + 1

        # Something may really have been written, so every earlier observation
        # may be stale and the generation moves on: every read taken before it
        # now has a different identity, and reading a file again after
        # changing it is the right thing to do. Nothing is cleared -- an
        # observation from the previous generation stays on file, in case the
        # turn comes back to it after another write.
        #
        # The decision is `invalidates_reads`, the same function the caller
        # uses for its own read cache, so the two can never disagree about
        # whether the tree has moved under them.

        if invalidates_reads(envelope.status, envelope.mutation, envelope.metadata):
            ledger["generation"] += 1

        trace_fields = {
            "tool_name": name,
            "metadata": {
                **safe_metadata,
                "output_bytes": envelope.output_bytes,
                "exit_code": envelope.exit_code,
                "truncated_for_model": envelope.truncated_for_model,
                "result_stored": envelope.result_reference is not None,
                "result_reference_present": envelope.result_reference is not None,
                "mutation_count": len(envelope.affected_paths),
            },
        }

        hook_event = ToolHookEvent(
            "after_tool" if envelope.success else "on_tool_failure",
            context.task_id, action_id, name, spec.category.value,
            argument_keys, envelope.success, envelope.status.value,
            envelope.duration_seconds,
            {"output_bytes": envelope.output_bytes,
             "stored": envelope.result_reference is not None},
        )

        if envelope.success:
            span.finish(**trace_fields)
            self.hooks.after_tool(hook_event)
        else:
            span.fail(
                envelope.error_summary or envelope.text.splitlines()[0],
                status=(EventStatus.CANCELLED
                        if envelope.status == ToolResultStatus.CANCELLED
                        else EventStatus.FAILED),
                error_category=envelope.error_category or envelope.status.value,
                **trace_fields,
            )
            self.hooks.on_tool_failure(hook_event)

        if envelope.status == ToolResultStatus.CACHED:
            context.trace.emit(
                EventType.REPEATED_ACTION_DETECTED, context.task_id,
                status=EventStatus.DETECTED, tool_name=name,
                action_id=action_id, metadata={"kind": "cached_tool_call"},
            )

        # Paths come from the handler's own report rather than from the
        # arguments, so what is traced is what the tool actually touched.

        for path in envelope.read_paths:
            context.trace.emit(
                EventType.FILE_READ, context.task_id, status=EventStatus.OK,
                tool_name=name, action_id=action_id,
                metadata={"path": path, "grounding": "structured_tool_result"},
            )

        for path in envelope.affected_paths:
            context.trace.emit(
                EventType.FILE_MODIFIED, context.task_id, status=EventStatus.OK,
                tool_name=name, action_id=action_id,
                metadata={"path": path, "grounding": "structured_tool_result"},
            )

        return envelope

    def _envelope(
        self, context: ToolExecutionContext, spec: ToolSpec,
        tool_call_id: str, action_id: str, result: ToolHandlerResult,
        duration: float,
    ) -> ToolResultEnvelope:
        # Both forms are kept: the byte length is what the size policy and the
        # trace report, while the text is what the model is shown.

        if isinstance(result.text, bytes):
            encoded = result.text
            text = result.text.decode("utf-8", errors="replace")
        else:
            text = str(result.text)
            encoded = text.encode("utf-8", errors="replace")

        policy = spec.result_policy

        # Two questions, and they were one: whether the whole result is too
        # large to keep in the window, and whether what the model is SHOWN
        # has to be cut. A handler that renders its own model-facing text
        # answers the second itself -- and a blind cut of the stored form is
        # what sent a fetched clause into the store while its metadata filled
        # the window.

        shown = text if result.model_text is None else str(result.model_text)
        oversize = len(text) > policy.model_context_chars
        truncated = len(shown) > policy.model_context_chars

        # A large result is kept whole in the store and only referenced in the
        # context, so the output stays retrievable without spending the window.

        reference = None

        if oversize and policy.store_large_results and context.result_store is not None:
            stored = context.result_store.put(
                encoded, task_id=context.task_id,
                metadata={"tool_name": spec.name, "action_id": action_id},
            )
            reference = stored.reference

        if truncated:
            preview = shown[:policy.preview_chars]
            suffix = (
                f"\n\n[tool output limited for model context: {len(encoded)} bytes; "
                + (f"full result reference {reference}]" if reference
                   else "no result store configured]")
            )
            model_content = preview + suffix
        else:
            model_content = shown

        # A cached result is a success: the call was answered, just not re-run.

        success = result.status in {ToolResultStatus.OK, ToolResultStatus.CACHED}

        return ToolResultEnvelope(
            tool_call_id, action_id, spec.name, success, result.status,
            text, model_content, spec.category.value, duration, len(encoded),
            result.exit_code, result.stdout, result.stderr,
            dict(result.metadata), truncated, result.mutation,
            tuple(result.affected_paths), tuple(result.read_paths),
            result.error_category, result.error_summary, reference,
        )

    _REPEAT_REFUSAL = (
        "ERROR: this exact call has already failed {count} times and was not run "
        "again. Repeating it will not change the result. Take a different route: "
        "use the tools that act on this workspace (bash, edit_file, write_file, "
        "search_corpus), or say what is blocking you and stop."
    )

    # An observation, not a failure, so it says what to do with the answer
    # rather than how to get a different one: the answer is already there.

    _CACHED_REFUSAL = (
        "ERROR: this exact observation is already present in the context and no "
        "state has changed. Use the existing evidence and answer the user, or "
        "choose an action that obtains genuinely new information."
    )

    @staticmethod
    def _evidence_visible(context: ToolExecutionContext, ledger: dict,
                          fingerprint: str) -> bool:
        """Whether this call's original result is still in the active context."""

        record = ledger["evidence"].get(fingerprint)
        marker = (record or {}).get("tool_call_id")

        # Nothing recorded, or a caller that cannot answer: assume it is
        # there. Assuming the opposite would re-send every cached result.

        if marker is None or context.evidence_available is None:
            return True

        try:
            return bool(context.evidence_available(marker))
        except Exception:
            return True

    @staticmethod
    def _rehydrated(envelope: ToolResultEnvelope, ledger: dict,
                    fingerprint: str) -> ToolResultEnvelope:
        """Put a lost observation back, without calling it new.

        The status stays CACHED and the count has already been taken: this
        restores evidence the model can no longer see, and restoring it is
        not progress. A result that was too large to send whole comes back as
        the same preview and the same stored reference it originally carried.
        """

        record = ledger["evidence"].get(fingerprint) or {}
        content = record.get("content")

        if not content:
            return envelope

        return replace(
            envelope, text=content, model_content=content,
            result_reference=record.get("result_reference"),
            metadata={**dict(envelope.metadata), "rehydrated_evidence": True},
        )

    def _suppressed_repeat(
        self, context: ToolExecutionContext, tool_call_id: str, action_id: str,
        name: str, started: float, tool_category: str,
        argument_keys: tuple[str, ...], count: int, kind: str, text: str,
        terminal: bool,
    ) -> ToolResultEnvelope:
        """Refuse an identical call that has already had its answer.

        Announced as a repeated action, because that is what it is, and
        answered with a result the model can act on rather than a silence it
        will fill with the same call again. ``terminal`` says the model has
        already been told once and asked anyway, which is the caller's cue to
        end the turn rather than spend the rest of the budget here.
        """

        context.trace.emit(
            EventType.REPEATED_ACTION_DETECTED, context.task_id,
            status=EventStatus.DETECTED, tool_name=name, action_id=action_id,
            metadata={"kind": kind, "repeat_count": count,
                      "terminal_repeat": terminal,
                      "argument_keys": argument_keys,
                      "arguments_recorded": False},
        )

        envelope = self._early_failure(
            context, tool_call_id, action_id, name, started,
            ToolResultStatus.DENIED, kind, text, tool_category, argument_keys,
        )

        return replace(envelope, metadata={
            "suppressed_repeat": True, "repeat_count": count,
            "repeat_kind": kind, "terminal_repeat": terminal,
            "category": tool_category,
        })

    def _early_failure(
        self, context: ToolExecutionContext, tool_call_id: str,
        action_id: str, name: str, started: float,
        status: ToolResultStatus, category: str, text: str,
        tool_category: str, argument_keys: tuple[str, ...],
    ) -> ToolResultEnvelope:
        """Refuse a call before it runs, still emitting a complete started/failed pair."""

        # The pair is emitted by hand rather than through a span: the call is
        # rejected, and a trace reader must still see a start for every failure.

        context.trace.emit(
            EventType.TOOL_CALL_STARTED, context.task_id,
            status=EventStatus.STARTED, tool_name=name, action_id=action_id,
            metadata={"category": tool_category,
                      "argument_keys": argument_keys,
                      "arguments_recorded": False},
        )
        context.trace.emit(
            EventType.TOOL_CALL_FAILED, context.task_id,
            status=EventStatus.FAILED, tool_name=name, action_id=action_id,
            error_category=category, error_summary=text[:240],
            metadata={"category": tool_category,
                      "argument_keys": argument_keys,
                      "arguments_recorded": False},
        )

        event = ToolHookEvent(
            "on_tool_failure", context.task_id, action_id, name, tool_category,
            argument_keys, False, status.value, time.monotonic() - started,
        )

        self.hooks.on_tool_failure(event)

        return ToolResultEnvelope(
            tool_call_id, action_id, name, False, status, text, text,
            tool_category, event.duration_seconds or 0.0,
            len(text.encode("utf-8")), error_category=category,
            error_summary=text,
        )


def _normalize_handler_result(value: object) -> ToolHandlerResult:
    """Accept the legacy string protocol as well as a structured result."""

    if isinstance(value, ToolHandlerResult):
        return value

    if isinstance(value, (str, bytes)):
        # A legacy handler encodes its outcome in the text, so the status has
        # to be read back out of the prefix and the trailing exit code.

        text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        exit_match = re.search(r"\(exit\s+(-?\d+)\)", text)
        exit_code = int(exit_match.group(1)) if exit_match else None

        if text.startswith("CANCELLED"):
            status = ToolResultStatus.CANCELLED
        elif text.startswith("ERROR"):
            status = (ToolResultStatus.TIMEOUT if "timed out" in text.lower()
                      else ToolResultStatus.FAILED)
        elif exit_code not in (None, 0):
            status = ToolResultStatus.FAILED
        elif text.startswith("(ALREADY EXECUTED"):
            status = ToolResultStatus.CACHED
        else:
            status = ToolResultStatus.OK

        return ToolHandlerResult(
            value, status, exit_code=exit_code,
            error_category=(status.value if status not in {
                ToolResultStatus.OK, ToolResultStatus.CACHED} else None),
            error_summary=(text.splitlines()[0][:240] if status not in {
                ToolResultStatus.OK, ToolResultStatus.CACHED} else None),
        )

    raise TypeError("tool handler must return ToolHandlerResult, str, or bytes")


def _validate_arguments(spec: ToolSpec, arguments: Mapping[str, object]) -> str | None:
    """The schema check the router runs before a handler ever sees the call."""

    if not isinstance(arguments, Mapping):
        return "arguments must be an object"

    schema = spec.input_schema
    properties = schema.get("properties", {})

    for name in schema.get("required", ()):
        if name not in arguments:
            return f"missing required argument '{name}'"

    expected_types = {
        "string": str, "integer": int, "number": (int, float),
        "boolean": bool, "array": (list, tuple), "object": Mapping,
    }

    for name, value in arguments.items():
        property_schema = properties.get(name)

        # An undeclared argument is left alone: the schema states what is
        # required and typed, not that nothing else may be passed.

        if property_schema is None:
            continue

        # bool is an int subclass, so a `true` passed where a number is
        # expected would otherwise satisfy the isinstance check.

        expected = expected_types.get(property_schema.get("type"))

        if expected is not None and (not isinstance(value, expected)
                                     or isinstance(value, bool)
                                     and property_schema.get("type") in {"integer", "number"}):
            return f"argument '{name}' must be {property_schema.get('type')}"

    return None
