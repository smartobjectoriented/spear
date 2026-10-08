"""What the legacy loop tells a turn, and what it lets the turn do.

The write gate and the tools a writing round may see, the repeated-read
guard, the reading ceiling, the claims an answer makes about changes, the
reading of a request as a request to write, and the demands that send a
turn back to its work.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import replace
from typing import Sequence

from models.model_backend import ModelTurn, StopReason
from harness.tool_router import ToolResultStatus
from harness.tool_registry import COMMAND_TOOLS
from context import request_intent
from runtime.budgets import BudgetKind
from runtime import work_phase
from runtime.agent_context import AgentContext


_WRITE_TOOLS = ("edit_file", "patch", "write_file", "append_file")

#: The tool that opens the write gate. It belongs with the writing tools
#: wherever the harness narrows a round to them: a round offered nothing but
#: `edit_file` while the gate is still shut is a round that can only be
#: refused, and a demand the harness has made impossible reads as the model
#: refusing to act.
_PLAN_TOOL = work_phase.PLAN_TOOL


def _write_round_tools(context) -> tuple[str, ...]:
    """The tools a round narrowed to writing may see."""
    phase = getattr(context, "work_phase", None)

    if phase is None or not getattr(phase, "engaged", False):
        return _WRITE_TOOLS

    return (_WRITE_TOOLS if phase.may_write().allowed
            else _WRITE_TOOLS + (_PLAN_TOOL,))


# A read that answers the same thing again has told the turn nothing. Said
# at the second, refused at the fourth: the first repeat can be a coincidence
# of two honest questions, the fourth is a loop.
_REPEAT_NOTICE = 2

# Refused at the third, not the fourth. Watched live: told twice that a
# result was one it already had, a turn asked a third time and kept going to
# a hundred and twenty-seven calls. Two warnings are enough warning.
_REPEAT_REFUSE = 3

# Below this a result is too small to identify: "0", "" and a one-line count
# repeat for perfectly good reasons.
_REPEAT_MIN_CHARS = 200


def _repeated_result(seen, envelope, visible=lambda _call_id: True) -> int:
    """How many times this exact result has come back this turn.

    Mutations are exempt: writing the same content twice is idempotent and
    fine. Failures are exempt too -- the same error twice is the tree
    telling the truth twice, and the repair loop depends on hearing it.

    Only deliveries the model can still see are counted. A turn compacted,
    re-read the recipe the summary had folded away, and was refused it as
    "already given 2 times": given, and then taken away. A refused call is
    not recorded, since it delivered nothing; while every delivery stays
    visible the count only grows, so the refusal stands as before.
    """
    text = envelope.text or ""

    if envelope.mutation or not envelope.success or len(text) < _REPEAT_MIN_CHARS:
        return 0

    key = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
    delivered = [call_id for call_id in seen.get(key, ()) if visible(call_id)]
    count = len(delivered) + 1

    if count < _REPEAT_REFUSE:
        delivered.append(envelope.tool_call_id)

    seen[key] = delivered

    return count


def _with_repeat_note(envelope, count):
    """The same result, carrying the fact that it is the same.

    Refused outright past the limit: a turn that has been told three times
    and asked a fourth is not reading the answer, and handing it the wall of
    text again is how the loop is fed.
    """
    if count >= _REPEAT_REFUSE:
        return replace(
            envelope,
            model_content=(
                f"REFUSED: this call returns a result you have already been "
                f"given {count - 1} times. Nothing about it has changed. Use "
                f"what you have, or do something different -- and if the task "
                f"asks you to change the code, change it."),
        )

    return replace(
        envelope,
        model_content=(
            f"{envelope.model_content}\n\n[the same result as "
            f"{count - 1} earlier call(s) this turn: reading it again will "
            f"not add anything]"),
    )


_INTENT_RE = re.compile(
    r"\b(let me|let's|let us|i'?ll|i will|i'?m going to|going to|"
    r"first,? (?:let|i|read|check)|now (?:let|i)|here'?s what)\b", re.I,
)
_FABRICATION_RE = re.compile(r"^\s*\[tool\].*$", re.M)


def strip_fabrications(text: str) -> str:
    return _FABRICATION_RE.sub("", text)


_REFUSED = frozenset({ToolResultStatus.DENIED, ToolResultStatus.UNKNOWN_TOOL,
                      ToolResultStatus.INVALID_ARGUMENTS})


# Reading rounds a turn gets before it is asked to conclude. Scaled to the
# budget, but CAPPED -- and the cap is the point. It used to be a plain
# fraction, which was fine at a 60-round ceiling and became absurd when the
# ceiling rose to 500: the nudge moved to a hundred rounds, and a master run
# read for thirty minutes with nothing to interrupt it and wrote nothing. A
# bigger budget should buy a turn more room to WORK, not more room to read
# before it starts.
INVESTIGATION_CEILING = int(os.environ.get("SPEAR_MAX_READING_ROUNDS", "40"))


def _tool_activity(name: str) -> str:
    """The status label for a running tool: what it is, in the same words.

    Same indicator, same colours, same clock as a model call -- only the
    label changes. Nothing here reformats the line: a tool that takes four
    seconds should look like work in progress, not like a different program.
    """

    return f"{name}…"


def _investigation_ceiling(round_limit) -> int:
    return min(INVESTIGATION_CEILING, max(12, round_limit // 5))


def _past_wall_fraction(context, fraction) -> bool:
    """Has the turn spent this much of its wall-clock budget?

    Read rather than counted: the manager updates the elapsed figure on
    every charge, so this needs no clock of its own and cannot drift from
    the limit that will actually end the turn.
    """
    manager = getattr(context, "budget_manager", None)

    if manager is None:
        return False

    limit = manager.limits.get(BudgetKind.WALL_TIME_MS)

    if limit is None or limit.amount <= 0:
        return False

    return manager.consumed.get(BudgetKind.WALL_TIME_MS, 0) >= limit.amount * fraction


def _round_was_all_refused(envelopes) -> bool:
    """Did the sandbox refuse every tool call of that round?

    A refusal is the harness's answer, not the turn's work, and a round made
    only of refusals gave the model nothing to go on.
    """
    if not envelopes:
        return False

    return all(envelope.status in _REFUSED for envelope in envelopes)


def _asked(conversation) -> str:
    """The operator's own words for this turn, which the policy routes on.

    Harness-authored messages are skipped. They carry the user role because
    the API has no other one, and reading them back as the request made the
    harness answer itself: a nudge that offers "make ONE small edit_file
    change" matched the write-request detector, so a turn that had only been
    asked a question was told it had changed nothing and demanded an edit.
    The model, correctly, refused to invent an old_text for a file it had
    never opened -- and the session's first prompt ended in an argument
    instead of an answer.
    """
    for message in reversed(list(conversation or ())):
        if getattr(message, "role", None) != "user":
            continue

        if getattr(message, "authored_by", "operator") != "operator":
            continue

        parts = [getattr(block, "text", "") for block in message.content
                 if getattr(block, "text", "")]

        if parts:
            return "\n".join(parts)

    return ""


def standard_policy_for(context: AgentContext, *, repair_ask=None):
    """The deterministic standard-bound policy for this turn, if it is one.

    Activation is the session's binding, never the user's phrasing, and never
    anything the model says. A turn with no bound standard gets no policy and
    behaves exactly as it did before -- the runtime stays generic and the
    normative reasoning stays in one place.

    `repair_ask` is the one tool-less question the policy may put back to the
    model when the guards reject an answer the evidence could still support.
    It travels from here because this is where the runtime's model access
    lives; without it the repair silently never ran in an interactive
    session, only in the evaluation harness.
    """
    if not context.standard_binding:
        return None

    from standard import standard_answer_policy

    return standard_answer_policy.policy_for(context.standard_binding,
                                             _asked(context.conversation),
                                             repair_ask=repair_ask)


def looks_like_preamble(text: str) -> bool:
    value = (text or "").strip()
    return 0 < len(value) <= 400 and bool(_INTENT_RE.search(value))


# "I have created /path/x.rst" -- a claim of a completed write, in the perfect
# or the passive. Intent ("I'll create") is deliberately excluded: saying what
# you are about to do is not a false report.

# The verbs a finished write is reported with. "implemented", "completed",
# "done" and "applied" were missing, and they are the ones a TASK is reported
# with rather than a file: on a fresh checkout, having read the task file and
# grepped for the enum it asks for -- exit 1, "not yet defined" -- the model
# answered "The task in doc/ack-task.md has already been implemented, adding
# the CmdAckKind enum and updating all relevant files". Nothing was written.
_DONE_VERB = (r"(?:created|written|added|updated|modified|fixed|implemented"
              r"|completed|applied|done|made)")

_CHANGE_CLAIM_RE = re.compile(
    r"\b(?:i (?:have |'ve )?" + _DONE_VERB
    # "has ALREADY been implemented": an adverb between the auxiliary and the
    # participle is the ordinary way to say it, and it defeated `has been`.
    + r"|(?:has|have|had|was|were|is|are)\s+(?:\w+\s+){0,2}been\s+" + _DONE_VERB
    # "was successfully applied", with no "been" at all. Past tense only:
    # "is implemented in command_wire.c" describes code that exists, which is
    # not a report of work done this turn.
    + r"|(?:was|were)\s+(?:\w+\s+){0,2}" + _DONE_VERB
    + r")\b",
    re.I)
_FILE_TOKEN_RE = re.compile(r"[\w./-]+\.[A-Za-z]\w{0,4}\b")

# The mirror image of _CHANGE_CLAIM_RE: not "I updated the file" but "I'll
# update the file". One reports a write that did not happen; the other
# promises one and stops.
_ANNOUNCED_CHANGE_RE = re.compile(
    r"\b(?:i'?(?:ll|m going to)|i will|let me|now i'?ll|next,? i'?ll|"
    r"i can|i'?m about to|going to)\s+"
    r"(?:\w+\s+){0,4}?"
    r"(?:implement|make|apply|add|create|write|update|modify|change|fix|"
    r"edit|introduce|refactor|rename|remove|delete)\b", re.I)


def announced_but_unmade_change(text: str, tool_log: Sequence[str],
                                did_modify: bool) -> bool:
    """Did the answer promise an edit the turn then never made?

    A bound turn read twelve windows of C, worked out the change, and ended
    on "I'll implement the change by adding the CmdAckKind enum and modifying
    the encode function". Nothing was written. The task file had asked for an
    implementation, so a plan is not a smaller answer -- it is no answer, and
    the next turn starts from nothing and pays for the reading again.

    The harness already refuses to hand over a claim of a write that did not
    happen (unsupported_change_claim). This is the same evidence read the
    other way: the write was announced, the tool log holds no write, so the
    turn is not finished. It costs one more round, which is what the nudge to
    conclude costs in the opposite direction.
    """
    if did_modify or not text:
        return False

    if any(entry.split(" ", 1)[0] in _WRITE_TOOLS for entry in tool_log):
        return False

    return bool(_ANNOUNCED_CHANGE_RE.search(text)
                and _FILE_TOKEN_RE.search(text))


# A turn whose request is a write, in the user's own words. Not a guess about
# intent: these are imperatives naming an edit.
_WRITE_REQUEST_RE = re.compile(
    # "do the necessary changes" is a writing request in anyone's reading, and
    # the determiner group knew "the" but not "the necessary": one adjective
    # between the two was enough for the whole deterministic floor to miss a
    # turn that asked for the code to be changed, leaving the decision to the
    # model classifier alone. Two adjectives, because "all the required minor
    # fixes" is the same sentence.
    r"\b(?:do|make|apply|carry\s+out|perform)\s+(?:the\s+|these\s+|those\s+|"
    r"all\s+(?:the\s+)?)?(?:\w+\s+){0,2}?(?:modifications?|changes?|edits?|"
    r"task|work|fix(?:es)?)"
    # "Make timer_init() set up the counter", "make the encoder emit one
    # packet per request": a causative imperative whose object is something
    # in the tree -- a function, a file, or the code by an ordinary name --
    # asks for that thing to behave differently. "make sure" and "make a
    # list" name nothing in the tree.
    r"|(?:^|[.!?;:]\s+)(?:please\s+)?make\s+(?:the\s+|this\s+|that\s+|our\s+)?"
    r"(?:[A-Za-z_]\w*\(\)|[\w/-]{2,}\.(?:c|h|cc|cpp|hpp|py|rs|go|js|ts|java|sh)\b|"
    r"(?:implementation|code|function|method|class|module|script|program|build|"
    r"tests?|driver|firmware|encoder|decoder|parser|header)\b)"
    r"|\b(?:implement|write|edit|patch|refactor|rename|add|create|update|fix|"
    r"modify|remove|delete|adapt|adjust|amend|revise|rework|correct|move|"
    r"relocate)\b"
    r"|\b(?:fais|faire|applique|implémente|implementer|implémenter|corrige|"
    r"modifie|modifier|adapte|adapter|ajuste|ajuster|ajoute|écris|ecris)\b",
    re.I)


def wants_write(context, question) -> bool:
    """The turn's answer to "was I asked to change the code", decided once.

    Cached on the context because every gate below asks it and the answer
    cannot change inside a turn -- and because the model-side half costs a
    call, which is worth paying once and not once per round.

    An explicit prohibition is checked FIRST, before the cache and before any
    reading of the words: it is the strongest statement of intent there is,
    and the pattern below cannot see past a single token. Asked "Answer which
    file defines add without editing files", `\badd\b` matched the FUNCTION'S
    NAME, the turn was classified as a writing request, and every gate that
    asks this then spent rounds demanding an edit the user had forbidden.
    """
    if getattr(context, "read_only", False):
        # Written through, so the cache and the decision cannot disagree
        # later in the turn.

        context._write_request = False

        return False

    cached = getattr(context, "_write_request", None)

    if cached is not None:
        return cached

    # A question about a change is settled here, before the model is asked
    # and whatever it would answer. "Could we have X in images/ instead?"
    # was read WRITE by the model, and the turn was then told it was not
    # finished until a file changed.

    if request_intent.advisory(question, _WRITE_REQUEST_RE):
        context._write_request = False

        return False

    # Accepting a proposal ("yes, do it") is a write, when there is a
    # proposal: the previous answer. The question that produced it was not.

    pattern = is_write_request(question) or request_intent.mutation_intent(
        question, _WRITE_REQUEST_RE,
        answered=request_intent.answered(getattr(context, "conversation", ())),
    ) == "write"
    model = None

    if not pattern and getattr(context, "judge_intent", False):
        # Only asked when the pattern is silent: the floor is already
        # decided, and a call that cannot change the answer is a call not
        # worth making.
        model = request_intent.judge(
            getattr(context, "backend", None), question,
            budget_manager=getattr(context, "budget_manager", None),
            budget_kind=BudgetKind.AUXILIARY_MODEL_CALLS)

        if model is not None:
            context.observer.notice(
                "intent_judged", {"write": model, "asked": question[:80]})

    answer = request_intent.resolve(pattern, model)
    context._write_request = answer

    return answer


def is_write_request(question: str) -> bool:
    """Did the user ask for the tree to change, in so many words?

    Read sentence by sentence: a verb inside a question about a change
    ("could we add a flag?") asks for nothing, and an acceptance of the
    previous proposal ("yes, do it") asks for the change it accepted.
    """
    intent = request_intent.mutation_intent(question or "", _WRITE_REQUEST_RE)

    if intent != "unknown":
        return intent == "write"

    return bool(_WRITE_REQUEST_RE.search(question or ""))


def may_demand_write(context) -> bool:
    """Whether this turn is allowed to be sent back to the code at all.

    One question, asked by every gate that would inject a write demand,
    reopen the write window, or narrow the tools to the writing ones. A turn
    the user told not to change anything is finished by an answer, and a
    harness that keeps asking for an edit is asking it to disobey.
    """

    return not getattr(context, "read_only", False)


def carried_obligations(context, question) -> tuple:
    """The clauses an earlier turn established that THIS turn owes work on.

    Clauses carried across turns are EVIDENCE, not a checklist. The carry
    exists for one shape -- "how should X work?", then "change the code" --
    where the reasoning was done in the first prompt and the second would
    otherwise start over on the wrong sections. It says what a change has to
    satisfy; it does not say that a change was asked for.

    Nothing enforced that. The completion gate read `prior_clauses` alone,
    so a session that had established ten clauses turned the NEXT question
    -- "is this explicitly required by the standard, and which clauses say
    so?" -- into an implementation audit: the answer was complete and
    correct, and the harness then demanded file-and-line for eight clauses
    the user had not asked about, which the turn duly supplied by editing
    the customer's tree, building it and running its tests.

    So the obligation is decided by THIS turn's request, by the same test
    that decides whether the clauses are put in front of the model at all.
    A gate may only ask for what the turn was actually given to do: if the
    request is a question, the carried clauses are context for answering it
    and the turn is finished when it is answered.

    The execution mode is deliberately absent. `--auto` says which actions
    may proceed without confirmation; it has never had a say in which
    actions are in scope, and a gate that widened the task because
    confirmation was cheap would be reading it as permission to do more.
    """
    prior = tuple(getattr(context, "prior_clauses", ()) or ())

    if not prior:
        return ()

    # A turn told not to change anything cannot be sent back to the code,
    # whatever its words otherwise match.

    if not may_demand_write(context):
        return ()

    return prior if is_write_request(question) else ()


def conclude_demand(question: str, is_write: bool | None = None) -> str:
    """The nudge that ends a turn which has only been reading.

    It used to offer two branches -- answer the question, or make the edit --
    and let the model pick. Asked "please do the modifications", a turn that
    had read eight files and knew the enum was missing took the first branch
    and re-emitted the previous turn's prose, word for word. Nothing was
    written, and the answer closed by saying the task "has already been
    implemented".

    A request to change the tree does not have a question branch. When the
    user's own words are an imperative to edit, the only way out of the nudge
    is the edit.

    ``is_write`` is the turn's already-resolved answer -- prohibition,
    pattern, and the model judgement if one was made. Re-reading the words
    here made this the last place that could still call "Answer which file
    defines add" an instruction to edit, after every other gate had agreed it
    was not. The pattern stays as the fallback for callers with no turn to
    ask (the tests, and nothing else).
    """
    if is_write_request(question) if is_write is None else is_write:
        return (
            "You have investigated enough. The user asked you to change the "
            "code, so this turn is not finished until a file changes. Make "
            "ONE small edit_file change now: the single most important edit, "
            "with a SHORT old_text (the few exact lines to replace, copied "
            "verbatim from what you read) and the new_text. Call NO other "
            "tool first: no shell command and no retrieval of any kind. "
            "Told only to stop reading files, a turn spent its last seven "
            "rounds on retrieval instead and wrote nothing. Do NOT rewrite "
            "whole files, do NOT summarise what you read, and do NOT report "
            "the task as already done -- nothing has been written yet. You "
            "can make more small edits after this one."
        )

    if request_intent.advisory(question or "", _WRITE_REQUEST_RE):
        return (
            "You have investigated enough. Conclude now, with no further "
            "bash/grep/find/cat/read commands. The user asked whether or how "
            "something could be changed; they did not ask for the change. "
            "Answer that: say whether it is feasible, what would have to "
            "change and in which files (cite the exact paths), and the "
            "trade-offs. Do NOT edit anything -- the change is theirs to ask "
            "for next.")

    return (
        "You have investigated enough. Conclude now, with no further "
        "bash/grep/find/cat/read commands. If the user asked a question, "
        "answer it from what you found, citing the exact paths. If a change "
        "was requested, make ONE small edit_file change: the single most "
        "important fix, with a SHORT old_text (the few exact lines to "
        "replace) and the new_text — do NOT rewrite the whole file, keep it "
        "under ~15 lines so it completes. You can make more small edits after."
    )


def make_it_demand(text: str) -> str:
    """The one message that turns an announcement into the edit itself."""
    return (
        "You said you would make this change and then stopped, so the turn "
        "wrote nothing. Make it now: call edit_file with a SHORT old_text "
        "(the few exact lines to replace, copied verbatim from the file you "
        "read) and the new_text. Do not re-read, do not restate the plan, do "
        "not rewrite whole files. If the change spans several files, make the "
        "single most important edit now; you can make the others after."
    )


def unsupported_change_claim(text: str, tool_log: Sequence[str],
                             did_modify: bool) -> str:
    """A note to append when the answer claims a write the turn never made.

    A cut turn asked to conclude produced, verbatim, the previous turn's
    summary from history: "I have created the documentation chapter at
    .../user_space_ls.rst", plus an account of a sed refusal that happened in
    another turn. Nothing had been written -- the tool log held four reads.

    The harness cannot stop a model from claiming that. It can refuse to hand
    the claim over as the turn's result while holding the evidence that it is
    false, which is the same rule it applies to sandboxes and budgets: what is
    reported is what was observed.
    """

    if did_modify or not text:
        return ""

    if any(entry.split(" ", 1)[0] in _WRITE_TOOLS for entry in tool_log):
        return ""

    if not _CHANGE_CLAIM_RE.search(text) or not _FILE_TOKEN_RE.search(text):
        return ""

    return ("\n\n⚠ Nothing was written this turn: no file was created or "
            "modified, and the tool log holds only reads. Any claim above that "
            "a file was created or updated describes work that did not happen "
            "here — treat it as a plan, not a result.")


def turn_evidence(tool_log: Sequence[str]) -> str:
    if not tool_log:
        return ""

    commands = failures = 0
    modified = []

    for entry in tool_log:
        head, _, output = entry.partition("\n")
        name = head.split(" ", 1)[0]

        if name in COMMAND_TOOLS:
            commands += 1

            if output.startswith("ERROR:") or re.search(r"\(exit [1-9]", output):
                failures += 1
        elif name in _WRITE_TOOLS:
            match = re.search(r'"path"\s*:\s*"([^"]+)"', head)

            if match and output.startswith("OK"):
                modified.append(match.group(1))

    files = ", ".join(dict.fromkeys(modified)) if modified else "none"

    return (f"\n\nFacts from this turn's tool log — {commands} command(s) run, "
            f"{failures} of them exited non-zero; files changed: {files}. "
            f"Base your summary on tool results, not on what you meant to do. "
            f"If you report behaviour as working, name the cases whose output "
            f"you actually saw, and say which cases you did not test. A command "
            f"that printed nothing did not confirm anything.")


def validate_agent_turn(turn: ModelTurn) -> str:
    if turn.stop_reason == StopReason.TOOL_USE:
        if not turn.tool_calls:
            raise RuntimeError("provider returned TOOL_USE without tool_calls")

        return "tool_use"

    if turn.tool_calls:
        raise RuntimeError(
            f"provider returned tool_calls with terminal stop_reason {turn.stop_reason}"
        )

    if turn.stop_reason == StopReason.END_TURN:
        return "end_turn"

    if turn.stop_reason == StopReason.REFUSAL:
        return "refusal"

    raise RuntimeError(turn.error or f"model backend stopped with {turn.stop_reason}")


def write_demand(question: str) -> str:
    """The one message that sends an empty writing turn back to the code."""
    return (
        "You were asked to change the code and this turn has not changed "
        "anything: the tool log holds reads only. Make the edit now, with "
        "edit_file and a SHORT old_text copied verbatim from a file you have "
        "read. Do not read anything else first -- no shell command and no "
        "retrieval. If after looking you believe no change is needed, say "
        "exactly which file and which lines already do what was asked, and "
        "why; an answer that describes the change instead of making it is "
        "not an answer to this request."
    )
