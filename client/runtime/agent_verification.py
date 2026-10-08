"""What a turn changed and what shows that the change works.

Read from the tool log and from the project's own build and test commands:
the verification runs, the requirement matrix, the notes an answer carries
when its change is unverified, and the work-order sections still open.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Sequence

from harness.tool_registry import COMMAND_TOOLS
from evidence import project_build
from runtime import work_order
from evidence.verification import (
    CompletionVerificationStatus, VerificationCategory, VerificationCoverage,
    VerificationEvidence, VerificationPolicy,
)
from normative.requirement_set import Disposition
from runtime.tracing import new_action_id
from runtime.working_state import TerminalStatus, VerificationOutcome
from runtime.agent_notes import _WRITE_TOOLS, is_write_request


def changed_files(tool_log: Sequence[str]) -> list[str]:
    result = []

    for entry in tool_log:
        head, _, output = entry.partition("\n")

        if head.split(" ", 1)[0] in _WRITE_TOOLS:
            match = re.search(r'"path"\s*:\s*"([^"]+)"', head)

            if match and output.startswith("OK"):
                result.append(match.group(1))

    return list(dict.fromkeys(result))


# What actually establishes that a change works. `ls`, `cat` and `git status`
# are not on this list: a turn that edited four files, ran `ls`, `cat
# CMakeLists.txt` and `cmake -S . -B build`, and stopped, had "run something
# after the change" and had verified nothing. It left the tree not compiling.
_VERIFYING = frozenset({
    VerificationCategory.BUILD,
    VerificationCategory.UNIT_TEST,
    VerificationCategory.INTEGRATION_TEST,
    VerificationCategory.LINT,
    VerificationCategory.STATIC_ANALYSIS,
    VerificationCategory.EXECUTION,
})

# rag_chat appends this to a command result whose status was not zero.
_EXIT_CODE_RE = re.compile(r"\(exit (\d+)\)")


def _unescaped(command: str) -> str:
    """The command as it was run, not as the log's JSON spelled it."""
    try:
        return json.loads(f'"{command}"')
    except ValueError:
        return command


def _refused(output: str) -> bool:
    """A command the boundary refused did not run, so it proved nothing."""
    return output.lstrip().startswith(("ERROR", "REFUSED"))


def _verification_runs(tool_log: Sequence[str], policy=None) -> list[tuple[str, bool]]:
    """Every command after the last write that verifies anything, and whether
    it succeeded. Configuring a build is not building it: `cmake -S . -B
    build` classifies as BUILD and would pass for verification on its name
    alone, so a run counts only when its own output carries no failing exit
    code.

    What changed is passed to the classifier, because a command that
    exercises none of it verifies none of it: a turn that edited a Python
    file and compiled a standalone C program was, without this, a turn with a
    passing build behind it.
    """
    policy = policy or VerificationPolicy()
    last_change, runs = -1, []
    changed = changed_files(tool_log)

    for index, entry in enumerate(tool_log):
        head, _, output = entry.partition("\n")
        name = head.split(" ", 1)[0]

        if name in _WRITE_TOOLS and output.startswith("OK"):
            last_change, runs = index, []
        elif name in COMMAND_TOOLS and last_change >= 0:
            # The command itself, not the JSON around it: classify_command
            # anchors on the start of the string, and `./test_command` inside
            # {"command": "..."} matches nothing.
            found = re.search(r'"command"\s*:\s*"(.*)"\s*\}?\s*$', head)
            command = _unescaped(found.group(1)) if found else ""
            category, _ = policy.classify_command(command, changed_paths=changed)

            if category in _VERIFYING:
                found = _EXIT_CODE_RE.search(output)
                runs.append((command, not _refused(output)
                             and (found is None or found.group(1) == "0")))

    return runs


def unverified_change(tool_log: Sequence[str]) -> bool:
    """A change with nothing at all run after it.

    Deliberately looser than _verification_runs: this drives the mid-turn
    nudge, which asks the model to go and verify, and a turn that ran
    something has at least been pointed at the question. What that something
    established is settled at the end, by unverified_write_note, where the
    answer is already written and the judgement costs no round.
    """
    last_change, ran_after = -1, False

    for index, entry in enumerate(tool_log):
        head, _, output = entry.partition("\n")
        name = head.split(" ", 1)[0]

        if name in _WRITE_TOOLS and output.startswith("OK"):
            last_change, ran_after = index, False
        elif name in COMMAND_TOOLS and last_change >= 0:
            ran_after = True

    return last_change >= 0 and not ran_after


#: Compiling the files that were touched, and nothing else. A legitimate step
#: on the way -- it finds a typo in one round instead of three -- and not a
#: result: it links nothing, runs nothing, and says nothing about the program
#: the change was made to. Recognised by the shape every such command has,
#: which is a compiler asked NOT to produce a program.
_SYNTAX_ONLY = re.compile(
    r"^\s*(?:\S*/)?(?:g?cc|clang|c\+\+|g\+\+|clang\+\+)\b[^|;&]*"
    r"(?:\s-fsyntax-only\b|\s-c\b)", re.I)


def syntax_only_verification(runs) -> bool:
    """Every verification this turn ran was a compile of loose files."""
    return bool(runs) and all(_SYNTAX_ONLY.search(command)
                              for command, _ in runs)


#: How a disposition reads in the closing table.
_DISPOSITION_LABEL = {
    "satisfied_already": "already satisfied",
    "change_planned": "CHANGE PLANNED, NOT MADE",
    "change_implemented": "changed",
    "explicitly_out_of_scope": "out of scope",
    "undetermined": "UNDETERMINED",
}


#: What a turn says when it believes it has finished. Matched only to compare
#: it against the ledger -- the prose is never the record, and a run that
#: ended "the implementation is complete and satisfies all the requirements"
#: over a ledger holding one out-of-scope rule and six unvalidated items is
#: why this is checked at all.
_CLAIMS_DONE = re.compile(
    r"\b(?:implementation|it|this|everything|all\s+(?:of\s+)?(?:the\s+)?"
    r"(?:requirements?|clauses?|rules?))\s+(?:is|are|now)\s+"
    r"(?:complete|compliant|satisfied|done)\b"
    r"|\bfully\s+(?:compliant|implemented|satisfied)\b"
    r"|\ball\s+(?:the\s+)?requirements?\s+(?:are\s+)?(?:met|satisfied)\b"
    r"|\bsatisfies\s+all\b|\bimplementation\s+is\s+complete\b",
    re.I)


def requirement_status(phase):
    """The turn's status, in the ledger's words and nobody else's.

    Three readings and no fourth, because there are only three things the
    ledger can be saying: everything reviewed is closed and validated, it is
    closed with stated exclusions, or there is work left. "Everything is
    compliant" is not among them and cannot be reached from here.
    """
    requirements = getattr(phase, "requirements", None)

    if not requirements or not len(requirements):
        return ""

    open_items = requirements.open_items()
    excluded = [item for item in requirements
                if item.disposition == str(Disposition.EXPLICITLY_OUT_OF_SCOPE)]
    unvalidated = phase.unvalidated_requirements()

    if open_items or unvalidated:
        return "implementation work incomplete"

    if excluded:
        return "supported profile validated, with exclusions"

    return "reviewed requirements satisfied"


def requirement_matrix_note(phase, answer="") -> str:
    """The canonical record: one line per requirement, and one status.

    Deterministic, built from the ledger rather than from the answer, and
    printed whether the news is good or bad. It is also the ONLY matrix: a
    real run wrote its own three-row summary table declaring every clause
    satisfied, directly above a harness matrix that called one of them out of
    scope and listed six items as unvalidated. Two tables that disagree are
    worse than one that says something unwelcome, so when the prose claims
    completion the ledger contradicts it here, by name.
    """
    requirements = getattr(phase, "requirements", None)

    if not requirements or not len(requirements):
        return ""

    lines = ["", "", f"REQUIREMENT MATRIX — {len(requirements)} requirement(s) "
             f"carried from the previous turn. This is the record; any summary "
             f"above that disagrees with it is wrong."]

    unvalidated = {item.key for item in phase.unvalidated_requirements()}

    for item in requirements:
        label = _DISPOSITION_LABEL.get(item.disposition, item.disposition)

        if item.key in unvalidated:
            label += ", NOT VALIDATED"

        where = item.source_id or item.section or "—"
        said = requirements.stated_revisions(item)
        revised = f" (revised {said}x)" if said > 1 else ""
        lines.append(f"- {item.key} [{where}] — {label}{revised}"
                     + (f" | condition: {item.trigger[:90]}"
                        if item.trigger else "")
                     + (f": {item.note[:120]}" if item.note else ""))

    unfinished = requirements.open_items()
    status = requirement_status(phase)

    if unfinished:
        lines.append("")
        lines.append(f"{len(unfinished)} of them are not closed: "
                     + ", ".join(found.key for found in unfinished)
                     + ". This turn is not finished.")

    if unvalidated:
        lines.append("")
        lines.append(f"{len(unvalidated)} changed requirement(s) have no test "
                     f"that reaches the changed path: "
                     + ", ".join(sorted(unvalidated))
                     + ". A suite that passed without exercising them proves "
                       "nothing about them.")

    carried_only = requirements.excluded()

    if carried_only:
        lines.append("")
        lines.append(f"Also retrieved and NOT in scope for this turn "
                     f"({len(carried_only)}), recorded so the choice is "
                     f"visible: "
                     + ", ".join(f"{found.key} ({found.origin})"
                                 for found in carried_only[:8]))

    gaps = phase.branch_gaps()

    if gaps:
        lines.append("")
        lines.append("Tested on one side of the condition only: "
                     + "; ".join(f"{key} — {why}"
                                 for key, why in sorted(gaps.items())[:6])
                     + ". An implementation can pass these and still ignore "
                       "the condition.")

    excluded = [item for item in requirements
                if item.disposition == str(Disposition.EXPLICITLY_OUT_OF_SCOPE)]

    if excluded:
        lines.append("")
        lines.append("Explicitly excluded from the supported profile: "
                     + ", ".join(found.key for found in excluded)
                     + ". The result is not compliance with the whole "
                       "document and must not be reported as such.")

    lines.append("")
    lines.append(f"STATUS (from the ledger): {status}.")

    if (unfinished or unvalidated) and _CLAIMS_DONE.search(answer or ""):
        lines.append("")
        lines.append("The answer above claims this work is complete. The "
                     "ledger says it is not, and the ledger is what was "
                     "measured: take the matrix, not the claim.")

    return "\n".join(lines)


def turn_changed_files(context, tool_log) -> tuple[str, ...]:
    """The files the turn changed: from the agent core's structured evidence
    when the turn ran there, from the tool log otherwise."""
    log = getattr(context, "core_evidence", None)

    if log is not None:
        from evidence import completion
        return completion.changed_files(log)

    return tuple(changed_files(tool_log))


def unverified_write_note(tool_log: Sequence[str], project_runs=(),
                          project_commands=None) -> str:
    """What to append when a turn changed files and never showed they work.

    The sibling of unsupported_change_claim, for the opposite fault. That one
    refuses to hand over a claim the tool log contradicts; this one refuses to
    hand over a change the tool log never stands behind.

    A turn edited four C files, spent its last rounds configuring a build,
    and was cut by the round budget on "The changes have been made. Let me
    verify the build works". The build did not work -- a declaration had been
    added without the old one being removed, and the tree no longer compiled.
    Nothing in the answer said so, and the nudge that exists for this could
    not fire, because `cmake` after an edit counted as having run something.

    So the last word on a writing turn is deterministic and comes from the
    tool log: what changed, whether anything verified it, and what that
    verification said. It fires on every exit path, including the budget's,
    which is the one that produced this.
    """
    # Anchored on files the log actually names. A mutation this cannot name
    # is one it cannot describe either, and a warning that says "something
    # changed" helps nobody.

    changed = changed_files(tool_log)

    if not changed:
        return ""

    return write_note(changed, _verification_runs(tool_log), project_runs,
                      project_commands)


def write_note(changed, runs, project_runs=(), project_commands=None) -> str:
    """unverified_write_note on facts already established: the files that
    changed and the verification runs after the last change. The agent core
    path (completion.decide) establishes them from structured records."""

    # The harness's own run of the project's build and tests counts here too.
    # Without this the note said "nothing was run afterwards" to a turn whose
    # tree had just been built and tested by the harness -- and a warning the
    # turn can see is false is a warning it learns to ignore. A command that
    # FAILED is reported by project_build.note, in its own words, so this one
    # stays quiet rather than saying the same thing less precisely.

    if any(status in {"passed", "failed"} for _, status, _ in project_runs):
        return ""

    files = ", ".join(changed)

    if not runs:
        return (f"\n\n⚠ UNVERIFIED: {files} changed, and nothing was run afterwards that "
                f"could show the change works — no build, no test, no lint. "
                f"Treat the change above as unverified.")

    # Order decides. A turn began with `make clean`, which failed because the
    # project has no clean target, and then compiled every edited file
    # successfully. Reporting the earlier failure called a working change
    # broken. What stands is the last verification: a failure still counts
    # when nothing after it passed.

    if runs[-1][1]:
        # It passed -- but a compile of the files that were touched is not a
        # verification of the program they belong to, and a tree that declares
        # how it builds and tests itself has said what one would be. Reported
        # rather than demanded: the suite may be unrunnable here for reasons
        # the turn cannot fix, and a note is honest where a nudge would just
        # spend the last round.

        declared = tuple(getattr(project_commands, "verifies", lambda: ())())

        if declared and syntax_only_verification(runs):
            return (f"\n\n⚠ UNVERIFIED: {files} changed, and the only thing run afterwards "
                    f"compiled the files in isolation. That is not the "
                    f"project's own verification: `{declared[-1][:120]}` was "
                    f"never run, so the change above is unproven at the level "
                    f"the project tests itself.")

        return ""

    failed = runs[-1][0]

    return (f"\n\n⚠ UNVERIFIED: {files} changed, and the last verification to run did "
            f"not pass: `{failed[:120]}`. The change above is not shown to "
            f"work as written.")


def verify_demand(tool_log: Sequence[str]) -> str:
    files = ", ".join(changed_files(tool_log)) or "a file"
    return (f"You changed {files} and ran nothing afterwards, so nothing shows "
            f"the change works. Verify it now with bash: build it, and run it "
            f"if it can run here — including the cases that could reasonably "
            f"fail, not only the one the user named. Then conclude, naming what "
            f"you actually observed. If it genuinely cannot be built or run "
            f"here, say that plainly instead of reporting it as working.")


# The completion states a turn may still be argued out of: either nothing
# covered the change, or what covered it did not pass.

_NEEDS_VERIFICATION = frozenset({
    CompletionVerificationStatus.UNVERIFIED,
    CompletionVerificationStatus.PARTIALLY_VERIFIED,
    CompletionVerificationStatus.FAILED,
})

_PROJECT_OUTCOMES = {
    "passed": VerificationOutcome.PASSED,
    "failed": VerificationOutcome.FAILED,
    "not_run": VerificationOutcome.NOT_RUN,
}


def _record_project_verification(context, commands, runs):
    """Record what the project's own build and tests just said about this turn.

    Full coverage, because these are the commands the project declared as its
    verification -- not a guess about which files a command happened to reach.
    A run that could not be performed is recorded as NOT_RUN, which is not a
    pass and never becomes one.
    """

    from runtime.agent_runtime import AgentRuntime

    if not runs or not getattr(commands, "configured", False):
        return

    policy = getattr(context, "verification_policy", None)
    state = getattr(context, "working_state", None)

    if policy is None or state is None:
        return

    # A turn that has already ended -- interrupted, failed, out of budget --
    # accepts no further state events. The verification still happened; there
    # is simply no longer a completion for it to settle.

    if state.terminal_status != TerminalStatus.RUNNING:
        return

    # Recorded once per generation, like the run itself. The completion gate
    # asks for this before deciding whether to demand a verification, and the
    # end of the turn asks again; both see the same single run, and the
    # working state must not carry it twice.

    if getattr(context, "_project_verification_recorded", None) == state.mutation_generation:
        return

    try:
        context._project_verification_recorded = state.mutation_generation
    except (AttributeError, TypeError):
        pass

    for command, status, text in runs:
        category = (VerificationCategory.UNIT_TEST if project_build.kind(command) == "test"
                    else VerificationCategory.BUILD)
        outcome = _PROJECT_OUTCOMES.get(status, VerificationOutcome.NOT_RUN)
        AgentRuntime._record_verification(context, VerificationEvidence(
            "verification_" + uuid.uuid4().hex, category,
            VerificationCoverage.FULL, status != "not_run", outcome,
            context.working_state.mutation_generation,
            action_id=new_action_id("project_verify"), command=command,
            summary=f"project {project_build.kind(command)} {status}"
                    + (f": {text.splitlines()[0][:120]}" if text else ""),
        ))


def project_build_runs(context, tool_log):
    """The project's own verification, run once per generation of changes.

    Each call used to re-run the tree's build and tests, and the gate is
    consulted up to six times in a turn: a project whose suite takes a minute
    paid six. Nothing can have changed between two calls at the same mutation
    generation, so the first answer is the answer until something is written.

    Returns one (command, status, output) per command, status being "passed",
    "failed" or "not_run" -- and "not_run" is not "passed": a verification
    that could not be performed decided nothing.
    """
    commands = getattr(context, "project_commands", None)
    root = getattr(context, "project_root", "") or "."
    verifier = getattr(context, "project_verifier", None)

    if commands is None or not turn_changed_files(context, tool_log):
        return ()

    state = getattr(context, "working_state", None)
    generation = getattr(state, "mutation_generation", 0)
    cached = getattr(context, "_project_verification", None)

    if cached is not None and cached[0] == generation:
        return cached[1]

    runs = []
    phase = getattr(context, "work_phase", None)

    for command in commands.verifies():
        # Already taken, at this exact state, with a pass. Re-running it
        # cannot say anything new and a real run spent minutes doing so.
        if (phase is not None and getattr(phase, "engaged", False)
                and phase.already_validated(command)):
            runs.append((command, "passed", ""))
            continue

        if verifier is not None:
            status, output = verifier(command)
        else:
            # No verifier: the caller has no execution boundary to offer, so
            # nothing is run rather than run somewhere unexamined.

            status, output = "not_run", "no confined runner for project verification"

        runs.append((command, status, output))

        if status != "passed":
            # `verifies()` is ordered: the tests do not run against a tree
            # that did not build.

            break

    runs = tuple(runs)

    # The lifecycle's TEST phase is about the PROJECT's own verification, and
    # this is where it happens -- once per generation of changes, for every
    # caller. Recorded here rather than at the five call sites above, which
    # would each have had to remember.

    if phase is not None and getattr(phase, "engaged", False):
        for command, status, _ in runs:
            phase.note_validation(command, status)

    try:
        context._project_verification = (generation, runs)
    except (AttributeError, TypeError):
        pass

    return runs


def project_build_gap(context, tool_log):
    """The first of the project's own commands that did not pass, if any.

    A change that has not been built is not a change that works. Reporting
    that was the first step; handing the round back is the one that finishes
    the turn. Nothing here is task-specific: it is the tree's own build and
    its own tests, and a turn that never wrote is never asked.
    """
    for command, status, output in project_build_runs(context, tool_log):
        if status != "passed":
            return command, output

    return "", ""


def asked_to_write_and_did_not(question, tool_log, did_modify,
                               *, is_write=None) -> bool:
    """The user asked for the code to change, and nothing changed.

    Narrower than it looks, and it has to be: the two other gates need the
    answer to say something first -- one catches a claim of a write, the
    other a promise of one. A turn that reads for sixty rounds and then
    answers in prose says neither, so both stay quiet and the turn is handed
    back with an empty diff. Asked "can you validate and make changes in the
    code accordingly", that is not a smaller answer; it is the wrong one.

    The request is the user's own words, and the evidence is the tool log.
    Neither is a guess about intent.
    """
    if did_modify or changed_files(tool_log):
        return False

    if any(entry.split(" ", 1)[0] in _WRITE_TOOLS for entry in tool_log):
        return False

    # `is_write` is the turn's decided answer -- pattern plus reading -- and
    # the pattern alone is only the fallback for callers that have no
    # context to decide from (the tests, and nothing else).
    return is_write_request(question or "") if is_write is None else is_write


def work_order_gap(context, tool_log, cache):
    """The sections this turn has not finished, and the order they are in.

    An order that supplies acceptance checks is judged by running them --
    file granularity marked a section done because a neighbouring section
    shares its file. One that supplies none falls back to the diff.
    """
    order = work_order.find(context.conversation)

    if not order:
        return "", []

    checks = work_order.acceptance(order)

    if not checks:
        return order, work_order.unaddressed(
            order, changed_files(tool_log))

    outputs = {}

    def run(command):
        try:
            envelope = context.tool_executor(
                context, f"acceptance-{abs(hash(command)) % 10**8}",
                "bash", {"command": command}, cache,
            )
        except Exception:                                   # noqa: BLE001
            return False

        # rag_chat appends "(exit N)" to a command that did not exit
        # zero, and nothing to one that did.
        text = envelope.text or ""
        found = re.search(r"\(exit (\d+)\)", text)

        ok = bool(envelope.success) and (found is None
                                         or found.group(1) == "0")

        if not ok:
            # What the check PRINTED, not only what it was. Told a section
            # failed `test $(... | grep -c ...) -eq 1`, a turn could not see
            # that the count was 2 and that the fix was a deletion; the
            # output says so in one number.
            outputs[command] = " ".join(text.split())[:200]

        return ok

    missing = work_order.unmet(order, run)

    for item in missing:
        item["output"] = outputs.get(item["check"], "")

    return order, missing
