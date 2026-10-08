"""The evidence plane of a coding turn: canonical log, verdict, final answer.

The agent core decides when it has answered; it does not decide whether the
answer is true. That is decided here, from what the turn's tools actually
did, and it is written into the answer itself:

  evidence          the structured record of one call: what it changed,
                    which command it ran and how that ended, whether it was
                    refused. Taken from the ToolRecord's own fields, never
                    from any rendering of it.
  canonical_entry   the compact text of one call for the transcript and the
                    tool log ("OK: <path> updated", "(exit N)", "ERROR: ...").
                    A display: its head is cut to 200 characters, so nothing
                    is decided from it.
  decide            VERIFIED when source changed and the checks that would
                    show what the answer claims passed after the last source
                    change (a build for a build, a clean, a build and a look
                    for a link surviving both); UNVERIFIED when source changed
                    without that, or when the project's own validation
                    failed on the final tree; NO_CHANGE otherwise. Decided
                    from the structured evidence alone.
  qualify           an UNVERIFIED answer opens with the verdict, and every
                    sentence that asserts success unconditionally is marked
                    as unverified where it stands, so the answer cannot say
                    one thing while the verdict says another.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from agent.host import ToolRecord

WRITE_TOOLS = ("patch", "write_file")


@dataclass(frozen=True)
class Verdict:
    state: str          # VERIFIED | UNVERIFIED | NO_CHANGE
    changed: tuple[str, ...]
    reason: str = ""


@dataclass(frozen=True)
class Evidence:
    """One call as the evidence plane reads it."""
    name: str
    changed: tuple[str, ...]        # files this call changed, labelled
    command: str | None = None      # the terminal command, in full
    exit_code: int | None = None
    refused: bool = False
    timed_out: bool = False
    output: str = ""                # a command's output, its tail


def evidence(record: ToolRecord, label=lambda path: path) -> Evidence:
    changed = (tuple(label(path) for path in record.changed_paths)
               if record.name in WRITE_TOOLS + ("delete_file",) and not record.refused else ())
    output = ""

    if record.name == "terminal":
        try:
            output = str(json.loads(record.result).get("output") or "")[-2000:]
        except (ValueError, AttributeError):
            output = ""

    return Evidence(record.name, changed, record.command, record.exit_code,
                    record.refused, record.timed_out, output)


def changed_files(log) -> tuple[str, ...]:
    """Every file the turn changed, in the order it first changed."""
    return tuple(dict.fromkeys(path for item in log for path in item.changed))


# ---------------------------------------------------------------- freshness
#
# A check proves the state the tree was in when it ran, and no later one.
# Every successful change to delivered source moves the turn to a new source
# epoch; a check counts for the final answer only when it ran in the final
# epoch. What a build writes into its own output areas is not a source change:
# the control plane's generated trees (/build/tmp/, /generated/) and, when the
# caller knows the project, whatever git ignores.

_GENERATED = re.compile(r"(?:^|/)(?:build/tmp|generated|__pycache__|node_modules|\.git)(?:/|$)")


def generated_path(path: str) -> bool:
    """A path whose change is a build's side effect, not a source change."""
    return bool(_GENERATED.search(str(path)))


@dataclass(frozen=True)
class Check:
    """One recognised validation stage, as it ran."""
    command: str
    kind: str           # build | clean | clean+build | task | test | syntax | check
    passed: bool
    epoch: int          # the source epoch it ran in
    index: int          # the call it belongs to, in log order


# A lone & ends a list and runs it in the background; the & of 2>&1, &> and |&
# is a redirection.
_SEPARATOR = re.compile(r"(&&|\|\||(?<![&>|])&(?![&>])|[;|\n])")
_FAILURE = re.compile(r"\bERROR\b|\berror:|\bFAILED\b|\bfailed\b|returned non-zero|"
                      r"No rule to make target|command not found|Traceback")
_CHECKERS = frozenset({"ls", "readlink", "test", "[", "stat", "file", "realpath"})


def _stages(command: str):
    """(argv, the separator after it) for each stage, in order."""
    import request_scope
    import shlex

    parts = _SEPARATOR.split(command.replace("\\;", "+"))

    for index in range(0, len(parts), 2):
        try:
            argv = shlex.split(parts[index], posix=True)
        except ValueError:
            argv = parts[index].split()

        while argv and (argv[0] in request_scope._SHELL_KEYWORDS
                        or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[0])):
            argv = argv[1:]

        after = parts[index + 1] if index + 1 < len(parts) else ""

        if argv:
            yield argv, after


def _backgrounded(stages) -> list[bool]:
    """Per stage, whether it runs in the background. `a && b &` sends the
    whole list there: its exit status is the shell's 0, not the commands'."""
    flags, group = [], []

    for _, after in stages:
        group.append(len(flags))
        flags.append(False)

        if after in (";", "\n", "&", ""):
            for index in group:
                flags[index] = after == "&"

            group = []

    return flags


def _in_tree(path: str, root: str) -> str | None:
    """A written path as the tree names it, or None when it is outside it."""
    if not path.startswith("/"):
        return path

    if root and (path == root or path.startswith(root.rstrip("/") + "/")):
        return path[len(root.rstrip("/")) + 1:]

    return None


def _kind(argv) -> str | None:
    """What one stage validates, if anything."""
    from pathlib import Path

    binary = Path(argv[0]).name
    args = argv[1:]

    if binary == "bitbake":
        # -g, -e, -p and friends inspect the build without running it.

        if any(flag in args for flag in ("-g", "-e", "-p", "-n", "-s", "-h", "--help",
                                          "--version", "--dry-run", "--show-versions")):
            return None

        task = args[args.index("-c") + 1] if "-c" in args[:-1] else None
        task = task.removeprefix("do_") if task else None

        if task in ("clean", "cleanall", "cleansstate"):
            return "clean"

        return "task" if task and task not in ("build", "compile") else "build"

    if binary == "build.sh":
        recipes = [arg for arg in args if not arg.startswith("-")]

        if not recipes or any(flag in args for flag in ("-h", "-l")):
            return None

        return "clean+build" if "-c" in args else "build"

    if binary in ("make", "ninja"):
        targets = [a for a in args if not a.startswith("-") and "=" not in a]
        if targets and all(t in ("clean", "distclean", "mrproper") for t in targets):
            return "clean"
        if any(t in ("test", "check") for t in targets):
            return "test"
        return "build"

    if binary in ("pytest", "ctest") or args[:2] == ["-m", "unittest"] or args[:2] == ["-m", "pytest"]:
        return "test"

    if (binary in ("bash", "sh") and "-n" in args) or "py_compile" in args:
        return "syntax"

    if binary in _CHECKERS:
        return "check"

    return None


def timeline(log, generated=generated_path, root=""):
    """(final source epoch, the checks, the source files changed), in order.

    A refused call changes nothing. A command's own writes move the epoch at
    the stage that makes them, so `sed -i ... && make` validates its edit and
    `make && sed -i ...` does not.
    """
    import request_scope
    from agent_runtime import _VERIFYING
    from verification import VerificationPolicy

    policy = VerificationPolicy()
    epoch, checks, changed = 0, [], []

    for index, item in enumerate(log):
        if item.refused:
            continue

        sources = [path for path in item.changed if not generated(path)]

        if sources:
            epoch += 1
            changed.extend(sources)
            continue

        if item.name != "terminal" or not item.command:
            continue

        stages = list(_stages(item.command))
        background = _backgrounded(stages)
        ran = not item.timed_out and item.exit_code == 0
        masked = False

        for position, (argv, after) in enumerate(stages):
            writes = [path for path in (_in_tree(item, root) for item in
                                        request_scope._shell_destinations(argv))
                      if path and not generated(path)]

            if writes:
                epoch += 1
                changed.extend(writes)

            kind = _kind(argv)

            # Sent to the background, a stage returns before it has run.

            if kind is None or background[position]:
                continue

            # Piped into a filter, a stage's exit status is the filter's; what
            # the output says is then all there is to go on.

            if after == "|" and "pipefail" not in item.command:
                masked = True

            passed = ran and not (masked and _FAILURE.search(item.output or ""))

            if kind == "check":
                passed = passed and not re.search(r"No such file|cannot access",
                                                  item.output or "")

            checks.append(Check(item.command, kind, passed, epoch, index))

        # What the verification policy counts as a check and no stage kind
        # names -- running the program just changed, a linter -- still counts,
        # as a run of the tree as it stands after the command.

        if not any(background) and not any(check.index == index and check.kind != "check"
                                           for check in checks):
            category, _ = policy.classify_command(item.command, changed_paths=tuple(changed))

            if category in _VERIFYING:
                passed = ran and not (masked and _FAILURE.search(item.output or ""))
                checks.append(Check(item.command, "run", passed, epoch, index))

    return epoch, checks, tuple(dict.fromkeys(changed))


# What a final answer claims, sentence by sentence; a hedged sentence claims
# nothing. Each claim names the checks that would have to have passed, in the
# final source epoch and in this order.

_CLAIMS = (
    ("survives a clean and rebuild",
     re.compile(r"surviv|persist|after (?:a |the |each |every )?clean|"
                r"clean(?:/| and | \+ |-and-)(?:re)?build|"
                r"always (?:be )?(?:present|there|available|recreated|created)|"
                r"recreated (?:automatically|on every|every|each|after)", re.I),
     (("clean", "clean+build"), ("build", "clean+build"), ("check",))),
    ("builds", re.compile(r"\bbuild(?:s|ing)? (?:succeed|pass|work|complete|clean)|"
                          r"\bbuilt successfully|\bcompiles\b", re.I),
     (("build", "clean+build"),)),
    ("tests pass", re.compile(r"\btests? (?:now )?pass", re.I), (("test",),)),
)


def _claims(answer: str):
    sentences = [part for part in _SENTENCE.split(answer or "") if part.strip()]

    for name, pattern, needs in _CLAIMS:
        if any(pattern.search(sentence) and not _HEDGE.search(sentence)
               for sentence in sentences):
            yield name, needs


def _shown(checks, needs) -> bool:
    """Whether passing checks of these kinds ran, in this order."""
    position = 0

    for check in checks:
        if position < len(needs) and check.passed and check.kind in needs[position]:
            # clean+build answers for the clean and the build at once.

            position += 2 if (check.kind == "clean+build" and position + 1 < len(needs)
                              and "clean+build" in needs[position + 1]) else 1

    return position >= len(needs)


def verification_runs(log, policy=None) -> list[tuple[str, bool]]:
    """Every verifying command after the last change, and whether it passed:
    ran (not refused), finished (not timed out), and exited 0."""
    from agent_runtime import _VERIFYING
    from verification import VerificationPolicy

    policy = policy or VerificationPolicy()
    changed = changed_files(log)
    runs, after_change = [], False

    for item in log:
        if item.changed:
            runs, after_change = [], True
        elif item.name == "terminal" and after_change and item.command:
            category, _ = policy.classify_command(item.command, changed_paths=changed)

            if category in _VERIFYING:
                runs.append((item.command, not item.refused and not item.timed_out
                             and item.exit_code == 0))

    return runs


def canonical_entry(record: ToolRecord, label=lambda path: path) -> str:
    head = f"{record.name} {json.dumps(dict(record.arguments), ensure_ascii=False)[:200]}"

    if record.refused:
        try:
            reason = json.loads(record.result).get("error", record.result)
        except (ValueError, AttributeError):
            reason = record.result
        return f"{head}\nERROR: {str(reason)[:300]}"

    if record.name == "terminal":
        try:
            data = json.loads(record.result)
        except ValueError:
            data = {}

        if record.timed_out:
            return f"{head}\nERROR: timed out\n{(data.get('output') or '')[-300:]}"

        if data.get("error"):
            return f"{head}\nERROR: {data['error']}"

        body = (data.get("output") or "(empty)")[:400]

        if record.exit_code not in (None, 0):
            body += f"\n(exit {record.exit_code})"

        return f"{head}\n{body}"

    if record.name in WRITE_TOOLS:
        if record.changed_paths:
            return f"{head}\nOK: {label(record.changed_paths[0])} updated"

        return f"{head}\nERROR: {record.result[:300]}"

    if not record.ok:
        return f"{head}\nERROR: {record.result[:300]}"

    return f"{head}\n{record.result[:400]}"


# ------------------------------------------------------- project validation
#
# The harness's own run of the project's declared build and tests. Its outcome
# is a fact about the tree it ran on, kept as such: a pass may stand for what
# it demonstrates, a failure is negative evidence, and a run that did not
# happen demonstrates nothing.

PASSED, FAILED, NOT_RUN = "PASSED", "FAILED", "NOT_RUN"


@dataclass(frozen=True)
class ProjectValidationEvidence:
    """One harness run of a project command, against one source epoch."""
    status: str                 # PASSED | FAILED | NOT_RUN
    validation_kind: str        # build | test
    source_epoch: int
    command: str
    exit_code: int | None = None
    evidence: str = ""          # what the run printed, when it said anything
    origin: str = ""            # configured | probed: where the command came from


def project_evidence(project_runs, project_commands=None, epoch=0):
    """Harness runs as ProjectValidationEvidence.

    A (command, status, output) run carries no epoch of its own: it is taken
    to have run on the tree as it stands, `epoch`. Its exit status counts only
    when nothing filters it -- piped into a filter without pipefail, a pass
    says nothing and a failure in its output is still a failure.
    """
    test = getattr(project_commands, "test", None)
    found = []

    for run in project_runs or ():
        if isinstance(run, ProjectValidationEvidence):
            found.append(run)
            continue

        command, status, output = run
        status = {"passed": PASSED, "failed": FAILED}.get(str(status).lower(), NOT_RUN)

        if status == PASSED and _masked(command):
            status = FAILED if _FAILURE.search(output or "") else NOT_RUN

        kind = "test" if test and command == test else "build"
        origin = project_commands.origin(command) if hasattr(project_commands, "origin") else ""
        found.append(ProjectValidationEvidence(
            status, kind, epoch, command, 0 if status == PASSED else None,
            str(output or "")[-2000:], origin))

    return found


def _masked(command: str) -> bool:
    """Whether a stage's exit status is hidden: behind a pipe, or by running
    in the background."""
    stages = list(_stages(command))

    return any(_backgrounded(stages)) or ("pipefail" not in command and any(
        after == "|" for _, after in stages))


def decide(log, *, project_runs=(), project_commands=None, answer="",
           generated=generated_path, root="") -> Verdict:
    """`log` is the turn's Evidence, in call order; `answer` the core's own.

    VERIFIED only when the checks that would show what the answer claims ran
    and passed in the final source epoch -- after the last change to the
    delivered source. A check from before that change proves an earlier tree,
    and a failed project validation on the final tree is never outweighed.
    """
    from agent_runtime import syntax_only_verification

    epoch, checks, changed = timeline(log, generated, root)

    if not changed:
        return Verdict("NO_CHANGE", ())

    files = ", ".join(changed)
    project = project_evidence(project_runs, project_commands, epoch)
    fresh = [run for run in project if run.source_epoch == epoch]
    failed = next((run for run in fresh if run.status == FAILED), None)

    if failed is not None:
        return Verdict("UNVERIFIED", changed, (
            f"{files} changed, and the project's own {failed.validation_kind} "
            f"failed on the final source: `{failed.command[:120]}`. The change "
            f"above is not shown to work as written."))

    harness = [Check(run.command, run.validation_kind, True, epoch, len(log))
               for run in fresh if run.status == PASSED]
    final = [check for check in checks if check.epoch == epoch] + harness
    earlier = ([check for check in checks if check.epoch < epoch and check.passed]
               + [run for run in project if run.source_epoch < epoch
                  and run.status == PASSED])
    runs = list(dict.fromkeys((check.command, check.passed) for check in final
                              if check.kind not in ("clean", "check")
                              and check not in harness))

    if not runs and not harness:
        if earlier:
            return Verdict("UNVERIFIED", changed, (
                f"{files} changed, and the final change was not revalidated "
                f"after the last source modification: what passed ran on an "
                f"earlier state of the tree."))

        return Verdict("UNVERIFIED", changed, (
            f"{files} changed, and nothing was run afterwards that could show "
            f"the change works — no build, no test, no lint. Treat the change "
            f"above as unverified."))

    if not harness:
        # Order decides: what stands is the last verification to run.

        if not runs[-1][1]:
            return Verdict("UNVERIFIED", changed, (
                f"{files} changed, and the last verification to run did not "
                f"pass: `{runs[-1][0][:120]}`. The change above is not shown "
                f"to work as written."))

        declared = tuple(getattr(project_commands, "verifies", lambda: ())())

        if declared and syntax_only_verification(runs):
            return Verdict("UNVERIFIED", changed, (
                f"{files} changed, and the only thing run afterwards compiled "
                f"the files in isolation. That is not the project's own "
                f"verification: `{declared[-1][:120]}` was never run, so the "
                f"change above is unproven at the level the project tests "
                f"itself."))

    unmet = [name for name, needs in _claims(answer) if not _shown(final, needs)]

    if unmet:
        return Verdict("UNVERIFIED", changed, (
            f"{files} changed; what passed after the last change does "
            f"not show that it {' or that it '.join(unmet)} -- no "
            f"{' and '.join(_needed(unmet))} ran on the final source state."))

    return Verdict("VERIFIED", changed)


def _needed(unmet):
    words = {"survives a clean and rebuild": "clean, build and check",
             "builds": "build", "tests pass": "test run"}
    return [words[name] for name in unmet]


# A sentence that states the change works, is done or was checked, without
# hedging. Matched per sentence; negations and conditionals are left alone.
_CLAIM = re.compile(
    r"\b(verified|verifies|confirm(?:ed|s)?|correct(?:ly)?|complete[ds]?|"
    r"works?|working|successful(?:ly)?|succeed(?:ed|s)?|ensures?|guarantees?|"
    r"always (?:be )?(?:present|there|available|recreated|created)|"
    r"will (?:always|now) |surviv(?:es?|ed|ing)|persist(?:s|ed|ing|ent)?|"
    r"(?:automatically|always) (?:re)?created|"
    r"recreated (?:automatically|on every|every|each|after))\b|✓|✔", re.IGNORECASE)
_HEDGE = re.compile(
    r"\b(not|n't|unverified|untested|should|may|might|could|would|expected|"
    r"if |once |assuming|likely|unless|cannot|could not|unable)\b", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`*(\[])|\n")


def qualify(answer: str, verdict: Verdict) -> str:
    """The answer as SPEAR hands it over, consistent with its verdict."""
    if verdict.state != "UNVERIFIED":
        return answer

    header = "**UNVERIFIED** — " + (verdict.reason or (
        f"{', '.join(verdict.changed)} changed, and nothing this turn ran shows "
        f"that the change works."))

    def mark(sentence: str) -> str:
        stripped = sentence.strip()

        if (stripped and _CLAIM.search(stripped) and not _HEDGE.search(stripped)
                and not stripped.startswith(("#", "|", "```", "-  ", "    "))):
            body = stripped.replace("✓", "").replace("✔", "").rstrip()
            return sentence.replace(stripped, f"{body} *(not verified)*")

        return sentence

    pieces, last = [], 0

    for match in _SENTENCE.finditer(answer or ""):
        pieces.append(mark(answer[last:match.start()]))
        pieces.append(match.group(0))
        last = match.end()

    pieces.append(mark((answer or "")[last:]))

    return header + "\n\n" + "".join(pieces)
