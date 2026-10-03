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
  decide            VERIFIED when files changed and the last verification
                    run after the change completed and passed; UNVERIFIED
                    when files changed without that; NO_CHANGE otherwise.
                    Decided from the structured evidence alone.
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


def evidence(record: ToolRecord, label=lambda path: path) -> Evidence:
    changed = (tuple(label(path) for path in record.changed_paths)
               if record.name in WRITE_TOOLS and not record.refused else ())

    return Evidence(record.name, changed, record.command, record.exit_code,
                    record.refused, record.timed_out)


def changed_files(log) -> tuple[str, ...]:
    """Every file the turn changed, in the order it first changed."""
    return tuple(dict.fromkeys(path for item in log for path in item.changed))


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


def decide(log, *, project_runs=(), project_commands=None) -> Verdict:
    """`log` is the turn's Evidence, in call order."""
    from agent_runtime import write_note

    changed = changed_files(log)

    if not changed:
        return Verdict("NO_CHANGE", ())

    note = write_note(changed, verification_runs(log), project_runs, project_commands)

    if note:
        return Verdict("UNVERIFIED", changed, note.strip().removeprefix("⚠ UNVERIFIED:").strip())

    return Verdict("VERIFIED", changed)


# A sentence that states the change works, is done or was checked, without
# hedging. Matched per sentence; negations and conditionals are left alone.
_CLAIM = re.compile(
    r"\b(verified|verifies|confirm(?:ed|s)?|correct(?:ly)?|complete[ds]?|"
    r"works?|working|successful(?:ly)?|succeed(?:ed|s)?|ensures?|guarantees?|"
    r"always (?:be )?(?:present|there|available|recreated|created)|"
    r"will (?:always|now) )\b|✓|✔", re.IGNORECASE)
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
