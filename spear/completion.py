"""The evidence plane of a coding turn: canonical log, verdict, final answer.

The agent core decides when it has answered; it does not decide whether the
answer is true. That is decided here, from what the turn's tools actually
did, and it is written into the answer itself:

  canonical_entry   one line per call in the form SPEAR's checks read --
                    what changed ("OK: <path> updated"), what ran and how it
                    ended ("(exit N)", "ERROR: ..."), what was refused --
                    whatever the model-facing result looked like.
  decide            VERIFIED when files changed and the last verification
                    run after the change completed and passed; UNVERIFIED
                    when files changed without that; NO_CHANGE otherwise.
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


def decide(tool_log, *, project_runs=(), project_commands=None) -> Verdict:
    from agent_runtime import changed_files, unverified_write_note

    changed = tuple(changed_files(tool_log))

    if not changed:
        return Verdict("NO_CHANGE", ())

    note = unverified_write_note(tool_log, project_runs, project_commands)

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
