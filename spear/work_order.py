"""A task file with numbered sections is a list, and the turn is judged by it.

Handed a seven-section work order -- A through G, across five files -- the
model changed the function signature the first section asks for, propagated it
until the tree compiled, and answered "The changes have been made
successfully", listing four items. Sections C, D and E had not been touched.
Twice in a row, at 60 rounds and at 120: the budget was not the reason. What
it does is the mechanical part, and it stops when the compiler stops
complaining.

Nothing here judges whether a section was done WELL; that needs reading the
code against the standard, and no deterministic layer can. What it judges is
whether the files a section names were written to at all, which is a fact the
tool log already holds. A section whose files are untouched was not attempted.

So the turn gets one round back, naming the sections it skipped, and the
answer carries the tally either way. Both are built from the work order the
operator supplied and the diff the turn produced, and the model neither writes
nor sees the tally.
"""

from __future__ import annotations

import os
import re

# `## A. command_wire.h -- the subtype becomes a parameter`. Lettered on
# purpose: "## Scope", "## Grounding" and "## Style" are the parts of a work
# order that ask for nothing, and a section that asks for nothing cannot be
# skipped.
_SECTION = re.compile(r"^#{1,4}\s+([A-Z])[.)]\s+(.+?)\s*$", re.M)

# A file named anywhere in a section, with or without its directory.
_FILE = re.compile(
    r"(?<![\w.\-/])/?(?:[\w.\-]+/)*[\w.\-]+"
    r"\.(?:c|h|cc|cpp|cxx|hh|hpp|py|rs|go|java|js|ts|sh|mk|cmake|rst|md)\b")

# The block a !read leaves in the conversation, and the marker rag_chat puts
# in front of it.
_READ_BLOCK = re.compile(r"\[file: ([^\]\n]+)\]\n(.*)", re.S)


def find(conversation) -> str:
    """The work order this turn is working from, or "".

    The operator reads it in with !read, so it arrives as an ordinary history
    entry. The most recent one wins, and it only counts as a work order if it
    has lettered sections: a file read for reference is not a task list.
    """
    for message in reversed(list(conversation or ())):
        for block in getattr(message, "content", ()):
            text = getattr(block, "text", "") or getattr(block, "content", "")

            if not isinstance(text, str) or "[file: " not in text:
                continue

            found = _READ_BLOCK.search(text)

            if found and len(_SECTION.findall(found.group(2))) >= 2:
                return found.group(2)

    return ""


def sections(text):
    """Each lettered section, with the files it names."""
    found = []
    marks = list(_SECTION.finditer(text or ""))

    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        body = text[mark.end():end]
        files = []

        for name in _FILE.findall(mark.group(2) + "\n" + body):
            base = os.path.basename(name)

            if base not in files:
                files.append(base)

        found.append({"label": mark.group(1), "title": mark.group(2),
                      "files": files})

    return found


# `A: grep -q CmdAckKind src/command/command_wire.h` under an "## Acceptance"
# heading. One line per section, the section's letter, a colon, a shell
# command that exits zero when the section is done.
_ACCEPTANCE = re.compile(r"^\s*([A-Z*])\s*:\s*(\S.*?)\s*$", re.M)

# The label of a check that gates all the others: `*: cmake --build ...`.
GATE = "*"
_ACCEPTANCE_HEAD = re.compile(r"^#{1,4}\s+Acceptance\s*$", re.M | re.I)


def acceptance(text):
    """The checks a work order supplies for its own sections.

    File granularity is not enough and this is why: section B and section C
    both live in command_wire.c, so a write for B marked C addressed, and a
    single new parameter threaded through cmdlink.c marked D done when D asks
    for two packets instead of one. A section knows what finishing it looks
    like; the work order is where that belongs, not in the harness.
    """
    head = _ACCEPTANCE_HEAD.search(text or "")

    if not head:
        return []

    body = (text or "")[head.end():]
    stop = re.search(r"^#{1,4}\s+\S", body, re.M)

    if stop:
        body = body[:stop.start()]

    return [{"label": found.group(1), "check": found.group(2)}
            for found in _ACCEPTANCE.finditer(body)]


def unmet(text, run):
    """Sections whose own acceptance check does not pass.

    A gate check (`*:`) is a precondition, and a failing one makes every
    section unverified rather than passing. Six of seven checks once passed
    on a tree that did not compile: the header carried the new signature, the
    .c file carried the new body and the OLD signature, and every grep for a
    new token found it. A grep sees presence, not coherence; only the build
    sees both.

    @param run  a callable taking a shell command and returning True when it
                exited zero. The runtime supplies the sandboxed executor; a
                test supplies a stub.
    """
    checks = acceptance(text)
    gates = [item for item in checks if item["label"] == GATE]
    sections = [item for item in checks if item["label"] != GATE]

    for gate in gates:
        if not run(gate["check"]):
            return [dict(gate, label=GATE)] + sections

    return [item for item in sections if not run(item["check"])]


def unaddressed(text, changed):
    """Sections naming files, none of which this turn wrote to."""
    written = {os.path.basename(path) for path in changed}

    return [item for item in sections(text)
            if item["files"] and not (set(item["files"]) & written)]


def _name(item):
    if item.get("check"):
        printed = item.get("output")
        tail = f" -> {printed[:90]}" if printed else ""

        return f"{item['label']} (`{item['check'][:70]}`{tail})"

    return f"{item['label']} ({', '.join(item['files'][:3])})"


def demand(missing) -> str:
    """The one message that sends a turn back to the sections it skipped."""
    listed = "; ".join(_name(item) for item in missing)

    checked = any(item.get("check") for item in missing)
    verify = (
        " Each one shows the command that decides it: RUN THAT COMMAND FIRST "
        "and read its output. Part of the section may already be done -- asked "
        "for a section it had half-written, a turn appended the same test "
        "functions a second time and the file stopped compiling on four "
        "redefinitions. Edit only what the check still needs."
        if checked else "")

    return (
        f"The work order has sections that are not done: {listed}." + verify
        + f" Changing the signature until the tree compiles is section A's "
        f"work, not the task. Make the edit each one needs, with a SHORT "
        f"old_text copied verbatim from the file. If a section genuinely "
        f"needs no change, say which and why -- but do not report the task "
        f"complete while its check fails."
    )


def record(text, changed, missing=None) -> str:
    """The tally appended to a work-order turn.

    From the acceptance checks when the order supplies them, and from the
    diff otherwise.
    """
    if missing is not None:
        checks = acceptance(text)

        if checks:
            failed = {item["label"] for item in missing}
            passed = ([] if GATE in failed else
                      [item["label"] for item in checks
                       if item["label"] not in failed and item["label"] != GATE])
            lines = ["WORK ORDER TALLY (written by the harness by running the "
                     "order's own acceptance checks; the model neither sees "
                     "nor writes it)"]

            if passed:
                lines.append(f"Sections whose check passes: {', '.join(passed)}")

            if not missing:
                return ""

            gated = [item for item in missing if item["label"] == GATE]

            if gated:
                lines.append("The order's gate check fails, so no section is "
                             f"verified: `{gated[0]['check'][:90]}`")
                lines.append("Sections not verified: " + ", ".join(
                    item["label"] for item in missing
                    if item["label"] != GATE))
            else:
                lines.append("Sections whose check fails: "
                             + "; ".join(_name(item) for item in missing))

            return "\n\n---\n" + "\n".join(lines)

    listed = sections(text)

    if not listed:
        return ""

    written = {os.path.basename(path) for path in changed}
    done = [item["label"] for item in listed
            if item["files"] and set(item["files"]) & written]
    missing = [item for item in listed if item["files"]
               and not (set(item["files"]) & written)]
    silent = [item["label"] for item in listed if not item["files"]]

    lines = ["WORK ORDER TALLY (written by the harness from the diff; the "
             "model neither sees nor writes it)"]

    if done:
        lines.append(f"Sections with a write this turn: {', '.join(done)}")

    if missing:
        lines.append("Sections whose files were never written to: "
                     + "; ".join(_name(item) for item in missing))

    if silent:
        lines.append(f"Sections naming no file, not tracked: {', '.join(silent)}")

    if not missing:
        return ""

    return "\n\n---\n" + "\n".join(lines)
