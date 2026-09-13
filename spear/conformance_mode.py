"""A turn that asks whether code conforms is shaped, not merely filtered.

Asked to "validate the ACK implementation against VITA49.2", a bound session
listed the structure registry, read all nine of its structures -- Tilt Angle,
Bandwidth, Manufacturer OUI, none of them about acknowledgement -- and rendered
them under "Validated Structures (PASS/ACCEPTABLE)". It never opened a source
file. The conformance guard correctly left that alone: nothing in it is a
verdict on the implementation. The reader was misled anyway, because the
answer LOOKED like an assessment and nothing said what it had not done.

No guard fixes that, because the fault is not a sentence. It is the shape of
the turn: a conformance question was answered without the two things a
conformance answer is made of -- the clauses, and the code. So this module
shapes the turn instead:

  - it recognises a conformance turn from the question, deterministically;
  - it asks the model for a per-clause form, once, in the bound-standard rule
    (cheap, and not what anything below relies on: model-side requests have
    been measured ineffective nine times over);
  - it keeps its own ledger of whether any source file was read; and
  - it appends an ASSESSMENT RECORD the model never sees and cannot write:
    the clauses read, the clauses the answer cites, the difference between
    the two in both directions, the files read, and whether any verdict has
    a leg to stand on.

The record is built only from the turn's own ledgers. It states nothing about
the code, which no deterministic layer can judge; it states what was and was
not looked at, which is exactly the information the reader never had. A turn
that read no source file says so in its first line, above everything else.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import conformance_guard

# What the question has to ask, and what it has to ask it about. "validate
# the config" is not a conformance turn; "is the encoder compliant" is.
_ASKS = re.compile(
    r"\b(?:validat(?:e|es|ed|ion)|verif(?:y|ies|ied|ication)|compliant|"
    r"compliance|conform(?:s|ant|ance|e|ité)?|audit(?:s|ed)?|"
    r"check(?:s|ed)?\s+(?:that|whether|if|against)|valide[rz]?|vérifie[rz]?)"
    r"\b", re.I)
_CODE_SUBJECT = re.compile(
    r"\b(?:implementation|implémentation|code(?:base)?|sources?|converter|"
    r"controllee|controller|encoder|decoder|driver|firmware|function|"
    r"cmdWire\w*|ctl\w+)\b", re.I)

# Tools that read the tree. edit_file reads its region first; bash reads
# whatever the command reads; search_corpus reads the indexed sources.
CODE_TOOLS = frozenset({"bash", "search_corpus", "read_file", "edit_file"})

# A source file named in a command, a path argument or a result.
# A path is taken whole, from its first segment. The lookbehind used to be
# (?<![\w/]), which refused to start on a path already preceded by a slash --
# so an absolute path yielded nothing at all, and a path with a dot in a
# directory name ("CMakeFiles/3.28.3/CompilerIdC/x.c") was picked up from
# after the dot, as "28.3/CompilerIdC/x.c". A line that names the wrong file
# is worse than a line that names none.
_SOURCE_PATH = re.compile(
    r"(?<![\w.\-/])/?(?:[\w.\-]+/)*[\w.\-]+\.(?:c|h|cc|cpp|cxx|hh|hpp|py|rs|go|"
    r"java|js|ts|sh|mk|cmake|dts|dtsi|s|S)\b")

# Commands that put a file's CONTENT in front of the model, as opposed to
# naming it. `find` and `ls` print paths by the dozen and show none of them:
# harvesting their output made "source files read" list files the turn had
# only seen the names of.
_READING_BINARY = frozenset({
    "cat", "head", "tail", "sed", "awk", "less", "more", "bat", "nl", "od",
    "hexdump", "xxd", "strings", "grep", "rg", "egrep", "fgrep", "diff",
})

# Sent to the model on a conformance turn, as part of the bound-standard rule.
FORMAT_RULE = "\n".join((
    "CONFORMANCE TURN",
    "This question asks whether code conforms. Answer as one row per clause:",
    "  clause (as the standard numbers it) | what the code does, with file and line | status",
    "Status is one of: satisfied, violated, not assessed. Cite only clauses you retrieved.",
    "Do not conclude about the implementation as a whole; the record appended after your "
    "answer states what was and was not read.",
))


def is_conformance_turn(question):
    """Does the question ask whether some code conforms?"""
    text = question or ""

    return bool(_ASKS.search(text) and _CODE_SUBJECT.search(text))


@dataclass
class CodeReadLedger:
    """Which source files the tools actually put in front of the model."""

    tools: Counter = field(default_factory=Counter)
    files: set = field(default_factory=set)

    def observe(self, name, arguments=None, text=""):
        if name not in CODE_TOOLS:
            return self

        self.tools[name] += 1

        # Only from the ARGUMENTS. A tool result is where `find` and `grep -l`
        # print the tree, and taking paths from there credited the turn with
        # reading every file it had listed. What the model was shown is what
        # it asked for, and that is in the call.

        for value in (arguments or {}).values():
            if isinstance(value, str):
                self.files |= set(self._read_paths(name, value))

        return self

    @staticmethod
    def _read_paths(name, value):
        """The source files this call put in front of the model."""
        if name != "bash":
            return _SOURCE_PATH.findall(value)

        # A pipeline reads whatever its stages read; a `find` reads nothing.
        # Judged per stage, so `find . -name '*.c' | head` credits nothing
        # while `sed -n 1,50p a.c | grep x` credits a.c.
        found = []

        for stage in re.split(r"[|;&]+|&&", value):
            words = stage.split()

            if words and words[0].rsplit("/", 1)[-1] in _READING_BINARY:
                found += _SOURCE_PATH.findall(stage)

        return found

    def read_anything(self):
        return bool(self.files) or bool(self.tools)


def _sort_sections(sections):
    return sorted(sections, key=lambda s: [int(p) for p in s.split(".")])


def _listed(sections, limit=12):
    ordered = _sort_sections(sections)
    text = ", ".join(f"§{section}" for section in ordered[:limit])

    if len(ordered) > limit:
        text += f" and {len(ordered) - limit} more"

    return text or "none"


def record(answer, clauses, code, *, standard_id="", revision=""):
    """The assessment record: what was read, what was cited, what was not."""
    name = " ".join(part for part in (standard_id, revision) if part) or (
        "the bound standard")
    read = set(clauses.sections)
    cited = conformance_guard.clauses_in(answer)
    cited_not_read = {section for section in cited if not clauses.covers(section)}
    read_not_cited = {section for section in read
                      if not any(section == c or section.startswith(c + ".")
                                 or c.startswith(section + ".") for c in cited)}
    files = sorted(code.files)

    lines = ["ASSESSMENT RECORD (written by the harness from the turn's own "
             "tool log; the model neither sees nor writes it)",
             f"Standard: {name}",
             f"Clauses read this turn: {_listed(read)} ({len(read)})",
             f"Clauses cited in the answer: {_listed(cited)} ({len(cited)})"]

    if cited_not_read:
        lines.append(f"  cited but never read: {_listed(cited_not_read)} — "
                     "nothing in this turn backs those citations")

    if read_not_cited:
        lines.append(f"  read but not assessed: {_listed(read_not_cited)}")

    if files:
        shown = ", ".join(files[:8]) + (f" and {len(files) - 8} more"
                                        if len(files) > 8 else "")
        lines.append(f"Source files read: {shown} ({len(files)})")
    elif code.tools:
        lines.append("Source files read: none identified "
                     f"({sum(code.tools.values())} code-reading calls named "
                     "no source file)")
    else:
        lines.append("Source files read: none")

    if not code.read_anything():
        lines.append("Verdict: none can be issued — no source file was read, "
                     "so nothing above assesses the implementation")
    elif not read:
        lines.append("Verdict: none can be issued — no clause of the "
                     "standard was read")
    else:
        lines.append("Verdict scope: at most the clauses read, on the files "
                     "read; nothing about the implementation as a whole")

    return "\n".join(lines)


def render(answer, clauses, code, *, standard_id="", revision=""):
    """The answer the reader sees on a conformance turn."""
    body = (answer or "").rstrip()
    block = record(body, clauses, code, standard_id=standard_id,
                   revision=revision)

    if not code.read_anything():
        # Above everything else, because everything else looks like an
        # assessment and is not one.
        head = ("No source file was read this turn: nothing in this answer "
                "assesses the implementation.")

        return f"{head}\n\n{body}\n\n---\n{block}" if body else (
            f"{head}\n\n---\n{block}")

    return f"{body}\n\n---\n{block}" if body else f"---\n{block}"


def unaddressed_clauses(answer, carried):
    """Clauses this session established that the answer never accounts for.

    The specification is carried into the writing turn and the turn is asked
    to say, for each rule, what the code does today. Nothing checked it: a
    turn handed ten clauses answered about two, changed one line, and
    stopped -- four runs in a row, each of them building and passing its
    tests, each of them a fraction of the work.

    Naming a clause is not addressing it. The first version accepted the
    clause appearing anywhere in the answer, and passed a turn that had
    restated twelve rules and implemented one -- restating them is precisely
    what the model does. A clause counts as addressed when the sentence that
    names it also says WHERE: a source file, with or without a line. That is
    the whole of "state for each one what the code does today", and it is
    checkable.
    """
    if not carried:
        return []

    placed = set()

    for sentence in _sentences(answer or ""):
        if not _SOURCE_PATH.search(sentence):
            continue

        placed |= _cited(sentence)

    return [item for item in carried if not any(_covers(one, item)
                                               for one in placed)]


# A clause as the standard writes it, KEEPING the rule suffix. The
# conformance guard normalises "8.4.1.1-2" to its section, which is right
# for judging a verdict and wrong here: this has to tell one rule of a
# section from another, or a turn that satisfied 8.4.1.1-2 and stopped would
# be credited with 8.4.1.1-3 as well -- which is the very failure the check
# exists to catch. The marker is required, so a version number or a date
# cannot pass for a citation.
_CITED = re.compile(r"(?:§|\bRule\s+|\bObservation\s+)(\d+(?:\.\d+)*(?:-\d+)?)")


def _cited(sentence):
    return set(_CITED.findall(sentence))


def _covers(cited, carried) -> bool:
    """Does this citation account for that carried clause?

    Two hierarchies meet here and they are not the same one. Sections nest
    with dots -- 8.4 contains 8.4.1.1 -- and a citation of a section
    accounts for everything inside it. Rules hang off a section with a dash,
    and they are SIBLINGS: 8.4.1.1-2 and 8.4.1.1-3 are two different
    obligations, and satisfying one says nothing about the other.

    So a citation reaches downwards and upwards through the dots, and never
    sideways across the dash. Reaching sideways would let a turn that did
    step one be credited with step two, which is the failure this check
    exists to catch; refusing to reach upwards would deny a turn credit for
    a section it had cited by one of its rules.
    """
    if cited == carried:
        return True

    root, kin = cited.split("-", 1)[0], "-" in cited
    base = carried.split("-", 1)[0]

    if kin and "-" in carried:
        return False        # two different rules of the same section

    if kin:
        # A rule accounts for its own section and for any section above it.
        return carried == root or root.startswith(carried + ".")

    return (cited == base or cited.startswith(base + ".")
            or base.startswith(cited + "."))


def _sentences(text):
    return [part for part in re.split(r"(?<=[.!?])\s+|\n+", text or "")
            if part.strip()]


def clause_demand(missing, carried):
    """The one message that sends a turn back to the rules it skipped."""
    listed = ", ".join(f"\u00a7{item}" for item in missing[:10])

    return (
        f"This session established {len(carried)} clauses and your answer "
        f"accounts for {len(carried) - len(missing)} of them. Not addressed: "
        f"{listed}.\n\nFor each one, either make the code satisfy it and say "
        f"which file and line now does, or say plainly that it is already "
        f"satisfied and where, or that you are not implementing it and why. "
        f"Do not restate the rule without saying what the code does about it. "
        f"Work through them one at a time, editing as you go.\n\n"
        f"The clauses are already established and listed above. Do not "
        f"retrieve them again -- read the code and change it."
    )
