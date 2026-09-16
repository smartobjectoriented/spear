"""Identifiers the turn read OUT OF A REPOSITORY, and nothing more.

A code-comparison answer names two kinds of thing: the fields a standard
defines, and the functions and types an implementation calls its own. The
identifier check in `normative_claims` knows only the first, because the only
ledger it reads is the normative one. On a turn that reads source files that is
the wrong question to ask of a function name: it was never going to be in a
clause, and flagging it says nothing about whether the model made it up.

Measured on one real two-turn session against a private repository: twenty
UNGROUNDED_IDENTIFIER findings, of which nineteen named things that were
plainly in front of the model -- ten defined in clauses read one turn earlier,
nine printed by the turn's own greps -- and exactly one existed nowhere. The
check was right to fire and wrong about nineteen twentieths of what it fired
on.

So this module keeps the other half of the answer's vocabulary. The rule it
implements is narrow on purpose:

    an identifier is grounded if its literal form was in content a
    code-reading tool RETURNED to the model during this turn

and not:

    an identifier is grounded if a file somewhere in the repository
    contains it.

The difference is the whole point. A name the model never saw is a name the
model invented, whatever some unopened file happens to hold. Accordingly
nothing here reads the filesystem, nothing here reads a tool's ARGUMENTS, and
a result that only lists paths -- `find`, `ls`, `grep -l` -- grounds nothing,
because a path is not content.

What this grounds is EXISTENCE, and only existence. A name found in a comment
is still just a name; it carries no modality, no authority and no clause. The
normative ledger remains the only source of those, and every other guard --
modality, coherence, cardinality, precedence -- goes on reading it alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import normative_claims

#: Tools whose results may ground an identifier. Named here rather than shared
#: with the conformance record's list, which answers a different question and
#: should be free to change without moving what counts as evidence.
#:
#: `edit_file` is deliberately absent: an edit echoes back what the MODEL
#: wrote, and a draft may not ground itself. Standard tools are absent for the
#: same reason in reverse -- their content is normative evidence, and it is
#: already held, with its modality, by the normative ledger.
GROUNDING_TOOLS = frozenset({"read_file", "search_corpus", "bash"})

#: `path:12:  some code` -- one matched line of one file, as every grep-like
#: tool prints it. The prefix names the file; the body is the content.
_LOCATED = re.compile(r"^\s*(?P<path>[^\s:]+):(?P<line>\d+):(?P<body>.*)$")

#: A lone path on a line of its own: what `find`, `ls` and `grep -l` print.
#: Listing a file is not reading it, so such a line contributes nothing --
#: neither an identifier nor provenance.
#:
#: A path, though, and not merely a lone word: a header prints an enum member
#: on a line by itself, and reading that as a directory listing would throw
#: away the one place some names are ever written.
_LISTED = re.compile(r"^\s*(?:"
                     r"\.{0,2}/?(?:[\w.\-]+/)+[\w.\-]*/?"   # has a directory
                     r"|[\w.\-]+\.[A-Za-z0-9]{1,5}"            # bare filename
                     r")\s*$")

#: A path appearing inside a line that also carries content, so the words of
#: a filename cannot be mistaken for the words of the file.
_EMBEDDED_PATH = re.compile(r"(?:[\w.\-]*/)+[\w.\-]*")


def _content_lines(text):
    """The lines of a tool result that are CONTENT, with their file if named.

    Yields `(path, body)`; `path` is "" when the result did not say which file
    the line came from.
    """
    for line in (text or "").splitlines():
        located = _LOCATED.match(line)

        if located is not None:
            yield located.group("path"), located.group("body")
            continue

        # A bare path is a listing. Checked after the located form, so
        # `src/a.c:12:x` is read as content and `src/a.c` is not.

        if not line.strip() or _LISTED.match(line):
            continue

        yield "", _EMBEDDED_PATH.sub(" ", line)


@dataclass
class CodeIdentifier:
    """One name, and the answer to "why was this considered grounded?"."""

    form: str
    files: set = field(default_factory=set)
    tools: set = field(default_factory=set)

    def to_dict(self):
        return {"identifier": self.form, "files": sorted(self.files),
                "tools": sorted(self.tools)}


@dataclass
class CodeEvidenceLedger:
    """Every identifier this turn's code-reading tools put in front of the
    model, kept by normalised spelling with where it came from."""

    seen: dict = field(default_factory=dict)

    def observe(self, name, text=""):
        """One tool result, exactly as the model was shown it.

        The result only. A tool's arguments say what was ASKED for, and a
        pattern the model grepped for is a name it already believed in --
        grounding on that would ground every invention that survives one
        fruitless search.
        """
        if name not in GROUNDING_TOOLS or not text:
            return self

        for path, body in _content_lines(text):
            for form in normative_claims.identifiers_in(body):
                key = normative_claims.normalise(form)
                entry = self.seen.get(key)

                if entry is None:
                    entry = self.seen[key] = CodeIdentifier(form)

                entry.tools.add(name)

                if path:
                    entry.files.add(path)

        return self

    def identifiers(self):
        """The spellings, for the identifier check to permit."""
        return {entry.form for entry in self.seen.values()}

    def knows(self, token):
        return normative_claims.normalise(token) in self.seen

    def provenance(self):
        """The audit's record, ordered so two runs write the same bytes."""
        return [self.seen[key].to_dict() for key in sorted(self.seen)]
