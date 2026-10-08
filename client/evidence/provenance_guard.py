"""Where an answer says its evidence came from, checked against where it came from.

Every deterministic layer before this one validates what an answer claims about
a structure. None of them looks at the citations, and a citation is the part a
reader is least able to check. So a supported control that answers correctly,
with the right word numbers and the right source ids, can also append

    [TICKER-TIME-9 2020-R2026 §7.2.2-1, p.85, std-5287d23d…](file:///home/user/verdin/…pdf#page=85)

and the fabricated half reaches the reader looking exactly like provenance --
past the evidence guard, the entailment proof, the oracle and the geometry
adapter, because none of them is about citations.

The rule is not that the path looks wrong. It is that the tools never produced
it. Probing all four tools against the real store and the synthetic corpus
returns no URL and no filesystem path of any kind, so what the model wrote came
from nowhere. Rather than trust that inventory, the check is turn-local: a
provenance value may appear in the answer only if it appeared in a tool result
this turn.

Removal is surgical. The answer's substance is not the sanitizer's business,
and replacing a fabricated citation with a guessed one would be the same fault
wearing a better disguise -- so a fabricated reference is deleted, never
substituted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FABRICATED_SOURCE_ID = "FABRICATED_SOURCE_ID"
FABRICATED_URL = "FABRICATED_URL"
FABRICATED_PATH = "FABRICATED_PATH"

# The identifier families the store actually issues. Probed, not assumed:
# std, bfd, bit, cel, fld, vgr from the real corpus, plus pkg from the
# synthetic one.

_IDENT = re.compile(
    r"\b(?:std|bfd|fld|pkg|vgr|bit|tbl|cel|evt|lnk)-[0-9a-f]{4,}\b")

# A URL of any scheme, and an absolute path that purports to name a document.
# Both are checked wherever they appear, because a file:// URI is provenance
# whatever section it sits in.

_URL = re.compile(r"\b(?:file|https?|ftp|s3|gs)://[^\s)>\]\"'`]+", re.I)
_UNIX_PATH = re.compile(
    r"(?<![\w.\-/])/(?:home|Users|mnt|var|opt|tmp|etc|usr|srv|root|data)"
    r"/[^\s)>\]\"'`,;]*")
_WIN_PATH = re.compile(r"(?<![\w])[A-Za-z]:[\\/](?:[^\s)>\]\"'`,;]*)")

# A markdown link whose target is the thing being checked. The visible text is
# the citation a reader reads and is kept; only the target is at issue.

_LINK = re.compile(r"\[([^\]\n]{0,200})\]\(([^)\s]{1,400})\)")

# Left behind once a reference is cut out.
_EMPTY_BRACKETS = re.compile(r"\(\s*[,;]?\s*\)|\[\s*\]|<\s*>")
_LOOSE_PUNCT = re.compile(r"[ \t]+([,.;:)])")
_BLANK_RUN = re.compile(r"\n{3,}")
_TRAILING = re.compile(r"[ \t]+$", re.M)


@dataclass
class ProvenanceLedger:
    """Exactly what the tools put in front of the model this turn."""

    source_ids: set = field(default_factory=set)
    urls: set = field(default_factory=set)
    paths: set = field(default_factory=set)

    def observe(self, text):
        """Record one tool result, verbatim. Nothing is derived from it.

        A document URL is not reconstructed from a standard_id, and a file
        path is not built from a page number: if the tool did not print it,
        the session never saw it.
        """
        if not text:
            return self

        self.source_ids |= set(_IDENT.findall(text))
        self.urls |= set(_URL.findall(text))
        self.paths |= set(_UNIX_PATH.findall(text)) | set(
            _WIN_PATH.findall(text))

        return self

    def knows(self, value):
        return (value in self.source_ids or value in self.urls
                or value in self.paths)


# A reference at the end of a sentence ends with the sentence. Without this,
# a URL the tools really did return would be read as a different, unknown one
# purely because a full stop followed it.

_SENTENCE_TAIL = ".,;:!?'\""


def _trim(match):
    """The reference itself, with the punctuation that merely followed it."""
    value = match.group(0).rstrip(_SENTENCE_TAIL)

    return value, (match.start(), match.start() + len(value))


def findings(answer, ledger):
    """Every provenance value the answer states that the tools did not."""
    text = answer or ""
    found = []

    for pattern, kind, known in ((_IDENT, FABRICATED_SOURCE_ID,
                                  ledger.source_ids),
                                 (_URL, FABRICATED_URL, ledger.urls),
                                 (_UNIX_PATH, FABRICATED_PATH, ledger.paths),
                                 (_WIN_PATH, FABRICATED_PATH, ledger.paths)):
        for match in pattern.finditer(text):
            value, span = _trim(match)

            if not value or value in known:
                continue

            found.append({"kind": kind, "value": value, "span": span})

    return sorted(found, key=lambda item: item["span"])


def _tidy(text):
    """Close up the punctuation a removed reference left behind."""
    text = _EMPTY_BRACKETS.sub("", text)
    text = _LOOSE_PUNCT.sub(r"\1", text)
    text = _TRAILING.sub("", text)

    return _BLANK_RUN.sub("\n\n", text)


def sanitize(answer, ledger):
    """The answer with its ungrounded references removed and nothing else.

    A markdown link to a fabricated target keeps its visible text, which is
    where the real citation usually is; a bare fabricated value is cut. No
    replacement reference is ever supplied, because choosing one would mean
    guessing which source backs which sentence.
    """
    text = answer or ""
    problems = findings(text, ledger)

    if not problems:
        return text, [], []

    removed = []

    def drop_link(match):
        target = (match.group(2) or "").rstrip(_SENTENCE_TAIL)

        if not target or ledger.knows(target) or not (
                _URL.fullmatch(target) or _UNIX_PATH.fullmatch(target)
                or _WIN_PATH.fullmatch(target) or _IDENT.fullmatch(target)):
            return match.group(0)

        removed.append(target)

        return match.group(1)

    cleaned = _LINK.sub(drop_link, text)

    for problem in problems:
        if problem["value"] in cleaned:
            cleaned = cleaned.replace(problem["value"], "")
            removed.append(problem["value"])

    cleaned = _tidy(cleaned)
    retained = sorted({value for value in _IDENT.findall(cleaned)
                       if value in ledger.source_ids})

    return cleaned, sorted(set(removed)), retained


def guard(answer, ledger):
    """Return the answer, or the same answer without its invented provenance."""
    problems = findings(answer, ledger)

    if not problems:
        return answer, [], [], False

    cleaned, removed, retained = sanitize(answer, ledger)

    return cleaned, problems, removed, True
