"""Normative facts a fetched source unit states outright.

FT3V's ledger learns only from standard.get_structure, so a session whose
decisive evidence arrives through standard.fetch reaches the final-answer guard
with nothing to check against. Two supported controls fail there: one is told a
twelve-octet record and two of its three offsets and then places the third at
octet 12, past the end of the record it just quoted.

Reading facts out of prose in general is a bad idea, and standard_prose_range
refuses it for good reasons -- sentences carry counts, scales and figure numbers
that all look like positions. That refusal stands. What is done here is
narrower: a numbered *requirement* is split out by the repository's own
statement splitter, and then a closed set of clause patterns is matched against
it. Each pattern is anchored on the normative verb ("shall begin at octet
offset", "shall be carried in word") and captures one determiner-led subject and
one number. A statement that does not fit a pattern yields nothing; there is no
fallback reading, and nothing here interprets.

Every fact carries the source unit, section, page and the statement it came out
of, because a fact the guard cannot attribute is not evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from standard_prose_range import normative_statements
from standard_stated_range import RangeContext, RangeStatus, parse_stated_ranges

# Only a requirement requires. A recommendation or an observation describes,
# and describing a layout does not establish one.

NORMATIVE_MODALITIES = frozenset({"Rule", "Requirement"})

_STATEMENT_HEAD = re.compile(
    r"^(Rule|Requirement|Recommendation|Permission|Observation|Suggestion)\s+"
    r"(\d+(?:\.\d+)*-\d+)\s*:\s*(.*)$", re.S)

# The document writes its counts as words as often as as digits. This is the
# whole vocabulary accepted; anything outside it is not a number here.

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "twenty": 20, "twenty-four": 24, "thirty-two": 32, "sixty-four": 64,
}
_NUMBER = r"(?:\d{1,4}|" + "|".join(sorted(_NUMBER_WORDS, key=len,
                                           reverse=True)) + r")"

# A subject is a determiner-led name of at most five words, each of which starts
# a word the document capitalised. That is what separates "The Primary Value
# shall begin" from a sentence whose subject is a clause.

_SUBJECT = r"[A-Z][A-Za-z0-9]*(?:[ \-][A-Za-z0-9]+){0,4}"

FIELD_OFFSET = "FIELD_OFFSET"
FIELD_LENGTH = "FIELD_LENGTH"
FIELD_WORD_INDEX = "FIELD_WORD_INDEX"
FIELD_RANGE = "FIELD_RANGE"
RECORD_LENGTH = "RECORD_LENGTH"
RECORD_WORD_COUNT = "RECORD_WORD_COUNT"
MEMBERSHIP = "MEMBERSHIP"

# The closed set. Each entry is (fact class, pattern); nothing outside it is
# read, and a statement matching none of them contributes nothing.

_OFFSET = re.compile(
    rf"\bThe\s+({_SUBJECT})\s+shall\s+begin\s+at\s+octet\s+offset\s+"
    rf"({_NUMBER})\b", re.I)
_WORD_INDEX = re.compile(
    rf"\bThe\s+({_SUBJECT})\s+shall\s+be\s+carried\s+in\s+word\s+"
    rf"({_NUMBER})\b", re.I)
_BIT_RANGE = re.compile(
    rf"\bThe\s+({_SUBJECT})\s+shall\s+occupy\s+bits\s+(\d{{1,2}})\s*"
    rf"(?:\.\.|-|–|:|through|to)\s*(\d{{1,2}})\b", re.I)
_RECORD_OCTETS = re.compile(rf"\ba[n]?\s+({_NUMBER})-octet\s+record\b", re.I)
_RECORD_WORDS = re.compile(
    rf"\bshall\s+consist\s+of\s+({_NUMBER})\s+(\d{{1,3}})-bit\s+words\b", re.I)
_SUBJECT_HEAD = re.compile(rf"^The\s+({_SUBJECT})\s+shall\b", re.I)

# Members the statement enumerates, and the size it gives all of them at once.

_MEMBER_PAIR = re.compile(
    rf"\b(?:carrying|carry|carries)\s+the\s+({_SUBJECT})\s+and\s+the\s+"
    rf"({_SUBJECT})\b", re.I)
_EACH_OCTETS = re.compile(rf"\beach\s+({_NUMBER})\s+octets\b", re.I)

# A length given to a region the statement declines to place. The label is the
# document's own word for it, and it stays unpositioned.

_RESERVED_OCTETS = re.compile(rf"\b({_NUMBER})\s+octets\s+reserved\b", re.I)
_RESERVED_LABEL = "Reserved"


def number(text):
    """A count the document wrote either way, or None if it wrote neither."""
    value = (text or "").strip().lower()

    if value.isdigit():
        return int(value)

    return _NUMBER_WORDS.get(value)


def _key(label):
    return re.sub(r"[^a-z0-9]+", "", (label or "").lower())


@dataclass(frozen=True)
class NormativeFact:
    """One fact, and exactly where it was stated."""

    fact_class: str
    label: str
    source_id: str
    section: str
    page: object
    statement: str
    span: str
    offset: int | None = None
    length: int | None = None
    word_index: int | None = None
    msb: int | None = None
    lsb: int | None = None
    octets: int | None = None
    words: int | None = None
    word_width: int | None = None

    @property
    def key(self):
        return _key(self.label)

    @property
    def provenance(self):
        return {"source_id": self.source_id, "section": self.section,
                "page": self.page, "statement": self.statement,
                "span": self.span}


@dataclass
class _Origin:
    """Where the statement being read came from."""

    source_id: str = ""
    section: str = ""
    page: object = None


def _unit_of(payload):
    """The fetched unit, in either shape standard.fetch returns."""
    if not isinstance(payload, dict):
        return None

    unit = payload.get("unit")

    if isinstance(unit, dict) and unit.get("text"):
        return unit

    if payload.get("text") and payload.get("source_id"):
        return payload

    return None


def statement_facts(text, origin, *, statement_id="", modality=""):
    """Every fact one numbered requirement states, and nothing more."""
    if modality not in NORMATIVE_MODALITIES:
        return ()

    found = []

    def add(fact_class, label, span, **over):
        found.append(NormativeFact(
            fact_class=fact_class, label=label.strip(),
            source_id=origin.source_id, section=origin.section,
            page=origin.page, statement=statement_id, span=span, **over))

    for match in _OFFSET.finditer(text):
        value = number(match.group(2))

        if value is not None:
            add(FIELD_OFFSET, match.group(1), match.group(0), offset=value)

    for match in _WORD_INDEX.finditer(text):
        value = number(match.group(2))

        if value is not None:
            add(FIELD_WORD_INDEX, match.group(1), match.group(0),
                word_index=value)

    for match in _BIT_RANGE.finditer(text):
        high, low = int(match.group(2)), int(match.group(3))

        if high >= low:
            add(FIELD_RANGE, match.group(1), match.group(0), msb=high, lsb=low)

    subject = _SUBJECT_HEAD.search(text)
    container = subject.group(1) if subject else ""

    for match in _RECORD_OCTETS.finditer(text):
        value = number(match.group(1))

        if value is not None and container:
            add(RECORD_LENGTH, container, match.group(0), octets=value)

    for match in _RECORD_WORDS.finditer(text):
        count, width = number(match.group(1)), int(match.group(2))

        if count is not None and container:
            add(RECORD_WORD_COUNT, container, match.group(0), words=count,
                word_width=width)

    # "carrying the A and the B, each four octets" gives both members one size.
    # The size is read only when the statement enumerates who it applies to.

    members = _MEMBER_PAIR.search(text)
    each = _EACH_OCTETS.search(text)

    if members:
        for label in members.groups():
            add(MEMBERSHIP, label, members.group(0))

            if each is not None and number(each.group(1)) is not None:
                add(FIELD_LENGTH, label, each.group(0),
                    length=number(each.group(1)))

    for match in _RESERVED_OCTETS.finditer(text):
        value = number(match.group(1))

        if value is not None:
            add(FIELD_LENGTH, _RESERVED_LABEL, match.group(0), length=value)

    return tuple(found)


def unit_facts(payload):
    """Every normative fact a standard.fetch result states outright."""
    unit = _unit_of(payload)

    if unit is None:
        return ()

    origin = _Origin(source_id=str(unit.get("source_id") or ""),
                     section=str(unit.get("section") or ""),
                     page=unit.get("page"))
    found = []

    for _index, statement in normative_statements(str(unit.get("text") or "")):
        head = _STATEMENT_HEAD.match(statement)

        if head is None:
            continue

        modality, statement_id, body = head.groups()
        found.extend(statement_facts(body, origin, statement_id=statement_id,
                                     modality=modality.capitalize()))

    return tuple(found)


def diagram_ranges(payload):
    """Ranges a fetched diagram line prints beside a name, if it prints any.

    The repository's own stated-range parser decides what counts; this only
    hands it the text and keeps what it accepted. A diagram range is local to
    its word, so it establishes a width, never a physical position.
    """
    unit = _unit_of(payload)

    if unit is None:
        return ()

    text = str(unit.get("text") or "")
    context = RangeContext(in_bitfield_region=True)
    found = []

    for parsed in parse_stated_ranges(text, context=context):
        if parsed.status is RangeStatus.ACCEPTED:
            found.append(parsed)

    return tuple(found)
