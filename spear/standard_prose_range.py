"""Bit positions a standard states in a rule instead of in its diagram.

A packet diagram usually writes a field's range beside its name, and STD2C made
that the authority. Sometimes it does not: the diagram prints the field's name
and leaves its bits to a numbered rule in the surrounding text. Page 148 of the
first real review is exactly that -- a named subfield in the diagram, its bits
given only by a Rule, and a four-bit hole in the word where it belongs.

Reading bit ranges out of prose in general is a bad idea, and STD2C refuses it
for good reasons: sentences describe radix points, value scales, figure numbers
and counts, and every one of those carries number pairs that look like bit
ranges. That refusal stands. This module does something narrower -- it does not
parse prose looking for ranges, it *links* a range the document already states
in a normative rule to a field the diagram already shows, and only when the rule
names that field, names its word, and leaves no room for a second reading.

The bar is deliberately high enough that most of the corpus fails it. That is
the point: a link is evidence about one named field, not a licence to read bits
out of sentences.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Iterable, Mapping, Sequence

from standard_schema import sha256_json
from standard_stated_range import (
    DEFAULT_WORD_WIDTH, RangeContext, RangeStatus, parse_stated_ranges,
)


PROSE_RANGE_MODEL_VERSION = "normative-prose-ranges-v1"

# The modality a statement must carry before its numbers may position a field.
# A recommendation or an example describes; only a requirement requires.

NORMATIVE_MODALITIES = frozenset({"REQUIREMENT"})

# A numbered statement opens a sentence we can scope a range to. Splitting here
# keeps one rule's range from being read against another rule's field name.

_STATEMENT = re.compile(
    r"(?=\b(?:Rule|Requirement|Recommendation|Permission|Observation|Suggestion)"
    r"\s+\d+(?:\.\d+)*-\d+\s*:)")

# The document quotes the subfield it is talking about. That quoting is what
# separates "the X subfield ... bits 31-28" from a sentence that merely mentions
# a bit, and every rejected sentence in the corpus lacks it.

_QUOTED_NAME = re.compile(r"[“\"]([^”\"]{1,48})[”\"]")

# Which word of the structure the rule is talking about, in the document's own
# ordinal. Used only to corroborate the word the diagram already assigned.

_ORDINAL_WORD = re.compile(
    r"\b(?:(1st|first)|(2nd|second)|(3rd|third)|(4th|fourth)|(5th|fifth)"
    r"|(6th|sixth)|(7th|seventh)|(8th|eighth))\s+(?:\w+\s+){0,2}?word\b", re.I)

# A name this short is a letter on a diagram, not a subfield to match on.

_MIN_NAME_CHARS = 3
_FIELD_NOUN = re.compile(r"\b(sub)?field\b", re.I)
_PUNCTUATION = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")

NOT_NORMATIVE = "NOT_NORMATIVE"
NO_CANONICAL_PROVENANCE = "NO_CANONICAL_PROVENANCE"
NO_QUOTED_FIELD_NAME = "NO_QUOTED_FIELD_NAME"
MULTIPLE_RANGES_IN_STATEMENT = "MULTIPLE_RANGES_IN_STATEMENT"
MULTIPLE_NAMES_IN_STATEMENT = "MULTIPLE_NAMES_IN_STATEMENT"
NAME_TOO_SHORT = "NAME_TOO_SHORT"
NAME_MATCHES_NO_FIELD = "NAME_MATCHES_NO_FIELD"
NAME_AMBIGUOUS = "NAME_AMBIGUOUS"
FIELD_HAS_LOCAL_RANGE = "FIELD_HAS_LOCAL_RANGE"
COMPETING_NORMATIVE_RANGES = "COMPETING_NORMATIVE_RANGES"
SECTION_MISMATCH = "SECTION_MISMATCH"
PROSE_RANGE_WORD_UNRESOLVED = "PROSE_RANGE_WORD_UNRESOLVED"
RANGE_EXCEEDS_WORD_WIDTH = "RANGE_EXCEEDS_WORD_WIDTH"


class MatchMethod(StrEnum):
    EXACT_NORMALIZED_NAME = "EXACT_NORMALIZED_NAME"


class LinkStatus(StrEnum):
    LINKED = "LINKED"
    REJECTED = "REJECTED"


def normalize_field_name(text: str) -> str:
    """A field name reduced to what two spellings of it must share.

    Case, spacing and punctuation only, plus the noun "field"/"subfield" which
    the prose adds and the diagram usually does not. Nothing here guesses at
    meaning: two names match or they do not.
    """
    value = _SPACES.sub(" ", text.strip().lower())
    value = _FIELD_NOUN.sub(" ", value)

    return _SPACES.sub(" ", _PUNCTUATION.sub(" ", value)).strip()


def normative_statements(text: str) -> tuple[tuple[int, str], ...]:
    """One numbered statement per entry, so a range cannot cross rules."""
    flat = _SPACES.sub(" ", text.strip())
    parts = [part.strip() for part in _STATEMENT.split(flat)]

    return tuple((index, part) for index, part in enumerate(parts) if part)


def ordinal_word(statement: str, diagram_words: Sequence[int]) -> int | None:
    """The word the statement names, in the diagram's own numbering.

    "the second header word" is the second word the diagram lists, whatever it
    happens to call it -- so a diagram numbering from zero is read correctly.
    """
    match = _ORDINAL_WORD.search(statement)

    if match is None:
        return None

    position = next(index for index, group in enumerate(match.groups())
                    if group is not None)
    ordered = sorted(set(int(value) for value in diagram_words))

    return ordered[position] if position < len(ordered) else None


@dataclass(frozen=True)
class StandardNormativeRangeLink:
    """One rule's range, tied to one field the diagram already shows."""

    candidate_id: str
    target_cell_id: str
    target_label: str
    normalized_name: str
    quoted_name: str
    requirement_source_id: str
    statement_index: int
    range_text: str
    high_bit: int
    low_bit: int
    width: int
    word_index: int | None
    ordinal_word_index: int | None
    match_method: MatchMethod
    status: LinkStatus
    warnings: tuple[str, ...] = ()
    link_fingerprint: str = ""

    @property
    def usable(self) -> bool:
        return self.status is LinkStatus.LINKED and not self.warnings

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["match_method"] = self.match_method.value
        value["status"] = self.status.value
        value["warnings"] = list(self.warnings)

        return value


def _fingerprint(link: StandardNormativeRangeLink) -> str:
    return "lnk-" + sha256_json({
        "model": PROSE_RANGE_MODEL_VERSION,
        "candidate": link.candidate_id, "cell": link.target_cell_id,
        "name": link.normalized_name, "source": link.requirement_source_id,
        "statement": link.statement_index,
        "bits": [link.high_bit, link.low_bit], "word": link.word_index,
    })[:16]


@dataclass(frozen=True)
class ProseFieldTarget:
    """A field the diagram shows but does not position."""

    cell_id: str
    label: str
    word_index: int | None
    source_ids: tuple[str, ...]
    provenance: str


def normative_range_links(
    targets: Sequence[ProseFieldTarget], units: Iterable, *,
    candidate_id: str, sections: frozenset[str], diagram_words: Sequence[int],
    word_width: int = DEFAULT_WORD_WIDTH,
) -> tuple[tuple[StandardNormativeRangeLink, ...],
           tuple[StandardNormativeRangeLink, ...]]:
    """Every link the evidence supports, and every one it refuses, with reasons.

    A statement qualifies only when it is a requirement, cites canonical text,
    carries exactly one range and exactly one quoted name, names a word, and
    that name picks out exactly one unpositioned field of this diagram in the
    same section. Anything less is returned as a refusal, never as a link.
    """
    by_name: dict[str, list[ProseFieldTarget]] = {}

    for target in targets:
        by_name.setdefault(normalize_field_name(target.label), []).append(target)

    context = RangeContext(in_bitfield_region=True, word_width=word_width)

    linked: list[StandardNormativeRangeLink] = []
    refused: list[StandardNormativeRangeLink] = []
    proposals: dict[str, list[StandardNormativeRangeLink]] = {}

    for unit in units:
        modality = getattr(unit.content_type, "value", str(unit.content_type))
        section = getattr(unit, "section", None)

        for index, statement in normative_statements(unit.text):
            ranges = [item for item in parse_stated_ranges(statement, context=context)
                      if item.high_bit < word_width]

            if not ranges:
                continue

            names = _QUOTED_NAME.findall(statement)
            reasons: list[str] = []

            if modality not in NORMATIVE_MODALITIES:
                reasons.append(NOT_NORMATIVE)

            if not getattr(unit, "source_id", ""):
                reasons.append(NO_CANONICAL_PROVENANCE)

            if len(ranges) > 1:
                reasons.append(MULTIPLE_RANGES_IN_STATEMENT)

            if not names:
                reasons.append(NO_QUOTED_FIELD_NAME)
            elif len(names) > 1:
                reasons.append(MULTIPLE_NAMES_IN_STATEMENT)

            if sections and section not in sections:
                reasons.append(SECTION_MISMATCH)

            name = normalize_field_name(names[0]) if names else ""

            if names and len(name) < _MIN_NAME_CHARS:
                reasons.append(NAME_TOO_SHORT)

            matches = by_name.get(name, []) if name else []

            if name and not matches:
                reasons.append(NAME_MATCHES_NO_FIELD)
            elif len(matches) > 1:
                reasons.append(NAME_AMBIGUOUS)

            ordinal = ordinal_word(statement, diagram_words)
            target = matches[0] if len(matches) == 1 else None

            if ordinal is None:
                reasons.append(PROSE_RANGE_WORD_UNRESOLVED)
            elif target is not None and target.word_index != ordinal:
                reasons.append(PROSE_RANGE_WORD_UNRESOLVED)

            first = ranges[0]

            if first.high_bit >= word_width:
                reasons.append(RANGE_EXCEEDS_WORD_WIDTH)

            link = StandardNormativeRangeLink(
                candidate_id=candidate_id,
                target_cell_id=target.cell_id if target else "",
                target_label=target.label if target else "",
                normalized_name=name, quoted_name=names[0] if names else "",
                requirement_source_id=getattr(unit, "source_id", ""),
                statement_index=index, range_text=first.range_text,
                high_bit=first.high_bit, low_bit=first.low_bit, width=first.width,
                word_index=target.word_index if target else None,
                ordinal_word_index=ordinal,
                match_method=MatchMethod.EXACT_NORMALIZED_NAME,
                status=LinkStatus.REJECTED if reasons else LinkStatus.LINKED,
                warnings=tuple(dict.fromkeys(reasons)))
            link = type(link)(**{**asdict(link),
                                 "match_method": link.match_method,
                                 "status": link.status,
                                 "link_fingerprint": _fingerprint(link)})
            (refused if reasons else linked).append(link)

            if not reasons:
                proposals.setdefault(link.target_cell_id, []).append(link)

    # A field two rules position differently is a field this cannot settle.

    settled: list[StandardNormativeRangeLink] = []

    for cell_id, group in proposals.items():
        if len({(item.high_bit, item.low_bit) for item in group}) == 1:
            settled.append(group[0])
            continue

        for item in group:
            refused.append(type(item)(**{
                **asdict(item), "match_method": item.match_method,
                "status": LinkStatus.REJECTED,
                "warnings": (COMPETING_NORMATIVE_RANGES,),
                "link_fingerprint": item.link_fingerprint}))

    keep = {item.link_fingerprint for item in settled}
    refused.extend(item for item in linked if item.link_fingerprint not in keep)

    return tuple(sorted(settled, key=lambda item: item.target_cell_id)), tuple(refused)


# --------------------------------------------------------------------------
# what a reviewer decided about a link
# --------------------------------------------------------------------------

_FINGERPRINT = re.compile(r"^lnk-[0-9a-f]{16}$")


class ProseRangeError(RuntimeError):
    pass


@dataclass(frozen=True)
class StandardReviewedLink:
    """One person's decision about one link, and the facts they decided on.

    A link is evidence from outside the diagram, so an approval that permits it
    has to say which link, positioning which field, from which requirement, at
    which bits, in which word. Binding the fingerprint alone would let the facts
    behind it move while the approval still looked current.
    """

    link_fingerprint: str
    target_cell_id: str
    normative_source_id: str
    high_bit: int
    low_bit: int
    word_index: int | None
    accepted: bool
    role: str = "FIELD"

    def __post_init__(self) -> None:
        if not _FINGERPRINT.fullmatch(self.link_fingerprint):
            raise ProseRangeError("malformed normative link fingerprint")

        if not self.target_cell_id or not self.normative_source_id:
            raise ProseRangeError("a reviewed link must name its field and source")

        if self.role not in ("FIELD", "RESERVED"):
            raise ProseRangeError(f"a linked field cannot be {self.role!r}")

        if self.high_bit < self.low_bit:
            raise ProseRangeError("reviewed link range is inverted")

    @property
    def bound(self) -> tuple[object, ...]:
        """Exactly the facts the reviewer saw, in a form that can be compared."""
        return (self.link_fingerprint, self.target_cell_id,
                self.normative_source_id, self.high_bit, self.low_bit,
                self.word_index)

    @classmethod
    def of(cls, link: StandardNormativeRangeLink, *, accepted: bool,
           role: str = "FIELD") -> "StandardReviewedLink":
        return cls(link_fingerprint=link.link_fingerprint,
                   target_cell_id=link.target_cell_id,
                   normative_source_id=link.requirement_source_id,
                   high_bit=link.high_bit, low_bit=link.low_bit,
                   word_index=link.word_index, accepted=accepted, role=role)

    def matches(self, link: StandardNormativeRangeLink) -> bool:
        return self.bound == StandardReviewedLink.of(
            link, accepted=self.accepted, role=self.role).bound

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "StandardReviewedLink":
        try:
            return cls(
                link_fingerprint=str(raw["link_fingerprint"]),
                target_cell_id=str(raw["target_cell_id"]),
                normative_source_id=str(raw["normative_source_id"]),
                high_bit=int(raw["high_bit"]), low_bit=int(raw["low_bit"]),
                word_index=(int(raw["word_index"])
                            if raw.get("word_index") is not None else None),
                accepted=bool(raw["accepted"]),
                role=str(raw.get("role", "FIELD")))
        except (KeyError, TypeError, ValueError) as exc:
            raise ProseRangeError(f"malformed reviewed link: {exc}") from exc
