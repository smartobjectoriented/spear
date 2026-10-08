"""Bit positions the document states in its own words.

A packet diagram is drawn, and a drawing can be measured. STD2A measured them,
and STD2B-R showed the measurement does not survive contact with a real
standard: a field label is centred inside its field, so its text box is inset
from the bits it names by however much whitespace the typesetter left. The
measurement is an association, not a boundary.

Where the document states a range in the label itself the guessing stops, so
that is the authority here: an explicit stated range outranks any geometry, and
geometry is kept only to say which row and which table a label belongs to. When
the two disagree the disagreement is recorded and the text wins; nothing is
averaged, intersected or reconciled.

Nothing in this module reads meaning out of prose. It recognises a small set of
range syntaxes, demands that they occur somewhere bit positions could plausibly
be stated, and refuses everything else.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Sequence


STATED_RANGE_MODEL_VERSION = "stated-ranges-v1"

# Beyond this a number is not a bit position in any document we are prepared to
# guess about. Wide enough for the 64-bit values a 32-bit word carries in two
# halves; narrow enough that a year or a page number is never mistaken for one.

MAX_BIT = 63

# The width of one row of a packet diagram. A range that will not fit inside it
# is stated against something else -- a wider value, a whole record -- and is
# not a word-local bit position, whatever it looks like.

DEFAULT_WORD_WIDTH = 32

# A shorter run of consecutive labels is a fragment of a longer ruler far more
# often than it is a ruler, and a fragment cannot say what lies outside it.

_MIN_TRUSTED_RULER = 8


class SyntaxKind(StrEnum):
    PAREN_DASH = "PAREN_DASH"
    PAREN_DOTDOT = "PAREN_DOTDOT"
    BRACKET_COLON = "BRACKET_COLON"
    BRACKET_DASH = "BRACKET_DASH"
    ANGLE_COLON = "ANGLE_COLON"
    BITS_DASH = "BITS_DASH"
    BITS_THROUGH = "BITS_THROUGH"
    BITS_SINGLE = "BITS_SINGLE"


class RangeStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class RangeDomain(StrEnum):
    WORD_LOCAL = "WORD_LOCAL"
    EXCEEDS_WORD_WIDTH = "EXCEEDS_WORD_WIDTH"


class RulerAgreement(StrEnum):
    INSIDE_RULER = "INSIDE_RULER"
    OUTSIDE_RULER = "OUTSIDE_RULER"
    NO_RULER = "NO_RULER"
    RULER_UNTRUSTED = "RULER_UNTRUSTED"


class GeometryAgreement(StrEnum):
    AGREES = "AGREES"
    PARTIAL_OVERLAP = "PARTIAL_OVERLAP"
    CONTRADICTS = "CONTRADICTS"
    GEOMETRY_NOT_MEANINGFUL = "GEOMETRY_NOT_MEANINGFUL"


class StatedOrder(StrEnum):
    HIGH_FIRST = "HIGH_FIRST"
    LOW_FIRST = "LOW_FIRST"
    SINGLE = "SINGLE"


# Warning vocabulary, so a reviewer sees the same word every time.

NO_BITFIELD_CONTEXT = "NO_BITFIELD_CONTEXT"
DISQUALIFYING_CONTEXT = "DISQUALIFYING_CONTEXT"
UNIT_FOLLOWS_RANGE = "UNIT_FOLLOWS_RANGE"
COUNT_DESCRIPTOR = "COUNT_DESCRIPTOR"
AMBIGUOUS_ASCENDING_RANGE = "AMBIGUOUS_ASCENDING_RANGE"
RANGE_OUT_OF_BOUNDS = "RANGE_OUT_OF_BOUNDS"
RANGE_EXCEEDS_WORD_WIDTH = "RANGE_EXCEEDS_WORD_WIDTH"
RANGE_OUTSIDE_RULER = "RANGE_OUTSIDE_RULER"
GEOMETRY_RANGE_CONFLICT = "GEOMETRY_RANGE_CONFLICT"
PROSE_CONTEXT = "PROSE_CONTEXT"


# --------------------------------------------------------------------------
# syntax
# --------------------------------------------------------------------------

_SEPARATOR = r"(?:\.\.|…|[-‐‑‒–—:])"

# Ordered: a form that names "bit" carries its own evidence, so it is tried
# first and is recognised even inside another form's brackets.

_SYNTAX: tuple[tuple[SyntaxKind, re.Pattern[str], bool], ...] = (
    (SyntaxKind.BITS_THROUGH,
     re.compile(r"\bbits?\s+(\d{1,2})\s+(?:through|thru|to)\s+(\d{1,2})\b", re.I), True),
    (SyntaxKind.BITS_DASH,
     re.compile(rf"\bbits?\s+(\d{{1,2}})\s*{_SEPARATOR}\s*(\d{{1,2}})\b", re.I), True),
    (SyntaxKind.BITS_SINGLE,
     re.compile(rf"\bbit\s+(\d{{1,2}})\b(?!\s*(?:{_SEPARATOR}|through|thru|to)\s*\d)",
                re.I), True),
    (SyntaxKind.BRACKET_COLON,
     re.compile(r"\[\s*(\d{1,2})\s*:\s*(\d{1,2})\s*\]"), False),
    (SyntaxKind.BRACKET_DASH,
     re.compile(rf"\[\s*(\d{{1,2}})\s*{_SEPARATOR}\s*(\d{{1,2}})\s*\]"), False),
    (SyntaxKind.ANGLE_COLON,
     re.compile(r"<\s*(\d{1,2})\s*:\s*(\d{1,2})\s*>"), False),
    (SyntaxKind.PAREN_DOTDOT,
     re.compile(r"\(\s*(\d{1,2})\s*(?:\.\.|…)\s*(\d{1,2})\s*\)"), False),
    (SyntaxKind.PAREN_DASH,
     re.compile(r"\(\s*(\d{1,2})\s*[-‐‑‒–—]\s*(\d{1,2})\s*\)"),
     False),
)

# A form that names bits by name needs no help from its surroundings.

_SELF_EVIDENCING = frozenset(
    kind for kind, _, keyword in _SYNTAX if keyword)

# A bare pair of numbers in brackets is a bit range only when the surroundings
# say bits are being discussed.

_STRONG_SYNTAX = frozenset(
    (SyntaxKind.BRACKET_COLON, SyntaxKind.BRACKET_DASH, SyntaxKind.ANGLE_COLON))

# Nouns that name a numbered thing in a document rather than a field.

_REFERENCE_NOUN = (
    r"version|clause|subclause|section|subsection|figure|fig|"
    r"table|annex|appendix|page|pages|rule|note|observation|recommendation|"
    r"requirement|permission|suggestion|part|chapter|paragraph|revision|rev|"
    r"date|dated|issue|edition|equation|eq|step|example|item|no|nr|num")

# "Figure 3-1", "clause 9.3", "Fig. 2": the noun names something the document
# numbers, and the number is what makes it a reference rather than a name.

_DOCUMENT_REFERENCE = re.compile(
    r"(?:^|[^\w])(?:" + _REFERENCE_NOUN + r")\.?\s*\d+(?:[.\-]\d+)*\b", re.I)

# The noun alone in front of the range, with nothing naming anything around it.

_BARE_REFERENCE = re.compile(r"^\s*(?:" + _REFERENCE_NOUN + r")[\s.]*$", re.I)


def is_document_reference(before: str) -> bool:
    """Whether the text before a range points at a numbered part of the document.

    A noun like "figure" disqualifies a range when it is referring -- when the
    document numbers the thing it names -- not merely because it turns up in a
    field's name. "Noise Figure" and "Figure of Merit" are names; "Figure 3-1"
    and "Fig. 2" are references, and so is a bare "Figure" with a range hung
    straight off it.
    """
    return bool(_DOCUMENT_REFERENCE.search(before)
                or _BARE_REFERENCE.match(before))

# A unit immediately after the numbers makes them a measurement, not a position.

_UNIT_SUFFIX = re.compile(
    r"^\s*(?:[kKMGTmunµp]?Hz|dB[mci]?|[munµ]?s(?:ec(?:onds?)?)?\b|"
    r"[mck]?m\b|[munµ]?V\b|[munµ]?A\b|[mkMG]?W\b|deg(?:rees?)?\b|"
    r"rad(?:ians?)?\b|[kMG]?bps\b|samples?\b|%)")

# "(2 Words, Optional)" counts words. It is not a position and never becomes one.

_COUNT_DESCRIPTOR = re.compile(
    r"\(\s*\d+\s+(?:word|byte|octet|bit|nibble|sample|record|entry|entries)s?\b", re.I)

# Enough to say the surroundings are talking about bits at all.

_BIT_TERMINOLOGY = re.compile(r"\bbits?\b|\bmsb\b|\blsb\b|\bbitmapped\b", re.I)

# A field label names a field. A sentence that happens to mention a bit is
# explaining something, and the bit it mentions is not that cell's position.

_SENTENCE = re.compile(r"\.\s|\.$")
_MAX_LABEL_CHARS = 90


def reads_as_prose(text: str) -> bool:
    """Whether a cell reads as a sentence rather than as a field label."""
    return bool(_SENTENCE.search(text.strip())) or len(text.strip()) > _MAX_LABEL_CHARS


def mentions_bits(text: str) -> bool:
    """Whether a piece of text talks about bits at all."""
    return bool(_BIT_TERMINOLOGY.search(text))


def is_count_descriptor(text: str) -> bool:
    """A word, byte or octet count -- never a bit range (STD2B-R found many)."""
    return bool(_COUNT_DESCRIPTOR.search(text))


def ruler_is_trusted(labels: Sequence[int], *,
                     word_width: int = DEFAULT_WORD_WIDTH) -> bool:
    """Whether a run of visible labels is a whole ruler rather than a piece of one.

    A whole ruler runs from zero to one below a power-of-two width and has every
    position in between. Anything else is a fragment, and a fragment cannot be
    used to decide that a stated range lies outside the diagram.
    """
    values = sorted(set(int(value) for value in labels))

    if len(values) < _MIN_TRUSTED_RULER or len(values) != len(tuple(labels)):
        return False

    if values[0] != 0 or values != list(range(len(values))):
        return False

    width = values[-1] + 1

    return width & (width - 1) == 0 and width <= max(word_width, MAX_BIT + 1)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RangeContext:
    """What the surroundings say about whether bit positions are being stated."""

    in_bitfield_region: bool = False
    ruler_labels: tuple[int, ...] = ()
    bit_terminology: bool = False
    word_width: int = DEFAULT_WORD_WIDTH

    @property
    def bitfield_like(self) -> bool:
        return bool(self.in_bitfield_region or self.ruler_labels
                    or self.bit_terminology)


@dataclass(frozen=True)
class ParsedBitRange:
    """One range token, normalized, with the reason it was kept or refused."""

    range_text: str
    high_bit: int
    low_bit: int
    width: int
    syntax_kind: SyntaxKind
    stated_order: StatedOrder
    domain: RangeDomain
    status: RangeStatus
    start: int
    end: int
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["syntax_kind"] = self.syntax_kind.value
        value["stated_order"] = self.stated_order.value
        value["domain"] = self.domain.value
        value["status"] = self.status.value
        value["warnings"] = list(self.warnings)

        return value


def _matches(text: str) -> list[tuple[SyntaxKind, re.Match[str]]]:
    """Every range token in reading order, each claimed by one syntax only."""
    found: list[tuple[SyntaxKind, re.Match[str]]] = []
    taken: list[tuple[int, int]] = []

    for kind, pattern, _ in _SYNTAX:
        for match in pattern.finditer(text):
            if any(match.start() < end and start < match.end()
                   for start, end in taken):
                continue

            taken.append((match.start(), match.end()))
            found.append((kind, match))

    return sorted(found, key=lambda item: item[1].start())


def parse_stated_ranges(text: str, *, context: RangeContext | None = None,
                        ) -> tuple[ParsedBitRange, ...]:
    """Every range token in `text`, accepted or refused, in reading order.

    Refused tokens are returned too: a reviewer needs to see that something
    range-shaped was found and why it was not believed.
    """
    context = context or RangeContext()

    if context.bit_terminology is False and mentions_bits(text):
        context = RangeContext(context.in_bitfield_region, context.ruler_labels,
                               True, context.word_width)

    results: list[ParsedBitRange] = []

    for kind, match in _matches(text):
        numbers = [int(value) for value in match.groups() if value is not None]
        high, low = max(numbers), min(numbers)
        order = (StatedOrder.SINGLE if len(numbers) == 1
                 else StatedOrder.HIGH_FIRST if numbers[0] > numbers[1]
                 else StatedOrder.LOW_FIRST)
        domain = (RangeDomain.WORD_LOCAL if high < context.word_width
                  else RangeDomain.EXCEEDS_WORD_WIDTH)
        warnings: list[str] = []

        if is_document_reference(text[:match.start()]):
            warnings.append(DISQUALIFYING_CONTEXT)

        if _UNIT_SUFFIX.match(text[match.end():]):
            warnings.append(UNIT_FOLLOWS_RANGE)

        if is_count_descriptor(match.group(0)):
            warnings.append(COUNT_DESCRIPTOR)

        if reads_as_prose(text):
            warnings.append(PROSE_CONTEXT)

        if high > MAX_BIT:
            warnings.append(RANGE_OUT_OF_BOUNDS)

        if kind not in _SELF_EVIDENCING:
            if not context.bitfield_like:
                warnings.append(NO_BITFIELD_CONTEXT)

            # Written low first with nothing naming bits, a pair in brackets is
            # as likely to be a scale or a count as a bit range. Refuse it.

            elif order is StatedOrder.LOW_FIRST and kind not in _STRONG_SYNTAX:
                warnings.append(AMBIGUOUS_ASCENDING_RANGE)

        if domain is RangeDomain.EXCEEDS_WORD_WIDTH:
            warnings.append(RANGE_EXCEEDS_WORD_WIDTH)

        blocking = [value for value in warnings if value != RANGE_EXCEEDS_WORD_WIDTH]
        results.append(ParsedBitRange(
            range_text=match.group(0), high_bit=high, low_bit=low,
            width=high - low + 1, syntax_kind=kind, stated_order=order,
            domain=domain,
            status=RangeStatus.REJECTED if blocking else RangeStatus.ACCEPTED,
            start=match.start(), end=match.end(), warnings=tuple(warnings)))

    return tuple(results)


def accepted_ranges(text: str, *, context: RangeContext | None = None,
                    ) -> tuple[ParsedBitRange, ...]:
    return tuple(item for item in parse_stated_ranges(text, context=context)
                 if item.status is RangeStatus.ACCEPTED)


def display_label(text: str, parsed: ParsedBitRange) -> str:
    """The label without its range token. The normative label is never changed."""
    stripped = re.sub(r"\s+", " ",
                      text[:parsed.start] + " " + text[parsed.end:]).strip()
    stripped = re.sub(r"\s+([,;:.])", r"\1", stripped)

    return re.sub(r"[\s,;:\-–]+$", "", stripped).strip() or text.strip()


# --------------------------------------------------------------------------
# agreement
# --------------------------------------------------------------------------

def agreement_with_ruler(parsed: ParsedBitRange, labels: Sequence[int], *,
                         word_width: int = DEFAULT_WORD_WIDTH) -> RulerAgreement:
    """Whether a visible ruler vouches for the stated range, if one can."""

    if not labels:
        return RulerAgreement.NO_RULER

    if not ruler_is_trusted(labels, word_width=word_width):
        return RulerAgreement.RULER_UNTRUSTED

    domain = set(int(value) for value in labels)

    return (RulerAgreement.INSIDE_RULER
            if set(range(parsed.low_bit, parsed.high_bit + 1)) <= domain
            else RulerAgreement.OUTSIDE_RULER)


def agreement_with_geometry(parsed: ParsedBitRange,
                            covered_labels: Sequence[int], *,
                            ruler_trusted: bool) -> GeometryAgreement:
    """How the measured span compares -- for the audit trail, not for authority."""

    if not covered_labels or not ruler_trusted:
        return GeometryAgreement.GEOMETRY_NOT_MEANINGFUL

    measured = set(int(value) for value in covered_labels)
    stated = set(range(parsed.low_bit, parsed.high_bit + 1))

    if measured == stated:
        return GeometryAgreement.AGREES

    return (GeometryAgreement.PARTIAL_OVERLAP if measured & stated
            else GeometryAgreement.CONTRADICTS)


# --------------------------------------------------------------------------
# provenance-bound result
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StandardStatedBitRange:
    """A stated range tied to the canonical cell and sources it was read from."""

    raw_source_cell_id: str
    supporting_source_ids: tuple[str, ...]
    provenance_grade: str
    normative_label: str
    display_label: str
    range_text: str
    high_bit: int
    low_bit: int
    width: int
    syntax_kind: SyntaxKind
    stated_order: StatedOrder
    domain: RangeDomain
    ruler_agreement: RulerAgreement
    geometry_agreement: GeometryAgreement
    geometry_conflict: bool
    status: RangeStatus
    warnings: tuple[str, ...] = ()

    @property
    def positional(self) -> bool:
        """Whether the range itself can state a word-local position."""
        return (self.status is RangeStatus.ACCEPTED
                and self.domain is RangeDomain.WORD_LOCAL
                and self.ruler_agreement is not RulerAgreement.OUTSIDE_RULER)

    @property
    def promotable(self) -> bool:
        """Positional, and cited. Promotion needs both; showing a reviewer does not."""
        return self.positional and bool(self.supporting_source_ids)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)

        for name in ("syntax_kind", "stated_order", "domain", "ruler_agreement",
                     "geometry_agreement", "status"):
            value[name] = getattr(self, name).value

        value["supporting_source_ids"] = list(self.supporting_source_ids)
        value["warnings"] = list(self.warnings)

        return value


def stated_ranges_for_cell(text: str, *, cell_id: str,
                           source_ids: Sequence[str], provenance: str,
                           context: RangeContext | None = None,
                           covered_labels: Sequence[int] = (),
                           ) -> tuple[StandardStatedBitRange, ...]:
    """Bind every range token in one cell to the cell's canonical provenance."""
    context = context or RangeContext()
    trusted = ruler_is_trusted(context.ruler_labels, word_width=context.word_width)
    bound: list[StandardStatedBitRange] = []

    for parsed in parse_stated_ranges(text, context=context):
        ruler = agreement_with_ruler(parsed, context.ruler_labels,
                                     word_width=context.word_width)
        geometry = agreement_with_geometry(parsed, covered_labels,
                                           ruler_trusted=trusted)
        warnings = list(parsed.warnings)

        if ruler is RulerAgreement.OUTSIDE_RULER:
            warnings.append(RANGE_OUTSIDE_RULER)

        conflict = geometry in (GeometryAgreement.PARTIAL_OVERLAP,
                                GeometryAgreement.CONTRADICTS)

        if conflict:
            warnings.append(GEOMETRY_RANGE_CONFLICT)

        status = (RangeStatus.REJECTED
                  if parsed.status is RangeStatus.REJECTED
                  or ruler is RulerAgreement.OUTSIDE_RULER
                  else RangeStatus.ACCEPTED)
        bound.append(StandardStatedBitRange(
            raw_source_cell_id=cell_id,
            supporting_source_ids=tuple(source_ids),
            provenance_grade=provenance,
            normative_label=text, display_label=display_label(text, parsed),
            range_text=parsed.range_text, high_bit=parsed.high_bit,
            low_bit=parsed.low_bit, width=parsed.width,
            syntax_kind=parsed.syntax_kind, stated_order=parsed.stated_order,
            domain=parsed.domain, ruler_agreement=ruler,
            geometry_agreement=geometry, geometry_conflict=conflict,
            status=status, warnings=tuple(dict.fromkeys(warnings))))

    return tuple(bound)
