"""A 64-bit value written across two words, and when that may be believed.

A bit range is local to its word. STD2D built the whole model on that, and it
is right for almost every label in the document. But some labels state bits in
a different coordinate system: a 64-bit quantity split over two 32-bit words
prints the high half as 63..32 and the low half as 31..0. The 63..32 is not a
position in any word; it is a position inside the value.

Until now such a label was parsed, understood, and then refused -- correctly,
because projecting it needs a partner, and inventing a partner is exactly the
guessing this pipeline exists to avoid. The refusal was recorded only as a
label with no position, which made it indistinguishable from a label that
states nothing at all.

This module supplies the missing proof. It pairs a 63..32 segment with a 31..0
segment only when the document itself says they are halves of one value: the
same name once the range is removed, the same units, explicit word numbers,
adjacent words, high before low, and the two halves covering 63..0 exactly
once. Nothing here measures anything, and nothing here reads prose. When any
part of that is unclear the pair is refused, with the reason, and the label
stays exactly as unresolved as it was.

Deliberately narrow. Repeated word-local ranges, three-way splits, values that
are not 64 bits and halves that are not exactly 32 bits wide are all outside
it. They are different problems and they need their own evidence.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Mapping, Sequence

from standard_schema import sha256_json
from standard_stated_range import (DEFAULT_WORD_WIDTH, RangeContext, RangeDomain,
                                   RangeStatus, parse_stated_ranges)
from standard_word_association import (AssociationSource, UNRESOLVED,
                                       associate_words, is_pure_numeric_ruler_cell,
                                       ruler_band, word_index_cells)

VALUE_PAIR_MODEL_VERSION = "value-local-pairs-v1"

# The only split this phase supports: one 64-bit value, two 32-bit halves.

SUPPORTED_VALUE_WIDTH = 64
HIGH_SEGMENT = (63, 32)
LOW_SEGMENT = (31, 0)
SEGMENT_WIDTH = 32

# Why the word-local placement of a high segment may be believed. It names the
# evidence, not the act, so a reader can tell what would have to change.

ADJACENT_WORD_PAIR = "ADJACENT_WORD_PAIR_64"

# Provenance a pairing will accept. A projection puts bits somewhere the label
# never named, so the label itself has to be the cell's own text.

DIRECT_TEXT_MATCH = "DIRECT_TEXT_MATCH"

NO_LOW_SEGMENT = "NO_LOW_SEGMENT"
NO_HIGH_SEGMENT = "NO_HIGH_SEGMENT"
AMBIGUOUS_SEGMENTS = "AMBIGUOUS_SEGMENTS"
WORDS_NOT_ADJACENT = "WORDS_NOT_ADJACENT"
WORD_ORDER_REVERSED = "WORD_ORDER_REVERSED"
WORD_IDENTITY_NOT_EXPLICIT = "WORD_IDENTITY_NOT_EXPLICIT"
PROVENANCE_NOT_DIRECT = "PROVENANCE_NOT_DIRECT"
NO_CANONICAL_SOURCE = "NO_CANONICAL_SOURCE"
WORD_WIDTH_UNSUPPORTED = "WORD_WIDTH_UNSUPPORTED"
SEGMENT_WIDTH_UNSUPPORTED = "SEGMENT_WIDTH_UNSUPPORTED"

_RULER_LABEL = re.compile(r"^\d{1,3}$")

# Everything that is not a letter or a digit is punctuation between words, and
# the range token has already been cut out by the time this runs.

_NOT_WORD = re.compile(r"[^0-9A-Za-z]+")


class SegmentRole(StrEnum):
    """Which half of the value a segment is."""

    HIGH = "HIGH"
    LOW = "LOW"


class CoordinateDomain(StrEnum):
    """Which coordinate system a declared range is expressed in.

    WORD_LOCAL is the ordinary case and the one everything before STD2E
    assumed: the label's numbers are bit positions in its own word. VALUE_LOCAL
    means they are positions inside a wider value, and the word-local placement
    had to be derived.
    """

    WORD_LOCAL = "WORD_LOCAL"
    VALUE_LOCAL = "VALUE_LOCAL"


def normalize_value_base(text: str, *, range_text: str = "") -> str:
    """What a label names once its range token and punctuation are gone.

    Units stay in. Two halves of one quantity print the same unit, so keeping
    it makes a unit that differs a mismatch rather than something to reconcile.
    """
    value = text.replace(range_text, " ") if range_text else text

    return " ".join(_NOT_WORD.sub(" ", value).upper().split())


@dataclass(frozen=True)
class StandardValueSegment:
    """One half of a value, and the cell that states it."""

    cell_id: str
    text: str
    base: str
    word_index: int
    declared_msb: int
    declared_lsb: int
    segment_role: SegmentRole
    segment_index: int
    provenance: str
    source_ids: tuple[str, ...]
    range_text: str

    @property
    def value_local(self) -> bool:
        """Whether the declared numbers mean something other than word bits."""
        return (self.declared_msb, self.declared_lsb) != LOW_SEGMENT

    @property
    def coordinate_domain(self) -> CoordinateDomain:
        return (CoordinateDomain.VALUE_LOCAL if self.value_local
                else CoordinateDomain.WORD_LOCAL)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["segment_role"] = self.segment_role.value
        value["coordinate_domain"] = self.coordinate_domain.value
        value["source_ids"] = list(self.source_ids)

        return value


@dataclass(frozen=True)
class StandardValuePair:
    """Two segments the document proves are halves of one 64-bit value."""

    group_id: str
    base: str
    value_width: int
    high: StandardValueSegment
    low: StandardValueSegment
    projection_source: str = ADJACENT_WORD_PAIR

    @property
    def segments(self) -> tuple[StandardValueSegment, ...]:
        return (self.high, self.low)

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return (self.high.cell_id, self.low.cell_id)

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    def by_cell(self) -> dict[str, StandardValueSegment]:
        return {item.cell_id: item for item in self.segments}

    def to_dict(self) -> dict[str, object]:
        return {"group_id": self.group_id, "base": self.base,
                "value_width": self.value_width,
                "projection_source": self.projection_source,
                "segment_count": self.segment_count,
                "segments": [item.to_dict() for item in self.segments]}


def _group_id(base: str, segments: Sequence[StandardValueSegment]) -> str:
    return "vgr-" + sha256_json(
        [base, SUPPORTED_VALUE_WIDTH,
         [[item.cell_id, item.declared_msb, item.declared_lsb, item.word_index]
          for item in segments]])[:16]


def _candidate_cells(bitfield: Mapping[str, object],
                     table: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Cells that could name something: not the ruler, not the word column."""
    heading, index_cells = word_index_cells(table)
    reserved = {str(cell["cell_id"]) for cell in index_cells}

    if heading is not None:
        reserved.add(str(heading["cell_id"]))

    band = ruler_band(bitfield.get("bit_labels", ()))
    found = []

    for row in table["rows"]:
        for cell in row["cells"]:
            text = str(cell.get("text", "")).strip()

            if (str(cell["cell_id"]) in reserved or not cell.get("semantic")
                    or not text or _RULER_LABEL.match(text)
                    or is_pure_numeric_ruler_cell(text, cell["bbox"], band)):
                continue

            found.append(cell)

    return found


def value_local_pairs(bitfield: Mapping[str, object], table: Mapping[str, object],
                      *, word_width: int = DEFAULT_WORD_WIDTH,
                      ) -> tuple[tuple[StandardValuePair, ...],
                                 tuple[dict[str, object], ...]]:
    """Every 64-bit word pair the diagram proves, and every one it refuses.

    The canonical cell text is what is read, never a label derived from it, so
    a pairing is the same fact wherever it is asked for: building a definition,
    counting what a reviewer must classify, and serving one back.
    """

    if word_width != SEGMENT_WIDTH:
        return (), ({"reason": WORD_WIDTH_UNSUPPORTED,
                     "detail": f"value pairing supports {SEGMENT_WIDTH}-bit words "
                               f"and this diagram uses {word_width}"},)

    labels = tuple(int(item["value"]) for item in bitfield.get("bit_labels", ()))
    context = RangeContext(in_bitfield_region=True, ruler_labels=labels,
                           word_width=word_width)
    associations = associate_words(table)

    by_base: dict[str, list[StandardValueSegment]] = {}
    refusals: list[dict[str, object]] = []

    for cell in _candidate_cells(bitfield, table):
        text = str(cell["text"]).strip()
        ranges = [item for item in parse_stated_ranges(text, context=context)
                  if item.status is RangeStatus.ACCEPTED]

        if len(ranges) != 1:
            # A label carrying no range, or several, is not one half of one
            # value. Both are ordinary and neither is worth a refusal note.

            continue

        found = ranges[0]
        declared = (found.high_bit, found.low_bit)

        if declared not in (HIGH_SEGMENT, LOW_SEGMENT):
            if found.domain is RangeDomain.EXCEEDS_WORD_WIDTH:
                refusals.append({
                    "cell_id": str(cell["cell_id"]), "text": text,
                    "reason": SEGMENT_WIDTH_UNSUPPORTED,
                    "detail": f"declared {found.high_bit}..{found.low_bit} is not "
                              f"{HIGH_SEGMENT[0]}..{HIGH_SEGMENT[1]}"})

            continue

        role = (SegmentRole.HIGH if declared == HIGH_SEGMENT else SegmentRole.LOW)
        base = normalize_value_base(text, range_text=found.range_text)
        association = associations.get(str(cell["cell_id"]), UNRESOLVED)
        by_base.setdefault(base, []).append(StandardValueSegment(
            cell_id=str(cell["cell_id"]), text=text, base=base,
            word_index=(association.word_index
                        if association.association_source
                        is AssociationSource.EXPLICIT_WORD_INDEX
                        and association.word_index is not None else -1),
            declared_msb=found.high_bit, declared_lsb=found.low_bit,
            segment_role=role, segment_index=0 if role is SegmentRole.HIGH else 1,
            provenance=str(cell["provenance"]),
            source_ids=tuple(str(value) for value in cell.get("source_ids", ())),
            range_text=found.range_text))

    pairs: list[StandardValuePair] = []

    for base in sorted(by_base):
        members = by_base[base]
        highs = [item for item in members if item.segment_role is SegmentRole.HIGH]
        lows = [item for item in members if item.segment_role is SegmentRole.LOW]

        if not highs:
            # A plain word-local 31..0 label, which is the common case and not
            # a value at all. Only say something when a high half was there.

            continue

        note = {"base": base,
                "cell_ids": sorted(item.cell_id for item in members)}

        if len(highs) > 1 or len(lows) > 1:
            refusals.append({**note, "reason": AMBIGUOUS_SEGMENTS,
                             "detail": f"{len(highs)} high and {len(lows)} low "
                                       "segments share this name"})
            continue

        if not lows:
            refusals.append({**note, "reason": NO_LOW_SEGMENT,
                             "detail": "nothing states the matching "
                                       f"{LOW_SEGMENT[0]}..{LOW_SEGMENT[1]} half"})
            continue

        high, low = highs[0], lows[0]

        if high.word_index < 0 or low.word_index < 0:
            refusals.append({**note, "reason": WORD_IDENTITY_NOT_EXPLICIT,
                             "detail": "a projected segment needs an explicit "
                                       "word number, not a measured row"})
            continue

        if low.word_index != high.word_index + 1:
            reason = (WORD_ORDER_REVERSED if low.word_index < high.word_index
                      else WORDS_NOT_ADJACENT)
            refusals.append({**note, "reason": reason,
                             "detail": f"high half is word {high.word_index} and "
                                       f"low half is word {low.word_index}"})

            continue

        if any(item.provenance != DIRECT_TEXT_MATCH for item in (high, low)):
            refusals.append({**note, "reason": PROVENANCE_NOT_DIRECT,
                             "detail": "a projected segment needs the cell's own "
                                       "canonical text"})
            continue

        if not high.source_ids or not low.source_ids:
            refusals.append({**note, "reason": NO_CANONICAL_SOURCE,
                             "detail": "a segment cites no canonical source"})
            continue

        # 63..32 and 31..0 cover 63..0 once each. Asserted rather than assumed,
        # because the two constants are the whole reason the projection holds.

        covered = set(range(low.declared_lsb, low.declared_msb + 1))
        upper = set(range(high.declared_lsb, high.declared_msb + 1))

        if covered & upper or len(covered | upper) != SUPPORTED_VALUE_WIDTH:
            refusals.append({**note, "reason": SEGMENT_WIDTH_UNSUPPORTED,
                             "detail": "the two halves do not cover the value once"})
            continue

        pairs.append(StandardValuePair(
            group_id=_group_id(base, (high, low)), base=base,
            value_width=SUPPORTED_VALUE_WIDTH, high=high, low=low))

    return tuple(pairs), tuple(refusals)


def pair_segments(bitfield: Mapping[str, object], table: Mapping[str, object],
                  *, word_width: int = DEFAULT_WORD_WIDTH,
                  ) -> dict[str, tuple[StandardValuePair, StandardValueSegment]]:
    """Every paired cell, with the pair it belongs to. Keyed by cell id."""
    pairs, _ = value_local_pairs(bitfield, table, word_width=word_width)

    return {cell_id: (pair, segment) for pair in pairs
            for cell_id, segment in pair.by_cell().items()}
