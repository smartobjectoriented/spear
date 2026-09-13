"""Which half of a word a packed 16-bit subfield occupies, when prose says so.

Some words of this standard hold two independent quantities side by side, and
each label numbers its own bits from zero. Both therefore print (15..0), and
read as word-local ranges they collide: two fields claiming bits 15..0 of one
word. Promotion refuses that, correctly, because the diagram alone does not
say which quantity sits in the upper half and which in the lower.

Several sections say it in prose, and that prose is the only authority this
module accepts. It looks for the shapes the document actually uses -- a named
quantity assigned to the upper or lower 16 bits, a portion named by explicit
bits expressing a quantity, and a pair assigned to the two halves
respectively -- and for one bounded indirection, a rule that defers to another
section's rule by naming it.

Nothing here measures anything. STD2E-B0 established that a label's box never
covers its field, and that the visual left/right split, though strikingly
regular, rests on no normative statement anywhere in the document. Where the
prose is absent, incomplete, or contradictory the pair stays exactly as
refused as it was, with the reason recorded.

The two quantities are not one value. They share a word and nothing else, so
they form a packing group rather than the value group STD2E-A uses for a
64-bit number split across two words. Each keeps its own 16-bit width, and the
model never invites a reader to concatenate them.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Iterable, Mapping, Sequence

from standard_schema import sha256_json
from standard_stated_range import (DEFAULT_WORD_WIDTH, RangeContext, RangeStatus,
                                   parse_stated_ranges)

SUBFIELD_ASSIGNMENT_MODEL_VERSION = "normative-subfield-assignment-v1"

# The only packing this phase supports: two 16-bit quantities in one 32-bit word.

PACKED_WIDTH = 16
PACKED_MEMBERS = 2
DECLARED_RANGE = (15, 0)
HIGH_BITS = (31, 16)
LOW_BITS = (15, 0)

# Where a projected placement came from. Never a stated range: the label said
# 15..0, and a rule elsewhere said which 16 bits those are.

NORMATIVE_SUBFIELD_ASSIGNMENT = "NORMATIVE_SUBFIELD_ASSIGNMENT"

NO_ASSIGNMENT = "NO_ASSIGNMENT"
PARTIAL_ASSIGNMENT = "PARTIAL_ASSIGNMENT"
SLOT_CONFLICT = "SLOT_CONFLICT"
AMBIGUOUS_NAME_MATCH = "AMBIGUOUS_NAME_MATCH"
NAME_NOT_MATCHED = "NAME_NOT_MATCHED"
STATED_RANGE_CONFLICT = "STATED_RANGE_CONFLICT"
CROSS_REFERENCE_UNRESOLVED = "CROSS_REFERENCE_UNRESOLVED"
MEMBER_COUNT_UNSUPPORTED = "MEMBER_COUNT_UNSUPPORTED"

_NOT_WORD = re.compile(r"[^0-9A-Za-z]+")

# A label carries its own range, and the range is not part of the name.

_RANGE_TOKEN = re.compile(r"\(?\s*\d{1,2}\s*\.\.\s*\d{1,2}\s*\)?")

# Written ordinals the document mixes with digits: "third" and "3rd" and a
# stray "rd" left behind when a superscript is extracted out of order.

_ORDINAL = {"first": "1", "second": "2", "third": "3", "fourth": "4",
            "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8",
            "1st": "1", "2nd": "2", "3rd": "3", "4th": "4", "5th": "5"}
_ORDINAL_SUFFIX = {"st", "nd", "rd", "th"}

# Words that carry no identity: they name the container or the act, not the
# quantity, and the document uses them inconsistently.

_NOISE = {"the", "a", "an", "of", "field", "fields", "subfield", "subfields",
          "value", "values", "bit", "bits", "word", "format", "db", "dbm",
          "degrees", "degree", "radians", "hz", "optional", "required"}

_HIGH_PHRASE = re.compile(r"upper\s+16\s+bits|upper\s+portion|bits\s+31\s*\.\.\s*16",
                          re.I)
_LOW_PHRASE = re.compile(r"lower\s+16\s+bits|lower\s+portion|bits\s+15\s*\.\.\s*0",
                         re.I)
_PORTION = re.compile(
    r"\b(upper|lower)\s+portion\s*(?:\(\s*bits\s*(\d{1,2})\s*\.\.\s*(\d{1,2})\s*\))?",
    re.I)
_EXPRESS = re.compile(r"shall\s+express\s+(?:the\s+)?(.+?)(?:\s+of\s+|\s*[.,;]|$)",
                      re.I)
_RESPECTIVELY = re.compile(
    r"(?P<names>[^,]+?)\s+and\s+(?P<second>[^,]+?)\s*,?\s+which\s+occupy\s+the\s+"
    r"(?P<first_slot>upper|lower)\s+and\s+(?P<second_slot>upper|lower)\s+16\s+bits"
    r".*?\brespectively\b", re.I)
_ALIAS = re.compile(
    r"^the\s+(?P<name>.+?)\s+field\s+shall\s+express\s+(?:the\s+)?"
    r"(?P<means>.+?)(?:\s+of\s+|\s*[.,;]|$)", re.I)
_SECTION_REFERENCE = re.compile(r"\bSection\s+(\d+(?:\.\d+)+)", re.I)
_SUBJECT = re.compile(r"^(?:the\s+)?(?P<name>.+?)\s+(?:shall|is|are)\b", re.I)


class PackingSlot(StrEnum):
    """Which half of the physical word a packed quantity occupies."""

    HIGH_SLOT = "HIGH_SLOT"
    LOW_SLOT = "LOW_SLOT"

    @property
    def bits(self) -> tuple[int, int]:
        return HIGH_BITS if self is PackingSlot.HIGH_SLOT else LOW_BITS


class AssignmentForm(StrEnum):
    """The sentence shape the assignment was read from."""

    SUBJECT_NAMED = "SUBJECT_NAMED"
    PORTION_NAMED = "PORTION_NAMED"
    RESPECTIVELY = "RESPECTIVELY"


def normalize_quantity(text: str) -> tuple[str, ...]:
    """A quantity's identity: its meaningful tokens, ordinals in digits.

    Bounded on purpose. Case, punctuation, units and container nouns come out;
    "third" becomes "3" because the document writes the same quantity both
    ways. Nothing here matches on similarity, so two different quantities stay
    two different quantities.
    """
    tokens = []

    for token in _NOT_WORD.sub(" ", _RANGE_TOKEN.sub(" ", text)).lower().split():
        token = _ORDINAL.get(token, token)

        if token in _ORDINAL_SUFFIX:
            # A superscript extracted away from its number, as in "3 rd".

            continue

        if token in _NOISE:
            continue

        tokens.append(token)

    return tuple(tokens)


@dataclass(frozen=True)
class StandardSubfieldAssignment:
    """One rule sentence putting one named quantity in one half of a word."""

    name: str
    normalized: tuple[str, ...]
    slot: PackingSlot
    source_id: str
    section: str
    form: AssignmentForm
    statement: int
    msb: int | None = None
    lsb: int | None = None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["slot"] = self.slot.value
        value["form"] = self.form.value
        value["normalized"] = list(self.normalized)

        return value


@dataclass(frozen=True)
class StandardQuantityAlias:
    """A rule defining one name as meaning another, in one section."""

    name: str
    normalized: tuple[str, ...]
    means: tuple[str, ...]
    source_id: str


@dataclass(frozen=True)
class StandardPackedSlot:
    """One diagram label, projected into its half of the physical word."""

    cell_id: str
    label: str
    word_index: int | None
    slot: PackingSlot
    msb: int
    lsb: int
    declared_msb: int
    declared_lsb: int
    value_width: int
    packing_group_id: str
    authority: str
    source_id: str
    section: str
    form: str
    cross_reference: str | None = None
    reference_source_id: str | None = None
    alias_of: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["slot"] = self.slot.value
        value["alias_of"] = list(self.alias_of) if self.alias_of else None

        return value


def _statements(unit) -> list[tuple[int, str]]:
    return [(index, item.strip()) for index, item
            in enumerate(re.split(r"(?<=[.:])\s+", unit.text)) if item.strip()]


def _is_requirement(unit) -> bool:
    return getattr(getattr(unit, "content_type", None), "value", "") == "REQUIREMENT"


def subfield_assignments(units: Iterable, *, section: str,
                         ) -> tuple[tuple[StandardSubfieldAssignment, ...],
                                    tuple[StandardQuantityAlias, ...]]:
    """Every slot assignment and name definition this section states.

    Only requirement text of this one section is read. A sentence that does not
    put a named quantity in a named half is not an assignment and is ignored
    rather than guessed at.
    """
    found: list[StandardSubfieldAssignment] = []
    aliases: list[StandardQuantityAlias] = []

    for unit in units:
        if getattr(unit, "section", None) != section or not _is_requirement(unit):
            continue

        source_id = str(getattr(unit, "source_id", ""))

        for index, sentence in _statements(unit):
            alias = _ALIAS.match(sentence)

            if alias and not (_HIGH_PHRASE.search(sentence)
                              or _LOW_PHRASE.search(sentence)):
                name = alias.group("name").strip()
                means = normalize_quantity(alias.group("means"))

                if means and normalize_quantity(name) != means:
                    aliases.append(StandardQuantityAlias(
                        name=name, normalized=normalize_quantity(name),
                        means=means, source_id=source_id))

            both = _RESPECTIVELY.search(sentence)

            if both:
                names = [both.group("names"), both.group("second")]

                # "contains two 16-bit subfields, A and B" -- the list starts
                # after the last comma before the first name.

                names[0] = names[0].split(",")[-1]
                slots = [PackingSlot.HIGH_SLOT
                         if both.group(key).lower() == "upper"
                         else PackingSlot.LOW_SLOT
                         for key in ("first_slot", "second_slot")]

                for name, slot in zip(names, slots):
                    tokens = normalize_quantity(name)

                    if tokens:
                        found.append(StandardSubfieldAssignment(
                            name=name.strip(), normalized=tokens, slot=slot,
                            source_id=source_id, section=section,
                            form=AssignmentForm.RESPECTIVELY, statement=index,
                            msb=slot.bits[0], lsb=slot.bits[1]))

                continue

            high, low = _HIGH_PHRASE.search(sentence), _LOW_PHRASE.search(sentence)

            if bool(high) == bool(low):
                # Neither half named, or both named in one sentence: not a
                # statement this module is willing to read as an assignment.

                continue

            slot = PackingSlot.HIGH_SLOT if high else PackingSlot.LOW_SLOT
            portion = _PORTION.search(sentence)

            if portion:
                express = _EXPRESS.search(sentence)

                if not express:
                    continue

                tokens = normalize_quantity(express.group(1))

                if not tokens:
                    continue

                msb, lsb = slot.bits

                if portion.group(2) and portion.group(3):
                    msb, lsb = int(portion.group(2)), int(portion.group(3))

                    if (msb, lsb) != slot.bits:
                        # The words and the numbers disagree; say nothing.

                        continue

                found.append(StandardSubfieldAssignment(
                    name=express.group(1).strip(), normalized=tokens, slot=slot,
                    source_id=source_id, section=section,
                    form=AssignmentForm.PORTION_NAMED, statement=index,
                    msb=msb, lsb=lsb))

                continue

            subject = _SUBJECT.match(sentence)

            if not subject:
                continue

            tokens = normalize_quantity(subject.group("name"))

            if not tokens:
                continue

            found.append(StandardSubfieldAssignment(
                name=subject.group("name").strip(), normalized=tokens, slot=slot,
                source_id=source_id, section=section,
                form=AssignmentForm.SUBJECT_NAMED, statement=index,
                msb=slot.bits[0], lsb=slot.bits[1]))

    return tuple(found), tuple(aliases)


def referenced_sections(units: Iterable, *, section: str) -> tuple[str, ...]:
    """Sections this one's rules explicitly defer to, in reading order.

    A deferral is only a pointer, and the authority still comes from the
    requirement text it points at, so the sentence carrying it is not itself
    required to be typed as a requirement. That matters in practice: one rule
    of this document is split by extraction, and its deferral lands in the
    untyped tail. The target section's rules remain requirement-gated.
    """
    found: list[str] = []

    for unit in units:
        if getattr(unit, "section", None) != section:
            continue

        for _index, sentence in _statements(unit):
            if not re.search(r"follow the regulations|as (?:per|specified|given) "
                             r"in|shall .{0,40}\bas per\b", sentence, re.I):
                continue

            for match in _SECTION_REFERENCE.finditer(sentence):
                value = match.group(1)

                if value != section and value not in found:
                    found.append(value)

    return tuple(found)


def _reference_source(units: Iterable, *, section: str) -> str | None:
    for unit in units:
        if getattr(unit, "section", None) != section:
            continue

        for _index, sentence in _statements(unit):
            if _SECTION_REFERENCE.search(sentence) and re.search(
                    r"follow the regulations|as (?:per|specified|given) in",
                    sentence, re.I):
                return str(getattr(unit, "source_id", ""))

    return None


def _packing_group_id(section: str, members: Sequence[tuple[str, str]]) -> str:
    return "pkg-" + sha256_json([section, sorted(members)])[:16]


def _match(assignment: StandardSubfieldAssignment,
           candidates: Mapping[str, tuple[str, ...]],
           aliases: Sequence[StandardQuantityAlias],
           ) -> tuple[list[str], tuple[str, ...] | None]:
    """The diagram labels this assignment names, and the alias that helped."""

    def hits(tokens):
        return [cell for cell, label in candidates.items()
                if tokens and set(tokens) <= set(label)]

    direct = hits(assignment.normalized)

    if direct:
        return direct, None

    for alias in aliases:
        if alias.normalized == assignment.normalized:
            found = hits(alias.means)

            if found:
                return found, alias.normalized

    return [], None


def packed_slot_projection(
    bitfield: Mapping[str, object], table: Mapping[str, object],
    units: Iterable, *, associations: Mapping[str, int | None],
    section: str, word_width: int = DEFAULT_WORD_WIDTH,
) -> tuple[dict[str, StandardPackedSlot], tuple[dict[str, object], ...]]:
    """Project a word's two packed quantities, or say why it cannot be done.

    The diagram supplies the pair; the prose supplies the halves. Both must be
    unambiguous, and a physical range the label states outright always wins.
    """
    units = list(units)
    labels = tuple(int(item["value"]) for item in bitfield.get("bit_labels", ()))
    context = RangeContext(in_bitfield_region=True, ruler_labels=labels,
                           word_width=word_width)

    # The candidate pair: cells of one word whose labels both declare 15..0.

    by_word: dict[object, list[Mapping[str, object]]] = {}
    stated_elsewhere: list[Mapping[str, object]] = []

    for row in table["rows"]:
        for cell in row["cells"]:
            text = str(cell.get("text", "")).strip()

            if not cell.get("semantic") or not text:
                continue

            ranges = [item for item in parse_stated_ranges(text, context=context)
                      if item.status is RangeStatus.ACCEPTED]

            if len(ranges) != 1:
                continue

            found = ranges[0]
            width = found.high_bit - found.low_bit + 1

            if (found.high_bit, found.low_bit) == DECLARED_RANGE:
                by_word.setdefault(associations.get(str(cell["cell_id"])),
                                   []).append(cell)
            elif width == PACKED_WIDTH:
                # A 16-bit quantity that names its physical half outright. It
                # needs no projection, and it is the stronger authority if a
                # rule ever disagrees with it.

                stated_elsewhere.append((cell, found.high_bit, found.low_bit))

    refusals: list[dict[str, object]] = []
    projected: dict[str, StandardPackedSlot] = {}
    section_assignments, section_aliases = subfield_assignments(units,
                                                                section=section)

    for word in sorted({associations.get(str(c["cell_id"]))
                        for group in by_word.values() for c in group}
                       | {associations.get(str(c["cell_id"]))
                          for c, _h, _l in stated_elsewhere},
                       key=str):
        if word is None:
            continue

        cells = by_word.get(word, [])
        explicit = [(c, h, l) for c, h, l in stated_elsewhere
                    if associations.get(str(c["cell_id"])) == word]
        note = {"word_index": word,
                "cell_ids": sorted(str(c["cell_id"]) for c in cells)}

        # A rule that puts a quantity in a half the label itself contradicts is
        # a disagreement in the document, and is never resolved by preferring
        # one of them silently.

        for cell, high, low in explicit:
            tokens = normalize_quantity(str(cell["text"]))

            for assignment in section_assignments:
                if not (assignment.normalized
                        and set(assignment.normalized) <= set(tokens)):
                    continue

                if (high, low) != assignment.slot.bits:
                    refusals.append({
                        **note, "reason": STATED_RANGE_CONFLICT,
                        "detail": f"{assignment.name!r} states bits {high}..{low} "
                                  f"and a rule assigns it to "
                                  f"{assignment.slot.value}"})
                    break

        if any(r["reason"] == STATED_RANGE_CONFLICT and r["word_index"] == word
               for r in refusals):
            continue

        if len(cells) < PACKED_MEMBERS:
            continue

        if len(cells) != PACKED_MEMBERS:
            refusals.append({**note, "reason": MEMBER_COUNT_UNSUPPORTED,
                             "detail": f"{len(cells)} quantities declare "
                                       f"{DECLARED_RANGE[0]}..{DECLARED_RANGE[1]} "
                                       "in this word"})
            continue

        if explicit:
            refusals.append({**note, "reason": STATED_RANGE_CONFLICT,
                             "detail": "a label of this word states its "
                                       "physical half outright"})
            continue

        names = {str(c["cell_id"]): normalize_quantity(str(c["text"]))
                 for c in cells}

        if len(set(names.values())) != PACKED_MEMBERS:
            refusals.append({**note, "reason": AMBIGUOUS_NAME_MATCH,
                             "detail": "the two quantities normalize alike"})
            continue

        found, aliases = section_assignments, section_aliases
        cross_reference = reference_source = None

        if not found:
            for other in referenced_sections(units, section=section):
                borrowed, borrowed_aliases = subfield_assignments(
                    units, section=other)

                if borrowed:
                    found = borrowed
                    aliases = tuple(aliases) + tuple(borrowed_aliases)
                    cross_reference = other
                    reference_source = _reference_source(units, section=section)

                    break

        if not found:
            refusals.append({**note, "reason": NO_ASSIGNMENT,
                             "detail": "no rule of this section assigns a "
                                       "quantity to a half of the word"})
            continue

        chosen: dict[str, tuple[StandardSubfieldAssignment, tuple[str, ...] | None]] = {}
        failed = False

        for assignment in found:
            hits, alias = _match(assignment, names, aliases)

            if not hits:
                continue

            if len(hits) > 1:
                refusals.append({**note, "reason": AMBIGUOUS_NAME_MATCH,
                                 "detail": f"{assignment.name!r} names "
                                           f"{len(hits)} of the diagram's labels"})
                failed = True

                break

            cell_id = hits[0]
            previous = chosen.get(cell_id)

            if previous is not None and previous[0].slot is not assignment.slot:
                refusals.append({**note, "reason": SLOT_CONFLICT,
                                 "detail": f"{assignment.name!r} is assigned to "
                                           "both halves"})
                failed = True

                break

            if previous is None:
                chosen[cell_id] = (assignment, alias)

        if failed:
            continue

        if len(chosen) != PACKED_MEMBERS:
            refusals.append({**note, "reason": PARTIAL_ASSIGNMENT,
                             "detail": f"{len(chosen)} of {PACKED_MEMBERS} "
                                       "quantities are assigned a half"})
            continue

        slots = {cell: item.slot for cell, (item, _) in chosen.items()}

        if set(slots.values()) != {PackingSlot.HIGH_SLOT, PackingSlot.LOW_SLOT}:
            refusals.append({**note, "reason": SLOT_CONFLICT,
                             "detail": "the assignments do not cover one high "
                                       "and one low half"})
            continue

        group = _packing_group_id(
            section, [(cell, slots[cell].value) for cell in sorted(chosen)])

        for cell_id, (assignment, alias) in sorted(chosen.items()):
            cell = next(c for c in cells if str(c["cell_id"]) == cell_id)
            msb, lsb = assignment.slot.bits
            projected[cell_id] = StandardPackedSlot(
                cell_id=cell_id, label=str(cell["text"]).strip(),
                word_index=word, slot=assignment.slot, msb=msb, lsb=lsb,
                declared_msb=DECLARED_RANGE[0], declared_lsb=DECLARED_RANGE[1],
                value_width=PACKED_WIDTH, packing_group_id=group,
                authority=NORMATIVE_SUBFIELD_ASSIGNMENT,
                source_id=assignment.source_id, section=assignment.section,
                form=assignment.form.value, cross_reference=cross_reference,
                reference_source_id=reference_source, alias_of=alias)

    return projected, tuple(refusals)
