"""Semantic bitfield structure derived from human-approved geometry.

Geometry says where the ink is. This says what a reviewed operator decided it
means, and only that.

A field's position comes from one of two places, in this order. If the label
states its own bit range, that range is the position: the document outranks any
measurement of it, and STD2B-R showed why -- a centred label's box is inset from
the field it names by whatever whitespace the typesetter left. Otherwise the
position comes from the visible ruler labels the span covers. Never from prose,
never from a model, and never from the two blended together: where a stated
range and the geometry disagree the stated range wins outright and the
disagreement is recorded for audit.

A position is local to its word. Bits 23..12 of one word and bits 23..12 of
another are different fields in different places, so fields are grouped by the
word the diagram assigns them before anything is checked for overlap, and the
word comes from the document's own word numbering wherever it has one.

Nothing here promotes anything on its own. A definition exists only when a
person approved the candidate it came from, classified its fields, and the
corpus, layout and structure it was approved against are still the current ones.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import Iterable, Mapping, Sequence

from standard_schema import sha256_json
from standard_stated_range import (
    DEFAULT_WORD_WIDTH, RangeContext, RangeDomain, RangeStatus,
    StandardStatedBitRange, parse_stated_ranges, ruler_is_trusted,
    stated_ranges_for_cell,
)
from standard_structure import (
    ProvenanceQuality, StandardStructureError, StandardBitfieldCandidate,
    StandardTableCandidate, unresolved_semantic_cells,
)
from standard_prose_range import (
    ProseFieldTarget, ProseRangeError, StandardNormativeRangeLink,
    StandardReviewedLink, normative_range_links,
)
from standard_subfield_assignment import (
    NORMATIVE_SUBFIELD_ASSIGNMENT, PackingSlot, StandardPackedSlot,
    packed_slot_projection,
)
from standard_value_pair import (
    ADJACENT_WORD_PAIR, CoordinateDomain, SegmentRole, StandardValuePair,
    StandardValueSegment, value_local_pairs,
)
from standard_word_association import (
    AssociationSource, StandardWordAssociation, UNRESOLVED, associate_words,
    field_candidates, page_words,
)


SEMANTIC_SCHEMA_VERSION = 7

# The promotion model: what becomes a field, at what position, in what role.
# It is bound into a reviewer's evidence, so a change here rightly expires
# every approval and must mean the promotion model itself moved. The payload's
# shape is the schema version's business, and derived output -- completeness,
# unresolved words -- is neither: recording those differently is not a reason
# to make a person read a diagram again.

SEMANTIC_MODEL_VERSION = "prose-linked-bitfields-v2"

# An approval record's own shape. v1 classified diagram candidates and nothing
# else; v2 also records what the reviewer decided about each normative link.
# The decision model a reviewer is bound to, and nothing else. It is written
# into review evidence, so moving it expires every approval in the store --
# which is right when what a reviewer decides changes, and wrong for anything
# else. The approvals *file* has its own version, APPROVAL_FILE_SCHEMA_VERSION.

APPROVAL_SCHEMA_VERSION = 2

APPROVING_VERDICTS = ("PASS", "ACCEPTABLE_WARNING")

# A review recorded by the assistant is evidence for a person, never a substitute
# for one. These identities can never approve a semantic promotion.

ASSISTANT_IDENTITY = re.compile(r"^(?:claude|assistant|model|gpt|llm)\b", re.I)
_CANDIDATE_ID = re.compile(r"^bit-[0-9a-f]{16}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"[^A-Za-z0-9]+")


class BitOrder(StrEnum):
    MSB_TO_LSB = "MSB_TO_LSB"
    LSB_TO_MSB = "LSB_TO_MSB"
    UNKNOWN = "UNKNOWN"


class SpanRole(StrEnum):
    """What an operator decided a geometric span actually is."""

    FIELD = "FIELD"
    RESERVED = "RESERVED"
    STRUCTURAL_LABEL = "STRUCTURAL_LABEL"
    ROW_LABEL = "ROW_LABEL"
    IGNORE = "IGNORE"
    UNKNOWN = "UNKNOWN"


PROMOTING_ROLES = (SpanRole.FIELD, SpanRole.RESERVED)


class SemanticRole(StrEnum):
    FIELD = "FIELD"
    RESERVED = "RESERVED"


class PositionSource(StrEnum):
    """Where a field's msb and lsb actually came from, strongest first.

    STATED_RANGE is the diagram's own label. VALUE_LOCAL_PROJECTION is also
    the diagram's own label, but one that states its bits inside a wider value
    rather than inside its word: the word-local position was derived from a
    proven pairing, and calling it STATED_RANGE would tell a caller the source
    said 31..0 when it said 63..32. NORMATIVE_PROSE_RANGE is a rule elsewhere
    in the text that names the field and gives its bits; it is kept distinct so
    an audit can always see the bits did not come from the diagram.

    NORMATIVE_SUBFIELD_PROJECTION is the third coordinate case: the label
    states 15..0 because it numbers its own bits from zero, and a rule says
    which half of the word those sixteen bits are. Distinct from
    VALUE_LOCAL_PROJECTION, which places a half of one wider value from the
    diagram alone; here the diagram cannot place anything and the authority is
    entirely textual.
    """

    STATED_RANGE = "STATED_RANGE"
    VALUE_LOCAL_PROJECTION = "VALUE_LOCAL_PROJECTION"
    NORMATIVE_SUBFIELD_PROJECTION = "NORMATIVE_SUBFIELD_PROJECTION"
    NORMATIVE_PROSE_RANGE = "NORMATIVE_PROSE_RANGE"
    GEOMETRIC_RULER = "GEOMETRIC_RULER"
    UNKNOWN = "UNKNOWN"


class DefinitionCompleteness(StrEnum):
    """Whether every span a reviewer called a field resolved to a position."""

    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"


class StructuralCompleteness(StrEnum):
    """Whether the definition describes the words it covers, or only part of them.

    Resolution completeness asks whether everything a reviewer classified got a
    position. That is a weaker question, and the first real review found the
    gap between them: a word can resolve every candidate it has and still leave
    bits to a field the diagram names but never positions.
    """

    STRUCTURALLY_COMPLETE = "STRUCTURALLY_COMPLETE"
    STRUCTURALLY_INCOMPLETE = "STRUCTURALLY_INCOMPLETE"


class UnresolvedCause(StrEnum):
    """Why a label the diagram names carries no position.

    Both used to be a bare string in a list, so a consumer had to re-parse the
    label to tell them apart. A label stating 63..32 is understood and deferred;
    a label stating nothing is not understood at all. Those are different facts
    about the document and the model now says which one it is.
    """

    NO_DETERMINISTIC_RANGE = "NO_DETERMINISTIC_RANGE"
    VALUE_LOCAL_DEFERRED = "VALUE_LOCAL_DEFERRED"


class StandardSemanticError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# human approval
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StandardBitfieldApproval:
    """One person's decision about one candidate, tied to what they saw."""

    candidate_id: str
    verdict: str
    reviewer: str
    reviewed_at: str
    structure_fingerprint: str
    span_roles: tuple[str, ...] = ()
    packet_identity: str | None = None
    notes: str | None = None
    schema_version: int = APPROVAL_SCHEMA_VERSION
    reviewed_links: tuple[StandardReviewedLink, ...] = ()
    review_evidence_fingerprint: str | None = None

    # Which recorded decision this approval is, and the one it replaced. An
    # approval taken before the history log existed names neither, and stays
    # perfectly usable: absence here means "written earlier", never "invalid".

    event_id: str | None = None
    supersedes: str | None = None

    def __post_init__(self) -> None:
        if not _CANDIDATE_ID.fullmatch(self.candidate_id):
            raise StandardSemanticError("malformed bitfield candidate id")

        if self.verdict not in ("PASS", "ACCEPTABLE_WARNING", "FAIL",
                                "NEEDS_FOLLOWUP"):
            raise StandardSemanticError(f"unknown verdict {self.verdict!r}")

        if not _SHA256.fullmatch(self.structure_fingerprint):
            raise StandardSemanticError("approval must name a structure fingerprint")

        if not self.reviewer or not self.reviewer.strip():
            raise StandardSemanticError("approval must name a reviewer")

        if ASSISTANT_IDENTITY.match(self.reviewer.strip()):
            # An assistant pass is recorded elsewhere; it cannot approve.

            raise StandardSemanticError(
                "an assistant identity cannot approve a semantic promotion")

        for role in self.span_roles:
            if role not in tuple(SpanRole):
                raise StandardSemanticError(f"unknown span role {role!r}")

        if self.schema_version > APPROVAL_SCHEMA_VERSION:
            raise StandardSemanticError(
                f"approval schema v{self.schema_version} is newer than this build")

        seen = [item.link_fingerprint for item in self.reviewed_links]

        if len(set(seen)) != len(seen):
            raise StandardSemanticError("a link is reviewed twice in one approval")

    @property
    def accepted_links(self) -> tuple[StandardReviewedLink, ...]:
        return tuple(item for item in self.reviewed_links if item.accepted)

    def reviewed(self, cell_id: str) -> StandardReviewedLink | None:
        for item in self.reviewed_links:
            if item.target_cell_id == cell_id:
                return item

        return None

    @property
    def approves(self) -> bool:
        return self.verdict in APPROVING_VERDICTS

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["span_roles"] = list(self.span_roles)
        value["reviewed_links"] = [item.to_dict() for item in self.reviewed_links]

        return value

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "StandardBitfieldApproval":
        try:
            return cls(
                candidate_id=str(raw["candidate_id"]), verdict=str(raw["verdict"]),
                reviewer=str(raw["reviewer"]), reviewed_at=str(raw["reviewed_at"]),
                structure_fingerprint=str(raw["structure_fingerprint"]),
                span_roles=tuple(str(item) for item in raw.get("span_roles", ())),
                packet_identity=(str(raw["packet_identity"])
                                 if raw.get("packet_identity") else None),
                notes=(str(raw["notes"]) if raw.get("notes") else None),
                # A record written before links existed reviewed none, and says
                # so; it is never read as having accepted one.
                schema_version=int(raw.get("schema_version", 1)),
                reviewed_links=tuple(
                    StandardReviewedLink.from_dict(item)
                    for item in raw.get("reviewed_links", ())),
                review_evidence_fingerprint=(
                    str(raw["review_evidence_fingerprint"])
                    if raw.get("review_evidence_fingerprint") else None),
                event_id=str(raw["event_id"]) if raw.get("event_id") else None,
                supersedes=(str(raw["supersedes"])
                            if raw.get("supersedes") else None))
        except (KeyError, TypeError, ProseRangeError) as exc:
            raise StandardSemanticError(f"malformed approval: {exc}") from exc


# --------------------------------------------------------------------------
# semantic types
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StandardFieldDefinition:
    field_id: str
    label: str
    display_label: str
    normalized_identifier: str
    semantic_role: SemanticRole
    word_index: int | None
    msb: int
    lsb: int
    width: int
    covered_bit_labels: tuple[int, ...]
    source_cell_ids: tuple[str, ...]
    supporting_source_ids: tuple[str, ...]
    provenance_grades: tuple[str, ...]
    word_label: str | None = None
    word_association_source: AssociationSource = AssociationSource.GEOMETRIC_ROW
    normative_source_id: str | None = None
    normative_link_fingerprint: str | None = None
    position_source: PositionSource = PositionSource.GEOMETRIC_RULER
    stated_range_text: str | None = None
    stated_range_syntax: str | None = None
    geometry_conflict: bool = False
    geometry_msb: int | None = None
    geometry_lsb: int | None = None

    # -- the two coordinate systems, kept apart ---------------------------
    # `msb`/`lsb` are always the physical word, because that is what coverage,
    # overlap and completeness are computed in. `declared_*` is what the label
    # actually says. For every ordinary field the two are the same; for a
    # projected 64-bit half they are not, and a caller must be able to see it.

    coordinate_domain: CoordinateDomain = CoordinateDomain.WORD_LOCAL
    declared_msb: int | None = None
    declared_lsb: int | None = None
    value_group_id: str | None = None
    value_width: int | None = None
    segment_index: int | None = None
    segment_count: int | None = None
    projection_source: str | None = None

    # A packing group is not a value group. Its members share a physical word
    # and nothing else: each stays its own quantity of its own width, and
    # nothing here invites a reader to concatenate them into one number.

    packing_group_id: str | None = None
    packing_slot: str | None = None
    projection_reference: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def value_local(self) -> bool:
        """Whether this field's own label states bits in value coordinates."""
        return self.coordinate_domain is CoordinateDomain.VALUE_LOCAL

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["semantic_role"] = self.semantic_role.value
        value["position_source"] = self.position_source.value
        value["coordinate_domain"] = self.coordinate_domain.value
        value["word_association_source"] = self.word_association_source.value

        for name in ("covered_bit_labels", "source_cell_ids",
                     "supporting_source_ids", "provenance_grades", "warnings"):
            value[name] = list(getattr(self, name))

        return value


@dataclass(frozen=True)
class StandardUnresolvedLabel:
    """One named label the diagram never positions, and why it could not be."""

    label: str
    cell_id: str
    cause: UnresolvedCause
    declared_msb: int | None = None
    declared_lsb: int | None = None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["cause"] = self.cause.value

        return value


@dataclass(frozen=True)
class StandardUnresolvedWord:
    """A word of the diagram this definition does not describe, and why.

    It carries no bits and never becomes a field. Its job is to stop a word
    disappearing: a word whose labels nobody could position used to leave no
    trace at all, so a definition could omit it and still call itself whole.
    """

    word_index: int | None
    word_label: str | None
    unpositioned_labels: tuple[str, ...]
    supporting_source_ids: tuple[str, ...]
    field_count: int

    # The same labels with the reason each one is here. The plain list above
    # stays exactly as it was, so a reader written against it keeps working.

    unresolved_labels: tuple[StandardUnresolvedLabel, ...] = ()

    @property
    def causes(self) -> tuple[str, ...]:
        return tuple(sorted({item.cause.value for item in self.unresolved_labels}))

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["unpositioned_labels"] = list(self.unpositioned_labels)
        value["supporting_source_ids"] = list(self.supporting_source_ids)
        value["unresolved_labels"] = [item.to_dict()
                                      for item in self.unresolved_labels]
        value["causes"] = list(self.causes)

        return value


@dataclass(frozen=True)
class StandardBitfieldDefinition:
    schema_version: int
    definition_id: str
    standard_id: str
    revision: str
    corpus_fingerprint: str
    layout_fingerprint: str
    structure_fingerprint: str
    semantic_model_version: str
    source_bitfield_candidate_id: str
    source_table_id: str
    pages: tuple[int, ...]
    bit_order: BitOrder
    visible_bit_labels: tuple[int, ...]
    fields: tuple[StandardFieldDefinition, ...]
    supporting_source_ids: tuple[str, ...]
    approved_by: str
    approved_at: str
    approval_verdict: str
    provenance_status: str
    completeness: DefinitionCompleteness = DefinitionCompleteness.COMPLETE
    structural_completeness: StructuralCompleteness = (
        StructuralCompleteness.STRUCTURALLY_COMPLETE)
    unresolved_words: tuple[StandardUnresolvedWord, ...] = ()
    unresolved_field_count: int = 0
    warnings: tuple[str, ...] = ()
    semantic_fingerprint: str = ""

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["bit_order"] = self.bit_order.value
        value["completeness"] = self.completeness.value
        value["structural_completeness"] = self.structural_completeness.value
        value["unresolved_words"] = [item.to_dict()
                                     for item in self.unresolved_words]
        value["pages"] = list(self.pages)
        value["visible_bit_labels"] = list(self.visible_bit_labels)
        value["fields"] = [item.to_dict() for item in self.fields]
        value["supporting_source_ids"] = list(self.supporting_source_ids)
        value["warnings"] = list(self.warnings)

        return value


@dataclass(frozen=True)
class StandardPacketDefinition:
    """Only built when a reviewer stated the packet identity explicitly."""

    schema_version: int
    packet_id: str
    standard_id: str
    revision: str
    identity: str
    bitfield_definition_ids: tuple[str, ...]
    supporting_source_ids: tuple[str, ...]
    approved_by: str
    approved_at: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)

        for name in ("bitfield_definition_ids", "supporting_source_ids"):
            value[name] = list(getattr(self, name))

        return value


@dataclass(frozen=True)
class StandardSemanticSet:
    bitfields: tuple[StandardBitfieldDefinition, ...]
    packets: tuple[StandardPacketDefinition, ...]
    blocked: tuple[Mapping[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {"bitfields": [item.to_dict() for item in self.bitfields],
                "packets": [item.to_dict() for item in self.packets],
                "blocked": [dict(item) for item in self.blocked]}


def semantic_fingerprint(semantics: StandardSemanticSet) -> str:
    return sha256_json({
        "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
        "semantic_model_version": SEMANTIC_MODEL_VERSION,
        "bitfields": [item.to_dict() for item in semantics.bitfields],
        "packets": [item.to_dict() for item in semantics.packets],
    })


# --------------------------------------------------------------------------
# derivation
# --------------------------------------------------------------------------

def bit_order_of(labels: Sequence[int]) -> BitOrder:
    """Read the ordering off the ruler itself; never assume left is the msb."""

    if len(labels) < 2:
        return BitOrder.UNKNOWN

    steps = {later - earlier for earlier, later in zip(labels, labels[1:])}

    if steps == {-1}:
        return BitOrder.MSB_TO_LSB

    if steps == {1}:
        return BitOrder.LSB_TO_MSB

    return BitOrder.UNKNOWN


def _normalized_identifier(label: str) -> str:
    """Convenience metadata only. The normative text is `label`."""
    return _IDENTIFIER.sub("_", label).strip("_").upper()


def _contiguous(labels: Sequence[int]) -> bool:
    ordered = sorted(labels)
    return all(later - earlier == 1
               for earlier, later in zip(ordered, ordered[1:]))


def unresolved_label_of(cell: Mapping[str, object], *,
                        ruler_labels: Sequence[int],
                        word_width: int = DEFAULT_WORD_WIDTH,
                        ) -> StandardUnresolvedLabel:
    """One unpositioned label, with the reason it has no position.

    A label whose range parses but names bits of a wider value is understood
    and deferred. A label with no usable range at all is not understood. The
    difference is recorded here rather than left for a consumer to recover by
    parsing the label text back out.
    """
    text = str(cell["text"]).strip()
    context = RangeContext(in_bitfield_region=True,
                           ruler_labels=tuple(ruler_labels), word_width=word_width)
    deferred = next((item for item in parse_stated_ranges(text, context=context)
                     if item.status is RangeStatus.ACCEPTED
                     and item.domain is RangeDomain.EXCEEDS_WORD_WIDTH), None)

    if deferred is not None:
        return StandardUnresolvedLabel(
            label=text, cell_id=str(cell["cell_id"]),
            cause=UnresolvedCause.VALUE_LOCAL_DEFERRED,
            declared_msb=deferred.high_bit, declared_lsb=deferred.low_bit)

    return StandardUnresolvedLabel(label=text, cell_id=str(cell["cell_id"]),
                                   cause=UnresolvedCause.NO_DETERMINISTIC_RANGE)


def _blocked(candidate_id: str, reason: str, detail: str) -> dict[str, object]:
    return {"candidate_id": candidate_id, "reason": reason, "detail": detail}


@dataclass(frozen=True)
class _Position:
    """One field's resolved position, and the authority that resolved it."""

    msb: int
    lsb: int
    covered: tuple[int, ...]
    source: PositionSource
    stated: StandardStatedBitRange | None = None
    geometry_msb: int | None = None
    geometry_lsb: int | None = None
    warnings: tuple[str, ...] = ()

    # What the label declares, which is the same as msb/lsb everywhere except
    # a projected half of a wider value.

    declared_msb: int | None = None
    declared_lsb: int | None = None

    @property
    def width(self) -> int:
        return self.msb - self.lsb + 1

    @property
    def declared(self) -> tuple[int, int]:
        return (self.msb if self.declared_msb is None else self.declared_msb,
                self.lsb if self.declared_lsb is None else self.declared_lsb)


# A bit number measured against a piece of a ruler is a bit number in that
# piece's numbering, which need not be the document's. Promotion still allows
# it -- a person classified the span -- but the definition says so out loud.

PARTIAL_RULER_WARNING = "position measured against a ruler that is not a whole one"

# Since STD2C a stated range carries its own position, so an incomplete ruler
# is a fact to record rather than a reason to refuse the document's own words.

RULER_FRAGMENTED = "RULER_FRAGMENTED"


def _geometric_position(covered: Sequence[int], labels: Sequence[int], *,
                        word_width: int = DEFAULT_WORD_WIDTH) -> _Position | None:
    """The STD2B rule, unchanged: a contiguous run of labels the span covers."""

    if not covered or not _contiguous(covered) or not set(covered) <= set(labels):
        return None

    warnings = () if ruler_is_trusted(labels, word_width=word_width) else (
        PARTIAL_RULER_WARNING,)

    return _Position(msb=max(covered), lsb=min(covered),
                     covered=tuple(sorted(covered)),
                     source=PositionSource.GEOMETRIC_RULER,
                     geometry_msb=max(covered), geometry_lsb=min(covered),
                     warnings=warnings)


def resolve_position(text: str, *, cell_id: str, source_ids: Sequence[str],
                     provenance: str, covered_labels: Sequence[int],
                     ruler_labels: Sequence[int],
                     word_width: int = DEFAULT_WORD_WIDTH,
                     prose: StandardNormativeRangeLink | None = None,
                     segment: StandardValueSegment | None = None,
                     packed: StandardPackedSlot | None = None,
                     ) -> tuple[_Position | None, str, bool]:
    """Where one span's bits are, preferring what the document says outright.

    Returns the position, the reason it could not be resolved, and whether that
    reason should block the whole candidate rather than leave one field out. A
    stated range and a measurement are never combined: if the label states a
    range, that range is the answer and the measurement becomes an audit note.
    """
    context = RangeContext(in_bitfield_region=True,
                           ruler_labels=tuple(ruler_labels),
                           word_width=word_width)
    ranges = stated_ranges_for_cell(
        text, cell_id=cell_id, source_ids=tuple(source_ids),
        provenance=provenance, context=context, covered_labels=covered_labels)

    if packed is not None:
        # The label states 15..0 for its own sixteen bits; a rule of the
        # document says which sixteen. Presenting that as a stated range would
        # claim the diagram placed it, and it did not.

        stated = next((item for item in ranges
                       if (item.high_bit, item.low_bit)
                       == (packed.declared_msb, packed.declared_lsb)), None)
        return _Position(
            msb=packed.msb, lsb=packed.lsb,
            covered=tuple(range(packed.lsb, packed.msb + 1)),
            source=PositionSource.NORMATIVE_SUBFIELD_PROJECTION, stated=stated,
            declared_msb=packed.declared_msb,
            declared_lsb=packed.declared_lsb), "", False

    if segment is not None and segment.value_local:
        # The label states 63..32, which is a position in the value and in no
        # word. STD2E proved which word holds it; the word-local placement is
        # this projection and is never presented as something the label said.

        stated = next((item for item in ranges
                       if (item.high_bit, item.low_bit)
                       == (segment.declared_msb, segment.declared_lsb)), None)
        return _Position(
            msb=word_width - 1, lsb=0,
            covered=tuple(range(0, word_width)),
            source=PositionSource.VALUE_LOCAL_PROJECTION, stated=stated,
            declared_msb=segment.declared_msb,
            declared_lsb=segment.declared_lsb), "", False

    usable = [item for item in ranges if item.promotable]

    if len(usable) > 1:
        return None, "the label states more than one bit range", False

    if usable and prose is not None and (
            prose.high_bit, prose.low_bit) != (usable[0].high_bit, usable[0].low_bit):

        # The diagram and a rule both position this field, differently. Picking
        # either one silently would hide a contradiction in the document.

        return None, (f"the label states bits {usable[0].high_bit}.."
                      f"{usable[0].low_bit} but a normative rule states "
                      f"{prose.high_bit}..{prose.low_bit}"), True

    if usable:
        stated = usable[0]
        geometry = _geometric_position(covered_labels, ruler_labels,
                                       word_width=word_width)
        warnings = list(stated.warnings)

        return _Position(
            msb=stated.high_bit, lsb=stated.low_bit,
            covered=tuple(range(stated.low_bit, stated.high_bit + 1)),
            source=PositionSource.STATED_RANGE, stated=stated,
            geometry_msb=geometry.msb if geometry else None,
            geometry_lsb=geometry.lsb if geometry else None,
            warnings=tuple(warnings)), "", False

    if prose is not None:
        # The diagram names this field and never positions it; a rule does.

        return _Position(
            msb=prose.high_bit, lsb=prose.low_bit,
            covered=tuple(range(prose.low_bit, prose.high_bit + 1)),
            source=PositionSource.NORMATIVE_PROSE_RANGE), "", False

    # Only now does geometry matter, so only now do its gates apply. Corrupt
    # geometry still blocks the whole candidate: it says the region was read
    # wrongly, which is not the same as a field whose bits simply go unstated.

    if covered_labels and not _contiguous(covered_labels):
        return None, "the span covers a broken label range", True

    if covered_labels and not set(covered_labels) <= set(ruler_labels):
        return None, "the span covers labels the ruler does not have", True

    geometry = _geometric_position(covered_labels, ruler_labels,
                                   word_width=word_width)

    if geometry is not None:
        return geometry, "", False

    if ranges:
        # Something range-shaped was found and refused. Say which reason, so a
        # reviewer is not left guessing why the label was not believed.

        why = ", ".join(ranges[0].warnings) or "the stated range is not usable"
        return None, f"stated range refused ({why})", False

    return None, "neither a stated range nor a usable ruler position", False


def promote_bitfield(
    candidate: StandardBitfieldCandidate, table: StandardTableCandidate,
    approval: StandardBitfieldApproval | None, *, standard_id: str, revision: str,
    corpus_fingerprint: str, layout_fingerprint: str, structure_fingerprint: str,
    word_width: int = DEFAULT_WORD_WIDTH,
    words: Sequence[Mapping[str, object]] = (),
    prose_links: Mapping[str, StandardNormativeRangeLink] | None = None,
    packed_slots: Mapping[str, StandardPackedSlot] | None = None,
) -> tuple[StandardBitfieldDefinition | None, dict[str, object] | None]:
    """Turn one reviewed candidate into a definition, or say why it cannot be."""

    if approval is None:
        return None, _blocked(candidate.bitfield_id, "NO_HUMAN_REVIEW",
                              "no human approval recorded for this candidate")

    if not approval.approves:
        return None, _blocked(candidate.bitfield_id, "HUMAN_REVIEW_REJECTED",
                              f"human verdict is {approval.verdict}")

    if approval.structure_fingerprint != structure_fingerprint:
        return None, _blocked(candidate.bitfield_id, "STALE_APPROVAL",
                              "approval was recorded against other geometry")

    if unresolved_semantic_cells(table):
        return None, _blocked(candidate.bitfield_id, "UNRESOLVED_PROVENANCE",
                              "the source table has semantic cells with no source")

    labels = [label.value for label in candidate.bit_labels]
    order = bit_order_of(labels)

    if order is BitOrder.UNKNOWN:
        return None, _blocked(candidate.bitfield_id, "AMBIGUOUS_BIT_ORDER",
                              "the ruler is not a single ascending or descending run")

    available = dict(prose_links or {})
    payload, table_payload = candidate.to_dict(), table.to_dict()
    associations = associate_words(table_payload)

    # A role list classifies what the diagram shows, and only that. A field a
    # rule positions from outside the diagram is not a role to assign; it is a
    # piece of evidence the reviewer either accepted or did not.
    # A label stating 63..32 becomes classifiable only once its 31..0 half is
    # proved, so the pairing has to run before the reviewer's role list is
    # counted against the candidates.

    packed = dict(packed_slots or {})
    pairs, _pair_refusals = value_local_pairs(payload, table_payload,
                                              word_width=word_width)
    segments = {cell_id: segment for pair in pairs
                for cell_id, segment in pair.by_cell().items()}
    groups = {cell_id: pair for pair in pairs for cell_id in pair.cell_ids}
    value_cells = frozenset(segments)
    originals = field_candidates(payload, table_payload, words=words,
                                 value_local_cells=value_cells)

    if len(approval.span_roles) != len(originals):
        return None, _blocked(
            candidate.bitfield_id, "SPANS_NOT_CLASSIFIED",
            f"the candidate has {len(originals)} diagram field candidates; "
            f"the approval classifies {len(approval.span_roles)}")

    accepted: dict[str, StandardReviewedLink] = {}

    for cell_id, link in sorted(available.items()):
        decision = approval.reviewed(cell_id)

        if decision is None:
            # Silence is not consent: an approval that never saw this link
            # cannot be read as permitting it.

            return None, _blocked(
                candidate.bitfield_id, "PROSE_LINK_NOT_REVIEWED",
                f"normative link {link.link_fingerprint} positions "
                f"{link.target_label!r} and the approval does not review it")

        if not decision.matches(link):
            return None, _blocked(
                candidate.bitfield_id, "STALE_PROSE_LINK",
                f"normative link for {link.target_label!r} no longer matches "
                "the evidence the reviewer accepted")

        if decision.accepted:
            accepted[cell_id] = decision

    for decision in approval.reviewed_links:
        if decision.target_cell_id not in available:
            return None, _blocked(
                candidate.bitfield_id, "PROSE_LINK_MISSING",
                f"the approval reviews link {decision.link_fingerprint}, "
                "which this candidate no longer offers")

    if approval.review_evidence_fingerprint is not None:
        current = review_evidence_fingerprint(
            candidate.bitfield_id, structure_fingerprint, table=table_payload,
            originals=originals, roles=approval.span_roles,
            associations=associations, reviewed_links=approval.reviewed_links,
            value_segments=segments, packed_slots=packed)

        if current != approval.review_evidence_fingerprint:
            return None, _blocked(
                candidate.bitfield_id, "STALE_REVIEW_EVIDENCE",
                "what the reviewer was shown is not what this candidate is now")

    links = {cell_id: available[cell_id] for cell_id in accepted}
    entries = field_candidates(payload, table_payload, words=words,
                               value_local_cells=value_cells,
                               prose_cells=frozenset(links))
    roles = list(approval.span_roles) + [
        accepted[entry.cell_id].role for entry in entries[len(originals):]]

    fields: list[StandardFieldDefinition] = []
    warnings: list[str] = []
    supporting: list[str] = []
    unresolved: list[str] = []

    for entry, role_name in zip(entries, roles):
        index = entry.index
        role = SpanRole(role_name)

        if role is SpanRole.UNKNOWN:
            return None, _blocked(candidate.bitfield_id, "SPAN_UNCLASSIFIED",
                                  f"field candidate {index} is still UNKNOWN")

        if role not in PROMOTING_ROLES:
            continue

        if not entry.cell_id:
            return None, _blocked(candidate.bitfield_id, "SPAN_CELL_MISSING",
                                  f"field candidate {index} has no cell in the table")

        if (entry.provenance == ProvenanceQuality.UNRESOLVED.value
                or not entry.source_ids):
            return None, _blocked(candidate.bitfield_id, "UNRESOLVED_PROVENANCE",
                                  f"field candidate {index} has no canonical source")

        association = associations.get(entry.cell_id, UNRESOLVED)

        if not association.resolved:
            # Several words share this band and its lines could not be told
            # apart. Guessing which word owns the bits is exactly the mistake
            # this phase exists to stop.

            return None, _blocked(
                candidate.bitfield_id, "WORD_ASSOCIATION_UNRESOLVED",
                f"field candidate {index} belongs to no resolvable word "
                f"({', '.join(association.warnings) or 'no word evidence'})")

        position, why, blocking = resolve_position(
            entry.text, cell_id=entry.cell_id, source_ids=entry.source_ids,
            provenance=entry.provenance,
            covered_labels=entry.covered_labels, ruler_labels=labels,
            word_width=word_width, prose=links.get(entry.cell_id),
            segment=segments.get(entry.cell_id),
            packed=packed.get(entry.cell_id))

        if blocking:
            reason = ("PROSE_RANGE_CONFLICTS_LOCAL" if "normative rule" in why
                      else "SPAN_NOT_CONTIGUOUS" if "broken" in why
                      else "SPAN_OUTSIDE_RULER")
            return None, _blocked(candidate.bitfield_id, reason,
                                  f"field candidate {index} {why}")

        if position is None:
            # A field a person named but whose bits nothing states and nothing
            # measures. Leave it out rather than invent it, and say so.

            unresolved.append(f"field candidate {index}: {why}")
            continue

        semantic = (SemanticRole.RESERVED if role is SpanRole.RESERVED
                    else SemanticRole.FIELD)
        stated = position.stated

        # The link only enters the identity of a field it actually positioned,
        # so introducing prose linking does not rename fields that predate it.

        identity = [candidate.bitfield_id, structure_fingerprint, entry.text,
                    list(position.covered), list(entry.source_ids),
                    semantic.value, position.source.value,
                    association.word_index, association.word_label]

        if position.source is PositionSource.NORMATIVE_PROSE_RANGE:
            identity.append(links[entry.cell_id].link_fingerprint)

        segment = segments.get(entry.cell_id)
        pair = groups.get(entry.cell_id)
        slot = packed.get(entry.cell_id)

        if slot is not None:
            identity.append([slot.packing_group_id, slot.slot.value,
                             slot.source_id])

        if position.source is PositionSource.VALUE_LOCAL_PROJECTION:
            # Only a projected field's identity moves, so pairing does not
            # rename any field that predates it.

            identity.append([pair.group_id, segment.declared_msb,
                             segment.declared_lsb])

        field_id = "fld-" + sha256_json(identity)[:16]
        declared_msb, declared_lsb = position.declared
        fields.append(StandardFieldDefinition(
            field_id=field_id, label=entry.text,
            display_label=stated.display_label if stated else entry.text,
            normalized_identifier=_normalized_identifier(
                stated.display_label if stated else entry.text),
            semantic_role=semantic, word_index=association.word_index,
            msb=position.msb, lsb=position.lsb, width=position.width,
            covered_bit_labels=position.covered,
            source_cell_ids=(entry.cell_id,),
            supporting_source_ids=tuple(entry.source_ids),
            provenance_grades=(entry.provenance,),
            word_label=association.word_label,
            word_association_source=association.association_source,
            normative_source_id=(links[entry.cell_id].requirement_source_id
                                 if position.source is
                                 PositionSource.NORMATIVE_PROSE_RANGE
                                 else slot.source_id if slot is not None else None),
            normative_link_fingerprint=(links[entry.cell_id].link_fingerprint
                                        if position.source is
                                        PositionSource.NORMATIVE_PROSE_RANGE
                                        else None),
            position_source=position.source,
            stated_range_text=stated.range_text if stated else None,
            stated_range_syntax=stated.syntax_kind.value if stated else None,
            geometry_conflict=bool(stated and stated.geometry_conflict),
            geometry_msb=position.geometry_msb, geometry_lsb=position.geometry_lsb,
            coordinate_domain=(
                CoordinateDomain.VALUE_LOCAL
                if slot is not None and slot.slot is PackingSlot.HIGH_SLOT
                else segment.coordinate_domain if segment is not None
                else CoordinateDomain.WORD_LOCAL),
            declared_msb=declared_msb, declared_lsb=declared_lsb,
            value_group_id=pair.group_id if pair is not None else None,
            value_width=(pair.value_width if pair is not None
                         else slot.value_width if slot is not None else None),
            segment_index=segment.segment_index if segment is not None else None,
            segment_count=pair.segment_count if pair is not None else None,
            projection_source=(pair.projection_source
                               if position.source
                               is PositionSource.VALUE_LOCAL_PROJECTION
                               else slot.authority if slot is not None else None),
            packing_group_id=slot.packing_group_id if slot is not None else None,
            packing_slot=slot.slot.value if slot is not None else None,
            projection_reference=(slot.cross_reference
                                  if slot is not None else None),
            warnings=tuple(dict.fromkeys(
                tuple(entry.warnings) + tuple(position.warnings)))))
        supporting.extend(entry.source_ids)

    if not fields:
        return None, _blocked(
            candidate.bitfield_id,
            "NO_RESOLVED_FIELD_POSITION" if unresolved else "NO_PROMOTABLE_SPAN",
            unresolved[0] if unresolved else "no candidate was classified as a field")

    warnings.extend(unresolved)

    # Bits are local to their word, so only fields sharing a word can collide.

    by_word: dict[tuple[object, object], list[StandardFieldDefinition]] = {}

    for field in fields:
        by_word.setdefault((field.word_index, field.word_label), []).append(field)

    for word, group in sorted(by_word.items(), key=lambda item: str(item[0])):
        occupied: set[int] = set()

        for field in group:
            positions = set(range(field.lsb, field.msb + 1))

            if positions & occupied:
                return None, _blocked(
                    candidate.bitfield_id, "OVERLAPPING_FIELDS",
                    f"fields overlap in word {word[0] if word[0] is not None else word[1]}")

            occupied |= positions

        # Gaps are the document's, and are left exactly as they are.

        if len(occupied) < (max(labels) - min(labels) + 1):
            warnings.append(
                f"word {word[0] if word[0] is not None else word[1]} "
                "does not cover every ruler position")

    # A gap is only a gap when nothing in the diagram claims those bits. Where
    # the word still has room and the diagram shows a name nobody positioned,
    # the definition is not describing that word, and must not say it is.

    from standard_word_association import unpositioned_labels

    structural = StructuralCompleteness.STRUCTURALLY_COMPLETE
    orphaned = [cell for cell in unpositioned_labels(payload, table_payload,
                                                     words=words)
                if str(cell["cell_id"]) not in {f.source_cell_ids[0]
                                                for f in fields}]

    # Every word the diagram shows something in, not only the words that
    # happened to yield a field. A word whose labels nobody could position
    # used to be absent from this loop, so it left no trace and the definition
    # called itself whole while a named word of it was simply missing.

    by_orphan: dict[tuple[object, object], list[Mapping[str, object]]] = {}

    for cell in orphaned:
        key = associations.get(str(cell["cell_id"]), UNRESOLVED).key
        by_orphan.setdefault(key, []).append(cell)

    unresolved_words: list[StandardUnresolvedWord] = []
    universe = sorted(set(by_word) | set(by_orphan), key=lambda item: str(item[0]))

    for word in universe:
        group = by_word.get(word, ())
        occupied: set[int] = set()

        for field in group:
            occupied |= set(range(field.lsb, field.msb + 1))

        named = by_orphan.get(word, [])

        if len(occupied) >= word_width or not named:
            continue

        structural = StructuralCompleteness.STRUCTURALLY_INCOMPLETE
        name = word[0] if word[0] is not None else word[1]
        classified = tuple(sorted(
            (unresolved_label_of(cell, ruler_labels=labels, word_width=word_width)
             for cell in named), key=lambda item: item.label))
        deferred = [item for item in classified
                    if item.cause is UnresolvedCause.VALUE_LOCAL_DEFERRED]

        # The sentence for a word of plain unpositioned labels is left exactly
        # as it was; a deferred value-local label earns its own clause rather
        # than being counted as something the diagram never positioned.

        warnings.append(
            f"word {name} has {word_width - len(occupied)} unclaimed bit(s) "
            f"and {len(named) - len(deferred)} label(s) the diagram never "
            f"positions" if len(deferred) != len(named) or not deferred else
            f"word {name} has {word_width - len(occupied)} unclaimed bit(s) "
            f"and {len(deferred)} value-local label(s) not yet projected")

        if deferred and len(deferred) != len(named):
            warnings.append(
                f"word {name} also has {len(deferred)} value-local label(s) "
                "not yet projected")

        unresolved_words.append(StandardUnresolvedWord(
            word_index=word[0], word_label=word[1],
            unpositioned_labels=tuple(sorted(
                str(cell["text"]).strip() for cell in named)),
            supporting_source_ids=tuple(sorted({
                str(value) for cell in named
                for value in cell.get("source_ids", ())})),
            field_count=len(group), unresolved_labels=classified))

    if not ruler_is_trusted(labels, word_width=word_width):
        warnings.append(RULER_FRAGMENTED)

    if any(field.word_association_source is AssociationSource.GEOMETRIC_ROW
           for field in fields) and len(by_word) > 1:

        # A claim about several words resting on horizontal bands is the very
        # thing that produced false overlaps before.

        warnings.append("word identity for a multi-word definition rests on geometry")
        unresolved.append("multi-word grouping is geometric, not stated")

    definition_id = "bfd-" + sha256_json([
        candidate.bitfield_id, corpus_fingerprint, layout_fingerprint,
        structure_fingerprint, [field.field_id for field in fields]])[:16]
    definition = StandardBitfieldDefinition(
        schema_version=SEMANTIC_SCHEMA_VERSION, definition_id=definition_id,
        standard_id=standard_id, revision=revision,
        corpus_fingerprint=corpus_fingerprint,
        layout_fingerprint=layout_fingerprint,
        structure_fingerprint=structure_fingerprint,
        semantic_model_version=SEMANTIC_MODEL_VERSION,
        source_bitfield_candidate_id=candidate.bitfield_id,
        source_table_id=candidate.table_id,
        pages=(table.page_start,) if table.page_start == table.page_end
        else tuple(range(table.page_start, table.page_end + 1)),
        bit_order=order, visible_bit_labels=tuple(labels),
        fields=tuple(fields),
        supporting_source_ids=tuple(dict.fromkeys(supporting)),
        approved_by=approval.reviewer, approved_at=approval.reviewed_at,
        approval_verdict=approval.verdict,
        provenance_status="COMPLETE",
        completeness=(DefinitionCompleteness.PARTIAL if unresolved
                      else DefinitionCompleteness.COMPLETE),
        structural_completeness=structural,
        unresolved_words=tuple(unresolved_words),
        unresolved_field_count=len(unresolved), warnings=tuple(warnings))

    return replace(definition, semantic_fingerprint=sha256_json(
        definition.to_dict())), None


def review_evidence(
    candidate_id: str, structure_fingerprint: str, *,
    table: Mapping[str, object], originals: Sequence[object],
    roles: Sequence[str], associations: Mapping[str, object],
    reviewed_links: Sequence[StandardReviewedLink],
    value_segments: Mapping[str, StandardValueSegment] | None = None,
    packed_slots: Mapping[str, StandardPackedSlot] | None = None,
) -> dict[str, object]:
    """Exactly what a reviewer decided, in one canonical form.

    It binds the candidate and the geometry it was read from, the field
    candidates in the order they were shown, the canonical text of the cell
    each one is, the role given to each, the word each was assigned, and for
    every normative link the decision plus the facts it rested on -- the link,
    its field, its requirement, its bits and its word.

    The text is the cell's canonical text from the geometry store, never a
    label derived from it. A derived label is presentation: STD2F-R cut ruler
    numbers out of one, and had this bound that instead, every approval would
    have expired the moment the derivation improved. The canonical text still
    catches a real change, because a cell's identity is a hash of it.

    Provenance grades, warnings, recommendations and rendering are outside it
    for the same reason: they move without changing what was decided, and
    expiring approvals over them teaches people to re-approve without reading.

    A candidate whose bits were projected from value coordinates also binds
    that projection, because a reviewer who accepted a 63..32 half accepted a
    transformation and not only a name. The key is written only where such a
    projection exists, so every candidate decided before STD2E hashes to
    exactly what it hashed to before.
    """
    canonical = {str(cell["cell_id"]): str(cell["text"])
                 for row in table["rows"] for cell in row["cells"]}
    segments = dict(value_segments or {})
    packed = dict(packed_slots or {})

    def shown(entry, role):
        item = {"cell": entry.cell_id,
                "text": canonical.get(entry.cell_id, entry.text),
                "role": role,
                "word": [getattr(associations.get(entry.cell_id, UNRESOLVED),
                                 "word_index", None),
                         getattr(associations.get(entry.cell_id, UNRESOLVED),
                                 "word_label", None)]}
        segment = segments.get(entry.cell_id)

        if segment is not None:
            item["value"] = [segment.declared_msb, segment.declared_lsb,
                             segment.segment_index, segment.base]

        slot = packed.get(entry.cell_id)

        if slot is not None:
            # A reviewer accepting this accepted a placement the diagram never
            # made, so the rule that made it is part of what was decided.

            item["packed"] = [slot.declared_msb, slot.declared_lsb,
                              slot.msb, slot.lsb, slot.slot.value,
                              slot.packing_group_id, slot.authority,
                              slot.source_id, slot.cross_reference]

        return item

    return {
        "model": SEMANTIC_MODEL_VERSION,
        "approval_schema": APPROVAL_SCHEMA_VERSION,
        "candidate": candidate_id,
        "structure": structure_fingerprint,
        "candidates": [shown(entry, role)
                       for entry, role in zip(originals, roles)],
        "links": [list(item.bound) + [item.accepted, item.role]
                  for item in sorted(reviewed_links,
                                     key=lambda item: item.link_fingerprint)],
    }


def review_evidence_fingerprint(*args, **kwargs) -> str:
    """The hash of the one canonical evidence payload."""
    return sha256_json(review_evidence(*args, **kwargs))


def packed_slots_for(candidate, table, units: Sequence[object], *,
                     word_width: int = DEFAULT_WORD_WIDTH,
                     ) -> dict[str, StandardPackedSlot]:
    """Where a word's two packed 16-bit quantities sit, when a rule says so.

    The sections consulted are the ones the diagram's own cells cite, and only
    those. Where more than one applies they must agree; a disagreement is the
    document contradicting itself and is never resolved by preferring one.

    Takes either the structure objects or the payloads they serialize to, so
    promotion, approval counting and serving can each ask in the form they
    already hold.
    """
    units = tuple(units)

    if not units:
        return {}

    payload = candidate.to_dict() if hasattr(candidate, "to_dict") else candidate
    table_payload = table.to_dict() if hasattr(table, "to_dict") else table
    associations = {cell_id: item.word_index
                    for cell_id, item in associate_words(table_payload).items()}
    by_id = {getattr(unit, "source_id", ""): unit for unit in units}
    cited = {value for row in table_payload["rows"] for cell in row["cells"]
             for value in cell.get("source_ids", ())}
    sections = sorted({getattr(by_id[value], "section", None) for value in cited
                       if value in by_id and getattr(by_id[value], "section", None)})
    found: dict[str, StandardPackedSlot] = {}

    for section in sections:
        projected, _refusals = packed_slot_projection(
            payload, table_payload, units, associations=associations,
            section=section, word_width=word_width)

        for cell_id, slot in projected.items():
            previous = found.get(cell_id)

            if previous is not None and (previous.msb, previous.lsb) != (
                    slot.msb, slot.lsb):

                # Two applicable sections place the same quantity differently.

                return {}

            found.setdefault(cell_id, slot)

    return found


def normative_links_for(candidate, table, units: Sequence[object], *,
                        layout: Mapping[str, object] | None = None,
                        word_width: int = DEFAULT_WORD_WIDTH,
                        ) -> dict[str, StandardNormativeRangeLink]:
    """The rules that position a field this diagram names but never places."""
    from standard_word_association import unpositioned_labels

    units = tuple(units)

    if not units:
        return {}

    payload, table_payload = candidate.to_dict(), table.to_dict()
    words = page_words(layout, candidate.page)

    # A label a proven word pair already positions is not waiting for a rule to
    # position it, so it is not offered as a target. Leaving it in would let a
    # rule and a projection claim the same cell.

    pairs, _ = value_local_pairs(payload, table_payload, word_width=word_width)
    paired = {cell_id for pair in pairs for cell_id in pair.cell_ids}
    orphans = [cell for cell in unpositioned_labels(payload, table_payload,
                                                    words=words)
               if str(cell["cell_id"]) not in paired]

    if not orphans:
        return {}

    associations = associate_words(table_payload)
    targets = [ProseFieldTarget(
        cell_id=str(cell["cell_id"]), label=str(cell["text"]).strip(),
        word_index=associations.get(str(cell["cell_id"]), UNRESOLVED).word_index,
        source_ids=tuple(str(value) for value in cell.get("source_ids", ())),
        provenance=str(cell["provenance"])) for cell in orphans]

    # The section the diagram itself cites. A rule from elsewhere is not
    # talking about this structure just because it uses the same word.

    cited = {value for row in table_payload["rows"] for cell in row["cells"]
             for value in cell.get("source_ids", ())}
    by_id = {getattr(unit, "source_id", ""): unit for unit in units}
    sections = frozenset(
        getattr(by_id[value], "section", None) for value in cited
        if value in by_id and getattr(by_id[value], "section", None))
    diagram_words = sorted({item.word_index for item in targets
                            if item.word_index is not None}
                           | {association.word_index
                              for association in associations.values()
                              if association.word_index is not None})
    linked, _ = normative_range_links(
        targets, units, candidate_id=candidate.bitfield_id, sections=sections,
        diagram_words=diagram_words, word_width=word_width)

    return {item.target_cell_id: item for item in linked}


def build_semantics(
    structures, approvals: Mapping[str, StandardBitfieldApproval], *,
    standard_id: str, revision: str, corpus_fingerprint: str,
    layout_fingerprint: str, structure_fingerprint: str,
    layout: Mapping[str, object] | None = None,
    units: Sequence[object] = (),
) -> StandardSemanticSet:
    """Promote every candidate a person approved, and record why the rest were not."""
    tables = {table.table_id: table for table in structures.tables}
    definitions: list[StandardBitfieldDefinition] = []
    blocked: list[dict[str, object]] = []

    for candidate in structures.bitfields:
        table = tables.get(candidate.table_id)

        if table is None:
            blocked.append(_blocked(candidate.bitfield_id, "SOURCE_TABLE_MISSING",
                                    "the candidate names a table that is not present"))
            continue

        definition, reason = promote_bitfield(
            candidate, table, approvals.get(candidate.bitfield_id),
            standard_id=standard_id, revision=revision,
            corpus_fingerprint=corpus_fingerprint,
            layout_fingerprint=layout_fingerprint,
            structure_fingerprint=structure_fingerprint,
            words=page_words(layout, candidate.page),
            prose_links=normative_links_for(candidate, table, units,
                                            layout=layout),
            packed_slots=packed_slots_for(candidate, table, units))

        if definition is not None:
            definitions.append(definition)
        else:
            blocked.append(reason)

    packets: list[StandardPacketDefinition] = []
    grouped: dict[str, list[StandardBitfieldDefinition]] = {}

    for definition in definitions:
        approval = approvals[definition.source_bitfield_candidate_id]

        if approval.packet_identity:
            grouped.setdefault(approval.packet_identity, []).append(definition)

    for identity, members in sorted(grouped.items()):
        approval = approvals[members[0].source_bitfield_candidate_id]
        packets.append(StandardPacketDefinition(
            schema_version=SEMANTIC_SCHEMA_VERSION,
            packet_id="pkt-" + sha256_json(
                [standard_id, revision, structure_fingerprint, identity,
                 [item.definition_id for item in members]])[:16],
            standard_id=standard_id, revision=revision, identity=identity,
            bitfield_definition_ids=tuple(item.definition_id for item in members),
            supporting_source_ids=tuple(dict.fromkeys(
                value for item in members for value in item.supporting_source_ids)),
            approved_by=approval.reviewer, approved_at=approval.reviewed_at))

    return StandardSemanticSet(tuple(definitions), tuple(packets), tuple(blocked))


def validate_semantics(semantics: StandardSemanticSet, *, structures,
                       source_ids: frozenset[str], corpus_fingerprint: str,
                       layout_fingerprint: str, structure_fingerprint: str) -> None:
    """Structural integrity of the definitions. Not implementation compliance."""
    candidates = {item.bitfield_id for item in structures.bitfields}
    known = {item.definition_id for item in semantics.bitfields}

    for definition in semantics.bitfields:
        for value, expected, name in (
                (definition.corpus_fingerprint, corpus_fingerprint, "corpus"),
                (definition.layout_fingerprint, layout_fingerprint, "layout"),
                (definition.structure_fingerprint, structure_fingerprint, "structure")):
            if value != expected:
                raise StandardSemanticError(f"definition pins a stale {name} fingerprint")

        if definition.source_bitfield_candidate_id not in candidates:
            raise StandardSemanticError("definition names an unknown candidate")

        if definition.bit_order is BitOrder.UNKNOWN:
            raise StandardSemanticError("definition has no bit order")

        if ASSISTANT_IDENTITY.match(definition.approved_by.strip()):
            raise StandardSemanticError("definition was approved by an assistant")

        ruler = set(definition.visible_bit_labels)
        seen: dict[tuple[object, object], set[int]] = {}

        for field in definition.fields:
            if field.width != abs(field.msb - field.lsb) + 1:
                raise StandardSemanticError("field width does not match its bounds")

            if set(field.covered_bit_labels) != set(range(field.lsb, field.msb + 1)):
                raise StandardSemanticError("field bounds do not match its labels")

            if field.position_source is PositionSource.UNKNOWN:
                raise StandardSemanticError("field has no position source")

            if (field.position_source is PositionSource.GEOMETRIC_RULER
                    and not set(field.covered_bit_labels) <= ruler):

                # A measured position that is not on the ruler was measured
                # against nothing. A stated position answers to the text, not
                # to a ruler the extractor may only have seen a piece of.

                raise StandardSemanticError("field lies outside the ruler")

            if (field.position_source is PositionSource.STATED_RANGE
                    and not field.stated_range_text):
                raise StandardSemanticError("stated field does not name its range")

            if not set(field.supporting_source_ids) <= source_ids:
                raise StandardSemanticError("field cites an unknown canonical source")

            if not field.supporting_source_ids:
                raise StandardSemanticError("field has no canonical source")

            if ProvenanceQuality.UNRESOLVED.value in field.provenance_grades:
                raise StandardSemanticError("field rests on unresolved provenance")

            if field.word_association_source is AssociationSource.UNKNOWN:
                raise StandardSemanticError("field belongs to no resolvable word")

            word = (field.word_index, field.word_label)
            positions = set(range(field.lsb, field.msb + 1))

            if positions & seen.setdefault(word, set()):
                raise StandardSemanticError("fields overlap within a word")

            seen[word] |= positions

    for packet in semantics.packets:
        if not set(packet.bitfield_definition_ids) <= known:
            raise StandardSemanticError("packet names an unknown definition")

        if not set(packet.supporting_source_ids) <= source_ids:
            raise StandardSemanticError("packet cites an unknown canonical source")
