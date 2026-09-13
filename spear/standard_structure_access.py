"""Read-only access to the reviewed structures a person actually approved.

Everything upstream of this file proposes; a human disposes; this file hands
back what was disposed of, and nothing else. It never builds a definition, so
an unapproved candidate has no route through here however it is asked for, and
it never fills a gap, so a structure that describes half a word says so.

A definition is served only while the evidence behind it still holds: the
corpus, layout and geometry it was approved against, the approval itself, the
evidence that approval recorded, and every normative link that approval
accepted. Any of those moving is a refusal with a name, never a quieter answer.
"""

from __future__ import annotations

import re

from typing import Mapping, Sequence

from standard_semantic import (
    APPROVING_VERDICTS, DEFAULT_WORD_WIDTH, StandardSemanticError,
    review_evidence_fingerprint,
)
from standard_semantic_store import StandardApprovalStore, StandardSemanticStore
from standard_value_pair import value_local_pairs
from standard_store import StandardStoreError
from standard_structure_store import StandardStructureStore
from standard_word_association import (
    UNRESOLVED, associate_words, field_candidates, unpositioned_labels,
)

# Machine-readable refusals. A caller that cannot have a structure is told
# which fact stopped it, so it can decide rather than guess.

STRUCTURE_NOT_FOUND = "STRUCTURE_NOT_FOUND"
STRUCTURE_NOT_APPROVED = "STRUCTURE_NOT_APPROVED"
STALE_SEMANTIC_STORE = "STALE_SEMANTIC_STORE"
STALE_APPROVAL = "STALE_APPROVAL"
STALE_REVIEW_EVIDENCE = "STALE_REVIEW_EVIDENCE"
SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
AMBIGUOUS_STRUCTURE = "AMBIGUOUS_STRUCTURE"

# An identifier that names nothing here. Distinct from STRUCTURE_NOT_FOUND,
# which answers a well-formed question about a revision; this one says the
# question itself cannot be asked that way, and says how to ask it.

INVALID_DEFINITION_ID = "INVALID_DEFINITION_ID"

# A well-formed identifier: the shape the tool itself hands out.

_DEFINITION_ID = re.compile(r"^bfd-[0-9a-f]{16}$")
_CANDIDATE_ID = re.compile(r"^bit-[0-9a-f]{16}$")
STRUCTURE_INCOMPLETE = "STRUCTURE_INCOMPLETE"

# A short label is what a reviewer read; a paragraph is what standard.fetch is
# for, under its own controls.

_MAX_LABEL_CHARS = 80


class StructureAccessError(RuntimeError):
    """A refusal carrying the reason a caller should act on."""

    def __init__(self, reason: str, detail: str = "",
                 recovery: Mapping[str, object] | None = None) -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail or reason
        self.recovery = dict(recovery or {})

    def to_dict(self) -> dict[str, object]:
        found = {"error": self.reason, "detail": self.detail}

        if self.recovery:
            found["recovery"] = self.recovery

        return found


def _ranges(bits: Sequence[int]) -> list[list[int]]:
    """Contiguous runs as [msb, lsb] pairs, high first, for a compact answer."""
    ordered = sorted(set(int(value) for value in bits))
    runs: list[list[int]] = []

    for value in ordered:
        if runs and value == runs[-1][1] + 1:
            runs[-1][1] = value
        else:
            runs.append([value, value])

    return [[high, low] for low, high in reversed(runs)]


def _field(field: Mapping[str, object]) -> dict[str, object]:
    """One field, with its two provenance paths kept apart.

    `canonical_source_ids` cite the cell the label was read from. A position
    taken from a rule elsewhere also names that rule, separately, so a reader
    can tell where the bits came from rather than inheriting one flat list.
    """
    return {
        "field_id": field["field_id"],
        "word_index": field["word_index"],
        "word_label": field.get("word_label"),
        "word_association_source": field["word_association_source"],
        "normative_label": str(field["label"])[:_MAX_LABEL_CHARS],
        "display_label": str(field["display_label"])[:_MAX_LABEL_CHARS],
        "semantic_role": field["semantic_role"],
        # The physical word is what coverage is computed in and what a
        # decoder needs. `declared_*` is what the label literally says; for a
        # projected half of a wider value the two differ, and a caller must
        # never be able to read one as the other.
        "msb": field["msb"], "lsb": field["lsb"], "width": field["width"],
        "coordinate_domain": field.get("coordinate_domain", "WORD_LOCAL"),
        "declared_msb": field.get("declared_msb", field["msb"]),
        "declared_lsb": field.get("declared_lsb", field["lsb"]),
        "value_group_id": field.get("value_group_id"),
        "value_width": field.get("value_width"),
        "segment_index": field.get("segment_index"),
        "segment_count": field.get("segment_count"),
        # A packing group shares a word; a value group composes one value. A
        # caller must be able to tell them apart without reading names.
        "packing_group_id": field.get("packing_group_id"),
        "packing_slot": field.get("packing_slot"),
        "projection_source": field.get("projection_source"),
        "projection_reference": field.get("projection_reference"),
        "position_source": field["position_source"],
        "stated_range_text": field.get("stated_range_text"),
        "provenance_grades": list(field["provenance_grades"]),
        "source_cell_ids": list(field["source_cell_ids"]),
        "canonical_source_ids": list(field["supporting_source_ids"]),
        "normative_source_id": field.get("normative_source_id"),
        "normative_link_fingerprint": field.get("normative_link_fingerprint"),
        "warnings": list(field.get("warnings", ())),
    }


def _packing_groups(fields: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Every physical word shared by independent quantities, high slot first."""
    groups: dict[str, list[Mapping[str, object]]] = {}

    for field in fields:
        group = field.get("packing_group_id")

        if group:
            groups.setdefault(str(group), []).append(field)

    found = []

    for group, members in sorted(groups.items()):
        members = sorted(members, key=lambda f: f.get("packing_slot") or "")
        found.append({
            "semantic_kind": "INDEPENDENT_VALUES_SHARED_CONTAINER",
            "interpretation": "these members are INDEPENDENT semantic values that "
                              "share one physical container; physical_width is the "
                              "container, not a combined value; do not concatenate "
                              "them",
            "packing_group_id": group,
            "physical_word_index": members[0]["word_index"],
            "physical_width": 32,
            "member_field_ids": [m["field_id"] for m in members],
            "projection_source": next(
                (m.get("projection_source") for m in members
                 if m.get("projection_source")), None),
            "projection_reference": next(
                (m.get("projection_reference") for m in members
                 if m.get("projection_reference")), None),
            "members": [{"field_id": m["field_id"],
                         "packing_slot": m.get("packing_slot"),
                         "declared_msb": m.get("declared_msb"),
                         "declared_lsb": m.get("declared_lsb"),
                         "msb": m["msb"], "lsb": m["lsb"],
                         "value_width": m.get("value_width"),
                         "normative_source_id": m.get("normative_source_id"),
                         "coordinate_domain": m.get("coordinate_domain",
                                                    "WORD_LOCAL")}
                        for m in members]})

    return found


def _packed_slots(store, standard_id: str, revision: str, bitfield, table):
    """The packed-slot placement, asked of the one helper the store also uses."""
    from standard_semantic_store import _packed_slots as ask

    return ask(store, standard_id, revision, bitfield, table)


def _value_groups(fields: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Every multi-word value, with its segments in most-significant order."""
    groups: dict[str, list[Mapping[str, object]]] = {}

    for field in fields:
        group = field.get("value_group_id")

        if group:
            groups.setdefault(str(group), []).append(field)

    found = []

    for group, members in sorted(groups.items()):
        members = sorted(members, key=lambda f: f.get("segment_index") or 0)
        found.append({
            # Said out loud at serving time, because a caller that only reads
            # the id has no way to know a group of segments is one quantity.
            "semantic_kind": "COMPOSITE_VALUE",
            "interpretation": "these members are segments of ONE semantic value "
                              "split across containers; do not describe them as "
                              "independent values",
            "value_group_id": group,
            "value_width": members[0].get("value_width"),
            "segment_count": members[0].get("segment_count"),
            "projection_source": next(
                (m.get("projection_source") for m in members
                 if m.get("projection_source")), None),
            "segments": [{"field_id": m["field_id"],
                          "segment_index": m.get("segment_index"),
                          "word_index": m["word_index"],
                          "declared_msb": m.get("declared_msb"),
                          "declared_lsb": m.get("declared_lsb"),
                          "msb": m["msb"], "lsb": m["lsb"],
                          "coordinate_domain": m.get("coordinate_domain",
                                                     "WORD_LOCAL")}
                         for m in members]})

    return found


class StandardStructureAccess:
    """Serves persisted definitions, and only while their evidence holds."""

    def __init__(self, store) -> None:
        self.store = store
        self.semantics = StandardSemanticStore(store)
        self.approvals = StandardApprovalStore(store)
        self.structures = StandardStructureStore(store)

    # -- loading ----------------------------------------------------------

    def _load(self, standard_id: str, revision: str):
        try:
            geometry, geometry_payload = self.structures.load(standard_id, revision)
            accepted, refused = self.approvals.load(standard_id, revision)
        except (StandardSemanticError, StandardStoreError, FileNotFoundError) as exc:
            raise StructureAccessError(STALE_SEMANTIC_STORE, str(exc)) from exc

        try:
            manifest, payload = self.semantics.load(standard_id, revision)
        except StandardSemanticError as exc:
            # Never approving anything is a state, not a fault. Only a store
            # that exists and no longer matches its inputs is stale.

            if "no semantic definitions" in str(exc):
                manifest, payload = None, {"bitfields": []}
            else:
                raise StructureAccessError(STALE_SEMANTIC_STORE, str(exc)) from exc
        except (StandardStoreError, FileNotFoundError) as exc:
            raise StructureAccessError(STALE_SEMANTIC_STORE, str(exc)) from exc

        return manifest, payload, geometry, geometry_payload, accepted, refused

    def index(self, standard_id: str, revision: str) -> dict[str, object]:
        """What can be asked for, by identity. No fields, no inference."""
        manifest, payload, _, _, accepted, _ = self._load(standard_id, revision)

        return {
            "standard_id": standard_id, "revision": revision,
            "semantic_fingerprint": (manifest.semantic_fingerprint
                                     if manifest is not None else None),
            "approved_structure_count": len(payload["bitfields"]),
            "structures": sorted(
                ({"definition_id": item["definition_id"],
                  "source_candidate_id": item["source_bitfield_candidate_id"],
                  "pages": list(item["pages"]),
                  "completeness": item["completeness"],
                  "structural_completeness": item["structural_completeness"],
                  "field_count": len(item["fields"])}
                 for item in payload["bitfields"]
                 if item["source_bitfield_candidate_id"] in accepted),
                key=lambda item: item["definition_id"]),
        }

    # -- one structure ----------------------------------------------------

    def get(self, standard_id: str, revision: str, *,
            definition_id: str | None = None,
            source_candidate_id: str | None = None,
            require_structurally_complete: bool = False,
            word_width: int = DEFAULT_WORD_WIDTH) -> dict[str, object]:
        manifest, payload, geometry, geometry_payload, accepted, refused = self._load(
            standard_id, revision)
        found = [item for item in payload["bitfields"]
                 if (definition_id is None
                     or item["definition_id"] == definition_id)
                 and (source_candidate_id is None
                      or item["source_bitfield_candidate_id"] == source_candidate_id)]

        if not found:
            approved = [(item["definition_id"], item["source_bitfield_candidate_id"])
                        for item in payload["bitfields"]
                        if item["source_bitfield_candidate_id"] in accepted]

            # A real definition paired with the wrong candidate, or a real
            # candidate paired with the wrong definition. This used to fall
            # through to "no human has approved it", which was false -- the
            # candidate WAS approved, for another definition -- and a caller
            # told that tried every other candidate against the same
            # definition, one refusal at a time, until its budget was gone.
            # The pairing is not a secret: say which one it is.

            by_definition = dict(approved)
            by_candidate = {candidate: definition
                            for definition, candidate in approved}
            partner = None

            if definition_id in by_definition and source_candidate_id is not None:
                partner = {"definition_id": definition_id,
                           "source_candidate_id": by_definition[definition_id]}
            elif source_candidate_id in by_candidate and definition_id is not None:
                partner = {"definition_id": by_candidate[source_candidate_id],
                           "source_candidate_id": source_candidate_id}

            if partner is not None:
                raise StructureAccessError(
                    STRUCTURE_NOT_FOUND,
                    "those two identifiers name different approved structures; "
                    "the pairing below is the one that exists",
                    recovery={"approved_pairing": partner,
                              "how": "call standard.get_structure with that "
                                     "pairing, or with the definition_id alone"})

            # A candidate nobody reviewed is a different answer from a name
            # that means nothing here, and the caller can act on the difference.

            known = {item["bitfield_id"]
                     for item in geometry_payload.get("bitfields", ())}

            if source_candidate_id in known:
                raise StructureAccessError(
                    STRUCTURE_NOT_APPROVED,
                    "that candidate exists but no human has approved it")

            # Nothing is guessed at, matched by name or read as a near miss. The
            # refusal instead carries what a valid identifier is and where to
            # get one, because a caller that has to guess will keep guessing.
            # A well-formed identifier that names nothing is a real answer
            # about this revision. An identifier of the wrong shape or kind is
            # a different fault, and the caller can act on the difference.

            malformed = [value for value, pattern
                         in ((definition_id, _DEFINITION_ID),
                             (source_candidate_id, _CANDIDATE_ID))
                         if value is not None and not pattern.fullmatch(value)]
            detail = "no approved structure has that identity in this revision"
            reason = STRUCTURE_NOT_FOUND

            if malformed:
                reason = INVALID_DEFINITION_ID
                asked = malformed[0]

                if asked.startswith("std-"):
                    detail = ("that is a canonical source id, not a structure "
                              "identifier; source ids belong to standard.fetch "
                              "and standard.cite")
                else:
                    detail = ("that is not a structure identifier; a "
                              "definition_id is 'bfd-' and 16 hex digits, a "
                              "source_candidate_id is 'bit-' and 16 hex digits. "
                              "A field name, wildcard or section number is "
                              "neither")

            raise StructureAccessError(
                reason, detail,
                recovery={
                    "how": "call standard.get_structure with no arguments to "
                           "list every approved structure and its identifiers",
                    "definition_id_form": "bfd-<16 hex>",
                    "source_candidate_id_form": "bit-<16 hex>",
                    # As pairs. Two flat lists of nine each made an 81-cell
                    # grid to guess in; the pairs are the nine cells that
                    # exist, and a definition_id alone is always enough.
                    "approved_structures": [
                        {"definition_id": definition,
                         "source_candidate_id": candidate}
                        for definition, candidate in sorted(approved)],
                    "valid_definition_ids": sorted(
                        str(item["definition_id"])
                        for item in payload["bitfields"]
                        if item["source_bitfield_candidate_id"] in accepted),
                    "valid_source_candidate_ids": sorted(
                        str(item["source_bitfield_candidate_id"])
                        for item in payload["bitfields"]
                        if item["source_bitfield_candidate_id"] in accepted),
                })

        if len(found) > 1:
            raise StructureAccessError(
                AMBIGUOUS_STRUCTURE,
                f"{len(found)} approved structures match; name one identity")

        definition = found[0]
        candidate = definition["source_bitfield_candidate_id"]

        # -- the approval this definition rests on, and its evidence -------

        if any(item.get("candidate_id") == candidate for item in refused):
            raise StructureAccessError(
                STALE_APPROVAL,
                "the approval behind this structure was refused as stale")

        approval = accepted.get(candidate)

        if approval is None or not approval.approves:
            raise StructureAccessError(
                STRUCTURE_NOT_APPROVED,
                "this structure has no current human approval")

        if definition["structure_fingerprint"] != geometry.structure_fingerprint:
            raise StructureAccessError(
                STALE_SEMANTIC_STORE,
                "the structure was built against other geometry")

        table, bitfield = self._geometry_of(geometry_payload, candidate)

        if table is None:
            raise StructureAccessError(
                SOURCE_UNAVAILABLE, "the geometry this structure came from is gone")

        if approval.review_evidence_fingerprint is not None:
            pairs, _ = value_local_pairs(bitfield, table)
            segments = {cell_id: segment for pair in pairs
                        for cell_id, segment in pair.by_cell().items()}
            originals = field_candidates(
                bitfield, table, value_local_cells=frozenset(segments))
            packed = _packed_slots(self.store, standard_id, revision,
                                   bitfield, table)
            current = review_evidence_fingerprint(
                candidate, geometry.structure_fingerprint, table=table,
                originals=originals, roles=approval.span_roles,
                associations=associate_words(table),
                reviewed_links=approval.reviewed_links, value_segments=segments,
                packed_slots=packed)

            if current != approval.review_evidence_fingerprint:
                raise StructureAccessError(
                    STALE_REVIEW_EVIDENCE,
                    "what the reviewer was shown is not what this candidate is now")

        self._check_links(definition, approval)

        if (require_structurally_complete
                and definition["structural_completeness"] != "STRUCTURALLY_COMPLETE"):
            raise StructureAccessError(
                STRUCTURE_INCOMPLETE,
                "this structure does not describe every word it covers")

        return self._render(definition, approval, manifest, bitfield, table,
                            word_width=word_width)

    @staticmethod
    def _geometry_of(payload: Mapping[str, object], candidate: str):
        bitfield = next((item for item in payload.get("bitfields", ())
                         if item["bitfield_id"] == candidate), None)

        if bitfield is None:
            return None, None

        table = next((item for item in payload["tables"]
                      if item["table_id"] == bitfield["table_id"]), None)

        return table, bitfield

    @staticmethod
    def _check_links(definition: Mapping[str, object], approval) -> None:
        """A prose-positioned field keeps its bits only while its link is accepted."""
        allowed = {item.link_fingerprint for item in approval.accepted_links}

        for field in definition["fields"]:
            fingerprint = field.get("normative_link_fingerprint")

            if fingerprint and fingerprint not in allowed:
                raise StructureAccessError(
                    STALE_APPROVAL,
                    f"the normative link positioning {field['display_label']!r} "
                    "is no longer accepted by the approval")

    # -- rendering --------------------------------------------------------

    def _render(self, definition: Mapping[str, object], approval,
                manifest, bitfield: Mapping[str, object],
                table: Mapping[str, object], *, word_width: int
                ) -> dict[str, object]:
        associations = associate_words(table)
        promoted = {value for field in definition["fields"]
                    for value in field["source_cell_ids"]}
        orphans: dict[object, list[str]] = {}

        for cell in unpositioned_labels(bitfield, table):
            if str(cell["cell_id"]) in promoted:
                continue

            key = associations.get(str(cell["cell_id"]), UNRESOLVED).word_index
            orphans.setdefault(key, []).append(
                str(cell["text"]).strip()[:_MAX_LABEL_CHARS])

        # Every word the definition knows about, including one the diagram
        # names and nothing positions. Such a word carries no fields and is
        # still reported, because hiding it is how a structure comes to look
        # whole while a word of it is missing.

        unresolved = {item["word_index"]: item
                      for item in definition.get("unresolved_words", ())}
        words = []

        for index in sorted(
                {field["word_index"] for field in definition["fields"]}
                | set(unresolved),
                key=lambda value: (value is None, value)):
            members = [f for f in definition["fields"] if f["word_index"] == index]
            covered: set[int] = set()

            for field in members:
                covered |= set(range(field["lsb"], field["msb"] + 1))

            unclaimed = set(range(word_width)) - covered
            named = sorted(set(orphans.get(index, ()))
                           | set(unresolved.get(index, {})
                                 .get("unpositioned_labels", ())))
            words.append({
                "word_index": index,
                "word_label": (members[0].get("word_label") if members
                               else unresolved.get(index, {}).get("word_label")),
                "word_width": word_width,
                "field_count": len(members),
                "covered_bits": _ranges(covered),
                "unclaimed_bits": _ranges(unclaimed),
                # Named on the page, positioned by nothing. Not fields, and
                # deliberately not turned into any.
                "unpositioned_labels": [value[:_MAX_LABEL_CHARS] for value in named],
                # Why each of those labels has no position, so a caller never
                # has to parse a label string to tell "not understood" from
                # "understood and not yet projected".
                "unresolved_labels": [
                    {"label": str(item["label"])[:_MAX_LABEL_CHARS],
                     "cause": item["cause"],
                     "declared_msb": item.get("declared_msb"),
                     "declared_lsb": item.get("declared_lsb")}
                    for item in unresolved.get(index, {}).get(
                        "unresolved_labels", ())],
                "unresolved_causes": list(
                    unresolved.get(index, {}).get("causes", ())),
                "structural_status": "unresolved" if named else "resolved",
            })

        citation_ids = sorted({value for field in definition["fields"]
                               for value in field["supporting_source_ids"]}
                              | {value for field in definition["fields"]
                                 if field.get("normative_source_id")
                                 for value in (field["normative_source_id"],)})

        return {
            "definition_id": definition["definition_id"],
            "source_candidate_id": definition["source_bitfield_candidate_id"],
            "standard_id": definition["standard_id"],
            "revision": definition["revision"],
            "pages": list(definition["pages"]),
            "bit_order": definition["bit_order"],
            "completeness": definition["completeness"],
            "structural_completeness": definition["structural_completeness"],
            "human_verdict": definition["approval_verdict"],
            "reviewer": definition["approved_by"],
            "reviewed_at": definition["approved_at"],
            "warnings": list(definition["warnings"]),
            "definition_fingerprint": definition["semantic_fingerprint"],
            "semantic_model_version": definition["semantic_model_version"],
            "semantic_fingerprint": (manifest.semantic_fingerprint
                                     if manifest is not None else None),
            "corpus_fingerprint": definition["corpus_fingerprint"],
            "layout_fingerprint": definition["layout_fingerprint"],
            "structure_fingerprint": definition["structure_fingerprint"],
            # One entry per wider value the diagram splits over words, so the
            # halves can be found from each other without matching on names.
            "value_groups": _value_groups(definition["fields"]),
            # One word holding several independent quantities. Never a
            # composite number: each member keeps its own width.
            "packing_groups": _packing_groups(definition["fields"]),
            "word_count": len(words),
            "unresolved_word_count": len(unresolved),
            "field_count": len(definition["fields"]),
            "words": words,
            "fields": [_field(field) for field in sorted(
                definition["fields"],
                key=lambda f: ((f["word_index"] is None), f["word_index"], -f["msb"]))],
            "citation_source_ids": citation_ids,
        }
