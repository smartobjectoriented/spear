"""Operator-only review of derived table geometry.

Geometry that an algorithm is confident about is still not geometry a person has
looked at. This walks the candidates, shows each as a text grid with its
warnings and the canonical sources behind it, and records a human verdict.

Nothing here is registered as a model tool, and no structure is rewritten by it:
the review records verdicts beside the structures, never over them.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from state_paths import standards_root
from standard_schema import canonical_json
from standard_store import StandardStore, StandardStoreError
from standard_structure import StandardStructureError
from standard_structure_store import StandardStructureStore

REVIEW_FILE = "review.json"
REVIEW_SCHEMA_VERSION = 1
UNREVIEWED = "UNREVIEWED"
HUMAN_VERDICTS = ("PASS", "ACCEPTABLE_WARNING", "FAIL", "NEEDS_FOLLOWUP")
BLOCKING_VERDICTS = ("FAIL", "NEEDS_FOLLOWUP")

# Judged independently of the single verdict, so a table can be a correct region
# with wrong columns, or right columns with provenance that is not good enough.

REVIEW_DIMENSIONS = ("geometry_correct", "row_structure_correct",
                     "column_structure_correct", "provenance_sufficient",
                     "bitfield_candidate_correct")
_VERDICT_KEYS = {"p": "PASS", "w": "ACCEPTABLE_WARNING",
                 "f": "FAIL", "n": "NEEDS_FOLLOWUP"}
_MAX_CELL_CHARS = 22
_MAX_GRID_ROWS = 14

_HELP = """
  p  PASS                 w  ACCEPTABLE_WARNING
  f  FAIL                 n  NEEDS_FOLLOWUP
  s  skip                 u  clear this verdict
  b  previous             v  write an SVG of this table's geometry
  q  save and quit        ?  this help
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_reviewer() -> str:
    for value in (os.environ.get("SPEAR_OPERATOR"), os.environ.get("USER")):
        if value and value.strip():
            return value.strip()

    return "operator"


def sample_tables(tables: Sequence[Mapping[str, object]], *,
                  per_stratum: int = 4) -> tuple[str, ...]:
    """A stratified spread of candidates rather than the first N on page one."""

    def rows(table):
        return int(table.get("row_count", 0))

    strata = (
        ("bitfield_like", lambda t: t.get("_bitfield")),
        ("multi_page", lambda t: bool(t.get("continuation_ids"))),
        ("no_caption", lambda t: not t.get("caption_source_ids")),
        ("large_table", lambda t: rows(t) >= 12),
        ("simple_table", lambda t: rows(t) <= 4 and not t.get("warnings")),
        ("wide_table", lambda t: int(t.get("column_count", 0)) >= 5),
        ("irregular_geometry", lambda t: "irregular_rows" in t.get("warnings", ())),
        ("unstable_columns", lambda t: "unstable_columns" in t.get("warnings", ())),
        ("ambiguous_span", lambda t: "ambiguous_cell_span" in t.get("warnings", ())),
        ("missing_header", lambda t: "missing_header" in t.get("warnings", ())),
        ("partial_provenance",
         lambda t: "source_mapping_partial" in t.get("warnings", ())),
        ("clean", lambda t: not t.get("warnings")),
    )
    chosen: list[str] = []

    for _, predicate in strata:
        taken = 0

        for table in tables:
            if taken >= per_stratum:
                break

            if table["table_id"] in chosen or not predicate(table):
                continue

            chosen.append(table["table_id"])
            taken += 1

    return tuple(chosen)


class StandardStructureReview:
    def __init__(self, store: StandardStore, standard_id: str, revision: str, *,
                 reviewer: str | None = None) -> None:
        self.store = store
        self.standard_id = standard_id
        self.revision = revision
        self.reviewer = reviewer or default_reviewer()
        self.structures_store = StandardStructureStore(store)
        self.manifest, payload = self.structures_store.load(standard_id, revision)
        self.tables = list(payload["tables"])
        self.bitfields = {item["table_id"]: item
                          for item in payload.get("bitfields", ())}

        for table in self.tables:
            table["_bitfield"] = table["table_id"] in self.bitfields

        self.path = (self.structures_store.directory(standard_id, revision)
                     / REVIEW_FILE)
        self.verdicts = self._load_verdicts()
        self._layout: Mapping[str, object] | None = None
        self._links: dict[str, Mapping[str, object]] = {}
        self._structures = None

    def normative_links(self, table_id: str) -> Mapping[str, object]:
        """Rules that position a field this diagram names but never places."""

        if table_id in self._links:
            return self._links[table_id]

        found: Mapping[str, object] = {}

        try:
            from standard_commands import load_structure_set
            from standard_semantic import normative_links_for

            if self._structures is None:
                self._structures = load_structure_set(
                    self.store, self.standard_id, self.revision)

            manifest, structures, _, layout = self._structures
            candidate = next((item for item in structures.bitfields
                              if item.table_id == table_id), None)
            table = next((item for item in structures.tables
                          if item.table_id == table_id), None)

            if candidate is not None and table is not None:
                found = normative_links_for(
                    candidate, table,
                    self.store.load_units(self.standard_id, self.revision),
                    layout=layout)
        except Exception:
            # Advisory only: a panel must still render without the links.

            found = {}

        self._links[table_id] = found

        return found

    def layout(self) -> Mapping[str, object] | None:
        """The pinned layout artifact, read once, for raw word geometry."""

        if self._layout is None:
            path = (self.store.revision_dir(self.standard_id, self.revision)
                    / "layout.json")

            if not path.is_file() or path.is_symlink():
                return None

            self._layout = json.loads(path.read_text("utf-8"))

        return self._layout

    def _load_verdicts(self) -> dict[str, dict[str, object]]:
        if not self.path.is_file() or self.path.is_symlink():
            return {}

        raw = json.loads(self.path.read_text("utf-8"))

        if raw.get("structure_fingerprint") != self.manifest.structure_fingerprint:
            raise StandardStructureError(
                "the recorded review belongs to different structures")

        return {row["table_id"]: row for row in raw.get("rows", ())}

    # -- selection --------------------------------------------------------

    def selection(self, *, only_unreviewed: bool = False, sample: bool = False,
                  bitfields: bool = False, provenance_sufficient: bool = False,
                  page: int | None = None) -> tuple[str, ...]:
        ids = (sample_tables(self.tables) if sample
               else tuple(table["table_id"] for table in self.tables))

        if bitfields:
            ids = tuple(value for value in ids if self.table(value).get("_bitfield"))

        if provenance_sufficient:
            ids = tuple(value for value in ids
                        if not any(cell["semantic"] and not cell["source_ids"]
                                   for row in self.table(value)["rows"]
                                   for cell in row["cells"]))

        if page is not None:
            ids = tuple(value for value in ids
                        if self.table(value)["page_start"] <= page
                        <= self.table(value)["page_end"])

        if only_unreviewed:
            ids = tuple(value for value in ids
                        if self.verdicts.get(value, {}).get("verdict", UNREVIEWED)
                        == UNREVIEWED)

        return ids

    def table(self, table_id: str) -> Mapping[str, object]:
        for table in self.tables:
            if table["table_id"] == table_id:
                return table

        raise StandardStructureError(f"unknown table {table_id}")

    def resume_at(self, ids: Sequence[str]) -> int:
        for offset, table_id in enumerate(ids):
            if self.verdicts.get(table_id, {}).get("verdict", UNREVIEWED) == UNREVIEWED:
                return offset

        return 0

    # -- verdicts ---------------------------------------------------------

    def record(self, table_id: str, verdict: str, **dimensions: str) -> None:
        """Record a verdict beside -- never over -- the automated status."""

        if verdict not in HUMAN_VERDICTS:
            raise StandardStructureError(f"unknown verdict {verdict}")

        unknown = set(dimensions) - set(REVIEW_DIMENSIONS)

        if unknown:
            raise StandardStructureError(f"unknown review dimension {sorted(unknown)}")

        table = self.table(table_id)
        self.verdicts[table_id] = {
            "table_id": table_id, "verdict": verdict,
            # The automated status is evidence and is kept as it was.
            "geometry_status": table["geometry_status"],
            "warnings": list(table.get("warnings", ())),
            "reviewed_at": _now(), "reviewed_by": self.reviewer,
            **{name: dimensions.get(name, "UNKNOWN") for name in REVIEW_DIMENSIONS},
        }

    def clear(self, table_id: str) -> None:
        self.verdicts.pop(table_id, None)

    def summary(self) -> dict[str, int]:
        counts = {"TOTAL": len(self.tables), UNREVIEWED: 0}
        counts.update({verdict: 0 for verdict in HUMAN_VERDICTS})

        for table in self.tables:
            verdict = self.verdicts.get(table["table_id"], {}).get("verdict")
            counts[verdict if verdict in HUMAN_VERDICTS else UNREVIEWED] += 1

        return counts

    def save(self) -> Path:
        current, _ = self.structures_store.load(self.standard_id, self.revision)

        if current.structure_fingerprint != self.manifest.structure_fingerprint:
            raise StandardStructureError(
                "structures changed during review; refusing to save")

        StandardStructureStore._atomic_write(self.path, canonical_json({
            "schema_version": REVIEW_SCHEMA_VERSION,
            "standard_id": self.standard_id, "revision": self.revision,
            "corpus_fingerprint": self.manifest.corpus_fingerprint,
            "layout_fingerprint": self.manifest.layout_fingerprint,
            "structure_fingerprint": self.manifest.structure_fingerprint,
            "reviewer": self.reviewer, "updated_at": _now(),
            "summary": self.summary(),
            "rows": [self.verdicts[key] for key in sorted(self.verdicts)],
        }))

        return self.path

    # -- rendering --------------------------------------------------------

    def grid_text(self, table_id: str) -> str:
        table = self.table(table_id)
        columns = int(table["column_count"])
        lines = []

        for row in table["rows"][:_MAX_GRID_ROWS]:
            cells = [""] * columns

            for cell in row["cells"]:
                if 0 <= cell["column_index"] < columns:
                    cells[cell["column_index"]] = cell["text"][:_MAX_CELL_CHARS]

            marker = "H" if row["header_candidate"] == "TRUE" else " "
            lines.append(f"   {marker} | " + " | ".join(
                value.ljust(min(_MAX_CELL_CHARS, 14)) for value in cells))

        if len(table["rows"]) > _MAX_GRID_ROWS:
            lines.append(f"     ... {len(table['rows']) - _MAX_GRID_ROWS} more rows")

        return "\n".join(lines)

    def svg(self, table_id: str) -> Path:
        """Write a local SVG of the inferred geometry, for eyeballing a table."""
        table = self.table(table_id)
        x0, y0, x1, y1 = table["bbox"]
        pad = 12
        width, height = (x1 - x0) + pad * 2, (y1 - y0) + pad * 2
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
                 f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
                 f'<rect x="{pad}" y="{pad}" width="{x1 - x0:.1f}" '
                 f'height="{y1 - y0:.1f}" fill="none" stroke="#333" stroke-width="1"/>']

        for row in table["rows"]:
            ry0 = row["bbox"][1] - y0 + pad
            parts.append(f'<line x1="{pad}" y1="{ry0:.1f}" x2="{width - pad:.0f}" '
                         f'y2="{ry0:.1f}" stroke="#c33" stroke-width="0.5"/>')

            for cell in row["cells"]:
                cx0, cy0, cx1, cy1 = cell["bbox"]
                parts.append(
                    f'<rect x="{cx0 - x0 + pad:.1f}" y="{cy0 - y0 + pad:.1f}" '
                    f'width="{max(cx1 - cx0, 1):.1f}" height="{max(cy1 - cy0, 1):.1f}" '
                    f'fill="none" stroke="#39c" stroke-width="0.4"/>')

        parts.append("</svg>")
        target = (self.structures_store.directory(self.standard_id, self.revision)
                  / "figures" / f"{table_id}.svg")
        StandardStructureStore._atomic_write(target, "\n".join(parts).encode("utf-8"))

        return target


_STRUCTURAL_LABEL = re.compile(r"^\d+\s+\w+s?$")

# A row of real bit fields covers most of the ruler between its fields.

_RULER_TILING_SHARE = 0.6

# A ruler this short is usually a fragment of a longer one, and mapping a label
# onto a fragment gives bit numbers that mean nothing.

_MIN_TRUSTED_RULER = 8

# A label that says how many words it occupies is describing a word of a packet,
# not a run of bits inside one.

_WORD_DESCRIPTOR = re.compile(r"\(\s*\d+\s+Words?\b", re.I)

# One wide centred label is not a tiled row; a bit-field row has several fields.

_MIN_TILING_FIELDS = 2
_RESERVED_LABEL = re.compile(r"^reserved\b", re.I)


def _span_evidence(review: "StandardStructureReview", table, bitfield,
                   labels: Sequence[int]) -> list[dict[str, object]]:
    """What each field candidate says, what its box measures, and which word.

    A candidate is a measured span or a cell that states its own bits; since
    STD2C the second kind needs no ruler span to name a field.
    """
    from standard_semantic import PositionSource, SpanRole
    from standard_stated_range import RangeContext, stated_ranges_for_cell
    from standard_word_association import (
        UNRESOLVED, associate_words, field_candidates, page_words,
    )

    context = RangeContext(in_bitfield_region=True, ruler_labels=tuple(labels))
    associations = associate_words(table)
    words = page_words(review.layout(), int(table["page_start"]))
    links = review.normative_links(table["table_id"])
    spans: list[dict[str, object]] = []

    for entry in field_candidates(bitfield, table, words=words,
                                  prose_cells=frozenset(links)):
        covered = list(entry.covered_labels)
        ranges = stated_ranges_for_cell(
            entry.text, cell_id=entry.cell_id, source_ids=entry.source_ids,
            provenance=entry.provenance, context=context,
            covered_labels=covered)
        usable = [item for item in ranges if item.positional]
        stated = usable[0] if len(usable) == 1 else None
        ordered = sorted(covered)
        contiguous = all(later - earlier == 1
                         for earlier, later in zip(ordered, ordered[1:]))

        if stated is not None:
            # The document named its own bits. No measurement outranks that.

            role = (SpanRole.RESERVED.value
                    if _RESERVED_LABEL.match(entry.text.strip())
                    else SpanRole.FIELD.value)
            why = f"the label states bits {stated.high_bit}..{stated.low_bit}"
            source = PositionSource.STATED_RANGE.value
        elif len(usable) > 1:
            role, why = SpanRole.UNKNOWN.value, "the label states more than one range"
            source = PositionSource.UNKNOWN.value
        elif ranges:
            role, why = (SpanRole.UNKNOWN.value,
                         "stated range refused: " + ", ".join(ranges[0].warnings))
            source = PositionSource.UNKNOWN.value
        elif not contiguous:
            role, why = SpanRole.UNKNOWN.value, "covers a broken label range"
            source = PositionSource.UNKNOWN.value
        elif covered and len(covered) >= len(labels):
            role, why = SpanRole.IGNORE.value, "spans the whole ruler, so it is a band"
            source = PositionSource.UNKNOWN.value
        elif _RESERVED_LABEL.match(entry.text.strip()):
            role, why = SpanRole.RESERVED.value, "the visible label says reserved"
            source = PositionSource.GEOMETRIC_RULER.value
        elif _STRUCTURAL_LABEL.match(entry.text.strip()):
            role, why = SpanRole.STRUCTURAL_LABEL.value, "reads as a count, not a field"
            source = PositionSource.UNKNOWN.value
        elif _WORD_DESCRIPTOR.search(entry.text):
            role, why = (SpanRole.STRUCTURAL_LABEL.value,
                         "names a whole packet word, not a run of bits")
            source = PositionSource.UNKNOWN.value
        elif covered:
            role, why = SpanRole.FIELD.value, f"covers {len(covered)} contiguous labels"
            source = PositionSource.GEOMETRIC_RULER.value
        else:
            role, why = (SpanRole.UNKNOWN.value,
                         "neither a stated range nor a measurable span")
            source = PositionSource.UNKNOWN.value

        link = links.get(entry.cell_id)

        if link is not None and stated is None:
            # The diagram names this field; a numbered rule positions it.

            role = (SpanRole.RESERVED.value
                    if _RESERVED_LABEL.match(entry.text.strip())
                    else SpanRole.FIELD.value)
            why = (f"a normative rule states bits {link.high_bit}..{link.low_bit} "
                   f"for this field")
            source = PositionSource.NORMATIVE_PROSE_RANGE.value

        association = associations.get(entry.cell_id, UNRESOLVED)
        spans.append({
            "prose": (f"{link.high_bit}..{link.low_bit}" if link else None),
            "prose_source": (link.requirement_source_id if link else None),
            "index": entry.index, "label": entry.text, "covered": covered,
            "recommended_role": role, "reason": why, "position_source": source,
            "measured": entry.from_geometric_span,
            "word": (association.word_index if association.word_index is not None
                     else association.word_label),
            "word_source": association.association_source.value,
            "word_warnings": list(association.warnings),
            "stated": (f"{stated.high_bit}..{stated.low_bit}" if stated else None),
            "geometry": (f"{ordered[-1]}..{ordered[0]}" if ordered else None),
            "agreement": (stated.geometry_agreement.value if stated else None),
            "cell_id": entry.cell_id,
            "refused": [item.range_text for item in ranges
                        if not item.positional],
            "provenance": entry.provenance,
            "source_count": len(entry.source_ids)})

    return spans


def recommend(review: "StandardStructureReview", table_id: str) -> dict[str, object]:
    """Advisory only: what the label text says, and what the shapes suggest.

    Nothing here reads meaning out of prose. A span is called a field because
    the label states its bits, or because it covers a contiguous run of ruler
    labels; reserved because the visible label says so.
    """
    from standard_semantic import BitOrder, PositionSource, SpanRole, bit_order_of
    from standard_stated_range import ruler_is_trusted

    table = review.table(table_id)
    bitfield = review.bitfields.get(table_id)

    if bitfield is None:
        return {"candidate": "RECOMMEND_REJECT",
                "reasons": ["no bit-label geometry on this candidate"],
                "spans": []}

    labels = [item["value"] for item in bitfield["bit_labels"]]
    order = bit_order_of(labels)
    reasons: list[str] = []
    spans = _span_evidence(review, table, bitfield, labels)

    # In a real bit layout the fields of a row tile the ruler between them. A
    # word table drawn under a ruler has one centred label per row covering a
    # fraction of it, and there the horizontal position carries no bit meaning.
    # This says nothing about a span whose bits the label states outright.

    measured = [item for item in spans
                if item["position_source"] == PositionSource.GEOMETRIC_RULER.value]
    by_row: dict[object, set[int]] = {}
    counts: dict[object, int] = {}

    for item in measured:
        row = item["word"]
        by_row.setdefault(row, set()).update(item["covered"])
        counts[row] = counts.get(row, 0) + 1

    tiled = [row for row, values in by_row.items()
             if counts.get(row, 0) >= _MIN_TILING_FIELDS
             and len(values) >= _RULER_TILING_SHARE * max(len(labels), 1)]
    coverage = (max((len(values) for values in by_row.values()), default=0)
                / max(len(labels), 1))

    if measured and not tiled:
        for item in measured:
            item["recommended_role"] = SpanRole.ROW_LABEL.value
            item["position_source"] = PositionSource.UNKNOWN.value
            item["reason"] = ("no row tiles the ruler with several fields; "
                              "a centred label's position need not encode bits")

    fields = [item for item in spans
              if item["recommended_role"] in ("FIELD", "RESERVED")]
    stated = [item for item in fields
              if item["position_source"] == PositionSource.STATED_RANGE.value]
    geometric = [item for item in fields
                 if item["position_source"] == PositionSource.GEOMETRIC_RULER.value]
    unresolved = any(item["provenance"] == "UNRESOLVED" or not item["source_count"]
                     for item in fields)
    conflicts = [item for item in stated
                 if item["agreement"] in ("CONTRADICTS", "PARTIAL_OVERLAP")]
    refused = [item for item in spans if item["refused"]]
    trusted = ruler_is_trusted(labels)
    unresolved_word = [item for item in fields if item["word_source"] == "UNKNOWN"]
    words = sorted({str(item["word"]) for item in fields})
    geometric_word = [item for item in fields
                      if item["word_source"] == "GEOMETRIC_ROW"]

    if order is BitOrder.UNKNOWN:
        reasons.append("the ruler is not one ascending or descending run")
        candidate = "RECOMMEND_REJECT"
    elif not bitfield["spans"]:
        reasons.append("no cell spans more than one ruler label")
        candidate = "RECOMMEND_FOLLOWUP"
    elif not fields:
        reasons.append(
            f"no span looks like a field; best measured row covers "
            f"{coverage:.0%} of the ruler" if by_row or coverage
            else "no span states its bits and no span looks like a field")
        candidate = "RECOMMEND_FOLLOWUP"
    elif unresolved:
        reasons.append("a field-shaped span has no canonical source")
        candidate = "RECOMMEND_FOLLOWUP"
    elif unresolved_word:
        reasons.append(
            f"{len(unresolved_word)} field(s) belong to no resolvable word")
        candidate = "RECOMMEND_FOLLOWUP"
    elif len(words) > 1 and geometric_word:
        reasons.append(
            f"{len(words)} words, but word identity rests on horizontal bands")
        candidate = "RECOMMEND_FOLLOWUP"
    elif refused:
        reasons.append(
            f"{len(refused)} label(s) look like a stated range but were refused")
        candidate = "RECOMMEND_FOLLOWUP"
    elif geometric and len(labels) < _MIN_TRUSTED_RULER:
        reasons.append(
            f"only {len(labels)} ruler labels; likely a fragment of a longer "
            "ruler, and a measured field would inherit its bit numbers")
        candidate = "RECOMMEND_FOLLOWUP"
    elif table["warnings"]:
        reasons.append(f"geometry warnings: {', '.join(table['warnings'])}")
        candidate = "RECOMMEND_ACCEPTABLE_WARNING"
    else:
        reasons.append("every field position is stated or measured against a "
                       "whole ruler, with provenance")
        candidate = "RECOMMEND_PASS"

    if stated:
        reasons.append(f"{len(stated)} field position(s) stated in the label text")

    if len(words) > 1:
        reasons.append(f"fields span {len(words)} words; bits are local to each")

    if conflicts:
        # Not a blocker: the geometry is centred text, and the text wins.

        reasons.append(f"{len(conflicts)} stated range(s) disagree with the "
                       "measured span; the stated range is the authority")

    reasons.append(f"{len(labels)} ruler labels"
                   + (" (a whole ruler)" if trusted else " (not a whole ruler)")
                   + f", {len(fields)} field-shaped spans")

    return {"candidate": candidate, "bit_order": order.value,
            "ruler_trusted": trusted, "words": words, "reasons": reasons,
            "spans": spans}


def _render_bitfield(review: "StandardStructureReview", table_id: str, out) -> None:
    bitfield = review.bitfields.get(table_id)

    if bitfield is None:
        return

    advice = recommend(review, table_id)
    labels = [item["value"] for item in bitfield["bit_labels"]]
    out.write(f"  bit order        : {advice['bit_order']}\n")
    ruler = " ".join(str(value) for value in labels)
    out.write(f"  visible ruler    : {ruler if len(ruler) <= 110 else ruler[:107] + '...'}"
              f"  [{'whole ruler' if advice['ruler_trusted'] else 'RULER_FRAGMENTED'}]\n")
    out.write(f"  words            : {', '.join(advice['words']) or 'none'}\n")
    out.write(f"  recommendation   : {advice['candidate']}\n")

    for reason in advice["reasons"]:
        out.write(f"      because      {reason}\n")

    # Grouped by word, because a bit range means nothing without its word.

    grouped: dict[object, list] = {}

    for span in advice["spans"]:
        grouped.setdefault(span["word"], []).append(span)

    out.write(f"  field candidates ({len(advice['spans'])}):\n")

    for word in sorted(grouped, key=lambda value: (value is None, str(value))):
        members = grouped[word]
        source = members[0]["word_source"]
        out.write(f"    word {str(word):<6} [{source}]"
                  + (f"  {', '.join(members[0]['word_warnings'])}"
                     if members[0]["word_warnings"] else "") + "\n")

        for span in members:
            covered = span["covered"]
            shown = (f"{covered[0]}..{covered[-1]}" if len(covered) > 3
                     else ",".join(str(value) for value in covered) or "-")
            out.write(f"      [{span['index']:2d}] {span['label'][:32]:<32} "
                      f"{span['provenance']:<22} sources {span['source_count']}\n")
            origin = ("measured span" if span["measured"]
                      else "prose-linked cell" if span.get("prose")
                      else "stated-range cell")
            out.write(f"           from     : {origin}\n")
            out.write(f"           stated   : {span['stated'] or '-'}\n")

            if span.get("prose"):
                out.write(f"           PROSE    : {span['prose']}  "
                          f"(normative rule {span['prose_source']})\n")

            out.write(f"           geometry : {span['geometry'] or '-'}"
                      f"   (measured over {shown})\n")

            if span["agreement"]:
                out.write(f"           status   : {span['agreement']}\n")

            if span["refused"]:
                out.write(f"           refused  : {', '.join(span['refused'])}\n")

            out.write(f"           authority: {span['position_source']}\n")
            out.write(f"           -> {span['recommended_role']}: {span['reason']}\n")

    # Roles classify what the diagram shows. A field a rule positions from
    # outside the diagram is a separate decision, and is shown as one.

    diagram = [item for item in advice["spans"] if not item.get("prose")]
    linked = [item for item in advice["spans"] if item.get("prose")]
    roles = ",".join(str(item["recommended_role"]) for item in diagram)
    out.write(f"  diagram field roles : {roles}\n")
    links = review.normative_links(table_id)

    if linked:
        out.write("  normative links to accept or decline:\n")

        for item in linked:
            link = next((value for value in links.values()
                         if value.target_cell_id == item.get("cell_id")), None)
            out.write(f"    {link.link_fingerprint if link else '?'}  "
                      f"{item['label'][:30]!r} -> word {item['word']} "
                      f"bits {item['prose']}\n")
            out.write(f"        source: {item['prose_source']}   "
                      f"suggested role: {item['recommended_role']}\n")

    out.write(f"  to approve (yours to run, records your identity):\n"
              f"    /standard approve-bitfield {review.standard_id} {review.revision} "
              f"{bitfield['bitfield_id']} <PASS|ACCEPTABLE_WARNING> {roles}")

    for item in linked:
        link = next((value for value in links.values()
                     if value.target_cell_id == item.get("cell_id")), None)

        if link is not None:
            suffix = ("=RESERVED" if item["recommended_role"] == "RESERVED" else "")
            out.write(f" \\\n        --accept-link {link.link_fingerprint}{suffix}")

    out.write("\n")

    if linked:
        out.write("    (use --decline-link to approve without a link; a link that "
                  "is neither accepted nor declined blocks the build)\n")


def _render(review: StandardStructureReview, table_id: str, offset: int,
            total: int, out) -> None:
    table = review.table(table_id)
    recorded = review.verdicts.get(table_id, {})
    out.write("\n" + "=" * 74 + "\n")
    out.write(f"  {offset + 1} / {total}   {table_id}\n")
    out.write("=" * 74 + "\n")
    pages = (f"{table['page_start']}" if table["page_start"] == table["page_end"]
             else f"{table['page_start']}-{table['page_end']}")
    out.write(f"  page {pages}   rows {table['row_count']}   "
              f"columns {table['column_count']}   {table['geometry_status']}\n")
    out.write(f"  caption sources   : {len(table['caption_source_ids'])}\n")
    out.write(f"  supporting sources: {len(table['supporting_source_ids'])}\n")

    if table.get("continuation_ids"):
        out.write(f"  continues into    : {', '.join(table['continuation_ids'])}\n")

    if table.get("_bitfield"):
        out.write("  bit-label geometry: yes (candidate only)\n")

    if table.get("warnings"):
        out.write(f"  warnings          : {', '.join(table['warnings'])}\n")

    out.write(f"  human verdict     : {recorded.get('verdict', UNREVIEWED)}\n")
    out.write("-" * 74 + "\n")
    out.write(review.grid_text(table_id) + "\n")
    out.write("-" * 74 + "\n")
    _render_bitfield(review, table_id, out)


def _write_summary(review: StandardStructureReview, out) -> None:
    counts = review.summary()
    out.write("\nSTRUCTURE REVIEW SUMMARY\n")

    for key in ("TOTAL", UNREVIEWED, *HUMAN_VERDICTS):
        out.write(f"  {key:<20} {counts[key]}\n")

    blocking = sum(counts[key] for key in BLOCKING_VERDICTS)

    if counts[UNREVIEWED] == 0:
        out.write("\nSTRUCTURE REVIEW COMPLETE\n")
        out.write("  approval blocked by review findings\n" if blocking
                  else "  eligible for structure approval\n")


def run_review(review: StandardStructureReview, ids: Sequence[str], *,
               stdin=None, stdout=None) -> int:
    stdin = stdin or sys.stdin
    out = stdout or sys.stdout

    if not ids:
        out.write("No table candidates match the selection.\n")
        return 0

    _write_summary(review, out)
    out.write(f"\nReviewing as {review.reviewer}. Press ? for help.\n")
    offset = review.resume_at(ids)
    recorded = 0

    while 0 <= offset < len(ids):
        table_id = ids[offset]
        _render(review, table_id, offset, len(ids), out)
        out.write("  [p]ass [w]arning [f]ail [n]eeds-followup [s]kip [b]ack "
                  "[v]iew [q]uit ? > ")
        out.flush()
        line = stdin.readline()

        if not line:
            break

        key = line.strip().lower()

        if key in _VERDICT_KEYS:
            review.record(table_id, _VERDICT_KEYS[key])
            recorded += 1
            offset += 1
        elif key == "u":
            review.clear(table_id)
        elif key == "v":
            out.write(f"  wrote {review.svg(table_id)}\n")
        elif key in ("s", ""):
            offset += 1
        elif key == "b":
            offset = max(0, offset - 1)
        elif key == "q":
            break
        elif key == "?":
            out.write(_HELP)
        else:
            out.write(f"  unknown key {key!r}; press ? for help\n")

    review.save()
    _write_summary(review, out)
    out.write(f"\nSaved to {review.path}\n")

    return recorded


def main(argv: Sequence[str] | None = None, *, stdin=None, stdout=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="standard_structure_review",
        description="Operator-only review of derived table geometry.")
    parser.add_argument("--standard", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--reviewer", default=None)
    parser.add_argument("--only-unreviewed", action="store_true")
    parser.add_argument("--sample", action="store_true",
                        help="review a stratified spread rather than every table")
    parser.add_argument("--bitfields", action="store_true",
                        help="review only candidates with bit-label geometry")
    parser.add_argument("--only-provenance-sufficient", action="store_true")
    parser.add_argument("--page", type=int, default=None)
    parser.add_argument("--summary", action="store_true")
    options = parser.parse_args(argv)
    out = stdout or sys.stdout
    root = options.root or str(standards_root())

    try:
        review = StandardStructureReview(
            StandardStore(root), options.standard, options.revision,
            reviewer=options.reviewer)
    except (StandardStructureError, StandardStoreError, FileNotFoundError) as exc:
        out.write(f"structure review unavailable: {exc}\n")
        return 2

    if options.summary:
        _write_summary(review, out)
        return 0

    run_review(review, review.selection(
        only_unreviewed=options.only_unreviewed, sample=options.sample,
        bitfields=options.bitfields,
        provenance_sufficient=options.only_provenance_sufficient,
        page=options.page), stdin=stdin, stdout=out)

    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
