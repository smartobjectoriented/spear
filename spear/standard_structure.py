"""Deterministic table and bitfield geometry derived from a standard's layout.

This reads the word boxes already extracted for a revision and proposes table
regions, rows, columns and cells. It proposes only: nothing here asserts what a
table *means*. A run of visible bit labels is recorded as visible labels with
their boxes, never as a msb/lsb/width/mask, and every candidate keeps the
canonical source IDs it came from so a citation still comes from the corpus.

The canonical text corpus is never read for anything but provenance, and is
never written.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import Iterable, Mapping, Sequence

from standard_schema import StandardContentType, canonical_json, sha256_json


STRUCTURE_SCHEMA_VERSION = 2
STRUCTURE_EXTRACTOR_VERSION = "geometry-tables-v2"

_INTEGER = re.compile(r"^\d{1,3}$")
_MIN_ROWS = 2
_MIN_FRAGMENTS_PER_ROW = 2
_MIN_BIT_LABELS = 4

# A region may legitimately support a cell from a few canonical units; beyond
# that the attribution stops being bounded and is not worth claiming.

_MAX_REGION_SOURCES = 8

# Two rows of aligned labels is what a diagram looks like. Without a caption
# vouching for it, that is not enough to accept a table without review.

_MIN_AUTO_ROWS = 3
_PUNCTUATION_ONLY = re.compile(r"^[^\w]*$")

# A bullet in the first column marks a list, whatever its columns look like.

_BULLET = re.compile("^(?:[\u2022\u25aa\u25e6\u2023\u2043\u00b7*\u2013\u2014-]+|[0-9]{1,3}[.)])$")
_LIST_SHARE = 0.6

# Fractions of the page, not absolute points, so page size cannot leak in.

_PAGE_BOTTOM_SHARE = 0.80
_PAGE_TOP_SHARE = 0.25
_CAPTION_GAP_SHARE = 0.08


class GeometryStatus(StrEnum):
    AUTO_GEOMETRY_OK = "AUTO_GEOMETRY_OK"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    REJECTED_GEOMETRY = "REJECTED_GEOMETRY"


class ProvenanceQuality(StrEnum):
    """Why a cell carries the canonical sources it does."""

    DIRECT_TEXT_MATCH = "DIRECT_TEXT_MATCH"
    ROW_INHERITED = "ROW_INHERITED"
    TABLE_REGION_INHERITED = "TABLE_REGION_INHERITED"
    UNRESOLVED = "UNRESOLVED"


class HeaderCandidate(StrEnum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


class StandardStructureError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# derived types
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StandardTableCell:
    cell_id: str
    row_index: int
    column_index: int
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    source_ids: tuple[str, ...] = ()
    colspan_candidate: int = 1
    rowspan_candidate: int = 1
    warnings: tuple[str, ...] = ()
    provenance: ProvenanceQuality = ProvenanceQuality.UNRESOLVED
    semantic: bool = True

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["bbox"] = list(self.bbox)
        value["source_ids"] = list(self.source_ids)
        value["warnings"] = list(self.warnings)
        value["provenance"] = self.provenance.value

        return value


@dataclass(frozen=True)
class StandardTableRow:
    row_index: int
    page: int
    bbox: tuple[float, float, float, float]
    cells: tuple[StandardTableCell, ...]
    header_candidate: HeaderCandidate = HeaderCandidate.UNKNOWN
    evidence: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {"row_index": self.row_index, "page": self.page,
                "bbox": list(self.bbox),
                "cells": [cell.to_dict() for cell in self.cells],
                "header_candidate": self.header_candidate.value,
                "evidence": list(self.evidence), "warnings": list(self.warnings)}


@dataclass(frozen=True)
class StandardBitLabel:
    value: int
    column_index: int
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, object]:
        return {"value": self.value, "column_index": self.column_index,
                "bbox": list(self.bbox)}


@dataclass(frozen=True)
class StandardBitfieldSpan:
    text: str
    row_index: int
    bbox: tuple[float, float, float, float]
    covered_labels: tuple[int, ...]
    source_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "row_index": self.row_index,
                "bbox": list(self.bbox),
                "covered_labels": list(self.covered_labels),
                "source_ids": list(self.source_ids),
                "warnings": list(self.warnings)}


@dataclass(frozen=True)
class StandardBitfieldCandidate:
    """Geometry that resembles a bit-position layout. Not a packet definition."""

    bitfield_id: str
    table_id: str
    page: int
    label_row_index: int
    bit_labels: tuple[StandardBitLabel, ...]
    spans: tuple[StandardBitfieldSpan, ...]
    evidence: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    review_status: str = "UNREVIEWED"

    def to_dict(self) -> dict[str, object]:
        return {"bitfield_id": self.bitfield_id, "table_id": self.table_id,
                "page": self.page, "label_row_index": self.label_row_index,
                "bit_labels": [item.to_dict() for item in self.bit_labels],
                "spans": [item.to_dict() for item in self.spans],
                "evidence": list(self.evidence), "warnings": list(self.warnings),
                "review_status": self.review_status}


@dataclass(frozen=True)
class StandardTableContinuation:
    from_table_id: str
    to_table_id: str
    evidence: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {"from_table_id": self.from_table_id,
                "to_table_id": self.to_table_id,
                "evidence": list(self.evidence), "warnings": list(self.warnings)}


@dataclass(frozen=True)
class StandardTableCandidate:
    schema_version: int
    table_id: str
    standard_id: str
    revision: str
    corpus_fingerprint: str
    layout_fingerprint: str
    structure_extractor_version: str
    page_start: int
    page_end: int
    bbox: tuple[float, float, float, float]
    row_count: int
    column_count: int
    rows: tuple[StandardTableRow, ...]
    caption_source_ids: tuple[str, ...] = ()
    supporting_source_ids: tuple[str, ...] = ()
    continuation_ids: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    geometry_status: GeometryStatus = GeometryStatus.NEEDS_REVIEW
    review_status: str = "UNREVIEWED"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "table_id": self.table_id,
            "standard_id": self.standard_id, "revision": self.revision,
            "corpus_fingerprint": self.corpus_fingerprint,
            "layout_fingerprint": self.layout_fingerprint,
            "structure_extractor_version": self.structure_extractor_version,
            "page_start": self.page_start, "page_end": self.page_end,
            "bbox": list(self.bbox), "row_count": self.row_count,
            "column_count": self.column_count,
            "rows": [row.to_dict() for row in self.rows],
            "caption_source_ids": list(self.caption_source_ids),
            "supporting_source_ids": list(self.supporting_source_ids),
            "continuation_ids": list(self.continuation_ids),
            "evidence": list(self.evidence), "warnings": list(self.warnings),
            "geometry_status": self.geometry_status.value,
            "review_status": self.review_status,
        }

    def grid(self) -> tuple[tuple[str, ...], ...]:
        """The table as text, for an operator to read at a glance."""
        table = []

        for row in self.rows:
            line = [""] * self.column_count

            for cell in row.cells:
                if 0 <= cell.column_index < self.column_count:
                    line[cell.column_index] = cell.text

            table.append(tuple(line))

        return tuple(table)


# --------------------------------------------------------------------------
# page geometry
# --------------------------------------------------------------------------

@dataclass
class _Fragment:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    column_index: int = -1


@dataclass
class _Band:
    y0: float
    y1: float
    fragments: list[_Fragment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(fragment.text for fragment in
                        sorted(self.fragments, key=lambda item: item.x0))

    @property
    def centre(self) -> float:
        return (self.y0 + self.y1) / 2


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _page_fragments(page: Mapping[str, object]) -> list[_Fragment]:
    fragments = []

    for block in page["blocks"]:
        for line in block["lines"]:
            words = line["words"]

            if not words:
                continue

            text = _normalize(" ".join(word["text"] for word in words))

            if not text:
                continue

            box = line["bbox"]
            fragments.append(_Fragment(float(box[0]), float(box[1]),
                                       float(box[2]), float(box[3]), text))

    return fragments


def _bands(fragments: Sequence[_Fragment]) -> list[_Band]:
    """Group fragments that sit on the same visual line into one band.

    Poppler emits a separate line per table cell, so a row of a table arrives as
    several fragments sharing a y range rather than as one line.
    """

    if not fragments:
        return []

    heights = [item.y1 - item.y0 for item in fragments if item.y1 > item.y0]
    tolerance = (statistics.median(heights) * 0.6) if heights else 1.0
    bands: list[_Band] = []

    for fragment in sorted(fragments, key=lambda item: (item.y0, item.x0)):
        for band in reversed(bands):
            if abs(band.centre - (fragment.y0 + fragment.y1) / 2) <= tolerance:
                band.fragments.append(fragment)
                band.y0 = min(band.y0, fragment.y0)
                band.y1 = max(band.y1, fragment.y1)

                break
        else:
            bands.append(_Band(fragment.y0, fragment.y1, [fragment]))

    bands.sort(key=lambda band: band.y0)

    return bands


def _character_width(fragments: Sequence[_Fragment]) -> float:
    widths = [(item.x1 - item.x0) / max(len(item.text), 1) for item in fragments
              if item.x1 > item.x0]
    return statistics.median(widths) if widths else 4.0


def _regions(bands: Sequence[_Band],
             excluded: frozenset[str] = frozenset()) -> list[list[_Band]]:
    """Runs of adjacent bands that look like table rows rather than prose.

    A wide vertical gap ends a run, so two tables on one page stay apart. A run
    has to open and close on a multi-fragment band, which is what keeps aligned
    prose and numbered lists out: those are one fragment per line.
    """

    if not bands:
        return []

    gaps = [later.y0 - earlier.y1 for earlier, later in zip(bands, bands[1:])
            if later.y0 >= earlier.y1]
    gap_limit = max(statistics.median(gaps) * 3, 12.0) if gaps else 24.0
    runs: list[list[_Band]] = []
    run: list[_Band] = []

    for band in bands:
        if band.text in excluded:
            # A heading or a caption ends the run rather than joining it.

            if run:
                runs.append(run)
                run = []

            continue

        if run and (band.y0 - run[-1].y1) > gap_limit:
            runs.append(run)
            run = []

        run.append(band)

    if run:
        runs.append(run)

    regions = []

    for candidate in runs:
        trimmed = list(candidate)

        while trimmed and len(trimmed[0].fragments) < _MIN_FRAGMENTS_PER_ROW:
            trimmed.pop(0)

        while trimmed and len(trimmed[-1].fragments) < _MIN_FRAGMENTS_PER_ROW:
            trimmed.pop()

        if sum(1 for band in trimmed
               if len(band.fragments) >= _MIN_FRAGMENTS_PER_ROW) >= _MIN_ROWS:
            regions.append(trimmed)

    return regions


def _columns(region: Sequence[_Band], tolerance: float) -> list[float]:
    """Cluster fragment left edges into column positions."""
    edges = sorted(fragment.x0 for band in region for fragment in band.fragments)

    if not edges:
        return []

    clusters: list[list[float]] = [[edges[0]]]

    for edge in edges[1:]:
        if edge - clusters[-1][-1] <= tolerance:
            clusters[-1].append(edge)
        else:
            clusters.append([edge])

    return [statistics.median(cluster) for cluster in clusters]


def _assign_columns(region: Sequence[_Band], centres: Sequence[float],
                    tolerance: float) -> None:
    for band in region:
        for fragment in band.fragments:
            distances = [abs(fragment.x0 - centre) for centre in centres]
            fragment.column_index = distances.index(min(distances))


def _span(members: Sequence[_Fragment], centres: Sequence[float],
          tolerance: float) -> tuple[int, bool]:
    """How many columns a cell reaches across, and whether it lines up with them.

    A field label covering four bit columns is ordinary table geometry. It is
    only ambiguous when its edge falls inside a column instead of beside one.
    """
    x0 = min(item.x0 for item in members)
    x1 = max(item.x1 for item in members)
    covered = [index for index, centre in enumerate(centres)
               if x0 - tolerance <= centre <= x1 + tolerance]

    if len(covered) <= 1:
        return 1, True

    return len(covered), abs(x0 - centres[covered[0]]) <= tolerance


def _merge_wrapped(region: Sequence[_Band]) -> list[tuple[_Band, list[_Band]]]:
    """Attach a band that only continues earlier cells to the row above it.

    A wrapped cell sits closer to the line above than a new row would, and never
    starts the first column.
    """
    rows: list[tuple[_Band, list[_Band]]] = []
    pitches = [later.y0 - earlier.y1 for earlier, later in zip(region, region[1:])
               if later.y0 >= earlier.y1]
    typical = statistics.median(pitches) if pitches else 0.0

    for band in region:
        opens_first_column = any(fragment.column_index == 0
                                 for fragment in band.fragments)

        if rows and not opens_first_column:
            gap = band.y0 - rows[-1][0].y1

            if gap <= typical:
                rows[-1][1].append(band)
                continue

        rows.append((band, []))

    return rows


def _looks_like_header(cells: Sequence[str], below: Sequence[Sequence[str]]) -> HeaderCandidate:
    text = [value for value in cells if value]

    if len(text) < 2:
        return HeaderCandidate.UNKNOWN

    labelled = sum(1 for value in text if not _INTEGER.match(value))

    if labelled != len(text):
        return HeaderCandidate.UNKNOWN

    numeric_below = sum(1 for row in below for value in row
                        if value and _INTEGER.match(value))

    if numeric_below:
        return HeaderCandidate.TRUE

    return HeaderCandidate.UNKNOWN if not below else HeaderCandidate.TRUE


def _bullet_led(grid: Sequence[Sequence[str]]) -> bool:
    """Whether the first column is a bullet, which makes this a list.

    Only rows that populate the first column count: the wrapped continuation of
    a list item leaves it empty and should not dilute the evidence.
    """
    first = [line[0] for line in grid if line and line[0]]

    if len(first) < 2:
        return False

    bullets = sum(1 for value in first if _BULLET.match(value))

    return bullets >= _LIST_SHARE * len(first)


def _bit_label_run(values: Sequence[str]) -> tuple[int, int] | None:
    """The longest run of consecutive integers in a row, by column index.

    A bit ruler is usually introduced by a label -- "Word", "Octet", "Bit" -- so
    the run is looked for inside the row rather than required to be the row.
    """
    best: tuple[int, int] | None = None
    start = None

    for index, value in enumerate(values):
        if value and _INTEGER.match(value):
            if start is None:
                start = index
                continue

            previous = int(values[index - 1]) if _INTEGER.match(values[index - 1] or "") else None

            if previous is None or abs(int(value) - previous) != 1:
                if best is None or (index - start) > (best[1] - best[0]):
                    best = (start, index)

                start = index
        else:
            if start is not None and (best is None or (index - start) > (best[1] - best[0])):
                best = (start, index)

            start = None

    if start is not None and (best is None or (len(values) - start) > (best[1] - best[0])):
        best = (start, len(values))

    if best is None:
        return None

    numbers = [int(values[index]) for index in range(*best) if values[index]]

    if len(numbers) < _MIN_BIT_LABELS:
        return None

    steps = {later - earlier for earlier, later in zip(numbers, numbers[1:])}

    if steps not in ({1}, {-1}):
        return None

    return best


def _identifier(prefix: str, *parts: object) -> str:
    return f"{prefix}-{sha256_json(list(parts))[:16]}"


@dataclass(frozen=True)
class StandardStructureSet:
    tables: tuple[StandardTableCandidate, ...]
    bitfields: tuple[StandardBitfieldCandidate, ...]
    continuations: tuple[StandardTableContinuation, ...]

    def to_dict(self) -> dict[str, object]:
        return {"tables": [item.to_dict() for item in self.tables],
                "bitfields": [item.to_dict() for item in self.bitfields],
                "continuations": [item.to_dict() for item in self.continuations]}


def _page_sources(units: Iterable) -> dict[int, list[tuple[str, str, str]]]:
    index: dict[int, list[tuple[str, str, str]]] = {}

    for unit in units:
        index.setdefault(unit.page, []).append(
            (unit.source_id, _normalize(unit.text), unit.content_type.value))

    return index


def _not_table_rows(units: Iterable, page: int) -> frozenset[str]:
    """Text the canonical pass already read as a heading, caption or furniture.

    A clause heading is often set as a number and a title far apart, which looks
    exactly like a two-column row. The canonical extractor has already decided
    what those lines are, so the geometry pass defers to it rather than reading
    them a second time.
    """
    from standard_schema import StandardLayoutKind

    excluded = set()

    for unit in units:
        if unit.page != page:
            continue

        if (unit.layout_kind in (StandardLayoutKind.HEADING,
                                 StandardLayoutKind.FURNITURE,
                                 StandardLayoutKind.TOC_ENTRY)
                or unit.content_type in (StandardContentType.TABLE,
                                         StandardContentType.FIGURE)):
            excluded.add(_normalize(unit.text))

    return frozenset(excluded)


def _sources_for(text: str, entries: Sequence[tuple[str, str, str]]) -> tuple[str, ...]:
    """Canonical units whose text this geometry came from."""
    probe = _normalize(text)

    if not probe:
        return ()

    found = [source_id for source_id, unit_text, _ in entries
             if probe == unit_text or (len(probe) >= 4 and probe in unit_text)
             or (len(unit_text) >= 4 and unit_text in probe)]

    return tuple(dict.fromkeys(found))


def _caption_for(region: Sequence[_Band], bands: Sequence[_Band],
                 entries: Sequence[tuple[str, str, str]],
                 page_height: float,
                 kind: str = StandardContentType.TABLE.value) -> tuple[str, ...]:
    """A genuine caption of this kind sitting just above the region."""
    captions = {unit_text: source_id for source_id, unit_text, entry_kind in entries
                if entry_kind == kind}

    if not captions:
        return ()

    top = region[0].y0
    limit = page_height * _CAPTION_GAP_SHARE
    best: tuple[float, str] | None = None

    for band in bands:
        if band.y1 > top or (top - band.y1) > limit:
            continue

        source_id = captions.get(band.text)

        if source_id is not None and (best is None or band.y1 > best[0]):
            best = (band.y1, source_id)

    return (best[1],) if best else ()


def _build_table(region: Sequence[_Band], *, page: int, page_size: Sequence[float],
                 bands: Sequence[_Band], entries: Sequence[tuple[str, str, str]],
                 standard_id: str, revision: str, corpus_fingerprint: str,
                 layout_fingerprint: str) -> StandardTableCandidate | None:
    fragments = [fragment for band in region for fragment in band.fragments]
    tolerance = max(_character_width(fragments) * 2.0, 3.0)
    centres = _columns(region, tolerance)

    if len(centres) < 2:
        return None

    _assign_columns(region, centres, tolerance)
    grouped = _merge_wrapped(region)

    warnings: list[str] = []
    evidence: list[str] = [f"columns inferred from {len(fragments)} fragments",
                           f"column tolerance {tolerance:.1f}pt"]
    rows: list[StandardTableRow] = []
    supporting: list[str] = []

    for row_index, (band, wrapped) in enumerate(grouped):
        cells: dict[int, list[_Fragment]] = {}

        for fragment in band.fragments:
            cells.setdefault(fragment.column_index, []).append(fragment)

        for extra in wrapped:
            for fragment in extra.fragments:
                cells.setdefault(fragment.column_index, []).append(fragment)

        # The canonical unit for a table row is the row, so a cell inherits the
        # row's provenance when its own text is too short to match on its own.

        row_text = _normalize(" ".join(
            fragment.text for fragment in
            sorted([f for group in cells.values() for f in group],
                   key=lambda item: (item.y0, item.x0))))
        row_sources = _sources_for(row_text, entries)
        row_cells: list[StandardTableCell] = []
        row_warnings: list[str] = []

        for column_index, members in sorted(cells.items()):
            members.sort(key=lambda item: (item.y0, item.x0))
            text = _normalize(" ".join(item.text for item in members))
            box = (min(m.x0 for m in members), min(m.y0 for m in members),
                   max(m.x1 for m in members), max(m.y1 for m in members))
            span, aligned = _span(members, centres, tolerance)
            cell_warnings: list[str] = []

            if span > 1 and not aligned:
                cell_warnings.append("ambiguous_cell_span")

            if len(members) > 1 and any(a.y0 != members[0].y0 for a in members):
                cell_warnings.append("multi_line_cell")

            direct = _sources_for(text, entries)
            row_cells.append(StandardTableCell(
                cell_id=_identifier("cel", standard_id, revision,
                                    corpus_fingerprint, layout_fingerprint,
                                    page, row_index, column_index,
                                    [round(value, 2) for value in box], text),
                row_index=row_index, column_index=column_index, page=page,
                bbox=tuple(round(value, 3) for value in box), text=text,
                source_ids=direct, colspan_candidate=span,
                warnings=tuple(cell_warnings),
                provenance=(ProvenanceQuality.DIRECT_TEXT_MATCH if direct
                            else ProvenanceQuality.UNRESOLVED),
                semantic=bool(text.strip()) and not _PUNCTUATION_ONLY.match(text)))
            supporting.extend(direct)

        if len({cell.column_index for cell in row_cells}) != len(row_cells):
            row_warnings.append("overlapping_cells")

        inherited = tuple(dict.fromkeys(
            list(row_sources) + [value for cell in row_cells
                                 for value in cell.source_ids]))
        row_cells = [
            cell if cell.source_ids else replace(
                cell, source_ids=inherited,
                provenance=ProvenanceQuality.ROW_INHERITED)
            for cell in row_cells] if inherited else row_cells

        if not inherited:
            row_warnings.append("source_mapping_partial")

        supporting.extend(inherited)
        rows.append(StandardTableRow(
            row_index=row_index, page=page,
            bbox=(min(c.bbox[0] for c in row_cells), band.y0,
                  max(c.bbox[2] for c in row_cells),
                  max([band.y1] + [c.bbox[3] for c in row_cells])),
            cells=tuple(row_cells), evidence=(f"{len(row_cells)} cells",),
            warnings=tuple(row_warnings)))

    if len(rows) < _MIN_ROWS:
        return None

    grid = [[""] * len(centres) for _ in rows]

    for row in rows:
        for cell in row.cells:
            if 0 <= cell.column_index < len(centres):
                grid[row.row_index][cell.column_index] = cell.text

    header = _looks_like_header(grid[0], grid[1:])
    rows[0] = StandardTableRow(
        rows[0].row_index, rows[0].page, rows[0].bbox, rows[0].cells,
        header, rows[0].evidence + ("first row of the region",), rows[0].warnings)

    if header is not HeaderCandidate.TRUE:
        warnings.append("missing_header")

    if _bullet_led(grid):
        warnings.append("list_not_table")

    figure_caption = _caption_for(region, bands, entries, page_size[3],
                                  StandardContentType.FIGURE.value)
    table_caption = _caption_for(region, bands, entries, page_size[3])

    if figure_caption and not table_caption:
        warnings.append("figure_region")

    occupancy = [sum(1 for line in grid if line[index]) / len(grid)
                 for index in range(len(centres))]

    if any(share < 0.34 for share in occupancy):
        warnings.append("unstable_columns")

    pitches = [later[0].y0 - earlier[0].y1 for earlier, later in zip(grouped, grouped[1:])]

    if len(pitches) >= 2 and statistics.pstdev(pitches) > max(
            statistics.median(pitches), 1.0):
        warnings.append("irregular_rows")

    if any("ambiguous_cell_span" in cell.warnings for row in rows for cell in row.cells):
        warnings.append("ambiguous_cell_span")

    if any("overlapping_cells" in row.warnings for row in rows):
        warnings.append("overlapping_cells")

    if any(cell.semantic and not cell.source_ids
           for row in rows for cell in row.cells):
        warnings.append("source_mapping_partial")

    box = (min(row.bbox[0] for row in rows), min(row.bbox[1] for row in rows),
           max(row.bbox[2] for row in rows), max(row.bbox[3] for row in rows))

    # Auto-accept is deliberately narrow. Without a caption or a header row a
    # regular block of labels -- a diagram, a bulleted list -- looks exactly
    # like a small table, so it goes to review instead.

    vouched = bool(table_caption) or (header is HeaderCandidate.TRUE
                                      and len(rows) >= _MIN_AUTO_ROWS)
    status = (GeometryStatus.AUTO_GEOMETRY_OK if not warnings and vouched
              else GeometryStatus.NEEDS_REVIEW)

    if not warnings and not vouched:
        evidence.append("no caption, and too few rows to accept on shape alone")

    table_id = _identifier("tbl", standard_id, revision, corpus_fingerprint,
                           layout_fingerprint, page,
                           [round(value, 2) for value in box],
                           [list(line) for line in grid])
    region_sources = tuple(dict.fromkeys(supporting))

    if 0 < len(region_sources) <= _MAX_REGION_SOURCES:
        rows = [replace(row, cells=tuple(
            cell if cell.source_ids or not cell.semantic else replace(
                cell, source_ids=region_sources,
                provenance=ProvenanceQuality.TABLE_REGION_INHERITED)
            for cell in row.cells)) for row in rows]

    return StandardTableCandidate(
        schema_version=STRUCTURE_SCHEMA_VERSION, table_id=table_id,
        standard_id=standard_id, revision=revision,
        corpus_fingerprint=corpus_fingerprint, layout_fingerprint=layout_fingerprint,
        structure_extractor_version=STRUCTURE_EXTRACTOR_VERSION,
        page_start=page, page_end=page,
        bbox=tuple(round(value, 3) for value in box),
        row_count=len(rows), column_count=len(centres), rows=tuple(rows),
        caption_source_ids=table_caption,
        supporting_source_ids=region_sources,
        evidence=tuple(evidence), warnings=tuple(dict.fromkeys(warnings)),
        geometry_status=status)


def _column_positions(table: StandardTableCandidate) -> list[float]:
    columns: dict[int, list[float]] = {}

    for row in table.rows:
        for cell in row.cells:
            columns.setdefault(cell.column_index, []).append(cell.bbox[0])

    return [statistics.median(columns[index]) for index in sorted(columns)]


def _page_size(artifact: Mapping[str, object], page: int) -> Sequence[float]:
    for entry in artifact["pages"]:
        if entry["page"] == page:
            return entry["size"]

    raise StandardStructureError(f"layout artifact has no page {page}")


def _continuations(tables: Sequence[StandardTableCandidate],
                   artifact: Mapping[str, object]) -> list[StandardTableContinuation]:
    """Propose, never merge: a continuation is recorded as a link with evidence."""
    by_page: dict[int, list[StandardTableCandidate]] = {}

    for table in tables:
        by_page.setdefault(table.page_start, []).append(table)

    found: list[StandardTableContinuation] = []

    for table in tables:
        height = _page_size(artifact, table.page_start)[3]

        if table.bbox[3] < height * _PAGE_BOTTOM_SHARE:
            continue

        for other in by_page.get(table.page_end + 1, ()):
            next_height = _page_size(artifact, other.page_start)[3]

            if other.bbox[1] > next_height * _PAGE_TOP_SHARE:
                continue

            if other.column_count != table.column_count:
                continue

            here, there = _column_positions(table), _column_positions(other)
            width = _page_size(artifact, table.page_start)[2]
            tolerance = width * 0.02
            drift = [abs(a - b) for a, b in zip(here, there)]

            if not drift or max(drift) > tolerance:
                continue

            evidence = [
                "table reaches the bottom of its page",
                "next table begins at the top of the following page",
                f"{len(drift)} column positions agree within {max(drift):.1f}pt",
            ]
            warnings: list[str] = []
            repeated = ([cell.text for cell in table.rows[0].cells]
                        == [cell.text for cell in other.rows[0].cells])

            if repeated:
                evidence.append("header row repeated on the continuation")
            else:
                warnings.append("continuation_uncertain")

            found.append(StandardTableContinuation(
                from_table_id=table.table_id, to_table_id=other.table_id,
                evidence=tuple(evidence), warnings=tuple(warnings)))

    return found


def _bitfield_for(table: StandardTableCandidate) -> StandardBitfieldCandidate | None:
    """Only geometry: a run of consecutive integer labels and what spans them."""
    grid = table.grid()

    for row in table.rows:
        values = [""] * table.column_count
        boxes: dict[int, tuple[float, float, float, float]] = {}

        for cell in row.cells:
            if 0 <= cell.column_index < table.column_count:
                values[cell.column_index] = cell.text
                boxes[cell.column_index] = cell.bbox

        run = _bit_label_run(values)

        if run is None:
            continue

        bit_labels = tuple(
            StandardBitLabel(value=int(values[index]), column_index=index,
                             bbox=boxes[index])
            for index in range(*run)
            if values[index] and _INTEGER.match(values[index]) and index in boxes)

        if len(bit_labels) < _MIN_BIT_LABELS:
            continue

        labels = tuple(label.value for label in bit_labels)
        spans: list[StandardBitfieldSpan] = []

        for other in table.rows:
            if other.row_index == row.row_index:
                continue

            for cell in other.cells:
                covered = tuple(
                    label.value for label in bit_labels
                    if cell.bbox[0] <= (label.bbox[0] + label.bbox[2]) / 2 <= cell.bbox[2])

                if len(covered) < 2:
                    continue

                spans.append(StandardBitfieldSpan(
                    text=cell.text, row_index=cell.row_index, bbox=cell.bbox,
                    covered_labels=covered, source_ids=cell.source_ids,
                    warnings=cell.warnings))

        descending = labels[0] > labels[-1]
        evidence = (f"{len(bit_labels)} consecutive integer labels",
                    "descending" if descending else "ascending",
                    f"{len(spans)} cells span more than one label")
        warnings = () if spans else ("no cell spans a label range",)

        return StandardBitfieldCandidate(
            bitfield_id=_identifier("bit", table.table_id, row.row_index),
            table_id=table.table_id, page=table.page_start,
            label_row_index=row.row_index, bit_labels=bit_labels,
            spans=tuple(spans), evidence=evidence, warnings=warnings)

    return None


def extract_structures(units: Iterable, artifact: Mapping[str, object], *,
                       standard_id: str, revision: str,
                       corpus_fingerprint: str) -> StandardStructureSet:
    """Propose table and bitfield geometry for one revision. Reads only."""
    from dataclasses import replace

    if not isinstance(artifact, Mapping) or not artifact.get("pages"):
        raise StandardStructureError("layout artifact has no pages")

    layout_fingerprint = artifact.get("layout_fingerprint")

    if not isinstance(layout_fingerprint, str):
        raise StandardStructureError("layout artifact has no fingerprint")

    units = tuple(units)
    sources = _page_sources(units)
    tables: list[StandardTableCandidate] = []

    for page in artifact["pages"]:
        bands = _bands(_page_fragments(page))
        entries = sources.get(page["page"], [])
        excluded = _not_table_rows(units, page["page"])

        for region in _regions(bands, excluded):
            table = _build_table(
                region, page=page["page"], page_size=page["size"], bands=bands,
                entries=entries, standard_id=standard_id, revision=revision,
                corpus_fingerprint=corpus_fingerprint,
                layout_fingerprint=layout_fingerprint)

            if table is not None:
                tables.append(table)

    continuations = _continuations(tables, artifact)
    onward: dict[str, list[str]] = {}

    for link in continuations:
        onward.setdefault(link.from_table_id, []).append(link.to_table_id)

    tables = [replace(table, continuation_ids=tuple(onward.get(table.table_id, ())),
                      page_end=table.page_end)
              for table in tables]
    bitfields = [item for item in (_bitfield_for(table) for table in tables)
                 if item is not None]

    return StandardStructureSet(tuple(tables), tuple(bitfields),
                                tuple(continuations))


def validate_structures(structures: StandardStructureSet, *,
                        artifact: Mapping[str, object],
                        source_ids: frozenset[str]) -> None:
    """Geometry and provenance integrity. Not normative compliance."""
    pages = {entry["page"]: entry["size"] for entry in artifact["pages"]}
    known = {table.table_id for table in structures.tables}

    for table in structures.tables:
        if table.page_start not in pages or table.page_end not in pages:
            raise StandardStructureError("table references a page that does not exist")

        width, height = pages[table.page_start][2], pages[table.page_start][3]

        for box in [table.bbox] + [row.bbox for row in table.rows] + [
                cell.bbox for row in table.rows for cell in row.cells]:
            if not all(isinstance(value, (int, float)) and math.isfinite(value)
                       for value in box):
                raise StandardStructureError("bbox is not finite")

            if box[2] < box[0] or box[3] < box[1]:
                raise StandardStructureError("bbox is inverted")

            if box[0] < -1 or box[1] < -1 or box[2] > width + 1 or box[3] > height + 1:
                raise StandardStructureError("bbox falls outside its page")

        if [row.row_index for row in table.rows] != sorted(
                row.row_index for row in table.rows):
            raise StandardStructureError("rows are not in order")

        for row in table.rows:
            indexes = [cell.column_index for cell in row.cells]

            if indexes != sorted(indexes):
                raise StandardStructureError("columns are not in order")

            for cell in row.cells:
                if not set(cell.source_ids) <= source_ids:
                    raise StandardStructureError("cell cites an unknown source")

        if not set(table.supporting_source_ids) <= source_ids:
            raise StandardStructureError("table cites an unknown source")

        if not set(table.caption_source_ids) <= source_ids:
            raise StandardStructureError("caption cites an unknown source")

        if not set(table.continuation_ids) <= known:
            raise StandardStructureError("continuation names an unknown table")

    for link in structures.continuations:
        if link.from_table_id not in known or link.to_table_id not in known:
            raise StandardStructureError("continuation names an unknown table")

    for bitfield in structures.bitfields:
        if bitfield.table_id not in known:
            raise StandardStructureError("bitfield names an unknown table")

        values = [label.value for label in bitfield.bit_labels]
        steps = {later - earlier for earlier, later in zip(values, values[1:])}

        if steps and steps not in ({1}, {-1}):
            raise StandardStructureError("bit labels are not a consecutive run")


def unresolved_semantic_cells(table: StandardTableCandidate,
                              ) -> tuple[StandardTableCell, ...]:
    """Text-bearing cells that carry no canonical source at all."""
    return tuple(cell for row in table.rows for cell in row.cells
                 if cell.semantic and not cell.source_ids)


def promotion_state(table: StandardTableCandidate) -> str:
    """Whether a later phase could promote this geometry to normative meaning.

    Provenance is the gate: a semantic cell with no canonical source behind it
    would become an uncited assertion, so the table stays blocked. Blank and
    punctuation-only cells are structural and do not carry meaning to cite.
    """
    return ("BLOCKED_UNRESOLVED_PROVENANCE" if unresolved_semantic_cells(table)
            else "PROVENANCE_SUFFICIENT")


def provenance_counts(tables: Sequence[StandardTableCandidate]) -> dict[str, int]:
    from collections import Counter

    counts = Counter()

    for table in tables:
        for row in table.rows:
            for cell in row.cells:
                counts[cell.provenance.value] += 1

                if cell.semantic and not cell.source_ids:
                    counts["UNRESOLVED_SEMANTIC"] += 1

    counts["TOTAL"] = sum(counts[key] for key in
                          (item.value for item in ProvenanceQuality))

    return dict(counts)


def structure_fingerprint(structures: StandardStructureSet) -> str:
    return sha256_json({
        "structure_schema_version": STRUCTURE_SCHEMA_VERSION,
        "structure_extractor_version": STRUCTURE_EXTRACTOR_VERSION,
        "structures": structures.to_dict(),
    })
