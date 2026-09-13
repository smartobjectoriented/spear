"""Synthetic bitfield tables with explicit word numbering.

Built as structures directly, so the word column, the printed lines and the
overlapping y-bands are exactly what each test needs rather than whatever a
renderer happens to produce. No licensed normative text appears here.
"""

from __future__ import annotations

from standard_structure import (
    STRUCTURE_EXTRACTOR_VERSION, STRUCTURE_SCHEMA_VERSION, GeometryStatus,
    HeaderCandidate, ProvenanceQuality, StandardBitfieldCandidate,
    StandardBitLabel, StandardBitfieldSpan, StandardTableCandidate,
    StandardTableCell, StandardTableRow,
)

SID, REV = "WRD", "R1"
FINGERPRINT = "a" * 64
LAYOUT = "b" * 64
SOURCE = "std-0123456789abcdef0123456789abcdef"
# One printed line, and the gap to the next one.
LINE_HEIGHT = 12.0
LINE_PITCH = 26.0
RULER_TOP = 60.0
FIRST_LINE = 100.0


def line_y(line: int) -> tuple[float, float]:
    top = FIRST_LINE + line * LINE_PITCH
    return top, top + LINE_HEIGHT


def cell(row: int, column: int, text: str, x0: float, x1: float, *,
         line: int = 0, top: float | None = None, bottom: float | None = None,
         sources: tuple[str, ...] = (SOURCE,),
         provenance: ProvenanceQuality = ProvenanceQuality.DIRECT_TEXT_MATCH,
         ) -> StandardTableCell:
    y0, y1 = line_y(line) if top is None else (top, bottom)
    return StandardTableCell(
        cell_id=f"cel-{row:02d}{column:02d}{abs(hash(text)) % 10 ** 8:08d}",
        row_index=row, column_index=column, page=1, bbox=(x0, y0, x1, y1),
        text=text, source_ids=sources,
        provenance=provenance if sources else ProvenanceQuality.UNRESOLVED,
        semantic=bool(text.strip()))


def ruler_row(row: int = 0, *, heading: str | None = "Word",
              labels: range = range(31, -1, -1)) -> StandardTableRow:
    """A heading and a full ruler on one line, the way a diagram prints them."""
    cells = []
    if heading is not None:
        cells.append(cell(row, 0, heading, 60.0, 92.0,
                          top=RULER_TOP, bottom=RULER_TOP + LINE_HEIGHT))
    for offset, value in enumerate(labels):
        left = 100.0 + offset * 14.0
        cells.append(cell(row, offset + 1, str(value), left, left + 8.0,
                          top=RULER_TOP, bottom=RULER_TOP + LINE_HEIGHT))
    return StandardTableRow(row_index=row, page=1,
                            bbox=(60.0, RULER_TOP, 560.0, RULER_TOP + LINE_HEIGHT),
                            cells=tuple(cells),
                            header_candidate=HeaderCandidate.TRUE)


def word_row(row: int, words: tuple[tuple[int | str, tuple[str, ...]], ...], *,
             first_line: int = 0, word_column: bool = True) -> StandardTableRow:
    """One band holding several printed lines, each line one word of the packet.

    The word numbers land in one cell, exactly as a y-band merge produces them.
    """
    cells = []
    if word_column:
        top, _ = line_y(first_line)
        _, bottom = line_y(first_line + len(words) - 1)
        cells.append(cell(row, 0, " ".join(str(name) for name, _ in words),
                          64.0, 88.0, top=top, bottom=bottom))
    column = 1
    for offset, (_, labels) in enumerate(words):
        for position, text in enumerate(labels):
            left = 100.0 + position * 140.0
            cells.append(cell(row, column, text, left, left + 120.0,
                              line=first_line + offset))
            column += 1
    top, _ = line_y(first_line)
    _, bottom = line_y(first_line + len(words) - 1)
    return StandardTableRow(row_index=row, page=1, bbox=(60.0, top, 560.0, bottom),
                            cells=tuple(cells),
                            header_candidate=HeaderCandidate.FALSE)


def table(rows: tuple[StandardTableRow, ...], *,
          warnings: tuple[str, ...] = ()) -> StandardTableCandidate:
    columns = max((cell.column_index for row in rows for cell in row.cells),
                  default=0) + 1
    return StandardTableCandidate(
        schema_version=STRUCTURE_SCHEMA_VERSION, table_id="tbl-wordfixture00001",
        standard_id=SID, revision=REV, corpus_fingerprint=FINGERPRINT,
        layout_fingerprint=LAYOUT,
        structure_extractor_version=STRUCTURE_EXTRACTOR_VERSION,
        page_start=1, page_end=1, bbox=(60.0, RULER_TOP, 560.0, 400.0),
        row_count=len(rows), column_count=columns, rows=rows,
        supporting_source_ids=(SOURCE,), warnings=warnings,
        geometry_status=GeometryStatus.NEEDS_REVIEW)


def bitfield(source: StandardTableCandidate, *, label_row: int = 0,
             spans: tuple[StandardBitfieldSpan, ...] = (),
             ) -> StandardBitfieldCandidate:
    row = next(item for item in source.rows if item.row_index == label_row)
    labels = tuple(
        StandardBitLabel(value=int(item.text), column_index=item.column_index,
                         bbox=item.bbox)
        for item in row.cells if item.text.strip().isdigit())
    return StandardBitfieldCandidate(
        bitfield_id="bit-0123456789abcdef", table_id=source.table_id, page=1,
        label_row_index=label_row, bit_labels=labels, spans=spans)


def span_of(source: StandardTableCandidate, text: str, *,
            covered: tuple[int, ...] = ()) -> StandardBitfieldSpan:
    """A measured span standing for one of the table's cells."""
    found = next(cell for row in source.rows for cell in row.cells
                 if cell.text == text)
    return StandardBitfieldSpan(
        text=found.text, row_index=found.row_index, bbox=found.bbox,
        covered_labels=covered, source_ids=found.source_ids)


# --------------------------------------------------------------------------
# raw word geometry, for the cases where a cell merges two printed lines
# --------------------------------------------------------------------------

def raw(text: str, x0: float, x1: float, *, line: int | None = None,
        on_ruler: bool = False) -> dict:
    """One word box, the way the layout artifact records it."""
    if on_ruler:
        y0, y1 = RULER_TOP, RULER_TOP + LINE_HEIGHT
    else:
        y0, y1 = line_y(line or 0)
    return {"text": text, "bbox": [x0, y0, x1, y1]}


def merged_cell_table(label: str, prefix: tuple[str, ...] = ("18",), *,
                      prefix_on_ruler: bool = True, word: int = 0,
                      ) -> tuple[object, tuple[dict, ...]]:
    """A table whose field cell also swallowed words from another printed line.

    The column pass groups by column and ignores which line a word sat on, so a
    wide label can arrive carrying the ruler numbers printed above it. The cell
    text and bbox here are exactly what that produces.
    """
    merged = " ".join(prefix + (label,))
    top = RULER_TOP if prefix_on_ruler else line_y(0)[0]
    rows = (
        ruler_row(),
        StandardTableRow(
            row_index=1, page=1, bbox=(60.0, top, 560.0, line_y(0)[1]),
            cells=(cell(1, 0, str(word), 64.0, 88.0),
                   cell(1, 1, merged, 240.0, 400.0, top=top,
                        bottom=line_y(0)[1])),
            header_candidate=HeaderCandidate.FALSE),
    )
    words = tuple(
        [raw(str(value), 100.0 + offset * 14.0, 108.0 + offset * 14.0,
             on_ruler=True)
         for offset, value in enumerate(range(31, -1, -1))]
        + [raw(text, 240.0 + index * 20.0, 258.0 + index * 20.0,
               on_ruler=prefix_on_ruler, line=0)
           for index, text in enumerate(prefix)]
        + [raw(label, 240.0, 400.0, line=0),
           raw(str(word), 64.0, 88.0, line=0)])
    return table(rows), words


def ruler_run_table(run: str, field: str = "FIELD_A (15-0)", *, word: int = 0,
                    on_ruler: bool = True):
    """A word row that also swallowed a cell reaching up to the ruler's line.

    The column pass groups by column without regard for printed line, so a cell
    can span from the ruler down into a word's row. When its text is a run of
    ruler numbers it is still ruler; when it is a name it is a field nobody
    positioned. Both shapes are built here from the same geometry.
    """
    top = RULER_TOP if on_ruler else line_y(0)[0]
    rows = (
        ruler_row(),
        StandardTableRow(
            row_index=1, page=1, bbox=(60.0, top, 560.0, line_y(0)[1]),
            cells=(cell(1, 0, str(word), 64.0, 88.0),
                   cell(1, 1, run, 240.0, 400.0, top=top, bottom=line_y(0)[1]),
                   cell(1, 2, field, 420.0, 540.0)),
            header_candidate=HeaderCandidate.FALSE),
    )
    return table(rows)


def two_word_table(first: tuple[str, ...], second: tuple[str, ...], *,
                   word_column: bool = True):
    """Two numbered words, each on its own printed line with its own labels.

    A word whose labels state no range yields no field, which is the shape
    that used to disappear from structural completeness altogether.
    """
    rows = (ruler_row(),
            word_row(1, ((0, first),), first_line=0, word_column=word_column),
            word_row(2, ((1, second),), first_line=2, word_column=word_column))
    return table(rows)
