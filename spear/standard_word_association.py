"""Which packet word a field belongs to, taken from what the diagram says.

A packet diagram is a stack of words, and a bit range is local to its word.
Bits 23..12 of word 0 and bits 23..12 of word 1 are different fields in
different places, and nothing about them overlaps.

STD2A grouped a diagram into rows by horizontal band. Where two words are set
close together the band swallows both, and two perfectly good local ranges then
look like two fields fighting over the same bits. The fix is not a better band:
these diagrams carry a Word column, and the document's own word numbers outrank
any measurement of where a line sits.

So word identity is read, in order, from an explicit word index, from an
explicit word label, and only then from the geometric row -- which is recorded
as the weaker evidence it is. When a row carries several word numbers and its
lines cannot be matched to them one for one, the association is UNKNOWN and
promotion stops there rather than guessing.

This module also decides which cells of a bitfield table are field candidates
at all. Since STD2C a stated range carries its own position, so a cell no
longer has to span two ruler labels to name a field.

Cells are grouped into columns without regard for which printed line a word sat
on, so a wide label can arrive carrying a ruler number from the line above it.
The bits are unharmed -- they come from the label's own text -- but the name
would be wrong, so a stated field's label is cut back to the printed lines that
are not the ruler's. That is done here rather than in the geometry, which stays
exactly as extracted.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Mapping, Sequence

from standard_stated_range import MAX_BIT, RangeContext, stated_ranges_for_cell


WORD_ASSOCIATION_MODEL_VERSION = "explicit-words-v1"

# The bare noun heading a column of word numbers. "1 Word" and "2 Words" are
# counts and never match this, because a count puts its number first.

_WORD_HEADING = re.compile(r"^words?(?:\s*(?:#|no\.?|number|index))?$", re.I)

# A marker naming its own word, wherever it sits.

_INLINE_WORD_INDEX = re.compile(r"^word\s+(\d{1,3})$", re.I)
_INLINE_WORD_LABEL = re.compile(r"^word\s+([A-Za-z][A-Za-z0-9_\-]{0,30})$", re.I)

# A cell of the word column: one or more word numbers, nothing else.

_WORD_TOKENS = re.compile(r"^\d{1,3}(?:\s+\d{1,3})*$")
_RULER_LABEL = re.compile(r"^\d{1,3}$")

# A ruler prints one number per position, but the column pass can gather a run
# of them into one cell. Such a cell is still ruler, not a name.

_MIN_RULER_RUN = 2

# A word number is a small number. A four-digit one is something else.

_MAX_WORD_INDEX = 999

# A cell taller than this multiple of the row's usual line height covers more
# than one line, so it cannot be assigned to one of them.

_MULTI_LINE_HEIGHT = 1.5

RULER_LINE_EXCLUDED = "ruler_line_excluded"
NO_WORD_INDEX_IN_ROW = "no_word_index_in_row"
WORD_LINE_COUNT_MISMATCH = "word_line_count_mismatch"
WORD_LINES_NOT_SEPARABLE = "word_lines_not_separable"


class AssociationSource(StrEnum):
    """Where a field's word identity came from, strongest first."""

    EXPLICIT_WORD_INDEX = "EXPLICIT_WORD_INDEX"
    EXPLICIT_WORD_LABEL = "EXPLICIT_WORD_LABEL"
    GEOMETRIC_ROW = "GEOMETRIC_ROW"
    UNKNOWN = "UNKNOWN"


EXPLICIT_SOURCES = (AssociationSource.EXPLICIT_WORD_INDEX,
                    AssociationSource.EXPLICIT_WORD_LABEL)


@dataclass(frozen=True)
class StandardWordAssociation:
    """One field's word, and the evidence that put it there."""

    word_index: int | None
    word_label: str | None
    association_source: AssociationSource
    source_cell_ids: tuple[str, ...] = ()
    supporting_source_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        return self.association_source is not AssociationSource.UNKNOWN

    @property
    def explicit(self) -> bool:
        return self.association_source in EXPLICIT_SOURCES

    @property
    def key(self) -> tuple[object, object]:
        """What two fields must share to be able to overlap at all."""
        return (self.word_index, self.word_label)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["association_source"] = self.association_source.value

        for name in ("source_cell_ids", "supporting_source_ids", "warnings"):
            value[name] = list(getattr(self, name))

        return value


UNRESOLVED = StandardWordAssociation(None, None, AssociationSource.UNKNOWN)


# --------------------------------------------------------------------------
# reading the word column
# --------------------------------------------------------------------------

def _text(cell: Mapping[str, object]) -> str:
    return str(cell.get("text", "")).strip()


def _box(cell: Mapping[str, object]) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in cell["bbox"])  # type: ignore[return-value]


def _centre(cell: Mapping[str, object]) -> float:
    box = _box(cell)
    return (box[1] + box[3]) / 2


def is_word_count(text: str) -> bool:
    """A size, not an identity: the number comes before the noun."""
    return bool(re.search(r"\b\d+\s+words?\b", text, re.I))


def word_heading(table: Mapping[str, object]) -> Mapping[str, object] | None:
    """The bare 'Word' heading of a word column, if the diagram has one."""
    headings = [cell for row in table["rows"] for cell in row["cells"]
                if _WORD_HEADING.match(_text(cell))]

    if not headings:
        return None

    return min(headings, key=lambda cell: (_box(cell)[0], _box(cell)[1]))


def word_index_cells(table: Mapping[str, object],
                     ) -> tuple[Mapping[str, object] | None, list[Mapping[str, object]]]:
    """The heading and the cells of word numbers standing under it.

    Matched by horizontal overlap rather than by column index: the heading is a
    word and the values are digits, so the column pass often splits them even
    though the page has them in one column.
    """
    heading = word_heading(table)

    if heading is None:
        return None, []

    left, right = _box(heading)[0], _box(heading)[2]
    found = []

    for row in table["rows"]:
        for cell in row["cells"]:
            if cell is heading or not _text(cell):
                continue

            box = _box(cell)

            if (box[0] >= left - 1 and box[2] <= right + 1
                    and _WORD_TOKENS.match(_text(cell))
                    and all(int(token) <= _MAX_WORD_INDEX
                            for token in _text(cell).split())):
                found.append(cell)

    return heading, found


def is_pure_numeric_ruler_cell(text: str, box: Sequence[float] | None = None,
                               band: tuple[float, float] | None = None) -> bool:
    """Whether a cell is a run of ruler numbers rather than something named.

    The column pass groups by column and can pull several ruler positions into
    one cell, which then looks like a label the diagram never positioned and
    would argue a word is incompletely described. It is not a label: it is the
    ruler.

    The test is deliberately narrow. Apart from a leading word-column heading,
    every token must be an integer, there must be at least two of them, they must all be plausible bit numbers, they must
    step by exactly one in a single direction, and the cell must sit on the
    ruler's own printed line. Anything else -- a gap in the numbering, a stride
    of two, one word among the digits, or no ruler to sit on -- is kept as
    possibly meaningful, because mistaking a name for the ruler hides a field
    while mistaking the ruler for a name only adds a warning.
    """

    if box is None or band is None:
        return False

    tokens = text.split()

    # The column pass can gather the word column's own heading together with
    # the ruler beside it. Dropping the heading leaves the same ruler run.

    if tokens and _WORD_HEADING.match(tokens[0]):
        tokens = tokens[1:]

    if len(tokens) < _MIN_RULER_RUN or not all(item.isdigit() for item in tokens):
        return False

    values = [int(item) for item in tokens]

    if any(value > MAX_BIT for value in values):
        return False

    steps = {later - earlier for earlier, later in zip(values, values[1:])}

    if steps not in ({1}, {-1}):
        return False

    top, bottom = float(box[1]), float(box[3])

    return not (bottom <= band[0] or top >= band[1])


def _lines(cells: Sequence[Mapping[str, object]],
           ) -> list[list[Mapping[str, object]]] | None:
    """Split cells into printed lines, or None if any cell covers several."""

    if not cells:
        return []

    heights = [_box(cell)[3] - _box(cell)[1] for cell in cells]
    usual = statistics.median(heights)

    if usual <= 0:
        return None

    if any(height > usual * _MULTI_LINE_HEIGHT for height in heights):
        # A wrapped label belongs to lines the split cannot tell apart.

        return None

    ordered = sorted(cells, key=lambda cell: (_centre(cell), _box(cell)[0]))
    groups: list[list[Mapping[str, object]]] = [[ordered[0]]]

    for cell in ordered[1:]:
        if _centre(cell) - _centre(groups[-1][-1]) > usual / 2:
            groups.append([cell])
        else:
            groups[-1].append(cell)

    return groups


def _word_tokens(cells: Sequence[Mapping[str, object]]) -> list[int]:
    """Every word number in the row's word cells, in reading order."""
    ordered = sorted(cells, key=lambda cell: (_box(cell)[1], _box(cell)[0]))

    return [int(token) for cell in ordered for token in _text(cell).split()]


def _sources(cells: Sequence[Mapping[str, object]]) -> tuple[tuple[str, ...],
                                                            tuple[str, ...]]:
    ids = tuple(dict.fromkeys(str(cell["cell_id"]) for cell in cells))
    sources = tuple(dict.fromkeys(
        str(value) for cell in cells for value in cell.get("source_ids", ())))

    return ids, sources


def associate_words(table: Mapping[str, object],
                    ) -> dict[str, StandardWordAssociation]:
    """Give every cell of a bitfield table the word it belongs to, by cell id."""
    heading, index_cells = word_index_cells(table)
    marked = {str(cell["cell_id"]) for cell in index_cells}

    if heading is not None:
        marked.add(str(heading["cell_id"]))

    associations: dict[str, StandardWordAssociation] = {}

    for row in table["rows"]:
        members = [cell for cell in row["cells"] if _text(cell)]
        word_cells = [cell for cell in members
                      if str(cell["cell_id"]) in marked
                      and _WORD_TOKENS.match(_text(cell))]
        others = [cell for cell in members if str(cell["cell_id"]) not in marked]

        # A marker naming its own word speaks for the whole row.

        inline_index = inline_label = None

        for cell in list(others):
            match = _INLINE_WORD_INDEX.match(_text(cell))

            if match is not None:
                inline_index, marker = int(match.group(1)), cell
                others.remove(cell)
                word_cells = word_cells or [marker]

                break

            match = _INLINE_WORD_LABEL.match(_text(cell))

            if match is not None and not is_word_count(_text(cell)):
                inline_label, marker = match.group(1), cell
                others.remove(cell)
                word_cells = word_cells or [marker]

                break

        ids, sources = _sources(word_cells)

        if inline_label is not None:
            association = StandardWordAssociation(
                None, inline_label, AssociationSource.EXPLICIT_WORD_LABEL,
                ids, sources)

            for cell in others:
                associations[str(cell["cell_id"])] = association

            continue

        tokens = ([inline_index] if inline_index is not None
                  else _word_tokens(word_cells))

        if not tokens:
            warning = (NO_WORD_INDEX_IN_ROW,) if index_cells else ()

            for cell in others:
                associations[str(cell["cell_id"])] = StandardWordAssociation(
                    int(row["row_index"]), None, AssociationSource.GEOMETRIC_ROW,
                    warnings=warning)

            continue

        if len(tokens) == 1:
            association = StandardWordAssociation(
                tokens[0], None, AssociationSource.EXPLICIT_WORD_INDEX, ids, sources)

            for cell in others:
                associations[str(cell["cell_id"])] = association

            continue

        # Several words share one band. Their lines are still distinct on the
        # page, so match line to word number in reading order -- and only if
        # there are exactly as many lines as the document named words.

        groups = _lines(others)

        if groups is None or len(groups) != len(tokens):
            reason = (WORD_LINES_NOT_SEPARABLE if groups is None
                      else WORD_LINE_COUNT_MISMATCH)

            for cell in others:
                associations[str(cell["cell_id"])] = StandardWordAssociation(
                    None, None, AssociationSource.UNKNOWN, ids, sources, (reason,))

            continue

        for index, group in zip(tokens, groups):
            association = StandardWordAssociation(
                index, None, AssociationSource.EXPLICIT_WORD_INDEX, ids, sources)

            for cell in group:
                associations[str(cell["cell_id"])] = association

    return associations


# --------------------------------------------------------------------------
# keeping a ruler line out of a field's name
# --------------------------------------------------------------------------

def page_words(layout: Mapping[str, object] | None,
               page: int) -> tuple[Mapping[str, object], ...]:
    """Every word box the extractor found on one page, in no particular order."""

    if not layout:
        return ()

    for entry in layout.get("pages", ()):
        if int(entry.get("page", -1)) != int(page):
            continue

        return tuple(word for block in entry.get("blocks", ())
                     for line in block.get("lines", ())
                     for word in line.get("words", ()))

    return ()


def ruler_band(bit_labels: Sequence[Mapping[str, object]],
               ) -> tuple[float, float] | None:
    """The vertical strip the visible ruler occupies, from the labels themselves."""
    boxes = [tuple(float(value) for value in item["bbox"]) for item in bit_labels]

    if not boxes:
        return None

    return (min(box[1] for box in boxes), max(box[3] for box in boxes))


def _words_within(words: Sequence[Mapping[str, object]],
                  box: Sequence[float], tolerance: float = 0.5,
                  ) -> list[Mapping[str, object]]:
    found = []

    for word in words:
        x0, y0, x1, y1 = (float(value) for value in word["bbox"])

        if (x0 >= box[0] - tolerance and x1 <= box[2] + tolerance
                and y0 >= box[1] - tolerance and y1 <= box[3] + tolerance):
            found.append(word)

    return found


def isolate_from_ruler_line(text: str, box: Sequence[float],
                            words: Sequence[Mapping[str, object]],
                            band: tuple[float, float] | None,
                            ) -> str | None:
    """The label without the words that sat on the ruler's printed line.

    Returns None when there is nothing to do: no raw geometry, no ruler band,
    the cell covers a single printed line, or every line is the ruler's. Only
    lines that actually overlap the ruler are dropped, so a genuine label that
    wraps onto a second line keeps both of them.
    """

    if band is None or not words:
        return None

    inside = _words_within(words, box)

    if len(inside) < 2:
        return None

    kept, dropped = [], []

    for word in inside:
        _, y0, _, y1 = (float(value) for value in word["bbox"])

        # An overlap with the ruler's strip is the evidence; nothing about the
        # word's own text is consulted, so a numeric field name is safe.

        (dropped if not (y1 <= band[0] or y0 >= band[1]) else kept).append(word)

    if not dropped or not kept:
        return None

    ordered = sorted(kept, key=lambda word: (float(word["bbox"][1]),
                                             float(word["bbox"][0])))
    rebuilt = re.sub(r"\s+", " ", " ".join(
        str(word["text"]) for word in ordered)).strip()

    return rebuilt or None


# --------------------------------------------------------------------------
# which cells can name a field
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldCandidate:
    """A cell that could be a semantic field, and the evidence it carries."""

    index: int
    cell_id: str
    text: str
    row_index: int
    column_index: int
    covered_labels: tuple[int, ...]
    source_ids: tuple[str, ...]
    provenance: str
    from_geometric_span: bool
    warnings: tuple[str, ...] = ()


def _span_cell(table: Mapping[str, object], span: Mapping[str, object]):
    for row in table["rows"]:
        if int(row["row_index"]) != int(span["row_index"]):
            continue

        for cell in row["cells"]:
            if (list(cell["bbox"]) == list(span["bbox"])
                    and _text(cell) == str(span["text"]).strip()):
                return cell

    return None


def _cell_of(table: Mapping[str, object], cell_id: str):
    for row in table["rows"]:
        for cell in row["cells"]:
            if str(cell["cell_id"]) == cell_id:
                return cell

    return None


def _stated_label(text: str, cell, context: RangeContext,
                  band: tuple[float, float] | None,
                  words: Sequence[Mapping[str, object]]) -> tuple[str, bool]:
    """A stated field's label, cut back to its own printed line if need be.

    Only a label that states its own bits is touched, and only when dropping
    the ruler's line leaves the stated range intact. A measured span keeps the
    text it always had.
    """

    def stated(value: str):
        return [item for item in stated_ranges_for_cell(
            value, cell_id=str(cell["cell_id"]) if cell is not None else "",
            source_ids=tuple(str(item) for item in cell.get("source_ids", ()))
            if cell is not None else (),
            provenance=str(cell["provenance"]) if cell is not None else "UNRESOLVED",
            context=context) if item.positional]

    if cell is None or not stated(text):
        return text, False

    rebuilt = isolate_from_ruler_line(text, tuple(cell["bbox"]), words, band)

    if rebuilt is None or rebuilt == text or not stated(rebuilt):
        return text, False

    return rebuilt, True


def unpositioned_labels(bitfield: Mapping[str, object],
                        table: Mapping[str, object], *,
                        words: Sequence[Mapping[str, object]] = (),
                        ) -> tuple[Mapping[str, object], ...]:
    """Cited cells the diagram shows as names but never positions.

    These are what a normative rule elsewhere may be talking about. A cell is
    only offered if it is cited, is not the ruler or the word column, and is
    not already a field candidate in its own right.
    """
    chosen = {item.cell_id for item in field_candidates(bitfield, table, words=words)}
    heading, index_cells = word_index_cells(table)
    reserved = {str(cell["cell_id"]) for cell in index_cells}

    if heading is not None:
        reserved.add(str(heading["cell_id"]))

    band = ruler_band(bitfield.get("bit_labels", ()))
    found = []

    for row in table["rows"]:
        for cell in row["cells"]:
            cell_id = str(cell["cell_id"])
            text = _text(cell)

            if (cell_id in chosen or cell_id in reserved or not cell.get("semantic")
                    or not text or _RULER_LABEL.match(text)
                    or is_pure_numeric_ruler_cell(text, cell["bbox"], band)
                    or not cell.get("source_ids")):
                continue

            found.append(cell)

    return tuple(sorted(found, key=lambda cell: (int(cell["row_index"]),
                                                 int(cell["column_index"]))))


def field_candidates(bitfield: Mapping[str, object], table: Mapping[str, object],
                     *, words: Sequence[Mapping[str, object]] = (),
                     prose_cells: frozenset[str] = frozenset(),
                     value_local_cells: frozenset[str] = frozenset(),
                     ) -> tuple[FieldCandidate, ...]:
    """Every cell that could name a field: measured spans, then stated ranges.

    The geometric spans come first and keep their order, so a role list stays
    aligned with what an operator saw. After them come cells the ruler never
    reached: since STD2C a stated range carries its own position, so a label
    that states its bits is a field candidate whether or not its box happens to
    cover two ruler labels.

    Given the page's raw word boxes, a stated field's label is also cut back to
    its own printed line, so a ruler number from the line above does not end up
    inside the field's name.

    `value_local_cells` names cells whose range is stated in the coordinates of
    a wider value -- a 64-bit half printed as 63..32 -- and which STD2E has
    since proved belongs to a word. They are the diagram's own labels, so a
    reviewer classifies them like any other, but they are appended rather than
    interleaved so no candidate a reviewer already saw changes position.

    `prose_cells` names cells a normative rule has positioned from outside the
    diagram. They come last, so the order of everything a reviewer already saw
    is unchanged.
    """
    labels = tuple(int(item["value"]) for item in bitfield.get("bit_labels", ()))
    context = RangeContext(in_bitfield_region=True, ruler_labels=labels)
    band = ruler_band(bitfield.get("bit_labels", ()))
    heading, index_cells = word_index_cells(table)
    reserved = {str(cell["cell_id"]) for cell in index_cells}

    if heading is not None:
        reserved.add(str(heading["cell_id"]))

    found: list[FieldCandidate] = []
    seen: set[str] = set()

    for span in bitfield.get("spans", ()):
        cell = _span_cell(table, span)
        cell_id = str(cell["cell_id"]) if cell is not None else ""
        label, cleaned = _stated_label(str(span["text"]), cell, context, band, words)
        found.append(FieldCandidate(
            index=len(found), cell_id=cell_id, text=label,
            row_index=int(span["row_index"]),
            column_index=int(cell["column_index"]) if cell is not None else -1,
            covered_labels=tuple(int(value) for value in span["covered_labels"]),
            source_ids=tuple(str(value) for value in span.get("source_ids", ())),
            provenance=str(cell["provenance"]) if cell is not None else "UNRESOLVED",
            from_geometric_span=True,
            warnings=tuple(str(value) for value in span.get("warnings", ()))
            + ((RULER_LINE_EXCLUDED,) if cleaned else ())))

        if cell_id:
            seen.add(cell_id)

    extra: list[Mapping[str, object]] = []

    for row in table["rows"]:
        for cell in row["cells"]:
            cell_id = str(cell["cell_id"])

            if cell_id in seen or cell_id in reserved or not cell.get("semantic"):
                continue

            if _RULER_LABEL.match(_text(cell)):
                # A bare ruler label is the ruler, not a field.

                continue

            stated = stated_ranges_for_cell(
                _text(cell), cell_id=cell_id,
                source_ids=tuple(str(value) for value in cell.get("source_ids", ())),
                provenance=str(cell["provenance"]), context=context)

            if any(item.positional for item in stated):
                extra.append(cell)

    for cell in sorted(extra, key=lambda cell: (int(cell["row_index"]),
                                                int(cell["column_index"]))):
        label, cleaned = _stated_label(_text(cell), cell, context, band, words)
        found.append(FieldCandidate(
            index=len(found), cell_id=str(cell["cell_id"]), text=label,
            row_index=int(cell["row_index"]),
            column_index=int(cell["column_index"]),
            covered_labels=(),
            source_ids=tuple(str(value) for value in cell.get("source_ids", ())),
            provenance=str(cell["provenance"]), from_geometric_span=False,
            warnings=tuple(str(value) for value in cell.get("warnings", ()))
            + ((RULER_LINE_EXCLUDED,) if cleaned else ())))
        seen.add(str(cell["cell_id"]))

    # Named directly by cell id, so this never needs to ask which cells are
    # unpositioned -- which is the question that produced these two sets.

    for cell in sorted((cell for row in table["rows"] for cell in row["cells"]
                        if str(cell["cell_id"]) in value_local_cells),
                       key=lambda cell: (int(cell["row_index"]),
                                         int(cell["column_index"]))):
        cell_id = str(cell["cell_id"])

        if cell_id in seen:
            continue

        found.append(FieldCandidate(
            index=len(found), cell_id=cell_id, text=_text(cell),
            row_index=int(cell["row_index"]),
            column_index=int(cell["column_index"]), covered_labels=(),
            source_ids=tuple(str(value) for value in cell.get("source_ids", ())),
            provenance=str(cell["provenance"]), from_geometric_span=False,
            warnings=tuple(str(value) for value in cell.get("warnings", ()))))
        seen.add(cell_id)

    for cell in sorted((cell for row in table["rows"] for cell in row["cells"]
                        if str(cell["cell_id"]) in prose_cells),
                       key=lambda cell: (int(cell["row_index"]),
                                         int(cell["column_index"]))):
        cell_id = str(cell["cell_id"])

        if cell_id in seen:
            continue

        found.append(FieldCandidate(
            index=len(found), cell_id=cell_id, text=_text(cell),
            row_index=int(cell["row_index"]),
            column_index=int(cell["column_index"]), covered_labels=(),
            source_ids=tuple(str(value) for value in cell.get("source_ids", ())),
            provenance=str(cell["provenance"]), from_geometric_span=False,
            warnings=tuple(str(value) for value in cell.get("warnings", ()))))

    return tuple(found)
