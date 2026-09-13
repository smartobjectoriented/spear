"""The diagram cells behind a flattened columnar unit, read back out of layout.

standard.fetch hands the model a packet diagram's row as one line of text:

    1    Horizontal Beamwidth (15..0), Degrees        Vertical Beamwidth (15..0), Degrees

Two cells, side by side, each printing its own value-local range. Flattened,
the second (15..0) reads like the tail of a single 32-bit description, and the
model completes the picture by putting Vertical in the upper half. Nothing in
the document says that.

The geometry was never lost -- it is in the store's own layout artifact, word
by word with bounding boxes, from the same extraction the corpus came from.
This module reads it, joins it to a fetched unit by page and text, and reports
the cells. It writes nothing, re-extracts nothing, and asks nothing of the PDF.

A cell's printed range is local to that cell. Turning two local ranges into one
global layout requires a normative statement about how the cells compose, and
where the document does not make one, neither does this.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from state_paths import standards_root
from standard_stated_range import (
    RangeContext, RangeStatus, display_label, parse_stated_ranges,
)

DIAGRAM_CELL = "DIAGRAM_CELL"
CELL_LOCAL_FIELD_RANGE = "CELL_LOCAL_FIELD_RANGE"

LAYOUT_ARTIFACT = "layout.json"

# A cell's name is words; a ruler's is digits. A line of nothing but numbers is
# the bit ruler above the row, never a field.

_RULER_LINE = re.compile(r"^[\d\s]+$")
_SPACES = re.compile(r"\s+")


def layout_path(standard_id, revision):
    return standards_root() / standard_id / revision / LAYOUT_ARTIFACT


@lru_cache(maxsize=4)
def _pages(standard_id, revision):
    """Every page's blocks, keyed by page number. Read once, never written."""
    path = layout_path(standard_id, revision)

    if not path.is_file():
        return {}

    document = json.loads(path.read_text("utf-8"))
    found = {}

    for index, page in enumerate(document.get("pages") or ()):
        number = page.get("page", index + 1)
        found[number] = page

    return found


@lru_cache(maxsize=4)
def layout_fingerprint(standard_id, revision):
    path = layout_path(standard_id, revision)

    if not path.is_file():
        return ""

    document = json.loads(path.read_text("utf-8"))

    return document.get("layout_fingerprint", "")


@dataclass(frozen=True)
class Cell:
    """One diagram cell, with the box it was printed in."""

    label: str
    text: str
    msb: int
    lsb: int
    page: int
    block: int
    line: int
    bbox: tuple
    source_id: str = ""
    range_text: str = ""

    @property
    def key(self):
        return re.sub(r"[^a-z0-9]+", "", self.label.lower())

    @property
    def column(self):
        """The horizontal span the cell occupies, which is its identity."""
        return (self.bbox[0], self.bbox[2])

    @property
    def provenance(self):
        return {"source_id": self.source_id, "page": self.page,
                "block": self.block, "line": self.line,
                "bbox": list(self.bbox), "text": self.text,
                "range_text": self.range_text}


def _normalize(text):
    return _SPACES.sub(" ", (text or "").strip())


def _line_text(line):
    return _normalize(" ".join(word.get("text", "")
                               for word in line.get("words") or ()))


def _line_box(line, words):
    if line.get("bbox"):
        return tuple(line["bbox"])

    boxes = [word["bbox"] for word in words if word.get("bbox")]

    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)

    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def cells_for_unit(unit, *, standard_id, revision):
    """The diagram cells whose printed text this flattened unit is made of.

    A layout line qualifies when its words appear in the unit's own text -- so
    the join is the document's own words, not a guess about which figure the
    unit came from -- and when the repository's stated-range parser accepts a
    range in it. A line of ruler digits carries no name and is skipped.
    """
    if not isinstance(unit, dict):
        return ()

    page_number = unit.get("page")
    flattened = _normalize(unit.get("text"))

    if page_number is None or not flattened:
        return ()

    page = _pages(standard_id, revision).get(page_number)

    if page is None:
        return ()

    context = RangeContext(in_bitfield_region=True)
    found = []

    for block in page.get("blocks") or ():
        for line in block.get("lines") or ():
            words = line.get("words") or ()
            text = _line_text(line)

            if not text or _RULER_LINE.match(text):
                continue

            if text not in flattened:
                continue

            accepted = [item for item in parse_stated_ranges(text,
                                                             context=context)
                        if item.status is RangeStatus.ACCEPTED]

            if len(accepted) != 1:
                # No range, or more than one, means this line does not name a
                # single positioned cell and nothing is read from it.
                continue

            parsed = accepted[0]
            label = _normalize(display_label(text, parsed))

            if not label:
                continue

            found.append(Cell(
                label=label, text=text, msb=parsed.high_bit,
                lsb=parsed.low_bit, page=page_number,
                block=block.get("block", -1), line=line.get("line", -1),
                bbox=_line_box(line, words),
                source_id=unit.get("source_id", ""),
                range_text=parsed.range_text))

    return tuple(sorted(found, key=lambda cell: (cell.bbox[1], cell.bbox[0])))


def independent(cells):
    """Cells whose printed columns do not overlap are separate fields."""
    pairs = []

    for index, first in enumerate(cells):
        for second in cells[index + 1:]:
            low, high = first.column, second.column

            if low[1] <= high[0] or high[1] <= low[0]:
                pairs.append((first, second))

    return pairs
