"""Which extracted units belong together, worked out at read time.

A table arrives from the extractor in pieces. The caption is one unit -- 85 of
this store's 98 table units are nothing BUT a caption -- and each row is
another, carrying no label, no content type and no link back. Asked which bit
of a field is reserved, a session read forty-one units over fourteen rounds
and then said the evidence did not settle it. The answer was in a unit it had
retrieved; nothing connected that unit to the table it came from.

So the relationship is derived, never written back: the licensed corpus is not
rewritten, and a caller that ignores this module sees exactly what it saw
before.

Conservative on purpose. Adjacency alone attaches prose to whatever table it
happens to follow, and a table's neighbourhood is mostly prose. A unit joins a
table only if it LOOKS like a row of one -- a leading key and a short name --
and the run stops at the first unit that does not, so a paragraph after the
last row ends the table rather than joining it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import provision_identity

#: A caption: a table unit whose text is a caption and little else.
_CAPTION = re.compile(r"^\s*table\b", re.I)

#: How far a table's rows may sit from its caption. A table continues onto the
#: next page; it does not continue into the next section.
_MAX_PAGE_SPAN = 1

#: A table is caption, then a column header, then rows. The header is not
#: row-shaped -- "Bit# Designation Name Function" has no leading key -- and
#: treating it as the end of the table lost every row of the one table this
#: was written for. So a bounded number of header-shaped units may precede
#: the first row; once rows begin, the first non-row unit ends the table.
_HEADER_TOLERANCE = 2

#: What may be tolerated before the first row: short, and not a sentence.
#: Prose is neither, which is what keeps a paragraph from being swallowed.
_HEADER_MAX_CHARS = 80


def _header_like(unit):
    text = _clean(unit.get("text"))

    return bool(text) and len(text) <= _HEADER_MAX_CHARS and "." not in text


def _clean(text):
    return " ".join((text or "").split())


def is_row_like(unit):
    """Does this unit look like a row of a table, rather than prose?

    The test is the shape the extractor leaves: a key, then a short name. A
    sentence fails it, which is the point -- absorbing the paragraph after a
    table is how a structural link turns into a source of wrong evidence.
    """
    if not isinstance(unit, dict):
        return False

    if (unit.get("content_type") or "").upper() not in ("UNKNOWN", "TABLE", ""):
        return False

    text = _clean(unit.get("text"))

    return bool(provision_identity._TABLE_ROW.match(text)) and not _CAPTION.match(text)


def is_caption(unit):
    if not isinstance(unit, dict):
        return False

    return ((unit.get("content_type") or "").upper() == "TABLE"
            and bool(_CAPTION.match(_clean(unit.get("text")))))


@dataclass
class Table:
    """One caption and the rows derived for it."""

    caption: dict
    rows: list = field(default_factory=list)

    @property
    def section(self):
        return str(self.caption.get("section") or "")

    @property
    def source_id(self):
        return str(self.caption.get("source_id") or "")

    @property
    def title(self):
        return _clean(self.caption.get("text"))

    def keys(self):
        """The ProvisionKey of every row, in the order they were read."""
        found = []

        identity = provision_identity.table_identity(self.caption)

        for row in self.rows:
            for record in provision_identity.records_from_unit(
                    row, table=identity):
                if isinstance(record.key, provision_identity.TableRowKey):
                    found.append(record.key)

        return found

    def row_for(self, key):
        """The row unit whose leading key is `key`, or None."""
        for row in self.rows:
            match = provision_identity._TABLE_ROW.match(_clean(row.get("text")))

            if match and int(match.group("key")) == key:
                return row

        return None


def tables_in(units):
    """Every table recoverable from a sequence of units.

    `units` is whatever the caller has -- one payload's worth, or a section
    read from the store. Order is taken from `unit_position` when present,
    because the extractor's order is the document's.
    """
    ordered = sorted((unit for unit in units if isinstance(unit, dict)),
                     key=lambda unit: (unit.get("page") or 0,
                                       unit.get("unit_position") or 0))
    found, current, header_budget = [], None, 0

    for unit in ordered:
        if is_caption(unit):
            current = Table(caption=unit)
            header_budget = _HEADER_TOLERANCE
            found.append(current)
            continue

        if current is None:
            continue

        same_section = (str(unit.get("section") or "") == current.section)
        near = (abs((unit.get("page") or 0)
                    - (current.caption.get("page") or 0)) <= _MAX_PAGE_SPAN)

        if same_section and near and is_row_like(unit):
            current.rows.append(unit)
            header_budget = 0            # rows have begun; no more tolerance
        elif (not current.rows and header_budget and same_section and near
              and _header_like(unit)):
            header_budget -= 1           # a column header, not the end
        else:
            # The run has ended. A later row-shaped unit belongs to whatever
            # comes next, not to this caption.
            current = None

    return found


def link(units):
    """{row source_id: caption unit} and {caption source_id: [row units]}.

    Both directions: a caption hit should be able to reach its rows, and a row
    hit should keep the table it came from.
    """
    to_caption, to_rows = {}, {}

    for table in tables_in(units):
        to_rows[table.source_id] = list(table.rows)

        for row in table.rows:
            to_caption[str(row.get("source_id") or "")] = table.caption

    return to_caption, to_rows
