"""Operator-only deterministic PDF ingestion into a canonical standard corpus.

Segmentation runs in three passes. A document pass finds the repeated page
furniture and the front matter, because neither can be recognised from a single
line. A page pass groups lines into blocks and measures their column geometry. A
document-order pass then decides which blocks are genuine clause headings, using
the clause number's syntax, the hierarchy it would continue, and the layout the
block sits in -- never the bare fact that a line begins with a digit.
"""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from standard_schema import (
    HumanValidationStatus, StandardContentType, StandardDocumentUnit,
    StandardIngestionManifest, StandardLayoutKind, StandardModality,
    STANDARD_UNIT_SCHEMA_VERSION, make_source_id, source_content_sha256,
)
from standard_store import StandardStore, corpus_fingerprint


EXTRACTOR_VERSION = "poppler-structure-v2"

# A line that opens with a number and a title. Matching this makes a line a
# heading *candidate* only; _classify_heading decides whether it is one.

_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\s+(\S.*)$")
_CROSS_REFERENCE = re.compile(
    r"\b(?:see|refer(?:red)?\s+to)\s+(?:section\s+)?(\d+(?:\.\d+)+)\b", re.I,
)
_TABLE = re.compile(r"^\s*(?:table|tab\.)\s+[A-Za-z0-9.-]+\b", re.I)
_FIGURE = re.compile(r"^\s*(?:figure|fig\.)\s+[A-Za-z0-9.-]+\b", re.I)

# A caption names the table; a sentence merely refers to it. The separator, or a
# short titular line, tells them apart.

_CAPTION_SEPARATOR = re.compile(r"^\s*[:\u2013\u2014]")
_MAX_CAPTION_CHARS = 120

# A standard that labels its own normative statements is better evidence about
# them than any keyword count; the keywords are the fallback, not the rule.

_NORMATIVE_LABEL = re.compile(
    r"^\s*(rule|requirement|recommendation|suggestion|permission|observation"
    r"|definition)\s+[A-Za-z0-9][A-Za-z0-9.\-]*\s*:", re.I)

# Quoting a keyword mentions it; it does not impose it.

_QUOTED_SPAN = re.compile(r"[\"\u201c][^\"\u201c\u201d]{0,120}[\"\u201d]")
_DEFINED_AS = re.compile(r"\bis defined as\b|\bshall mean\b", re.I)

# A definition names a term and then says what it means. Ordinary prose uses the
# noun ("a means of", "by means of") or takes a clause ("means that ..."), and
# opens with a pronoun rather than the term being defined.

_DEFINITION_TERM = re.compile(
    r"(?:^|(?<=[.;]\s))\s*"
    r"(?!(?:this|that|these|those|it|its|which|there|they|we|you|he|she)\b)"
    r"[^.;:]{1,60}?\s+means\s+(?!that\b|of\b|to\b|for\b|by\b)\S", re.I)

# The same labels, unanchored, so a unit that merged several can be spotted.

_LABEL_MARKER = re.compile(
    r"\b(?:rule|requirement|recommendation|suggestion|permission|observation"
    r"|definition)\s+[A-Za-z0-9][A-Za-z0-9.\-]*\s*:", re.I)
COARSE_UNIT_WARNING = "unit merges more than one labelled paragraph"
_ANNEX = re.compile(r"^\s*(?:annex|appendix)\s+[A-Z0-9]+\b", re.I)

# A clause component is at most two digits and never zero-padded. This is what
# separates a real clause number from a binary literal, a zero-padded code, a
# bare magnitude or a measurement; it encodes no document's numbering.

_CLAUSE_COMPONENT = re.compile(r"0|[1-9][0-9]?")
_MAX_CLAUSE_DEPTH = 4
_MAX_HEADING_CHARS = 120
_MAX_SIBLING_STEP = 3

# A gap this wide separates a heading from the block above it. The first line of
# a page is treated as fully separated.

_ISOLATED_GAP = 2

# A heading sets its title beside its number; a table sets a cell under a column
# header, far to the right. The dividing distance is a property of how wide the
# document's lines are, so it is measured rather than assumed.

_HEADING_GAP_WIDTH_SHARE = 0.25
_HEADING_GAP_FLOOR = 12
_LINE_WIDTH_QUANTILE = 0.95

# Cells are separated by a run of at least three spaces, which is what
# ``pdftotext -layout`` emits between columns.

_COLUMN_GAP = re.compile(r" {3,}")
_DOT_LEADER = re.compile(r"\.{4,}")
_PAGE_TARGET = re.compile(r"(?:\.{4,}|\s{2,})\s*(\d{1,4})$")

# A contents entry names a clause and the page it opens on. That page is the
# document's own statement about its structure, so it outranks any heuristic.

_CONTENTS_TARGET = re.compile(
    r"^(\d+(?:\.\d+)*)\s+.*?(?:\.{2,}|\s{2,})\s*(\d{1,4})\s*$")
_FRONT_MATTER_TITLE = re.compile(
    r"^(?:table of contents|contents|list of tables|list of figures"
    r"|list of abbreviations|list of acronyms|index)\s*$", re.I)
_DIGITS = re.compile(r"\d+")

# Furniture must repeat across at least this share of the document's pages.

_FURNITURE_PAGE_SHARE = 0.5
_FURNITURE_MIN_PAGES = 3
_FURNITURE_MAX_CHARS = 200

# Furniture appears about once per page. A line that repeats far more often
# than the pages carrying it is repeated content, not a running head or foot.

_FURNITURE_MAX_PER_PAGE = 1.5

# Front matter must be mostly page-target entries, not merely contain a few.

_FRONT_MATTER_MIN_ENTRIES = 5
_FRONT_MATTER_CONTINUATION_ENTRIES = 3
_FRONT_MATTER_SHARE = 0.5


class StandardIngestionError(RuntimeError):
    pass


def _run(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StandardIngestionError(f"PDF extraction unavailable: {exc}") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        raise StandardIngestionError(
            "malformed or unsupported PDF: " + (detail[-1][:240] if detail else "extractor failed")
        )

    return completed.stdout


def extract_pdf_pages(path: str | Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    source = Path(path).expanduser()

    if not source.is_file() or source.is_symlink():
        raise StandardIngestionError("PDF path must name a regular local file")

    info = _run(["pdfinfo", str(source)])
    match = re.search(r"^Pages:\s+(\d+)\s*$", info, re.M)

    if not match or int(match.group(1)) < 1:
        raise StandardIngestionError("PDF page count is unavailable")

    page_count = int(match.group(1))
    text = _run(["pdftotext", "-layout", "-enc", "UTF-8", str(source), "-"])
    pages = text.split("\f")

    if pages and not pages[-1].strip():
        pages.pop()

    warnings = []

    if len(pages) < page_count:
        pages.extend("" for _ in range(page_count - len(pages)))
        warnings.append("extractor returned fewer page boundaries than PDF metadata")
    elif len(pages) > page_count:
        pages = pages[:page_count]
        warnings.append("extractor returned extra page boundaries; extras ignored")

    for number, page in enumerate(pages, 1):
        if not page.strip():
            warnings.append(f"page {number} yielded no extractable text")

    if not any(page.strip() for page in pages):
        raise StandardIngestionError("PDF contains no extractable text; OCR is not enabled")

    return tuple(pages), tuple(warnings)


# --------------------------------------------------------------------------
# clause numbers
# --------------------------------------------------------------------------

def _clause_parts(number: str) -> tuple[int, ...] | None:
    """Return the clause components, or None when the number cannot be a clause."""
    parts = number.split(".")

    if not 1 <= len(parts) <= _MAX_CLAUSE_DEPTH:
        return None

    if not all(_CLAUSE_COMPONENT.fullmatch(part) for part in parts):
        return None

    return tuple(int(part) for part in parts)


def _continues(previous: tuple[int, ...] | None, candidate: tuple[int, ...]) -> bool:
    """Whether ``candidate`` plausibly follows ``previous`` in one clause tree."""

    if previous is None:
        return False

    if (len(candidate) == len(previous) + 1
            and candidate[:-1] == previous and candidate[-1] <= 1):
        return True

    for depth in range(1, len(previous) + 1):
        if (len(candidate) == depth
                and candidate[:depth - 1] == previous[:depth - 1]
                and 0 < candidate[depth - 1] - previous[depth - 1] <= _MAX_SIBLING_STEP):
            return True

    return False


# --------------------------------------------------------------------------
# lines and blocks
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _Line:
    page: int
    text: str
    blank_before: int     # blank lines above, i.e. how far the gap opened
    ordinal: int          # position among the page's non-blank lines
    total: int            # number of non-blank lines on the page

    @property
    def preceded_by_blank(self) -> bool:
        return self.blank_before > 0


def _page_lines(pages: tuple[str, ...]) -> tuple[tuple[_Line, ...], ...]:
    result = []

    for number, page in enumerate(pages, 1):
        raw = page.splitlines()
        stripped = [line.strip() for line in raw]
        content = [index for index, line in enumerate(stripped) if line]
        lines = []

        for ordinal, index in enumerate(content):
            blank = 0
            probe = index - 1

            while probe >= 0 and not stripped[probe]:
                blank += 1
                probe -= 1

            lines.append(_Line(
                page=number, text=stripped[index],
                blank_before=(blank if index else _ISOLATED_GAP),
                ordinal=ordinal, total=len(content)))

        result.append(tuple(lines))

    return tuple(result)


def _heading_gap_limit(page_lines: tuple[tuple[_Line, ...], ...]) -> int:
    """How far right a clause title may sit before it is a column, not a title."""
    lengths = sorted(len(line.text) for lines in page_lines for line in lines)

    if not lengths:
        return _HEADING_GAP_FLOOR

    width = lengths[min(int(len(lengths) * _LINE_WIDTH_QUANTILE), len(lengths) - 1)]

    return max(_HEADING_GAP_FLOOR, int(_HEADING_GAP_WIDTH_SHARE * width))


def _signature(text: str) -> str:
    """Collapse whitespace and page numbers so a running footer matches itself."""
    return _DIGITS.sub("#", " ".join(text.split()))


def _furniture_signatures(page_lines: tuple[tuple[_Line, ...], ...]) -> frozenset[str]:
    """Short lines that repeat in the margins of most pages are page furniture."""
    page_count = len(page_lines)

    if page_count < _FURNITURE_MIN_PAGES:
        return frozenset()

    threshold = max(_FURNITURE_MIN_PAGES,
                    math.ceil(_FURNITURE_PAGE_SHARE * page_count))
    pages_seen: dict[str, set[int]] = defaultdict(set)
    occurrences: Counter[str] = Counter()

    for lines in page_lines:
        for line in lines:
            if len(line.text) > _FURNITURE_MAX_CHARS:
                continue

            signature = _signature(line.text)
            pages_seen[signature].add(line.page)
            occurrences[signature] += 1

    return frozenset(
        signature for signature, pages in pages_seen.items()
        if len(pages) >= threshold
        and occurrences[signature] <= _FURNITURE_MAX_PER_PAGE * len(pages)
    )


def _is_page_target(text: str) -> bool:
    """A contents entry: a title trailing off to a page number."""
    return bool(_PAGE_TARGET.search(text))


def _front_matter_pages(page_lines: tuple[tuple[_Line, ...], ...],
                        furniture: frozenset[str]) -> frozenset[int]:
    """Pages whose body is a contents/list-of table, not normative prose."""
    entries: dict[int, tuple[int, int, bool]] = {}

    for lines in page_lines:
        body = [line for line in lines if _signature(line.text) not in furniture]

        if not body:
            continue

        targets = sum(1 for line in body if _is_page_target(line.text))
        titled = any(_FRONT_MATTER_TITLE.match(line.text) for line in body[:3])
        entries[body[0].page] = (targets, len(body), titled)

    front: set[int] = set()

    for page, (targets, total, titled) in entries.items():
        dense = targets >= _FRONT_MATTER_MIN_ENTRIES and targets >= _FRONT_MATTER_SHARE * total

        if titled or dense:
            front.add(page)

    # A contents table usually runs on; keep following pages while they stay dense.

    for page in sorted(front):
        follower = page + 1

        while follower in entries:
            targets, total, _ = entries[follower]

            if (targets < _FRONT_MATTER_CONTINUATION_ENTRIES
                    or targets < _FRONT_MATTER_SHARE * total):
                break

            front.add(follower)
            follower += 1

    return frozenset(front)


@dataclass
class _Block:
    page: int
    lines: list[_Line]
    blank_before: int
    is_furniture: bool = False
    after_annex: bool = False
    text: str = ""
    cells: int = 1
    layout_kind: StandardLayoutKind = StandardLayoutKind.PROSE
    section: str | None = None
    warnings: list[str] = field(default_factory=list)

    def finish(self) -> "_Block":
        self.text = " ".join(line.text for line in self.lines)
        self.cells = max(len(_COLUMN_GAP.split(line.text)) for line in self.lines)

        return self

    def slice(self, start: int, stop: int | None = None) -> "_Block":
        """A new block over part of these lines, keeping the page and furniture."""
        chosen = self.lines[start:stop]

        return _Block(page=self.page, lines=list(chosen),
                      blank_before=(self.blank_before if start == 0 else 0),
                      is_furniture=self.is_furniture,
                      after_annex=self.after_annex,
                      layout_kind=self.layout_kind).finish()


def _blocks(lines: tuple[_Line, ...], furniture: frozenset[str]) -> list[_Block]:
    """Group a page's lines into blocks, splitting on blanks and on candidates."""
    result: list[_Block] = []
    current: _Block | None = None

    for line in lines:
        is_furniture = _signature(line.text) in furniture
        opens = (line.preceded_by_blank or is_furniture
                 or (current is not None and current.is_furniture)
                 or _NUMBERED.match(line.text) or _TABLE.match(line.text)
                 or _FIGURE.match(line.text) or _ANNEX.match(line.text))

        if current is None or opens:
            if current is not None:
                result.append(current.finish())

            current = _Block(page=line.page, lines=[line],
                             blank_before=line.blank_before,
                             is_furniture=is_furniture)
        else:
            current.lines.append(line)

    if current is not None:
        result.append(current.finish())

    return result


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def _heading_layout_evidence(block: _Block, neighbour: _Block | None) -> bool:
    """Is this block laid out like an isolated heading rather than a body line?

    A clause number that does not continue the hierarchy needs the page to argue
    for it: standing alone, titled, and not one more row of a table already
    running down the page.
    """
    title = _NUMBERED.match(block.text).group(2)

    return (block.blank_before > 0
            and len(block.lines) == 1
            and len(block.text) <= _MAX_HEADING_CHARS
            and block.cells <= 2
            and title[:1].isupper()
            and not block.text.rstrip().endswith((".", ";", ",", ":"))
            and (neighbour is None
                 or neighbour.layout_kind is not StandardLayoutKind.COLUMNAR
                 or block.blank_before >= _ISOLATED_GAP))


def _annex_heading(block: _Block) -> bool:
    """An annex opens on its own titled line, not mid-sentence in a paragraph."""
    return (block.blank_before > 0 and block.cells <= 2
            and len(block.text) <= _MAX_HEADING_CHARS
            and not _DOT_LEADER.search(block.text))


def _classify_heading(block: _Block, previous: tuple[int, ...] | None,
                      neighbour: _Block | None, gap_limit: int,
                      ) -> tuple[str, tuple[int, ...]] | None:
    """Accept a block as a clause heading, or return None to leave it as body."""
    match = _NUMBERED.match(block.text)

    if match is None:
        return None

    number = match.group(1)
    parts = _clause_parts(number)

    if parts is None:
        return None

    if block.cells >= 3 or _DOT_LEADER.search(block.text):
        return None

    if _title_gap(block.text) > gap_limit:
        # The title sits under a column header rather than beside its number.

        return None

    if _continues(previous, parts):
        # Two cells is ambiguous between a heading and a row; a number that
        # continues the clause tree settles it.

        return number, parts

    if block.layout_kind is StandardLayoutKind.COLUMNAR:
        return None

    if _heading_layout_evidence(block, neighbour):
        block.warnings.append("clause number does not continue the previous heading")
        return number, parts

    return None


def _title_gap(text: str) -> int:
    """Spaces between a clause number and its title, i.e. how it was set."""
    match = _NUMBERED.match(text.strip())

    return 10 ** 6 if match is None else match.start(2) - len(match.group(1))


def _contents_target(text: str) -> tuple[str, int] | None:
    match = _CONTENTS_TARGET.match(" ".join(text.split(" ")))

    if match is None or _clause_parts(match.group(1)) is None:
        return None

    return match.group(1), int(match.group(2))


def _printed_page_offset(headings: list[_Block],
                         targets: Mapping[str, int]) -> int | None:
    """How far the printed page numbers sit from the extracted page numbers."""
    claims: Counter[str] = Counter(block.section for block in headings)
    offsets = Counter(block.page - targets[block.section] for block in headings
                      if claims[block.section] == 1 and block.section in targets)

    return offsets.most_common(1)[0][0] if offsets else None


def _drop_reopened_clauses(headings: list[_Block],
                           targets: Mapping[str, int]) -> None:
    """A clause number opens one clause. Later re-openings are table columns.

    Where a number was claimed twice, a body clause always outranks one read
    after an annex began, because an annex renumbers in its own namespace. Then
    comes the page the contents table gives for that clause, which is the
    document speaking for itself. Only when it is silent do the heuristics
    decide: the reading that continued the hierarchy, then the one set like a
    heading rather than like a column -- a title pushed far to the right is a
    cell under a column header, not a title beside its number -- and finally the
    earlier reading.
    """
    RESYNC = "clause number does not continue the previous heading"
    offset = _printed_page_offset(headings, targets)

    def contradicts_contents(block: _Block) -> int:
        target = targets.get(block.section or "")

        if target is None or offset is None:
            return 0

        return 0 if block.page == target + offset else 1

    def rank(block: _Block, order: int) -> tuple[int, int, int, int, int]:
        return (1 if block.after_annex else 0,
                contradicts_contents(block),
                1 if RESYNC in block.warnings else 0,
                _title_gap(block.text), order)

    claimed: dict[str, tuple[_Block, tuple[int, int, int]]] = {}

    for order, block in enumerate(headings):
        score = rank(block, order)
        held = claimed.get(block.section)

        if held is None:
            claimed[block.section] = (block, score)
            continue

        loser = block if score >= held[1] else held[0]

        if loser is held[0]:
            claimed[block.section] = (block, score)

        loser.layout_kind = StandardLayoutKind.PROSE
        loser.section = None
        loser.warnings.append("clause number was already opened earlier")


def _mark_columns(blocks: list[_Block]) -> None:
    """Flag column-shaped blocks from layout alone, before any clause is read.

    Three or more cells is a table row on its own evidence. Two cells is only a
    row while a table caption is open and nothing has broken the run.
    """
    table_run = False

    for block in blocks:
        if block.is_furniture:
            continue

        if block.blank_before >= _ISOLATED_GAP and block.cells < 3:
            # A gap this wide ends the table; rows are set tight against
            # each other, so whatever follows starts something new.

            table_run = False

        if _is_caption(block.text, _TABLE):
            table_run = True
            continue

        if block.cells >= 3 or (table_run and block.cells == 2):
            block.layout_kind = StandardLayoutKind.COLUMNAR
            table_run = True
        elif block.cells == 1:
            table_run = False


def _is_caption(text: str, reference: re.Pattern[str]) -> bool:
    """Whether this line captions a table or figure rather than discussing one."""
    match = reference.match(text)

    if match is None:
        return False

    if _CAPTION_SEPARATOR.match(text[match.end():]):
        return True

    stripped = text.strip()

    return len(stripped) <= _MAX_CAPTION_CHARS and not stripped.endswith(".")


_LABEL_CONTENT_TYPES = {
    "rule": StandardContentType.REQUIREMENT,
    "requirement": StandardContentType.REQUIREMENT,
    "recommendation": StandardContentType.RECOMMENDATION,
    "suggestion": StandardContentType.RECOMMENDATION,
    "definition": StandardContentType.DEFINITION,
    "permission": StandardContentType.INFORMATIVE,
    "observation": StandardContentType.INFORMATIVE,
}


def _modality(text: str) -> StandardModality:
    text = _QUOTED_SPAN.sub(" ", text)

    for word, value in (("shall", StandardModality.SHALL),
                        ("should", StandardModality.SHOULD),
                        ("must", StandardModality.MUST),
                        ("may", StandardModality.MAY)):
        if re.search(rf"\b{word}\b", text, re.I):
            return value

    return StandardModality.NONE


def _content_type(text: str, modality: StandardModality) -> tuple[StandardContentType, bool]:
    if _is_caption(text, _TABLE):
        return StandardContentType.TABLE, True

    if _is_caption(text, _FIGURE):
        return StandardContentType.FIGURE, True

    label = _NORMATIVE_LABEL.match(text)

    if label is not None:
        # The document says what this paragraph is; believe it over the keywords.

        return _LABEL_CONTENT_TYPES[label.group(1).casefold()], False

    if re.search(r"\bformula\b|^\s*equation\s+", text, re.I):
        return StandardContentType.FORMULA, True

    if modality in {StandardModality.SHALL, StandardModality.MUST}:
        return StandardContentType.REQUIREMENT, False

    if modality == StandardModality.SHOULD:
        return StandardContentType.RECOMMENDATION, False

    if _DEFINED_AS.search(text) or _DEFINITION_TERM.search(text):
        return StandardContentType.DEFINITION, False

    if re.search(r"\bexample\b", text, re.I):
        return StandardContentType.EXAMPLE, False

    if re.search(r"\binformative\b", text, re.I):
        return StandardContentType.INFORMATIVE, False

    return StandardContentType.UNKNOWN, False


def canonical_units(
    pages: tuple[str, ...], *, standard_id: str, revision: str,
    pdf_sha256: str, extractor_version: str = EXTRACTOR_VERSION,
) -> tuple[StandardDocumentUnit, ...]:
    page_lines = _page_lines(pages)
    gap_limit = _heading_gap_limit(page_lines)
    furniture = _furniture_signatures(page_lines)
    front_matter = _front_matter_pages(page_lines, furniture)

    prepared: list[tuple[_Block, bool, bool, bool]] = []
    headings: list[_Block] = []
    contents_targets: dict[str, int] = {}
    previous: tuple[int, ...] | None = None
    after_annex = False

    for lines in page_lines:
        blocks = _blocks(lines, furniture)
        _mark_columns(blocks)
        neighbour: _Block | None = None
        current_section_reset = False
        pending = list(blocks)

        while pending:
            block = pending.pop(0)
            block.after_annex = after_annex
            is_front = block.page in front_matter
            is_toc = is_front and _is_page_target(block.text)

            if block.is_furniture:
                block.layout_kind = StandardLayoutKind.FURNITURE
            elif is_toc:
                block.layout_kind = StandardLayoutKind.TOC_ENTRY
            elif is_front:
                pass
            elif _ANNEX.match(block.text) and _annex_heading(
                    block.slice(0, 1) if len(block.lines) > 1 else block):

                # An annex restarts numbering in a namespace of its own. Forget
                # the clause we were in rather than invent a heading for it, and
                # stop filing what follows under the last body clause.

                previous = None
                current_section_reset = True
                after_annex = True
                block.warnings.append("annex or appendix heading resets clause numbering")
            else:
                # A heading and the paragraph under it often arrive with no
                # blank line between them. Judge the opening line on its own,
                # and give the heading its own canonical unit.

                probe = block.slice(0, 1) if len(block.lines) > 1 else block
                heading = _classify_heading(probe, previous, neighbour, gap_limit)

                if heading is not None and probe is not block:
                    pending.insert(0, block.slice(1))
                    block = probe

                if heading is not None:
                    block.section, previous = heading[0], heading[1]
                    block.layout_kind = StandardLayoutKind.HEADING
                    headings.append(block)

            if is_toc:
                target = _contents_target(block.text)

                if target is not None:
                    contents_targets.setdefault(target[0], target[1])

            prepared.append((block, is_front, is_toc, current_section_reset))
            current_section_reset = False

            if not block.is_furniture:
                neighbour = block

    _drop_reopened_clauses(headings, contents_targets)

    units: list[StandardDocumentUnit] = []
    current_section: str | None = None
    parent_source_id: str | None = None
    heading_levels: dict[int, str] = {}
    position = 0
    previous_body: _Block | None = None

    for block, is_front, is_toc, resets in prepared:
        position += 1

        if resets:
            current_section = None
            heading_levels = {}
            parent_source_id = None

        is_heading = block.layout_kind is StandardLayoutKind.HEADING
        excluded = block.is_furniture or is_toc

        if is_heading:
            current_section = block.section
            level = current_section.count(".") + 1
            heading_levels[level] = block.text
            heading_levels = {key: value for key, value in heading_levels.items()
                              if key <= level}

        section = None if (excluded or is_front) else current_section
        heading_path = () if (excluded or is_front) else tuple(
            heading_levels[key] for key in sorted(heading_levels))
        modality = _modality(block.text)

        # Content type says what a unit is; modality records the words it uses.
        # Front matter may still say "shall" -- that is evidence worth keeping.

        if block.is_furniture:
            content_type, structured = StandardContentType.PAGE_FURNITURE, False
        elif is_front:
            content_type, structured = StandardContentType.FRONT_MATTER, False
        else:
            content_type, structured = _content_type(block.text, modality)

        if block.layout_kind is StandardLayoutKind.COLUMNAR:
            structured = True

        continuation = bool(
            block.layout_kind is StandardLayoutKind.COLUMNAR
            and previous_body is not None
            and previous_body.page == block.page - 1
            and previous_body.layout_kind is StandardLayoutKind.COLUMNAR)
        source_id = make_source_id(
            standard_id=standard_id, revision=revision, page=block.page,
            section=section, unit_position=position, text=block.text)
        units.append(StandardDocumentUnit(
            source_id=source_id, standard_id=standard_id, revision=revision,
            section=section, page=block.page, heading_path=heading_path,
            content_type=content_type, modality=modality, text=block.text,
            parent_source_id=(None if is_heading or excluded else parent_source_id),
            cross_references=(() if excluded else
                              tuple(dict.fromkeys(_CROSS_REFERENCE.findall(block.text)))),
            source_pdf_sha256=pdf_sha256,
            source_content_sha256=source_content_sha256(block.text),
            extractor_version=extractor_version,
            warnings=tuple(block.warnings) + (
                ("complex visual structure requires human review",) if structured else ())
            + ((COARSE_UNIT_WARNING,)
               if len(_LABEL_MARKER.findall(block.text)) > 1 else ()),
            needs_review=structured, needs_structured_review=structured,
            schema_version=STANDARD_UNIT_SCHEMA_VERSION,
            layout_kind=block.layout_kind,
            retrievable=not excluded,
            is_front_matter=is_front, is_toc_entry=is_toc,
            possible_table_continuation=continuation,
            unit_position=position))

        if is_heading:
            parent_source_id = source_id

        if not block.is_furniture:
            previous_body = block

    return tuple(units)


def extraction_diagnostics(units: tuple[StandardDocumentUnit, ...]) -> dict[str, object]:
    """Aggregate, metadata-only evidence about how a corpus was segmented.

    Order-dependent measures need reading order. A schema-2 unit records its
    position; for an older corpus the store can only offer page-then-id order,
    so those measures are reported as approximate rather than as fact.
    """
    from standard_crossrefs import _section_anchor

    ordered_known = bool(units) and all(unit.schema_version >= 2 for unit in units)
    ordered = sorted(units, key=(lambda unit: unit.unit_position) if ordered_known
                     else (lambda unit: (unit.page, unit.source_id)))
    headings = [unit for unit in units
                if unit.layout_kind is StandardLayoutKind.HEADING]
    sections = [unit.section for unit in units if unit.section]
    distinct = sorted(set(sections))
    invalid = {value for value in distinct if _clause_parts(value) is None}
    transitions: list[tuple[int, ...]] = []

    for unit in ordered:
        parts = _clause_parts(unit.section or "")

        if parts is not None and (not transitions or transitions[-1] != parts):
            transitions.append(parts)

    backwards = sum(1 for before, after in zip(transitions, transitions[1:])
                    if after < before)
    anchors: Counter[str] = Counter(
        unit.section for unit in units if _section_anchor(unit))
    versions = sorted({unit.extractor_version for unit in units})

    return {
        "extractor_version": versions[0] if len(versions) == 1 else ",".join(versions),
        "corpus_schema_version": max((unit.schema_version for unit in units),
                                     default=0),
        "reading_order_recoverable": ordered_known,
        "unit_count": len(units),
        "page_count": max((unit.page for unit in units), default=0),
        "heading_units": len(headings) if ordered_known else None,
        "distinct_sections": len(distinct),
        "invalid_section_ids": len(invalid),
        "units_on_invalid_sections": sum(1 for unit in units
                                         if unit.section in invalid),
        "units_without_section": sum(1 for unit in units if not unit.section),
        "section_transitions": len(transitions),
        "backwards_transitions": backwards,
        "duplicate_section_anchors": sum(1 for count in anchors.values() if count > 1),
        "section_anchors": len(anchors),
        "front_matter_units": sum(1 for unit in units if unit.is_front_matter),
        "toc_entry_units": sum(1 for unit in units if unit.is_toc_entry),
        "page_furniture_units": sum(
            1 for unit in units
            if unit.content_type is StandardContentType.PAGE_FURNITURE),
        "retrievable_units": sum(1 for unit in units if unit.retrievable),
        "columnar_units": sum(1 for unit in units
                              if unit.layout_kind is StandardLayoutKind.COLUMNAR),
        "structured_review_units": sum(1 for unit in units
                                       if unit.needs_structured_review),
        "table_continuation_units": sum(1 for unit in units
                                        if unit.possible_table_continuation),
        "table_caption_units": sum(1 for unit in units
                                   if unit.content_type is StandardContentType.TABLE),
        "figure_caption_units": sum(1 for unit in units
                                    if unit.content_type is StandardContentType.FIGURE),
        "modality_counts": dict(sorted(Counter(
            unit.modality.value for unit in units).items())),
        "content_type_counts": dict(sorted(Counter(
            unit.content_type.value for unit in units).items())),
    }


def verify_against_contents(
    units: tuple[StandardDocumentUnit, ...],
) -> dict[str, object]:
    """Check detected clause headings against the document's own contents table.

    The contents table names each clause and the page it opens on. Comparing the
    two is the one check on extraction that does not consult the extractor, so it
    is what an operator should trust over any internal verdict.
    """
    targets: dict[str, int] = {}

    for unit in units:
        if unit.is_toc_entry:
            target = _contents_target(unit.text)

            if target is not None:
                targets.setdefault(target[0], target[1])

    headings = {unit.section: unit for unit in units
                if unit.layout_kind is StandardLayoutKind.HEADING and unit.section}
    shared = sorted(set(targets) & set(headings), key=lambda value: _clause_parts(value))
    offsets = Counter(headings[value].page - targets[value] for value in shared)
    offset = offsets.most_common(1)[0][0] if offsets else None
    agreed, disagreed = [], []

    for value in shared:
        expected = targets[value] + (offset or 0)
        (agreed if headings[value].page == expected else disagreed).append({
            "clause": value, "contents_page": targets[value],
            "expected_page": expected, "detected_page": headings[value].page,
            "delta": headings[value].page - expected,
        })

    return {
        "contents_entries_with_a_page": len(targets),
        "headings_detected": len(headings),
        "clauses_in_both": len(shared),
        "printed_page_offset": offset,
        "page_agreement": len(agreed),
        "page_disagreement": len(disagreed),
        "agreement_rate": round(len(agreed) / len(shared), 4) if shared else None,
        "advertised_not_detected": sorted(set(targets) - set(headings),
                                          key=lambda value: _clause_parts(value)),
        "detected_not_advertised": len(set(headings) - set(targets)),
        "disagreements": disagreed,
    }


def build_manifest(
    units: tuple[StandardDocumentUnit, ...], *, standard_id: str, revision: str,
    pdf_sha256: str, filename: str, page_count: int,
    warnings: tuple[str, ...], source_origin: str, retain_pdf: bool,
    ingestion_timestamp: str | None = None,
) -> StandardIngestionManifest:
    return StandardIngestionManifest(
        standard_id=standard_id, revision=revision,
        source_pdf_sha256=pdf_sha256, logical_source_filename=filename,
        ingestion_timestamp=(ingestion_timestamp or
                             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        extractor_version=EXTRACTOR_VERSION, page_count=page_count,
        canonical_unit_count=len(units),
        section_count=len({unit.section for unit in units if unit.section}),
        requirement_count=sum(unit.content_type == StandardContentType.REQUIREMENT
                              for unit in units),
        recommendation_count=sum(
            unit.content_type == StandardContentType.RECOMMENDATION for unit in units),
        definition_count=sum(unit.content_type == StandardContentType.DEFINITION
                             for unit in units),
        table_figure_marker_count=sum(unit.content_type in {
            StandardContentType.TABLE, StandardContentType.FIGURE} for unit in units),
        warnings=warnings, extraction_errors=(),
        human_validation_status=HumanValidationStatus.NOT_REVIEWED,
        corpus_manifest_sha256=corpus_fingerprint(units), source_origin=source_origin,
        raw_pdf_retained=retain_pdf,
        corpus_schema_version=STANDARD_UNIT_SCHEMA_VERSION)


def extract_corpus(
    pdf_path: str | Path, *, standard_id: str, revision: str,
    retain_pdf: bool = False, source_origin: str = "LICENSED_STANDARD",
    ingestion_timestamp: str | None = None,
) -> tuple[StandardIngestionManifest, tuple[StandardDocumentUnit, ...], bytes]:
    """Extract a corpus without deciding where -- or whether -- it is stored."""
    source = Path(pdf_path).expanduser()

    if not source.is_file() or source.is_symlink():
        raise StandardIngestionError("PDF path must name a regular local file")

    pdf_bytes = source.read_bytes()
    pdf_sha = hashlib.sha256(pdf_bytes).hexdigest()
    pages, extraction_warnings = extract_pdf_pages(source)
    units = canonical_units(pages, standard_id=standard_id, revision=revision,
                            pdf_sha256=pdf_sha)

    if not units:
        raise StandardIngestionError("PDF extraction produced no canonical units")

    manifest = build_manifest(
        units, standard_id=standard_id, revision=revision, pdf_sha256=pdf_sha,
        filename=source.name, page_count=len(pages), warnings=extraction_warnings,
        source_origin=source_origin, retain_pdf=retain_pdf,
        ingestion_timestamp=ingestion_timestamp)

    return manifest, units, pdf_bytes


def compare_extractions(
    old: tuple[StandardDocumentUnit, ...], candidate: tuple[StandardDocumentUnit, ...],
) -> dict[str, object]:
    """Metadata-only OLD vs CANDIDATE comparison; never any normative text."""
    old_ids = {unit.source_id for unit in old}
    new_ids = {unit.source_id for unit in candidate}
    old_manifest = {unit.source_content_sha256 for unit in old}
    new_manifest = {unit.source_content_sha256 for unit in candidate}

    return {
        "old": extraction_diagnostics(old),
        "candidate": extraction_diagnostics(candidate),
        "source_ids": {
            "retained": len(old_ids & new_ids),
            "removed": len(old_ids - new_ids),
            "added": len(new_ids - old_ids),
        },
        "unit_text": {
            "retained": len(old_manifest & new_manifest),
            "removed": len(old_manifest - new_manifest),
            "added": len(new_manifest - old_manifest),
        },
    }


def ingest_candidate(
    store: StandardStore, pdf_path: str | Path, *, standard_id: str,
    revision: str, with_layout: bool = False,
    source_origin: str = "LICENSED_STANDARD",
    ingestion_timestamp: str | None = None,
) -> tuple[str, dict[str, object]]:
    """Re-extract into an isolated candidate corpus; nothing becomes active."""
    from standard_layout import extract_layout, layout_bytes

    active_manifest = store.load_manifest(standard_id, revision)
    active_units = store.load_units(standard_id, revision)
    manifest, units, _ = extract_corpus(
        pdf_path, standard_id=standard_id, revision=revision,
        source_origin=source_origin, ingestion_timestamp=ingestion_timestamp)

    if manifest.source_pdf_sha256 != active_manifest.source_pdf_sha256:
        from standard_store import StandardCollisionError
        raise StandardCollisionError(
            "candidate was extracted from a different source PDF")

    layout = None
    layout_seconds = None

    if with_layout:
        started = time.monotonic()
        layout = layout_bytes(extract_layout(
            pdf_path, pdf_sha256=manifest.source_pdf_sha256))
        layout_seconds = round(time.monotonic() - started, 3)

    diagnostics = {
        "schema_version": 1,
        "standard_id": standard_id, "revision": revision,
        "source_pdf_sha256": manifest.source_pdf_sha256,
        "candidate_corpus_sha256": manifest.corpus_manifest_sha256,
        "active_corpus_sha256": active_manifest.corpus_manifest_sha256,
        "active_extractor_version": active_manifest.extractor_version,
        "candidate_extractor_version": manifest.extractor_version,
        "layout_artifact_present": layout is not None,
        "layout_extraction_seconds": layout_seconds,
        "layout_artifact_bytes": len(layout) if layout is not None else None,
        "comparison": compare_extractions(active_units, units),
    }
    candidate_id = store.save_candidate(
        manifest, units, diagnostics=diagnostics, layout=layout)

    return candidate_id, diagnostics


def ingest_pdf(
    store: StandardStore, pdf_path: str | Path, *, standard_id: str,
    revision: str, retain_pdf: bool = False,
    source_origin: str = "LICENSED_STANDARD",
    ingestion_timestamp: str | None = None,
) -> StandardIngestionManifest:
    source = Path(pdf_path).expanduser()

    if not source.is_file() or source.is_symlink():
        raise StandardIngestionError("PDF path must name a regular local file")

    pdf_sha = hashlib.sha256(source.read_bytes()).hexdigest()

    # Reject a colliding revision before parsing potentially sensitive content.

    try:
        existing = store.load_manifest(standard_id, revision)
    except FileNotFoundError:
        existing = None

    if existing is not None and existing.source_pdf_sha256 != pdf_sha:
        from standard_store import StandardCollisionError
        raise StandardCollisionError("revision label already exists with different source PDF")

    manifest, units, pdf_bytes = extract_corpus(
        source, standard_id=standard_id, revision=revision, retain_pdf=retain_pdf,
        source_origin=source_origin, ingestion_timestamp=ingestion_timestamp)

    # Identical reingestion returns the prior timestamp-bearing manifest.

    return store.save_ingestion(
        manifest, units, source_pdf=(pdf_bytes if retain_pdf else None))
