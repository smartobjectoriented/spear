"""Operator-only human review of a standard's extraction review file.

The automated review that produced the file is evidence, not approval. This
module lets an operator walk the rows, see enough local text to judge each one,
and record a human verdict beside the automated one -- never over it.

It is deliberately not a model tool: nothing here is registered with the tool
registry, and normative text is rendered to the operator's terminal only, never
written into the review file or any log.
"""

from __future__ import annotations

import getpass
import json
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from state_paths import standards_root
from standard_schema import canonical_json
from standard_store import StandardStore, StandardStoreError, candidate_id_for


REVIEW_SCHEMA_VERSION = 2
REVIEW_REPORT_KIND = "STD1E_OPERATOR_REVIEW"
UNREVIEWED = "UNREVIEWED"
HUMAN_VERDICTS = ("PASS", "ACCEPTABLE_WARNING", "FAIL", "NEEDS_FOLLOWUP")
BLOCKING_VERDICTS = ("FAIL", "NEEDS_FOLLOWUP")
_EXCERPT_CHARS = 700
_CONTEXT_CHARS = 180
_CONTEXT_UNITS = 1
_VERDICT_KEYS = {"p": "PASS", "w": "ACCEPTABLE_WARNING",
                 "f": "FAIL", "n": "NEEDS_FOLLOWUP"}


class StandardReviewError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReviewItem:
    """One row of the review file, as the operator sees it."""

    position: int
    source_id: str
    page: int
    section: str | None
    content_type: str
    layout_kind: str
    stratum: str
    warnings: tuple[str, ...]
    automated_verdict: str | None
    automated_note: str | None
    human_verdict: str | None
    review_status: str
    reviewed_at: str | None
    reviewed_by: str | None
    previous_source_id: str | None
    previous_section: str | None

    @property
    def reviewed(self) -> bool:
        return self.review_status != UNREVIEWED


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_reviewer() -> str:
    """One operator identity for the whole session, never per row."""

    for value in (os.environ.get("SPEAR_OPERATOR"), os.environ.get("USER")):
        if value and value.strip():
            return value.strip()

    try:
        return getpass.getuser()
    except Exception:
        return "operator"


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    if path.exists() and path.is_symlink():
        raise StandardReviewError("refusing to replace a symlink")

    handle, temporary = tempfile.mkstemp(prefix=".review-", dir=path.parent)

    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())

        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)

        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


DEFAULT_SAMPLE_PER_STRATUM = 6


def _strata(units):
    """Named slices of the corpus an operator should look at, as predicates.

    The annex slice deliberately excludes front matter: a contents entry naming
    an appendix is not an annex opener, and sampling one reviews the wrong thing.
    """
    from standard_ingest import _ANNEX, _NUMBERED
    from standard_schema import StandardContentType, StandardLayoutKind

    def heading(unit):
        return unit.layout_kind is StandardLayoutKind.HEADING

    return (
        ("genuine_top_level_heading",
         lambda u: heading(u) and "." not in (u.section or ".")),
        ("genuine_deep_heading",
         lambda u: heading(u) and (u.section or "").count(".") >= 2),
        ("heading_accepted_by_resync",
         lambda u: heading(u) and any("does not continue" in w for w in u.warnings)),
        ("demoted_reopened_clause",
         lambda u: any("already opened earlier" in w for w in u.warnings)),
        ("coarse_unit",
         lambda u: any("merges more than one" in w for w in u.warnings)),
        ("contents_entry", lambda u: u.is_toc_entry),
        ("page_furniture",
         lambda u: u.content_type is StandardContentType.PAGE_FURNITURE),
        ("front_matter_non_entry",
         lambda u: u.is_front_matter and not u.is_toc_entry),
        ("table_caption", lambda u: u.content_type is StandardContentType.TABLE),
        ("figure_caption", lambda u: u.content_type is StandardContentType.FIGURE),
        ("columnar_row",
         lambda u: u.layout_kind is StandardLayoutKind.COLUMNAR),
        ("cross_page_table_continuation", lambda u: u.possible_table_continuation),
        ("numbered_body_paragraph_not_a_heading",
         lambda u: _NUMBERED.match(u.text) and u.layout_kind is StandardLayoutKind.PROSE),
        # A body annex opener, not the contents line that lists it.
        ("annex_heading",
         lambda u: bool(_ANNEX.match(u.text)) and not u.is_front_matter
         and any("resets clause numbering" in w for w in u.warnings)),
        ("requirement",
         lambda u: u.content_type is StandardContentType.REQUIREMENT),
        ("recommendation",
         lambda u: u.content_type is StandardContentType.RECOMMENDATION),
        ("definition", lambda u: u.content_type is StandardContentType.DEFINITION),
        ("informative",
         lambda u: u.content_type is StandardContentType.INFORMATIVE),
    )


def build_review_sample(
    store: StandardStore, standard_id: str, revision: str, *,
    per_stratum: int = DEFAULT_SAMPLE_PER_STRATUM,
) -> dict[str, object]:
    """A stratified sample of the active corpus for an operator to review."""
    manifest = store.verify_corpus(standard_id, revision)
    units = sorted(store.load_units(standard_id, revision),
                   key=lambda unit: unit.unit_position)
    rows: list[dict[str, object]] = []

    for name, predicate in _strata(units):
        taken = 0

        for unit in units:
            if taken >= per_stratum:
                break

            if not predicate(unit):
                continue

            taken += 1
            rows.append({
                "reason": name,
                "candidate_source_id": unit.source_id,
                "page": unit.page,
                "candidate_section": unit.section,
                "content_type": unit.content_type.value,
                "layout_kind": unit.layout_kind.value,
                "retrievable": unit.retrievable,
                "is_front_matter": unit.is_front_matter,
                "is_toc_entry": unit.is_toc_entry,
                "possible_table_continuation": unit.possible_table_continuation,
                "needs_structured_review": unit.needs_structured_review,
                "chars": len(unit.text),
                "warnings": list(unit.warnings),
                "automated_verdict": None,
                "automated_note": None,
                "human_verdict": None,
                "review_status": UNREVIEWED,
            })

    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "report_kind": REVIEW_REPORT_KIND,
        "generated_at": _now(),
        "standard_id": standard_id, "revision": revision,
        "corpus_manifest_sha256": manifest.corpus_manifest_sha256,
        "source_pdf_sha256": manifest.source_pdf_sha256,
        "extractor_version": manifest.extractor_version,
        "rows": rows,
    }


def write_review_sample(
    store: StandardStore, standard_id: str, revision: str, *,
    per_stratum: int = DEFAULT_SAMPLE_PER_STRATUM,
) -> tuple[Path, Path | None]:
    """Write a fresh sample, keeping any existing review beside it as history."""
    target = review_path(store, standard_id, revision)
    archived = None

    if target.is_file() and not target.is_symlink():
        archived = target.parent / "history" / f"{target.stem}-{_now()}.json"
        _atomic_write(archived, target.read_bytes())

    document = build_review_sample(store, standard_id, revision,
                                   per_stratum=per_stratum)
    _atomic_write(target, canonical_json(document))

    return target, archived


def review_path(store: StandardStore, standard_id: str, revision: str) -> Path:
    return (store.revision_dir(standard_id, revision) / "evaluation"
            / "std1e-operator-review.json")


class StandardReview:
    """A review file bound to one canonical corpus, with human verdicts."""

    def __init__(self, store: StandardStore, standard_id: str, revision: str,
                 document: Mapping[str, object], path: Path,
                 reviewer: str | None = None) -> None:
        self.store = store
        self.standard_id = standard_id
        self.revision = revision
        self.path = path
        self.reviewer = reviewer or default_reviewer()
        self._document = dict(document)
        self._rows = [dict(row) for row in document["rows"]]
        self._ordered: tuple | None = None

    # -- loading ---------------------------------------------------------

    @classmethod
    def load(cls, store: StandardStore, standard_id: str, revision: str, *,
             path: Path | None = None, reviewer: str | None = None) -> "StandardReview":
        target = path or review_path(store, standard_id, revision)

        if target.is_symlink():
            raise StandardReviewError("review file must not be a symlink")

        try:
            raw = json.loads(target.read_text("utf-8"))
        except FileNotFoundError as exc:
            raise StandardReviewError(f"no review file at {target}") from exc
        except (ValueError, OSError) as exc:
            raise StandardReviewError(f"review file is unreadable: {exc}") from exc

        document = cls._validated(raw, store, standard_id, revision)

        return cls(store, standard_id, revision, document, target, reviewer)

    @staticmethod
    def _validated(raw: object, store: StandardStore, standard_id: str,
                   revision: str) -> Mapping[str, object]:
        if not isinstance(raw, dict):
            raise StandardReviewError("review file must be a JSON object")

        if raw.get("report_kind") != REVIEW_REPORT_KIND:
            raise StandardReviewError("review file is not a standard review report")

        if (raw.get("standard_id"), raw.get("revision")) != (standard_id, revision):
            raise StandardReviewError("review file targets a different standard")

        rows = raw.get("rows")

        if not isinstance(rows, list) or not rows:
            raise StandardReviewError("review file has no rows")

        # The corpus is the authority. A review of an older generation must not
        # be applied to the corpus that replaced it.

        active = store.verify_corpus(standard_id, revision).corpus_manifest_sha256
        declared = raw.get("corpus_manifest_sha256")

        if declared is None:
            # A first-generation file names the candidate it reviewed; accept it
            # only when that candidate is what is active now, then pin it.

            if raw.get("candidate_id") != candidate_id_for(active):
                raise StandardReviewError(
                    "review file predates the active corpus; re-run the review")

            raw = {**raw, "corpus_manifest_sha256": active}
        elif declared != active:
            raise StandardReviewError(
                "review file was written against a different canonical corpus")

        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise StandardReviewError(f"row {index} is malformed")

            source_id = row.get("candidate_source_id") or row.get("source_id")

            if not isinstance(source_id, str):
                raise StandardReviewError(f"row {index} has no source id")

            # Resolve through the store, which validates the id and confines the
            # path. Any inspect_path recorded in the file is ignored.

            try:
                store.resolve_source(standard_id, revision, source_id)
            except (StandardStoreError, KeyError) as exc:
                raise StandardReviewError(
                    f"row {index} names a source the corpus does not hold: {exc}") from exc

            status = row.get("review_status", UNREVIEWED)

            if status != UNREVIEWED and status not in HUMAN_VERDICTS:
                raise StandardReviewError(f"row {index} has an unknown review status")

        return raw

    # -- items -----------------------------------------------------------

    def _item(self, position: int) -> ReviewItem:
        row = self._rows[position]
        return ReviewItem(
            position=position,
            source_id=row.get("candidate_source_id") or row["source_id"],
            page=int(row.get("page", 0)),
            section=row.get("candidate_section"),
            content_type=str(row.get("content_type", "")),
            layout_kind=str(row.get("layout_kind", "")),
            stratum=str(row.get("reason", "")),
            warnings=tuple(row.get("warnings", ())),
            automated_verdict=row.get("automated_verdict", row.get("review_verdict")),
            automated_note=row.get("automated_note", row.get("review_note")),
            human_verdict=row.get("human_verdict"),
            review_status=row.get("review_status", UNREVIEWED),
            reviewed_at=row.get("reviewed_at"),
            reviewed_by=row.get("reviewed_by"),
            previous_source_id=row.get("old_source_id"),
            previous_section=row.get("old_section"),
        )

    def items(self) -> tuple[ReviewItem, ...]:
        return tuple(self._item(index) for index in range(len(self._rows)))

    def select(self, *, only_unreviewed: bool = False, only_warning: bool = False,
               stratum: str | None = None) -> tuple[int, ...]:
        chosen = []

        for item in self.items():
            if only_unreviewed and item.reviewed:
                continue

            if only_warning and item.automated_verdict != "ACCEPTABLE_WARNING":
                continue

            if stratum is not None and item.stratum != stratum:
                continue

            chosen.append(item.position)

        return tuple(chosen)

    def strata(self) -> tuple[str, ...]:
        return tuple(sorted({item.stratum for item in self.items()}))

    def resume_at(self, positions: Sequence[int]) -> int:
        """The first position still awaiting a human verdict."""

        for offset, position in enumerate(positions):
            if not self._item(position).reviewed:
                return offset

        return 0

    # -- local evidence --------------------------------------------------

    def _units(self) -> tuple:
        if self._ordered is None:
            units = self.store.load_units(self.standard_id, self.revision)
            self._ordered = tuple(sorted(
                units, key=lambda unit: (unit.unit_position, unit.page, unit.source_id)))

        return self._ordered

    @staticmethod
    def _bounded(text: str, limit: int) -> str:
        collapsed = " ".join(text.split())
        return collapsed if len(collapsed) <= limit else collapsed[:limit - 1] + "…"

    def excerpt(self, position: int, limit: int = _EXCERPT_CHARS) -> str:
        """A bounded excerpt of the unit, read through the store."""
        unit = self.store.resolve_source(
            self.standard_id, self.revision, self._item(position).source_id)

        return self._bounded(unit.text, limit)

    def context(self, position: int, *, neighbours: int = _CONTEXT_UNITS,
                limit: int = _CONTEXT_CHARS) -> tuple[tuple[str, str], ...]:
        """Bounded text of the units immediately before and after this one."""
        source_id = self._item(position).source_id
        ordered = self._units()

        try:
            index = next(offset for offset, unit in enumerate(ordered)
                         if unit.source_id == source_id)
        except StopIteration:  # pragma: no cover - resolve_source already guards
            return ()

        window = []

        for offset in range(max(0, index - neighbours), index):
            window.append(("before", self._bounded(ordered[offset].text, limit)))

        for offset in range(index + 1, min(len(ordered), index + 1 + neighbours)):
            window.append(("after", self._bounded(ordered[offset].text, limit)))

        return tuple(window)

    # -- verdicts --------------------------------------------------------

    def record(self, position: int, verdict: str) -> ReviewItem:
        if verdict not in HUMAN_VERDICTS:
            raise StandardReviewError(f"unknown human verdict: {verdict}")

        row = self._rows[position]
        row.setdefault("automated_verdict", row.pop("review_verdict", None))
        row.setdefault("automated_note", row.pop("review_note", None))
        row.pop("review_verdict", None)
        row.pop("review_note", None)
        row.update({"human_verdict": verdict, "review_status": verdict,
                    "reviewed_at": _now(), "reviewed_by": self.reviewer})

        return self._item(position)

    def clear(self, position: int) -> ReviewItem:
        row = self._rows[position]
        row.update({"human_verdict": None, "review_status": UNREVIEWED,
                    "reviewed_at": None, "reviewed_by": None})

        return self._item(position)

    # -- summary and gate ------------------------------------------------

    def summary(self) -> dict[str, int]:
        counts = {"TOTAL": len(self._rows), UNREVIEWED: 0}
        counts.update({verdict: 0 for verdict in HUMAN_VERDICTS})

        for item in self.items():
            counts[item.review_status if item.reviewed else UNREVIEWED] += 1

        return counts

    def is_complete(self) -> bool:
        return all(item.reviewed for item in self.items())

    def blocking(self) -> dict[str, int]:
        counts = self.summary()
        return {verdict: counts[verdict] for verdict in BLOCKING_VERDICTS
                if counts[verdict]}

    def approval_state(self) -> str:
        if not self.is_complete():
            return "REVIEW_INCOMPLETE"

        return "APPROVAL_BLOCKED" if self.blocking() else "ELIGIBLE_FOR_APPROVAL"

    # -- persistence -----------------------------------------------------

    def save(self) -> Path:
        """Rewrite the private review file atomically, verdicts and IDs only."""
        active = self.store.verify_corpus(
            self.standard_id, self.revision).corpus_manifest_sha256

        if active != self._document.get("corpus_manifest_sha256"):
            raise StandardReviewError(
                "active canonical corpus changed during review; refusing to save")

        rows = []

        for row in self._rows:
            value = {key: item for key, item in row.items()
                     # A stored path is not evidence, and must never be read back.
                     if key != "inspect_path"}
            value.setdefault("automated_verdict", value.pop("review_verdict", None))
            value.setdefault("automated_note", value.pop("review_note", None))
            value.pop("review_verdict", None)
            value.pop("review_note", None)
            value.setdefault("human_verdict", None)
            value.setdefault("review_status", UNREVIEWED)
            rows.append(value)

        document = {
            **self._document,
            "schema_version": REVIEW_SCHEMA_VERSION,
            "report_kind": REVIEW_REPORT_KIND,
            "corpus_manifest_sha256": active,
            "human_review": {
                "reviewer": self.reviewer,
                "updated_at": _now(),
                "summary": self.summary(),
                "complete": self.is_complete(),
                "approval_state": self.approval_state(),
            },
            "rows": rows,
        }
        self._document = document
        _atomic_write(self.path, canonical_json(document))

        return self.path


# ---------------------------------------------------------------------------
# terminal review
# ---------------------------------------------------------------------------

_HELP = """
  p  PASS                 w  ACCEPTABLE_WARNING
  f  FAIL                 n  NEEDS_FOLLOWUP
  s  skip (leave unreviewed)      u  clear this row's verdict
  b  previous item        j  next item
  q  save and quit        ?  this help
"""


def _render(review: StandardReview, position: int, offset: int, total: int,
            out) -> None:
    item = review._item(position)
    out.write("\n" + "=" * 72 + "\n")
    out.write(f"  {offset + 1} / {total}   row {position + 1} of "
              f"{len(review.items())}   [{item.stratum}]\n")
    out.write("=" * 72 + "\n")
    out.write(f"  page {item.page}    section {item.section!r}    "
              f"{item.content_type} / {item.layout_kind}\n")
    out.write(f"  source        {item.source_id}\n")

    if item.previous_source_id:
        out.write(f"  previous      {item.previous_source_id} "
                  f"(section {item.previous_section!r})\n")
    elif item.previous_section is not None:
        out.write(f"  previous section {item.previous_section!r}\n")

    if item.warnings:
        for warning in item.warnings:
            out.write(f"  warning       {warning}\n")

    out.write(f"  automated     {item.automated_verdict or '-'}"
              f"{' -- ' + item.automated_note if item.automated_note else ''}\n")
    human = item.human_verdict or UNREVIEWED
    out.write(f"  human         {human}"
              f"{' by ' + item.reviewed_by + ' at ' + item.reviewed_at if item.reviewed else ''}\n")
    out.write("-" * 72 + "\n")

    for where, text in review.context(position):
        if where == "before":
            out.write(f"  ...{text}\n\n")

    out.write(f"  {review.excerpt(position)}\n")

    for where, text in review.context(position):
        if where == "after":
            out.write(f"\n  {text}...\n")

    out.write("-" * 72 + "\n")


def _write_summary(review: StandardReview, out, *, heading: str) -> None:
    counts = review.summary()
    out.write(f"\n{heading}\n")

    for key in ("TOTAL", UNREVIEWED, *HUMAN_VERDICTS):
        out.write(f"  {key:<20} {counts[key]}\n")

    if review.is_complete():
        if review.blocking():
            detail = ", ".join(f"{key}={value}"
                               for key, value in review.blocking().items())
            out.write(f"\nHUMAN REVIEW COMPLETE\n"
                      f"REVIEW COMPLETE -- approval blocked by review findings "
                      f"({detail})\n")
        else:
            out.write("\nHUMAN REVIEW COMPLETE\n"
                      "REVIEW COMPLETE -- eligible for operator approval\n"
                      f"  next: /standard approve {review.standard_id} "
                      f"{review.revision}\n")


def run_review(review: StandardReview, positions: Sequence[int], *,
               stdin=None, stdout=None) -> int:
    """Drive the terminal review loop. Returns the number of verdicts recorded."""
    stdin = stdin or sys.stdin
    out = stdout or sys.stdout

    if not positions:
        out.write("No items match the selected filter.\n")
        return 0

    _write_summary(review, out, heading="REVIEW SUMMARY")
    out.write(f"\nReviewing as {review.reviewer}. Press ? for help.\n")
    offset = review.resume_at(positions)
    recorded = 0

    while 0 <= offset < len(positions):
        position = positions[offset]
        _render(review, position, offset, len(positions), out)
        out.write("  [p]ass [w]arning [f]ail [n]eeds-followup [s]kip "
                  "[b]ack [q]uit ? > ")
        out.flush()
        line = stdin.readline()

        if not line:
            break

        key = line.strip().lower()

        if key in _VERDICT_KEYS:
            review.record(position, _VERDICT_KEYS[key])
            recorded += 1
            offset += 1
        elif key == "u":
            review.clear(position)
        elif key in ("s", "j", ""):
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
    _write_summary(review, out, heading="REVIEW SUMMARY")
    out.write(f"\nSaved to {review.path}\n")

    return recorded


def main(argv: Sequence[str] | None = None, *, stdin=None, stdout=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="standard_review",
        description="Operator-only human review of a standard's extraction review file.")
    parser.add_argument("--standard", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--root", default=None,
                        help="standards store root (default: $SPEAR_STATE_DIR/standards)")
    parser.add_argument("--reviewer", default=None)
    parser.add_argument("--only-unreviewed", action="store_true")
    parser.add_argument("--only-warning", action="store_true")
    parser.add_argument("--stratum", default=None)
    parser.add_argument("--summary", action="store_true",
                        help="print the summary and exit without reviewing")
    parser.add_argument("--build-sample", action="store_true",
                        help="write a fresh stratified sample, archiving any existing "
                             "review beside it, then exit")
    options = parser.parse_args(argv)
    out = stdout or sys.stdout

    root = options.root or str(standards_root())

    try:
        store = StandardStore(root)

        if options.build_sample:
            target, archived = write_review_sample(
                store, options.standard, options.revision)
            out.write(f"wrote {target}\n")
            out.write(f"previous review archived to {archived}\n" if archived
                      else "no previous review to archive\n")

            return 0

        review = StandardReview.load(store, options.standard, options.revision,
                                     reviewer=options.reviewer)
    except (StandardReviewError, StandardStoreError, FileNotFoundError) as exc:
        out.write(f"review unavailable: {exc}\n")
        return 2

    if options.stratum is not None and options.stratum not in review.strata():
        out.write(f"unknown stratum {options.stratum!r}; available: "
                  f"{', '.join(review.strata())}\n")
        return 2

    if options.summary:
        _write_summary(review, out, heading="REVIEW SUMMARY")
        return 0

    positions = review.select(only_unreviewed=options.only_unreviewed,
                              only_warning=options.only_warning,
                              stratum=options.stratum)
    run_review(review, positions, stdin=stdin, stdout=out)

    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
