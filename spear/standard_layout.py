"""Derived page geometry for a standard, kept beside the canonical corpus.

STD1E keeps canonical text on ``pdftotext -layout`` and records geometry as a
*separate* derived artifact rather than rebuilding canonical units around
coordinates. Citation identity therefore stays a property of the text corpus,
while a later geometry-aware pass can reconstruct table structure from the same
locally extracted PDF. This artifact carries no semantics: no cells, no fields,
no bit positions.
"""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
from pathlib import Path

from standard_schema import canonical_json, sha256_json


LAYOUT_ARTIFACT_VERSION = 1
LAYOUT_EXTRACTOR_VERSION = "poppler-tsv-v1"
_PAGE, _BLOCK, _LINE, _WORD = 1, 3, 4, 5
_MARKER = re.compile(r"^###[A-Z]+###$")


class StandardLayoutError(RuntimeError):
    pass


def _finite(value: str) -> float:
    number = float(value)

    if not math.isfinite(number):
        raise StandardLayoutError("layout coordinate is not finite")

    return number


def _bbox(left: str, top: str, width: str, height: str) -> list[float]:
    x0, y0 = _finite(left), _finite(top)
    w, h = _finite(width), _finite(height)

    if w < 0 or h < 0:
        raise StandardLayoutError("layout box has a negative extent")

    return [round(x0, 3), round(y0, 3), round(x0 + w, 3), round(y0 + h, 3)]


def extract_layout(pdf_path: str | Path, *, pdf_sha256: str) -> dict[str, object]:
    """Return word-level geometry for every page, from a local Poppler run."""
    source = Path(pdf_path).expanduser()

    if not source.is_file() or source.is_symlink():
        raise StandardLayoutError("PDF path must name a regular local file")

    try:
        completed = subprocess.run(
            ["pdftotext", "-tsv", "-enc", "UTF-8", str(source), "-"],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StandardLayoutError(f"layout extraction unavailable: {exc}") from exc

    if completed.returncode != 0:
        raise StandardLayoutError("layout extraction failed")

    pages: list[dict[str, object]] = []
    page = block = line = None

    for row in completed.stdout.splitlines()[1:]:
        columns = row.split("\t")

        if len(columns) < 12:
            continue

        try:
            level = int(columns[0])
            number = int(columns[1])
        except ValueError:
            raise StandardLayoutError("layout row has a malformed level or page")

        text = columns[11]

        if level == _PAGE:
            # Poppler leaves the previous page's coordinates in a page row's
            # left/top; only its width and height are meaningful. Word
            # coordinates are given in a page frame whose origin is (0, 0).

            width, height = _finite(columns[8]), _finite(columns[9])

            if width <= 0 or height <= 0:
                raise StandardLayoutError("page has no positive extent")

            page = {"page": number, "size": [0.0, 0.0, round(width, 3),
                                             round(height, 3)], "blocks": []}
            pages.append(page)
            block = line = None
        elif page is None:
            raise StandardLayoutError("layout row appeared before any page")
        elif level == _BLOCK:
            block = {"block": len(page["blocks"]), "bbox": _bbox(*columns[6:10]),
                     "lines": []}
            page["blocks"].append(block)
            line = None
        elif level == _LINE and block is not None:
            line = {"line": len(block["lines"]), "bbox": _bbox(*columns[6:10]),
                    "words": []}
            block["lines"].append(line)
        elif level == _WORD and line is not None and not _MARKER.match(text):
            line["words"].append({"text": text, "bbox": _bbox(*columns[6:10])})

    if not pages:
        raise StandardLayoutError("layout extraction produced no pages")

    artifact = {
        "schema_version": 1,
        "artifact_version": LAYOUT_ARTIFACT_VERSION,
        "layout_extractor_version": LAYOUT_EXTRACTOR_VERSION,
        "source_pdf_sha256": pdf_sha256,
        "page_count": len(pages),
        "word_count": sum(len(line["words"]) for entry in pages
                          for block in entry["blocks"] for line in block["lines"]),
        "pages": pages,
    }
    artifact["layout_fingerprint"] = sha256_json({
        "artifact_version": LAYOUT_ARTIFACT_VERSION,
        "layout_extractor_version": LAYOUT_EXTRACTOR_VERSION,
        "pages": pages,
    })

    return artifact


def validate_layout(artifact: object, *, pdf_sha256: str | None = None) -> dict[str, object]:
    """Fail closed on a layout artifact that is malformed or from another PDF."""

    if not isinstance(artifact, dict):
        raise StandardLayoutError("layout artifact must be an object")

    if artifact.get("artifact_version") != LAYOUT_ARTIFACT_VERSION:
        raise StandardLayoutError("unsupported layout artifact version")

    pages = artifact.get("pages")

    if not isinstance(pages, list) or not pages:
        raise StandardLayoutError("layout artifact has no pages")

    if pdf_sha256 is not None and artifact.get("source_pdf_sha256") != pdf_sha256:
        raise StandardLayoutError("layout artifact belongs to a different PDF")

    for entry in pages:
        if not isinstance(entry, dict) or not isinstance(entry.get("blocks"), list):
            raise StandardLayoutError("layout page is malformed")

        size = entry.get("size")

        if (not isinstance(size, list) or len(size) != 4 or size[:2] != [0.0, 0.0]
                or size[2] <= 0 or size[3] <= 0):
            raise StandardLayoutError("layout page has no usable page box")

        for box in _boxes(entry):
            if (not isinstance(box, list) or len(box) != 4
                    or not all(isinstance(value, (int, float)) and math.isfinite(value)
                               for value in box)
                    or box[2] < box[0] or box[3] < box[1]):
                raise StandardLayoutError("layout box is malformed")

    expected = sha256_json({
        "artifact_version": artifact["artifact_version"],
        "layout_extractor_version": artifact.get("layout_extractor_version"),
        "pages": pages,
    })

    if artifact.get("layout_fingerprint") != expected:
        raise StandardLayoutError("layout fingerprint mismatch")

    return artifact


def _boxes(page: dict[str, object]):
    yield page.get("size")

    for block in page["blocks"]:
        if not isinstance(block, dict) or not isinstance(block.get("lines"), list):
            raise StandardLayoutError("layout block is malformed")

        yield block.get("bbox")

        for line in block["lines"]:
            if not isinstance(line, dict) or not isinstance(line.get("words"), list):
                raise StandardLayoutError("layout line is malformed")

            yield line.get("bbox")

            for word in line["words"]:
                if not isinstance(word, dict) or not isinstance(word.get("text"), str):
                    raise StandardLayoutError("layout word is malformed")

                yield word.get("bbox")


def layout_bytes(artifact: dict[str, object]) -> bytes:
    return canonical_json(artifact)
