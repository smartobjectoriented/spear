"""Conservative one-hop resolution of explicit standard section references."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from standard_schema import (
    StandardCrossReferenceIndexManifest, StandardLayoutKind, sha256_json,
)
from standard_progress import report
from standard_store import StandardStore


RESOLVER_VERSION = "standard-crossrefs-v1"
_REFERENCE = re.compile(
    r"\b(?:see|refer(?:red)?\s+to|(?:as|is)\s+specified\s+in)\s+(?:Section\s+)?"
    r"(?P<section>\d+(?:\.\d+)+)\b", re.IGNORECASE,
)


def _section_anchor(unit) -> bool:
    """The one unit that opens a clause, so a reference to it is unambiguous."""

    if not unit.section:
        return False

    if unit.schema_version >= 2:
        # An extractor that classified the heading says so; a numbered list item
        # or a contents entry that merely starts with the digits does not.

        return unit.layout_kind is StandardLayoutKind.HEADING

    text = unit.text.strip()

    return bool(re.match(rf"^{re.escape(unit.section)}(?:\s|$)", text))


def rebuild_cross_reference_index(
    store: StandardStore, standard_id: str, revision: str, *,
    resolver_version: str = RESOLVER_VERSION, created_at: str | None = None,
    progress=None,
) -> StandardCrossReferenceIndexManifest:
    report(progress, "verifying corpus")
    source = store.verify_corpus(standard_id, revision)
    units = store.load_units(standard_id, revision)
    sections: dict[str, list] = {}

    for unit in units:
        if unit.section:
            sections.setdefault(unit.section, []).append(unit)

    entries = {}
    counts = {"resolved": 0, "ambiguous": 0, "unresolved": 0}

    for position, unit in enumerate(units, 1):
        report(progress, "cross references", position, len(units))
        literals = list(unit.cross_references)
        known_sections = {literal for literal in literals
                          if re.fullmatch(r"\d+(?:\.\d+)+", literal)}

        for match in _REFERENCE.finditer(unit.text):
            literal = match.group(0)

            if match.group("section") not in known_sections:
                literals.append(literal)
                known_sections.add(match.group("section"))

        relations = []

        for literal in literals:
            match = _REFERENCE.search(literal)
            section = (match.group("section") if match else literal
                       if re.fullmatch(r"\d+(?:\.\d+)+", literal) else None)
            candidates = sections.get(section or "", [])
            anchors = [candidate for candidate in candidates if _section_anchor(candidate)]

            if len(anchors) == 1:
                status, targets = "resolved", [anchors[0].source_id]
            elif not candidates:
                status, targets = "unresolved", []
            else:
                status, targets = "ambiguous", []

            counts[status] += 1
            relations.append({"literal": literal, "section": section,
                              "status": status, "target_source_ids": targets})

        if relations:
            entries[unit.source_id] = relations

    index = {
        "schema_version": 1, "standard_id": standard_id, "revision": revision,
        "resolver_version": resolver_version, "entries": dict(sorted(entries.items())),
        "counts": counts,
    }
    fingerprint = sha256_json({
        "resolver_version": resolver_version,
        "source_corpus_sha256": source.corpus_manifest_sha256,
        "index": index,
    })
    manifest = StandardCrossReferenceIndexManifest(
        standard_id, revision, source.source_pdf_sha256,
        source.corpus_manifest_sha256, resolver_version,
        counts["resolved"], counts["ambiguous"], counts["unresolved"],
        fingerprint,
        created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    store.save_cross_reference_index(manifest, index)

    return manifest
