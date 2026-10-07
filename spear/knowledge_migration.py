"""Legacy remembered notes into workspace knowledge, on request only.

Before workspace knowledge, /remember and the legacy runtime's remember tool
appended notes to a Markdown file per corpus, with an optional sidecar saying
who wrote each. Those notes are no longer shown to a turn. This moves the ones
worth keeping into the knowledge store, with no more authority than their
origin earns:

    written by the operator (/remember)    ACTIVE, USER_CONFIRMED
    written by the model (remember tool)   PROPOSED, MODEL_DERIVED
    origin unknown (no sidecar entry)      PROPOSED, IMPORTED, UNVERIFIED
    an instruction                         skipped -- it belongs in a rule
    transient, too long, or empty          skipped as invalid
    already migrated, or already known     skipped as a duplicate

A dry run counts and changes nothing. Every migrated record keeps the note's
id as its source, so running it twice adds nothing. The Markdown file is left
as it was -- a marker beside it records the migration -- so it can be read
again, or migrated again, if needed.
"""

from __future__ import annotations

import json
import os
import time

import workspace_knowledge as wk
from memory_store import MarkdownMemoryStore, MemoryScope, MemorySource, MemoryStoreError

OUTCOMES = ("user_confirmed", "proposed_model_derived", "imported_unknown", "duplicate",
            "skipped_rule_like", "invalid", "superseded")

_ORIGIN = {
    MemorySource.USER: (wk.Provenance.USER_CONFIRMED, "user_confirmed"),
    MemorySource.AUTHORIZED_TOOL: (wk.Provenance.MODEL_DERIVED, "proposed_model_derived"),
    MemorySource.IMPORTED: (wk.Provenance.IMPORTED, "imported_unknown"),
}


def marker(path: str) -> str:
    return f"{path}.migrated.json"


def legacy_notes(path: str):
    """The notes of one legacy file, read with their sidecar when it reads."""
    try:
        return MarkdownMemoryStore(path, default_scope=MemoryScope.PROJECT).records()
    except MemoryStoreError:
        return MarkdownMemoryStore(path, default_scope=MemoryScope.PROJECT,
                                   read_metadata=False).records()


def migrate(store: wk.KnowledgeStore, workspace_id: str, path: str, *,
            apply: bool = False) -> dict:
    """Counts by outcome (and, when applied, the ids created). Note text is
    never part of the report."""
    counts = {outcome: 0 for outcome in OUTCOMES}
    created = []

    if not path or not os.path.isfile(path):
        return {"file": path, "found": 0, "applied": apply, "counts": counts, "created": []}

    notes = legacy_notes(path)

    for note in notes:
        if not note.active or note.superseded_by:
            counts["superseded"] += 1
            continue

        if store.find_source(workspace_id, "legacy-note", note.memory_id) is not None:
            counts["duplicate"] += 1
            continue

        statement = " ".join(note.content.split())

        if wk.instruction(statement):
            counts["skipped_rule_like"] += 1
            continue

        if (not statement or wk.transient(statement)
                or len(statement) > wk.STATEMENT_CHARS):
            counts["invalid"] += 1
            continue

        provenance, outcome = _ORIGIN.get(note.source, _ORIGIN[MemorySource.IMPORTED])
        source = wk.SourceRef("legacy-note", note.memory_id,
                              wk._digest(note.content.encode("utf-8")))

        if store.duplicate(workspace_id, wk.Kind.PROJECT_FACT, statement[:60], statement):
            counts["duplicate"] += 1
            continue

        counts[outcome] += 1

        if apply:
            record = store.add(workspace_id, kind=wk.Kind.PROJECT_FACT, subject=statement[:60],
                               statement=statement, provenance=provenance, sources=(source,))
            created.append(record.record_id)

    report = {"file": path, "found": len(notes), "applied": apply, "counts": counts,
              "created": created}

    if apply:
        with open(marker(path), "w", encoding="utf-8") as handle:
            json.dump({"workspace": workspace_id,
                       "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "counts": counts, "created": created}, handle, indent=1)

    return report
