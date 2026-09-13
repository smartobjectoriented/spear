"""Private store for derived table geometry, kept beside the canonical corpus.

Structures are derived, never authoritative. The store pins the corpus and
layout fingerprints they were built from, so a structure set that no longer
matches the corpus fails closed instead of being read as current.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from standard_schema import canonical_json
from standard_store import StandardStore, StandardStoreError
from standard_structure import (
    STRUCTURE_EXTRACTOR_VERSION, STRUCTURE_SCHEMA_VERSION, GeometryStatus,
    StandardStructureSet, StandardStructureError, structure_fingerprint,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
STRUCTURE_FILE = "structures.json"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class StandardStructureManifest:
    schema_version: int
    standard_id: str
    revision: str
    corpus_fingerprint: str
    layout_fingerprint: str
    extractor_version: str
    structure_extractor_version: str
    table_count: int
    bitfield_count: int
    continuation_count: int
    warning_count: int
    auto_geometry_ok: int
    needs_review: int
    rejected_geometry: int
    structure_fingerprint: str
    created_at: str

    def to_dict(self) -> dict[str, object]:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "StandardStructureManifest":
        try:
            return cls(**{name: raw[name] for name in cls.__dataclass_fields__})
        except (KeyError, TypeError) as exc:
            raise StandardStructureError(
                f"malformed structure manifest: {exc}") from exc


def build_manifest(structures: StandardStructureSet, *, standard_id: str,
                   revision: str, corpus_fingerprint: str,
                   layout_fingerprint: str, extractor_version: str,
                   created_at: str | None = None) -> StandardStructureManifest:
    """Summarise a structure set, pinning what it was derived from."""

    # The three fingerprints are the point of the manifest: they say which
    # corpus and layout these structures describe, so a later read can tell
    # whether they still describe the current ones.

    statuses = [table.geometry_status for table in structures.tables]

    return StandardStructureManifest(
        schema_version=STRUCTURE_SCHEMA_VERSION,
        standard_id=standard_id, revision=revision,
        corpus_fingerprint=corpus_fingerprint,
        layout_fingerprint=layout_fingerprint,
        extractor_version=extractor_version,
        structure_extractor_version=STRUCTURE_EXTRACTOR_VERSION,
        table_count=len(structures.tables),
        bitfield_count=len(structures.bitfields),
        continuation_count=len(structures.continuations),
        warning_count=sum(len(table.warnings) for table in structures.tables),
        auto_geometry_ok=statuses.count(GeometryStatus.AUTO_GEOMETRY_OK),
        needs_review=statuses.count(GeometryStatus.NEEDS_REVIEW),
        rejected_geometry=statuses.count(GeometryStatus.REJECTED_GEOMETRY),
        structure_fingerprint=structure_fingerprint(structures),
        created_at=(created_at or
                    datetime.now(timezone.utc).isoformat(timespec="seconds")))


class StandardStructureStore:
    """Reads and writes one revision's derived structures, fail-closed."""

    def __init__(self, store: StandardStore) -> None:
        self.store = store

    def directory(self, standard_id: str, revision: str, *,
                  create: bool = False) -> Path:
        """The structures directory for one revision, refusing symlink escapes."""

        base = self.store.revision_dir(standard_id, revision, create=create)
        path = base / "structures"

        if path.exists() and path.is_symlink():
            raise StandardStructureError("structure path uses a symlink")

        if create:
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(path, 0o700)

        if path.exists() and path.resolve().parent != base.resolve():
            raise StandardStructureError("structure path escapes its revision")

        return path

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        if path.exists() and path.is_symlink():
            raise StandardStructureError("refusing to replace a symlink")

        handle, temporary = tempfile.mkstemp(prefix=".structure-", dir=path.parent)

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

    def save(self, manifest: StandardStructureManifest,
             structures: StandardStructureSet) -> Path:
        """Write a structure set, but only if its manifest is true of it.

        Three things are established before anything is written: the
        fingerprints are well formed, the manifest really describes THESE
        structures, and they were derived from the corpus now active.
        """

        for value, name in ((manifest.corpus_fingerprint, "corpus fingerprint"),
                            (manifest.layout_fingerprint, "layout fingerprint"),
                            (manifest.structure_fingerprint, "structure fingerprint")):
            if not _SHA256.fullmatch(value):
                raise StandardStructureError(f"{name} must be a lowercase SHA-256")

        if structure_fingerprint(structures) != manifest.structure_fingerprint:
            raise StandardStructureError("manifest does not describe these structures")

        active = self.store.verify_corpus(manifest.standard_id, manifest.revision)

        if active.corpus_manifest_sha256 != manifest.corpus_fingerprint:
            raise StandardStructureError(
                "structures were built against a different canonical corpus")

        directory = self.directory(manifest.standard_id, manifest.revision,
                                   create=True)
        self._atomic_write(directory / STRUCTURE_FILE,
                           canonical_json(structures.to_dict()))

        # The manifest is the commit marker and is always written last.

        self._atomic_write(directory / MANIFEST_FILE,
                           canonical_json(manifest.to_dict()))

        return directory

    def load_manifest(self, standard_id: str, revision: str,
                      ) -> StandardStructureManifest:
        path = self.directory(standard_id, revision) / MANIFEST_FILE

        if path.is_symlink():
            raise StandardStructureError("structure manifest uses a symlink")

        try:
            raw = json.loads(path.read_text("utf-8"))
        except FileNotFoundError as exc:
            raise StandardStructureError("no structures for this revision") from exc
        except ValueError as exc:
            raise StandardStructureError(f"structure manifest is unreadable: {exc}") from exc

        return StandardStructureManifest.from_dict(raw)

    def load(self, standard_id: str, revision: str,
             ) -> tuple[StandardStructureManifest, Mapping[str, object]]:
        """Load structures, refusing any that no longer match the corpus."""

        manifest = self.load_manifest(standard_id, revision)
        directory = self.directory(standard_id, revision)
        path = directory / STRUCTURE_FILE

        if path.is_symlink():
            raise StandardStructureError("structure file uses a symlink")

        try:
            structures = json.loads(path.read_text("utf-8"))
        except (FileNotFoundError, ValueError) as exc:
            raise StandardStructureError(f"structures are unreadable: {exc}") from exc

        # Both inputs are re-checked, because either changing invalidates the
        # geometry: the corpus is what was read, the layout is how it was read.

        active = self.store.verify_corpus(standard_id, revision)

        if manifest.corpus_fingerprint != active.corpus_manifest_sha256:
            raise StandardStructureError(
                "structures are stale: the canonical corpus has changed")

        layout = self.store.revision_dir(standard_id, revision) / "layout.json"

        if layout.is_file() and not layout.is_symlink():
            current = json.loads(layout.read_text("utf-8")).get("layout_fingerprint")

            if current != manifest.layout_fingerprint:
                raise StandardStructureError(
                    "structures are stale: the layout artifact has changed")

        if not isinstance(structures, Mapping) or "tables" not in structures:
            raise StandardStructureError("structure file is malformed")

        return manifest, structures

    def state(self, standard_id: str, revision: str) -> str:
        """READY, or the reason the structures cannot be used."""

        # A full load is the only honest test: every staleness check lives
        # there, and a cheaper probe would report READY on stale geometry.

        try:
            self.load(standard_id, revision)
        except (StandardStructureError, StandardStoreError, FileNotFoundError) as exc:
            return f"UNAVAILABLE ({exc})"

        return "READY"
