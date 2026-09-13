"""Structured access to SPEAR's human-readable Markdown memories.

Markdown remains the source of truth.  An optional JSON sidecar stores metadata
that Markdown cannot represent without making the file unpleasant to edit by
hand.  Deleting the sidecar loses only enrichment (tags, supersession, and
validation metadata), never the memories themselves.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
_LINE_RE = re.compile(r"^\s*-\s*(?:\[(\d{4}-\d{2}-\d{2})\]\s*)?(.+?)\s*$")
_TERM_RE = re.compile(r"[\w./+-]{2,}", re.UNICODE)


class MemoryStoreError(ValueError):
    pass


class MemoryScope(StrEnum):
    USER = "user"
    PROJECT = "project"
    CORPUS = "corpus"
    PROCEDURE = "task_derived_procedure"


class MemorySource(StrEnum):
    IMPORTED = "markdown_import"
    USER = "user"
    AUTHORIZED_TOOL = "authorized_tool"


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    content: str
    scope: MemoryScope
    source: MemorySource
    provenance: str
    created_at: str
    updated_at: str
    confidence: float = 1.0
    tags: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()
    superseded_by: str | None = None
    last_validated: str | None = None
    active: bool = True

    def __post_init__(self) -> None:
        if not self.memory_id.startswith("mem_") or not self.content.strip():
            raise MemoryStoreError("malformed memory record")

        if not 0.0 <= self.confidence <= 1.0:
            raise MemoryStoreError("memory confidence must be between zero and one")

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id, "content": self.content,
            "scope": self.scope.value, "source": self.source.value,
            "provenance": self.provenance, "created_at": self.created_at,
            "updated_at": self.updated_at, "confidence": self.confidence,
            "tags": list(self.tags), "supersedes": list(self.supersedes),
            "superseded_by": self.superseded_by,
            "last_validated": self.last_validated, "active": self.active,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "MemoryRecord":
        try:
            return cls(
                str(raw["memory_id"]), str(raw["content"]),
                MemoryScope(str(raw["scope"])), MemorySource(str(raw["source"])),
                str(raw["provenance"]), str(raw["created_at"]),
                str(raw["updated_at"]), float(raw.get("confidence", 1.0)),
                _strings(raw.get("tags", ())), _strings(raw.get("supersedes", ())),
                _optional_string(raw.get("superseded_by")),
                _optional_string(raw.get("last_validated")),
                _boolean(raw.get("active", True), "active"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MemoryStoreError("malformed memory record") from exc


@dataclass(frozen=True)
class MemorySelection:
    records: tuple[MemoryRecord, ...]
    query: str
    total_active: int
    excluded_count: int

    def render(self) -> str:
        if not self.records:
            return ""

        lines = ["## Relevant memories (durable knowledge, not task truth)", ""]
        lines.extend(f"- [{item.created_at[:10]}] {item.content}" for item in self.records)

        return "\n\n" + "\n".join(lines)


class MarkdownMemoryStore:
    """Markdown-authoritative memory store with optional metadata enrichment."""

    def __init__(
        self, markdown_path: str | os.PathLike[str], *,
        default_scope: MemoryScope = MemoryScope.PROJECT,
        metadata_path: str | os.PathLike[str] | None = None,
        read_metadata: bool = True,
    ) -> None:
        self.path = Path(markdown_path)
        self.metadata_path = Path(metadata_path or f"{self.path}.metadata.json")
        self.default_scope = default_scope
        self.read_metadata = read_metadata

    def records(self) -> tuple[MemoryRecord, ...]:
        """Every memory in the Markdown file, enriched by the sidecar where present.

        The Markdown decides what exists; the sidecar only adds what Markdown
        cannot express. A memory with no sidecar entry is complete, just plain.
        """

        metadata = self._load_metadata()
        enrichments = metadata.get("records", {})

        if not isinstance(enrichments, Mapping):
            raise MemoryStoreError("memory metadata records must be an object")

        records: list[MemoryRecord] = []
        occurrences: dict[tuple[str, str], int] = {}

        if not self.path.exists():
            return ()

        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise MemoryStoreError(f"cannot read memory Markdown: {exc}") from exc

        for line in lines:
            match = _LINE_RE.match(line)

            if not match:
                continue

            date, content = match.groups()
            content = content.strip()
            created = f"{date}T00:00:00+00:00" if date else "1970-01-01T00:00:00+00:00"

            # The id is derived from the line's content and date, so it is
            # stable across edits elsewhere in the file. The occurrence counter
            # keeps two identical lines on the same day distinguishable.

            key = (created, content)
            occurrence = occurrences.get(key, 0)
            occurrences[key] = occurrence + 1
            memory_id = _memory_id(self.default_scope, created, content, occurrence)
            raw = enrichments.get(memory_id, {})

            if raw and not isinstance(raw, Mapping):
                raise MemoryStoreError(f"metadata for {memory_id} must be an object")

            try:
                records.append(MemoryRecord(
                    memory_id, content,
                    MemoryScope(str(raw.get("scope", self.default_scope.value))),
                    MemorySource(str(raw.get("source", MemorySource.IMPORTED.value))),
                    str(raw.get("provenance", str(self.path))), created,
                    str(raw.get("updated_at", created)),
                    float(raw.get("confidence", 1.0)), _strings(raw.get("tags", ())),
                    _strings(raw.get("supersedes", ())),
                    _optional_string(raw.get("superseded_by")),
                    _optional_string(raw.get("last_validated")),
                    _boolean(raw.get("active", True), "active"),
                ))
            except (TypeError, ValueError) as exc:
                raise MemoryStoreError(f"malformed metadata for {memory_id}") from exc

        return tuple(records)

    def add(
        self, content: str, *, scope: MemoryScope | None = None,
        source: MemorySource = MemorySource.USER, provenance: str = "explicit",
        confidence: float = 1.0, tags: Sequence[str] = (),
        supersedes: Sequence[str] = (), now: datetime | None = None,
    ) -> MemoryRecord:
        """Append one memory to the Markdown, and record its metadata beside it."""

        # One line, because that is what the Markdown format is: a bullet list
        # a person can read and edit by hand.

        content = content.strip()

        if not content or "\n" in content:
            raise MemoryStoreError("memory content must be one non-empty line")

        instant = (now or datetime.now(UTC)).astimezone(UTC)
        created = instant.isoformat()
        date = instant.date().isoformat()
        scope = scope or self.default_scope
        existing = self.records()
        occurrence = sum(1 for item in existing
                         if item.created_at[:10] == date and item.content == content)
        memory_id = _memory_id(scope, f"{date}T00:00:00+00:00", content, occurrence)
        record = MemoryRecord(
            memory_id, content, scope, source, provenance,
            f"{date}T00:00:00+00:00", created, confidence,
            tuple(dict.fromkeys(tag.strip() for tag in tags if tag.strip())),
            tuple(dict.fromkeys(supersedes)), last_validated=created,
        )

        # Checked before anything is written: superseding a memory that does
        # not exist would leave the sidecar pointing at nothing.

        known = {item.memory_id: item for item in existing}
        missing = [item for item in record.supersedes if item not in known]

        if missing:
            raise MemoryStoreError(f"unknown superseded memory: {missing[0]}")

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(f"- [{date}] {content}\n")

        metadata = self._load_metadata()
        entries = dict(metadata.get("records", {}))
        entries[memory_id] = self._enrichment(record)

        # Superseded memories stay in the Markdown -- they are still part of
        # the record -- but are marked inactive so select() stops offering them.

        for old_id in record.supersedes:
            old = known[old_id]
            entries[old_id] = self._enrichment(replace(
                old, active=False, superseded_by=memory_id, updated_at=created,
            ))

        self._write_metadata({"schema_version": SCHEMA_VERSION, "records": entries})

        return record

    def select(
        self, query: str = "", *, scopes: Iterable[MemoryScope] | None = None,
        limit: int = 12,
    ) -> MemorySelection:
        """The memories worth injecting for this query, most relevant first."""

        if limit < 1:
            raise MemoryStoreError("memory selection limit must be positive")

        allowed = set(scopes) if scopes is not None else None
        active = [item for item in self.records()
                  if item.active and item.superseded_by is None
                  and (allowed is None or item.scope in allowed)]

        terms = _terms(query)
        scored = []

        for position, item in enumerate(active):
            haystack = _terms(" ".join((item.content, *item.tags)))
            overlap = len(terms & haystack)

            # With a query, only matching memories are offered; without one,
            # everything is eligible and file order decides.

            if terms and overlap == 0:
                continue

            # Term overlap dominates, confidence breaks ties among equal
            # matches, and position gives later memories a slight edge.

            score = overlap * 100 + item.confidence * 10 + position / max(1, len(active))
            scored.append((score, position, item))

        scored.sort(key=lambda value: (-value[0], -value[1], value[2].memory_id))
        selected = tuple(item for _, _, item in scored[:limit])

        return MemorySelection(selected, query, len(active), len(active) - len(selected))

    def count(self) -> int:
        return len(self.records())

    @staticmethod
    def _enrichment(record: MemoryRecord) -> dict[str, object]:
        """The sidecar half of a record: everything the Markdown cannot say."""

        # These three come from the Markdown line itself, so storing them again
        # would create a second source of truth that could disagree with it.

        value = record.to_dict()
        value.pop("memory_id")
        value.pop("content")
        value.pop("created_at")

        return value

    def _load_metadata(self) -> dict[str, object]:
        if not self.read_metadata:
            return {"schema_version": SCHEMA_VERSION, "records": {}}

        if not self.metadata_path.exists():
            return {"schema_version": SCHEMA_VERSION, "records": {}}

        try:
            raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MemoryStoreError("malformed memory metadata") from exc

        if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
            raise MemoryStoreError("unsupported memory metadata schema")

        return raw

    def _write_metadata(self, value: Mapping[str, object]) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.metadata_path.name}.", dir=self.metadata_path.parent,
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())

            os.replace(temporary, self.metadata_path)
        except Exception:
            # The Markdown has already been appended to at this point, so a
            # failed sidecar write loses enrichment, never the memory itself.

            try:
                os.unlink(temporary)
            except OSError:
                pass

            raise


def _memory_id(scope: MemoryScope, created: str, content: str, occurrence: int) -> str:
    payload = f"{scope.value}\0{created}\0{content}\0{occurrence}".encode("utf-8")
    return "mem_" + hashlib.sha256(payload).hexdigest()[:24]


def _terms(value: str) -> set[str]:
    return {item.casefold() for item in _TERM_RE.findall(value)}


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(x, str) for x in value):
        raise MemoryStoreError("memory metadata sequence must contain strings")

    return tuple(value)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise MemoryStoreError("memory metadata value must be a string or null")

    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise MemoryStoreError(f"memory metadata {name} must be boolean")

    return value
