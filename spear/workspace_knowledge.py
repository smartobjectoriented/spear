"""Workspace knowledge: what SPEAR has been told, or has verified, about a
workspace, kept across sessions.

It is descriptive and nothing more. A record says what is the case in one
workspace -- where a driver lives, which recipe builds an image, what the team
decided -- and where that came from. It is not a rule (rules.d says how to
behave), not configuration (projects.json is authoritative for the build and
the standard), not normative evidence (only the bound standard is) and not
conversation. It never grants a permission or a capability, and the current
request, rules, configuration and constraint packet all outrank it.

    record      one fact, decision or relation; a stable id, versions kept
    provenance  who or what it came from; MODEL_DERIVED and
                EXTERNAL_CAPABILITY can only ever propose
    verification  how it is known: told by the operator, read in a source
                file (bound to that file's content), or observed
    lifecycle   PROPOSED -> ACTIVE -> STALE or REVOKED; nothing is deleted
                unless the operator purges a workspace

A record bound to a file goes STALE when the file no longer reads as it did,
and is left out until the evidence is found again. Two active records that
disagree about one subject are a conflict, shown as one; the newer never
silently wins. Storage is SQLite under SPEAR's state root.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Callable, Iterable


class Kind(StrEnum):
    PROJECT_FACT = "PROJECT_FACT"
    ARCHITECTURE_FACT = "ARCHITECTURE_FACT"
    BUILD_FACT = "BUILD_FACT"
    COMPONENT_RELATION = "COMPONENT_RELATION"
    DECISION = "DECISION"
    COMMAND_KNOWLEDGE = "COMMAND_KNOWLEDGE"


class Provenance(StrEnum):
    USER_CONFIRMED = "USER_CONFIRMED"
    PROJECT_SOURCE = "PROJECT_SOURCE"
    TOOL_OBSERVATION = "TOOL_OBSERVATION"
    EXTERNAL_CAPABILITY = "EXTERNAL_CAPABILITY"
    IMPORTED = "IMPORTED"
    MODEL_DERIVED = "MODEL_DERIVED"


class Verification(StrEnum):
    UNVERIFIED = "UNVERIFIED"
    OBSERVED = "OBSERVED"
    SOURCE_VERIFIED = "SOURCE_VERIFIED"
    USER_CONFIRMED = "USER_CONFIRMED"


class Lifecycle(StrEnum):
    PROPOSED = "PROPOSED"
    ACTIVE = "ACTIVE"
    STALE = "STALE"
    REVOKED = "REVOKED"


#: Provenances that can never make a record active on their own word.
PROPOSE_ONLY = frozenset({Provenance.MODEL_DERIVED, Provenance.EXTERNAL_CAPABILITY,
                          Provenance.TOOL_OBSERVATION, Provenance.IMPORTED})

STATEMENT_CHARS = 600
SUBJECT_CHARS = 120
QUOTE_CHARS = 400


class KnowledgeError(ValueError):
    """A record cannot be stored as asked; the message says why."""


@dataclass(frozen=True)
class SourceRef:
    """What a record rests on, enough to check it again -- never the content."""

    kind: str                      # "file", "command" or "capability"
    locator: str                   # path relative to the workspace, command, provider/capability
    digest: str = ""               # of the file, or of the bounded result
    line: int | None = None
    quote: str = ""                # the evidence a file must still contain

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value not in ("", None)}


@dataclass(frozen=True)
class Record:
    record_id: str
    workspace_id: str
    kind: Kind
    subject: str
    statement: str
    provenance: Provenance
    verification: Verification
    lifecycle: Lifecycle
    tags: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    sources: tuple[SourceRef, ...] = ()
    version: int = 1
    created_at: str = ""
    updated_at: str = ""
    reason: str = ""               # why it is in its current state

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(
            [self.kind, self.subject, self.statement,
             [source.to_dict() for source in self.sources]], sort_keys=True).encode()
        ).hexdigest()[:16]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["sources"] = [source.to_dict() for source in self.sources]
        data["fingerprint"] = self.fingerprint
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Record":
        data = dict(data)
        data.pop("fingerprint", None)
        data["sources"] = tuple(SourceRef(**source) for source in data.get("sources") or ())
        data["tags"], data["paths"] = tuple(data.get("tags") or ()), tuple(data.get("paths") or ())
        data["kind"] = Kind(data["kind"])
        data["provenance"] = Provenance(data["provenance"])
        data["verification"] = Verification(data["verification"])
        data["lifecycle"] = Lifecycle(data["lifecycle"])
        return cls(**data)


# ── what a statement is ──────────────────────────────────────────────

# An instruction says what to DO. Knowledge says what IS. Read from the
# opening words, deterministically, and kept narrow: "the build uses make"
# is a fact, "always run make" is a rule.
_INSTRUCTION = re.compile(
    r"^\s*(?:please\s+)?(?:always|never|do\s+not|don'?t|must|make\s+sure|ensure|avoid|"
    r"remember\s+to|use|run|prefer|keep|stop|should|you\s+(?:must|should))\b", re.I)


def instruction(statement: str) -> bool:
    """Does this read as an instruction rather than a description?"""
    return bool(_INSTRUCTION.match(statement or ""))


# What holds for a moment, not for the workspace: a guess, the state of this
# session, how someone feels. Structural and narrow -- a durable fact that
# happens to contain "currently" in another sense is still accepted.
_TRANSIENT = re.compile(
    r"^\s*(?:i\s+(?:think|believe|guess|suspect)|maybe|perhaps|it\s+seems)\b"
    r"|\b(?:this|the\s+current)\s+(?:session|conversation|chat)\b"
    r"|\bturn\s+\d+\b|\bright\s+now\b|\bjust\s+(?:now|failed|happened)\b"
    r"|\b(?:last|previous)\s+(?:command|run|attempt)\b|\bfailed\s+once\b"
    r"|\bthe\s+user\s+(?:is|was|seems|feels)\b", re.I)


def transient(statement: str) -> bool:
    """Does this describe a moment rather than the workspace?"""
    return bool(_TRANSIENT.search(statement or ""))


def _words(text: str) -> str:
    return " ".join(re.findall(r"\w+", (text or "").lower()))


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w./-]+", " ", (text or "").lower()).split())


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: str) -> str | None:
    try:
        with open(path, "rb") as handle:
            return _digest(handle.read())
    except OSError:
        return None


def source_evidence(root: str, path: str, quote: str = "", line: int | None = None):
    """(a file SourceRef, "") when the file holds the evidence, else (None, why).

    The evidence is the quote, or the named line when no quote is given; it
    has to be in the file now, as written, for the fact to be source-verified.
    """
    relative = os.path.normpath(path)

    if relative.startswith("..") or os.path.isabs(relative):
        return None, f"{path} is not inside the workspace"

    full = os.path.join(root, relative)

    try:
        with open(full, "rb") as handle:
            data = handle.read()
    except OSError:
        return None, f"{path} cannot be read"

    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()

    if line is not None and not 1 <= line <= len(lines):
        return None, f"{path} has no line {line}"

    if not quote:
        if line is None:
            return None, "a source needs the line or the quoted text it rests on"

        quote = lines[line - 1].strip()

    quote = quote[:QUOTE_CHARS]

    if not quote or quote not in (lines[line - 1] if line is not None else text):
        return None, f"{path} does not contain the quoted evidence" + (
            f" on line {line}" if line is not None else "")

    return SourceRef("file", relative, _digest(data), line, quote), ""


# ── the store ────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    record_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    lifecycle TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS records_workspace ON records (workspace_id, lifecycle);
CREATE TABLE IF NOT EXISTS versions (
    record_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    changed_at TEXT NOT NULL,
    body TEXT NOT NULL,
    PRIMARY KEY (record_id, version)
);
"""


class KnowledgeStore:
    """Every workspace's records, in one SQLite file. Queries always name the
    workspace: there is no lookup that crosses one."""

    def __init__(self, path: str, *, audit: Callable[[str, dict], None] | None = None):
        self.path = path

        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        self._db = sqlite3.connect(path, timeout=10, isolation_level=None)
        self._db.executescript(_SCHEMA)
        self.audit = audit or (lambda kind, metadata: None)

    def close(self):
        self._db.close()

    # ── writing ─────────────────────────────────────────────────────

    def _write(self, record: Record, reason: str) -> Record:
        record = replace(record, updated_at=_now(), reason=reason)
        body = json.dumps(record.to_dict(), sort_keys=True)

        with self._db:
            self._db.execute("BEGIN")
            self._db.execute(
                "INSERT INTO records (record_id, workspace_id, lifecycle, body) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(record_id) DO UPDATE SET "
                "lifecycle = excluded.lifecycle, body = excluded.body",
                (record.record_id, record.workspace_id, record.lifecycle, body))
            self._db.execute("INSERT OR REPLACE INTO versions VALUES (?, ?, ?, ?)",
                             (record.record_id, record.version, record.updated_at, body))

        return record

    def _event(self, kind: str, record: Record, reason: str = ""):
        self.audit(kind, {"workspace": record.workspace_id, "record": record.record_id,
                          "kind": record.kind, "provenance": record.provenance,
                          "verification": record.verification,
                          "status": record.lifecycle, "version": record.version,
                          "reason": reason})

    def add(self, workspace_id: str, *, kind, subject: str, statement: str,
            provenance, verification=None, tags=(), paths=(), sources=()) -> Record:
        """Store one record, in the state its provenance allows."""
        kind, provenance = Kind(kind), Provenance(provenance)
        subject, statement = (subject or "").strip(), " ".join((statement or "").split())

        if not workspace_id:
            raise KnowledgeError("knowledge needs an exact workspace")

        if not statement:
            raise KnowledgeError("a record needs a statement")

        if len(statement) > STATEMENT_CHARS or len(subject) > SUBJECT_CHARS:
            raise KnowledgeError(f"keep a record to one fact: at most {STATEMENT_CHARS} "
                                 f"characters, and a subject of at most {SUBJECT_CHARS}")

        if instruction(statement):
            raise KnowledgeError(
                "this reads as an instruction, not a fact about the workspace: an "
                "instruction belongs in a rule (rules.d, or /recall for one that holds "
                "everywhere), where it has a rule's authority")

        if transient(statement):
            raise KnowledgeError("this describes a moment, not the workspace: knowledge "
                                 "keeps what stays true across sessions")

        sources = tuple(sources)
        same = self.duplicate(workspace_id, kind, subject or statement[:60], statement, sources)

        if same is not None:
            self.audit("knowledge_duplicate", {
                "workspace": workspace_id, "record": same.record_id, "kind": same.kind,
                "provenance": provenance, "status": same.lifecycle,
                "reason": "the same fact is already recorded"})
            return same

        if provenance == Provenance.USER_CONFIRMED:
            verification, lifecycle = Verification.USER_CONFIRMED, Lifecycle.ACTIVE
        elif provenance == Provenance.PROJECT_SOURCE:
            if not any(source.kind == "file" and source.digest for source in sources):
                raise KnowledgeError("a source-verified record needs the file it rests on")

            verification, lifecycle = Verification.SOURCE_VERIFIED, Lifecycle.ACTIVE
        else:
            verification = Verification(verification or (
                Verification.OBSERVED if provenance == Provenance.TOOL_OBSERVATION
                else Verification.UNVERIFIED))
            lifecycle = Lifecycle.PROPOSED

        record = Record(
            record_id="k-" + secrets.token_hex(6), workspace_id=workspace_id, kind=kind,
            subject=subject or statement[:60], statement=statement, provenance=provenance,
            verification=verification, lifecycle=lifecycle,
            tags=tuple(sorted({normalise(tag) for tag in tags if normalise(tag)})),
            paths=tuple(paths), sources=sources, created_at=_now())
        record = self._write(record, "created")
        self._event("knowledge_proposed" if lifecycle == Lifecycle.PROPOSED
                    else "knowledge_activated", record, "created")

        return record

    def duplicate(self, workspace_id, kind, subject, statement, sources=()):
        """An active or proposed record saying the same thing, or None: same
        kind, same subject, the same statement word for word, the same sources."""
        wanted = (Kind(kind), _words(subject), _words(statement),
                  sorted((source.kind, source.locator) for source in sources))

        for record in self.list(workspace_id, (Lifecycle.ACTIVE, Lifecycle.PROPOSED)):
            if (record.kind, _words(record.subject), _words(record.statement),
                    sorted((source.kind, source.locator) for source in record.sources)) \
                    == wanted:
                return record

        return None

    def find_source(self, workspace_id: str, kind: str, locator: str) -> Record | None:
        """The record, in any state, that rests on this source."""
        for record in self.list(workspace_id):
            if any(source.kind == kind and source.locator == locator
                   for source in record.sources):
                return record

        return None

    def accept(self, workspace_id: str, record_id: str) -> Record:
        """The operator confirms a proposed record; it becomes theirs."""
        record = self._own(workspace_id, record_id)

        if record.lifecycle != Lifecycle.PROPOSED:
            raise KnowledgeError(f"{record_id} is {record.lifecycle}, not PROPOSED")

        record = self._write(replace(record, lifecycle=Lifecycle.ACTIVE,
                                     verification=Verification.USER_CONFIRMED,
                                     version=record.version + 1), "accepted by the operator")
        self._event("knowledge_activated", record, "accepted by the operator")

        return record

    def amend(self, workspace_id: str, record_id: str, statement: str) -> Record:
        """A corrected statement, as the record's next version; the operator's
        word, so it is user-confirmed and any source binding is dropped."""
        record = self._own(workspace_id, record_id)

        if record.lifecycle == Lifecycle.REVOKED:
            raise KnowledgeError(f"{record_id} is revoked")

        statement = " ".join((statement or "").split())

        if not statement or instruction(statement) or len(statement) > STATEMENT_CHARS:
            raise KnowledgeError("the amended statement must be one fact, not an instruction")

        record = self._write(replace(record, statement=statement, lifecycle=Lifecycle.ACTIVE,
                                     verification=Verification.USER_CONFIRMED,
                                     provenance=Provenance.USER_CONFIRMED, sources=(),
                                     version=record.version + 1), "amended by the operator")
        self._event("knowledge_activated", record, "amended by the operator")

        return record

    def revoke(self, workspace_id: str, record_id: str, reason: str = "") -> Record:
        record = self._own(workspace_id, record_id)
        record = self._write(replace(record, lifecycle=Lifecycle.REVOKED,
                                     version=record.version + 1),
                             reason or "revoked by the operator")
        self._event("knowledge_revoked", record, record.reason)

        return record

    def purge(self, workspace_id: str) -> int:
        """Physically delete one workspace's records and their history."""
        with self._db:
            self._db.execute("BEGIN")
            ids = [row[0] for row in self._db.execute(
                "SELECT record_id FROM records WHERE workspace_id = ?", (workspace_id,))]
            self._db.executemany("DELETE FROM versions WHERE record_id = ?",
                                 [(item,) for item in ids])
            self._db.execute("DELETE FROM records WHERE workspace_id = ?", (workspace_id,))

        return len(ids)

    # ── reading ─────────────────────────────────────────────────────

    def get(self, workspace_id: str, record_id: str) -> Record | None:
        row = self._db.execute("SELECT body FROM records WHERE record_id = ? AND "
                               "workspace_id = ?", (record_id, workspace_id)).fetchone()
        return Record.from_dict(json.loads(row[0])) if row else None

    def _own(self, workspace_id, record_id) -> Record:
        record = self.get(workspace_id, record_id)

        if record is None:
            raise KnowledgeError(f"{record_id}: no such record in this workspace")

        return record

    def list(self, workspace_id: str, lifecycles: Iterable = ()) -> list[Record]:
        wanted = [str(item) for item in lifecycles]
        query = "SELECT body FROM records WHERE workspace_id = ?"
        query += f" AND lifecycle IN ({','.join('?' * len(wanted))})" if wanted else ""

        return sorted((Record.from_dict(json.loads(row[0])) for row in self._db.execute(
            query, (workspace_id, *wanted))), key=lambda record: record.created_at)

    def history(self, workspace_id: str, record_id: str) -> list[Record]:
        self._own(workspace_id, record_id)

        return [Record.from_dict(json.loads(row[0])) for row in self._db.execute(
            "SELECT body FROM versions WHERE record_id = ? ORDER BY version", (record_id,))]

    def export(self, workspace_id: str) -> dict:
        return {"workspace": workspace_id, "exported_at": _now(),
                "records": [{**record.to_dict(),
                             "history": [item.to_dict() for item in
                                         self.history(workspace_id, record.record_id)]}
                            for record in self.list(workspace_id)]}

    # ── keeping it true ─────────────────────────────────────────────

    def validate(self, workspace_id: str, root: str) -> list[Record]:
        """Mark STALE every active record whose source file no longer reads as
        it did; return them. A file's digest is computed once per call."""
        digests, stale = {}, []

        for record in self.list(workspace_id, (Lifecycle.ACTIVE,)):
            changed = []

            for source in record.sources:
                if source.kind != "file":
                    continue

                if source.locator not in digests:
                    digests[source.locator] = file_digest(os.path.join(root, source.locator))

                if digests[source.locator] != source.digest:
                    changed.append(source.locator)

            if changed:
                why = (f"{', '.join(changed)} changed since the fact was verified"
                       if all(digests[path] for path in changed)
                       else f"{', '.join(changed)} is gone")
                record = self._write(replace(record, lifecycle=Lifecycle.STALE,
                                             version=record.version + 1), why)
                self._event("knowledge_stale", record, why)
                stale.append(record)

        return stale

    def revalidate(self, workspace_id: str, root: str) -> list[Record]:
        """Make active again every stale record whose evidence is found again,
        bound to the file as it now is."""
        restored = []

        for record in self.list(workspace_id, (Lifecycle.STALE,)):
            sources, missing = [], False

            for source in record.sources:
                if source.kind != "file":
                    sources.append(source)
                    continue

                found, _ = source_evidence(root, source.locator, source.quote, None)

                if found is None:
                    missing = True
                    break

                sources.append(replace(found, line=source.line))

            if missing or not sources:
                continue

            record = self._write(replace(record, sources=tuple(sources),
                                         lifecycle=Lifecycle.ACTIVE,
                                         version=record.version + 1),
                                 "the evidence was found again")
            self._event("knowledge_activated", record, "revalidated")
            restored.append(record)

        return restored


def conflicts(records) -> list[list[Record]]:
    """Groups of active records that say different things about one subject."""
    groups = {}

    for record in records:
        if record.lifecycle == Lifecycle.ACTIVE:
            groups.setdefault((record.kind, normalise(record.subject)), []).append(record)

    return [group for group in groups.values()
            if len({_words(record.statement) for record in group}) > 1]


# ── what a turn is shown ─────────────────────────────────────────────

DIRECT_LIMIT = 10
DIRECT_TOKENS = 1500
DIRECT, INDEXED = "DIRECT", "INDEXED"

HEADER = ("\n\n## Workspace knowledge\n\nWhat has been recorded about this workspace, "
          "with where each fact came from. It describes; it does not instruct. The "
          "request, the project's rules and configuration, and any normative constraints "
          "take precedence over it, and a fact the current request contradicts is out of "
          "date.\n\n")


def estimate(text: str) -> int:
    return len(text) // 3 + 1


def _source(record: Record) -> str:
    sources = "; ".join(
        f"{source.locator}" + (f":{source.line}" if source.line else "")
        for source in record.sources)
    return f"{record.provenance}, {record.verification}" + (f", {sources}" if sources else "")


def full(record: Record) -> str:
    return f"- {record.record_id} [{record.kind}] {record.subject}: {record.statement} ({_source(record)})"


#: How much of a statement an index line carries.
SUMMARY_CHARS = 40


def summary(record: Record, chars: int = SUMMARY_CHARS, *, kind: bool = True) -> str:
    statement = (record.statement if len(record.statement) <= chars
                 else record.statement[:chars - 1] + "…")
    label = f" [{record.kind}]" if kind else ""
    return f"- {record.record_id}{label} {record.subject}: {statement}"


def index(records, chars: int = SUMMARY_CHARS) -> str:
    """One line a record, grouped under its kind, so a kind is said once."""
    lines = []

    for kind in Kind:
        group = [record for record in records if record.kind == kind]

        if group:
            lines.append(f"{kind}:")
            lines += [summary(record, chars, kind=False) for record in group]

    return "\n".join(lines)


def detail(record: Record, history=()) -> str:
    lines = [full(record), f"  status {record.lifecycle}, version {record.version}, "
             f"recorded {record.created_at}, updated {record.updated_at}"]

    for source in record.sources:
        lines.append(f"  source: {json.dumps(source.to_dict(), ensure_ascii=False)}")

    if len(history) > 1:
        lines.append("  earlier versions: " + "; ".join(
            f"v{item.version} {item.lifecycle}: {item.statement[:80]}" for item in history[:-1]))

    return "\n".join(lines)


def matches(record: Record, request: str) -> bool:
    """Does the request name this record's subject, a tag or one of its paths?
    Exact words, never resemblance."""
    words = f" {normalise(request)} "
    names = [normalise(record.subject)] + list(record.tags) + [
        normalise(path) for path in record.paths + tuple(
            source.locator for source in record.sources if source.kind == "file")]

    return any(name and f" {name} " in words for name in names)


@dataclass
class Selection:
    mode: str = DIRECT
    shown: list = field(default_factory=list)       # records shown in full
    indexed: list = field(default_factory=list)     # records shown as one line
    conflicts: list = field(default_factory=list)
    text: str = ""


def select(records, request: str) -> Selection:
    """What a turn is shown of a workspace's active records: all of them in
    full when they are few, else those the request names in full and the rest
    as an index. A conflict is shown as one, both sides with their sources."""
    active = [record for record in records if record.lifecycle == Lifecycle.ACTIVE]
    result = Selection(conflicts=conflicts(active))

    if not active:
        return result

    clashing = {record.record_id for group in result.conflicts for record in group}
    whole = "".join(full(record) + "\n" for record in active)

    if len(active) <= DIRECT_LIMIT and estimate(whole) <= DIRECT_TOKENS:
        result.mode, result.shown = DIRECT, active
    else:
        result.mode = INDEXED
        result.shown = [record for record in active
                        if matches(record, request) or record.record_id in clashing][:8]
        named = {record.record_id for record in result.shown}
        result.indexed = [record for record in active if record.record_id not in named]

    body = [full(record) for record in result.shown if record.record_id not in clashing]

    for group in result.conflicts:
        body.append(f"- KNOWLEDGE CONFLICT about {group[0].subject}: these records disagree "
                    f"and neither is preferred --")
        body += ["  " + full(record) for record in group]

    if result.indexed:
        body.append("\nOther records, by id (`spear-knowledge show <id>` in the terminal "
                    "prints one in full):")
        body.append(index(result.indexed))

    result.text = HEADER + "\n".join(body) + "\n"

    return result


# ── the turn's read-only door ────────────────────────────────────────

COMMAND = "spear-knowledge"


@dataclass
class Door:
    """`spear-knowledge list | show <id> | propose '<json>'` for one turn.

    Read-only on what exists. A proposal is stored PROPOSED and MODEL_DERIVED:
    never shown to a turn, never acted on, until the operator accepts it.
    """

    store: KnowledgeStore
    workspace_id: str

    def run(self, command: str) -> tuple[int, str]:
        import host_commands

        words, problem = host_commands.words(command, COMMAND)

        if problem:
            return 2, problem

        verb, rest = (words[1] if len(words) > 1 else ""), words[2:]

        if verb == "list" and not rest:
            records = self.store.list(self.workspace_id, (Lifecycle.ACTIVE,))
            return 0, "\n".join(summary(record) for record in records) or "(no records)"

        if verb == "show" and len(rest) == 1:
            record = self.store.get(self.workspace_id, rest[0])

            if record is None or record.lifecycle not in (Lifecycle.ACTIVE, Lifecycle.STALE):
                return 1, f"{rest[0]}: no such record in this workspace."

            return 0, detail(record, self.store.history(self.workspace_id, rest[0]))

        if verb == "propose" and len(rest) == 1:
            try:
                data = json.loads(rest[0])
                record = self.store.add(
                    self.workspace_id, kind=data.get("kind", Kind.PROJECT_FACT),
                    subject=str(data.get("subject") or ""),
                    statement=str(data.get("statement") or ""),
                    provenance=Provenance.MODEL_DERIVED,
                    tags=tuple(str(tag) for tag in data.get("tags") or ()))
            except (ValueError, KnowledgeError, AttributeError) as exc:
                return 2, f"{COMMAND} propose: {exc}"

            return 0, (f"{record.record_id} proposed. It is not used until the operator "
                       f"accepts it.")

        return 2, (f"usage: {COMMAND} list | {COMMAND} show <id> | "
                   f"{COMMAND} propose '<{{\"kind\", \"subject\", \"statement\"}} as JSON>'")


class RealStateInTest(RuntimeError):
    """A test reached for the operator's own knowledge store."""


def default_path() -> str:
    """SPEAR_KNOWLEDGE_DB, else knowledge.sqlite3 under the state root. A test
    run that names neither it nor SPEAR_STATE_DIR is refused: the default is a
    person's own knowledge, and no test may open it."""
    import state_paths

    if os.environ.get("SPEAR_KNOWLEDGE_DB"):
        return os.environ["SPEAR_KNOWLEDGE_DB"]

    if state_paths.under_test() and not os.environ.get("SPEAR_STATE_DIR"):
        raise RealStateInTest("a test must give the knowledge store its own path "
                              "(SPEAR_KNOWLEDGE_DB or SPEAR_STATE_DIR)")

    return str(state_paths.state_dir() / "knowledge.sqlite3")
