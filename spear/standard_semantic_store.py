"""Private stores for human approvals and the semantic definitions they permit.

Approvals live beside the geometry they judge; definitions live in their own
derived store. Both pin every upstream fingerprint, so an approval recorded
against other geometry, or a definition built on a corpus that has since moved,
fails closed rather than being read as current.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from standard_approval_history import (
    APPROVAL_FILE_SCHEMA_VERSION, ApprovalEventType, StandardApprovalHistory,
    event_for,
)
from standard_schema import canonical_json
from standard_store import StandardStore, StandardStoreError
from standard_structure_store import StandardStructureStore
from standard_value_pair import value_local_pairs
from standard_word_association import field_candidates
from standard_prose_range import StandardReviewedLink
from standard_semantic import (
    APPROVAL_SCHEMA_VERSION, ASSISTANT_IDENTITY, SEMANTIC_MODEL_VERSION,
    SEMANTIC_SCHEMA_VERSION, SpanRole, StandardBitfieldApproval,
    StandardSemanticError, StandardSemanticSet, review_evidence_fingerprint,
    semantic_fingerprint,
)

APPROVALS_FILE = "approvals.json"
SEMANTIC_DIRECTORY = "semantic-structures"
SEMANTIC_FILE = "definitions.json"
MANIFEST_FILE = "manifest.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def operator_identity(explicit: str | None = None) -> str:
    """Who is approving, taken from the local session -- never from a file."""
    import getpass

    for value in (explicit, os.environ.get("SPEAR_OPERATOR"),
                  os.environ.get("USER")):
        if value and value.strip():
            candidate = value.strip()
            break
    else:
        try:
            candidate = getpass.getuser()
        except Exception:  # pragma: no cover - exotic environments
            candidate = ""

    if not candidate:
        raise StandardSemanticError("no operator identity is configured")

    if ASSISTANT_IDENTITY.match(candidate):
        raise StandardSemanticError(
            "an assistant identity cannot approve a semantic promotion")

    return candidate


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    if path.exists() and path.is_symlink():
        raise StandardSemanticError("refusing to replace a symlink")

    handle, temporary = tempfile.mkstemp(prefix=".semantic-", dir=path.parent)

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


def _packed_slots(store, standard_id: str, revision: str, bitfield, table):
    """The normative packed-slot placement, as promotion will compute it.

    Approval counting, promotion and serving must agree about what a reviewer
    was shown, and this is the one place each of them asks.
    """
    from standard_semantic import packed_slots_for

    try:
        units = store.load_units(standard_id, revision)
    except (StandardStoreError, StandardSemanticError, FileNotFoundError):
        return {}

    return packed_slots_for(bitfield, table, units)


def _same_decision(previous, current) -> bool:
    """Whether two approvals say exactly the same thing about a candidate.

    The time of the decision is left out: two runs of one command differ only
    by when they ran, and that alone is not a second review.
    """

    def content(approval):
        return canonical_json({
            "verdict": approval.verdict,
            "span_roles": list(approval.span_roles),
            "reviewer": approval.reviewer,
            "structure_fingerprint": approval.structure_fingerprint,
            "review_evidence_fingerprint": approval.review_evidence_fingerprint,
            "reviewed_links": [item.to_dict()
                               for item in approval.reviewed_links],
            "packet_identity": approval.packet_identity,
            "notes": approval.notes,
        })

    return content(previous) == content(current)


class StandardApprovalStore:
    """Human decisions about bitfield candidates, kept with the geometry."""

    def __init__(self, store: StandardStore) -> None:
        self.store = store
        self.structures = StandardStructureStore(store)
        self.history = StandardApprovalHistory(store)

    def path(self, standard_id: str, revision: str) -> Path:
        return self.structures.directory(standard_id, revision) / APPROVALS_FILE

    def load(self, standard_id: str, revision: str,
             ) -> tuple[dict[str, StandardBitfieldApproval], list[dict[str, object]]]:
        """Return usable approvals and the ones refused, with the reason."""
        manifest, payload = self.structures.load(standard_id, revision)
        known = {item["bitfield_id"] for item in payload.get("bitfields", ())}
        target = self.path(standard_id, revision)

        if not target.is_file() or target.is_symlink():
            return {}, []

        try:
            raw = json.loads(target.read_text("utf-8"))
        except ValueError as exc:
            raise StandardSemanticError(f"approvals are unreadable: {exc}") from exc

        if not isinstance(raw, Mapping) or not isinstance(raw.get("rows"), list):
            raise StandardSemanticError("approval file is malformed")

        accepted: dict[str, StandardBitfieldApproval] = {}
        refused: list[dict[str, object]] = []

        for row in raw["rows"]:
            try:
                approval = StandardBitfieldApproval.from_dict(row)
            except StandardSemanticError as exc:
                refused.append({"candidate_id": row.get("candidate_id"),
                                "reason": "MALFORMED", "detail": str(exc)})
                continue

            if approval.candidate_id not in known:
                refused.append({"candidate_id": approval.candidate_id,
                                "reason": "UNKNOWN_CANDIDATE",
                                "detail": "no such bitfield candidate"})
                continue

            if approval.structure_fingerprint != manifest.structure_fingerprint:
                refused.append({"candidate_id": approval.candidate_id,
                                "reason": "STALE_APPROVAL",
                                "detail": "recorded against other geometry"})
                continue

            accepted[approval.candidate_id] = approval

        return accepted, refused

    def normative_links(self, standard_id: str, revision: str,
                        candidate_id: str) -> dict[str, object]:
        """The links this candidate offers right now, read from the live data."""
        from standard_commands import load_structure_set
        from standard_semantic import normative_links_for

        manifest, structures, _, layout = load_structure_set(
            self.store, standard_id, revision)
        candidate = next((item for item in structures.bitfields
                          if item.bitfield_id == candidate_id), None)

        if candidate is None:
            return {}

        table = next(item for item in structures.tables
                     if item.table_id == candidate.table_id)

        return normative_links_for(
            candidate, table, self.store.load_units(standard_id, revision),
            layout=layout)

    def approve(self, standard_id: str, revision: str, candidate_id: str, *,
                verdict: str, span_roles: Sequence[str],
                reviewer: str | None = None, packet_identity: str | None = None,
                notes: str | None = None,
                accept_links: Sequence[str] = (),
                decline_links: Sequence[str] = (),
                link_roles: Mapping[str, str] | None = None,
                ) -> StandardBitfieldApproval:
        """Record one human decision, stamped with the live fingerprint."""
        manifest, payload = self.structures.load(standard_id, revision)
        candidates = {item["bitfield_id"]: item
                      for item in payload.get("bitfields", ())}

        if candidate_id not in candidates:
            raise StandardSemanticError(f"unknown bitfield candidate {candidate_id}")

        bitfield = candidates[candidate_id]
        table = next((item for item in payload["tables"]
                      if item["table_id"] == bitfield["table_id"]), None)

        if table is None:
            raise StandardSemanticError(
                f"candidate {candidate_id} names a table that is not present")

        # One role per field the DIAGRAM shows -- including a half of a wider
        # value, which is the diagram's own label and is classified like any
        # other. A field positioned by a rule from outside the diagram is
        # reviewed as a link, not as a role. The pairing is computed exactly as
        # promotion and serving compute it, so the three cannot disagree.

        pairs, _ = value_local_pairs(bitfield, table)
        segments = {cell_id: segment for pair in pairs
                    for cell_id, segment in pair.by_cell().items()}
        originals = field_candidates(bitfield, table,
                                     value_local_cells=frozenset(segments))

        # A packed pair changes no candidate, only where its members sit, so
        # this is read purely to reproduce the evidence a reviewer will bind.

        packed = _packed_slots(self.store, standard_id, revision,
                               bitfield, table)

        if len(span_roles) != len(originals):
            raise StandardSemanticError(
                f"candidate has {len(originals)} diagram field candidates; "
                f"{len(span_roles)} were classified")

        available = self.normative_links(standard_id, revision, candidate_id)
        by_fingerprint = {item.link_fingerprint: item
                          for item in available.values()}
        decisions = {str(value): True for value in accept_links}

        for value in decline_links:
            if str(value) in decisions:
                raise StandardSemanticError(
                    f"link {value} is both accepted and declined")

            decisions[str(value)] = False

        unknown = set(decisions) - set(by_fingerprint)

        if unknown:
            raise StandardSemanticError(
                f"this candidate offers no normative link {sorted(unknown)[0]}")

        pending = set(by_fingerprint) - set(decisions)

        if pending:
            missing = sorted(pending)[0]
            raise StandardSemanticError(
                f"normative link {missing} positions "
                f"{by_fingerprint[missing].target_label!r} and must be accepted "
                "or declined before this candidate can be approved")

        roles = dict(link_roles or {})

        # The facts are read from the live link, never from the command line:
        # the reviewer names a link, the store records what that link says.

        reviewed = tuple(sorted(
            (StandardReviewedLink.of(by_fingerprint[key], accepted=value,
                                     role=roles.get(key, "FIELD"))
             for key, value in decisions.items()),
            key=lambda item: item.link_fingerprint))
        from standard_word_association import associate_words

        approval = StandardBitfieldApproval(
            candidate_id=candidate_id, verdict=verdict,
            reviewer=operator_identity(reviewer), reviewed_at=_now(),
            # Taken from the live store, never from anything supplied.
            structure_fingerprint=manifest.structure_fingerprint,
            span_roles=tuple(str(role) for role in span_roles),
            packet_identity=packet_identity, notes=notes,
            schema_version=APPROVAL_SCHEMA_VERSION, reviewed_links=reviewed,
            review_evidence_fingerprint=review_evidence_fingerprint(
                candidate_id, manifest.structure_fingerprint, table=table,
                originals=originals, roles=span_roles,
                associations=associate_words(table), reviewed_links=reviewed,
                value_segments=segments, packed_slots=packed))
        existing, _ = self.load(standard_id, revision)
        previous = existing.get(candidate_id)

        # A decision is recorded before it takes effect. Two files cannot be
        # swapped as one, so the order is chosen for which half-done state is
        # survivable: a decision logged but not yet active changes nothing and
        # loses nothing, while an active approval no decision records would be
        # exactly the hole this log exists to close.

        recorded = {item.event_id
                    for item in self.history.events(standard_id, revision)}
        pending = []
        supersedes = None

        if previous is not None:
            if _same_decision(previous, approval):
                # Nothing about what governs would change, so there is nothing
                # to record. Refusing keeps an up-arrow-and-enter out of the
                # log without ever refusing a decision that differs.

                raise StandardSemanticError(
                    f"this decision is already the active one for "
                    f"{candidate_id}"
                    + (f" ({previous.event_id})" if previous.event_id else "")
                    + "; change the verdict, roles, links or notes to record a "
                      "new review")

            # An approval taken before the log existed is carried in first, so
            # what it replaced is on the record before anything replaces it.
            # The copy is faithful: its reviewer, time and evidence are its own.

            if previous.event_id is not None:
                carried = previous.event_id
            else:
                copy = event_for(previous, standard_id=standard_id,
                                 revision=revision,
                                 event_type=ApprovalEventType
                                 .MIGRATED_INITIAL_APPROVAL)
                carried = copy.event_id

                # A migration may already have carried it in. Carrying is a
                # faithful copy, so the same decision always yields the same
                # event, and doing it twice must add nothing.

                if carried not in recorded:
                    pending.append(copy)

            supersedes = carried

        event = event_for(
            approval, standard_id=standard_id, revision=revision,
            event_type=(ApprovalEventType.REAPPROVAL if previous is not None
                        else ApprovalEventType.INITIAL_APPROVAL),
            supersedes=supersedes)

        if event.event_id in recorded:
            # Same candidate, verdict, roles, evidence, links and second. This
            # is one command run twice, not a person reviewing twice.

            raise StandardSemanticError(
                f"this exact decision is already recorded as {event.event_id}")

        pending.append(event)
        self.history.append(standard_id, revision, pending)

        approval = replace(approval, event_id=event.event_id,
                           supersedes=supersedes)
        existing[candidate_id] = approval
        _atomic_write(self.path(standard_id, revision), canonical_json({
            "schema_version": APPROVAL_FILE_SCHEMA_VERSION,
            "standard_id": standard_id, "revision": revision,
            "structure_fingerprint": manifest.structure_fingerprint,
            "updated_at": _now(),
            "rows": [existing[key].to_dict() for key in sorted(existing)],
        }))

        return approval


    def migrate_history(self, standard_id: str, revision: str, *,
                        dry_run: bool = True) -> list[dict[str, object]]:
        """Carry approvals taken before the history log into it, unchanged.

        Each becomes one initial event holding the decision exactly as it was
        recorded -- the same reviewer, the same time, the same evidence, and no
        evidence where there never was one. The active row then names its
        event, so the pointer from what governs to what was decided is written
        down rather than inferred.

        Returns what it would do. Nothing is written unless `dry_run` is off.
        """
        manifest, _ = self.structures.load(standard_id, revision)
        active, _ = self.load(standard_id, revision)

        # Reading the log validates it whole, so a corrupt or forked history
        # stops here. An active approval already naming an event that does not
        # describe it is the other way this can be unsafe, and carrying more
        # decisions in on top of that would only bury it.

        problems = self.history.verify(standard_id, revision, active)

        if problems:
            raise StandardSemanticError(
                "approval history is inconsistent with the active approvals: "
                + "; ".join(f"{item['candidate_id']}: {item['reason']}"
                            for item in problems))

        planned = self.history.migration_plan(standard_id, revision, active)
        report = [{"candidate_id": item.candidate_id, "event_id": item.event_id,
                   "event_type": item.event_type.value,
                   "verdict": item.verdict, "roles": len(item.span_roles),
                   "reviewer": item.reviewer, "approved_at": item.approved_at,
                   "review_evidence_fingerprint":
                       item.review_evidence_fingerprint,
                   "evidence_status": item.evidence_status.value}
                  for item in planned]

        if dry_run or not planned:
            return report

        self.history.append(standard_id, revision, planned)
        by_candidate = {item.candidate_id: item.event_id for item in planned}

        for candidate_id, event_id in by_candidate.items():
            active[candidate_id] = replace(active[candidate_id],
                                           event_id=event_id)

        _atomic_write(self.path(standard_id, revision), canonical_json({
            "schema_version": APPROVAL_FILE_SCHEMA_VERSION,
            "standard_id": standard_id, "revision": revision,
            "structure_fingerprint": manifest.structure_fingerprint,
            "updated_at": _now(),
            "rows": [active[key].to_dict() for key in sorted(active)],
        }))

        return report


@dataclass(frozen=True)
class StandardSemanticManifest:
    schema_version: int
    standard_id: str
    revision: str
    corpus_fingerprint: str
    layout_fingerprint: str
    structure_fingerprint: str
    semantic_model_version: str
    approved_candidate_count: int
    bitfield_definition_count: int
    field_definition_count: int
    packet_definition_count: int
    blocked_count: int
    semantic_fingerprint: str
    created_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "StandardSemanticManifest":
        try:
            return cls(**{name: raw[name] for name in cls.__dataclass_fields__})
        except (KeyError, TypeError) as exc:
            raise StandardSemanticError(f"malformed semantic manifest: {exc}") from exc


def build_semantic_manifest(semantics: StandardSemanticSet, *, standard_id: str,
                            revision: str, corpus_fingerprint: str,
                            layout_fingerprint: str, structure_fingerprint: str,
                            approved: int, created_at: str | None = None,
                            ) -> StandardSemanticManifest:
    return StandardSemanticManifest(
        schema_version=SEMANTIC_SCHEMA_VERSION, standard_id=standard_id,
        revision=revision, corpus_fingerprint=corpus_fingerprint,
        layout_fingerprint=layout_fingerprint,
        structure_fingerprint=structure_fingerprint,
        semantic_model_version=SEMANTIC_MODEL_VERSION,
        approved_candidate_count=approved,
        bitfield_definition_count=len(semantics.bitfields),
        field_definition_count=sum(len(item.fields) for item in semantics.bitfields),
        packet_definition_count=len(semantics.packets),
        blocked_count=len(semantics.blocked),
        semantic_fingerprint=semantic_fingerprint(semantics),
        created_at=created_at or _now())


class StandardSemanticStore:
    """The derived semantic definitions, private and fail-closed."""

    def __init__(self, store: StandardStore) -> None:
        self.store = store
        self.structures = StandardStructureStore(store)

    def directory(self, standard_id: str, revision: str, *,
                  create: bool = False) -> Path:
        base = self.store.revision_dir(standard_id, revision, create=create)
        path = base / SEMANTIC_DIRECTORY

        if path.exists() and path.is_symlink():
            raise StandardSemanticError("semantic path uses a symlink")

        if create:
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(path, 0o700)

        if path.exists() and path.resolve().parent != base.resolve():
            raise StandardSemanticError("semantic path escapes its revision")

        return path

    def save(self, manifest: StandardSemanticManifest,
             semantics: StandardSemanticSet) -> Path:
        for value, name in ((manifest.corpus_fingerprint, "corpus"),
                            (manifest.layout_fingerprint, "layout"),
                            (manifest.structure_fingerprint, "structure"),
                            (manifest.semantic_fingerprint, "semantic")):
            if not _SHA256.fullmatch(value):
                raise StandardSemanticError(
                    f"{name} fingerprint must be a lowercase SHA-256")

        if semantic_fingerprint(semantics) != manifest.semantic_fingerprint:
            raise StandardSemanticError("manifest does not describe these definitions")

        structures, _ = self.structures.load(manifest.standard_id, manifest.revision)

        if structures.structure_fingerprint != manifest.structure_fingerprint:
            raise StandardSemanticError(
                "definitions were built against other geometry")

        directory = self.directory(manifest.standard_id, manifest.revision,
                                   create=True)
        _atomic_write(directory / SEMANTIC_FILE, canonical_json(semantics.to_dict()))

        # The manifest is the commit marker and is always written last.

        _atomic_write(directory / MANIFEST_FILE, canonical_json(manifest.to_dict()))

        return directory

    def load(self, standard_id: str, revision: str,
             ) -> tuple[StandardSemanticManifest, Mapping[str, object]]:
        directory = self.directory(standard_id, revision)

        for name in (MANIFEST_FILE, SEMANTIC_FILE):
            if (directory / name).is_symlink():
                raise StandardSemanticError("semantic file uses a symlink")

        try:
            manifest = StandardSemanticManifest.from_dict(
                json.loads((directory / MANIFEST_FILE).read_text("utf-8")))
            payload = json.loads((directory / SEMANTIC_FILE).read_text("utf-8"))
        except FileNotFoundError as exc:
            raise StandardSemanticError("no semantic definitions for this revision") from exc
        except ValueError as exc:
            raise StandardSemanticError(f"semantic store is unreadable: {exc}") from exc

        corpus = self.store.verify_corpus(standard_id, revision)

        if manifest.corpus_fingerprint != corpus.corpus_manifest_sha256:
            raise StandardSemanticError(
                "semantic definitions are stale: the canonical corpus has changed")

        layout = self.store.revision_dir(standard_id, revision) / "layout.json"

        if layout.is_file() and not layout.is_symlink():
            current = json.loads(layout.read_text("utf-8")).get("layout_fingerprint")

            if current != manifest.layout_fingerprint:
                raise StandardSemanticError(
                    "semantic definitions are stale: the layout has changed")

        structures, _ = self.structures.load(standard_id, revision)

        if structures.structure_fingerprint != manifest.structure_fingerprint:
            raise StandardSemanticError(
                "semantic definitions are stale: the geometry has changed")

        return manifest, payload

    def state(self, standard_id: str, revision: str) -> str:
        try:
            self.load(standard_id, revision)
        except (StandardSemanticError, StandardStoreError, FileNotFoundError) as exc:
            return f"UNAVAILABLE ({exc})"

        return "READY"
