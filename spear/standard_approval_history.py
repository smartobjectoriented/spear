"""Every human decision about a bitfield candidate, kept for good.

Approvals used to live in one file keyed by candidate, so a second review of
the same candidate replaced the first and the first stopped existing. That was
survivable only while no candidate had ever been reviewed twice. STD2E-A made
p155's candidate set grow, a re-review became necessary, and the cost of the
design became a real one: re-approving would have erased what the reviewer
approved the first time, on what evidence, and when.

So the decisions move here, append-only, and the approvals file keeps only the
pointer to the one that governs now. A decision is written once and never
rewritten. Superseding one appends another that names it, which makes the chain
readable in either direction: what governs today, and everything it replaced.

An event is self-contained. It carries the verdict, the roles, the reviewer,
the time, the geometry it was recorded against and the evidence it was bound
to, so reading it needs nothing else -- not the active approval, and not a
build. That matters most for the events that will never be active again.

History is not semantic input. Promotion reads the active approval and only
the active approval; nothing here has ever produced a definition. Nor is a
past event rewritten when the model moves on: it stated what was reviewed
then, and it goes on stating it. Whether it may still govern promotion is a
separate question, asked of the active approval, and answered elsewhere.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from standard_schema import canonical_json, sha256_json

HISTORY_FILE = "approval-history.jsonl"

# The file layout, not the decision model. The number bound into review
# evidence is APPROVAL_SCHEMA_VERSION and is deliberately not this one: an
# event id says nothing about what a reviewer decided, and moving the evidence
# marker would expire every approval in the store to record that fact.

APPROVAL_FILE_SCHEMA_VERSION = 3
_EVENT_ID = re.compile(r"^evt-[0-9a-f]{16}$")


class StandardApprovalHistoryError(RuntimeError):
    """Raised when recorded review history cannot be trusted as it stands."""


class ApprovalEventType(StrEnum):
    """How a decision came to be recorded."""

    INITIAL_APPROVAL = "INITIAL_APPROVAL"
    REAPPROVAL = "REAPPROVAL"

    # A decision that predates this file and was carried into it unchanged.

    MIGRATED_INITIAL_APPROVAL = "MIGRATED_INITIAL_APPROVAL"


class EvidenceStatus(StrEnum):
    """Whether the decision was bound to a review-evidence fingerprint.

    The earliest approvals were recorded before evidence binding existed. That
    is a fact about when they were made, not a gap to be filled in later, and
    it is written down rather than left as an unexplained absence.
    """

    BOUND = "BOUND"
    PREDATES_EVIDENCE_BINDING = "PREDATES_EVIDENCE_BINDING"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class StandardApprovalEvent:
    """One human decision, exactly as it was made, for ever."""

    candidate_id: str
    standard_id: str
    revision: str
    verdict: str
    span_roles: tuple[str, ...]
    reviewer: str
    approved_at: str
    structure_fingerprint: str
    review_evidence_fingerprint: str | None
    reviewed_links: tuple[Mapping[str, object], ...]
    packet_identity: str | None
    notes: str | None
    approval_schema_version: int
    event_type: ApprovalEventType
    supersedes: str | None = None

    # When the event reached the log, which is not when the human decided. A
    # migrated decision keeps its own approved_at and gains a recorded_at.

    recorded_at: str = ""
    event_id: str = ""

    @property
    def evidence_status(self) -> EvidenceStatus:
        return (EvidenceStatus.BOUND if self.review_evidence_fingerprint
                else EvidenceStatus.PREDATES_EVIDENCE_BINDING)

    def identity(self) -> dict[str, object]:
        """Everything that makes this decision the decision that it is.

        `recorded_at` is outside it, so carrying an existing approval into the
        log twice produces the same event and the second carry is a no-op.
        """
        return {
            "candidate_id": self.candidate_id,
            "standard_id": self.standard_id,
            "revision": self.revision,
            "verdict": self.verdict,
            "span_roles": list(self.span_roles),
            "reviewer": self.reviewer,
            "approved_at": self.approved_at,
            "structure_fingerprint": self.structure_fingerprint,
            "review_evidence_fingerprint": self.review_evidence_fingerprint,
            "reviewed_links": [dict(item) for item in self.reviewed_links],
            "packet_identity": self.packet_identity,
            "notes": self.notes,
            "approval_schema_version": self.approval_schema_version,
            "event_type": self.event_type.value,
            "supersedes": self.supersedes,
        }

    def sealed(self, *, recorded_at: str | None = None) -> "StandardApprovalEvent":
        """The same decision, with its identity computed and stamped."""
        from dataclasses import replace

        return replace(self, recorded_at=recorded_at or _now(),
                       event_id="evt-" + sha256_json(self.identity())[:16])

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["event_type"] = self.event_type.value
        value["span_roles"] = list(self.span_roles)
        value["reviewed_links"] = [dict(item) for item in self.reviewed_links]
        value["evidence_status"] = self.evidence_status.value

        return value

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "StandardApprovalEvent":
        try:
            return cls(
                candidate_id=str(raw["candidate_id"]),
                standard_id=str(raw["standard_id"]),
                revision=str(raw["revision"]),
                verdict=str(raw["verdict"]),
                span_roles=tuple(str(item) for item in raw.get("span_roles", ())),
                reviewer=str(raw["reviewer"]),
                approved_at=str(raw["approved_at"]),
                structure_fingerprint=str(raw["structure_fingerprint"]),
                review_evidence_fingerprint=(
                    str(raw["review_evidence_fingerprint"])
                    if raw.get("review_evidence_fingerprint") else None),
                reviewed_links=tuple(dict(item)
                                     for item in raw.get("reviewed_links", ())),
                packet_identity=(str(raw["packet_identity"])
                                 if raw.get("packet_identity") else None),
                notes=str(raw["notes"]) if raw.get("notes") else None,
                approval_schema_version=int(raw["approval_schema_version"]),
                event_type=ApprovalEventType(str(raw["event_type"])),
                supersedes=(str(raw["supersedes"])
                            if raw.get("supersedes") else None),
                recorded_at=str(raw.get("recorded_at", "")),
                event_id=str(raw["event_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise StandardApprovalHistoryError(
                f"malformed approval event: {exc}") from exc


def event_for(approval, *, standard_id: str, revision: str,
              event_type: ApprovalEventType,
              supersedes: str | None = None) -> StandardApprovalEvent:
    """The event that records one approval, taken from the approval itself.

    Nothing is recomputed here. The reviewer, the time and the evidence are
    whatever the decision already says they are, which is what makes carrying
    an existing approval into the log a faithful copy rather than a new claim.
    """
    return StandardApprovalEvent(
        candidate_id=approval.candidate_id, standard_id=standard_id,
        revision=revision, verdict=approval.verdict,
        span_roles=tuple(approval.span_roles), reviewer=approval.reviewer,
        approved_at=approval.reviewed_at,
        structure_fingerprint=approval.structure_fingerprint,
        review_evidence_fingerprint=approval.review_evidence_fingerprint,
        reviewed_links=tuple(item.to_dict() for item in approval.reviewed_links),
        packet_identity=approval.packet_identity, notes=approval.notes,
        approval_schema_version=approval.schema_version,
        event_type=event_type, supersedes=supersedes).sealed()


class StandardApprovalHistory:
    """The append-only log of review decisions for one standard revision."""

    def __init__(self, store) -> None:
        from standard_structure_store import StandardStructureStore

        self.store = store
        self.structures = StandardStructureStore(store)

    def path(self, standard_id: str, revision: str) -> Path:
        return self.structures.directory(standard_id, revision) / HISTORY_FILE

    # -- reading ----------------------------------------------------------

    def events(self, standard_id: str, revision: str, *,
               candidate_id: str | None = None,
               ) -> tuple[StandardApprovalEvent, ...]:
        """Every recorded decision, oldest first, in the order it was written.

        A line that cannot be read is not skipped. A review log that has been
        truncated or corrupted is refused whole, because a decision quietly
        missing from it is exactly the thing this file exists to prevent.
        """
        target = self.path(standard_id, revision)

        if target.is_symlink():
            raise StandardApprovalHistoryError("approval history uses a symlink")

        if not target.is_file():
            return ()

        found: list[StandardApprovalEvent] = []

        for number, line in enumerate(
                target.read_text("utf-8").splitlines(), start=1):
            if not line.strip():
                continue

            try:
                raw = json.loads(line)
            except ValueError as exc:
                raise StandardApprovalHistoryError(
                    f"approval history line {number} is unreadable "
                    f"(truncated or corrupt): {exc}") from exc

            event = StandardApprovalEvent.from_dict(raw)

            if event.standard_id != standard_id or event.revision != revision:
                raise StandardApprovalHistoryError(
                    f"approval history line {number} belongs to "
                    f"{event.standard_id} {event.revision}")

            found.append(event)

        self._check(found)

        if candidate_id is not None:
            found = [item for item in found if item.candidate_id == candidate_id]

        return tuple(found)

    @staticmethod
    def _check(events: Sequence[StandardApprovalEvent]) -> None:
        """Refuse a history that cannot be read as one chain per candidate."""
        by_id: dict[str, StandardApprovalEvent] = {}

        for event in events:
            if not _EVENT_ID.fullmatch(event.event_id):
                raise StandardApprovalHistoryError(
                    f"malformed approval event id {event.event_id!r}")

            if event.event_id in by_id:
                raise StandardApprovalHistoryError(
                    f"approval event {event.event_id} is recorded twice")

            by_id[event.event_id] = event

        superseded: set[str] = set()

        for event in events:
            if event.supersedes is None:
                continue

            previous = by_id.get(event.supersedes)

            if previous is None:
                raise StandardApprovalHistoryError(
                    f"approval event {event.event_id} supersedes "
                    f"{event.supersedes}, which is not recorded")

            if previous.candidate_id != event.candidate_id:
                raise StandardApprovalHistoryError(
                    f"approval event {event.event_id} supersedes a decision "
                    "about another candidate")

            if event.supersedes in superseded:
                raise StandardApprovalHistoryError(
                    f"approval event {event.supersedes} is superseded twice")

            superseded.add(event.supersedes)

        # A chain is written in order and never rewritten, so a predecessor
        # always precedes. That also makes a cycle impossible to express.

        seen: set[str] = set()

        for event in events:
            if event.supersedes is not None and event.supersedes not in seen:
                raise StandardApprovalHistoryError(
                    f"approval event {event.event_id} supersedes "
                    f"{event.supersedes}, which it does not follow")

            seen.add(event.event_id)

    def tip(self, standard_id: str, revision: str, candidate_id: str,
            ) -> StandardApprovalEvent | None:
        """The one decision for this candidate that nothing has replaced."""
        events = self.events(standard_id, revision, candidate_id=candidate_id)
        superseded = {item.supersedes for item in events if item.supersedes}
        tips = [item for item in events if item.event_id not in superseded]

        if not tips:
            return None

        if len(tips) > 1:
            raise StandardApprovalHistoryError(
                f"candidate {candidate_id} has {len(tips)} unsuperseded "
                "decisions; the chain is ambiguous")

        return tips[0]

    def chain(self, standard_id: str, revision: str, candidate_id: str,
              ) -> tuple[StandardApprovalEvent, ...]:
        """This candidate's decisions, oldest first, ending at the tip."""
        return self.events(standard_id, revision, candidate_id=candidate_id)

    # -- writing ----------------------------------------------------------

    def append(self, standard_id: str, revision: str,
               events: Sequence[StandardApprovalEvent]) -> None:
        """Add decisions to the end of the log. Nothing already there moves.

        Written and flushed before the active approval is swapped, so the only
        way a crash can leave the two disagreeing is with a decision recorded
        that never took effect. That direction loses nothing.
        """

        if not events:
            return

        target = self.path(standard_id, revision)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        if target.exists() and target.is_symlink():
            raise StandardApprovalHistoryError(
                "refusing to append through a symlink")

        existing = {item.event_id for item in self.events(standard_id, revision)}
        payload = []

        for event in events:
            if event.event_id in existing:
                raise StandardApprovalHistoryError(
                    f"approval event {event.event_id} is already recorded")

            existing.add(event.event_id)
            payload.append(canonical_json(event.to_dict()) + b"\n")

        handle = os.open(str(target),
                         os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)

        try:
            os.write(handle, b"".join(payload))
            os.fsync(handle)
        finally:
            os.close(handle)

        directory = os.open(str(target.parent), os.O_RDONLY)

        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    # -- consistency with the active approvals ----------------------------

    def verify(self, standard_id: str, revision: str,
               approvals: Mapping[str, object]) -> list[dict[str, object]]:
        """Every way the active approvals and this log fail to agree.

        An active approval naming an event the log does not hold is the serious
        case: it means a decision is being enforced that nothing records. An
        active approval naming no event at all is not a fault -- it is what
        every approval taken before this file existed looks like.
        """
        events = {item.event_id: item
                  for item in self.events(standard_id, revision)}
        problems: list[dict[str, object]] = []

        for candidate_id, approval in sorted(approvals.items()):
            event_id = getattr(approval, "event_id", None)

            if event_id is None:
                continue

            event = events.get(event_id)

            if event is None:
                problems.append({"candidate_id": candidate_id,
                                 "reason": "ACTIVE_EVENT_MISSING",
                                 "detail": f"active approval names {event_id}, "
                                           "which the history does not hold"})
                continue

            if event.candidate_id != candidate_id:
                problems.append({"candidate_id": candidate_id,
                                 "reason": "ACTIVE_EVENT_MISMATCH",
                                 "detail": f"{event_id} records a decision "
                                           "about another candidate"})
                continue

            if event.identity() != event_for(
                    approval, standard_id=standard_id, revision=revision,
                    event_type=event.event_type,
                    supersedes=event.supersedes).identity():
                problems.append({"candidate_id": candidate_id,
                                 "reason": "ACTIVE_EVENT_DIVERGED",
                                 "detail": f"{event_id} does not describe the "
                                           "active approval"})
                continue

            tip = self.tip(standard_id, revision, candidate_id)

            if tip is not None and tip.event_id != event_id:
                problems.append({"candidate_id": candidate_id,
                                 "reason": "ACTIVE_NOT_TIP",
                                 "detail": f"active approval is {event_id} but "
                                           f"the chain ends at {tip.event_id}"})

        return problems

    # -- carrying existing approvals in ------------------------------------

    def migration_plan(self, standard_id: str, revision: str,
                       approvals: Mapping[str, object],
                       ) -> list[StandardApprovalEvent]:
        """The events that would carry today's approvals into the log.

        One per approval that has none, each a faithful copy: the reviewer, the
        time and the evidence are the ones already recorded. An approval that
        is already in the log produces nothing, so running this twice is the
        same as running it once.
        """
        recorded = {item.event_id
                    for item in self.events(standard_id, revision)}
        planned: list[StandardApprovalEvent] = []

        for candidate_id in sorted(approvals):
            approval = approvals[candidate_id]

            if getattr(approval, "event_id", None) in recorded:
                continue

            event = event_for(
                approval, standard_id=standard_id, revision=revision,
                event_type=ApprovalEventType.MIGRATED_INITIAL_APPROVAL)

            if event.event_id in recorded:
                continue

            recorded.add(event.event_id)
            planned.append(event)

        return planned
