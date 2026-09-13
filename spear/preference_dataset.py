"""Grounded paired and unpaired preference mining from canonical episodes.

Pairing is intentionally stricter than recovery linkage: an automatically
approved pair must contain two observed completions for the exact same
model-visible decision context.  Changed-context recoveries remain independent
outcome-labelled observations.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Sequence

from sft_dataset import (
    LengthStatus, ReviewStatus, SFTDatasetError, _canonical_bytes, _hash,
    _openai_context, _openai_tools, _target_message, _validate_structure,
)
from training_data import (
    TrainingEligibility, TrainingEpisode, TrainingProvenance,
    TrainingRedactionPolicy, TrainingTurn,
)
from training_splits import SplitConfiguration, split_group_for_episode
from training_store import TrainingStore, TrainingStoreError


PREFERENCE_SCHEMA_VERSION = 1
OUTCOME_POLICY_VERSION = 1
PREFERENCE_EXPORTER_VERSION = 1
CONTEXT_INDEX_SCHEMA_VERSION = 1


class PreferenceDatasetError(RuntimeError):
    pass


class OutcomeLabel(StrEnum):
    DESIRABLE = "DESIRABLE"
    UNDESIRABLE = "UNDESIRABLE"
    UNKNOWN = "UNKNOWN"


class EvidenceStrength(StrEnum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


class PreferenceProfile(StrEnum):
    PAIRED_DPO = "paired_dpo"
    UNPAIRED_OUTCOME = "unpaired_outcome"


class ComparisonStatus(StrEnum):
    EXACT_CONTEXT = "EXACT_CONTEXT"
    NOT_PAIRABLE_CONTEXT_CHANGED = "NOT_PAIRABLE_CONTEXT_CHANGED"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class PreferenceConfiguration:
    profile: str = PreferenceProfile.PAIRED_DPO.value
    split: SplitConfiguration = field(default_factory=SplitConfiguration)
    allow_external_web: bool = False
    max_approximate_tokens: int = 32768
    project_filters: tuple[str, ...] = ()
    split_filters: tuple[str, ...] = ()
    superseded_memory_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.profile not in {item.value for item in PreferenceProfile}:
            raise ValueError(f"unsupported preference profile: {self.profile}")

        if self.max_approximate_tokens <= 0:
            raise ValueError("max approximate tokens must be positive")

        if any(item not in {"train", "validation", "test"}
               for item in self.split_filters):
            raise ValueError("invalid split filter")


class DecisionContextFingerprint:
    """Strong identity over only what was visible at the model boundary."""

    VERSION = 1

    @classmethod
    def compute(cls, *, prompt: Sequence[Mapping[str, object]],
                tools: Sequence[Mapping[str, object]], role: str,
                purpose: str) -> str:
        """Identify the decision the model faced, and nothing else.

        Two completions are only comparable if the model saw exactly the same
        thing before producing them, which is prompt, tools, role and purpose.
        Anything from outside the model boundary would make a pair claim more
        than the evidence supports.
        """

        value = {
            "version": cls.VERSION,
            "prompt": list(prompt),
            "tools": list(tools),
            "role": role,
            "purpose": purpose,
        }

        return "context_" + _hash(value)


@dataclass(frozen=True)
class PreferenceCandidate:
    preference_id: str
    schema_version: int
    decision_context_hash: str
    split_group_id: str
    chosen_episode_id: str
    chosen_turn_id: str
    rejected_episode_id: str
    rejected_turn_id: str
    prompt: tuple[Mapping[str, object], ...]
    chosen: tuple[Mapping[str, object], ...]
    rejected: tuple[Mapping[str, object], ...]
    tool_view_hash: str
    tools: tuple[Mapping[str, object], ...]
    chosen_outcome_evidence: Mapping[str, object]
    rejected_outcome_evidence: Mapping[str, object]
    comparison_basis: tuple[str, ...]
    evidence_strength: str
    comparison_status: str
    review_status: str
    review_flags: tuple[str, ...]
    provenance: tuple[str, ...]
    memory_provenance_ids: tuple[str, ...]
    redaction_metadata: Mapping[str, object]
    source_checksums: Mapping[str, str]
    harness_commits: tuple[str, ...]
    character_count: int
    token_count: int
    token_count_kind: str
    length_status: str
    fingerprint: str
    split: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def trainer_record(self) -> dict[str, object]:
        value: dict[str, object] = {
            "prompt": [dict(item) for item in self.prompt],
            "chosen": [dict(item) for item in self.chosen],
            "rejected": [dict(item) for item in self.rejected],
        }

        if self.tools:
            value["tools"] = [dict(item) for item in self.tools]

        return value


@dataclass(frozen=True)
class OutcomePreferenceSample:
    sample_id: str
    schema_version: int
    source_episode_id: str
    source_turn_id: str
    decision_context_hash: str
    prompt: tuple[Mapping[str, object], ...]
    completion: tuple[Mapping[str, object], ...]
    desirable: bool
    grounded_label_basis: tuple[str, ...]
    evidence_strength: str
    tool_view_hash: str
    tools: tuple[Mapping[str, object], ...]
    action_ids: tuple[str, ...]
    tool_outcomes: tuple[Mapping[str, object], ...]
    recovery_turn_ids: tuple[str, ...]
    provenance: tuple[str, ...]
    memory_provenance_ids: tuple[str, ...]
    review_status: str
    review_flags: tuple[str, ...]
    redaction_metadata: Mapping[str, object]
    source_checksum: str
    harness_commit: str | None
    split_group_id: str
    character_count: int
    token_count: int
    token_count_kind: str
    length_status: str
    fingerprint: str
    split: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def trainer_record(self) -> dict[str, object]:
        value: dict[str, object] = {
            "prompt": [dict(item) for item in self.prompt],
            "completion": [dict(item) for item in self.completion],
            "label": self.desirable,
        }

        if self.tools:
            value["tools"] = [dict(item) for item in self.tools]

        return value


@dataclass(frozen=True)
class _TurnObservation:
    episode_id: str
    turn_id: str
    task_id: str
    prompt: tuple[Mapping[str, object], ...]
    completion: Mapping[str, object]
    tools: tuple[Mapping[str, object], ...]
    context_hash: str
    completion_hash: str
    completion_type: str
    label: str
    label_basis: tuple[str, ...]
    evidence_strength: str
    review_status: str
    review_flags: tuple[str, ...]
    tool_view_hash: str
    action_ids: tuple[str, ...]
    tool_outcomes: tuple[Mapping[str, object], ...]
    recovery_turn_ids: tuple[str, ...]
    provenance: tuple[str, ...]
    memory_ids: tuple[str, ...]
    redaction: Mapping[str, object]
    source_checksum: str
    harness_commit: str | None
    split_group_id: str
    split: str
    character_count: int
    token_count: int
    length_status: str


@dataclass
class PreferenceBuildResult:
    paired: list[PreferenceCandidate]
    unpaired: list[OutcomePreferenceSample]
    all_paired: list[PreferenceCandidate]
    all_unpaired: list[OutcomePreferenceSample]
    observations: list[_TurnObservation]
    report: dict[str, object]
    source_checksums: dict[str, str]
    tool_schemas: dict[str, Mapping[str, object]]


class OutcomeLabelPolicy:
    VERSION = OUTCOME_POLICY_VERSION
    NEGATIVE_LABELS = frozenset({
        "failed_tool_action", "repeated_action", "stalled", "invalid_model_turn",
        "failed_model_call", "policy_failure", "security_failure",
        "stale_verification", "budget_exhausted", "ungrounded_completion_claim",
        "cancelled", "cancelled_action",
    })
    PROGRESS_LABELS = frozenset({
        "new_discovery", "useful_read", "useful_search", "meaningful_progress",
        "plan_progress", "recovery_after_failure",
    })

    def label(self, episode: TrainingEpisode, turn: TrainingTurn
              ) -> tuple[OutcomeLabel, tuple[str, ...], EvidenceStrength]:
        """Label one turn from observed outcomes, or refuse to.

        UNKNOWN is the honest default and is used freely: a turn is only called
        desirable or undesirable when something actually observed says so. Every
        return also carries the basis, so a label can be audited afterwards.
        """

        # A delegated turn's outcome belongs to its own child task, not to the
        # decision the main agent was making.

        if not turn.primary_candidate or turn.role.lower() in {"explorer", "reviewer"}:
            return OutcomeLabel.UNKNOWN, ("auxiliary_turn",), EvidenceStrength.WEAK

        labels = set(turn.labels)
        statuses = [call.status for call in turn.tool_calls]
        successes = sum(status in {"ok", "success"} for status in statuses)
        failures = sum(status not in {"ok", "success", None} for status in statuses)

        # Some worked and some did not, so the turn as a whole cannot be
        # attributed either way.

        if turn.tool_calls and successes and failures:
            return (OutcomeLabel.UNKNOWN, ("mixed_multi_tool_outcome",),
                    EvidenceStrength.WEAK)

        negative = sorted(labels & self.NEGATIVE_LABELS)

        if failures or negative:
            basis = list(negative)

            if failures:
                basis.append("grounded_tool_failure")

            if any(status in {"denied", "invalid_arguments", "unknown_tool"}
                   for status in statuses):
                basis.append("policy_or_argument_failure")

            return OutcomeLabel.UNDESIRABLE, tuple(dict.fromkeys(basis)), EvidenceStrength.STRONG

        if turn.error:
            return OutcomeLabel.UNDESIRABLE, ("malformed_model_turn",), EvidenceStrength.STRONG

        positive_episode = (episode.training_metadata.get("eligibility") ==
                            TrainingEligibility.POSITIVE_CANDIDATE.value)

        # A change is only desirable once the episode that contains it was
        # verified and finalized; on its own, editing a file proves nothing.

        if "grounded_mutation" in labels:
            if (positive_episode and episode.outcome.get("verification_status") == "verified"
                    and episode.outcome.get("checkpoint_status") == "finalized"):
                return OutcomeLabel.DESIRABLE, ("verified_grounded_mutation",), EvidenceStrength.STRONG

            return OutcomeLabel.UNKNOWN, ("mutation_contribution_unclear",), EvidenceStrength.WEAK

        # A verification counts only if it covered the code as it then stood:
        # a check of an older generation says nothing about this turn.

        if "verification_action" in labels or "meaningful_verification" in labels:
            current = (turn.verification_generation is None or
                       turn.verification_generation == turn.mutation_generation_after)

            if successes and current:
                return OutcomeLabel.DESIRABLE, ("successful_current_verification",), EvidenceStrength.STRONG

            return OutcomeLabel.UNKNOWN, ("verification_outcome_unclear",), EvidenceStrength.WEAK

        progress = labels & (self.PROGRESS_LABELS - {"recovery_after_failure"})

        if progress and (not turn.tool_calls or successes == len(statuses)):
            basis = tuple(sorted(progress | (labels & {"recovery_after_failure"})))
            return OutcomeLabel.DESIRABLE, basis, EvidenceStrength.STRONG

        # Recovering from a failure is not itself progress: what matters is
        # whether the recovery accomplished anything, which nothing here shows.

        if "recovery_after_failure" in labels:
            return OutcomeLabel.UNKNOWN, ("recovery_progress_unclear",), EvidenceStrength.WEAK

        if ("final_success" in labels and positive_episode
                and episode.outcome.get("terminal_task_status") == "completed"):
            return OutcomeLabel.DESIRABLE, ("eligible_successful_final_response",), EvidenceStrength.STRONG

        return OutcomeLabel.UNKNOWN, ("insufficient_attributable_outcome",), EvidenceStrength.WEAK


class DecisionContextIndex:
    """Rebuildable, checksummed filesystem index of observed decision inputs."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).resolve()
        self.directory = self.root / "datasets" / "preferences" / "index"
        self.path = self.directory / "decision-contexts.json"
        self.checksum_path = self.directory / "decision-contexts.sha256"
        self.lock_path = self.directory / ".index.lock"

    def load(self) -> dict[str, object]:
        # Neither file present means the index has simply never been built.
        # One present without the other is corruption, and is raised below.

        if not self.path.exists() and not self.checksum_path.exists():
            return {"schema_version": CONTEXT_INDEX_SCHEMA_VERSION, "contexts": {}}

        try:
            content = self.path.read_bytes()
            expected = self.checksum_path.read_text(encoding="ascii").strip()

            if hashlib.sha256(content).hexdigest() != expected:
                raise PreferenceDatasetError("preference context index checksum mismatch")

            value = json.loads(content)
        except (OSError, json.JSONDecodeError) as exc:
            raise PreferenceDatasetError(f"corrupt preference context index: {exc}") from exc

        if (not isinstance(value, dict)
                or value.get("schema_version") != CONTEXT_INDEX_SCHEMA_VERSION
                or not isinstance(value.get("contexts"), dict)):
            raise PreferenceDatasetError("unsupported preference context index")

        return value

    def update(self, observations: Sequence[_TurnObservation], *, rebuild: bool = False) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)

        try:
            fcntl.flock(lock, fcntl.LOCK_EX)

            value = ({"schema_version": CONTEXT_INDEX_SCHEMA_VERSION, "contexts": {}}
                     if rebuild else self.load())
            contexts = value["contexts"]

            # Each observation replaces any earlier row for the same turn, so
            # re-indexing an episode updates rather than duplicates it. The
            # rows are then sorted, which keeps the file byte-stable.

            for observation in observations:
                rows = contexts.setdefault(observation.context_hash, [])
                reference = {
                    "episode_id": observation.episode_id,
                    "turn_id": observation.turn_id,
                    "completion_hash": observation.completion_hash,
                    "label": observation.label,
                    "source_checksum": observation.source_checksum,
                }
                key = (observation.episode_id, observation.turn_id)
                rows[:] = [item for item in rows
                           if (item.get("episode_id"), item.get("turn_id")) != key]
                rows.append(reference)
                rows.sort(key=lambda item: (item["episode_id"], item["turn_id"]))

            encoded = _canonical_bytes(value) + b"\n"
            self._atomic_write(self.path, encoded)
            self._atomic_write(
                self.checksum_path,
                (hashlib.sha256(encoded).hexdigest() + "\n").encode("ascii"),
            )
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            os.close(lock)

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)

        try:
            os.fchmod(descriptor, 0o600)

            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())

            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class PreferenceDatasetBuilder:
    def __init__(self, store: TrainingStore | str | os.PathLike[str], *,
                 configuration: PreferenceConfiguration | None = None,
                 redaction_policy: TrainingRedactionPolicy | None = None,
                 outcome_policy: OutcomeLabelPolicy | None = None) -> None:
        self.store = store if isinstance(store, TrainingStore) else TrainingStore(store)
        self.configuration = configuration or PreferenceConfiguration()
        self.redaction_policy = redaction_policy or TrainingRedactionPolicy.from_environment()
        self.outcome_policy = outcome_policy or OutcomeLabelPolicy()
        self.index = DecisionContextIndex(self.store.root)

    def _source_checksum(self, episode_id: str) -> str:
        try:
            value = (self.store.episodes / f"{episode_id}.sha256").read_text(
                encoding="ascii",
            ).strip()
        except OSError as exc:
            raise PreferenceDatasetError(f"missing source checksum: {episode_id}") from exc

        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise PreferenceDatasetError(f"malformed source checksum: {episode_id}")

        return value

    @staticmethod
    def _completion_type(episode: TrainingEpisode, turn: TrainingTurn) -> str:
        """What kind of thing the turn produced; pairs across kinds need review."""

        if turn.tool_calls:
            if "verification_action" in turn.labels:
                return "verification_action"

            return "tool_action"

        if ("final_success" in turn.labels or
                turn.assistant_output == episode.outcome.get("final_response")):
            return "final_response"

        return "assistant_text"

    def _observe_turn(self, episode: TrainingEpisode, turn: TrainingTurn,
                      checksum: str) -> _TurnObservation:
        """Turn one captured turn into a labelled, redacted observation."""

        # The tool schemas are part of what the model saw, so they are verified
        # against the hash the turn recorded before anything is built on them.

        snapshot = episode.tool_views.get(turn.tool_view_hash)

        if not isinstance(snapshot, Mapping) or _hash(snapshot) != turn.tool_view_hash:
            raise PreferenceDatasetError("missing or corrupt captured tool view")

        prompt = tuple(_openai_context(turn))
        completion = _target_message(turn)
        has_tool_context = bool(turn.tool_calls) or any(
            isinstance(block, Mapping) and block.get("type") in {"tool_use", "tool_result"}
            for message in turn.messages for block in message.get("content", ())
        )
        tools = _openai_tools(snapshot) if turn.tools_enabled or has_tool_context else ()

        try:
            _validate_structure(
                (*prompt, completion), tools, len(prompt),
                allow_target_schema_mismatch=any(
                    call.status == "invalid_arguments" for call in turn.tool_calls
                ),
            )
        except SFTDatasetError as exc:
            raise PreferenceDatasetError(str(exc)) from exc

        context_hash = DecisionContextFingerprint.compute(
            prompt=prompt, tools=tools, role=turn.role, purpose=turn.purpose,
        )
        label, basis, strength = self.outcome_policy.label(episode, turn)
        cleaned, redaction = self.redaction_policy.redact({
            "prompt": list(prompt), "completion": completion, "tools": list(tools),
        })

        # Redacting the PROMPT is harmless; redacting the COMPLETION changes
        # what would be trained on, so it is flagged for a human to look at.

        completion_changed = _canonical_bytes(cleaned["completion"]) != _canonical_bytes(completion)
        prompt = tuple(cleaned["prompt"])
        completion = cleaned["completion"]
        tools = tuple(cleaned["tools"])
        flags: list[str] = []
        review = ReviewStatus.AUTO_APPROVED

        if completion_changed:
            flags.append("completion_redacted")
            review = ReviewStatus.NEEDS_REVIEW
        elif int(redaction.get("redaction_count", 0)):
            flags.append("prompt_redacted")

        if (TrainingProvenance.EXTERNAL_WEB.value in turn.provenance
                and not self.configuration.allow_external_web):
            flags.append("external_content")
            review = ReviewStatus.REJECTED

        if set(turn.memory_provenance_ids) & set(self.configuration.superseded_memory_ids):
            flags.append("superseded_memory")
            review = ReviewStatus.NEEDS_REVIEW

        payload = {"prompt": prompt, "completion": completion, "tools": tools}
        characters = len(_canonical_bytes(payload).decode("utf-8"))
        tokens = max(1, math.ceil(characters / 3.5))
        length_status = (LengthStatus.OVERLONG if tokens >
                         self.configuration.max_approximate_tokens else
                         LengthStatus.EXPORTABLE)

        if length_status == LengthStatus.OVERLONG:
            flags.append("overlong")

            if review != ReviewStatus.REJECTED:
                review = ReviewStatus.NEEDS_REVIEW

        split_group = split_group_for_episode(episode)

        return _TurnObservation(
            episode.episode_id, turn.turn_id, episode.task_id, prompt, completion,
            tools, context_hash, _hash(completion), self._completion_type(episode, turn),
            label.value, basis, strength.value, review.value, tuple(flags),
            turn.tool_view_hash,
            tuple(call.action_id for call in turn.tool_calls if call.action_id),
            tuple({
                "tool_call_id": call.tool_call_id, "name": call.name,
                "action_id": call.action_id, "status": call.status,
                "result_reference": call.result_reference,
            } for call in turn.tool_calls),
            tuple(turn.recovers_turn_ids), tuple(turn.provenance),
            tuple(turn.memory_provenance_ids), {**dict(redaction),
                                               "completion_changed": completion_changed},
            checksum,
            (str(episode.provenance.get("harness_git_commit"))
             if episode.provenance.get("harness_git_commit") else None),
            split_group, self.configuration.split.assign(split_group), characters,
            tokens, length_status.value,
        )

    def _load_observations(self) -> tuple[list[_TurnObservation], dict[str, str],
                                          dict[str, Mapping[str, object]], Counter]:
        observations: list[_TurnObservation] = []
        checksums: dict[str, str] = {}
        schemas: dict[str, Mapping[str, object]] = {}
        exclusions: Counter[str] = Counter()

        for metadata in sorted(self.store.iterate_metadata(),
                               key=lambda item: str(item.get("episode_id"))):
            episode_id = str(metadata.get("episode_id", ""))

            try:
                episode = self.store.load_episode(episode_id)
            except TrainingStoreError as exc:
                raise PreferenceDatasetError(f"corrupt source episode {episode_id}: {exc}") from exc

            checksum = self._source_checksum(episode_id)

            if checksum != metadata.get("checksum"):
                raise PreferenceDatasetError(f"source manifest checksum mismatch: {episode_id}")

            checksums[episode_id] = checksum
            project = str(episode.provenance.get("project") or "unknown")

            if self.configuration.project_filters and project not in self.configuration.project_filters:
                exclusions["project_filtered"] += len(episode.turns)
                continue

            for view_hash, snapshot in episode.tool_views.items():
                if _hash(snapshot) != view_hash:
                    raise PreferenceDatasetError("captured tool-view hash mismatch")

                schemas.setdefault(view_hash, snapshot)

            for turn in episode.turns:
                try:
                    observation = self._observe_turn(episode, turn, checksum)
                except PreferenceDatasetError:
                    exclusions["malformed_tool_structure"] += 1
                    continue

                if (self.configuration.split_filters
                        and observation.split not in self.configuration.split_filters):
                    exclusions["split_filtered"] += 1
                    continue

                observations.append(observation)

        return observations, checksums, schemas, exclusions

    @staticmethod
    def _mark_completion_conflicts(observations: list[_TurnObservation]
                                   ) -> tuple[list[_TurnObservation], int]:
        """Flag identical completions that were observed with opposite outcomes.

        The same answer to the same question cannot be both good and bad. When
        it appears as both, the labelling is not trustworthy for either.
        """

        labels: dict[tuple[str, str], set[str]] = defaultdict(set)

        for item in observations:
            if item.label != OutcomeLabel.UNKNOWN.value:
                labels[(item.context_hash, item.completion_hash)].add(item.label)

        conflicts = {key for key, values in labels.items() if len(values) > 1}
        result = []

        for item in observations:
            if (item.context_hash, item.completion_hash) in conflicts:
                result.append(replace(
                    item, review_status=ReviewStatus.NEEDS_REVIEW.value,
                    review_flags=tuple(dict.fromkeys((*item.review_flags,
                                                       "contradictory_outcome"))),
                ))
            else:
                result.append(item)

        return result, len(conflicts)

    @staticmethod
    def _make_unpaired(observation: _TurnObservation) -> OutcomePreferenceSample:
        desirable = observation.label == OutcomeLabel.DESIRABLE.value
        logical = {
            "context": observation.context_hash,
            "completion": observation.completion,
            "label": desirable,
            "tools": observation.tools,
            "source_episode": observation.episode_id,
            "source_turn": observation.turn_id,
        }
        fingerprint = _hash(logical)

        return OutcomePreferenceSample(
            "outcome_" + fingerprint[:32], PREFERENCE_SCHEMA_VERSION,
            observation.episode_id, observation.turn_id, observation.context_hash,
            observation.prompt, (observation.completion,), desirable,
            observation.label_basis, observation.evidence_strength,
            observation.tool_view_hash, observation.tools, observation.action_ids,
            observation.tool_outcomes,
            observation.recovery_turn_ids, observation.provenance,
            observation.memory_ids, observation.review_status,
            observation.review_flags, observation.redaction,
            observation.source_checksum, observation.harness_commit,
            observation.split_group_id, observation.character_count,
            observation.token_count, "approximate", observation.length_status,
            fingerprint, observation.split,
        )

    def _make_pair(self, chosen: _TurnObservation, rejected: _TurnObservation
                   ) -> PreferenceCandidate:
        """Build one pair, downgrading it for every doubt it carries."""

        # The flags accumulate; the review status only ever gets stricter,
        # never relaxed by a later check.

        flags: list[str] = []
        review = ReviewStatus.AUTO_APPROVED
        comparison = ComparisonStatus.EXACT_CONTEXT

        # Two halves from different splits would put the same comparison on
        # both sides of the train/test boundary.

        if chosen.split_group_id != rejected.split_group_id or chosen.split != rejected.split:
            flags.append("source_split_group_mismatch")
            review = ReviewStatus.REJECTED
            comparison = ComparisonStatus.NOT_COMPARABLE

        if chosen.completion_type != rejected.completion_type:
            flags.append("mixed_completion_types")
            review = ReviewStatus.NEEDS_REVIEW

        if (chosen.review_status != ReviewStatus.AUTO_APPROVED.value
                or rejected.review_status != ReviewStatus.AUTO_APPROVED.value):
            flags.append("source_requires_review")

            if review != ReviewStatus.REJECTED:
                review = ReviewStatus.NEEDS_REVIEW

        # A pair is only as strong as its weaker half.

        strength = (EvidenceStrength.STRONG if
                    chosen.evidence_strength == rejected.evidence_strength ==
                    EvidenceStrength.STRONG.value else EvidenceStrength.MODERATE)

        if strength != EvidenceStrength.STRONG:
            flags.append("non_strong_evidence")

            if review != ReviewStatus.REJECTED:
                review = ReviewStatus.NEEDS_REVIEW

        logical = {
            "context": chosen.context_hash, "chosen": chosen.completion,
            "rejected": rejected.completion, "tools": chosen.tools,
        }
        fingerprint = _hash(logical)
        chars = len(_canonical_bytes(logical).decode("utf-8"))
        tokens = max(1, math.ceil(chars / 3.5))
        length_status = (LengthStatus.OVERLONG if tokens >
                         self.configuration.max_approximate_tokens or
                         chosen.length_status == LengthStatus.OVERLONG.value or
                         rejected.length_status == LengthStatus.OVERLONG.value else
                         LengthStatus.EXPORTABLE)

        if length_status == LengthStatus.OVERLONG:
            flags.append("overlong")

            if review != ReviewStatus.REJECTED:
                review = ReviewStatus.NEEDS_REVIEW

        redaction = {
            "chosen": chosen.redaction, "rejected": rejected.redaction,
        }

        return PreferenceCandidate(
            "preference_" + fingerprint[:32], PREFERENCE_SCHEMA_VERSION,
            chosen.context_hash, chosen.split_group_id,
            chosen.episode_id, chosen.turn_id, rejected.episode_id, rejected.turn_id,
            chosen.prompt, (chosen.completion,), (rejected.completion,),
            chosen.tool_view_hash, chosen.tools,
            {"label_basis": chosen.label_basis, "action_ids": chosen.action_ids,
             "tool_outcomes": chosen.tool_outcomes},
            {"label_basis": rejected.label_basis, "action_ids": rejected.action_ids,
             "tool_outcomes": rejected.tool_outcomes},
            ("same_exact_model_input", "grounded_desirable_vs_undesirable"),
            strength.value, comparison.value, review.value, tuple(flags),
            tuple(sorted(set(chosen.provenance) | set(rejected.provenance))),
            tuple(sorted(set(chosen.memory_ids) | set(rejected.memory_ids))),
            redaction,
            {chosen.episode_id: chosen.source_checksum,
             rejected.episode_id: rejected.source_checksum},
            tuple(sorted({item for item in (chosen.harness_commit,
                                             rejected.harness_commit) if item})),
            chars, tokens, "approximate", length_status.value, fingerprint,
            chosen.split,
        )

    def build(self) -> PreferenceBuildResult:
        """Mine every pair and unpaired sample the stored episodes support."""

        observations, checksums, schemas, exclusions = self._load_observations()
        observations, completion_conflicts = self._mark_completion_conflicts(observations)

        # Recovery links are counted as grounded good/bad relationships even
        # when they cannot be paired, which is what the pairability rate below
        # is measured against.

        initial_by_source = {(item.episode_id, item.turn_id): item
                             for item in observations}
        recovery_relationships = 0
        recovery_not_pairable = 0
        recovery_grounded_relationships = 0
        annotated_observations = []

        for recovery in observations:
            context_changed = False

            for failed_turn_id in recovery.recovery_turn_ids:
                failed = initial_by_source.get((recovery.episode_id, failed_turn_id))

                if failed is None:
                    continue

                recovery_relationships += 1

                if (recovery.label == OutcomeLabel.DESIRABLE.value
                        and failed.label == OutcomeLabel.UNDESIRABLE.value):
                    recovery_grounded_relationships += 1

                # The retry saw something the first attempt did not, so the
                # two answers are not answers to the same question.

                if recovery.context_hash != failed.context_hash:
                    context_changed = True
                    recovery_not_pairable += 1
                    exclusions["context_mismatch"] += 1

            if context_changed:
                recovery = replace(
                    recovery,
                    review_flags=tuple(dict.fromkeys((
                        *recovery.review_flags,
                        ComparisonStatus.NOT_PAIRABLE_CONTEXT_CHANGED.value,
                    ))),
                )

            annotated_observations.append(recovery)

        observations = annotated_observations
        by_context: dict[str, list[_TurnObservation]] = defaultdict(list)

        for item in observations:
            by_context[item.context_hash].append(item)

        raw_pairs: list[PreferenceCandidate] = []
        exact_context_candidates = 0

        # Every desirable answer is paired against every undesirable one for
        # the same decision context. That is the whole pairing rule.

        for items in by_context.values():
            desirable = [item for item in items if item.label == OutcomeLabel.DESIRABLE.value]
            undesirable = [item for item in items if item.label == OutcomeLabel.UNDESIRABLE.value]

            for chosen in desirable:
                for rejected in undesirable:
                    # Identical text labelled both ways: already flagged as a
                    # conflict above, and useless as a preference either way.

                    if chosen.completion_hash == rejected.completion_hash:
                        continue

                    exact_context_candidates += 1
                    raw_pairs.append(self._make_pair(chosen, rejected))

        deduped: dict[str, PreferenceCandidate] = {}

        for pair in sorted(raw_pairs, key=lambda item: (
                item.preference_id, item.chosen_episode_id, item.rejected_episode_id)):
            if pair.fingerprint in deduped:
                exclusions["duplicate"] += 1
            else:
                deduped[pair.fingerprint] = pair

        pairs = list(deduped.values())

        # The same two completions preferred in both directions somewhere in
        # the corpus. Keyed on the sorted pair so the two orderings collide.

        directions: dict[tuple[str, tuple[str, str]], set[tuple[str, str]]] = defaultdict(set)

        for pair in pairs:
            chosen_hash = _hash(pair.chosen[0])
            rejected_hash = _hash(pair.rejected[0])
            key = (pair.decision_context_hash, tuple(sorted((chosen_hash, rejected_hash))))
            directions[key].add((chosen_hash, rejected_hash))

        contradictory_keys = {key for key, values in directions.items() if len(values) > 1}

        if contradictory_keys:
            updated = []

            for pair in pairs:
                chosen_hash = _hash(pair.chosen[0])
                rejected_hash = _hash(pair.rejected[0])
                key = (pair.decision_context_hash, tuple(sorted((chosen_hash, rejected_hash))))

                if key in contradictory_keys:
                    updated.append(replace(
                        pair, comparison_status=ComparisonStatus.CONFLICT.value,
                        review_status=ReviewStatus.NEEDS_REVIEW.value,
                        review_flags=tuple(dict.fromkeys((*pair.review_flags, "conflict"))),
                    ))
                else:
                    updated.append(pair)

            pairs = updated

        unpaired_by_fingerprint: dict[str, OutcomePreferenceSample] = {}
        unknown = 0

        for observation in observations:
            if observation.label == OutcomeLabel.UNKNOWN.value:
                unknown += 1
                continue

            sample = self._make_unpaired(observation)
            existing = unpaired_by_fingerprint.get(sample.fingerprint)

            if existing is None:
                unpaired_by_fingerprint[sample.fingerprint] = sample
            else:
                exclusions["duplicate"] += 1

        unpaired = list(unpaired_by_fingerprint.values())

        grounded_relationships = len(pairs) + recovery_grounded_relationships

        approved_pairs = [item for item in pairs
                          if item.review_status == ReviewStatus.AUTO_APPROVED.value]
        approved_unpaired = [item for item in unpaired
                             if item.review_status == ReviewStatus.AUTO_APPROVED.value]

        for observation in observations:
            if "mixed_multi_tool_outcome" in observation.label_basis:
                exclusions["mixed_multi_tool_outcome"] += 1

            if observation.label == OutcomeLabel.UNKNOWN.value:
                exclusions["weak_evidence"] += 1

            flag_reasons = {
                "external_content": "external_content",
                "completion_redacted": "redaction",
                "superseded_memory": "superseded_memory",
                "overlong": "overlong",
            }

            for flag, reason in flag_reasons.items():
                if flag in observation.review_flags:
                    exclusions[reason] += 1

        for pair in pairs:
            if "conflict" in pair.review_flags:
                exclusions["conflict"] += 1

            if "non_strong_evidence" in pair.review_flags:
                exclusions["weak_evidence"] += 1

            if "source_split_group_mismatch" in pair.review_flags:
                exclusions["source_split_group_mismatch"] += 1

        self._check_leakage(approved_pairs, approved_unpaired)
        report = self._report(
            observations, pairs, approved_pairs, unpaired, approved_unpaired,
            exclusions, exact_context_candidates, completion_conflicts,
            len(contradictory_keys), recovery_relationships,
            recovery_not_pairable, unknown, grounded_relationships,
        )

        return PreferenceBuildResult(
            approved_pairs, approved_unpaired, pairs, unpaired, observations,
            report, checksums, schemas,
        )

    @staticmethod
    def _check_leakage(pairs: Sequence[PreferenceCandidate],
                       unpaired: Sequence[OutcomePreferenceSample]) -> None:
        """Refuse an export where a group or episode spans more than one split."""

        groups: dict[str, set[str]] = defaultdict(set)
        episodes: dict[str, set[str]] = defaultdict(set)

        for item in pairs:
            groups[item.split_group_id].add(item.split)
            episodes[item.chosen_episode_id].add(item.split)
            episodes[item.rejected_episode_id].add(item.split)

        for item in unpaired:
            groups[item.split_group_id].add(item.split)
            episodes[item.source_episode_id].add(item.split)

        if any(len(value) > 1 for value in groups.values()):
            raise PreferenceDatasetError("training split-group leakage")

        if any(len(value) > 1 for value in episodes.values()):
            raise PreferenceDatasetError("source episode leakage")

    @staticmethod
    def _report(observations, pairs, approved_pairs, unpaired, approved_unpaired,
                exclusions, exact_context_candidates, completion_conflicts,
                direction_conflicts, recoveries, recovery_not_pairable, unknown,
                grounded_relationships) -> dict[str, object]:
        labels = Counter(item.label for item in observations)
        pair_review = Counter(item.review_status for item in pairs)
        unpaired_labels = Counter("positive" if item.desirable else "negative"
                                  for item in approved_unpaired)
        contexts: dict[str, set[str]] = defaultdict(set)

        for item in observations:
            contexts[item.context_hash].add(item.completion_hash)

        collisions = sum(len(values) > 1 for values in contexts.values())
        potentially_pairable = sum(
            any(item.label == OutcomeLabel.DESIRABLE.value for item in values)
            and any(item.label == OutcomeLabel.UNDESIRABLE.value for item in values)
            for values in (group for group in (
                [item for item in observations if item.context_hash == context]
                for context in contexts
            ))
        )
        pairability = (len(approved_pairs) / grounded_relationships
                       if grounded_relationships else 0.0)

        return {
            "source_turns": {
                "considered": (len(observations)
                               + int(exclusions.get("malformed_tool_structure", 0))),
                "desirable": labels[OutcomeLabel.DESIRABLE.value],
                "undesirable": labels[OutcomeLabel.UNDESIRABLE.value],
                "unknown": labels[OutcomeLabel.UNKNOWN.value],
            },
            "paired_candidates": {
                "exact_context_candidates": exact_context_candidates,
                "AUTO_APPROVED": len(approved_pairs),
                "NEEDS_REVIEW": pair_review[ReviewStatus.NEEDS_REVIEW.value],
                "REJECTED": pair_review[ReviewStatus.REJECTED.value],
                "conflicts": completion_conflicts + direction_conflicts,
                "not_pairable": recovery_not_pairable,
                "exact_context_collisions": collisions,
                "potentially_pairable_contexts": potentially_pairable,
            },
            "recovery": {
                "relationships": recoveries,
                "not_pairable_context_changed": recovery_not_pairable,
            },
            "unpaired": {
                "positive": unpaired_labels["positive"],
                "negative": unpaired_labels["negative"],
                "unknown_excluded": unknown,
                "all_labeled_candidates": len(unpaired),
            },
            "exclusion_reasons": dict(sorted(exclusions.items())),
            "pairability_rate": pairability,
            "grounded_good_bad_relationships": grounded_relationships,
            "split_sizes": {
                "paired": dict(Counter(item.split for item in approved_pairs)),
                "unpaired": dict(Counter(item.split for item in approved_unpaired)),
            },
        }

    def index_episode(self, episode: TrainingEpisode) -> None:
        """Add one episode's decision contexts to the index as it is captured."""

        checksum = self._source_checksum(episode.episode_id)
        observations = []

        # A turn that cannot be observed is skipped rather than fatal: indexing
        # is an optimisation, and build() re-derives everything from scratch.

        for turn in episode.turns:
            try:
                observations.append(self._observe_turn(episode, turn, checksum))
            except PreferenceDatasetError:
                continue

        self.index.update(observations)

    def rebuild_index(self) -> None:
        observations, _, _, _ = self._load_observations()
        self.index.update(observations, rebuild=True)

    @staticmethod
    def _write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        # Identical content is left alone, so re-materializing an unchanged
        # dataset does not churn mtimes.

        if path.exists() and path.read_bytes() == content:
            return

        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)

        try:
            os.fchmod(descriptor, 0o600)

            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())

            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def materialize(self, output_directory: str | os.PathLike[str], *,
                    dry_run: bool = False, report_only: bool = False
                    ) -> tuple[PreferenceBuildResult, Path | None]:
        result = self.build()

        if dry_run or report_only:
            return result, None

        logical = {
            "sources": result.source_checksums, "profile": self.configuration.profile,
            "split": asdict(self.configuration.split),
            "external": self.configuration.allow_external_web,
            "projects": self.configuration.project_filters,
            "split_filters": self.configuration.split_filters,
        }

        # Derived from the inputs and the configuration, so the same build
        # always lands in the same directory.

        dataset_id = "preference_" + _hash(logical)[:20]
        dataset_dir = Path(output_directory).resolve() / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        selected = (result.paired if self.configuration.profile ==
                    PreferenceProfile.PAIRED_DPO.value else result.unpaired)
        checksums: dict[str, str] = {}

        for split in ("train", "validation", "test"):
            content = b"".join(
                _canonical_bytes(item.trainer_record()) + b"\n"
                for item in selected if item.split == split
            )
            path = dataset_dir / f"{split}.jsonl"
            self._write(path, content)
            checksums[path.name] = hashlib.sha256(content).hexdigest()

        # The splits hold only what was auto-approved; the manifest beside them
        # lists every candidate, including those held back, so a reviewer can
        # see what was excluded and why.

        all_items = (result.all_paired if self.configuration.profile ==
                     PreferenceProfile.PAIRED_DPO.value else result.all_unpaired)
        side = b"".join(_canonical_bytes(item.to_dict()) + b"\n"
                        for item in sorted(all_items, key=lambda value:
                                           getattr(value, "preference_id", None) or value.sample_id))
        self._write(dataset_dir / "sample-manifest.jsonl", side)
        checksums["sample-manifest.jsonl"] = hashlib.sha256(side).hexdigest()
        manifest = {
            "dataset_id": dataset_id, "schema_version": PREFERENCE_SCHEMA_VERSION,
            "profile": self.configuration.profile,
            "source_training_store": str(self.store.root),
            "source_episode_count": len(result.source_checksums),
            "outcome_policy_version": self.outcome_policy.VERSION,
            "exporter_version": PREFERENCE_EXPORTER_VERSION,
            "decision_context_fingerprint_version": DecisionContextFingerprint.VERSION,
            "split_policy": {"method": "shared_sha256_split_group",
                             **asdict(self.configuration.split)},
            "output_checksums": checksums,
            "source_episode_checksums": result.source_checksums,
            "sample_count": len(selected),
            "quality": result.report,
        }
        self._write(dataset_dir / "dataset-manifest.json", _canonical_bytes(manifest) + b"\n")
        self._write(dataset_dir / "quality-report.json", _canonical_bytes(result.report) + b"\n")
        self._write(dataset_dir / "quality-report.md",
                    self._human_report(result.report).encode("utf-8"))
        self._persist_artifacts(result, dataset_id)

        return result, dataset_dir

    def _persist_artifacts(self, result: PreferenceBuildResult, dataset_id: str) -> None:
        root = self.store.root / "datasets" / "preferences"

        for item in result.all_paired:
            self._write(root / "paired" / f"{item.preference_id}.json",
                        _canonical_bytes(item.to_dict()) + b"\n")

        for item in result.all_unpaired:
            self._write(root / "unpaired" / f"{item.sample_id}.json",
                        _canonical_bytes(item.to_dict()) + b"\n")

        self._write(root / "manifests" / f"{dataset_id}.json",
                    _canonical_bytes({
                        "schema_version": PREFERENCE_SCHEMA_VERSION,
                        "dataset_id": dataset_id,
                        "paired": sorted(item.preference_id for item in result.all_paired),
                        "unpaired": sorted(item.sample_id for item in result.all_unpaired),
                    }) + b"\n")

    @staticmethod
    def _human_report(report: Mapping[str, object]) -> str:
        turns = report["source_turns"]
        paired = report["paired_candidates"]
        unpaired = report["unpaired"]

        return "\n".join((
            "# Preference quality report", "",
            f"Source turns considered: {turns['considered']}",
            f"Desirable / undesirable / unknown: {turns['desirable']} / "
            f"{turns['undesirable']} / {turns['unknown']}",
            f"AUTO_APPROVED pairs: {paired['AUTO_APPROVED']}",
            f"Pair conflicts: {paired['conflicts']}",
            f"Unpaired positive / negative: {unpaired['positive']} / {unpaired['negative']}",
            f"Pairability rate: {report['pairability_rate']:.6f}", "",
        ))
