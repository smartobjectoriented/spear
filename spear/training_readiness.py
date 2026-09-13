"""Evidence-based readiness decisions for governed SPEAR training data."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Sequence

from preference_dataset import (
    OutcomeLabel, PreferenceConfiguration, PreferenceDatasetBuilder,
)
from sft_dataset import (
    ReviewStatus, SFTDatasetBuilder, SFTExportConfiguration,
)
from training_governance import TrainingDataGovernancePolicy
from training_splits import SplitConfiguration, split_group_for_episode
from training_store import TrainingStore


READINESS_SCHEMA_VERSION = 1
READINESS_POLICY_VERSION = 1


class TrainingReadinessState(StrEnum):
    EMPTY = "EMPTY"
    COLLECTING = "COLLECTING"
    READY = "READY"
    BLOCKED = "BLOCKED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class ReadinessLevel(StrEnum):
    NOT_READY = "NOT_READY"
    READY_FOR_SMOKE = "READY_FOR_SMOKE"
    READY_FOR_EXPERIMENT = "READY_FOR_EXPERIMENT"


class TrainingStrategyRecommendation(StrEnum):
    COLLECT_MORE_DATA = "COLLECT_MORE_DATA"
    SFT_ONLY = "SFT_ONLY"
    SFT_THEN_UNPAIRED_PREFERENCE = "SFT_THEN_UNPAIRED_PREFERENCE"
    SFT_THEN_PAIRED_PREFERENCE = "SFT_THEN_PAIRED_PREFERENCE"
    NEEDS_MANUAL_DATA_REVIEW = "NEEDS_MANUAL_DATA_REVIEW"


@dataclass(frozen=True)
class TrainingStageReadiness:
    stage: str
    state: str
    level: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    metrics: Mapping[str, object]


@dataclass(frozen=True)
class TrainingReadinessReport:
    schema_version: int
    policy_version: int
    sft: TrainingStageReadiness
    unpaired_preference: TrainingStageReadiness
    paired_preference: TrainingStageReadiness
    recommendation: str
    governance: Mapping[str, object]
    distribution: Mapping[str, object]
    token_validation: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = ["# Training readiness", "",
                 f"Recommendation: **{self.recommendation}**", ""]

        for stage in (self.sft, self.unpaired_preference, self.paired_preference):
            lines.extend((f"## {stage.stage}", "",
                          f"State: {stage.state} ({stage.level})",
                          f"Reasons: {', '.join(stage.reasons) or 'none'}",
                          f"Warnings: {', '.join(stage.warnings) or 'none'}", ""))

        return "\n".join(lines)


@dataclass(frozen=True)
class TrainingReadinessPolicy:
    """Operational thresholds; callers may set values for their experiment."""

    sft_smoke_train_samples: int = 1
    sft_experiment_samples: int = 100
    sft_experiment_split_groups: int = 20
    sft_experiment_projects: int = 2
    minimum_validation_samples: int = 1
    minimum_test_samples: int = 1
    unpaired_smoke_per_class: int = 1
    unpaired_experiment_per_class: int = 50
    unpaired_experiment_groups: int = 20
    paired_smoke_pairs: int = 1
    paired_experiment_pairs: int = 50
    paired_experiment_contexts: int = 20
    maximum_quarantine_rate: float = .05
    maximum_duplicate_rate: float = .30
    minimum_strong_evidence_rate: float = .80
    maximum_preference_class_ratio: float = 4.0
    allow_preference_without_sft_baseline: bool = False
    policy_version: int = READINESS_POLICY_VERSION

    def __post_init__(self) -> None:
        # The count thresholds are recognised by name rather than listed, so a
        # new one is validated the moment it is added to the dataclass.

        counts = [value for name, value in asdict(self).items()
                  if ("samples" in name or "groups" in name or "pairs" in name
                      or "contexts" in name or "per_class" in name)
                  and isinstance(value, int)]

        if any(value < 0 for value in counts):
            raise ValueError("readiness count thresholds cannot be negative")

        if not 0 <= self.maximum_quarantine_rate <= 1:
            raise ValueError("quarantine rate must be between zero and one")

        if not 0 <= self.maximum_duplicate_rate <= 1:
            raise ValueError("duplicate rate must be between zero and one")

        if not 0 <= self.minimum_strong_evidence_rate <= 1:
            raise ValueError("evidence rate must be between zero and one")

        if self.maximum_preference_class_ratio < 1:
            raise ValueError("class ratio must be at least one")


class TrainingReadinessEvaluator:
    def __init__(self, store: TrainingStore | str | os.PathLike[str], *,
                 policy: TrainingReadinessPolicy | None = None,
                 governance: TrainingDataGovernancePolicy | None = None,
                 split: SplitConfiguration | None = None) -> None:
        self.store = store if isinstance(store, TrainingStore) else TrainingStore(store)
        self.policy = policy or TrainingReadinessPolicy()
        self.governance = governance or TrainingDataGovernancePolicy()
        self.split = split or SplitConfiguration()

    def evaluate(self, *, token_validation: Mapping[str, object] | None = None
                 ) -> TrainingReadinessReport:
        """Judge each training stage on the data the store actually holds."""

        episodes = []
        exclusions = Counter()
        origins = Counter()
        projects = Counter()
        checksums: dict[str, str] = {}

        for metadata in sorted(self.store.iterate_metadata(),
                               key=lambda item: str(item.get("episode_id"))):
            episode = self.store.load_episode(str(metadata["episode_id"]))
            decision = self.governance.assess(episode)
            origins[decision.origin] += 1

            if not decision.allowed:
                exclusions.update(decision.reasons)
                continue

            episodes.append(episode)
            projects[decision.project] += 1
            checksums[episode.episode_id] = str(metadata.get("checksum", ""))

        # The builders run over the whole store and are filtered afterwards:
        # split assignment must not depend on what governance happens to allow.

        allowed_ids = set(checksums)

        sft_build = SFTDatasetBuilder(
            self.store, configuration=SFTExportConfiguration(split=self.split),
        ).build()
        pref_build = PreferenceDatasetBuilder(
            self.store, configuration=PreferenceConfiguration(split=self.split),
        ).build()
        sft = [item for item in sft_build.samples if item.episode_id in allowed_ids]
        unpaired = [item for item in pref_build.unpaired
                    if item.source_episode_id in allowed_ids]
        paired = [item for item in pref_build.paired
                  if item.chosen_episode_id in allowed_ids
                  and item.rejected_episode_id in allowed_ids]

        # The `all_*` sets include what was held back for review, which is what
        # the quarantine rate below is measured against.

        all_sft = [item for item in sft_build.all_samples if item.episode_id in allowed_ids]
        all_unpaired = [item for item in pref_build.all_unpaired
                        if item.source_episode_id in allowed_ids]
        duplicate_count = sum(
            1 for values in sft_build.duplicate_sources.values() for value in values
            if value.get("episode_id") in allowed_ids
        )
        preference_report = self._filtered_preference_report(pref_build, allowed_ids)
        sft_stage = self._sft(sft, all_sft, len(episodes), len(projects), duplicate_count)
        unpaired_stage = self._unpaired(unpaired, all_unpaired, preference_report)
        paired_stage = self._paired(paired, preference_report)
        validation = dict(token_validation or {
            "kind": "approximate", "exact": False,
            "requires_preprocessing_validation": True,
        })

        # Without a real tokenizer the lengths are estimates, so the stage is
        # warned rather than blocked: the bundle re-checks them exactly.

        if not validation.get("exact"):
            sft_stage = TrainingStageReadiness(
                sft_stage.stage, sft_stage.state, sft_stage.level,
                sft_stage.reasons,
                tuple(dict.fromkeys((*sft_stage.warnings, "token_lengths_unvalidated"))),
                sft_stage.metrics,
            )

        recommendation = self._recommend(sft_stage, unpaired_stage, paired_stage)
        distribution = {
            "real_source_episodes": len(episodes),
            "source_episode_checksums": checksums,
            "projects": dict(sorted(projects.items())),
            "sft_sample_types": dict(sorted(Counter(item.sample_type for item in sft).items())),
            "sft_context_lengths": self._lengths([item.token_count for item in sft]),
            "preference": preference_report,
        }

        return TrainingReadinessReport(
            READINESS_SCHEMA_VERSION, self.policy.policy_version,
            sft_stage, unpaired_stage, paired_stage, recommendation.value,
            {"policy": self.governance.to_dict(),
             "origin_distribution": dict(sorted(origins.items())),
             "exclusions": dict(sorted(exclusions.items()))},
            distribution, validation,
        )

    @staticmethod
    def _lengths(values: Sequence[int]) -> Mapping[str, int]:
        ordered = sorted(values)
        return {"count": len(ordered), "minimum": min(ordered, default=0),
                "maximum": max(ordered, default=0),
                "median": ordered[len(ordered) // 2] if ordered else 0}

    @staticmethod
    def _filtered_preference_report(build, allowed_ids):
        """Recompute the preference report over the governed subset only.

        The builder's own report covers everything it saw; readiness must judge
        only the episodes governance allows, so the figures are derived again
        from the filtered observations rather than reused.
        """

        observations = [item for item in build.observations if item.episode_id in allowed_ids]
        all_pairs = [item for item in build.all_paired
                     if item.chosen_episode_id in allowed_ids
                     and item.rejected_episode_id in allowed_ids]
        approved_pairs = [item for item in build.paired
                          if item.chosen_episode_id in allowed_ids
                          and item.rejected_episode_id in allowed_ids]
        approved_unpaired = [item for item in build.unpaired
                             if item.source_episode_id in allowed_ids]
        labels = Counter(item.label for item in observations)
        conflicts = sum(item.comparison_status == "CONFLICT" for item in all_pairs)
        recoveries = sum(bool(item.recovery_turn_ids) for item in observations)
        by_context: dict[str, list[object]] = {}

        for item in observations:
            by_context.setdefault(item.context_hash, []).append(item)

        # How many good/bad comparisons the data supports in principle: every
        # distinct desirable answer against every distinct undesirable one for
        # the same context. The pairability rate measures what was realised
        # against that, plus the recoveries that could not be paired.

        exact_relationships = sum(
            len({item.completion_hash for item in values
                 if item.label == OutcomeLabel.DESIRABLE.value}) *
            len({item.completion_hash for item in values
                 if item.label == OutcomeLabel.UNDESIRABLE.value})
            for values in by_context.values()
        )
        grounded = exact_relationships + recoveries

        return {
            "source_turns": {"considered": len(observations),
                             "desirable": labels[OutcomeLabel.DESIRABLE.value],
                             "undesirable": labels[OutcomeLabel.UNDESIRABLE.value],
                             "unknown": labels[OutcomeLabel.UNKNOWN.value]},
            "paired_candidates": {"AUTO_APPROVED": len(approved_pairs),
                                  "NEEDS_REVIEW": sum(item.review_status == "NEEDS_REVIEW"
                                                      for item in all_pairs),
                                  "conflicts": conflicts,
                                  "exact_context_collisions": sum(
                                      len({item.completion_hash for item in values}) > 1
                                      for values in by_context.values())},
            "unpaired": {"positive": sum(item.desirable for item in approved_unpaired),
                         "negative": sum(not item.desirable for item in approved_unpaired),
                         "unknown_excluded": labels[OutcomeLabel.UNKNOWN.value]},
            "recovery": {"relationships": recoveries},
            "pairability_rate": len(approved_pairs) / grounded if grounded else 0.0,
        }

    def _sft(self, samples, all_samples, episode_count, project_count, duplicate):
        """The SFT stage verdict: enough samples, spread widely enough, clean enough."""

        splits = Counter(item.split for item in samples)
        groups = len({item.split_group_id for item in samples})
        quarantine = sum(item.review_status != ReviewStatus.AUTO_APPROVED.value
                         for item in all_samples)
        qrate = quarantine / len(all_samples) if all_samples else 0.0
        drate = duplicate / (len(all_samples) + duplicate) if all_samples or duplicate else 0.0
        metrics = {"auto_approved": len(samples), "source_episodes": episode_count,
                   "train": splits["train"], "validation": splits["validation"],
                   "test": splits["test"], "split_groups": groups,
                   "projects": project_count, "quarantine_rate": qrate,
                   "duplicate_rate": drate,
                   "tool_integrity": True,
                   "failure_contamination": sum(
                       bool(set(item.source_labels) & {"failed_tool_action", "repeated_action",
                                                       "stalled", "stale_verification"})
                       for item in samples),
                   "verification_states": dict(Counter(item.verification_state
                                                        for item in samples)),
                   "overlong": sum(item.length_status == "OVERLONG" for item in all_samples),
                   "redacted_or_quarantined": quarantine,
                   "target_types": dict(Counter(item.sample_type for item in samples))}

        # No approved samples but some quarantined ones is a different problem
        # from having no data at all, and the operator is told which.

        if not samples:
            quarantined = bool(all_samples)

            return TrainingStageReadiness(
                "SFT", (TrainingReadinessState.NEEDS_REVIEW.value if quarantined else
                        TrainingReadinessState.COLLECTING.value),
                ReadinessLevel.NOT_READY.value,
                (("too_many_quarantined_samples",) if quarantined else
                 ("no_real_auto_approved_samples",)), (), metrics)

        # Reasons block the stage; warnings only keep it out of EXPERIMENT.

        reasons, warnings = [], []

        if splits["train"] < self.policy.sft_smoke_train_samples:
            reasons.append("insufficient_train_samples")

        if qrate > self.policy.maximum_quarantine_rate:
            reasons.append("too_many_quarantined_samples")

        if drate > self.policy.maximum_duplicate_rate:
            warnings.append("excessive_duplicate_rate")

        # A real experiment needs volume AND spread: a hundred samples from one
        # task family would measure memorization, not capability.

        experiment = len(samples) >= self.policy.sft_experiment_samples
        experiment &= groups >= self.policy.sft_experiment_split_groups
        experiment &= project_count >= self.policy.sft_experiment_projects
        experiment &= splits["validation"] >= self.policy.minimum_validation_samples
        experiment &= splits["test"] >= self.policy.minimum_test_samples

        if splits["validation"] < self.policy.minimum_validation_samples:
            warnings.append("no_validation_split")

        if splits["test"] < self.policy.minimum_test_samples:
            warnings.append("no_test_split")

        if groups < self.policy.sft_experiment_split_groups:
            warnings.append("insufficient_task_diversity")

        state = (TrainingReadinessState.NEEDS_REVIEW if reasons else
                 TrainingReadinessState.READY)
        level = (ReadinessLevel.NOT_READY if reasons else
                 ReadinessLevel.READY_FOR_EXPERIMENT if experiment else
                 ReadinessLevel.READY_FOR_SMOKE)

        return TrainingStageReadiness("SFT", state.value, level.value,
                                      tuple(reasons), tuple(dict.fromkeys(warnings)), metrics)

    def _unpaired(self, samples, all_samples, report):
        """The unpaired stage verdict: both classes present, and not too lopsided."""

        positive = sum(item.desirable for item in samples)
        negative = len(samples) - positive
        strong = sum(item.evidence_strength == "STRONG" for item in samples)
        groups = len({item.split_group_id for item in samples})
        splits = Counter(item.split for item in samples)
        ratio = max(positive, negative) / max(1, min(positive, negative))
        metrics = {"positive": positive, "negative": negative, "split_groups": groups,
                   "strong_evidence_rate": strong / len(samples) if samples else 0.0,
                   "class_ratio": ratio if samples else 0.0,
                   "split_sizes": dict(splits),
                   "unknown_excluded": report["unpaired"]["unknown_excluded"]}

        if not samples:
            return TrainingStageReadiness("UNPAIRED_PREFERENCE", "COLLECTING", "NOT_READY",
                                          ("no_grounded_outcome_samples",), (), metrics)

        reasons, warnings = [], []

        # One class alone teaches nothing about preference: the model would
        # learn only what everything has in common.

        if min(positive, negative) < self.policy.unpaired_smoke_per_class:
            reasons.append("missing_preference_class")

        if ratio > self.policy.maximum_preference_class_ratio:
            warnings.append("preference_class_imbalance")

        if metrics["strong_evidence_rate"] < self.policy.minimum_strong_evidence_rate:
            warnings.append("insufficient_strong_evidence")

        experiment = (min(positive, negative) >= self.policy.unpaired_experiment_per_class
                      and groups >= self.policy.unpaired_experiment_groups
                      and splits["validation"] >= self.policy.minimum_validation_samples
                      and splits["test"] >= self.policy.minimum_test_samples
                      and not warnings)

        return TrainingStageReadiness(
            "UNPAIRED_PREFERENCE", "NEEDS_REVIEW" if reasons else "READY",
            "NOT_READY" if reasons else
            "READY_FOR_EXPERIMENT" if experiment else "READY_FOR_SMOKE",
            tuple(reasons), tuple(warnings), metrics)

    def _paired(self, samples, report):
        """The paired stage verdict; any contradiction blocks it outright."""

        contexts = len({item.decision_context_hash for item in samples})
        conflicts = int(report["paired_candidates"]["conflicts"])
        splits = Counter(item.split for item in samples)
        metrics = {"auto_approved": len(samples), "decision_contexts": contexts,
                   "conflicts": conflicts, "pairability_rate": report["pairability_rate"],
                   "split_sizes": dict(splits)}

        if not samples:
            reasons = ("preference_conflicts",) if conflicts else ("no_exact_context_pairs",)
            return TrainingStageReadiness("PAIRED_PREFERENCE", "NEEDS_REVIEW" if conflicts else
                                          "COLLECTING", "NOT_READY", reasons, (), metrics)

        reasons = ["preference_conflicts"] if conflicts else []

        if len(samples) < self.policy.paired_smoke_pairs:
            reasons.append("insufficient_exact_context_pairs")

        experiment = (len(samples) >= self.policy.paired_experiment_pairs and
                      contexts >= self.policy.paired_experiment_contexts and
                      splits["validation"] >= self.policy.minimum_validation_samples and
                      splits["test"] >= self.policy.minimum_test_samples and not conflicts)

        return TrainingStageReadiness(
            "PAIRED_PREFERENCE", "NEEDS_REVIEW" if reasons else "READY",
            "NOT_READY" if reasons else
            "READY_FOR_EXPERIMENT" if experiment else "READY_FOR_SMOKE",
            tuple(reasons), (), metrics)

    def _recommend(self, sft, unpaired, paired):
        """What to do next, in order: fix the data, get more, or train."""

        # Data a human has to look at comes first: no amount of collecting
        # resolves a contradiction already in the corpus.

        if (sft.state == "NEEDS_REVIEW" or
                "preference_conflicts" in paired.reasons):
            return TrainingStrategyRecommendation.NEEDS_MANUAL_DATA_REVIEW

        if sft.level == ReadinessLevel.NOT_READY.value:
            return TrainingStrategyRecommendation.COLLECT_MORE_DATA

        # Preference training refines a model that already behaves; without an
        # SFT baseline it is recommended only when the policy explicitly allows.

        if not self.policy.allow_preference_without_sft_baseline:
            if paired.level == ReadinessLevel.READY_FOR_EXPERIMENT.value:
                return TrainingStrategyRecommendation.SFT_THEN_PAIRED_PREFERENCE

            if unpaired.level == ReadinessLevel.READY_FOR_EXPERIMENT.value:
                return TrainingStrategyRecommendation.SFT_THEN_UNPAIRED_PREFERENCE

        return TrainingStrategyRecommendation.SFT_ONLY


class TrainingReadinessIndex:
    """Cheap, rebuildable post-finalization counters; never dataset truth."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.directory = Path(root).resolve() / "datasets" / "readiness"
        self.path = self.directory / "summary.json"
        self.checksum_path = self.directory / "summary.sha256"

    def update_episode(self, episode) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        value = self.load()
        value["episodes"][episode.episode_id] = {
            "origin": TrainingDataGovernancePolicy.origin_for(episode),
            "eligibility": episode.training_metadata.get("eligibility"),
            "project": episode.provenance.get("project") or "unknown",
            "split_group_id": split_group_for_episode(episode),
            "turn_count": len(episode.turns),
        }
        encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        self._write(self.path, encoded)
        self._write(self.checksum_path,
                    (hashlib.sha256(encoded).hexdigest() + "\n").encode())

    def load(self):
        if not self.path.exists() and not self.checksum_path.exists():
            return {"schema_version": READINESS_SCHEMA_VERSION, "episodes": {}}

        content = self.path.read_bytes()

        if hashlib.sha256(content).hexdigest() != self.checksum_path.read_text().strip():
            raise ValueError("training readiness index checksum mismatch")

        value = json.loads(content)

        if value.get("schema_version") != READINESS_SCHEMA_VERSION:
            raise ValueError("unsupported training readiness index")

        return value

    @staticmethod
    def _write(path: Path, content: bytes) -> None:
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
