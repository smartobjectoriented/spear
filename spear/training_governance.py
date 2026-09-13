"""Shared origin, project, and permanent-holdout governance for training data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from training_data import TrainingEpisode
from training_splits import split_group_for_episode


GOVERNANCE_POLICY_VERSION = 1


class DataOrigin(StrEnum):
    NORMAL_USAGE = "NORMAL_USAGE"
    BENCHMARK = "BENCHMARK"
    TEST_FIXTURE = "TEST_FIXTURE"
    SYNTHETIC = "SYNTHETIC"
    MANUAL_IMPORT = "MANUAL_IMPORT"
    LICENSED_STANDARD = "LICENSED_STANDARD"


@dataclass(frozen=True)
class GovernanceDecision:
    allowed: bool
    origin: str
    project: str
    split_group_id: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingDataGovernancePolicy:
    """Fail-closed policy used by readiness and frozen training bundles.

    Legacy FT0 episodes have no explicit origin field and are treated as normal
    usage. New capture always writes the field. Benchmark/test/synthetic data is
    never allowed by the production profile, even if project filters match.
    """

    include_projects: tuple[str, ...] = ()
    exclude_projects: tuple[str, ...] = ()
    permanent_holdout_groups: tuple[str, ...] = ()
    allowed_origins: tuple[str, ...] = (
        DataOrigin.NORMAL_USAGE.value,
        DataOrigin.MANUAL_IMPORT.value,
    )
    policy_version: int = GOVERNANCE_POLICY_VERSION

    def __post_init__(self) -> None:
        known = {item.value for item in DataOrigin}

        if any(item not in known for item in self.allowed_origins):
            raise ValueError("unknown allowed data origin")

        if set(self.include_projects) & set(self.exclude_projects):
            raise ValueError("project cannot be both included and excluded")

    @staticmethod
    def origin_for(episode: TrainingEpisode) -> str:
        """An episode's data origin, defaulting the way the docstring above says."""

        # A standard binding overrides whatever the episode claims: work done
        # against a licensed standard is licensed data whatever produced it.

        binding = episode.provenance.get("standard_binding")

        if isinstance(binding, Mapping):
            value = str(binding.get("data_origin") or
                        DataOrigin.LICENSED_STANDARD.value).upper()
            return value if value in {item.value for item in DataOrigin} else "UNKNOWN"

        # An unrecognised origin becomes UNKNOWN, which no profile allows: a
        # value nobody can classify must not fall through as usable data.

        value = str(episode.provenance.get("data_origin") or
                    DataOrigin.NORMAL_USAGE.value).upper()

        return value if value in {item.value for item in DataOrigin} else "UNKNOWN"

    def assess(self, episode: TrainingEpisode) -> GovernanceDecision:
        """Whether one episode may be used, and every reason it may not."""

        # All four checks run: the report names every ground for exclusion, not
        # just the first, so an operator sees the whole picture at once.

        origin = self.origin_for(episode)
        project = str(episode.provenance.get("project") or "unknown")
        group = split_group_for_episode(episode)
        reasons: list[str] = []

        if origin not in self.allowed_origins:
            reasons.append({
                DataOrigin.BENCHMARK.value: "benchmark_excluded",
                DataOrigin.TEST_FIXTURE.value: "test_fixture_excluded",
                DataOrigin.SYNTHETIC.value: "synthetic_excluded",
                DataOrigin.LICENSED_STANDARD.value: "licensed_standard_review_required",
            }.get(origin, "origin_not_allowed"))

        if self.include_projects and project not in self.include_projects:
            reasons.append("project_not_included")

        if project in self.exclude_projects:
            reasons.append("project_excluded")

        if group in self.permanent_holdout_groups:
            reasons.append("permanent_holdout")

        return GovernanceDecision(not reasons, origin, project, group,
                                  tuple(dict.fromkeys(reasons)))

    def to_dict(self) -> Mapping[str, object]:
        return {
            "policy_version": self.policy_version,
            "include_projects": list(self.include_projects),
            "exclude_projects": list(self.exclude_projects),
            "permanent_holdout_groups": list(self.permanent_holdout_groups),
            "allowed_origins": list(self.allowed_origins),
        }
