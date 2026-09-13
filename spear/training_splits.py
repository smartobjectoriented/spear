"""Shared deterministic task-family splits for every training-data product."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from training_data import TrainingEpisode


def _hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_task_text(value: object) -> str:
    """Reduce task text to what makes two tasks the same family.

    Paths, hex identifiers and redaction markers vary between runs of the same
    task; leaving them in would put near-identical work either side of the
    train/test boundary.
    """

    text = str(value or "").lower()
    text = re.sub(r"\[redacted:[^]]+\]", "[redacted]", text, flags=re.I)
    text = re.sub(r"(?:[a-z]:)?[/\\][\w./\\-]+", "<path>", text)
    text = re.sub(r"\b[0-9a-f]{8,}\b", "<id>", text)

    return " ".join(text.split())


@dataclass(frozen=True)
class SplitConfiguration:
    train: int = 90
    validation: int = 5
    test: int = 5

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)

        if any(not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("split percentages must be non-negative integers")

        if sum(values) != 100:
            raise ValueError("split percentages must total 100")

    def assign(self, split_group_id: str) -> str:
        """The split a group belongs to, by hash rather than by arrival order.

        Deriving it from the group id alone is what keeps the assignment stable
        as the store grows: adding episodes never moves the existing ones.
        """

        bucket = int(hashlib.sha256(split_group_id.encode()).hexdigest()[:16], 16) % 100

        if bucket < self.train:
            return "train"

        if bucket < self.train + self.validation:
            return "validation"

        return "test"


def split_group_for_episode(episode: TrainingEpisode) -> str:
    """The family an episode belongs to; everything in one family shares a split."""

    fixture = episode.task.get("fixture_id") or episode.task.get("task_identifier")
    basis = {
        "project": episode.provenance.get("project"),
        "workspace": episode.provenance.get("workspace_fingerprint"),
        "fixture": fixture,
        "objective": _normalize_task_text(episode.task.get("objective")),
        "acceptance": [_normalize_task_text(item)
                       for item in episode.task.get("acceptance_criteria", ())],
    }

    return "split_" + _hash(basis)[:32]
