"""Serializable benchmark configurations; production defaults are unchanged."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json


@dataclass(frozen=True)
class AblationConfig:
    name: str = "full"
    explorer: bool = True
    reviewer: bool = True
    planning: bool = True
    compaction: bool = True
    selected_memory: bool = True
    role_aware_tools: bool = True
    review_repair: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


CONFIGURATIONS = {
    "full": AblationConfig(),
    "production-default": AblationConfig(
        "production-default", explorer=False, reviewer=False,
        review_repair=False,
    ),
    "no-explorer": AblationConfig("no-explorer", explorer=False),
    "no-reviewer": AblationConfig("no-reviewer", reviewer=False),
    "no-planning": AblationConfig("no-planning", planning=False),
    "no-compaction": AblationConfig("no-compaction", compaction=False),
    "full-tools": AblationConfig("full-tools", role_aware_tools=False),
    "no-memory": AblationConfig("no-memory", selected_memory=False),
    "no-repair": AblationConfig("no-repair", review_repair=False),
}


def get_configuration(name: str) -> AblationConfig:
    try:
        return CONFIGURATIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark configuration: {name}") from exc
