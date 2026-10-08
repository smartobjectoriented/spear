"""Compact JSON/Markdown aggregation for benchmark manifests."""
from __future__ import annotations
import json
import statistics
from pathlib import Path
from typing import Iterable


def aggregate(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    groups = {}

    for row in rows:
        key = (row.get("configuration", {}).get("name", "unknown"), row.get("task_class", "unknown"))
        groups.setdefault(key, []).append(row)

    # `task_success` is the name; `success` is what older manifests carry.

    for row in rows:
        row.setdefault("task_success", row.get("success"))

    result = {"n": len(rows), "successes": sum(bool(row.get("success")) for row in rows),
              "functionally_correct": sum(bool(row.get("functionally_correct")) for row in rows),
              "groups": {}}
    result["success_rate"] = result["successes"] / len(rows) if rows else 0.0

    # Reported beside the success rate, never instead of it: the gap between
    # them is runs that produced the right file without doing the task.

    result["functional_rate"] = (result["functionally_correct"] / len(rows)
                                 if rows else 0.0)

    for (config, task_class), values in sorted(groups.items()):
        durations = [float(v["duration_seconds"]) for v in values
                     if isinstance(v.get("duration_seconds"), (int, float))]
        successes = sum(bool(v.get("success")) for v in values)
        correct = sum(bool(v.get("functionally_correct")) for v in values)
        result["groups"][f"{config}/{task_class}"] = {
            "n": len(values), "successes": successes,
            "success_rate": successes / len(values),
            "functionally_correct": correct,
            "functional_rate": correct / len(values),
            "mean_wall_seconds": statistics.mean(durations) if durations else None,
            "median_wall_seconds": statistics.median(durations) if durations else None,
        }

    return result


def write_report(rows: list[dict], output: str | Path) -> tuple[Path, Path]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows)
    json_path = output / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# SPEAR benchmark summary", "",
        f"Runs: {summary['n']}",
        f"Task successes: {summary['successes']} ({summary['success_rate']:.1%})",
        f"Functionally correct: {summary['functionally_correct']} "
        f"({summary['functional_rate']:.1%})", "",
        "A gap between the two is runs that left the right file behind "
        "without doing what the task asked.", "",
        "| configuration/task class | N | task success | functionally correct | median wall (s) |",
        "|---|---:|---:|---:|---:|",
    ]

    for key, value in summary["groups"].items():
        median = "—" if value["median_wall_seconds"] is None else f"{value['median_wall_seconds']:.2f}"
        lines.append(f"| {key} | {value['n']} | {value['success_rate']:.1%} | "
                     f"{value['functional_rate']:.1%} | {median} |")

    md_path = output / "summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return json_path, md_path

