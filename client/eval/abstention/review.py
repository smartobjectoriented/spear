"""Turn a harvest into the table a person has to sign off before pairs exist.

Counting behaviours is not the same as having looked at them. This prints one
row per sampled scenario with the facts that decide whether a sample can be a
losing side: what the model did, whether the defect stands alone, whether the
citations and the tool path are clean, and whether the same thing happened
again when asked again.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from harvest import FAILURE_CLASSES, NO_TOOL_USE


def latest(rows):
    """One row per scenario and attempt, preferring the most recent sampling."""
    found = {}

    for row in rows:
        found[(row["scenario_id"], row["attempt"])] = row

    return list(found.values())


def summarise(rows):
    rows = latest(rows)
    first = [row for row in rows if row["attempt"] == 1]
    repeats = collections.defaultdict(list)

    for row in rows:
        if row["attempt"] > 1:
            repeats[row["scenario_id"]].append(row["behavior"])

    table = []

    for row in sorted(first, key=lambda item: item["scenario_id"]):
        again = repeats.get(row["scenario_id"], [])
        table.append({
            "sample_id": row["scenario_id"],
            "class": row["behavior"],
            "family": row["family"],
            "shape": row["shape"],
            "isolated": not row["contamination"],
            "contamination": row["contamination"],
            "tool_path_clean": not any(item.startswith("unrecovered")
                                       for item in row["contamination"]),
            "citations_clean": not any(
                item.startswith(("fabricated", "invented_rule"))
                for item in row["contamination"]),
            "reproductions": 1 + sum(1 for item in again
                                     if item == row["behavior"]),
            "stable": bool(again) and all(item == row["behavior"] for item in again),
            "rounds": row["rounds"],
            "saturated_round": row["saturated_round"],
            "keep": (row["behavior"] in FAILURE_CLASSES
                     and not row["contamination"]
                     and bool(again)
                     and all(item == row["behavior"] for item in again)),
        })

    return table


def main(path):
    rows = json.loads(Path(path).read_text())
    table = summarise(rows)
    header = (f"{'sample_id':<44} {'class':<34} {'family':<14} "
              f"{'iso':<4} {'cit':<4} {'tool':<5} {'rep':<4} keep")
    print(header)
    print("-" * len(header))

    for item in table:
        print(f"{item['sample_id']:<44} {item['class']:<34} "
              f"{item['family']:<14} {'y' if item['isolated'] else 'n':<4} "
              f"{'y' if item['citations_clean'] else 'n':<4} "
              f"{'y' if item['tool_path_clean'] else 'n':<5} "
              f"{item['reproductions']:<4} {'KEEP' if item['keep'] else ''}")

    kept = [item for item in table if item["keep"]]
    print(f"\n{len(table)} scenarios, {len(kept)} keepable failures")
    print("by class :", dict(collections.Counter(item["class"] for item in kept)))
    print("families :", sorted({item["family"] for item in kept}))
    print("set aside as no-tool-use:",
          sum(1 for item in table if item["class"] == NO_TOOL_USE))

    return table


if __name__ == "__main__":
    main(sys.argv[1])
