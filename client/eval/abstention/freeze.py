"""Write the held-out manifest and hash everything that must not move again.

The hashes are the pre-training reference: if any of these files changes after
a run starts, the run's numbers stop meaning what they claimed to mean.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# These scripts are run directly (``python eval/abstention/<name>.py``), so the
# package root is not on the path. The sibling modules come first, then the
# harness modules two levels up.

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from heldout import HELDOUT
from scenarios import serialize

REAL_HELDOUT = ("F1", "F2", "H4")
REGRESSION_ANCHORS = ("C1", "D4", "E3", "H1", "F3", "G1", "G2")


def manifest(baselines):
    """One row per held-out scenario, with the behaviour the base model showed."""
    rows = []

    for scenario in HELDOUT:
        rows.append({
            "scenario_id": scenario.scenario_id, "shape": scenario.shape,
            "semantic_family": scenario.family_key,
            "expected": "ANSWERED" if scenario.answerable
                        else scenario.target_class,
            "answerable": scenario.answerable,
            "baseline_behavior": baselines.get(scenario.scenario_id),
            "definition": json.loads(serialize(scenario)),
        })

    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(state_dir, baseline_path):
    state = Path(state_dir)
    baselines = {}

    for row in json.loads(Path(baseline_path).read_text()):
        if row.get("attempt", 1) == 1:
            baselines[row["scenario_id"]] = row["behavior"]

    out = state / "heldout_eval.jsonl"

    with out.open("w") as handle:
        for row in manifest(baselines):
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"{len(HELDOUT)} held-out scenarios -> {out}")
    print("\n=== frozen artifact hashes (SHA-256) ===")

    for name in ("preference_train.jsonl", "preference_eval.jsonl",
                 "heldout_eval.jsonl", "preference_train_trainer.jsonl",
                 "preference_eval_trainer.jsonl"):
        path = state / name

        if path.exists():
            print(f"  {name:<34} {sha256(path)}")

    print(f"\nreal held-out cases      {', '.join(REAL_HELDOUT)}")
    print(f"regression anchors       {', '.join(REGRESSION_ANCHORS)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
