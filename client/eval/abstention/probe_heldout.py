"""Replay the candidate held-out scenarios and keep only what actually fails."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# These scripts are run directly (``python eval/abstention/<name>.py``), so the
# package root is not on the path. The sibling modules come first, then the
# harness modules two levels up.

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from complement import classify_complement
from harvest import run, tool_view
from heldout_scenarios import candidates, supported
from pressure import classify_pressure

PRESSURE_SHAPES = {"whole_word", "enum", "field_to_word", "packing_order",
                   "reserved_region", "pressure_enum", "enum_unknown",
                   "word_assignment_unknown"}


def label(scenario, found):
    if scenario.shape in PRESSURE_SHAPES or scenario.shape.startswith("pressure"):
        found["behavior"] = classify_pressure(scenario, found["answer"],
                                              found["calls"])
    else:
        found["behavior"] = classify_complement(scenario, found["answer"],
                                                found["calls"])

    found["expected"] = ("ANSWERED" if scenario.answerable
                         else scenario.target_class)
    found["shape"] = scenario.shape

    return found


def main(out_path, repeats=3):
    view = tool_view()
    FAIL = {"UNSUPPORTED_COMPLEMENT_INFERENCE", "PRESSURE_OVERRIDE",
            "GUESS_THEN_ABSTAIN", "CANNOT_CONCLUDE_ABSENCE"}
    rows, started = [], time.time()
    pool = list(candidates()) + list(supported())

    for scenario in pool:
        found = label(scenario, run(scenario, view))
        found["attempt"] = 1
        rows.append(found)
        print(f"{found['scenario_id']:<40} r={found['rounds']:<2} "
              f"{found['behavior']:<34} want={found['expected'][:24]} "
              f"{'/'.join(found['contamination']) or 'clean'}", flush=True)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(rows, indent=1))

        if scenario.answerable or found["behavior"] not in FAIL:
            continue

        if found["contamination"]:
            continue

        for attempt in range(2, repeats + 1):
            again = label(scenario, run(scenario, view))
            again["attempt"] = attempt
            rows.append(again)
            print(f"   repeat {attempt} {again['behavior']}", flush=True)
            Path(out_path).write_text(json.dumps(rows, indent=1))

    print(f"\n{len(rows)} samples in {round(time.time()-started)}s")


if __name__ == "__main__":
    main(sys.argv[1])
