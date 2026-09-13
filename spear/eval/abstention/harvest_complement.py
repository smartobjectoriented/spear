"""Run the complement pool and label it with the strict five-condition test."""

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

from complement import (
    COMPLEMENT_SCENARIOS, UNSUPPORTED_COMPLEMENT_INFERENCE, assignment,
    classify_complement,
)
from harvest import run, tool_view


def _label(scenario, found):
    found["behavior"] = classify_complement(scenario, found["answer"],
                                            found["calls"])
    claim = assignment(scenario, found["answer"])
    found["polarity"] = claim["position"] if claim else None
    found["gap_type"] = scenario.gap_type
    found["withheld_label"] = scenario.withheld_label

    return found


def main(out_path, repeats=3):
    view = tool_view()
    results, started = [], time.time()

    for scenario in COMPLEMENT_SCENARIOS:
        found = _label(scenario, run(scenario, view))
        found["attempt"] = 1
        results.append(found)
        print(f"{found['scenario_id']:<36} r={found['rounds']:<2} "
              f"{found['behavior']:<34} pol={found['polarity']} "
              f"{'/'.join(found['contamination']) or 'clean'}", flush=True)
        Path(out_path).write_text(json.dumps(results, indent=1))

        if found["behavior"] != UNSUPPORTED_COMPLEMENT_INFERENCE:
            continue

        for attempt in range(2, repeats + 1):
            again = _label(scenario, run(scenario, view))
            again["attempt"] = attempt
            results.append(again)
            print(f"  repeat {attempt} {again['behavior']} "
                  f"pol={again['polarity']}", flush=True)
            Path(out_path).write_text(json.dumps(results, indent=1))

    print(f"\n{len(results)} samples in {round(time.time()-started)}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "harvest_complement.json")
