"""Run the pressure pool and label it with the strict definition."""

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

from harvest import contamination, run, tool_view
from pressure import PRESSURE_OVERRIDE, PRESSURE_SCENARIOS, classify_pressure


def main(out_path, repeats=3):
    view = tool_view()
    results, started = [], time.time()

    for scenario in PRESSURE_SCENARIOS:
        found = run(scenario, view)
        found["behavior"] = classify_pressure(scenario, found["answer"],
                                              found["calls"])
        found["attempt"] = 1
        found["gap_type"] = scenario.gap_type
        found["pressure_strength"] = scenario.pressure_strength
        results.append(found)
        print(f"{found['scenario_id']:<52} r={found['rounds']:<2} "
              f"{found['behavior']:<20} "
              f"{'/'.join(found['contamination']) or 'clean'}", flush=True)
        Path(out_path).write_text(json.dumps(results, indent=1))

        if found["behavior"] != PRESSURE_OVERRIDE:
            continue

        for attempt in range(2, repeats + 1):
            again = run(scenario, view)
            again["behavior"] = classify_pressure(scenario, again["answer"],
                                                  again["calls"])
            again["attempt"] = attempt
            again["gap_type"] = scenario.gap_type
            again["pressure_strength"] = scenario.pressure_strength
            results.append(again)
            print(f"  repeat {attempt} {again['behavior']}", flush=True)
            Path(out_path).write_text(json.dumps(results, indent=1))

    print(f"\n{len(results)} samples in {round(time.time()-started)}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "harvest_pressure.json")
