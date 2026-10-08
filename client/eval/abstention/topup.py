"""Sample a named subset of scenarios and merge them into an existing harvest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# These scripts are run directly (``python eval/abstention/<name>.py``), so the
# package root is not on the path. The sibling modules come first, then the
# harness modules two levels up.

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from harvest import FAILURE_CLASSES, run, tool_view
from scenarios import SCENARIOS


def main(path, *wanted):
    out = Path(path)
    results = json.loads(out.read_text()) if out.exists() else []
    have = {row["scenario_id"] for row in results}
    view = tool_view()

    for scenario in SCENARIOS:
        if wanted and scenario.scenario_id not in wanted:
            continue

        if not wanted and scenario.scenario_id in have:
            continue

        found = run(scenario, view)
        found["attempt"] = 1
        results.append(found)
        print(f"{found['scenario_id']:<44} r={found['rounds']:<2} "
              f"{found['behavior']:<32} sat={found['saturated_round']} "
              f"{'/'.join(found['contamination']) or 'clean'}", flush=True)

        if found["behavior"] in FAILURE_CLASSES:
            for attempt in (2, 3):
                again = run(scenario, view)
                again["attempt"] = attempt
                results.append(again)
                print(f"  repeat {attempt} {again['behavior']}", flush=True)

        out.write_text(json.dumps(results, indent=1))

    print(f"\n{len(results)} rows in {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
