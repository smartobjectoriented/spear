"""Sample the current base model on the reserved evaluation scenarios."""

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
from heldout import HELDOUT
from pressure import classify_pressure


def label(scenario, found):
    if scenario.gap_type in ("whole_word", "enum", "field_to_word",
                             "packing_order", "reserved_region"):
        found["behavior"] = classify_pressure(scenario, found["answer"],
                                              found["calls"])
    else:
        found["behavior"] = classify_complement(scenario, found["answer"],
                                                found["calls"])

    found["expected"] = ("ANSWERED" if scenario.answerable
                         else scenario.target_class)

    return found


def main(out_path):
    view = tool_view()
    rows, started = [], time.time()

    for scenario in HELDOUT:
        found = label(scenario, run(scenario, view))
        rows.append(found)
        print(f"{found['scenario_id']:<44} r={found['rounds']:<2} "
              f"got={found['behavior']:<34} want={found['expected']}",
              flush=True)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(rows, indent=1))

    print(f"\n{len(rows)} scenarios in {round(time.time()-started)}s")


if __name__ == "__main__":
    main(sys.argv[1])
