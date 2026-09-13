"""Assemble the trainer records, choosing a real losing answer where one exists."""

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

from authored import FALLBACK, OVER_ABSTENTION
from build import prompt_messages, tool_view
from pairs import PAIRS
from validate import _ABSTENTION_TELLS, ANSWERING_CLASS


def _shows_the_defect(item, answer):
    """Did the model actually fail the way this item is built to elicit?"""

    if not answer:
        return False

    lowered = answer.lower()

    if item["class"] == ANSWERING_CLASS:
        # The losing side here is over-abstention, which the model rarely
        # produces on evidence it can read.

        return any(tell in lowered for tell in _ABSTENTION_TELLS)

    # For the three discipline classes, failing means never marking the gap.

    return not any(tell in lowered for tell in _ABSTENTION_TELLS)


def assemble(samples):
    by_id = {item["pair_id"]: item for item in samples}
    tools = tool_view()
    records = []

    for item in PAIRS:
        sampled = by_id.get(item["pair_id"], {})
        answer = sampled.get("sampled_answer") or ""

        if _shows_the_defect(item, answer):
            rejected, origin = answer, "sampled"
        elif item["pair_id"] in OVER_ABSTENTION:
            rejected, origin = OVER_ABSTENTION[item["pair_id"]], "authored"
        else:
            rejected, origin = FALLBACK[item["class"]], "authored_fallback"

        # Where the model kept looking, its own settled context is the honest
        # prompt: the decision we are training is the one taken after the
        # looking stopped paying, not the one taken on first sight.

        prompt = sampled.get("prompt") or prompt_messages(item)
        records.append({
            "pair_id": item["pair_id"], "class": item["class"],
            "source": "synthetic", "rejected_origin": origin,
            "licensed_content": False,
            "extra_evidence_rounds": sampled.get("extra_rounds", 0),
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": item["chosen"]}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "tools": tools,
        })

    return records


def main():
    samples_path = HERE / "synthetic_samples.json"
    samples = json.loads(samples_path.read_text()) if samples_path.exists() else []
    records = assemble(samples)
    out = HERE / "preference_train.jsonl"

    with out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    counts = {}

    for record in records:
        counts[(record["class"], record["rejected_origin"])] = counts.get(
            (record["class"], record["rejected_origin"]), 0) + 1

    for key in sorted(counts):
        print(f"{key[0]:<36} {key[1]:<18} {counts[key]}")

    print(f"\n{len(records)} records -> {out}")


if __name__ == "__main__":
    main()
