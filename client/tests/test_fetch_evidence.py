"""What a fetch showed the model is what the guards read.

Seen in use: `standard.fetch` is shown as rendered text, not JSON, and the
policy parsed it as JSON. Every fetch then counted for nothing: a field the
model quoted from the fetched register's neighbours ("RS, bits [47:44]
RangeSelector") was judged to come from nowhere and removed by the repair,
and a miscopied position beside it went unchecked.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import synthetic_standard
from standard import standard_answer_policy as sap
from standard.standard_tools import render_fetch_for_model, shown_fetch

MODE = "WidgetMode, bits [7:4] Widget Mode. Selects how the reply is encoded."
LATE = "WidgetTail, bits [31:28] Widget Tail. Never shown to the model."


def fetched():
    found = synthetic_standard.fetch(neighbours=())
    found["unit"]["source_id"] = "std-" + "1" * 32
    found["neighbors"] = [
        {"section": synthetic_standard.SECTION, "source_id": "std-" + "2" * 32,
         "text": MODE},
        {"section": synthetic_standard.SECTION, "source_id": "std-" + "3" * 32,
         "text": LATE},
    ]

    return found


def rendered(found, budget=None):
    if budget is None:
        budget = len(render_fetch_for_model(found, budget=100000)) - len(LATE)

    return render_fetch_for_model(found, budget=budget)


class Shown(unittest.TestCase):
    def test_only_the_neighbours_the_rendering_kept_are_evidence(self):
        found = fetched()
        text = rendered(found)

        self.assertIn("WidgetMode", text)
        self.assertNotIn("WidgetTail", text)
        self.assertEqual([item["text"] for item in shown_fetch(found, text)["neighbors"]],
                         [MODE])

    def test_a_unit_cut_short_counts_as_far_as_it_was_shown(self):
        found = fetched()
        found["unit"]["text"] = "A" * 400 + " WidgetLate appears only past the cut."
        text = render_fetch_for_model(found, budget=600)

        self.assertIn("[INCOMPLETE", text)
        self.assertNotIn("WidgetLate", shown_fetch(found, text)["unit"]["text"])


class Policy(unittest.TestCase):
    """A register description: a document that declares no provisions."""

    def policy(self, **options):
        policy = sap.policy_for(synthetic_standard.BINDING, "Which fields?", **options)
        found = fetched()
        found["unit"]["text"] = "Purpose Generates widgets for the current state."
        policy.observe_tool_result("standard.fetch", rendered(found),
                                   structured=json.dumps(found))

        return policy

    def test_a_rendered_fetch_reaches_every_ledger(self):
        policy = self.policy()

        self.assertIn("widgetmode", policy.claim_evidence.identifiers())
        self.assertNotIn("widgettail", policy.claim_evidence.identifiers())
        self.assertIn(synthetic_standard.SECTION, policy.clauses.sections)

    def test_the_repair_rewrites_the_answer_not_the_guards_notes(self):
        seen = []

        def ask(text):
            seen.append(text)

            return ("WidgetMode, bits [7:4] selects the encoding.\n"
                    "The standard requires a barrier after the write, as §9.9 says.")

        policy = self.policy(repair_ask=ask)
        answer = ("WidgetMode, bits [7:4] selects the encoding, and WidgetBogus "
                  "picks the target.\n"
                  "The standard requires a barrier after the write, as §9.9 says.")

        shown = policy.finalize(answer, stopped_by=sap.STOPPED_BY_MODEL)

        self.assertTrue(seen, "the repair was never attempted")
        self.assertNotIn("No requirement of", seen[0])
        self.assertIn("requires a barrier", seen[0])
        self.assertIn(MODE[:40], seen[0])
        self.assertNotIn("WidgetBogus", shown)
        self.assertIn("No requirement of", shown)
        self.assertTrue(policy.precedence_fired)


if __name__ == "__main__":
    unittest.main()
