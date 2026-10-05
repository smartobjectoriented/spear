"""A model's reading of a constraint is a candidate; only the source decides.

On real protocol code the post-check model claimed violations that were not
there, a second call to the same model confirmed them, and the confirmed
findings drove a repair. A constraint is now SATISFIED or VIOLATED only where
a deterministic predicate reads it off the final source; everything else the
model says is kept as a candidate finding and decides nothing.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import normative_constraints as nc
import normative_predicates as np
from normative_constraints import Constraint, NormativeConstraintSet


def constraint(text, cid="C1", modality="SHALL", condition="", resolved=True):
    return Constraint(cid, f"Rule {cid}", f"i-{cid}", f"s-{cid}", 1, modality,
                      f"Rule {cid}: {text}", condition=condition, resolved=resolved,
                      unresolved_reason="" if resolved else "needs review")


COUNT = constraint("The count field shall contain exactly four entries.")
MODE = constraint("The mode field shall be 2 when flag X is set.", "C2",
                  condition="when flag X is set")
OPTIONAL = constraint("The metadata field may be absent.", "C3", modality="MAY")
SUBTYPE = constraint("Only one of the AckV, AckX, or AckS bits shall be set to 1.", "C4")


def packet(*items):
    return NormativeConstraintSet("ncs-test", "STD", "1", "o", tuple(items))


def answer(*items):
    return json.dumps({"constraints": [
        {"id": cid, "status": status, "reason": reason,
         "evidence": [{"file": "a.py", "line": line, "text": text}]}
        for cid, status, reason, line, text in items]})


def adjudicate(items, files, check):
    chosen = packet(*items)
    return {item.constraint_id: item for item in nc.adjudicate(
        chosen, nc.read_candidates(chosen, check, files), files)}


class Predicates(unittest.TestCase):
    def decide(self, item, source, path="a.py"):
        return np.evaluate(item, {path: source})

    def test_an_exact_count_is_read_off_a_literal(self):
        held = self.decide(COUNT, "count = [0, 0, 0, 0]\n")
        broken = self.decide(COUNT, "count = [0, 0, 0]\n")

        self.assertEqual((held.predicate, held.observed, held.holds), (np.EXACT_COUNT, "4", True))
        self.assertEqual((broken.observed, broken.holds), ("3", False))
        self.assertEqual(broken.facts[0].line, 1)
        self.assertEqual(broken.facts[0].symbol, "count")

    def test_a_value_under_its_condition(self):
        self.assertTrue(self.decide(MODE, "mode = 2 if flag_x else 1\n").holds)
        self.assertFalse(self.decide(MODE, "mode = 1 if flag_x else 2\n").holds)
        self.assertTrue(self.decide(MODE, "void f(void) {\n  h->mode = flag_x ? 2 : 1;\n}\n",
                                    "a.c").holds)

    def test_anything_a_literal_cannot_show_is_undecided(self):
        for source in ("count = [0] * 4\n",                      # an expression
                       "count = [0, 0, 0]\ncount.append(0)\n",   # changed later
                       "count = [0, 0, 0]\ncount += [0]\n",
                       "count = [0, 0, 0]\nfill(count)\n",       # handed to a callee
                       "a, count = [1], [0, 0, 0]\n",
                       "count = [0, 0, 0, 0]\ncount = [0]\n",    # two values
                       "def f(x):\n    if x:\n        count = [0, 0, 0]\n",
                       "build(count=[0, 0, 0])\n",
                       "x = 1\n"):                               # nothing assigned
            with self.subTest(source=source):
                self.assertIsNone(self.decide(COUNT, source))

    def test_a_c_element_write_or_address_is_undecided(self):
        for source in ("int count[4] = {0, 0, 0, 0};\ncount[2] = 5;\n",
                       "int count[3] = {0, 0, 0};\nmemset(&count, 0, 3);\n",
                       "void f(int x) {\n  if (x)\n    count = 3;\n}\n"):
            with self.subTest(source=source):
                self.assertIsNone(self.decide(COUNT, source, "a.c"))

    def test_a_comment_or_a_pure_read_does_not_stand_in_the_way(self):
        source = ("/* count = 3 entries */\nstatic int count[4] = {0, 0, 0, 0};\n"
                  "int n(void) { return sizeof(count) / sizeof(count[0]); }\n")
        python = "count = [0, 0, 0, 0]  # count = 3 in older readers\nassert len(count) == 4\n"

        self.assertIsNone(self.decide(COUNT, source, "a.c"))   # count[0] indexes it
        self.assertTrue(self.decide(COUNT, python).holds)

    def test_a_subject_naming_several_things_is_never_decided(self):
        self.assertIsNone(self.decide(SUBTYPE, "acks = 1;\n", "a.c"))

    def test_only_unconditional_requirements_are_decided(self):
        self.assertIsNone(self.decide(OPTIONAL, "metadata = None\n"))
        self.assertIsNone(self.decide(constraint("The count field shall not contain exactly "
                                                 "four entries."), "count = [1, 2, 3]\n"))
        self.assertIsNone(self.decide(constraint("The mode field shall be 2 or 3."),
                                      "mode = 2\n"))
        self.assertIsNone(self.decide(constraint("The count field shall contain exactly "
                                                 "four entries.", resolved=False),
                                      "count = [0, 0, 0]\n"))


class Adjudication(unittest.TestCase):
    def test_a_model_only_violation_is_not_demonstrated_and_kept(self):
        files = {"a.py": "def f():\n    count = [0] * 4\n"}
        statuses = adjudicate([COUNT], files, answer(
            ("C1", "VIOLATED", "a list built by repetition is not four entries", 2,
             "count = [0] * 4")))
        status = statuses["C1"]

        self.assertEqual(status.status, nc.NOT_DEMONSTRATED)
        self.assertEqual(status.authority, "none")
        self.assertEqual(status.candidate.candidate, nc.POSSIBLE_VIOLATION)
        self.assertEqual(status.candidate.facts[0].excerpt, "count = [0] * 4")
        self.assertIn("possible violation", status.reason)

    def test_a_model_only_satisfaction_is_not_demonstrated_either(self):
        files = {"a.py": "count = [0] * 4\n"}
        statuses = adjudicate([COUNT], files, answer(
            ("C1", "SATISFIED", "four entries", 1, "count = [0] * 4")))

        self.assertEqual(statuses["C1"].status, nc.NOT_DEMONSTRATED)
        self.assertEqual(statuses["C1"].candidate.candidate, nc.POSSIBLE_SATISFACTION)

    def test_the_source_decides_whatever_the_model_said(self):
        files = {"a.py": "count = [0, 0, 0, 0]\n"}
        statuses = adjudicate([COUNT], files, answer(
            ("C1", "VIOLATED", "plausible but wrong", 1, "count = [0, 0, 0, 0]")))

        self.assertEqual(statuses["C1"].status, nc.SATISFIED)
        self.assertEqual(statuses["C1"].authority, "predicate:EXACT_COUNT")
        self.assertEqual(statuses["C1"].candidate.candidate, nc.POSSIBLE_VIOLATION)

    def test_a_deterministic_violation_is_authoritative_without_the_model(self):
        files = {"a.py": "count = [0, 0, 0]\n"}
        statuses = adjudicate([COUNT], files, "{}")

        self.assertEqual(statuses["C1"].status, nc.VIOLATED)
        self.assertIn("states 4, the final source gives 3", statuses["C1"].reason)
        self.assertEqual(statuses["C1"].candidate.candidate, nc.NO_FINDING)

    def test_an_unresolved_constraint_is_ambiguous(self):
        item = constraint("The count field shall contain exactly four entries.",
                          resolved=False)
        statuses = adjudicate([item], {"a.py": "count = [0, 0, 0]\n"}, "{}")

        self.assertEqual(statuses["C1"].status, nc.AMBIGUOUS)
        self.assertEqual(statuses["C1"].authority, "packet")

    def test_a_quote_the_source_lacks_is_no_fact(self):
        statuses = adjudicate([SUBTYPE], {"a.py": "x = 1\n"}, answer(
            ("C4", "VIOLATED", "two bits", 1, "set_bits(AckX | AckEr)")))

        self.assertEqual(statuses["C4"].candidate.facts, ())
        self.assertEqual(statuses["C4"].candidate.ungrounded, 1)
        self.assertEqual(statuses["C4"].status, nc.NOT_DEMONSTRATED)


class Authority(unittest.TestCase):
    def status(self, cid, state, authority):
        return nc.ConstraintStatus(cid, state, authority=authority)

    def test_only_an_established_violation_of_a_requirement_is_repairable(self):
        chosen = packet(COUNT, OPTIONAL, SUBTYPE)
        statuses = (self.status("C1", nc.VIOLATED, "predicate:EXACT_COUNT"),
                    self.status("C3", nc.VIOLATED, "predicate:VALUE_EQUALS"),
                    self.status("C4", nc.VIOLATED, "none"))

        self.assertEqual([item.constraint_id for item in nc.repairable(chosen, statuses)],
                         ["C1"])
        self.assertEqual(nc.repairable(chosen, statuses[1:]), [])

    def test_compliance_needs_every_requirement_established(self):
        chosen = packet(COUNT, MODE, OPTIONAL)
        satisfied = self.status("C1", nc.SATISFIED, "predicate:EXACT_COUNT")
        mode = self.status("C2", nc.SATISFIED, "predicate:CONDITIONAL_VALUE")
        optional = self.status("C3", nc.NOT_DEMONSTRATED, "none")

        self.assertEqual(nc.normative_status(chosen, (satisfied, mode, optional)), nc.SATISFIED)
        self.assertEqual(nc.normative_status(chosen, (satisfied, optional)), nc.NOT_DEMONSTRATED)
        self.assertEqual(nc.normative_status(chosen, (
            satisfied, self.status("C2", nc.NOT_DEMONSTRATED, "none"), optional)),
            nc.NOT_DEMONSTRATED)
        self.assertEqual(nc.normative_status(chosen, (
            satisfied, self.status("C2", nc.AMBIGUOUS, "packet"))), nc.AMBIGUOUS)
        self.assertEqual(nc.normative_status(chosen, (
            self.status("C1", nc.VIOLATED, "predicate:EXACT_COUNT"), mode)), nc.VIOLATED)

    def test_a_may_alone_establishes_nothing(self):
        self.assertEqual(nc.normative_status(packet(OPTIONAL), (
            self.status("C3", nc.SATISFIED, "predicate:VALUE_EQUALS"),)), nc.NOT_DEMONSTRATED)


if __name__ == "__main__":
    unittest.main()
