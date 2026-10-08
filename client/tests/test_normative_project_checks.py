"""A project check bound to a provision is authoritative evidence; nothing else is.

End to end through the MIXED orchestration on the synthetic standard, with a
count requirement no source predicate can read (`[0] * 4`) and a project check
bound to it in the project's own configuration.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest

from tests.test_agent_loop import call, turn
from tests.test_mixed_orchestration import (CODING, FIXED, OPAQUE, MixedTurn, fix, opaque,
                                            verdicts)
from runtime.tracing import EventType

from normative import normative_constraints as nc

CHECK = (f"{sys.executable} -B -c \"import record; "
         f"assert len(record.build_header(True)['count']) == 4\"")
WRONG = OPAQUE.replace("[0] * 4", "[0] * 3")
NOTHING = turn('{"constraints": []}')


def project(command=CHECK, semantics="PASS_AND_FAIL_DECISIVE", provisions=("Rule 4.2.1-2",),
            standard="SYNTH-MIXED", **extra):
    return {"normative_checks": [{
        "id": "count-four", "standard": standard, "revision": "M-1",
        "provisions": list(provisions), "command": command,
        "evidence": {"kind": "test", "success": "exit_zero", "semantics": semantics}}],
        **extra}


class ProjectChecks(MixedTurn):
    def runner(self, command):
        self.commands.append(command)
        done = subprocess.run(["bash", "-c", command], cwd=self.repo, capture_output=True,
                              text=True)
        code = done.returncode
        status = "PASSED" if code == 0 else "ERROR" if code in (126, 127) else "FAILED"

        return status, code, done.stdout + done.stderr

    def setUp(self):
        super().setUp()
        self.commands = []

    def statuses(self, record):
        return {item.constraint_id: item for item in record.statuses}

    def test_a_a_passing_bound_check_satisfies_its_constraint(self):
        record = self.run_mixed(fix(content=OPAQUE) + [NOTHING], project=project(),
                                runner=self.runner)
        c2 = self.statuses(record)["C2"]

        self.assertEqual((c2.status, c2.authority), (nc.SATISFIED, "check:count-four"))
        self.assertEqual(self.statuses(record)["C1"].status, nc.NOT_DEMONSTRATED)
        self.assertEqual(record.check_runs[0].status, "PASSED")
        self.assertEqual(record.check_runs[0].source_epoch, record.checks[0]["fingerprint"])

    def test_b_a_decisive_failure_violates_and_repairs_once(self):
        record = self.run_mixed(fix(content=WRONG) + [
            NOTHING,
            turn("", call("p2", "write_file", path="record.py", content=OPAQUE)),
            turn("Repaired."), NOTHING], project=project(), runner=self.runner)
        first = {item.constraint_id: item.status for item in record.checks[0]["statuses"]}
        repair = self.backend.raw_calls[3]

        self.assertEqual(first["C2"], nc.VIOLATED)
        self.assertTrue(record.repaired)
        self.assertIn("project check count-four", repair["messages"][-1]["content"])
        for word in ("POSSIBLE_", "candidate", "advisory", "check:", "predicate:"):
            self.assertNotIn(word, json.dumps(repair["messages"]))
        self.assertEqual(sorted(t["function"]["name"] for t in repair["tools"]), sorted(CODING))
        self.assertEqual([run.status for run in record.check_runs], ["FAILED", "PASSED"])
        self.assertNotEqual(record.check_runs[0].source_epoch, record.check_runs[1].source_epoch)
        self.assertEqual(self.statuses(record)["C2"].status, nc.SATISFIED)

    def test_b_a_violation_the_repair_does_not_fix_stays(self):
        record = self.run_mixed(fix(content=WRONG) + [
            NOTHING, turn("I could not change it."), NOTHING],
            project=project(), runner=self.runner)

        self.assertTrue(record.repaired)
        self.assertEqual(record.verdict, "NOT COMPLIANT")
        self.assertEqual(len(record.check_runs), 2)

    def test_b_a_failure_that_is_not_decisive_proves_nothing(self):
        record = self.run_mixed(fix(content=WRONG) + [NOTHING],
                                project=project(semantics="PASS_ESTABLISHES_SATISFIED"),
                                runner=self.runner)
        c2 = self.statuses(record)["C2"]

        self.assertEqual(c2.status, nc.NOT_DEMONSTRATED)
        self.assertIn("does not make a failure decisive", c2.reason)
        self.assertFalse(record.repaired)

    def test_c_a_check_that_cannot_run_is_not_demonstrated(self):
        record = self.run_mixed(fix(content=WRONG) + [NOTHING],
                                project=project(command="no-such-checker --count"),
                                runner=self.runner)

        self.assertEqual(record.check_runs[0].status, "ERROR")
        self.assertEqual(self.statuses(record)["C2"].status, nc.NOT_DEMONSTRATED)
        self.assertFalse(record.repaired)

    def test_c_no_runner_is_not_run(self):
        record = self.run_mixed(fix(content=OPAQUE) + [NOTHING], project=project())

        self.assertEqual(record.check_runs[0].status, "NOT_RUN")
        self.assertEqual(self.statuses(record)["C2"].status, nc.NOT_DEMONSTRATED)

    def test_d_a_pass_on_a_source_the_check_then_changed_is_stale(self):
        edit = CHECK + " && echo '# touched' >> record.py"
        record = self.run_mixed(fix(content=OPAQUE) + [NOTHING], project=project(command=edit),
                                runner=self.runner)
        c2 = self.statuses(record)["C2"]

        self.assertEqual(record.check_runs[0].status, "PASSED")
        self.assertEqual(c2.status, nc.NOT_DEMONSTRATED)
        self.assertIn("stale", c2.reason)
        self.assertEqual(record.normative, nc.NOT_DEMONSTRATED)

    def test_e_an_unrelated_pass_has_no_effect(self):
        for spec in (project(provisions=("Rule 4.2.1-1",)),
                     project(standard="OTHER-STD")):
            with self.subTest(spec=spec["normative_checks"][0]["provisions"]):
                record = self.run_mixed([
                    turn("", call("p1", "write_file", path="record.py", content=OPAQUE)),
                    turn("", call("t1", "terminal", command="make")),
                    turn("Done. The tests pass."), NOTHING], project=spec, runner=self.runner)

                self.assertEqual(self.statuses(record)["C2"].status, nc.NOT_DEMONSTRATED)

    def test_f_a_compliance_claim_against_a_failing_bound_check(self):
        record = self.run_mixed(fix(content=WRONG) + [
            opaque(("C2", "SATISFIED")), turn("Nothing to change."),
            opaque(("C2", "SATISFIED"))], project=project(), runner=self.runner)

        self.assertEqual(self.statuses(record)["C2"].status, nc.VIOLATED)
        self.assertEqual(self.statuses(record)["C2"].candidate.candidate,
                         nc.POSSIBLE_SATISFACTION)
        self.assertEqual(record.verdict, "NOT COMPLIANT")

    def test_g_a_violation_claim_against_a_passing_bound_check(self):
        from tests.test_mixed_orchestration import CONFIRMED

        record = self.run_mixed(fix(content=OPAQUE) + [opaque(("C2", "VIOLATED")), CONFIRMED],
                                project=project(), runner=self.runner)
        c2 = self.statuses(record)["C2"]

        self.assertEqual(c2.status, nc.SATISFIED)
        self.assertEqual(c2.candidate.advisory, "CONFIRMED")
        self.assertFalse(record.repaired)

    def test_disagreeing_authorities_are_a_conflict_not_a_repair(self):
        failing = f"{sys.executable} -c \"raise SystemExit(1)\""
        record = self.run_mixed(fix(content=FIXED) + [NOTHING],
                                project=project(command=failing), runner=self.runner)
        c2 = self.statuses(record)["C2"]

        self.assertEqual(c2.status, nc.EVIDENCE_CONFLICT)
        self.assertEqual(c2.authority, "check:count-four+predicate:EXACT_COUNT")
        self.assertFalse(record.repaired)
        self.assertEqual(record.verdict, "EVIDENCE CONFLICT")
        self.assertTrue(self.events(EventType.NORMATIVE_EVIDENCE_CONFLICT))

    def test_the_check_and_its_binding_never_reach_the_model(self):
        self.run_mixed(fix(content=OPAQUE) + [NOTHING], project=project(), runner=self.runner)
        seen = json.dumps([call_["messages"] for call_ in self.backend.raw_calls])

        self.assertNotIn("count-four", seen)
        self.assertNotIn("assert len(record.build_header", seen)
        self.assertEqual(sorted(t["function"]["name"] for t in self.backend.raw_calls[0]["tools"]),
                         sorted(CODING))

    def test_the_bindings_and_runs_are_audited(self):
        self.run_mixed(fix(content=OPAQUE) + [NOTHING], project=project(), runner=self.runner)
        loaded = self.events(EventType.NORMATIVE_CHECK_BINDING_LOADED)[0].metadata
        finished = self.events(EventType.NORMATIVE_CHECK_FINISHED)[0].metadata

        self.assertEqual(loaded["binding_id"], "count-four")
        self.assertEqual(len(loaded["instances"]), 1)
        self.assertEqual((finished["status"], finished["exit_code"]), ("PASSED", 0))
        self.assertTrue(finished["source_epoch_matches"])
        self.assertTrue(self.events(EventType.NORMATIVE_CHECK_STARTED))

    def test_a_binding_to_a_provision_the_revision_lacks_is_reported(self):
        record = self.run_mixed(fix(content=OPAQUE) + [NOTHING],
                                project=project(provisions=("Rule 9.9.9-9",)),
                                runner=self.runner)
        problems = [event.metadata.get("problems") for event in
                    self.events(EventType.NORMATIVE_CHECK_BINDING_LOADED)]

        self.assertEqual(self.commands, [])
        self.assertTrue(any(items and "Rule 9.9.9-9" in items[0] for items in problems))
        self.assertEqual(self.statuses(record)["C2"].status, nc.NOT_DEMONSTRATED)


if __name__ == "__main__":
    unittest.main()
