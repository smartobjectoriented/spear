"""Where the lifecycle actually stops a write, and where it stays out of the way.

The ledger's own tests decide when the gate should be open. These decide that
the production paths consult it: the router that dispatches every tool call,
the shell that can write without one, the exposure policy that decides what the
model is offered, and the round the harness narrows to the writing tools.

Four separate doors, and the failure mode is always the same one -- closing
three of them. A turn refused `edit_file` and handed a shell reaches for
`sed -i`; a turn told to plan and offered only `edit_file` can do neither.
"""

from __future__ import annotations

import io
import re
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent_runtime
import rag_chat
import work_phase
from agent_roles import AgentRole
from tool_exposure import ToolExposurePolicy
from tool_runtime import ExecutionMode
from work_phase import Phase, WorkPhaseLedger

CLAUSE = "4.2.1"
SOURCE = "src/link/handshake.c"

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

#: The refusals this gate issues, so a test can say "not one of ours" about
#: whatever else a sandboxless machine does to a shell command.
_GATE_REASONS = frozenset({
    work_phase.NO_AUTHORITY, work_phase.NO_IMPLEMENTATION,
    work_phase.NO_PLAN, work_phase.REVIEW_IS_READ_ONLY})


def ledger(*, investigated=False, planned=False):
    found = WorkPhaseLedger()
    found.engage(authority_bound=True, write_requested=True)

    if investigated or planned:
        found.observe_call(authority_keys={CLAUSE}, authority_units=1)
        found.observe_call(implementation_paths={SOURCE})

    if planned:
        found.record_plan([{
            "requirement": "the responder answers every accepted request",
            "requirement_evidence": f"Rule {CLAUSE}-1",
            "current_behaviour": "it answers only the first",
            "implementation_evidence": SOURCE,
            "gap": "later requests get no answer",
            "correction": "answer each request in handshake_accept()",
            "validation": "the project's own test suite",
        }])

    return found


def call(name, arguments, *, phase=None, mode=ExecutionMode.AUTO, cache=None):
    """One tool call through the production routing path."""
    context = SimpleNamespace(
        read_only=False, role="main", task_id="task_gate",
        trace=rag_chat.TRACE, checkpoint_manager=None, checkpoint=None,
        standard_binding={"standard_id": "SYNTHETIC-1"}, work_phase=phase,
        standard_source_ids_used=set(), session=None)
    screen = io.StringIO()

    with patch.object(rag_chat, "EXECUTION_MODE", mode), redirect_stdout(screen):
        envelope = rag_chat.route_tool_envelope(
            name, arguments, {} if cache is None else cache,
            trace=rag_chat.TRACE, agent_context=context)

    return envelope, screen.getvalue()


class TheRouterConsultsTheGate(unittest.TestCase):
    def test_an_edit_before_investigation_is_refused(self):
        envelope, _ = call("edit_file",
                           {"path": "a.c", "old_text": "x", "new_text": "y"},
                           phase=ledger())

        self.assertFalse(envelope.success)
        self.assertEqual(envelope.error_category, work_phase.NO_AUTHORITY)
        self.assertIn(work_phase.WRITE_BLOCKED, envelope.text)

    def test_an_edit_before_a_plan_is_refused(self):
        envelope, _ = call("write_file", {"path": "a.c", "content": "x"},
                           phase=ledger(investigated=True))

        self.assertFalse(envelope.success)
        self.assertEqual(envelope.error_category, work_phase.NO_PLAN)

    def test_the_refusal_is_not_silent(self):
        """A withheld tool reads as incapacity; a refusal reads as a task."""
        _, screen = call("edit_file",
                         {"path": "a.c", "old_text": "x", "new_text": "y"},
                         phase=ledger())
        printed = [_ANSI.sub("", line).strip()
                   for line in screen.splitlines() if "refused:" in line]

        self.assertEqual(printed, [
            "⎿  edit_file refused: the authoritative source has not been "
            "read yet"])

    def test_the_handler_never_ran(self):
        with self.subTest("the file is untouched"):
            target = Path("test_work_phase_gate_should_not_exist.txt")
            envelope, _ = call("write_file",
                               {"path": str(target), "content": "x"},
                               phase=ledger())

            self.assertFalse(target.exists())
            self.assertFalse(envelope.mutation)

    def test_a_reading_tool_is_never_gated(self):
        """Whatever else stops a shell here, it is not this gate."""
        envelope, _ = call("bash", {"command": "ls"}, phase=ledger())

        self.assertNotIn(envelope.error_category, _GATE_REASONS)

    def test_no_gate_at_all_leaves_a_write_alone(self):
        target = Path("test_work_phase_gate_ungoverned.txt")

        try:
            envelope, _ = call("write_file",
                               {"path": str(target), "content": "x"},
                               phase=None)

            self.assertTrue(envelope.success, envelope.text)
        finally:
            target.unlink(missing_ok=True)

    def test_a_disengaged_ledger_gates_nothing(self):
        found = WorkPhaseLedger()
        found.engage(authority_bound=True, write_requested=False)
        target = Path("test_work_phase_gate_disengaged.txt")

        try:
            envelope, _ = call("write_file",
                               {"path": str(target), "content": "x"},
                               phase=found)

            self.assertTrue(envelope.success, envelope.text)
        finally:
            target.unlink(missing_ok=True)


class TheShellCannotRouteAroundTheGate(unittest.TestCase):
    """Refusing edit_file and handing over a shell refuses nothing."""

    def test_a_writing_command_is_refused_while_the_gate_is_shut(self):
        envelope, _ = call("bash", {"command": "sed -i s/a/b/ " + SOURCE},
                           phase=ledger())

        self.assertEqual(envelope.status.value, "denied")

    def test_the_refusal_says_what_would_open_the_gate(self):
        envelope, _ = call("bash", {"command": "sed -i s/a/b/ " + SOURCE},
                           phase=ledger(investigated=True))

        self.assertIn(work_phase.WRITE_BLOCKED, envelope.text)

    def test_a_reading_command_is_not_this_gates_business(self):
        envelope, _ = call("bash", {"command": "ls"}, phase=ledger())

        self.assertNotIn(envelope.error_category, _GATE_REASONS)


class TheGateOpensOnAnAcceptedPlan(unittest.TestCase):
    def test_the_decision_flips(self):
        found = ledger(investigated=True)

        self.assertFalse(found.may_write().allowed)

        found.record_plan([{
            "requirement": "r", "requirement_evidence": f"§{CLAUSE}",
            "current_behaviour": "b", "implementation_evidence": SOURCE,
            "gap": "g", "correction": "c", "validation": "v"}])

        self.assertTrue(found.may_write().allowed)

    def test_a_write_then_reaches_the_handler(self):
        target = Path("test_work_phase_gate_written.txt")

        try:
            envelope, _ = call("write_file",
                               {"path": str(target), "content": "x"},
                               phase=ledger(planned=True))

            self.assertTrue(envelope.success, envelope.text)
            self.assertTrue(target.exists())
        finally:
            target.unlink(missing_ok=True)


class ThePlanToolIsOfferedOnTheShapeThatNeedsIt(unittest.TestCase):
    def setUp(self):
        self.registry = rag_chat.TOOL_REGISTRY
        self.policy = ToolExposurePolicy()

    def view(self, objective, **kwargs):
        return self.policy.select(self.registry, AgentRole.MAIN,
                                  objective=objective, **kwargs)

    def test_offered_when_a_bound_turn_asks_for_a_change(self):
        view = self.view("make the changes needed to comply with the spec",
                         standard_bound=True)

        self.assertIn("plan_change", view.names)

    def test_withheld_on_a_pure_question_about_the_document(self):
        view = self.view("what does the spec require of the responder?",
                         standard_bound=True)

        self.assertNotIn("plan_change", view.names)

    def test_withheld_on_a_code_task_with_no_bound_document(self):
        view = self.view("fix the crash in handshake_accept",
                         standard_bound=False)

        self.assertNotIn("plan_change", view.names)

    def test_withheld_when_the_user_forbade_changes(self):
        view = self.view("review the code against the spec without editing "
                         "anything", standard_bound=True)

        self.assertNotIn("plan_change", view.names)
        self.assertNotIn("edit_file", view.names)

    def test_a_plural_verb_still_reads_as_a_change(self):
        """`\\bchange\\b` does not match "changes", and that was the bug."""
        self.assertTrue(ToolExposurePolicy._MUTATION.search(
            "check the code base and do the necessary changes"))

    def test_the_write_request_floor_reads_an_adjective(self):
        self.assertTrue(agent_runtime.is_write_request(
            "check the code base and do the necessary changes to be compliant"))

    def test_an_ordinary_question_is_still_not_a_write_request(self):
        self.assertFalse(agent_runtime.is_write_request(
            "how should the handshake be managed?"))


class ANarrowedRoundCanStillPlan(unittest.TestCase):
    """A round offered only `edit_file` while the gate is shut is a dead round."""

    def test_the_plan_tool_joins_the_writing_tools_while_shut(self):
        context = SimpleNamespace(work_phase=ledger(investigated=True))

        self.assertIn("plan_change", agent_runtime._write_round_tools(context))

    def test_it_drops_out_once_the_gate_is_open(self):
        context = SimpleNamespace(work_phase=ledger(planned=True))

        self.assertEqual(agent_runtime._write_round_tools(context),
                         agent_runtime._WRITE_TOOLS)

    def test_an_ungoverned_turn_sees_exactly_what_it_always_did(self):
        self.assertEqual(
            agent_runtime._write_round_tools(SimpleNamespace(work_phase=None)),
            agent_runtime._WRITE_TOOLS)


class ThePlanHandlerReportsWhatItRefused(unittest.TestCase):
    """One requirement per call, seven flat strings.

    The first shape was an array of objects. Measured on a real run, every one
    of forty-three calls was rejected before its arguments were parsed, and
    the turn concluded the tool was broken and went back to trying to write.
    """

    def test_an_accepted_item_says_the_gate_is_open(self):
        found = ledger(investigated=True)
        cache = {rag_chat.WORK_PHASE: found}
        envelope, _ = call("plan_change", {
            "requirement": "r", "requirement_evidence": f"§{CLAUSE}",
            "current_behaviour": "b", "implementation_evidence": SOURCE,
            "gap": "g", "correction": "c", "validation": "v"},
            phase=found, cache=cache)

        self.assertIn("1 planned change(s) now stand", envelope.text)
        self.assertIn("write gate is open", envelope.text)
        self.assertEqual(found.phase, Phase.EDIT)

    def test_a_refused_item_comes_back_with_its_reason(self):
        found = ledger(investigated=True)
        cache = {rag_chat.WORK_PHASE: found}
        envelope, _ = call("plan_change", {
            "requirement": "r", "requirement_evidence": "§9.9.9",
            "current_behaviour": "b", "implementation_evidence": SOURCE,
            "gap": "g", "correction": "c", "validation": "v"},
            phase=found, cache=cache)

        self.assertIn("is not among the evidence this session holds",
                      envelope.text)
        self.assertIn(work_phase.WRITE_BLOCKED, envelope.text)
        self.assertFalse(found.may_write().allowed)

    def test_a_vague_item_is_refused_by_field(self):
        found = ledger(investigated=True)
        cache = {rag_chat.WORK_PHASE: found}
        envelope, _ = call(
            "plan_change",
            {"requirement": "be compliant", "requirement_evidence": "",
             "current_behaviour": "", "implementation_evidence": "",
             "gap": "", "correction": "I will update the handshake handling",
             "validation": ""},
            phase=found, cache=cache)

        self.assertIn("missing", envelope.text)
        self.assertFalse(found.may_write().allowed)

    def test_the_schema_asks_for_nothing_nested(self):
        """What a model can reliably produce, not what the data looks like."""
        spec = rag_chat.TOOL_REGISTRY.get("plan_change")

        for name, field in spec.input_schema["properties"].items():
            with self.subTest(field=name):
                self.assertEqual(field["type"], "string")

        self.assertEqual(spec.input_schema["required"],
                         list(work_phase.GapItem.FIELDS))

    def test_planning_on_an_ungoverned_turn_says_so_and_blocks_nothing(self):
        envelope, _ = call("plan_change", {
            "requirement": "r", "requirement_evidence": "r",
            "current_behaviour": "b", "implementation_evidence": SOURCE,
            "gap": "g", "correction": "c", "validation": "v"}, phase=None)

        self.assertIn("no investigation gate", envelope.text)
        self.assertTrue(envelope.success)


if __name__ == "__main__":
    unittest.main()
