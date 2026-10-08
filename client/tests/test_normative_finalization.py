"""A turn investigates with the tools it is offered, then finalizes with none.

FINALIZING is said to the model when the window closes. A tool call the model
writes as text is never run and never the answer; it earns one more request
for the answer, and a second one ends the turn saying it has none. A MIXED
pre-pass that ends that way leaves an empty packet that says why -- distinct
from one that answered and cited nothing, and from one the guards withheld.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from normative import mixed_orchestration as mo
from runtime.agent_context import RuntimeTerminalReason
from runtime.agent_finalization import (
    DONE, FAILED, FINALIZATION_DEMAND, FINALIZATION_RETRY,
)
from runtime.agent_runtime import AgentRuntime
from models.model_backend import ModelToolCall, ModelTurn, StopReason, TextBlock
from tests.test_agent_runtime import (GroundedToolExecutor, ScriptedBackend, make_context,
                                      text_turn, tool_turn)
from tests.test_mixed_orchestration import PREPASS_ANSWER, SATISFIED_ALL, MixedTurn, fix
from runtime.tracing import EventType

WRITTEN = ("<tool_call>\n<function=standard.search>\n<parameter=query>\nacknowledge "
           "packet\n</parameter>\n</function>\n</tool_call>")


def written_turn(text=WRITTEN):
    return ModelTurn(text, (), StopReason.END_TURN)


def harness_messages(context):
    return [block.text for message in context.conversation if message.role == "user"
            for block in message.content if isinstance(block, TextBlock)]


def run(turns, rounds=5):
    backend, executor = ScriptedBackend(turns), GroundedToolExecutor()
    context = make_context(backend, executor=executor, rounds=rounds)
    result = AgentRuntime().run(context)

    return result, context, backend, executor


class Investigating(unittest.TestCase):
    def test_tools_then_a_final_answer(self):
        # 15
        result, context, backend, executor = run([
            tool_turn("a", "bash", command="pwd"),
            tool_turn("b", "bash", command="ls"),
            text_turn("The answer, from what was read.")])

        self.assertEqual(result.final_response, "The answer, from what was read.")
        self.assertEqual(len(executor.calls), 2)
        self.assertTrue(all(call["use_tools"] for call in backend.calls))
        self.assertNotIn(FINALIZATION_DEMAND, harness_messages(context))
        self.assertEqual(context.finalization.state, DONE)

    def test_a_draft_beside_a_real_call_is_not_the_answer(self):
        # 19
        draft = ModelTurn("Rule 4-1 seems to govern this; checking its sibling.",
                          (ModelToolCall("a", "bash", {"command": "pwd"}),),
                          StopReason.TOOL_USE)
        result, _, _, executor = run([draft, text_turn("Final: Rule 4-1 and 4-2.")])

        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(result.final_response, "Final: Rule 4-1 and 4-2.")

    def test_a_written_call_while_investigating_runs_nothing(self):
        result, context, backend, executor = run([written_turn(), text_turn("The answer.")])

        self.assertEqual(executor.calls, [])
        self.assertEqual(result.final_response, "The answer.")
        self.assertFalse(backend.calls[-1]["use_tools"])
        self.assertIn(FINALIZATION_RETRY, harness_messages(context))


class Finalizing(unittest.TestCase):
    def test_the_closed_window_is_announced(self):
        # 12: the round budget closes the window on the last round.
        result, context, backend, _ = run([
            tool_turn("a", "bash", command="pwd"), text_turn("Answer.")], rounds=2)

        self.assertEqual([call["use_tools"] for call in backend.calls], [True, False])
        self.assertIn(FINALIZATION_DEMAND, harness_messages(context))
        self.assertEqual(result.final_response, "Answer.")

    def test_a_written_call_earns_one_retry_and_is_never_run(self):
        # 16
        result, context, backend, executor = run([
            tool_turn("a", "bash", command="pwd"), written_turn(),
            text_turn("Grounded answer.")], rounds=2)

        self.assertEqual(len(executor.calls), 1)
        self.assertEqual([call["use_tools"] for call in backend.calls], [True, False, False])
        self.assertEqual(result.final_response, "Grounded answer.")
        self.assertEqual(context.finalization.retries, 1)
        self.assertEqual(context.finalization.state, DONE)

    def test_a_second_written_call_ends_the_turn_without_an_answer(self):
        # 17
        result, context, backend, executor = run([
            tool_turn("a", "bash", command="pwd"), written_turn(), written_turn(),
            text_turn("never asked for")], rounds=2)

        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(len(backend.calls), 3, "no third request")
        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.STALLED)
        self.assertTrue(result.final_response.startswith("ANSWER NOT COMPLETED"))
        self.assertNotIn("<tool_call>", result.final_response)
        self.assertEqual(context.finalization.state, FAILED)

    def test_a_structured_call_with_no_tool_offered_is_not_run(self):
        result, _, _, executor = run([
            tool_turn("a", "bash", command="pwd"), tool_turn("b", "bash", command="ls"),
            text_turn("Answer.")], rounds=2)

        self.assertEqual([name for _, name, _ in executor.calls], ["bash"])
        self.assertEqual(result.final_response, "Answer.")

    def test_saying_the_evidence_is_insufficient_is_an_answer(self):
        # 18
        insufficient = "The evidence read does not establish this; no provision states it."
        result, context, _, _ = run([
            tool_turn("a", "bash", command="pwd"), text_turn(insufficient)], rounds=2)

        self.assertEqual(result.final_response, insufficient)
        self.assertEqual(context.finalization.retries, 0)


class ThePrePassSaysWhyItsPacketIsEmpty(MixedTurn):
    def packet_event(self):
        return self.events(EventType.NORMATIVE_CONSTRAINT_SET_CREATED)[-1].metadata

    def test_a_failed_pass_is_not_an_empty_answer(self):
        # 17 and 24B, through the orchestration
        record = self.run_mixed(fix() + [SATISFIED_ALL], legacy=[
            tool_turn("s", "standard.search", query="header"), written_turn(),
            written_turn()] + [text_turn("never asked for")] * 9)

        self.assertEqual(record.prepass, mo.PREPASS_FAILED)
        self.assertEqual(record.packet.constraints, ())
        self.assertIn("did not complete", record.packet.coverage_reason)
        self.assertEqual(self.packet_event()["prepass"], mo.PREPASS_FAILED)
        self.assertTrue(self.events(EventType.NORMATIVE_PREPASS_FAILED))
        self.assertTrue(self.events(EventType.NORMATIVE_FINALIZATION_RETRY))

    def test_an_answer_citing_nothing_is_a_valid_empty_packet(self):
        # 18 and 24A
        record = self.run_mixed(fix() + [SATISFIED_ALL], legacy=[text_turn(
            "The retrieved sections do not establish a requirement for this change.")])

        self.assertEqual(record.prepass, mo.PREPASS_ANSWERED)
        self.assertEqual(record.packet.coverage_reason, "the normative pass cited no provision")
        self.assertFalse(self.events(EventType.NORMATIVE_PREPASS_FAILED))

    def test_a_withheld_answer_is_neither(self):
        # 24C: a name nothing grounds, and nothing cited around it.
        record = self.run_mixed(fix() + [SATISFIED_ALL], legacy=[text_turn(
            "The header must carry the HdrZeta field.")])

        self.assertEqual(record.prepass, mo.PREPASS_WITHHELD)
        self.assertIn("withheld", record.packet.coverage_reason)
        self.assertFalse(self.events(EventType.NORMATIVE_PREPASS_FAILED))

    def test_a_cited_answer_is_unchanged(self):
        record = self.run_mixed(fix() + [SATISFIED_ALL], legacy=[text_turn(PREPASS_ANSWER)])

        self.assertEqual(record.prepass, mo.PREPASS_ANSWERED)
        self.assertEqual(len(record.packet.constraints), 3)
        self.assertTrue(self.events(EventType.NORMATIVE_INVESTIGATION_STARTED))

    def test_the_audit_is_not_shown_to_the_model(self):
        self.run_mixed(fix() + [SATISFIED_ALL], legacy=[
            tool_turn("s", "standard.search", query="header"), written_turn(),
            written_turn()] + [text_turn("never asked for")] * 9)
        shown = repr(self.backend.calls) + repr(self.backend.raw_calls)

        for name in ("normative_finalization_retry", "normative_prepass_failed",
                     "normative_investigation_started", "normative_finalization_started"):
            self.assertNotIn(name, shown)


if __name__ == "__main__":
    unittest.main()
