"""The deterministic normative boundary, on the production AgentRuntime path.

The guards were validated through an evaluation harness, which proves what they
do and not that anything runs them. These tests drive the real runtime loop with
a scripted backend and the real normative tools, and assert on
`AgentResult.final_response` -- the string a user actually receives. A turn with
no bound standard must come through the loop untouched; a bound turn must have
the whole validated sequence applied to it.
"""

from __future__ import annotations

import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import standard_answer_policy
from agent_runtime import AgentContext, AgentRuntime, standard_policy_for
from context_engine import ContextEngine, ContextItem, ContextLayer, Freshness
from model_backend import (
    ConversationMessage, ModelToolCall, ModelTurn, StopReason, TextBlock,
    ToolDefinition,
)
from standard_answer_policy import StandardAnswerPolicy, policy_for
from tool_router import ToolResultEnvelope, ToolResultStatus
from tracing import TraceEmitter
from working_state import WorkingState

BINDING = {"standard_id": "ANSI-VITA-49.2", "revision": "2017-R2024"}
REAL_ID = "std-5287d23d6932688558e46a628f8aaec2"
OTHER_ID = "std-e1d69bac5d63695f703b9e201c727b5d"

STRUCTURE = json.dumps({
    "definition_id": "bfd-0ccd61a481e290d5",
    "structural_completeness": "STRUCTURALLY_INCOMPLETE",
    "citation_source_ids": [REAL_ID],
    "fields": [{"display_label": "Phase Offset, Radians", "msb": 15, "lsb": 0,
                "width": 16, "word_index": 1, "normative_source_id": REAL_ID}],
    "unresolved": [{"label": "Reserved"}],
    "words": [{"word_index": 1, "word_width": 32}]})
EMPTY_STRUCTURE = json.dumps({"standard_id": "X", "revision": "Y",
                              "structure_count": 0, "structures": []})
WORD_UNIT = json.dumps({
    "unit": {"source_id": REAL_ID, "section": "7.2.2", "page": 85,
             "content_type": "requirement",
             "text": "Rule 7.2.2-1: The Coarse Time shall be carried in word 1 "
                     "of the Epoch Field."},
    "citation": {"source_id": REAL_ID}})
SEARCH = json.dumps({"results": [{"source_id": REAL_ID, "section": "7.2.2",
                                  "page": 85, "snippet": "Coarse Time"}]})


class ScriptedBackend:
    """Replays a fixed sequence of model turns."""

    model = "scripted-local"
    max_tokens = 64

    def __init__(self, turns):
        self.turns = list(turns)

    def complete(self, **kwargs):
        if not self.turns:
            raise AssertionError("unexpected primary model call")

        return self.turns.pop(0)


def text_turn(text):
    return ModelTurn(text, (), StopReason.END_TURN)


def tool_turn(call_id, name, **arguments):
    return ModelTurn("", (ModelToolCall(call_id, name, arguments),),
                     StopReason.TOOL_USE)


class ScriptedTools:
    """The normative tools, answering from a fixed script."""

    def __init__(self, script):
        self.script = dict(script)
        self.seen = []

    def __call__(self, context, call_id, name, arguments, cache):
        self.seen.append(name)
        text = self.script.get(name, "{}")

        return ToolResultEnvelope(
            call_id, f"action-{len(self.seen)}", name, True,
            ToolResultStatus.OK, text, text, "standard", 0.0,
            len(text.encode("utf-8")))


def make_context(backend, *, binding=None, executor=None, question="q",
                 rounds=6):
    state = WorkingState.start("task_policy", "answer", max_model_rounds=rounds,
                               max_tool_actions=rounds * 2)

    return AgentContext(
        working_state=state, backend=backend, context_engine=ContextEngine(),
        trace=TraceEmitter(), system_prompt="SYSTEM",
        context_items=(ContextItem("system", ContextLayer.SYSTEM_RULES, "t",
                                   "SYSTEM", priority=100,
                                   freshness=Freshness.CURRENT,
                                   protected=True, inclusion_reason="t"),),
        conversation=[ConversationMessage("user", (TextBlock(question),))],
        tools=(ToolDefinition("standard.get_structure", "s",
                              {"type": "object"}),),
        tool_executor=executor or ScriptedTools({}),
        max_model_rounds=rounds, max_tool_actions=rounds * 2,
        context_limit=8192, output_reserve=64, safety_margin=16,
        standard_binding=binding)


class Activation(unittest.TestCase):
    """Only a bound turn gets the policy."""

    def test_a_turn_with_no_binding_gets_no_policy(self):
        self.assertIsNone(standard_policy_for(make_context(ScriptedBackend([]))))

    def test_a_bound_turn_gets_one(self):
        policy = standard_policy_for(
            make_context(ScriptedBackend([]), binding=BINDING))

        self.assertIsInstance(policy, StandardAnswerPolicy)
        self.assertEqual(policy.standard_id, "ANSI-VITA-49.2")

    def test_activation_never_reads_the_user_wording(self):
        context = make_context(
            ScriptedBackend([]),
            question="what bits does the Reserved field of the standard occupy?")

        self.assertIsNone(standard_policy_for(context))

    def test_the_question_comes_from_the_conversation(self):
        policy = standard_policy_for(make_context(
            ScriptedBackend([]), binding=BINDING, question="which word?"))

        self.assertEqual(policy.question, "which word?")


class NonStandardTurn(unittest.TestCase):
    """A: everything that is not a bound normative turn is untouched."""

    def test_a_plain_answer_is_returned_verbatim(self):
        answer = ("Reserved occupies bits 31..16 and the file is at "
                  "/home/user/spec.pdf, see std-deadbeefdeadbeefdeadbeefdead1234.")
        context = make_context(ScriptedBackend([text_turn(answer)]))
        result = AgentRuntime().run(context)

        self.assertEqual(result.final_response, answer)
        self.assertIsNone(context.standard_policy)

    def test_no_standard_evidence_is_read_for_it(self):
        tools = ScriptedTools({"bash": "ok"})
        context = make_context(ScriptedBackend([text_turn("done")]),
                               executor=tools)
        AgentRuntime().run(context)

        self.assertEqual(tools.seen, [])


class BoundTurn(unittest.TestCase):
    """The validated sequence, applied by the real runtime."""

    def run_bound(self, turns, script, question="Give the bit range."):
        tools = ScriptedTools(script)
        context = make_context(ScriptedBackend(turns), binding=BINDING,
                               executor=tools, question=question)
        result = AgentRuntime().run(context)

        return result, context.standard_policy, tools

    def test_b_a_correct_bound_answer_is_unchanged(self):
        answer = f"Phase Offset, Radians occupies bits 15..0 of word 1 ({REAL_ID})."
        result, policy, _ = self.run_bound(
            [tool_turn("c1", "standard.get_structure",
                       definition_id="bfd-0ccd61a481e290d5"),
             text_turn(answer)],
            {"standard.get_structure": STRUCTURE})

        self.assertEqual(result.final_response, answer)
        self.assertFalse(policy.guard_fired)
        self.assertFalse(policy.provenance_fired)

    def test_e_an_h4_style_placement_never_reaches_the_user(self):
        raw = ("struct s {\n    uint32_t phase_offset : 16;\n"
               "    uint32_t reserved : 16;  // bits 31..16, best guess\n};")
        result, policy, _ = self.run_bound(
            [tool_turn("c1", "standard.get_structure",
                       definition_id="bfd-0ccd61a481e290d5"),
             text_turn(raw)],
            {"standard.get_structure": STRUCTURE})

        self.assertIn("reserved : 16", policy.raw_answer)
        self.assertTrue(policy.guard_fired)
        self.assertNotIn("reserved : 16", result.final_response)
        self.assertIn("Phase Offset, Radians — bits 15..0", result.final_response)

    def test_g_fabricated_provenance_is_removed(self):
        raw = (f"Phase Offset, Radians occupies bits 15..0 of word 1 "
               f"([{REAL_ID}](file:///home/user/vita.pdf)), see also "
               f"std-deadbeefdeadbeefdeadbeefdeadbeef.")
        result, policy, _ = self.run_bound(
            [tool_turn("c1", "standard.get_structure",
                       definition_id="bfd-0ccd61a481e290d5"),
             text_turn(raw)],
            {"standard.get_structure": STRUCTURE})

        self.assertTrue(policy.provenance_fired)
        self.assertNotIn("file://", result.final_response)
        self.assertNotIn("deadbeef", result.final_response)
        self.assertIn(REAL_ID, result.final_response)

    def test_c_a_bound_turn_reads_the_standard_before_its_first_round(self):
        """The zero-tool answer this used to catch can no longer happen.

        It used to be caught at the boundary: the model answered "I cannot
        find that standard" having called nothing, the bootstrap searched for
        it, and the model was given the round back. The same retrieval now runs
        BEFORE the first round, so the standard is in context when the model
        first speaks and the boundary has nothing left to add -- re-issuing the
        identical search into a context that already holds its result buys a
        round and no evidence.

        What the boundary keeps is the case this cannot cover: an opening call
        that could not be executed leaves no standard tool in the log, and the
        bootstrap is reachable again.
        """
        result, policy, tools = self.run_bound(
            [text_turn(f"Coarse Time is carried in word 1 ({REAL_ID}).")],
            {"standard.search": SEARCH},
            question="In the bound standard: give the word number of Coarse Time.")

        self.assertTrue(policy.opening_fired)
        self.assertFalse(policy.bootstrap_fired)

        # Before the model said anything, and exactly once.
        self.assertEqual(tools.seen, ["standard.search"])
        self.assertIn("word 1", result.final_response)

    def test_d_an_empty_structure_registry_recovers_once(self):
        result, policy, tools = self.run_bound(
            [tool_turn("c1", "standard.get_structure"),
             text_turn("The standard defines no bit ranges at all."),
             text_turn(f"Coarse Time is carried in word 1 ({REAL_ID}).")],
            {"standard.get_structure": EMPTY_STRUCTURE,
             "standard.search": SEARCH},
            question="In the bound standard: give the word number of Coarse Time.")

        self.assertTrue(policy.recovery_fired)
        self.assertFalse(policy.bootstrap_fired)

        # The opening retrieval already asked this exact question, and the
        # recovery routes to the same call for a question carrying no
        # structure identifier. It is decided, then dropped as a repeat.
        self.assertEqual(tools.seen.count("standard.search"), 1)
        self.assertEqual(policy.suppressed, ["EMPTY_STRUCTURE_RECOVERY"])

    def test_i_a_bit_range_caveat_does_not_retract_a_word_number(self):
        answer = ("Coarse Time is carried in word 1 of the Epoch Field, but the "
                  "standard does not specify the bit ranges within that word.")
        result, policy, _ = self.run_bound(
            [tool_turn("c1", "standard.fetch", source_id=REAL_ID),
             text_turn(answer)],
            {"standard.fetch": WORD_UNIT})

        self.assertEqual(policy.evidence.word_fields["coarsetime"]["word_index"],
                         1)
        self.assertFalse(policy.guard_fired)
        self.assertEqual(result.final_response, answer)

    def test_a_denial_of_the_word_number_is_still_caught(self):
        answer = "The standard does not specify the word number for Coarse Time."
        result, policy, _ = self.run_bound(
            [tool_turn("c1", "standard.fetch", source_id=REAL_ID),
             text_turn(answer)],
            {"standard.fetch": WORD_UNIT})

        self.assertTrue(policy.guard_fired)
        self.assertIn("Coarse Time — word 1", result.final_response)


class TurnScope(unittest.TestCase):
    """J: policy state is per turn and never shared."""

    def test_two_turns_get_two_policies(self):
        first = standard_policy_for(make_context(ScriptedBackend([]),
                                                 binding=BINDING))
        second = standard_policy_for(make_context(ScriptedBackend([]),
                                                  binding=BINDING))

        self.assertIsNot(first, second)
        self.assertIsNot(first.evidence, second.evidence)
        self.assertIsNot(first.provenance, second.provenance)

    def test_evidence_does_not_leak_between_policies(self):
        first = policy_for(BINDING, "q")
        first.observe_tool_result("standard.get_structure", STRUCTURE)
        second = policy_for(BINDING, "q")

        self.assertTrue(first.evidence.established)
        self.assertFalse(second.evidence.established)


class Rendering(unittest.TestCase):
    """H: a loop that ends with nothing still says what it read."""

    def test_the_budget_exit_renders_a_grounded_conclusion(self):
        policy = policy_for(BINDING, "What bit positions does Reserved occupy?")
        policy.observe_tool_result("standard.get_structure", STRUCTURE)
        seen = policy.finalize(
            "", rounds=14, stopped_by=standard_answer_policy.STOPPED_BY_BUDGET)

        self.assertTrue(seen.strip())
        self.assertIn("does not establish the requested bit positions", seen)
        self.assertIn(REAL_ID, seen)

    def test_a_model_stop_with_no_answer_is_left_empty(self):
        policy = policy_for(BINDING, "What bit positions does Reserved occupy?")
        policy.observe_tool_result("standard.get_structure", STRUCTURE)

        self.assertEqual(
            policy.finalize("", rounds=2,
                            stopped_by=standard_answer_policy.STOPPED_BY_MODEL),
            "")


class NothingSurvivedTheGuards(unittest.TestCase):
    """An answer removed by the guards is not an answer never written."""

    def test_a_fully_removed_answer_says_so(self):
        """The guards are entitled to remove everything; silence is not.

        Stubbed at the guard rather than fed an input that happens to
        trigger it: what is under test is the policy's contract when a
        guard leaves nothing, not any one guard's judgement.
        """
        policy = policy_for(BINDING, "What bit positions does Reserved occupy?")
        policy.observe_tool_result("standard.get_structure", STRUCTURE)

        with patch.object(standard_answer_policy.provenance_guard, "guard",
                          return_value=("", ["invented source"],
                                        ["a sentence"], True)):
            seen = policy.finalize("Reserved occupies bits 31 down to 24.",
                                   rounds=6,
                                   stopped_by=standard_answer_policy.STOPPED_BY_MODEL)

        self.assertTrue(seen.strip())
        self.assertIn("none of it survived the guards", seen)
        self.assertIn("provenance guard", seen)

    def test_an_answer_that_survives_is_untouched_by_the_note(self):
        policy = policy_for(BINDING, "What bit positions does Reserved occupy?")
        policy.observe_tool_result("standard.get_structure", STRUCTURE)
        seen = policy.finalize("Nothing was retrieved about that.", rounds=2,
                               stopped_by=standard_answer_policy.STOPPED_BY_MODEL)

        self.assertNotIn("none of it survived the guards", seen)

    def test_a_model_that_said_nothing_still_renders_nothing(self):
        """The note is about removal, not about silence."""
        policy = policy_for(BINDING, "What bit positions does Reserved occupy?")
        policy.observe_tool_result("standard.get_structure", STRUCTURE)

        self.assertEqual(
            policy.finalize("", rounds=2,
                            stopped_by=standard_answer_policy.STOPPED_BY_MODEL),
            "")


class SingleImplementation(unittest.TestCase):
    """K: the evaluation runners own no policy of their own."""

    def test_the_runners_contain_no_decision_logic(self):
        for name in ("eval/abstention/harvest.py", "eval/modeluse/harness.py"):
            source = (ROOT / name).read_text("utf-8")

            with self.subTest(name=name):
                for tell in ("should_bootstrap", "should_recover",
                             "should_terminate(", "bounded_absence",
                             "EvidenceLedger(", "ProvenanceLedger("):
                    self.assertNotIn(tell, source)

    def test_the_model_sees_only_the_four_normative_tools(self):
        self.assertEqual(set(StandardAnswerPolicy.STANDARD_TOOLS),
                         {"standard.search", "standard.fetch",
                          "standard.cite", "standard.get_structure"})


if __name__ == "__main__":
    unittest.main()
