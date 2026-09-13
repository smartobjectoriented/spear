"""What a turn owes, and what merely happened before it.

A bound session answered a question about reply selectors,
which established ten clauses. The next prompt was another question --
"are two separate replies explicitly required, and which clauses
say so?" -- and the model answered it, correctly, from the standard. The
harness then reported "8 of 10 established clauses not addressed", sent the
turn back to the code, and under `--auto` it went on to read, edit, build
and test the customer's source tree. Nothing had asked for any of that.

Clauses carried from an earlier turn are evidence available to this one.
They become obligations only when this turn is itself asked to change the
code -- the same test that decides whether they are put in front of the
model at all. The execution mode has no part in it: `--auto` says which
actions may proceed without being confirmed, never which actions are in
scope.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent_runtime
from agent_runtime import AgentRuntime, carried_obligations
from model_backend import (
    ConversationMessage, TextBlock, ToolDefinition,
)

from test_agent_runtime import (
    GroundedToolExecutor, RecordingObserver, ScriptedBackend, make_context,
    text_turn, tool_turn,
)

# The clauses an earlier turn of the session read. Ten of them, none of
# which this turn was asked to do anything about.
ESTABLISHED = ("4.1.1-1", "4.2.1-1", "4.2.1.4-1", "4.2.1.4-2", "4.2.1.1-1",
               "4.2.1.1-2", "4.2.1.1-3", "4.2.1.2-1", "4.5.1-1", "5.4.3-1")

QUESTION = ("Is it explicitly required by SYNTH-WIDGET-1 that a request "
            "naming both SelA and SelB results in two separate "
            "Widget Replies? Please distinguish what the standard "
            "explicitly states from what is inferred, and cite the relevant "
            "clauses.")

IMPLEMENTATION = ("Inspect the implementation against the clauses we "
                  "established and fix whatever does not comply.")

# An answer of the shape the question asks for: clauses cited, no source
# file named, because there is no source file in the answer to a question
# about what a standard says. Judged as a diff it accounts for nothing --
# which is exactly why the gate must not judge it as one.
NORMATIVE_ANSWER = (
    "Explicitly stated: Rule 4.2.1.1-2 requires exactly one of SelA/SelB/SelC "
    "per Widget Reply. Inferred: that a request naming both SelA and SelB "
    "therefore yields two replies follows from Rule 4.2.1.1-2 together with "
    "Observation 4.1.1-1; the standard does not state it in those words."
)

WRITE_TOOLS = ("edit_file", "write_file", "append_file")


def tools():
    """The view a bound, answering turn is given: standard, read, write."""

    schema = {"type": "object"}

    return tuple(ToolDefinition(name, name, schema) for name in (
        "standard.search", "standard.fetch", "standard.cite", "read_file",
        "bash", "edit_file", "write_file"))


def ask(text):
    return [ConversationMessage("user", (TextBlock(text),))]


def answering_turn(question, *, observer=None, prior=ESTABLISHED,
                   answer=NORMATIVE_ANSWER, executor=None, mode=None):
    """One turn: two standard lookups, then the answer. Nothing else."""

    backend = ScriptedBackend([
        tool_turn("s1", "standard.search", query="acknowledge"),
        tool_turn("s2", "standard.fetch", section="4.2.1.1"),
    ] + [text_turn(answer)] * 6)
    executor = executor or GroundedToolExecutor()
    context = make_context(backend, executor=executor, rounds=12, actions=20,
                           observer=observer or RecordingObserver(),
                           conversation=ask(question))
    context.tools = tools()
    context.prior_clauses = tuple(prior)
    context.prior_answer = "An earlier turn's answer about acknowledgements."

    if mode is not None:
        # What `--auto` sets. It reaches the tool layer, which decides what
        # needs confirming; the runtime is handed it here so the test can
        # show the scope decision does not read it.

        context.execution_mode = mode

    return context, executor, AgentRuntime().run(context)


class ANormativeQuestionIsNotAnImplementationTask(unittest.TestCase):
    """CASE 1 -- the turn that was reported."""

    def setUp(self):
        self.observer = RecordingObserver()
        self.context, self.executor, self.result = answering_turn(
            QUESTION, observer=self.observer)

    def test_the_answer_comes_back(self):
        self.assertIn("Rule 4.2.1.1-2", self.result.final_response)

    def test_nothing_is_sent_back_to_the_clauses(self):
        self.assertNotIn("clauses_unaddressed",
                         [kind for kind, _ in self.observer.notices])

    def test_the_turn_is_not_told_to_edit_build_or_test(self):
        harness = [message for message in self.context.conversation
                   if getattr(message, "authored_by", "operator") == "harness"]

        self.assertEqual(harness, [])

    def test_no_write_and_no_command_was_run(self):
        used = {name for _, name, _ in self.executor.calls}

        self.assertEqual(used & set(WRITE_TOOLS), set())
        self.assertNotIn("bash", used)

    def test_the_standard_tools_were_available_and_used(self):
        used = {name for _, name, _ in self.executor.calls}

        self.assertTrue({"standard.search", "standard.fetch"} <= used)

    def test_it_ends_when_the_question_is_answered(self):
        # Two lookups and the answer. A forced follow-up is visible here as
        # model calls the turn never needed.

        self.assertEqual(self.result.model_calls, 3)
        self.assertEqual(len(self.executor.calls), 2)

    def test_the_clauses_are_not_this_turns_obligations(self):
        self.assertEqual(carried_obligations(self.context, QUESTION), ())


class AnImplementationTaskStillOwesThem(unittest.TestCase):
    """CASE 2 -- the gate this check exists for, unweakened."""

    def setUp(self):
        self.observer = RecordingObserver()
        self.context, self.executor, self.result = answering_turn(
            IMPLEMENTATION, observer=self.observer,
            answer="I reviewed Rule 4.2.1.1-2 and Rule 4.2.1.4-1.")

    def test_the_clauses_are_obligations(self):
        self.assertEqual(carried_obligations(self.context, IMPLEMENTATION),
                         ESTABLISHED)

    def test_the_turn_is_sent_back_to_them(self):
        notices = [metadata for kind, metadata in self.observer.notices
                   if kind == "clauses_unaddressed"]

        self.assertTrue(notices)
        self.assertEqual(notices[0]["carried"], len(ESTABLISHED))

    def test_the_demand_names_what_was_skipped(self):
        harness = [block.text for message in self.context.conversation
                   if getattr(message, "authored_by", "") == "harness"
                   for block in message.content]

        self.assertTrue(any("Not addressed" in text for text in harness))

    def test_a_turn_that_accounts_for_them_is_left_alone(self):
        observer = RecordingObserver()
        placed = " ".join(f"src/ack.c:{index} satisfies Rule {item}."
                          for index, item in enumerate(ESTABLISHED))
        answering_turn(IMPLEMENTATION, observer=observer, answer=placed)

        self.assertNotIn("clauses_unaddressed",
                         [kind for kind, _ in observer.notices])


class AutoDoesNotEnlargeTheTask(unittest.TestCase):
    """CASE 3 -- `--auto` is a confirmation policy, not a scope."""

    def outcome(self, mode):
        observer = RecordingObserver()
        context, executor, result = answering_turn(
            QUESTION, observer=observer, mode=mode)

        return (result.final_response, [kind for kind, _ in observer.notices],
                [name for _, name, _ in executor.calls],
                [getattr(message, "authored_by", "operator")
                 for message in context.conversation])

    def test_the_task_is_the_same_under_every_mode(self):
        safe = self.outcome("safe")

        for mode in ("ask", "auto"):
            with self.subTest(mode=mode):
                self.assertEqual(self.outcome(mode), safe)

    def test_auto_asks_for_no_clauses(self):
        observer = RecordingObserver()
        answering_turn(QUESTION, observer=observer, mode="auto")

        self.assertEqual(observer.notices, [])

    def test_the_scope_decision_cannot_read_the_mode(self):
        """Not a property of today's wiring: of the decision itself.

        A gate that widened the task because confirmation happened to be
        cheap would be reading `--auto` as permission to do more, and the
        way that gets written is a mode check inside the predicate.
        """
        source = inspect.getsource(carried_obligations)

        for token in ("execution_mode", "ExecutionMode", "AUTO", "auto"):
            with self.subTest(token=token):
                self.assertNotIn(token, source.split('"""')[2])


class ANewTurnInheritsNothing(unittest.TestCase):
    """CASE 4 -- turn N established them; turn N+1 was asked something else."""

    def test_the_question_after_the_implementation_turn_is_free_of_it(self):
        first = RecordingObserver()
        answering_turn(IMPLEMENTATION, observer=first,
                       answer="I reviewed Rule 4.2.1.1-2.")

        self.assertIn("clauses_unaddressed",
                      [kind for kind, _ in first.notices])

        second = RecordingObserver()
        context, executor, result = answering_turn(
            "Which clause defines the SelB bit position?", observer=second)

        self.assertEqual(second.notices, [])
        self.assertEqual(
            [message for message in context.conversation
             if getattr(message, "authored_by", "operator") == "harness"], [])

    def test_continuing_the_work_carries_it_again(self):
        # The user asking for the work to go on is what brings the
        # obligations back, and the only thing that does.

        context, _, _ = answering_turn(QUESTION)

        self.assertEqual(carried_obligations(
            context, "carry on and fix the remaining clauses"), ESTABLISHED)

    def test_an_explicit_prohibition_outranks_the_words(self):
        context, _, _ = answering_turn(QUESTION)
        context.read_only = True

        self.assertEqual(
            carried_obligations(context, "fix the remaining clauses"), ())


class TheGateAndThePromptAskOneQuestion(unittest.TestCase):
    """The demand may only name clauses the turn was given as work.

    Two spellings of "is this turn an implementation turn" would let the
    gate ask for file-and-line on clauses the prompt never presented as
    obligations -- and the demand says "the clauses are already established
    and listed above", which would then be false.
    """

    def test_the_runtime_gate_reads_the_predicate(self):
        source = inspect.getsource(AgentRuntime.run)

        self.assertEqual(source.count("carried_obligations("), 2)
        self.assertNotIn('getattr(context, "prior_clauses"', source)

    def test_the_prompt_reads_the_same_predicate(self):
        import task_controller

        source = inspect.getsource(task_controller.TaskController.run)

        self.assertIn("carried_obligations(context, asked)", source)
        self.assertNotIn('getattr(context, "prior_clauses"', source)


if __name__ == "__main__":
    unittest.main()
