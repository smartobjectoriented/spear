import unittest
from types import SimpleNamespace

import request_intent


class Backend:
    """A classifier that answers whatever it was told to."""

    def __init__(self, text="", error=None, boom=False):
        self.text = text
        self.error = error
        self.boom = boom
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)

        if self.boom:
            raise RuntimeError("provider down")

        return SimpleNamespace(text=self.text, error=self.error)


class ReadingTheRequest(unittest.TestCase):
    def test_write_and_ask_are_believed(self):
        self.assertIs(request_intent.judge(Backend("WRITE"), "adapt it"), True)
        self.assertIs(request_intent.judge(Backend("ASK"), "what is it?"), False)

    def test_the_word_is_taken_however_it_is_dressed(self):
        for text in ("write", "Write.", "**WRITE**", "`ask`", "ASK — no change"):
            with self.subTest(text=text):
                self.assertIsNotNone(request_intent.judge(Backend(text), "x"))

    def test_anything_else_is_no_answer_at_all(self):
        """A classifier that has to be interpreted is a second guess."""
        for text in ("", "It depends on what you mean", "maybe", "1"):
            with self.subTest(text=text):
                self.assertIsNone(request_intent.judge(Backend(text), "x"))

    def test_a_broken_provider_teaches_nothing_rather_than_no(self):
        self.assertIsNone(request_intent.judge(Backend(boom=True), "x"))
        self.assertIsNone(request_intent.judge(Backend("WRITE", error="502"), "x"))
        self.assertIsNone(request_intent.judge(None, "x"))

    def test_a_spent_auxiliary_budget_falls_back_instead_of_failing(self):
        from budgets import BudgetKind, BudgetLimit, BudgetManager

        manager = BudgetManager("task", {BudgetKind.AUXILIARY_MODEL_CALLS:
                                         BudgetLimit(0)})
        backend = Backend("WRITE")

        self.assertIsNone(request_intent.judge(
            backend, "adapt it", budget_manager=manager,
            budget_kind=BudgetKind.AUXILIARY_MODEL_CALLS))
        self.assertEqual(backend.calls, [], "a refused budget makes no call")


class TheFloorIsNeverLowered(unittest.TestCase):
    """The pattern is the deterministic core; the reading only adds to it."""

    def test_the_pattern_wins_even_against_a_no(self):
        self.assertTrue(request_intent.resolve(True, False))
        self.assertTrue(request_intent.resolve(True, None))

    def test_the_reading_can_only_add(self):
        self.assertTrue(request_intent.resolve(False, True))
        self.assertFalse(request_intent.resolve(False, False))
        self.assertFalse(request_intent.resolve(False, None))


class InTheLoop(unittest.TestCase):
    def test_the_verb_list_being_silent_is_what_triggers_the_call(self):
        import agent_runtime

        context = SimpleNamespace(backend=Backend("WRITE"), judge_intent=True,
                                  budget_manager=None,
                                  observer=SimpleNamespace(notice=lambda *a: None))

        # A phrase no verb list covers: the operator says what is wrong and
        # leaves the instruction implicit.
        self.assertTrue(agent_runtime.wants_write(
            context, "the acknowledgement path does not match 8.4.1.1-3"))
        self.assertEqual(len(context.backend.calls), 1)

    def test_a_recognised_verb_costs_no_call_at_all(self):
        import agent_runtime

        context = SimpleNamespace(backend=Backend("ASK"), judge_intent=True,
                                  budget_manager=None,
                                  observer=SimpleNamespace(notice=lambda *a: None))

        self.assertTrue(agent_runtime.wants_write(context, "implement it"))
        self.assertEqual(context.backend.calls, [])

    def test_the_answer_is_decided_once_per_turn(self):
        import agent_runtime

        context = SimpleNamespace(backend=Backend("WRITE"), judge_intent=True,
                                  budget_manager=None,
                                  observer=SimpleNamespace(notice=lambda *a: None))

        for _ in range(4):
            agent_runtime.wants_write(context, "have a look at this")

        self.assertEqual(len(context.backend.calls), 1)


if __name__ == "__main__":
    unittest.main()
