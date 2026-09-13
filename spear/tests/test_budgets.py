import tempfile
import unittest

from budgets import BudgetExceeded, BudgetKind, BudgetLimit, BudgetManager
from agent_runtime import AgentRuntime, RuntimeTerminalReason
from compaction import ModelBackendSummarizer, StructuredCompactionState
from model_backend import ModelTurn, StopReason
from session_store import FileSessionStore, SessionConfiguration, SessionHandle, SessionSnapshot
from tests.test_agent_runtime import ScriptedBackend, make_context, text_turn
from working_state import WorkingState


class BudgetManagerTests(unittest.TestCase):
    def test_primary_tool_command_and_approximate_tokens(self):
        budget = BudgetManager("task", {
            BudgetKind.PRIMARY_MODEL_CALLS: BudgetLimit(2),
            BudgetKind.TOOL_CALLS: BudgetLimit(2),
            BudgetKind.COMMAND_EXECUTIONS: BudgetLimit(1),
            BudgetKind.INPUT_TOKENS: BudgetLimit(100, True),
        })
        budget.consume(BudgetKind.PRIMARY_MODEL_CALLS)
        budget.consume(BudgetKind.TOOL_CALLS)
        budget.consume(BudgetKind.COMMAND_EXECUTIONS)
        budget.consume(BudgetKind.INPUT_TOKENS, 20, approximate=True)
        self.assertEqual(budget.remaining(BudgetKind.PRIMARY_MODEL_CALLS), 1)
        self.assertTrue(budget.metrics()["input_tokens"]["approximate"])

    def test_exhaustion_is_typed(self):
        budget = BudgetManager("main", {BudgetKind.TOOL_CALLS: BudgetLimit(1)})
        budget.consume(BudgetKind.TOOL_CALLS)
        with self.assertRaises(BudgetExceeded) as caught:
            budget.consume(BudgetKind.TOOL_CALLS)
        self.assertEqual(caught.exception.kind, BudgetKind.TOOL_CALLS)

    def test_child_allocation_debits_parent_and_cannot_exceed_it(self):
        parent = BudgetManager("task", {
            BudgetKind.CHILD_AGENT_CALLS: BudgetLimit(2),
            BudgetKind.CHILD_MODEL_TURNS: BudgetLimit(3),
            BudgetKind.PRIMARY_MODEL_CALLS: BudgetLimit(3),
            BudgetKind.TOOL_CALLS: BudgetLimit(4),
        })
        child = parent.allocate_child("explore", component="explorer", limits={
            BudgetKind.MODEL_TURNS: 2, BudgetKind.PRIMARY_MODEL_CALLS: 2,
            BudgetKind.TOOL_CALLS: 2,
        })
        child.consume(BudgetKind.MODEL_TURNS)
        child.consume(BudgetKind.PRIMARY_MODEL_CALLS)
        child.consume(BudgetKind.TOOL_CALLS)
        self.assertEqual(parent.consumed[BudgetKind.CHILD_MODEL_TURNS], 1)
        self.assertEqual(parent.consumed[BudgetKind.PRIMARY_MODEL_CALLS], 1)
        with self.assertRaises(BudgetExceeded):
            parent.allocate_child("too_big", component="reviewer", limits={
                BudgetKind.MODEL_TURNS: 3,
            })

    def test_snapshot_round_trip_does_not_reset_consumption(self):
        budget = BudgetManager("task", {BudgetKind.MODEL_TURNS: BudgetLimit(3)})
        budget.consume(BudgetKind.MODEL_TURNS, 2)
        restored = BudgetManager.from_dict(budget.to_dict())
        self.assertEqual(restored.remaining(BudgetKind.MODEL_TURNS), 1)
        with self.assertRaises(BudgetExceeded):
            restored.consume(BudgetKind.MODEL_TURNS, 2)

    def test_main_runtime_budget_exhaustion_is_typed(self):
        context = make_context(ScriptedBackend([text_turn("unused")]))
        context.budget_manager = BudgetManager("main", {
            BudgetKind.PRIMARY_MODEL_CALLS: BudgetLimit(0),
        })
        result = AgentRuntime().run(context)
        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.BUDGET_EXHAUSTED)
        self.assertTrue(result.budget_exhausted)

    def test_auxiliary_compaction_has_separate_budget(self):
        state = WorkingState.start("task_auxbudget", "compact")
        backend = ScriptedBackend([])
        budget = BudgetManager("task", {
            BudgetKind.PRIMARY_MODEL_CALLS: BudgetLimit(2),
            BudgetKind.AUXILIARY_MODEL_CALLS: BudgetLimit(0),
        })
        summarizer = ModelBackendSummarizer(backend, budget_manager=budget)
        with self.assertRaises(BudgetExceeded) as caught:
            summarizer.summarize(
                structured_state=StructuredCompactionState.from_working_state(state),
                existing_summary=None, source_text="old", max_summary_chars=40,
            )
        self.assertEqual(caught.exception.kind, BudgetKind.AUXILIARY_MODEL_CALLS)
        self.assertEqual(budget.consumed.get(BudgetKind.PRIMARY_MODEL_CALLS, 0), 0)

    def test_session_snapshot_persists_budget_and_delegation_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FileSessionStore(directory)
            state = WorkingState.start("task_budget", "work")
            config = SessionConfiguration("/workspace", "project", "safe")
            handle = SessionHandle(store, "session_budget123", config)
            budget = BudgetManager("task", {BudgetKind.TOOL_CALLS: BudgetLimit(4)})
            budget.consume(BudgetKind.TOOL_CALLS, 2)
            snapshot = SessionSnapshot(
                handle.session_id, state.task_id, 0, config, state, (),
                budget_state=budget.to_dict(),
                delegation_results=({"delegation_id": "delegate_one"},),
                tool_exposure={"names": ["bash"]},
            )
            store.save_snapshot(snapshot)
            loaded = store.load_snapshot(handle.session_id)
            restored = BudgetManager.from_dict(loaded.budget_state)
            self.assertEqual(restored.consumed[BudgetKind.TOOL_CALLS], 2)
            self.assertEqual(loaded.delegation_results[0]["delegation_id"], "delegate_one")
            self.assertEqual(loaded.tool_exposure["names"], ["bash"])


if __name__ == "__main__":
    unittest.main()


class RefundingWhatProducedNothing(unittest.TestCase):
    """A charge for an attempt the turn got nothing from is a charge twice."""

    def manager(self, amount=3):
        return BudgetManager("task", {BudgetKind.MODEL_TURNS: BudgetLimit(amount)})

    def test_a_refund_gives_the_unit_back(self):
        manager = self.manager()
        manager.consume(BudgetKind.MODEL_TURNS)
        manager.consume(BudgetKind.MODEL_TURNS)
        manager.refund(BudgetKind.MODEL_TURNS)

        self.assertEqual(manager.remaining(BudgetKind.MODEL_TURNS), 2)

    def test_a_refund_never_goes_below_zero(self):
        manager = self.manager()
        manager.consume(BudgetKind.MODEL_TURNS)
        manager.refund(BudgetKind.MODEL_TURNS, 99)

        self.assertEqual(manager.consumed[BudgetKind.MODEL_TURNS], 0)

    def test_a_budget_with_no_such_limit_is_untouched(self):
        manager = self.manager()
        manager.refund(BudgetKind.TOOL_CALLS)

        self.assertNotIn(BudgetKind.TOOL_CALLS, manager.consumed)
