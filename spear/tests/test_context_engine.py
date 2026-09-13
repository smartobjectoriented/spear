import json
import unittest
from pathlib import Path

from context_engine import (
    ApproximateTokenEstimator,
    ContextEngine,
    ContextError,
    ContextItem,
    ContextLayer,
    ContextRequest,
    Freshness,
    working_state_context_item,
    working_state_projection,
)
from agent_runtime import AgentContext, AgentRuntime
from compaction import CompactionPolicy
from model_backend import ToolDefinition
from tracing import EventType, TraceEmitter
from working_state import (
    ActionKind,
    StateEvent,
    StateEventType,
    StateSource,
    VerificationOutcome,
    WorkingState,
)


def item(
    item_id, layer, content, *, priority=50, protected=False,
    freshness=Freshness.RECENT, source="test", group=None,
):
    return ContextItem(
        item_id=item_id,
        layer=layer,
        source=source,
        content=content,
        priority=priority,
        protected=protected,
        freshness=freshness,
        inclusion_reason="test fixture",
        eviction_group=group,
    )


def request(*items, limit=1000, reserve=0, margin=0, fixed=0):
    return ContextRequest(
        items=tuple(items), context_limit=limit, output_reserve=reserve,
        safety_margin=margin, fixed_input_tokens=fixed,
    )


class ContextEngineUnitTests(unittest.TestCase):
    def setUp(self):
        self.engine = ContextEngine(ApproximateTokenEstimator())

    def test_empty_context(self):
        snapshot = self.engine.compose(request())
        self.assertEqual(snapshot.selected_items, ())
        self.assertEqual(snapshot.excluded_items, ())
        self.assertEqual(snapshot.estimated_input_tokens, 0)
        self.assertTrue(snapshot.within_budget)

    def test_deterministic_ordering_preserves_caller_render_order(self):
        items = (
            item("system", ContextLayer.SYSTEM_RULES, "system"),
            item("memory", ContextLayer.DURABLE_MEMORY, "memory"),
            item("project", ContextLayer.PROJECT_RULES, "project"),
        )
        first = self.engine.compose(request(*items))
        second = self.engine.compose(request(*items))
        self.assertEqual(
            [entry.item_id for entry in first.selected_items],
            ["system", "memory", "project"],
        )
        self.assertEqual(first, second)

    def test_layer_and_provenance_are_preserved(self):
        original = item(
            "rag", ContextLayer.RETRIEVED_CONTEXT, "retrieved",
            source="project_index",
        )
        selected = self.engine.compose(request(original)).selected_items[0]
        self.assertEqual(selected.layer, ContextLayer.RETRIEVED_CONTEXT)
        self.assertEqual(selected.source, "project_index")
        stats = self.engine.compose(request(original)).layer_statistics
        self.assertEqual(stats[ContextLayer.RETRIEVED_CONTEXT].included_items, 1)

    def test_budget_accounts_for_output_margin_and_fixed_tokens(self):
        snapshot = self.engine.compose(request(
            item("rules", ContextLayer.SYSTEM_RULES, "x" * 80, protected=True),
            limit=100, reserve=20, margin=10, fixed=5,
        ))
        self.assertEqual(snapshot.available_input_tokens, 70)
        self.assertEqual(snapshot.fixed_input_tokens, 5)
        self.assertLessEqual(snapshot.estimated_input_tokens, 70)
        self.assertTrue(snapshot.estimate_is_approximate)

    def test_protected_items_survive_eviction(self):
        protected = item(
            "rules", ContextLayer.SYSTEM_RULES, "r" * 120,
            priority=100, protected=True,
        )
        expendable = item(
            "old", ContextLayer.RECENT_CONVERSATION, "o" * 120,
            priority=1,
        )
        snapshot = self.engine.compose(request(protected, expendable, limit=40))
        self.assertIn("rules", [entry.item_id for entry in snapshot.selected_items])
        self.assertEqual(
            [entry.item.item_id for entry in snapshot.excluded_items], ["old"]
        )

    def test_low_priority_item_is_evicted_first_within_a_layer(self):
        low = item(
            "low", ContextLayer.RETRIEVED_CONTEXT, "l" * 100, priority=10
        )
        high = item(
            "high", ContextLayer.RETRIEVED_CONTEXT, "h" * 100, priority=80
        )
        snapshot = self.engine.compose(request(low, high, limit=30))
        self.assertEqual(snapshot.excluded_items[0].item.item_id, "low")

    def test_explicit_eviction_order_prefers_obsolete_conversation(self):
        old = item(
            "old", ContextLayer.RECENT_CONVERSATION, "o" * 100,
            priority=90, freshness=Freshness.STALE,
        )
        rag = item(
            "rag", ContextLayer.RETRIEVED_CONTEXT, "r" * 100,
            priority=1,
        )
        snapshot = self.engine.compose(request(old, rag, limit=30))
        self.assertEqual(snapshot.excluded_items[0].item.item_id, "old")
        self.assertEqual(
            snapshot.excluded_items[0].reason, "obsolete_conversation_evicted"
        )

    def test_eviction_groups_are_atomic(self):
        call = item(
            "call", ContextLayer.TOOL_EVIDENCE, "call" * 20, group="exchange"
        )
        result = item(
            "result", ContextLayer.TOOL_EVIDENCE, "result" * 20, group="exchange"
        )
        snapshot = self.engine.compose(request(call, result, limit=10))
        self.assertEqual(
            [entry.item.item_id for entry in snapshot.excluded_items],
            ["call", "result"],
        )

    def test_oversized_protected_context_is_truncated_not_evicted(self):
        rules = item(
            "rules", ContextLayer.SYSTEM_RULES, "important " * 100,
            priority=100, protected=True,
        )
        snapshot = self.engine.compose(request(rules, limit=40))
        self.assertEqual(len(snapshot.selected_items), 1)
        self.assertEqual(snapshot.excluded_items, ())
        self.assertTrue(snapshot.truncation_decisions)
        self.assertTrue(snapshot.within_budget)
        self.assertIn("context truncated", snapshot.selected_items[0].content)

    def test_unavoidable_fixed_overflow_is_reported(self):
        snapshot = self.engine.compose(request(limit=10, fixed=15))
        self.assertFalse(snapshot.within_budget)
        self.assertEqual(snapshot.overflow_tokens, 5)

    def test_malformed_context_items_and_requests_are_rejected(self):
        with self.assertRaises(ContextError):
            ContextItem("", ContextLayer.SYSTEM_RULES, "source", "content")
        with self.assertRaises(ContextError):
            ContextItem("x", "system_rules", "source", "content")
        with self.assertRaises(ContextError):
            self.engine.compose(ContextRequest(items=("not-an-item",)))
        duplicate = item("same", ContextLayer.SYSTEM_RULES, "one")
        with self.assertRaises(ContextError):
            self.engine.compose(request(duplicate, duplicate))


class WorkingStateProjectionTests(unittest.TestCase):
    def populated_state(self):
        state = WorkingState.start("task_projection", "Fix the build")
        for event_type, data in (
            (StateEventType.CONSTRAINT_RECORDED,
             {"constraint": "Remain provider-neutral"}),
            (StateEventType.ACCEPTANCE_CRITERION_RECORDED,
             {"criterion": "Tests pass"}),
            (StateEventType.PLAN_STEP_ADDED,
             {"step_id": "inspect", "description": "Inspect failure"}),
        ):
            state.apply(StateEvent.create(
                event_type, state.task_id, StateSource.USER, **data
            ))
        state.apply(StateEvent.create(
            StateEventType.ACTION_FAILED, state.task_id, StateSource.TOOL_RUNTIME,
            action_id="make_1", kind=ActionKind.COMMAND.value, name="make",
            observed_status="failed", category="command_nonzero",
            summary="exit 2", exit_code=2,
        ))
        state.apply(StateEvent.create(
            StateEventType.DISCOVERY_RECORDED, state.task_id, StateSource.HARNESS,
            summary="Build entry point located", file_path="Makefile", symbol="all",
        ))
        state.apply(StateEvent.create(
            StateEventType.VERIFICATION_RECORDED, state.task_id, StateSource.HARNESS,
            verification_id="bench_1", kind="project_bench", executed=True,
            outcome=VerificationOutcome.FAILED.value, summary="bench failed",
        ))
        return state

    def test_projection_contains_selected_truth_not_event_history(self):
        state = self.populated_state()
        projection = working_state_projection(state)
        self.assertIn("Objective: Fix the build", projection)
        self.assertIn("Remain provider-neutral", projection)
        self.assertIn("Tests pass", projection)
        self.assertIn("Inspect failure", projection)
        self.assertIn("command_nonzero: exit 2", projection)
        self.assertIn("Build entry point located (Makefile: all)", projection)
        self.assertIn("project_bench: failed", projection)
        self.assertNotIn("stateevt_", projection)
        self.assertNotIn("make_1", projection)
        self.assertNotIn("action_failed", projection)

    def test_projection_and_context_item_are_deterministic_and_protected(self):
        state = self.populated_state()
        first = working_state_context_item(state)
        second = working_state_context_item(state)
        self.assertEqual(first, second)
        self.assertEqual(first.layer, ContextLayer.TASK_WORKING_STATE)
        self.assertTrue(first.protected)
        self.assertEqual(first.content, working_state_projection(state))


class MemoryRecorder:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


class ContextRuntimeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rag_chat
        cls.rag_chat = rag_chat

    def test_simple_provider_neutral_turn_preserves_layers_and_message_types(self):
        from model_backend import ConversationMessage, ModelTurn, StopReason, TextBlock

        class FakeBackend:
            model = "scripted-local"
            max_tokens = 64

            def __init__(self):
                self.request = None

            def complete(self, **kwargs):
                self.request = kwargs
                return ModelTurn("done", (), StopReason.END_TURN)

        state = WorkingState.start("task_context", "Answer accurately")
        fragments = {
            "system_instructions": "SYSTEM",
            "system_source": "test_system",
            "global_rules": "|GLOBAL",
            "project_rules": "|PROJECT",
            "tool_guide": "|TOOLS",
            "tool_guide_source": "test_tool_guide",
            "memories": "|MEMORY",
            "skills": "|SKILL",
            "working_directory": "|WORKSPACE",
            "working_state": state,
            "retrieval": "|RAG",
            "retrieval_source": "test_retrieval",
        }
        items = self.rag_chat.build_task_context_items(**fragments)
        state.apply(StateEvent.create(
            StateEventType.CONSTRAINT_RECORDED, state.task_id, StateSource.USER,
            constraint="Added after initial context collection",
        ))
        backend = FakeBackend()
        conversation = [ConversationMessage("user", (TextBlock("hello"),))]
        context = AgentContext(
            state, backend, ContextEngine(), TraceEmitter(),
            "unused compatibility string", items, conversation,
            (ToolDefinition("test", "test", {}),),
            lambda *_: "OK", 3, 3, 4096,
            output_reserve=64,
            compaction_policy=CompactionPolicy(minimum_compactable_tokens=10000),
        )
        turn = AgentRuntime().complete_model_turn(context, use_tools=False)
        self.assertEqual(turn.text, "done")
        rendered = backend.request["system"]

        # The legacy render order, with the answer-scope rule sitting among
        # the other system rules: after the tool guide, before the memories.

        expected_prefix = ("SYSTEM|GLOBAL|PROJECT|TOOLS"
                           + self.rag_chat.ANSWER_SCOPE_RULE
                           + "|MEMORY|SKILL|WORKSPACE")
        self.assertTrue(rendered.startswith(expected_prefix))
        self.assertIn("## Current task state", rendered)
        self.assertIn("Added after initial context collection", rendered)
        self.assertTrue(rendered.endswith("|RAG"))
        self.assertEqual(backend.request["conversation"], tuple(conversation))

    def test_context_trace_contains_counts_not_prompt_content(self):
        from model_backend import ConversationMessage, ModelTurn, StopReason, TextBlock

        class FakeBackend:
            max_tokens = 16

            def complete(self, **kwargs):
                return ModelTurn("done", (), StopReason.END_TURN)

        recorder = MemoryRecorder()
        secret = "authorization=do-not-record"
        state = WorkingState.start("trace_context", "trace safely")
        backend = FakeBackend()
        conversation = [
                    ConversationMessage("user", (TextBlock("old " * 200),)),
                    ConversationMessage("assistant", (TextBlock("answer " * 100),)),
                    ConversationMessage("user", (TextBlock("follow up"),)),
                    ConversationMessage("assistant", (TextBlock("follow up answer"),)),
                    ConversationMessage("user", (TextBlock(secret),)),
        ]
        items = (ContextItem(
            "system", ContextLayer.SYSTEM_RULES, "test", "SYSTEM",
            priority=100, freshness=Freshness.CURRENT, protected=True,
            inclusion_reason="test",
        ),)
        context = AgentContext(
            state, backend, ContextEngine(), TraceEmitter(recorder), "SYSTEM",
            items, conversation, (), lambda *_: "OK", 3, 3, 128,
            output_reserve=16,
            compaction_policy=CompactionPolicy(minimum_compactable_tokens=10000),
        )
        AgentRuntime().complete_model_turn(context, use_tools=False)
        event = next(entry for entry in recorder.events
                     if entry.event_type == EventType.CONTEXT_COMPOSED)
        self.assertIn("tokens_by_layer", event.metadata)
        self.assertIn("included_item_count", event.metadata)
        self.assertIn("truncation_reasons", event.metadata)
        self.assertGreaterEqual(event.metadata["excluded_item_count"], 1)
        encoded = json.dumps(event.to_dict())
        self.assertNotIn("do-not-record", encoded)
        self.assertNotIn("old old", encoded)

    def test_context_engine_has_no_provider_specific_imports(self):
        source = Path(__file__).resolve().parents[1].joinpath(
            "context_engine.py"
        ).read_text()
        self.assertNotIn("model_backend", source)
        self.assertNotIn("OpenAI", source)
        self.assertNotIn("Anthropic", source)


class AnswerScopeRuleTests(unittest.TestCase):
    """The writing rule has to reach the model, in every corpus.

    A stability campaign lost four runs the same way: the agent found the
    right answer, obeyed the restriction, and then named the excluded item to
    explain why it was excluded -- which is the restriction broken in the act
    of claiming to honour it.
    """

    SENTENCE = "omit excluded items entirely"

    @classmethod
    def setUpClass(cls):
        import rag_chat

        cls.rag_chat = rag_chat

    def items(self, base):
        from working_state import WorkingState

        return self.rag_chat.build_task_context_items(
            system_instructions=base, system_source="probe",
            global_rules="rules", project_rules="project",
            tool_guide=self.rag_chat.TOOL_GUIDE, tool_guide_source="probe",
            memories="m" * 4000, skills="s" * 4000, working_directory="cwd",
            working_state=WorkingState.start("task_probe", "probe"),
            retrieval="r" * 40000, retrieval_source="probe")

    def bases(self):
        """The two mutually exclusive system-instruction sources."""

        # The prompt this tree ships. A corpus may name another through
        # prompt_file, but that file is a deployment's, not ours to assert on
        # -- which is why the rule is injected beside the prompt rather than
        # written into it.
        return {"generic": self.rag_chat.ADHOC_PROMPT}

    def test_the_rule_says_the_thing_and_names_nothing(self):
        rule = self.rag_chat.ANSWER_SCOPE_RULE

        self.assertIn(self.SENTENCE, rule)
        self.assertIn("Do not name them to explain their exclusion", rule)
        self.assertIn("claim compliance with the restriction", rule)

        # It is a general writing rule, so it knows nothing about any task.

        for token in ("test_", "main.py", "explorer", "causal", "benchmark",
                      "unrelated", "fixture"):
            self.assertNotIn(token, rule.lower())

        # And it is stated ONCE: the base prompts must not carry their own.

        for label, base in self.bases().items():
            with self.subTest(corpus=label):
                self.assertNotIn(self.SENTENCE, base)

    def test_it_reaches_the_system_context_of_every_corpus(self):
        for label, base in self.bases().items():
            with self.subTest(corpus=label):
                items = self.items(base)
                rule = next(item for item in items
                            if item.item_id == "system:answer-scope")

                self.assertEqual(rule.layer, ContextLayer.SYSTEM_RULES)
                self.assertTrue(rule.protected)
                self.assertFalse(rule.truncatable)
                self.assertIn(self.SENTENCE,
                              "".join(item.content for item in items))

    def test_it_survives_a_window_that_is_truncating_the_prompt_itself(self):
        """Marked not truncatable, so pressure cuts the long blocks first."""

        engine = ContextEngine()

        for label, base in self.bases().items():
            items = tuple(self.items(base))

            for limit in (32768, 16384, 9600, 9200):
                with self.subTest(corpus=label, limit=limit):
                    snapshot = engine.compose(ContextRequest(
                        items=items, context_limit=limit))
                    text = "".join(item.content
                                   for item in snapshot.selected_items)

                    self.assertIn(self.SENTENCE, text)
                    self.assertNotIn(
                        "system:answer-scope",
                        [decision.item_id
                         for decision in snapshot.truncation_decisions])


if __name__ == "__main__":
    unittest.main()
