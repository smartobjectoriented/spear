import json
import tempfile
import unittest
from pathlib import Path

from agent_roles import explorer_role
from agent_runtime import AgentContext, AgentRuntime
from cancellation import CancellationScope, CancellationSource
from compaction import CompactionPolicy
from context_engine import ContextEngine, ContextItem, ContextLayer, Freshness
from model_backend import (
    ConversationMessage, ModelToolCall, ModelTurn, StopReason, TextBlock,
    ToolDefinition,
)
from orchestration import (
    ExplorationPolicy, ExplorationRequest, ExplorationService,
    ExplorationStatus,
)
from session_store import (
    FileSessionStore, SessionConfiguration, SessionEventType, SessionHandle,
    new_session_id,
)
from tool_router import ToolResultEnvelope, ToolResultStatus
from tracing import EventType, TraceEmitter
from working_state import WorkingState


class Recorder:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


class ScriptedBackend:
    model = "local-scripted"
    max_tokens = 128

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if "compact agent context" in kwargs["system"]:
            return ModelTurn(json.dumps({"narrative_summary": "compact"}), (),
                             StopReason.END_TURN)
        turn = self.turns.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        return turn


class CancellingBackend:
    model = "local-scripted"
    max_tokens = 128

    def __init__(self):
        self.cancel = None

    def complete(self, **kwargs):
        self.cancel()
        return ModelTurn("ignored", (), StopReason.END_TURN)


class ReadExecutor:
    def __init__(self):
        self.calls = []

    def __call__(self, context, call_id, name, arguments, cache):
        self.calls.append((context, name, dict(arguments)))
        path = "service.py" if "service" in str(arguments) else "main.py"
        text = f"{path}: def entry(): service.run()"
        return ToolResultEnvelope(
            call_id, "action_" + str(len(self.calls)), name, True,
            ToolResultStatus.OK, text, text, "command", .001, len(text),
            read_paths=(path,),
        )


class RepositoryReadExecutor:
    def __init__(self, root):
        self.root = Path(root)
        self.calls = []

    def __call__(self, context, call_id, name, arguments, cache):
        command = str(arguments.get("command", ""))
        candidates = [relative for relative in (
            "main.py", "service.py", "storage.py", "tests/test_service.py",
            "unrelated_a.py", "unrelated_b.py",
        ) if relative in command]
        inspected = tuple(candidates)
        text = "\n".join((self.root / item).read_text() for item in inspected)
        self.calls.append((context, command, inspected))
        return ToolResultEnvelope(
            call_id, "repo_action_" + str(len(self.calls)), name, True,
            ToolResultStatus.OK, text, text, "command", .001, len(text),
            read_paths=inspected,
        )


def final_report():
    return json.dumps({
        "summary": "main.py calls service.py",
        "key_findings": ["entry delegates to service"],
        "relevant_files": ["main.py", "service.py"],
        "entry_points": ["main.py:entry"],
        "symbols": ["entry", "run"],
        "call_chains": ["main.entry -> service.run"],
        "tests": ["tests/test_service.py"],
        "architecture": ["thin entry point, service layer"],
        "uncertainties": ["storage implementation not inspected"],
        "unanswered_questions": [],
        "evidence_references": ["action_1"],
    })


def parent_context(backend, *, trace=None, cancellation=None, session=None):
    state = WorkingState.start("task_parent123", "Map repository architecture",
                               max_model_rounds=8, max_tool_actions=20)
    items = (
        ContextItem("system", ContextLayer.SYSTEM_RULES, "main", "MAIN SECRET TRANSCRIPT",
                    protected=True),
        ContextItem("project", ContextLayer.PROJECT_RULES, "rules", "Follow project rules",
                    protected=True),
    )
    return AgentContext(
        state, backend, ContextEngine(), trace or TraceEmitter(), "MAIN", items,
        [ConversationMessage("user", (TextBlock("private parent transcript"),))],
        (), lambda *_: None, 8, 20, 4096, output_reserve=64,
        safety_margin=16,
        compaction_policy=CompactionPolicy(minimum_compactable_tokens=10000),
        cancellation=cancellation or CancellationSource().token,
        session=session,
    )


def child_turns():
    return [
        ModelTurn("", (ModelToolCall("read1", "bash",
                  {"command": "rg service"}),), StopReason.TOOL_USE),
        ModelTurn(final_report(), (), StopReason.END_TURN),
    ]


class OrchestrationTests(unittest.TestCase):
    def test_policy_bypasses_simple_and_known_file_tasks(self):
        policy = ExplorationPolicy()
        self.assertFalse(policy.decide("Change the constant in config.py").should_explore)
        self.assertFalse(policy.decide("small edit", known_files=("a.py",)).should_explore)
        self.assertTrue(policy.decide("Map the repository architecture").should_explore)

    def test_isolated_runtime_returns_structured_report_only(self):
        backend = ScriptedBackend(child_turns())
        parent = parent_context(backend)
        executor = ReadExecutor()
        report = ExplorationService(AgentRuntime()).explore(
            ExplorationRequest(parent.task_id, "Find repository entry points"),
            parent, tools=(ToolDefinition("bash", "read", {"type": "object"}),),
            tool_executor=executor,
        )
        self.assertEqual(report.status, ExplorationStatus.COMPLETED)
        self.assertEqual(report.relevant_files[:2], ("main.py", "service.py"))
        self.assertIn("service.py", report.files_inspected)
        self.assertNotEqual(report.child_task_id, parent.task_id)
        child = executor.calls[0][0]
        self.assertIsNot(child.working_state, parent.working_state)
        self.assertIsNot(child.progress_monitor, parent.progress_monitor)
        self.assertIsNone(child.compaction_artifact)
        self.assertIsNone(child.checkpoint_manager)
        self.assertIsNone(child.checkpoint)
        self.assertEqual(child.role, "explorer")
        child_input = json.dumps([item.content for item in child.context_items])
        self.assertNotIn("private parent transcript", child_input)
        self.assertNotIn("MAIN SECRET TRANSCRIPT", child_input)
        self.assertEqual(len(parent.conversation), 1)
        self.assertNotIn(final_report(), str(parent.conversation))
        self.assertEqual(parent.context_items[-1].layer,
                         ContextLayer.RETRIEVED_CONTEXT)
        self.assertIn("Explorer report", parent.context_items[-1].content)

    def test_synthetic_multifile_exploration_then_parent_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                "main.py": "from service import run\nrun()\n",
                "service.py": "from storage import load\ndef run(): return load()\n",
                "storage.py": "def load(): return 1\n",
                "config.py": "MODE = 'test'\n",
                "tests/test_service.py": "def test_run(): pass\n",
                "unrelated_a.py": "noise = 1\n",
                "unrelated_b.py": "noise = 2\n",
            }
            for relative, content in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            original = {name: (root / name).read_bytes() for name in files}
            turns = [
                ModelTurn("", (ModelToolCall("one", "bash", {
                    "command": "cat main.py service.py"}),), StopReason.TOOL_USE),
                ModelTurn("", (ModelToolCall("two", "bash", {
                    "command": "cat storage.py tests/test_service.py"}),),
                          StopReason.TOOL_USE),
                ModelTurn(final_report(), (), StopReason.END_TURN),
            ]
            parent = parent_context(ScriptedBackend(turns))
            parent.checkpoint_manager = object()  # parent-only ownership sentinel
            parent.checkpoint = object()
            executor = RepositoryReadExecutor(root)
            report = ExplorationService(AgentRuntime()).explore(
                ExplorationRequest(parent.task_id, "Map repository architecture"),
                parent,
                tools=(ToolDefinition("bash", "read", {"type": "object"}),),
                tool_executor=executor,
            )
            self.assertEqual(
                set(report.files_inspected),
                {"main.py", "service.py", "storage.py", "tests/test_service.py"},
            )
            self.assertNotIn("unrelated_a.py", report.files_inspected)
            self.assertTrue(all(call[0].checkpoint_manager is None
                                and call[0].checkpoint is None for call in executor.calls))
            self.assertEqual(original,
                             {name: (root / name).read_bytes() for name in files})
            parent.checkpoint_manager = None
            parent.checkpoint = None
            main_backend = ScriptedBackend([
                ModelTurn("Implementation target identified.", (), StopReason.END_TURN),
            ])
            parent.backend = main_backend
            result = AgentRuntime().run(parent)
            self.assertEqual(result.final_response, "Implementation target identified.")
            self.assertIn("Explorer report", main_backend.calls[0]["system"])
            self.assertEqual(parent.working_state.files_read, set())

    def test_trace_attribution_and_independent_usage(self):
        recorder = Recorder()
        parent = parent_context(ScriptedBackend(child_turns()),
                                trace=TraceEmitter(recorder))
        report = ExplorationService(AgentRuntime()).explore(
            ExplorationRequest(parent.task_id, "Find repository entry points"),
            parent, tools=(ToolDefinition("bash", "read", {"type": "object"}),),
            tool_executor=ReadExecutor(),
        )
        types = [event.event_type for event in recorder.events]
        self.assertIn(EventType.EXPLORATION_STARTED, types)
        self.assertIn(EventType.EXPLORATION_FINISHED, types)
        self.assertEqual(report.model_calls, 2)
        self.assertEqual(report.tool_calls, 1)
        event = next(e for e in recorder.events
                     if e.event_type == EventType.EXPLORATION_FINISHED)
        self.assertEqual(event.task_id, parent.task_id)
        self.assertEqual(event.metadata["child_task_id"], report.child_task_id)

    def test_parent_cancellation_propagates_without_child_to_parent_link(self):
        source = CancellationSource()
        source.cancel("stop parent", CancellationScope.TASK)
        parent = parent_context(ScriptedBackend([]), cancellation=source.token)
        report = ExplorationService(AgentRuntime()).explore(
            ExplorationRequest(parent.task_id, "Map repository architecture"),
            parent, tools=(), tool_executor=ReadExecutor(),
        )
        self.assertEqual(report.status, ExplorationStatus.CANCELLED)
        self.assertTrue(source.token.is_cancelled)

    def test_child_cancellation_does_not_cancel_parent(self):
        parent_source = CancellationSource()
        backend = CancellingBackend()
        parent = parent_context(backend, cancellation=parent_source.token)
        service = ExplorationService(AgentRuntime())
        backend.cancel = service.cancel_active
        report = service.explore(
            ExplorationRequest(parent.task_id, "Map repository architecture"),
            parent, tools=(), tool_executor=ReadExecutor(),
        )
        self.assertEqual(report.status, ExplorationStatus.CANCELLED)
        self.assertFalse(parent_source.token.is_cancelled)

    def test_report_persists_and_completed_explorer_is_not_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FileSessionStore(directory)
            handle = SessionHandle(
                store, new_session_id(),
                SessionConfiguration("/workspace", "project", "safe"),
            )
            handle.append(SessionEventType.SESSION_STARTED, "task_parent123")
            backend = ScriptedBackend(child_turns())
            parent = parent_context(backend, session=handle)
            service = ExplorationService(AgentRuntime())
            request = ExplorationRequest(parent.task_id, "Map repository architecture")
            first = service.explore(
                request, parent,
                tools=(ToolDefinition("bash", "read", {"type": "object"}),),
                tool_executor=ReadExecutor(),
            )
            loaded = store.load_snapshot(handle.session_id)
            self.assertEqual(len(loaded.exploration_reports), 1)
            resumed = AgentContext.from_session_snapshot(
                loaded, session=handle, backend=backend,
                context_engine=ContextEngine(), trace=TraceEmitter(), tools=(),
                tool_executor=ReadExecutor(), context_limit=4096,
            )
            second = service.explore(request, resumed, tools=(),
                                     tool_executor=ReadExecutor())
            self.assertEqual(second.exploration_id, first.exploration_id)
            self.assertEqual(len(backend.calls), 2)

    def test_failure_is_bounded_and_does_not_fail_parent(self):
        backend = ScriptedBackend([RuntimeError("offline")])
        parent = parent_context(backend)
        report = ExplorationService(AgentRuntime()).explore(
            ExplorationRequest(parent.task_id, "Map repository architecture"),
            parent, tools=(), tool_executor=ReadExecutor(),
        )
        self.assertIn(report.status, {ExplorationStatus.FAILED,
                                     ExplorationStatus.PARTIAL})
        self.assertEqual(parent.working_state.terminal_status.value, "running")
        self.assertEqual(parent.working_state.failures, {})

    def test_off_on_ablation_metrics_are_explicit(self):
        baseline = {"main_files_read": 5, "main_context_items": 7,
                    "explorer_tool_calls": 0, "success": True}
        backend = ScriptedBackend(child_turns())
        parent = parent_context(backend)
        report = ExplorationService(AgentRuntime()).explore(
            ExplorationRequest(parent.task_id, "Map repository architecture"),
            parent, tools=(ToolDefinition("bash", "read", {"type": "object"}),),
            tool_executor=ReadExecutor(),
        )
        enabled = {"main_files_read": 2,
                   "main_context_items": len(parent.context_items),
                   "explorer_tool_calls": report.tool_calls,
                   "success": report.status == ExplorationStatus.COMPLETED}
        self.assertTrue(baseline["success"] and enabled["success"])
        self.assertLess(enabled["main_files_read"], baseline["main_files_read"])
        self.assertGreater(enabled["explorer_tool_calls"], 0)


if __name__ == "__main__":
    unittest.main()
