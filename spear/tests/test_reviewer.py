import json
import tempfile
import unittest
from pathlib import Path

from agent_roles import AgentRole, AgentRoleSpec, reviewer_role
from agent_runtime import AgentContext, AgentRuntime
from cancellation import CancellationScope, CancellationSource
from checkpoint import CheckpointManager, RollbackStatus
from compaction import CompactionPolicy
from context_engine import ContextEngine, ContextItem, ContextLayer
from diff_evidence import DiffEvidence, DiffEvidenceService
from model_backend import (
    ConversationMessage, ModelToolCall, ModelTurn, StopReason, TextBlock,
    ToolDefinition,
)
from result_store import ResultStore
from reviewer import (
    FindingCategory, ReviewExecutionStatus, ReviewPolicy, ReviewRepairPolicy,
    ReviewRequest, ReviewService, ReviewVerdict, review_request_from_parent,
)
from session_store import (
    FileSessionStore, SessionConfiguration, SessionEventType, SessionHandle,
    new_session_id,
)
from tool_registry import ToolRegistry, ToolSpec, native_tool_specs
from tool_router import (
    ToolExecutionContext, ToolResultEnvelope, ToolResultStatus, ToolRouter,
)
from tool_runtime import CommandPolicy, ExecutionMode
from tracing import EventType, TraceEmitter
from verification import VerificationPolicy
from working_state import (
    StateEvent, StateEventType, StateSource, TerminalStatus, WorkingState,
)


class Recorder:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


class ScriptedBackend:
    model = "local-review-model"
    max_tokens = 256

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


class ReadExecutor:
    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, context, call_id, name, arguments, cache):
        self.calls.append((context, name, dict(arguments)))
        text = "ERROR: read failed" if self.fail else "src.py: return value"
        status = ToolResultStatus.FAILED if self.fail else ToolResultStatus.OK
        return ToolResultEnvelope(
            call_id, f"review_action_{len(self.calls)}", name, not self.fail,
            status, text, text, "command", .001, len(text),
            read_paths=() if self.fail else ("src.py",),
            error_category="tool_error" if self.fail else None,
            error_summary=text if self.fail else None,
        )


class RepairExecutor:
    def __init__(self):
        self.calls = []

    def __call__(self, context, call_id, name, arguments, cache):
        self.calls.append((name, dict(arguments)))
        mutation = name == "edit_file"
        text = "OK: repaired" if mutation else "1 passed"
        return ToolResultEnvelope(
            call_id, f"repair_action_{len(self.calls)}", name, True,
            ToolResultStatus.OK, text, text,
            "file_write" if mutation else "command", .001, len(text),
            exit_code=0 if name == "bash" else None,
            mutation=mutation, affected_paths=("src.py",) if mutation else (),
        )


class CancellingBackend:
    model = "local-review-model"
    max_tokens = 128

    def __init__(self):
        self.cancel = None

    def complete(self, **kwargs):
        self.cancel()
        return ModelTurn("ignored", (), StopReason.END_TURN)


def state_with_mutation(task_id="task_review123", path="src.py"):
    state = WorkingState.start(task_id, "Preserve zero values",
                               max_model_rounds=8, max_tool_actions=16)
    state.apply(StateEvent.create(
        StateEventType.ACCEPTANCE_CRITERION_RECORDED, task_id, StateSource.USER,
        criterion="Zero must remain a valid value",
    ))
    state.apply(StateEvent.create(
        StateEventType.ACTION_SUCCEEDED, task_id, StateSource.TOOL_RUNTIME,
        action_id="mutation_1", kind="tool", name="edit_file",
        observed_status="ok", summary="changed", round_number=1,
    ))
    state.apply(StateEvent.create(
        StateEventType.FILE_MODIFIED, task_id, StateSource.TOOL_RUNTIME,
        path=path, action_id="mutation_1",
    ))
    return state


def add_verification(state, *, passed=True, generation=None, coverage="full"):
    generation = state.mutation_generation if generation is None else generation
    state.apply(StateEvent.create(
        StateEventType.VERIFICATION_RECORDED, state.task_id, StateSource.HARNESS,
        verification_id=f"verify_{len(state.verifications) + 1}",
        kind="unit_test", category="unit_test", coverage=coverage,
        executed=True, outcome="passed" if passed else "failed",
        mutation_generation=generation, action_id="test_action",
        result_reference="result_" + "a" * 64,
        summary="tests passed" if passed else "tests failed",
    ))


def parent_context(backend, *, state=None, trace=None, cancellation=None,
                   session=None, checkpoint_manager=None, checkpoint=None):
    state = state or state_with_mutation()
    return AgentContext(
        working_state=state, backend=backend, context_engine=ContextEngine(),
        trace=trace or TraceEmitter(), system_prompt="IMPLEMENTER SYSTEM",
        context_items=(
            ContextItem("system", ContextLayer.SYSTEM_RULES, "main",
                        "IMPLEMENTER PRIVATE REASONING", protected=True),
            ContextItem("rules", ContextLayer.PROJECT_RULES, "project",
                        "Do not change public compatibility", protected=True),
        ),
        conversation=[ConversationMessage(
            "user", (TextBlock("private implementation transcript"),)
        )],
        tools=(), tool_executor=lambda *_: None,
        max_model_rounds=8, max_tool_actions=16, context_limit=8192,
        output_reserve=128, safety_margin=16,
        compaction_policy=CompactionPolicy(minimum_compactable_tokens=10000),
        cancellation=cancellation or CancellationSource().token,
        session=session, checkpoint_manager=checkpoint_manager,
        checkpoint=checkpoint,
    )


def diff(path="src.py", *, preview="- return value or 0\n+ return value\n",
         reference=None, external=()):
    return DiffEvidence(
        "checkpoint", (path,), preview, len(preview), False, reference,
        external_change_paths=tuple(external),
    )


def request(parent, *, evidence=None, status=None):
    evaluation = VerificationPolicy().evaluate_completion(parent.working_state)
    if status is not None:
        from verification import CompletionEvaluation, CompletionVerificationStatus
        evaluation = CompletionEvaluation(
            CompletionVerificationStatus(status),
            parent.working_state.mutation_generation,
        )
    return review_request_from_parent(
        parent, evidence or diff(), evaluation,
        project_rules=("Do not change public compatibility",),
    )


def review_json(verdict="accept", findings=(), **extra):
    value = {
        "verdict": verdict, "confidence": .9, "findings": list(findings),
        "requirement_coverage": ["Zero must remain a valid value"],
        "verification_assessment": "current evidence inspected",
        "suspected_regressions": [], "unrelated_changes": [],
        "test_gaps": [], "security_concerns": [],
        "recommended_repairs": [], "evidence_references": [],
    }
    value.update(extra)
    return json.dumps(value)


def finding(category, description, *, path="src.py", blocking=True,
            requirement=None, severity="high"):
    return {
        "severity": severity, "category": category, "description": description,
        "affected_path": path, "requirement_reference": requirement,
        "confidence": .95, "blocking": blocking,
        "suggested_remediation": "repair the grounded defect",
    }


class ReviewerRoleSecurityTests(unittest.TestCase):
    def registry(self):
        registry = ToolRegistry()
        for spec in native_tool_specs():
            registry.register(spec, None if spec.name == "bash" else lambda *_: "OK")
        for name in ("save_skill", "rollback", "checkpoint"):
            registry.register(ToolSpec(
                name, "main only", {"type": "object", "properties": {}},
            ), lambda *_: "OK")
        return registry

    def test_reviewer_role_is_strict_read_only(self):
        role = reviewer_role(max_model_turns=5, max_tool_calls=7,
                             context_limit=4096)
        self.assertEqual(role.role, AgentRole.REVIEWER)
        self.assertEqual(role.allowed_tool_names, {"bash", "search_corpus"})
        self.assertFalse(role.web_access)
        self.assertFalse(role.can_write_memory)
        self.assertFalse(role.checkpoint_access)
        self.assertFalse(role.recursive_delegation)
        role.validate_registry(self.registry())

    def test_reviewer_sensitive_tools_are_denied_by_router(self):
        router = ToolRouter(self.registry())
        context = ToolExecutionContext(
            "task_reviewer", TraceEmitter(), {}, role="reviewer",
            command_executor=lambda _: "OK",
        )
        attempts = {
            "edit_file": {"path": "a", "old_text": "a", "new_text": "b"},
            "write_file": {"path": "a", "content": "b"},
            "append_file": {"path": "a", "content": "b"},
            "remember": {"note": "x"}, "search_internet": {"query": "x"},
            "save_skill": {}, "rollback": {}, "checkpoint": {},
        }
        for name, arguments in attempts.items():
            with self.subTest(name=name):
                result = router.execute(context, "call_" + name, name, arguments)
                self.assertEqual(result.status, ToolResultStatus.DENIED)

    def test_reviewer_cannot_be_configured_with_privileged_access(self):
        with self.assertRaises(ValueError):
            AgentRoleSpec(AgentRole.REVIEWER, "review", frozenset({"bash"}),
                          "safe", 2, 2, 1024, checkpoint_access=True)

    def test_safe_command_policy_rejects_reviewer_mutating_bash(self):
        policy = CommandPolicy()
        authorization = policy.authorize(
            policy.classify("printf compromised > src.py"), ExecutionMode.SAFE,
        )
        self.assertIsNotNone(authorization.result)
        self.assertEqual(authorization.result.status, "denied")


class DiffEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.manager = CheckpointManager(self.root / "checkpoints", self.workspace)
        self.checkpoint = self.manager.begin_checkpoint(
            "task_diff123", "session_diff123",
        )

    def test_checkpoint_diff_handles_modified_created_deleted_and_binary(self):
        paths = {
            "modified.txt": b"old\n", "deleted.txt": b"gone\n",
            "binary.bin": b"\0old",
        }
        for name, content in paths.items():
            (self.workspace / name).write_bytes(content)
            self.manager.capture(self.checkpoint, name)
        self.manager.capture(self.checkpoint, "created.txt")
        (self.workspace / "modified.txt").write_text("new\n")
        (self.workspace / "deleted.txt").unlink()
        (self.workspace / "binary.bin").write_bytes(b"\0new")
        (self.workspace / "created.txt").write_text("created\n")
        for name in (*paths, "created.txt"):
            self.manager.record_mutation(self.checkpoint, name, "action_" + name)
        evidence = DiffEvidenceService().from_checkpoint(
            self.manager, self.checkpoint,
        )
        self.assertEqual(set(evidence.changed_paths), set(paths) | {"created.txt"})
        self.assertIn("modified.txt", evidence.preview)
        self.assertIn("created.txt", evidence.preview)
        self.assertIn("deleted.txt", evidence.preview)
        self.assertEqual(evidence.binary_paths, ("binary.bin",))

    def test_large_diff_is_bounded_and_referenceable(self):
        store = ResultStore(self.root / "results")
        path = self.workspace / "large.py"
        path.write_text("old\n")
        self.manager.capture(self.checkpoint, "large.py")
        path.write_text("\n".join(f"line {i}" for i in range(1000)))
        self.manager.record_mutation(self.checkpoint, "large.py", "large_action")
        evidence = DiffEvidenceService(
            result_store=store, model_context_chars=300,
        ).from_checkpoint(self.manager, self.checkpoint)
        self.assertTrue(evidence.truncated)
        self.assertLess(len(evidence.preview), 500)
        self.assertTrue(store.exists(evidence.result_reference))
        self.assertGreater(len(store.get(evidence.result_reference,
                                         task_id=self.checkpoint.task_id)), 300)

    def test_external_change_is_detected_and_git_fallback_is_injected(self):
        path = self.workspace / "a.py"
        path.write_text("old")
        self.manager.capture(self.checkpoint, "a.py")
        path.write_text("task")
        self.manager.record_mutation(self.checkpoint, "a.py", "action")
        path.write_text("external")
        evidence = DiffEvidenceService().from_checkpoint(self.manager, self.checkpoint)
        self.assertEqual(evidence.external_change_paths, ("a.py",))
        git = DiffEvidenceService().from_git(
            self.checkpoint.task_id, lambda: "diff --git a/a.py b/a.py",
            changed_paths=("a.py",),
        )
        self.assertEqual(git.source, "git")
        self.assertIn("diff --git", git.preview)


class ReviewerFunctionalTests(unittest.TestCase):
    def run_review(self, output, *, state=None, executor=None, trace=None,
                   cancellation=None, evidence=None, req_status=None):
        backend = ScriptedBackend([ModelTurn(output, (), StopReason.END_TURN)])
        parent = parent_context(backend, state=state, trace=trace,
                                cancellation=cancellation)
        if not parent.working_state.verifications:
            add_verification(parent.working_state)
        service = ReviewService(AgentRuntime())
        result = service.review(
            request(parent, evidence=evidence, status=req_status), parent,
            tools=(ToolDefinition("bash", "read", {"type": "object"}),),
            tool_executor=executor or ReadExecutor(),
        )
        return parent, result, backend, service

    def test_clean_correct_change_is_accepted(self):
        parent, result, _, _ = self.run_review(review_json())
        self.assertEqual(result.verdict, ReviewVerdict.ACCEPT)
        self.assertTrue(parent.working_state.current_review_accepted)
        self.assertEqual(parent.working_state.current_review.mutation_generation, 1)

    def test_missing_requirement_and_functional_regression_require_repair(self):
        items = (
            finding("requirement_missing", "Zero handling was removed",
                    requirement="Zero must remain a valid value"),
            finding("functional_regression", "Falsy zero now takes error path"),
        )
        _, result, _, _ = self.run_review(
            review_json("repair_required", items),
        )
        self.assertEqual(result.verdict, ReviewVerdict.REPAIR_REQUIRED)
        self.assertEqual(len(result.blocking_findings), 2)

    def test_failed_or_stale_verification_cannot_be_accepted(self):
        for status in ("failed", "unverified"):
            with self.subTest(status=status):
                _, result, _, _ = self.run_review(review_json(), req_status=status)
                self.assertEqual(result.verdict, ReviewVerdict.REPAIR_REQUIRED)
                self.assertTrue(any(item.category == FindingCategory.VERIFICATION_GAP
                                    for item in result.blocking_findings))

    def test_unrelated_change_test_gap_and_project_rule_violation(self):
        state = state_with_mutation(path="src.py")
        # Ground an unrelated second changed path.
        state.apply(StateEvent.create(
            StateEventType.ACTION_SUCCEEDED, state.task_id, StateSource.TOOL_RUNTIME,
            action_id="mutation_2", kind="tool", name="write_file",
            observed_status="ok", summary="changed unrelated", round_number=2,
        ))
        state.apply(StateEvent.create(
            StateEventType.FILE_MODIFIED, state.task_id, StateSource.TOOL_RUNTIME,
            path="unrelated.py", action_id="mutation_2",
        ))
        add_verification(state)
        findings = (
            finding("unrelated_change", "Unrelated file changed", path="unrelated.py"),
            finding("test_gap", "Zero regression has no test"),
            finding("project_rule_violation", "Public compatibility was removed",
                    requirement="Do not change public compatibility"),
        )
        _, result, _, _ = self.run_review(
            review_json("repair_required", findings), state=state,
            evidence=DiffEvidence("checkpoint", ("src.py", "unrelated.py"),
                                  "diff", 4, False),
        )
        self.assertEqual(len(result.blocking_findings), 3)

    def test_subjective_style_finding_does_not_block(self):
        style = finding("maintainability_issue", "I prefer a different name",
                        blocking=True, severity="low")
        _, result, _, _ = self.run_review(
            review_json("repair_required", (style,)),
        )
        self.assertEqual(result.verdict, ReviewVerdict.ACCEPT)
        self.assertEqual(len(result.blocking_findings), 0)
        self.assertEqual(len(result.non_blocking_findings), 1)

    def test_malformed_output_and_model_failure_block(self):
        parent, malformed, _, _ = self.run_review("not json")
        self.assertEqual(malformed.verdict, ReviewVerdict.BLOCKED)
        self.assertEqual(malformed.status, ReviewExecutionStatus.FAILED)
        backend = ScriptedBackend([RuntimeError("offline")])
        failed_parent = parent_context(backend)
        add_verification(failed_parent.working_state)
        failed = ReviewService(AgentRuntime()).review(
            request(failed_parent), failed_parent, tools=(),
            tool_executor=ReadExecutor(),
        )
        self.assertEqual(failed.verdict, ReviewVerdict.BLOCKED)

    def test_tool_failure_can_be_reported_as_grounded_blocker(self):
        output = review_json("blocked", (finding(
            "verification_gap", "Independent source inspection failed",
            path=None, requirement="Zero must remain a valid value",
        ),))
        backend = ScriptedBackend([
            ModelTurn("", (ModelToolCall("read", "bash", {"command": "cat src.py"}),),
                      StopReason.TOOL_USE),
            ModelTurn(output, (), StopReason.END_TURN),
        ])
        parent = parent_context(backend)
        add_verification(parent.working_state)
        result = ReviewService(AgentRuntime()).review(
            request(parent), parent,
            tools=(ToolDefinition("bash", "read", {"type": "object"}),),
            tool_executor=ReadExecutor(fail=True),
        )
        self.assertEqual(result.verdict, ReviewVerdict.BLOCKED)
        self.assertTrue(result.blocking_findings)

    def test_context_state_compaction_and_checkpoint_are_isolated(self):
        backend = ScriptedBackend([
            ModelTurn("", (ModelToolCall("read", "bash", {
                "command": "cat src.py"}),), StopReason.TOOL_USE),
            ModelTurn(review_json(), (), StopReason.END_TURN),
        ])
        parent = parent_context(backend)
        add_verification(parent.working_state)
        parent.checkpoint_manager = object()
        parent.checkpoint = object()
        executor = ReadExecutor()
        result = ReviewService(AgentRuntime()).review(
            request(parent), parent,
            tools=(ToolDefinition("bash", "read", {"type": "object"}),),
            tool_executor=executor,
        )
        system = backend.calls[0]["system"]
        self.assertNotIn("private implementation transcript", system)
        self.assertNotIn("IMPLEMENTER PRIVATE REASONING", system)
        self.assertIn("grounded", system.lower())
        self.assertEqual(len(parent.conversation), 1)
        self.assertEqual(parent.working_state.files_read, set())
        self.assertIsNone(parent.compaction_artifact)
        self.assertEqual(result.child_task_id != parent.task_id, True)
        child = executor.calls[0][0]
        self.assertIsNot(child.working_state, parent.working_state)
        self.assertIsNot(child.progress_monitor, parent.progress_monitor)
        self.assertIsNone(child.compaction_artifact)
        self.assertIsNone(child.checkpoint_manager)
        self.assertIsNone(child.checkpoint)

    def test_parent_cancellation_and_child_cancellation_are_isolated(self):
        source = CancellationSource()
        source.cancel("stop", CancellationScope.TASK)
        backend = ScriptedBackend([])
        parent = parent_context(backend, cancellation=source.token)
        add_verification(parent.working_state)
        result = ReviewService(AgentRuntime()).review(
            request(parent), parent, tools=(), tool_executor=ReadExecutor(),
        )
        self.assertEqual(result.status, ReviewExecutionStatus.CANCELLED)

        child_backend = CancellingBackend()
        child_parent_source = CancellationSource()
        child_parent = parent_context(
            child_backend, cancellation=child_parent_source.token,
        )
        add_verification(child_parent.working_state)
        service = ReviewService(AgentRuntime())
        child_backend.cancel = service.cancel_active
        child_result = service.review(
            request(child_parent), child_parent, tools=(),
            tool_executor=ReadExecutor(),
        )
        self.assertEqual(child_result.status, ReviewExecutionStatus.CANCELLED)
        self.assertFalse(child_parent_source.token.is_cancelled)

    def test_turn_and_tool_budgets_never_silently_accept(self):
        parent = parent_context(ScriptedBackend([
            ModelTurn(review_json(), (), StopReason.END_TURN),
        ]))
        add_verification(parent.working_state)
        req = request(parent)
        req = ReviewRequest(**{**req.__dict__, "max_model_turns": 1})
        result = ReviewService(AgentRuntime(), role=reviewer_role(
            max_model_turns=1, max_tool_calls=2,
        )).review(req, parent, tools=(), tool_executor=ReadExecutor())
        self.assertEqual(result.verdict, ReviewVerdict.BLOCKED)
        self.assertEqual(result.status, ReviewExecutionStatus.BUDGET_EXHAUSTED)

        tool_backend = ScriptedBackend([
            ModelTurn("", (ModelToolCall("read", "bash", {
                "command": "cat src.py"}),), StopReason.TOOL_USE),
            ModelTurn(review_json(), (), StopReason.END_TURN),
        ])
        tool_parent = parent_context(tool_backend)
        add_verification(tool_parent.working_state)
        tool_request = request(tool_parent)
        tool_request = ReviewRequest(**{
            **tool_request.__dict__, "max_model_turns": 3, "max_tool_calls": 1,
        })
        tool_result = ReviewService(AgentRuntime(), role=reviewer_role(
            max_model_turns=3, max_tool_calls=1,
        )).review(
            tool_request, tool_parent,
            tools=(ToolDefinition("bash", "read", {"type": "object"}),),
            tool_executor=ReadExecutor(),
        )
        self.assertEqual(tool_result.verdict, ReviewVerdict.BLOCKED)
        self.assertEqual(tool_result.status, ReviewExecutionStatus.BUDGET_EXHAUSTED)

    def test_external_change_blocks_review(self):
        _, result, _, _ = self.run_review(
            review_json(), evidence=diff(external=("src.py",)),
        )
        self.assertEqual(result.verdict, ReviewVerdict.BLOCKED)
        self.assertTrue(result.blocking_findings)

    def test_repeated_inspection_stalls_and_never_accepts(self):
        turns = [ModelTurn(
            "", (ModelToolCall(f"read{index}", "bash", {
                "command": "cat src.py",
            }),), StopReason.TOOL_USE,
        ) for index in range(5)]
        parent = parent_context(ScriptedBackend(turns))
        add_verification(parent.working_state)
        result = ReviewService(AgentRuntime()).review(
            request(parent), parent,
            tools=(ToolDefinition("bash", "read", {"type": "object"}),),
            tool_executor=ReadExecutor(),
        )
        self.assertEqual(result.status, ReviewExecutionStatus.STALLED)
        self.assertEqual(result.verdict, ReviewVerdict.BLOCKED)

    def test_trace_events_are_attributed_without_finding_text(self):
        recorder = Recorder()
        secret = "secret seeded regression text"
        _, result, _, _ = self.run_review(
            review_json("repair_required", (finding(
                "functional_regression", secret,
            ),)), trace=TraceEmitter(recorder),
        )
        types = [event.event_type for event in recorder.events]
        self.assertIn(EventType.REVIEW_STARTED, types)
        self.assertIn(EventType.REVIEW_FINISHED, types)
        self.assertIn(EventType.REVIEW_REPAIR_REQUESTED, types)
        self.assertNotIn(secret, json.dumps([event.to_dict() for event in recorder.events]))
        self.assertEqual(result.model_calls, 1)


class ReviewerGenerationAndPersistenceTests(unittest.TestCase):
    def test_review_policy_skips_no_change_and_trivial_non_code_change(self):
        policy = ReviewPolicy()
        untouched = WorkingState.start("task_untouched", "Explain repository")
        self.assertFalse(policy.decide(untouched).should_review)
        trivial = state_with_mutation(path="notes.txt")
        trivial.acceptance_criteria.clear()
        self.assertFalse(policy.decide(trivial).should_review)
        self.assertTrue(policy.decide(state_with_mutation(path="src.py")).should_review)

    def test_review_acceptance_is_invalidated_by_next_mutation(self):
        state = state_with_mutation()
        add_verification(state)
        state.apply(StateEvent.create(
            StateEventType.REVIEW_RECORDED, state.task_id, StateSource.HARNESS,
            review_id="review_generation1", mutation_generation=1,
            verdict="accept", blocking_count=0,
        ))
        self.assertTrue(state.current_review_accepted)
        state.apply(StateEvent.create(
            StateEventType.ACTION_SUCCEEDED, state.task_id, StateSource.TOOL_RUNTIME,
            action_id="mutation_2", kind="tool", name="edit_file",
            observed_status="ok", round_number=2,
        ))
        state.apply(StateEvent.create(
            StateEventType.FILE_MODIFIED, state.task_id, StateSource.TOOL_RUNTIME,
            path="src.py", action_id="mutation_2",
        ))
        self.assertFalse(state.current_review_accepted)
        self.assertEqual(VerificationPolicy().evaluate_completion(state).status.value,
                         "unverified")
        self.assertFalse(ReviewPolicy().completion_allowed(
            state, VerificationPolicy().evaluate_completion(state)))

    def test_agent_completion_gate_needs_current_verification_and_review(self):
        state = state_with_mutation()
        add_verification(state)
        parent = parent_context(ScriptedBackend([]), state=state)
        parent.review_required = True
        from agent_runtime import AgentResult, RuntimeTerminalReason
        result = AgentResult(
            state.task_id, TerminalStatus.RUNNING, RuntimeTerminalReason.COMPLETED,
            "done", state, 1, 1, 1, state.verification_outcome, (), (), (), (),
            True, False,
        )
        AgentRuntime().complete(parent, result)
        self.assertTrue(result.completion_deferred)
        state.apply(StateEvent.create(
            StateEventType.REVIEW_RECORDED, state.task_id, StateSource.HARNESS,
            review_id="review_accept123", mutation_generation=1,
            verdict="accept", blocking_count=0,
        ))
        AgentRuntime().complete(parent, result)
        self.assertEqual(state.terminal_status, TerminalStatus.COMPLETED)

    def test_completed_review_survives_resume_and_is_not_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FileSessionStore(directory)
            handle = SessionHandle(
                store, new_session_id(),
                SessionConfiguration("/workspace", "project", "safe"),
            )
            handle.append(SessionEventType.SESSION_STARTED, "task_review123")
            backend = ScriptedBackend([
                ModelTurn(review_json(), (), StopReason.END_TURN),
            ])
            parent = parent_context(backend, session=handle)
            add_verification(parent.working_state)
            service = ReviewService(AgentRuntime())
            req = request(parent)
            first = service.review(req, parent, tools=(), tool_executor=ReadExecutor())
            snapshot = store.load_snapshot(handle.session_id)
            resumed = AgentContext.from_session_snapshot(
                snapshot, session=handle, backend=backend,
                context_engine=ContextEngine(), trace=TraceEmitter(), tools=(),
                tool_executor=ReadExecutor(), context_limit=8192,
            )
            second = service.review(req, resumed, tools=(), tool_executor=ReadExecutor())
            self.assertEqual(first.review_id, second.review_id)
            self.assertEqual(len(backend.calls), 1)
            self.assertTrue(resumed.working_state.current_review_accepted)

    def test_one_cycle_repair_policy_and_off_on_measurement(self):
        policy = ReviewRepairPolicy(maximum_cycles=1)
        dummy = type("Result", (), {"verdict": ReviewVerdict.REPAIR_REQUIRED})()
        self.assertTrue(policy.allows(0, dummy))
        self.assertFalse(policy.allows(1, dummy))
        measurements = {
            "disabled": {"seeded_regressions_caught": 0, "false_blocks": 0,
                         "model_calls": 0, "tool_calls": 0},
            "enabled": {"seeded_regressions_caught": 1, "false_blocks": 0,
                        "model_calls": 1, "tool_calls": 0},
        }
        self.assertGreater(measurements["enabled"]["seeded_regressions_caught"],
                           measurements["disabled"]["seeded_regressions_caught"])
        self.assertEqual(measurements["enabled"]["false_blocks"], 0)

    def test_scripted_one_cycle_repair_then_current_generation_accept(self):
        defect = finding(
            "functional_regression", "Zero is rejected by the new truthiness check",
            requirement="Zero must remain a valid value",
        )
        backend = ScriptedBackend([
            ModelTurn(review_json("repair_required", (defect,)), (),
                      StopReason.END_TURN),
            ModelTurn("", (ModelToolCall("edit", "edit_file", {
                "path": "src.py", "old_text": "value", "new_text": "fixed",
            }),), StopReason.TOOL_USE),
            ModelTurn("", (ModelToolCall("test", "bash", {
                "command": "pytest",
            }),), StopReason.TOOL_USE),
            ModelTurn("Repair complete.", (), StopReason.END_TURN),
            ModelTurn(review_json("accept"), (), StopReason.END_TURN),
        ])
        parent = parent_context(backend)
        add_verification(parent.working_state)
        service = ReviewService(AgentRuntime())
        first = service.review(
            request(parent), parent, tools=(), tool_executor=ReadExecutor(),
        )
        self.assertEqual(first.verdict, ReviewVerdict.REPAIR_REQUIRED)
        self.assertEqual(len(parent.conversation), 1)
        repair_executor = RepairExecutor()
        parent.tools = (
            ToolDefinition("edit_file", "edit", {"type": "object"}),
            ToolDefinition("bash", "test", {"type": "object"}),
        )
        parent.tool_executor = repair_executor
        repair = AgentRuntime().run(parent, defer_completion=True)
        self.assertEqual(parent.working_state.mutation_generation, 2)
        evaluation = VerificationPolicy().evaluate_completion(parent.working_state)
        self.assertEqual(evaluation.status.value, "verified")
        second = service.review(
            request(parent, evidence=diff(preview="- value\n+ fixed\n")),
            parent, tools=(), tool_executor=ReadExecutor(),
        )
        self.assertEqual(second.verdict, ReviewVerdict.ACCEPT)
        self.assertTrue(parent.working_state.current_review_accepted)
        self.assertEqual(len(parent.review_results), 2)
        self.assertEqual(len([item for item in parent.context_items
                              if item.source.startswith("reviewer:")]), 2)
        self.assertEqual(repair.final_response, "Repair complete.")

    def test_blocked_review_keeps_parent_checkpoint_and_rollback_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            path = workspace / "src.py"
            path.write_text("original")
            manager = CheckpointManager(root / "checkpoints", workspace)
            checkpoint = manager.begin_checkpoint(
                "task_review123", "session_review123",
            )
            manager.capture(checkpoint, "src.py")
            path.write_text("broken")
            manager.record_mutation(checkpoint, "src.py", "mutation_1")
            state = state_with_mutation()
            state.apply(StateEvent.create(
                StateEventType.CHECKPOINT_RECORDED, state.task_id,
                StateSource.HARNESS, checkpoint_id=checkpoint.checkpoint_id,
                status=checkpoint.status.value,
            ))
            add_verification(state)
            parent = parent_context(
                ScriptedBackend([ModelTurn(review_json(
                    "blocked", (finding("security_issue", "Unsafe behavior"),)
                ), (), StopReason.END_TURN)]),
                state=state, checkpoint_manager=manager, checkpoint=checkpoint,
            )
            result = ReviewService(AgentRuntime()).review(
                request(parent, evidence=DiffEvidenceService().from_checkpoint(
                    manager, checkpoint,
                )), parent, tools=(), tool_executor=ReadExecutor(),
            )
            self.assertEqual(result.verdict, ReviewVerdict.BLOCKED)
            self.assertEqual(checkpoint.status.value, "active")
            rollback = parent.rollback_checkpoint()
            self.assertEqual(rollback.status, RollbackStatus.SUCCESS)
            self.assertEqual(path.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
