import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_runtime import AgentRuntime
from checkpoint import CheckpointManager
from context_engine import ContextItem, ContextLayer, Freshness
from task_controller import TaskController, TaskRequest, TaskStatus
from tests.test_agent_runtime import (
    GroundedToolExecutor, ScriptedBackend, make_context, text_turn, tool_turn,
)
from tests.test_task_controller import registry
from training_data import (
    TRAINING_SCHEMA_VERSION, TrainingDataError, TrainingEligibility,
    TrainingEpisode, TrainingRecorder, TrainingRedactionPolicy,
)
from training_store import TrainingStore, TrainingStoreError
from verification import CompletionVerificationStatus
from working_state import StateEventType


def make_controller(context, store, **kwargs):
    return TaskController(
        AgentRuntime(), registry(), tool_executor=context.tool_executor,
        training_store=store, **kwargs,
    )


def one_episode(store):
    metadata = list(store.iterate_metadata())
    if len(metadata) != 1:
        raise AssertionError(metadata)
    return store.load_episode(metadata[0]["episode_id"])


class TrainingCaptureTests(unittest.TestCase):
    def test_readiness_update_failure_does_not_change_task_success(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrainingStore(Path(directory) / "training")
            context = make_context(ScriptedBackend([text_turn("answer")]))
            with patch("training_readiness.TrainingReadinessIndex.update_episode",
                       side_effect=RuntimeError("secondary failure")):
                result = make_controller(context, store).run(TaskRequest(
                    "answer", context, (directory,), enable_planning=False,
                ))
            self.assertEqual(result.status, TaskStatus.COMPLETED)
            self.assertEqual(one_episode(store).outcome["terminal_task_status"],
                             TaskStatus.COMPLETED.value)

    def test_successful_informational_task_and_controller_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrainingStore(Path(directory) / "training")
            context = make_context(ScriptedBackend([text_turn("grounded answer")]))
            result = make_controller(context, store).run(TaskRequest(
                "answer a local question", context, (directory,),
                enable_planning=False,
            ))
            episode = one_episode(store)
            self.assertEqual(result.status, TaskStatus.COMPLETED)
            self.assertEqual(episode.schema_version, TRAINING_SCHEMA_VERSION)
            self.assertEqual(episode.training_metadata["eligibility"],
                             TrainingEligibility.POSITIVE_CANDIDATE.value)
            self.assertEqual(episode.turns[0].assistant_output, "grounded answer")
            self.assertTrue(episode.turns[0].primary_candidate)
            self.assertEqual(episode.outcome["final_response"], "grounded answer")
            self.assertEqual(episode.provenance["data_origin"], "NORMAL_USAGE")

    def test_failed_action_recovery_verified_mutation_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            (workspace / "service.py").write_text("value = 1\n")
            store = TrainingStore(Path(directory) / "training")
            backend = ScriptedBackend([
                tool_turn("bad", "bash", command="false"),
                tool_turn("read", "read_file", path="service.py"),
                tool_turn("edit", "edit_file", path="service.py",
                          old_text="1", new_text="2"),
                tool_turn("verify", "bash", command="pytest"),
                text_turn("implemented and verified"),
            ])
            executor = GroundedToolExecutor([
                "ERROR: failed (exit 2)", "value = 1", "OK: edited", "1 passed",
            ])
            context = make_context(backend, executor=executor, rounds=10, actions=12)
            manager = CheckpointManager(Path(directory) / "checkpoints", workspace)
            checkpoint = manager.begin_checkpoint(context.task_id, "session_training")
            context.checkpoint_manager = manager
            context.checkpoint = checkpoint
            context.apply_state_event(
                StateEventType.CHECKPOINT_RECORDED,
                checkpoint_id=checkpoint.checkpoint_id,
                status=checkpoint.status.value,
            )

            def bench(ctx):
                evidence = ctx.verification_policy.project_bench_evidence(
                    ctx.working_state, executed=True, passed=True, action_id="bench",
                )
                AgentRuntime._record_verification(ctx, evidence)
                return True

            result = make_controller(
                context, store, verification_runner=bench,
            ).run(TaskRequest(
                "fix and verify service.py", context, (str(workspace),),
                enable_planning=False,
            ))
            episode = one_episode(store)
            self.assertEqual(result.verification.status,
                             CompletionVerificationStatus.VERIFIED)
            self.assertEqual(episode.training_metadata["eligibility"],
                             TrainingEligibility.POSITIVE_CANDIDATE.value)
            failed = next(turn for turn in episode.turns
                          if "failed_tool_action" in turn.labels)
            recovery = next(turn for turn in episode.turns
                            if "recovery_after_failure" in turn.labels)
            mutation = next(turn for turn in episode.turns
                            if "grounded_mutation" in turn.labels)
            self.assertIn(failed.turn_id, recovery.recovers_turn_ids)
            self.assertNotIn(failed.turn_id,
                             episode.training_metadata["default_sft_turns"])
            self.assertGreater(mutation.mutation_generation_after,
                               mutation.mutation_generation_before)
            bad_call = failed.tool_calls[0]
            self.assertEqual(bad_call.model_content, "ERROR: failed (exit 2)")
            self.assertEqual(bad_call.status, "failed")
            self.assertEqual(episode.outcome["checkpoint_status"], "finalized")
            self.assertTrue(episode.execution["verifications"])
            self.assertNotIn("reasoning", json.dumps(episode.to_dict()).lower())

    def test_unverified_and_failed_verification_are_not_positive(self):
        for output in (None, "ERROR: tests failed (exit 2)"):
            with self.subTest(output=output), tempfile.TemporaryDirectory() as directory:
                turns = [
                    tool_turn("edit", "edit_file", path="a.py", old_text="a", new_text="b"),
                ]
                outputs = ["OK: edited"]
                if output:
                    turns.append(tool_turn("test", "bash", command="pytest"))
                    outputs.append(output)
                turns.append(text_turn("done"))
                context = make_context(
                    ScriptedBackend(turns), executor=GroundedToolExecutor(outputs),
                    rounds=8,
                )
                store = TrainingStore(Path(directory) / "training")
                make_controller(context, store).run(TaskRequest(
                    "change a.py", context, (directory,), enable_planning=False,
                ))
                episode = one_episode(store)
                self.assertEqual(episode.training_metadata["eligibility"],
                                 TrainingEligibility.FAILURE_TRAJECTORY.value)

    def test_later_passing_verification_preserves_failed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            context = make_context(ScriptedBackend([
                tool_turn("edit", "edit_file", path="a.py", old_text="a", new_text="b"),
                tool_turn("fail", "bash", command="pytest first"),
                tool_turn("pass", "bash", command="pytest second"),
                text_turn("done"),
            ]), executor=GroundedToolExecutor([
                "OK: edited", "ERROR: failed (exit 2)", "1 passed",
            ]), rounds=8)
            store = TrainingStore(Path(directory) / "training")
            make_controller(context, store).run(TaskRequest(
                "change and test a.py", context, (directory,), enable_planning=False,
            ))
            episode = one_episode(store)
            outcomes = [item["outcome"] for item in episode.execution["verifications"]]
            self.assertIn("failed", outcomes)
            self.assertIn("passed", outcomes)

    def test_cancelled_budget_and_stalled_are_failure_trajectories(self):
        from budgets import BudgetKind, BudgetLimit, BudgetManager
        from cancellation import CancellationSource

        cases = []
        cancelled = make_context(ScriptedBackend([text_turn()]), task_id="cancelled")
        source = CancellationSource()
        source.cancel("stop")
        cancelled.cancellation = source.token
        cases.append(cancelled)
        budget = make_context(ScriptedBackend([text_turn()]), task_id="budget")
        budget.budget_manager = BudgetManager("task", {
            BudgetKind.PRIMARY_MODEL_CALLS: BudgetLimit(0),
        })
        cases.append(budget)
        stalled = make_context(
            ScriptedBackend([tool_turn(str(i), "bash", command=f"cmd {i}")
                             for i in range(4)]),
            executor=GroundedToolExecutor(["same"] * 4), task_id="stalled",
            rounds=4,
        )
        stalled.progress_monitor.stall_threshold = 1
        cases.append(stalled)
        for context in cases:
            with self.subTest(task=context.task_id), tempfile.TemporaryDirectory() as directory:
                store = TrainingStore(Path(directory) / "training")
                make_controller(context, store).run(TaskRequest(
                    "exercise terminal state", context, (directory,),
                    enable_planning=False,
                ))
                self.assertEqual(one_episode(store).training_metadata["eligibility"],
                                 TrainingEligibility.FAILURE_TRAJECTORY.value)

    def test_compacted_exact_input_and_memory_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = ContextItem(
                "memory:item-7", ContextLayer.DURABLE_MEMORY, "memory_store",
                "remember this", priority=100, freshness=Freshness.CURRENT,
                protected=True, inclusion_reason="selected memory",
            )
            backend = ScriptedBackend([text_turn("answer")])
            context = make_context(backend, extra_items=(memory,))
            store = TrainingStore(Path(directory) / "training")
            make_controller(context, store).run(TaskRequest(
                "answer", context, (directory,), enable_planning=False,
            ))
            turn = one_episode(store).turns[0]
            self.assertEqual(turn.system, backend.calls[0]["system"])
            self.assertEqual(list(turn.messages), [
                {"role": message.role, "content": [
                    {"type": "text", "text": block.text} for block in message.content
                ]} for message in backend.calls[0]["conversation"]
            ])
            self.assertIn("memory:item-7", turn.memory_provenance_ids)

    def test_auxiliary_role_is_not_default_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrainingStore(Path(directory) / "training")
            context = make_context(ScriptedBackend([text_turn("report")]))
            context.role = "explorer"
            recorder = TrainingRecorder(
                store, task_id=context.task_id, session_id=None,
                objective="explore", workspace=directory,
            )
            context.training_recorder = recorder
            result = AgentRuntime().run(context)
            recorder.finalize(
                state=context.working_state, task_status="completed",
                terminal_reason=result.terminal_reason.value,
                verification_status="not_required", checkpoint_status=None,
                final_response=result.final_response,
            )
            episode = one_episode(store)
            self.assertFalse(episode.turns[0].primary_candidate)
            self.assertEqual(episode.training_metadata["default_sft_turns"], [])

    def test_two_tasks_do_not_leak_state_and_disabled_creates_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrainingStore(Path(directory) / "training")
            for number in range(2):
                context = make_context(
                    ScriptedBackend([text_turn(f"answer {number}")]),
                    task_id=f"task_{number}",
                )
                make_controller(context, store).run(TaskRequest(
                    f"objective {number}", context, (directory,), enable_planning=False,
                ))
            episodes = [store.load_episode(item["episode_id"])
                        for item in store.iterate_metadata()]
            self.assertEqual({item.task_id for item in episodes}, {"task_0", "task_1"})
            self.assertNotEqual(episodes[0].episode_id, episodes[1].episode_id)

            disabled_root = Path(directory) / "disabled"
            context = make_context(ScriptedBackend([text_turn("answer")]),
                                   task_id="disabled")
            TaskController(AgentRuntime(), registry(),
                           tool_executor=context.tool_executor).run(TaskRequest(
                               "answer", context, (directory,), enable_planning=False,
                           ))
            self.assertFalse(disabled_root.exists())


class TrainingSecurityAndStoreTests(unittest.TestCase):
    def test_redacts_secrets_arguments_results_and_environment_values(self):
        policy = TrainingRedactionPolicy(["environment-secret-value"])
        value, metadata = policy.redact({
            "authorization": "Bearer abcdef",
            "text": "token=abc123 environment-secret-value sk-abcdefghijklmnop",
            "input_tokens": 12,
        })
        rendered = json.dumps(value)
        self.assertNotIn("abcdef", rendered)
        self.assertNotIn("environment-secret-value", rendered)
        self.assertNotIn("sk-abcdefghijklmnop", rendered)
        self.assertEqual(value["input_tokens"], 12)
        self.assertGreater(metadata["redaction_count"], 0)

    def test_draft_is_incomplete_and_finalization_atomic_idempotent_private(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrainingStore(Path(directory) / "training")
            recorder = TrainingRecorder(
                store, task_id="task", session_id="session", objective="answer",
                workspace=directory,
            )
            draft = store.drafts / f"{recorder.episode_id}.json"
            self.assertEqual(json.loads(draft.read_text())["status"], "incomplete")
            context = make_context(ScriptedBackend([text_turn("answer")]))
            result = AgentRuntime().run(context)
            episode = recorder.finalize(
                state=context.working_state, task_status="completed",
                terminal_reason=result.terminal_reason.value,
                verification_status="not_required", checkpoint_status=None,
                final_response="answer",
            )
            first = store.save_episode(episode)
            second = store.save_episode(episode)
            self.assertEqual(first, second)
            self.assertFalse(draft.exists())
            self.assertEqual(os.stat(first).st_mode & 0o777, 0o600)
            self.assertEqual(len(list(store.iterate_metadata())), 1)

    def test_schema_malformed_checksum_and_traversal_rejected(self):
        with self.assertRaises(TrainingDataError):
            TrainingEpisode.from_dict({"schema_version": 999})
        with self.assertRaises(TrainingDataError):
            TrainingEpisode.from_dict({"schema_version": TRAINING_SCHEMA_VERSION})
        with tempfile.TemporaryDirectory() as directory:
            store = TrainingStore(Path(directory) / "training")
            for unsafe in ("../escape", "/absolute", "a/b", ""):
                with self.subTest(unsafe=unsafe), self.assertRaises(TrainingStoreError):
                    store.exists(unsafe)

    def test_result_reference_kept_without_full_result_duplication(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrainingStore(Path(directory) / "training")
            huge = "bounded"

            class ReferencingExecutor(GroundedToolExecutor):
                def __call__(self, context, tool_call_id, name, arguments, cache):
                    envelope = super().__call__(context, tool_call_id, name, arguments, cache)
                    from dataclasses import replace
                    return replace(envelope, model_content=huge,
                                   text="X" * 100_000,
                                   result_reference="result_abcdef")

            context = make_context(
                ScriptedBackend([tool_turn("read", "read_file", path="a"),
                                 text_turn("done")]),
                executor=ReferencingExecutor(["original"]),
            )
            make_controller(context, store).run(TaskRequest(
                "read a", context, (directory,), enable_planning=False,
            ))
            episode = one_episode(store)
            call = episode.turns[0].tool_calls[0]
            self.assertEqual(call.model_content, "bounded")
            self.assertEqual(call.result_reference, "result_abcdef")
            self.assertNotIn("X" * 1000, json.dumps(episode.to_dict()))


if __name__ == "__main__":
    unittest.main()
