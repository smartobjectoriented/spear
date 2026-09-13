import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sft_dataset import (
    ExportProfile, LengthStatus, ReviewStatus, SFTDatasetBuilder,
    SFTDatasetError, SFTExportConfiguration, SFTSampleType,
    SplitConfiguration,
)
from training_data import (
    TRAINING_SCHEMA_VERSION, TrainingEpisode, TrainingToolCall, TrainingTurn,
)
from training_store import TrainingStore


TOOL_SNAPSHOT = {"schema_version": 1, "tools": [{
    "name": "read_file", "description": "Read", "input_schema": {
        "type": "object", "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}, {
    "name": "edit_file", "description": "Edit", "input_schema": {
        "type": "object", "properties": {"path": {"type": "string"}},
    },
}, {
    "name": "bash", "description": "Run", "input_schema": {
        "type": "object", "properties": {"command": {"type": "string"}},
    },
}]}
VIEW_HASH = hashlib.sha256(json.dumps(
    TOOL_SNAPSHOT, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
).encode()).hexdigest()
TOOLS = {VIEW_HASH: TOOL_SNAPSHOT}


def turn(index, *, output="", calls=(), labels=(), messages=None,
         provenance=("generated",), verification_generation=None,
         mutation_before=0, mutation_after=0, recovers=(), role="main",
         primary=True):
    return TrainingTurn(
        turn_id=f"turn_{index:06d}", episode_id="episode_test", task_id="task",
        session_id=None, turn_index=index, role=role, purpose="primary_agent",
        primary_candidate=primary, system="system", messages=tuple(messages or ({
            "role": "user", "content": ({"type": "text", "text": "do task"},),
        },)), tool_view_hash=VIEW_HASH,
        tool_names=("read_file", "edit_file", "bash"),
        tools_enabled=True, assistant_output=output,
        stop_reason="tool_use" if calls else "end_turn", tool_calls=tuple(calls),
        labels=tuple(labels), recovers_turn_ids=tuple(recovers),
        mutation_generation_before=mutation_before,
        mutation_generation_after=mutation_after,
        verification_generation=verification_generation,
        provenance=tuple(provenance),
    )


def call(call_id, name, arguments, status="ok"):
    return TrainingToolCall(call_id, name, arguments, f"action_{call_id}", status,
                            "bounded result")


def episode(turns, *, episode_id="episode_test", task_id="task", project="proj",
            eligibility="positive_candidate", mutated=0, verified="not_required",
            checkpoint=None, objective="do task"):
    normalized = []
    for item in turns:
        item.episode_id = episode_id
        item.task_id = task_id
        normalized.append(item)
    return TrainingEpisode.from_dict({
        "schema_version": TRAINING_SCHEMA_VERSION, "episode_id": episode_id,
        "task_id": task_id, "session_id": None,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "provenance": {
            "project": project, "workspace_fingerprint": "workspace",
            "harness_git_commit": "a" * 40, "system_prompt_hash": "b" * 64,
        },
        "task": {"objective": objective, "acceptance_criteria": ["done"]},
        "turns": [item.__dict__ | {
            "tool_calls": [value.__dict__ for value in item.tool_calls]
        } for item in normalized],
        "tool_views": TOOLS,
        "execution": {"mutation_generation": mutated},
        "outcome": {
            "terminal_task_status": "completed", "verification_status": verified,
            "checkpoint_status": checkpoint, "unresolved_failures": [],
        },
        "training_metadata": {"eligibility": eligibility},
    })


class SFTDatasetTests(unittest.TestCase):
    def store_episode(self, root, value):
        store = TrainingStore(root)
        store.save_episode(value)
        return store

    def test_positive_final_is_one_target_openai_qwen_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            value = episode([turn(0, output="grounded answer", labels=("final_success",))])
            result = SFTDatasetBuilder(self.store_episode(source, value)).build()
            self.assertEqual(len(result.samples), 1)
            sample = result.samples[0]
            self.assertEqual(sample.sample_type, SFTSampleType.FINAL_RESPONSE.value)
            self.assertEqual(sample.target_message_index, len(sample.messages) - 1)
            self.assertEqual(sample.messages[-1], sample.target)
            self.assertEqual(sample.messages[-1]["role"], "assistant")
            self.assertNotIn("reasoning_content", json.dumps(sample.trainer_record()))
            self.assertEqual(sample.token_count_kind, "approximate")

    def test_verified_mutation_final_and_unverified_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            store = self.store_episode(source, episode([
                turn(0, output="done", labels=("final_success",), mutation_after=1),
            ], mutated=1, verified="verified", checkpoint="finalized"))
            self.assertEqual(len(SFTDatasetBuilder(store).build().samples), 1)
            failed = episode([
                turn(0, output="claimed done", labels=("final_success",), mutation_after=1),
            ], episode_id="episode_unverified", task_id="other", mutated=1,
                verified="not_verified", eligibility="failure_trajectory")
            store.save_episode(failed)
            result = SFTDatasetBuilder(store).build()
            self.assertEqual(len(result.samples), 1)
            self.assertGreater(result.report["exclusion_reasons"]["episode_not_positive"], 0)

    def test_tool_recovery_atomic_structure_and_failed_history(self):
        failed_call = call("bad", "bash", {"command": "false"}, "failed")
        recovery_context = ({"role": "user", "content": (
            {"type": "text", "text": "fix it"},
        )}, {"role": "assistant", "content": (
            {"type": "tool_use", "id": "bad", "name": "bash",
             "arguments": {"command": "false"}},
        )}, {"role": "user", "content": (
            {"type": "tool_result", "tool_call_id": "bad",
             "content": "ERROR", "is_error": True},
        )})
        with tempfile.TemporaryDirectory() as directory:
            value = episode([
                turn(0, calls=(failed_call,), labels=("failed_tool_action",)),
                turn(1, calls=(call("read", "read_file", {"path": "a"}),
                                    call("edit", "edit_file", {"path": "a"})),
                     labels=("successful_tool_action", "recovery_after_failure",
                             "grounded_mutation"),
                     mutation_after=1,
                     messages=recovery_context, recovers=("turn_000000",)),
                turn(2, output="done", labels=("final_success",)),
            ])
            result = SFTDatasetBuilder(self.store_episode(Path(directory) / "s", value)).build()
            recoveries = [item for item in result.samples
                          if item.sample_type == SFTSampleType.RECOVERY_ACTION.value]
            self.assertEqual(len(recoveries), 1)
            sample = recoveries[0]
            self.assertEqual(len(sample.target["tool_calls"]), 2)
            self.assertIn("bad", json.dumps(sample.messages[:-1]))
            self.assertNotIn("turn_000000", {item.source_turn_id for item in result.samples})
            self.assertTrue(sample.tools)

    def test_bad_repeated_stalled_stale_auxiliary_and_external_excluded(self):
        cases = [
            (("repeated_action", "successful_tool_action"), (call("a", "read_file", {"path": "a"}),), ("generated",), "main", True),
            (("stalled",), (), ("generated",), "main", True),
            (("verification_action", "stale_verification"), (call("b", "bash", {"command": "test"}),), ("generated",), "main", True),
            (("successful_tool_action",), (call("c", "read_file", {"path": "a"}),), ("generated", "external_web"), "main", True),
            (("final_success",), (), ("generated",), "explorer", False),
            (("final_success",), (), ("generated",), "reviewer", False),
        ]
        turns = [turn(i, output="answer" if not calls else "", calls=calls,
                      labels=labels, provenance=provenance, role=role, primary=primary)
                 for i, (labels, calls, provenance, role, primary) in enumerate(cases)]
        with tempfile.TemporaryDirectory() as directory:
            result = SFTDatasetBuilder(self.store_episode(Path(directory) / "s",
                                                           episode(turns))).build()
            self.assertEqual(result.samples, [])
            reasons = result.report["exclusion_reasons"]
            for reason in ("repeated_action", "stalled", "stale_verification",
                           "external_web_dependency", "auxiliary_turn"):
                self.assertIn(reason, reasons)

    def test_useful_read_mutation_and_current_verification_types(self):
        turns = [
            turn(0, calls=(call("r", "read_file", {"path": "a"}),),
                 labels=("successful_tool_action",)),
            turn(1, calls=(call("m", "edit_file", {"path": "a"}),),
                 labels=("successful_tool_action", "grounded_mutation"), mutation_after=1),
            turn(2, calls=(call("v", "bash", {"command": "test"}),),
                 labels=("successful_tool_action", "verification_action"),
                 mutation_before=1, mutation_after=1, verification_generation=1),
            turn(3, output="done", labels=("final_success",), mutation_after=1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = SFTDatasetBuilder(self.store_episode(Path(directory) / "s", episode(
                turns, mutated=1, verified="verified", checkpoint="finalized"))).build()
            self.assertEqual({item.sample_type for item in result.samples}, {
                item.value for item in SFTSampleType
            } - {SFTSampleType.RECOVERY_ACTION.value})

    def test_redaction_overlong_memory_review_and_external_override(self):
        secret = "sk-abcdefghijklmnop"
        with tempfile.TemporaryDirectory() as directory:
            value = episode([turn(0, output=f"answer {secret}", labels=("final_success",))])
            store = self.store_episode(Path(directory) / "s", value)
            result = SFTDatasetBuilder(store).build()
            self.assertEqual(result.samples, [])
            self.assertEqual(result.all_samples[0].review_status, ReviewStatus.NEEDS_REVIEW.value)
            self.assertNotIn(secret, json.dumps(result.all_samples[0].to_dict()))

            long_value = episode([turn(0, output="x" * 500, labels=("final_success",))],
                                 episode_id="episode_long", task_id="long")
            store.save_episode(long_value)
            cfg = SFTExportConfiguration(max_approximate_tokens=10)
            built = SFTDatasetBuilder(store, configuration=cfg).build()
            self.assertTrue(any(item.length_status == LengthStatus.OVERLONG.value
                                for item in built.all_samples))

    def test_dedup_split_profiles_filtering_and_materialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self.store_episode(root / "s", episode([
                turn(0, output="same", labels=("final_success",)),
            ], project="keep"))
            store.save_episode(episode([
                turn(0, output="same", labels=("final_success",)),
            ], episode_id="episode_replay", task_id="replay", project="keep"))
            cfg = SFTExportConfiguration(project_filters=("keep",))
            builder = SFTDatasetBuilder(store, configuration=cfg)
            first = builder.build()
            second = builder.build()
            self.assertEqual([item.to_dict() for item in first.samples],
                             [item.to_dict() for item in second.samples])
            self.assertEqual(len(first.samples), 1)
            self.assertEqual(first.report["exclusion_reasons"]["duplicate"], 1)
            self.assertEqual(first.samples[0].split_group_id,
                             second.samples[0].split_group_id)
            materialized, output = builder.materialize(root / "out")
            self.assertTrue((output / "dataset-manifest.json").is_file())
            self.assertTrue((output / "quality-report.json").is_file())
            self.assertTrue((output / "quality-report.md").is_file())
            manifest = json.loads((output / "dataset-manifest.json").read_text())
            for name, checksum in manifest["output_checksums"].items():
                import hashlib
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(),
                                 checksum)
            _, same_output = builder.materialize(root / "out")
            self.assertEqual(output, same_output)

            dry = root / "dry"
            _, no_output = builder.materialize(dry, dry_run=True)
            self.assertIsNone(no_output)
            self.assertFalse(dry.exists())

    def test_final_and_agent_profiles_are_views(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.store_episode(Path(directory) / "s", episode([
                turn(0, calls=(call("r", "read_file", {"path": "a"}),),
                     labels=("successful_tool_action",)),
                turn(1, output="done", labels=("final_success",)),
            ]))
            final = SFTDatasetBuilder(store, configuration=SFTExportConfiguration(
                profile=ExportProfile.FINAL_RESPONSE_ONLY.value)).build()
            agent = SFTDatasetBuilder(store, configuration=SFTExportConfiguration(
                profile=ExportProfile.AGENT_TOOL_USE.value)).build()
            self.assertEqual([item.sample_type for item in final.samples], ["final_response"])
            self.assertEqual([item.sample_type for item in agent.samples], ["tool_action"])

    def test_corrupt_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.store_episode(Path(directory) / "s", episode([
                turn(0, output="done", labels=("final_success",)),
            ]))
            (store.episodes / "episode_test.sha256").write_text("0" * 64 + "\n")
            with self.assertRaises(SFTDatasetError):
                SFTDatasetBuilder(store).build()

    def test_split_configuration_validation(self):
        with self.assertRaises(ValueError):
            SplitConfiguration(90, 9, 9)


if __name__ == "__main__":
    unittest.main()
