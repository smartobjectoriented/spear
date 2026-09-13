import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from preference_dataset import (
    DecisionContextFingerprint, OutcomeLabel, PreferenceConfiguration,
    PreferenceDatasetBuilder, PreferenceDatasetError, PreferenceProfile,
)
from sft_dataset import ReviewStatus, SFTDatasetBuilder
from tests.test_sft_dataset import call, episode, turn
from training_store import TrainingStore


def context_with_failure():
    return (
        {"role": "user", "content": ({"type": "text", "text": "do task"},)},
        {"role": "assistant", "content": ({
            "type": "tool_use", "id": "bad", "name": "read_file",
            "arguments": {"path": "missing"},
        },)},
        {"role": "user", "content": ({
            "type": "tool_result", "tool_call_id": "bad",
            "content": "ERROR: missing", "is_error": True,
        },)},
    )


def bad_turn(index=0, *, call_id="bad", path="missing", extra_labels=()):
    return turn(index, calls=(call(call_id, "read_file", {"path": path}, "failed"),),
                labels=("failed_tool_action", *extra_labels))


def good_turn(index=0, *, call_id="good", path="present", messages=None,
              extra_labels=(), recovers=()):
    return turn(index, calls=(call(call_id, "read_file", {"path": path}),),
                labels=("successful_tool_action", "useful_read", *extra_labels),
                messages=messages, recovers=recovers)


class PreferenceDatasetTests(unittest.TestCase):
    def store(self, root, *episodes):
        store = TrainingStore(root)
        for value in episodes:
            store.save_episode(value)
        return store

    def valid_pair_store(self, root):
        return self.store(
            root,
            episode([bad_turn()], episode_id="episode_bad", task_id="bad",
                    eligibility="failure_trajectory"),
            episode([good_turn()], episode_id="episode_good", task_id="good"),
        )

    def test_decision_context_fingerprint_is_strong_and_deterministic(self):
        prompt = ({"role": "system", "content": "rules"},
                  {"role": "user", "content": "task"})
        first = DecisionContextFingerprint.compute(
            prompt=prompt, tools=(), role="main", purpose="primary_agent")
        second = DecisionContextFingerprint.compute(
            prompt=prompt, tools=(), role="main", purpose="primary_agent")
        self.assertEqual(first, second)
        self.assertNotEqual(first, DecisionContextFingerprint.compute(
            prompt=(*prompt, {"role": "tool", "tool_call_id": "x",
                              "content": "compiler failed"}),
            tools=(), role="main", purpose="primary_agent"))
        # Episode/session IDs are not fingerprint inputs because they were not
        # rendered into this model-visible prompt.
        self.assertEqual(first, second)

    def test_exact_context_success_failure_pair_and_unpaired_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            result = PreferenceDatasetBuilder(
                self.valid_pair_store(Path(directory) / "source")
            ).build()
            self.assertEqual(len(result.paired), 1)
            pair = result.paired[0]
            self.assertEqual(pair.review_status, ReviewStatus.AUTO_APPROVED.value)
            self.assertEqual(pair.chosen[0]["role"], "assistant")
            self.assertEqual(pair.rejected[0]["role"], "assistant")
            self.assertTrue(pair.tools)
            self.assertEqual(len(result.unpaired), 2)
            self.assertEqual({item.desirable for item in result.unpaired}, {True, False})
            self.assertNotIn("reasoning", json.dumps(pair.trainer_record()).lower())
            self.assertEqual(result.report["paired_candidates"]["AUTO_APPROVED"], 1)

    def test_changed_context_recovery_is_unpaired_not_dpo(self):
        with tempfile.TemporaryDirectory() as directory:
            value = episode([
                bad_turn(0),
                good_turn(1, messages=context_with_failure(),
                          extra_labels=("recovery_after_failure",),
                          recovers=("turn_000000",)),
            ], episode_id="episode_recovery")
            result = PreferenceDatasetBuilder(
                self.store(Path(directory) / "source", value)
            ).build()
            self.assertEqual(result.paired, [])
            self.assertEqual(len(result.unpaired), 2)
            self.assertEqual(result.report["recovery"]["relationships"], 1)
            self.assertEqual(result.report["recovery"]["not_pairable_context_changed"], 1)
            self.assertEqual(result.report["exclusion_reasons"]["context_mismatch"], 1)
            recovery = next(item for item in result.unpaired if item.desirable)
            self.assertIn("NOT_PAIRABLE_CONTEXT_CHANGED", recovery.review_flags)

    def test_three_episode_end_to_end_pairability(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.store(
                Path(directory) / "source",
                episode([bad_turn()], episode_id="episode_a", task_id="a",
                        eligibility="failure_trajectory"),
                episode([good_turn()], episode_id="episode_b", task_id="b"),
                episode([
                    bad_turn(0),
                    good_turn(1, messages=context_with_failure(),
                              extra_labels=("recovery_after_failure",),
                              recovers=("turn_000000",)),
                ], episode_id="episode_c", task_id="c"),
            )
            result = PreferenceDatasetBuilder(store).build()
            self.assertEqual(len(result.paired), 1)
            self.assertEqual(len(result.unpaired), 4)
            self.assertEqual(sum(item.desirable for item in result.unpaired), 2)
            self.assertEqual(result.report["recovery"]["not_pairable_context_changed"], 1)
            self.assertEqual(result.report["pairability_rate"], 0.5)

    def test_failed_turn_stays_negative_and_useful_failed_episode_not_flipped(self):
        with tempfile.TemporaryDirectory() as directory:
            value = episode([
                bad_turn(0), good_turn(1),
            ], episode_id="episode_eventual", eligibility="positive_candidate")
            failed_episode_useful = episode([
                good_turn(0),
            ], episode_id="episode_failed_useful", task_id="useful",
                eligibility="failure_trajectory")
            result = PreferenceDatasetBuilder(self.store(
                Path(directory) / "source", value, failed_episode_useful,
            )).build()
            labels = {(item.episode_id, item.label) for item in result.observations}
            self.assertIn(("episode_eventual", OutcomeLabel.UNDESIRABLE.value), labels)
            self.assertIn(("episode_failed_useful", OutcomeLabel.DESIRABLE.value), labels)

    def test_unknown_and_success_success_have_no_automatic_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            neutral = turn(0, calls=(call("neutral", "bash", {"command": "pwd"}),),
                           labels=("successful_tool_action",))
            result = PreferenceDatasetBuilder(self.store(
                Path(directory) / "source",
                episode([neutral], episode_id="neutral"),
                episode([good_turn(path="a")], episode_id="good_a", task_id="a"),
                episode([good_turn(path="b")], episode_id="good_b", task_id="b"),
            )).build()
            self.assertEqual(result.paired, [])
            self.assertEqual(result.report["source_turns"]["unknown"], 1)

    def test_mixed_multiple_tools_atomic_and_unknown(self):
        mixed = turn(0, calls=(
            call("one", "read_file", {"path": "a"}),
            call("two", "bash", {"command": "false"}, "failed"),
        ), labels=("successful_tool_action", "failed_tool_action"))
        with tempfile.TemporaryDirectory() as directory:
            result = PreferenceDatasetBuilder(self.store(
                Path(directory) / "source", episode([mixed]),
            )).build()
            self.assertEqual(result.observations[0].label, OutcomeLabel.UNKNOWN.value)
            self.assertIn("mixed_multi_tool_outcome", result.observations[0].label_basis)
            self.assertEqual(result.unpaired, [])

    def test_invalid_security_malformed_and_false_completion_labels(self):
        denied = turn(0, calls=(call("deny", "bash", {"command": "rm x"}, "denied"),),
                      labels=("security_failure", "failed_tool_action"))
        false_final = turn(0, output="done", labels=("ungrounded_completion_claim",))
        malformed = turn(0, output="broken", labels=("invalid_model_turn",))
        with tempfile.TemporaryDirectory() as directory:
            result = PreferenceDatasetBuilder(self.store(
                Path(directory) / "source",
                episode([denied], episode_id="denied"),
                episode([false_final], episode_id="false", task_id="false"),
                episode([malformed], episode_id="malformed", task_id="malformed"),
            )).build()
            self.assertEqual({item.label for item in result.observations},
                             {OutcomeLabel.UNDESIRABLE.value})

    def test_invalid_arguments_are_preserved_as_structured_negative(self):
        invalid = turn(0, calls=(
            call("invalid", "read_file", {}, "invalid_arguments"),
        ), labels=("failed_tool_action",))
        with tempfile.TemporaryDirectory() as directory:
            result = PreferenceDatasetBuilder(self.store(
                Path(directory) / "source", episode([invalid]),
            )).build()
            self.assertEqual(len(result.unpaired), 1)
            self.assertFalse(result.unpaired[0].desirable)
            arguments = result.unpaired[0].completion[0]["tool_calls"][0]["function"]["arguments"]
            self.assertEqual(json.loads(arguments), {})

    def test_pair_source_split_groups_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.store(
                Path(directory) / "source",
                episode([bad_turn()], episode_id="bad_project", task_id="bad",
                        project="project-a", eligibility="failure_trajectory"),
                episode([good_turn()], episode_id="good_project", task_id="good",
                        project="project-b"),
            )
            result = PreferenceDatasetBuilder(store).build()
            self.assertEqual(result.paired, [])
            self.assertEqual(result.all_paired[0].review_status,
                             ReviewStatus.REJECTED.value)
            self.assertIn("source_split_group_mismatch",
                          result.all_paired[0].review_flags)

    def test_external_redaction_superseded_and_overlong_are_not_approved(self):
        external = replace(good_turn(), provenance=("generated", "external_web"))
        secret = good_turn()
        secret.tool_calls = (call("secret", "read_file", {"path": "sk-abcdefghijklmnop"}),)
        memory = replace(good_turn(), memory_provenance_ids=("mem_old",))
        long = turn(0, output="x" * 1000, labels=("final_success",))
        with tempfile.TemporaryDirectory() as directory:
            store = self.store(
                Path(directory) / "source",
                episode([external], episode_id="external"),
                episode([secret], episode_id="secret", task_id="secret"),
                episode([memory], episode_id="memory", task_id="memory"),
                episode([long], episode_id="long", task_id="long"),
            )
            config = PreferenceConfiguration(
                max_approximate_tokens=50, superseded_memory_ids=("mem_old",),
            )
            result = PreferenceDatasetBuilder(store, configuration=config).build()
            statuses = {(item.episode_id, item.review_status) for item in result.observations}
            self.assertIn(("external", ReviewStatus.REJECTED.value), statuses)
            self.assertIn(("memory", ReviewStatus.NEEDS_REVIEW.value), statuses)
            self.assertTrue(any("completion_redacted" in item.review_flags
                                for item in result.observations))
            self.assertTrue(any("overlong" in item.review_flags
                                for item in result.observations))

    def test_deterministic_ids_dedup_and_shared_sft_split(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.valid_pair_store(Path(directory) / "source")
            first = PreferenceDatasetBuilder(store).build()
            second = PreferenceDatasetBuilder(store).build()
            self.assertEqual(first.paired[0].preference_id, second.paired[0].preference_id)
            self.assertEqual([item.sample_id for item in first.unpaired],
                             [item.sample_id for item in second.unpaired])
            sft = SFTDatasetBuilder(store).build()
            good_preference = next(item for item in first.unpaired if item.desirable)
            self.assertEqual(good_preference.split_group_id,
                             sft.samples[0].split_group_id)
            self.assertEqual(good_preference.split, sft.samples[0].split)

    def test_contradictory_preference_direction_is_conflict(self):
        def observed(ep_id, path, desired):
            item = (good_turn(path=path, call_id=path) if desired
                    else bad_turn(path=path, call_id=path))
            return episode([item], episode_id=ep_id, task_id=ep_id,
                           eligibility=("positive_candidate" if desired
                                        else "failure_trajectory"))
        with tempfile.TemporaryDirectory() as directory:
            store = self.store(
                Path(directory) / "source",
                observed("a_good", "a", True), observed("b_bad", "b", False),
                observed("a_bad", "a", False), observed("b_good", "b", True),
            )
            result = PreferenceDatasetBuilder(store).build()
            self.assertEqual(result.paired, [])
            self.assertGreater(result.report["paired_candidates"]["conflicts"], 0)
            self.assertTrue(all(item.review_status == ReviewStatus.NEEDS_REVIEW.value
                                for item in result.all_paired))

    def test_index_incremental_rebuild_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.valid_pair_store(Path(directory) / "source")
            builder = PreferenceDatasetBuilder(store)
            for metadata in store.iterate_metadata():
                builder.index_episode(store.load_episode(metadata["episode_id"]))
            value = builder.index.load()
            self.assertEqual(len(value["contexts"]), 1)
            self.assertEqual(len(next(iter(value["contexts"].values()))), 2)
            builder.rebuild_index()
            self.assertEqual(value, builder.index.load())
            builder.index.checksum_path.write_text("0" * 64 + "\n")
            with self.assertRaises(PreferenceDatasetError):
                builder.index.load()

    def test_source_checksum_mismatch_rejected_and_zero_pairs_normal(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.store(Path(directory) / "source", episode([
                turn(0, output="neutral"),
            ]))
            self.assertEqual(PreferenceDatasetBuilder(store).build().paired, [])
            (store.episodes / "episode_test.sha256").write_text("0" * 64 + "\n")
            with self.assertRaises(PreferenceDatasetError):
                PreferenceDatasetBuilder(store).build()


if __name__ == "__main__":
    unittest.main()
