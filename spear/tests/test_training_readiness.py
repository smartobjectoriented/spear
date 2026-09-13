import tempfile
import unittest
from pathlib import Path

from tests.test_preference_dataset import bad_turn, good_turn
from tests.test_sft_dataset import episode, turn
from training_governance import DataOrigin, TrainingDataGovernancePolicy
from training_readiness import (
    ReadinessLevel, TrainingReadinessEvaluator, TrainingReadinessIndex,
    TrainingReadinessPolicy, TrainingStrategyRecommendation,
)
from training_store import TrainingStore
from training_splits import split_group_for_episode


def with_origin(value, origin):
    raw = value.to_dict()
    raw["provenance"]["data_origin"] = origin
    return type(value).from_dict(raw)


class TrainingReadinessTests(unittest.TestCase):
    def store(self, root, *values):
        store = TrainingStore(root)
        for value in values:
            store.save_episode(value)
        return store

    def test_empty_store_collects_and_recommends_more(self):
        with tempfile.TemporaryDirectory() as directory:
            report = TrainingReadinessEvaluator(Path(directory) / "store").evaluate()
            self.assertEqual(report.sft.state, "COLLECTING")
            self.assertEqual(report.sft.level, ReadinessLevel.NOT_READY.value)
            self.assertEqual(report.recommendation,
                             TrainingStrategyRecommendation.COLLECT_MORE_DATA.value)
            self.assertIn("no_real_auto_approved_samples", report.sft.reasons)

    def test_small_sft_is_smoke_not_meaningful_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.store(Path(directory) / "s", episode([
                turn(0, output="done", labels=("final_success",)),
            ]))
            report = TrainingReadinessEvaluator(store).evaluate()
            self.assertEqual(report.sft.level, ReadinessLevel.READY_FOR_SMOKE.value)
            self.assertEqual(report.recommendation, "SFT_ONLY")
            self.assertIn("no_validation_split", report.sft.warnings)
            self.assertIn("insufficient_task_diversity", report.sft.warnings)

    def test_configured_meaningful_sft_and_recommendations(self):
        policy = TrainingReadinessPolicy(
            sft_experiment_samples=1, sft_experiment_split_groups=1,
            sft_experiment_projects=1, minimum_validation_samples=0,
            minimum_test_samples=0, unpaired_experiment_per_class=1,
            unpaired_experiment_groups=1, paired_experiment_pairs=1,
            paired_experiment_contexts=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            store = self.store(
                Path(directory) / "s",
                episode([bad_turn()], episode_id="bad", task_id="bad",
                        eligibility="failure_trajectory"),
                episode([good_turn(), turn(1, output="done", labels=("final_success",))],
                        episode_id="good", task_id="good"),
            )
            report = TrainingReadinessEvaluator(store, policy=policy).evaluate()
            self.assertEqual(report.sft.level, "READY_FOR_EXPERIMENT")
            self.assertEqual(report.unpaired_preference.level, "READY_FOR_EXPERIMENT")
            self.assertEqual(report.paired_preference.level, "READY_FOR_EXPERIMENT")
            self.assertEqual(report.recommendation, "SFT_THEN_PAIRED_PREFERENCE")

            no_pairs = self.store(Path(directory) / "single", episode([
                good_turn(), turn(1, output="done", labels=("final_success",)),
            ]))
            unpaired = TrainingReadinessEvaluator(no_pairs, policy=policy).evaluate()
            self.assertEqual(unpaired.recommendation, "SFT_ONLY")

    def test_governance_excludes_evaluation_synthetic_holdout_and_projects(self):
        with tempfile.TemporaryDirectory() as directory:
            normal = episode([turn(0, output="done", labels=("final_success",))],
                             episode_id="normal", project="allowed")
            synthetic = with_origin(episode([
                turn(0, output="fake", labels=("final_success",)),
            ], episode_id="synthetic"), DataOrigin.SYNTHETIC.value)
            benchmark = with_origin(episode([
                turn(0, output="bench", labels=("final_success",)),
            ], episode_id="benchmark"), DataOrigin.BENCHMARK.value)
            test = with_origin(episode([
                turn(0, output="test", labels=("final_success",)),
            ], episode_id="fixture"), DataOrigin.TEST_FIXTURE.value)
            store = self.store(Path(directory) / "s", normal, synthetic, benchmark, test)
            report = TrainingReadinessEvaluator(store).evaluate()
            self.assertEqual(report.distribution["real_source_episodes"], 1)
            self.assertEqual(report.governance["exclusions"]["synthetic_excluded"], 1)
            self.assertEqual(report.governance["exclusions"]["benchmark_excluded"], 1)
            self.assertEqual(report.governance["exclusions"]["test_fixture_excluded"], 1)

            holdout = TrainingReadinessEvaluator(
                store, governance=TrainingDataGovernancePolicy(
                    permanent_holdout_groups=(split_group_for_episode(normal),),
                )).evaluate()
            self.assertEqual(holdout.distribution["real_source_episodes"], 0)
            self.assertEqual(holdout.governance["exclusions"]["permanent_holdout"], 1)
            # Project deny takes effect independently of origin.
            denied = TrainingReadinessEvaluator(store, governance=
                TrainingDataGovernancePolicy(exclude_projects=("allowed",))).evaluate()
            self.assertEqual(denied.distribution["real_source_episodes"], 0)
            included = TrainingReadinessEvaluator(store, governance=
                TrainingDataGovernancePolicy(include_projects=("missing",))).evaluate()
            self.assertEqual(included.distribution["real_source_episodes"], 0)

    def test_readiness_index_is_checksummed_and_rebuildable_secondary_state(self):
        with tempfile.TemporaryDirectory() as directory:
            value = episode([turn(0, output="done", labels=("final_success",))])
            index = TrainingReadinessIndex(directory)
            index.update_episode(value)
            self.assertIn(value.episode_id, index.load()["episodes"])
            index.path.write_text("{}")
            with self.assertRaises(ValueError):
                index.load()

    def test_target_redaction_quarantine_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.store(Path(directory) / "s", episode([
                turn(0, output="secret sk-abcdefghijklmnop",
                     labels=("final_success",)),
            ]))
            report = TrainingReadinessEvaluator(store).evaluate()
            self.assertEqual(report.sft.state, "NEEDS_REVIEW")
            self.assertEqual(report.sft.level, "NOT_READY")
            self.assertIn("too_many_quarantined_samples", report.sft.reasons)
            self.assertEqual(report.recommendation, "NEEDS_MANUAL_DATA_REVIEW")


if __name__ == "__main__":
    unittest.main()
