"""The operator's corrected answer, and the data built from it.

A /bad says an answer was wrong; it teaches nothing about what was right,
and the store held no two answers to one context, so no preference pair
could ever be mined. /fix records what the answer should have been: a
target over the answer's own context, and a pair against the answer given.
"""

import tempfile
import unittest
from pathlib import Path

from tests.test_sft_dataset import call, episode, turn
from training.preference_dataset import PreferenceDatasetBuilder
from training.sft_dataset import SFTDatasetBuilder, SFTSelectionPolicy
from training.training_data import TrainingEpisode
from training.training_governance import TrainingDataGovernancePolicy as GovernancePolicy
from training.training_store import TrainingStore, TrainingStoreError

WRONG = "IRM is bit [4]."
RIGHT = "IRM is bit [40] of ICC_SGI1R_EL1 (IHI 0069 §12.2.22)."


def answered(episode_id, **options):
    return episode([turn(0, output="reading", calls=(call("c1", "read_file", {"path": "a"}),),
                         labels=("useful_read",)),
                    turn(1, output=WRONG, labels=("final_success",))],
                   episode_id=episode_id, task_id=episode_id, **options)


class Store(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = TrainingStore(Path(self.temp.name) / "training")
        self.store.save_episode(answered("episode_a"))

    def tearDown(self):
        self.temp.cleanup()

    def test_the_latest_correction_wins_and_an_empty_one_is_refused(self):
        self.store.record_correction("episode_a", "first")
        self.store.record_correction("episode_a", RIGHT)

        self.assertEqual(self.store.operator_corrections(), {"episode_a": RIGHT})

        for episode_id, text in (("episode_a", "  "), ("../x", RIGHT)):
            with self.assertRaises(TrainingStoreError):
                self.store.record_correction(episode_id, text)

    def test_verdicts_still_read_back(self):
        self.store.record_verdict("episode_a", "bad")

        self.assertEqual(self.store.operator_verdicts(), {"episode_a": "bad"})


class Datasets(Store):
    def setUp(self):
        super().setUp()
        self.store.record_verdict("episode_a", "bad")
        self.store.record_correction("episode_a", RIGHT)

    def test_the_correction_is_a_target_over_the_answers_context(self):
        samples = SFTDatasetBuilder(self.store).build().samples
        corrected = [item for item in samples if "operator_correction" in item.review_flags]

        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0].target["content"], RIGHT)
        self.assertNotIn(WRONG, str(corrected[0].messages))
        self.assertEqual(corrected[0].sample_type, "final_response")

        # /bad still keeps every turn of the episode out; only the correction
        # comes from it.

        self.assertEqual(len(samples), 1)

    def test_without_a_bad_verdict_the_corrected_answer_is_still_no_target(self):
        self.store.save_episode(answered("episode_c"))
        self.store.record_correction("episode_c", RIGHT + " Bit [40] only.")
        targets = [str(item.target.get("content")) for item in
                   SFTDatasetBuilder(self.store).build().samples
                   if item.episode_id == "episode_c"]

        self.assertIn(RIGHT + " Bit [40] only.", targets)
        self.assertNotIn(WRONG, targets)

    def test_the_correction_and_the_answer_are_an_exact_context_pair(self):
        result = PreferenceDatasetBuilder(self.store).build()
        pairs = [item for item in result.all_paired if item.review_status == "AUTO_APPROVED"]

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].chosen[0]["content"], RIGHT)
        self.assertEqual(pairs[0].rejected[0]["content"], WRONG)


class Origins(unittest.TestCase):
    def origin(self, binding):
        raw = answered("episode_b").to_dict()
        raw["provenance"]["standard_binding"] = binding

        return GovernancePolicy.origin_for(TrainingEpisode.from_dict(raw))

    def test_only_a_licensed_standard_overrides_the_episodes_origin(self):
        self.assertEqual(self.origin({"standard_id": "S", "data_origin": "PUBLIC"}),
                         "NORMAL_USAGE")
        self.assertEqual(self.origin({"standard_id": "S", "data_origin": "LICENSED_STANDARD"}),
                         "LICENSED_STANDARD")
        self.assertEqual(self.origin({"standard_id": "S"}), "LICENSED_STANDARD")


class StandardReads(unittest.TestCase):
    def test_reading_the_standard_is_progress(self):
        for name in ("standard.search", "standard.fetch", "standard.get_structure",
                     "standard.cite"):
            source = episode([turn(0, output="", calls=(call("c1", name, {}),))])
            kind, reasons = SFTSelectionPolicy().select(source, source.turns[0])

            self.assertIsNotNone(kind, (name, reasons))


if __name__ == "__main__":
    unittest.main()
