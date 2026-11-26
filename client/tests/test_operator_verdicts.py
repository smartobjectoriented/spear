"""The operator's /good and /bad reach the training data.

They used to be archived beside the conversation and nowhere else, so an
answer called bad kept the automatic final_success label of its episode
and stayed a positive example.
"""

import tempfile
import unittest
from pathlib import Path

from tests.test_sft_dataset import episode, turn
from training.preference_dataset import OutcomeLabel, OutcomeLabelPolicy
from training.sft_dataset import SFTDatasetBuilder
from training.training_store import TrainingStore, TrainingStoreError


def answered(episode_id):
    return episode([turn(0, output="the answer", labels=("final_success",))],
                   episode_id=episode_id, task_id=episode_id)


class Journal(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = TrainingStore(Path(self.temp.name) / "training")

    def tearDown(self):
        self.temp.cleanup()

    def test_the_latest_verdict_wins_and_bad_names_are_refused(self):
        self.store.save_episode(answered("episode_a"))
        self.store.record_verdict("episode_a", "good")
        self.store.record_verdict("episode_a", "bad")

        self.assertEqual(self.store.operator_verdicts(), {"episode_a": "bad"})

        for episode_id, verdict in (("episode_a", "meh"), ("../x", "bad")):
            with self.assertRaises(TrainingStoreError):
                self.store.record_verdict(episode_id, verdict)

    def test_the_latest_episode_is_found(self):
        self.assertIsNone(self.store.latest_episode_id())
        self.store.save_episode(answered("episode_a"))
        self.store.save_episode(answered("episode_b"))

        self.assertEqual(self.store.latest_episode_id(), "episode_b")
        self.assertIsNone(self.store.latest_episode_id("another-session"))

    def test_a_bad_answer_is_never_an_sft_target(self):
        self.store.save_episode(answered("episode_a"))
        self.assertEqual(len(SFTDatasetBuilder(self.store).build().samples), 1)

        self.store.record_verdict("episode_a", "bad")
        result = SFTDatasetBuilder(self.store).build()

        self.assertEqual(list(result.samples), [])
        self.assertGreater(result.report["exclusion_reasons"]["operator_rejected"], 0)


class Preferences(unittest.TestCase):
    def test_the_verdict_outranks_the_automatic_label(self):
        source = answered("episode_a")
        last = source.turns[-1]

        self.assertEqual(OutcomeLabelPolicy().label(source, last)[0], OutcomeLabel.DESIRABLE)
        self.assertEqual(OutcomeLabelPolicy({"episode_a": "bad"}).label(source, last)[:2],
                         (OutcomeLabel.UNDESIRABLE, ("operator_rejected",)))
        self.assertEqual(OutcomeLabelPolicy({"episode_a": "good"}).label(source, last)[:2],
                         (OutcomeLabel.DESIRABLE, ("operator_approved",)))


if __name__ == "__main__":
    unittest.main()
