import json
import tempfile
import unittest
from pathlib import Path

import sft_dataset
import trajectory_episodes as convert
from sft_dataset import SFTDatasetBuilder
from training_data import TrainingEligibility
from training_store import TrainingStore


def row(**overrides):
    sample = {
        "verdict": "pass",
        "source": "project_build",
        "project": "adhoc:vita492",
        "task_id": "task_1",
        "bench": "",
        "question": "can you validate and make changes in the code accordingly",
        "steps": [
            {"tool": "bash", "arguments": {"command": "grep -n x a.c"},
             "result": "12: x", "action_id": "a1", "status": "ok"},
            {"tool": "edit_file",
             "arguments": {"path": "a.c", "old_text": "x", "new_text": "y"},
             "result": "OK", "action_id": "a2", "status": "ok"},
        ],
        "answer": "Changed a.c line 12.",
    }
    sample.update(overrides)

    return sample


class Shape(unittest.TestCase):
    def test_one_turn_per_emission_and_only_the_answer_is_a_candidate(self):
        episode = convert.episode_from_row(row(), system="SYSTEM")

        self.assertEqual([turn.purpose for turn in episode.turns],
                         ["tool_action", "tool_action", "final_response"])
        self.assertEqual([turn.primary_candidate for turn in episode.turns],
                         [False, False, True])

    def test_a_turn_sees_only_what_came_before_it(self):
        """A sample built from later history would train on hindsight."""
        episode = convert.episode_from_row(row(), system="SYSTEM")

        self.assertEqual(len(episode.turns[0].messages), 1)
        self.assertEqual(len(episode.turns[1].messages), 3)
        self.assertEqual(len(episode.turns[-1].messages), 5)

    def test_the_identifier_is_stable_across_conversions(self):
        """Converting the same file twice must not duplicate its episodes."""

        self.assertEqual(convert.episode_id_for(row()),
                         convert.episode_id_for(row()))
        self.assertNotEqual(convert.episode_id_for(row()),
                            convert.episode_id_for(row(answer="different")))

    def test_a_row_with_no_tool_call_still_converts(self):
        episode = convert.episode_from_row(row(steps=[]), system="SYSTEM")

        self.assertEqual(len(episode.turns), 1)
        self.assertTrue(episode.turns[0].primary_candidate)
        self.assertFalse(episode.turns[0].tools_enabled)


class Verdicts(unittest.TestCase):
    """Not judged is not judged wrong."""

    def test_pass_is_a_positive_candidate(self):
        episode = convert.episode_from_row(row(verdict="pass"))

        self.assertEqual(episode.training_metadata["eligibility"],
                         TrainingEligibility.POSITIVE_CANDIDATE.value)

    def test_fail_is_kept_as_a_failure_trajectory(self):
        """A corpus of successes cannot teach what to stop doing."""
        episode = convert.episode_from_row(row(verdict="fail"))

        self.assertEqual(episode.training_metadata["eligibility"],
                         TrainingEligibility.FAILURE_TRAJECTORY.value)

    def test_unrated_is_incomplete_rather_than_failed(self):
        episode = convert.episode_from_row(row(verdict="unrated"))

        self.assertEqual(episode.training_metadata["eligibility"],
                         TrainingEligibility.INCOMPLETE.value)

    def test_an_unknown_verdict_is_refused(self):
        with self.assertRaises(convert.TrajectoryConversionError):
            convert.episode_from_row(row(verdict="probably fine"))


class Refusals(unittest.TestCase):
    def test_a_row_without_a_question_is_refused(self):
        with self.assertRaises(convert.TrajectoryConversionError):
            convert.episode_from_row(row(question=""))

    def test_a_step_that_names_no_tool_is_refused(self):
        with self.assertRaises(convert.TrajectoryConversionError):
            convert.episode_from_row(row(steps=[{"arguments": {}}]))

    def test_one_bad_line_does_not_cost_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectories.jsonl"
            path.write_text(
                json.dumps(row()) + "\n"
                + "{not json\n"
                + json.dumps(row(answer="second")) + "\n", encoding="utf-8")
            seen = list(convert.episodes_from_file(path))

        self.assertEqual([episode is not None for episode, _ in seen],
                         [True, False, True])
        self.assertIn("line 2", seen[1][1])


class ItReachesTheTrainer(unittest.TestCase):
    """The point of the exercise: the exporters accept what usage records."""

    def build(self, sample):
        with tempfile.TemporaryDirectory() as directory:
            store = TrainingStore(directory)
            store.save_episode(convert.episode_from_row(sample, system="SYSTEM"))

            return SFTDatasetBuilder(store).build()

    def test_a_recorded_turn_becomes_an_sft_sample(self):
        result = self.build(row())

        self.assertEqual(len(result.samples), 1)
        self.assertEqual([message["role"] for message in result.samples[0].messages],
                         ["system", "user", "assistant", "tool",
                          "assistant", "tool", "assistant"])

    def test_a_failed_turn_yields_no_sample_to_imitate(self):
        self.assertEqual(len(self.build(row(verdict="fail")).samples), 0)

    def test_an_unrated_turn_yields_no_sample_to_imitate(self):
        self.assertEqual(len(self.build(row(verdict="unrated")).samples), 0)

    def test_the_tool_view_digest_matches_the_exporters_own(self):
        """The builder re-hashes every snapshot and refuses a mismatch.

        It caught this converter's first attempt. The two functions live in
        different modules on purpose -- the recorder must not depend on the
        exporter -- so this is what stops them drifting apart.
        """
        episode = convert.episode_from_row(row())

        for view_hash, snapshot in episode.tool_views.items():
            self.assertEqual(sft_dataset._hash(snapshot), view_hash)

        for turn in episode.turns:
            self.assertIn(turn.tool_view_hash, episode.tool_views)

    def test_supplied_schemas_reach_the_exported_tool_definitions(self):
        schemas = {"bash": {"description": "Run a command",
                            "input_schema": {"type": "object",
                                             "properties": {"command": {"type": "string"}}}}}
        episode = convert.episode_from_row(row(), tool_schemas=schemas)
        snapshot = next(iter(episode.tool_views.values()))
        described = {item["name"]: item for item in snapshot["tools"]}

        self.assertEqual(described["bash"]["description"], "Run a command")
        self.assertEqual(described["edit_file"]["input_schema"], {})


class TheOperatorCommand(unittest.TestCase):
    """/finetune ingest, end to end from a recorded file to stored episodes."""

    def test_ingesting_a_file_stores_its_episodes(self):
        import finetune_commands
        from training_controller import TrainingController

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "trajectories.jsonl"
            source.write_text(
                json.dumps(row()) + "\n"
                + json.dumps(row(verdict="fail", answer="broke it")) + "\n"
                + "{not json\n", encoding="utf-8")
            controller = TrainingController(str(root / "store"), root / "audit")
            message = finetune_commands.handle_finetune_command(
                f"/finetune ingest {source}", controller)
            stored = list((root / "store" / "episodes").glob("*.json"))

        self.assertIn("2 episode(s) stored", message)
        self.assertIn("1 to imitate", message)
        self.assertIn("1 row(s) skipped", message)
        self.assertEqual(len(stored), 2)

    def test_ingesting_twice_does_not_duplicate(self):
        import finetune_commands
        from training_controller import TrainingController

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "trajectories.jsonl"
            source.write_text(json.dumps(row()) + "\n", encoding="utf-8")
            controller = TrainingController(str(root / "store"), root / "audit")

            for _ in range(2):
                finetune_commands.handle_finetune_command(
                    f"/finetune ingest {source}", controller)

            stored = list((root / "store" / "episodes").glob("*.json"))

        self.assertEqual(len(stored), 1)

    def test_a_missing_file_is_refused_rather_than_reported_empty(self):
        import finetune_commands
        from training_controller import TrainingController

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = TrainingController(str(root / "store"), root / "audit")
            message = finetune_commands.handle_finetune_command(
                f"/finetune ingest {root / 'absent.jsonl'}", controller)

        self.assertIn("no trajectory file", message)


if __name__ == "__main__":
    unittest.main()
