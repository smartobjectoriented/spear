import json
import tempfile
import unittest
from pathlib import Path

import training_promotion as promotion
from training_data import TrainingEligibility, TrainingRedactionPolicy

PRIVATE_KEY = ("-----BEGIN RSA PRIVATE KEY-----\n"
               "AAAAB3NzaC1yc2EAAAADAQABAAABgQ\n"
               "-----END RSA PRIVATE KEY-----")


def row(**overrides):
    sample = {
        "timestamp": "2026-09-09T10:00:00+00:00",
        "verdict": "pass",
        "source": "project_build",
        "project": "adhoc:vita492",
        "task_id": "task_1",
        "question": "make the acknowledgement follow the rule",
        "answer": "Changed command_wire.c.",
        "steps": [{"tool": "bash", "arguments": {"command": "cat a.c"},
                   "result": "int main(void)", "action_id": "a1",
                   "status": "ok"}],
    }
    sample.update(overrides)

    return sample


class OnlyWhatSurvived(unittest.TestCase):
    """Recording is wide on purpose; promotion is not."""

    def test_a_verified_change_is_promoted(self):
        episode, decision = promotion.promote(row())

        self.assertTrue(decision.stored)
        self.assertTrue(decision.positive)
        self.assertEqual(episode.training_metadata["eligibility"],
                         TrainingEligibility.POSITIVE_CANDIDATE.value)

    def test_a_failed_turn_is_stored_but_never_imitated(self):
        episode, decision = promotion.promote(row(verdict="fail"))

        self.assertTrue(decision.stored)
        self.assertFalse(decision.positive)
        self.assertEqual(episode.training_metadata["eligibility"],
                         TrainingEligibility.FAILURE_TRAJECTORY.value)

    def test_an_unjudged_turn_is_stored_but_never_imitated(self):
        """"Nothing judged this" is not "this was judged wrong"."""
        _, decision = promotion.promote(row(verdict="unrated"))

        self.assertTrue(decision.stored)
        self.assertFalse(decision.positive)
        self.assertIn("nothing judged this turn", decision.describe())

    def test_a_pass_with_no_tool_call_was_verified_vacuously(self):
        """The build gate only runs on a turn that changed something."""
        episode, decision = promotion.promote(row(steps=[]))

        self.assertTrue(decision.stored)
        self.assertFalse(decision.positive)
        self.assertEqual(episode.training_metadata["eligibility"],
                         TrainingEligibility.INCOMPLETE.value)


class Secrets(unittest.TestCase):
    def test_a_credential_is_scrubbed_before_the_store_sees_it(self):
        leaky = row(steps=[{"tool": "bash", "arguments": {"command": "env"},
                            "result": "API_KEY=sk-abcdefghijklmnop1234",
                            "action_id": "a1", "status": "ok"}])
        episode, decision = promotion.promote(leaky)
        carried = episode.turns[0].tool_calls[0].model_content

        self.assertNotIn("sk-abcdefghijklmnop1234", carried)
        self.assertNotIn("sk-abcdefghijklmnop1234", json.dumps(episode.to_dict()))
        self.assertTrue(decision.positive, "a scrubbed turn is still usable")
        self.assertIn("redacted", decision.describe())

    def test_a_private_key_refuses_the_row_outright(self):
        """Scrubbing the text does not undo the key having been printed."""
        episode, decision = promotion.promote(
            row(steps=[{"tool": "bash", "arguments": {"command": "cat id_rsa"},
                        "result": PRIVATE_KEY, "action_id": "a1",
                        "status": "ok"}]))

        self.assertIsNone(episode)
        self.assertFalse(decision.stored)
        self.assertIn("private key", decision.describe())

    def test_an_environment_secret_is_taken_as_a_literal_to_scrub(self):
        policy = TrainingRedactionPolicy(("hunter2-correct-horse",))
        leaky = row(steps=[{"tool": "bash", "arguments": {"command": "echo $P"},
                            "result": "hunter2-correct-horse",
                            "action_id": "a1", "status": "ok"}])
        episode, _ = promotion.promote(leaky, policy=policy)

        self.assertNotIn("hunter2-correct-horse", json.dumps(episode.to_dict()))

    def test_the_decision_travels_with_the_episode(self):
        """A reviewer should see that a sample was cleaned, not trust it was."""
        leaky = row(steps=[{"tool": "bash", "arguments": {"command": "env"},
                            "result": "API_KEY=sk-abcdefghijklmnop1234",
                            "action_id": "a1", "status": "ok"}])
        episode, _ = promotion.promote(leaky)
        recorded = episode.training_metadata["promotion"]

        self.assertTrue(recorded["positive"])
        self.assertIn("api_key", recorded["redactions"]["categories"])


class ThroughTheCommand(unittest.TestCase):
    def test_ingest_reports_stored_imitable_and_refused_separately(self):
        import finetune_commands
        from training_controller import TrainingController

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "trajectories.jsonl"
            source.write_text("\n".join(json.dumps(item) for item in (
                row(),
                row(verdict="fail", answer="broke it"),
                row(answer="leaked",
                    steps=[{"tool": "bash", "arguments": {"command": "cat k"},
                            "result": PRIVATE_KEY, "action_id": "a1",
                            "status": "ok"}]),
            )) + "\n", encoding="utf-8")
            controller = TrainingController(str(root / "store"), root / "audit")
            message = finetune_commands.handle_finetune_command(
                f"/finetune ingest {source}", controller)
            stored = list((root / "store" / "episodes").glob("*.json"))

        self.assertIn("2 episode(s) stored", message)
        self.assertIn("1 to imitate", message)
        self.assertIn("1 refused", message)
        self.assertEqual(len(stored), 2)


class WhereTheRecorderWrites(unittest.TestCase):
    """A default that points at a stale file ingests it and says nothing."""

    def test_the_default_follows_the_recorder_not_a_guess(self):
        import os
        from unittest.mock import patch
        import finetune_commands

        with patch.dict(os.environ, {"SPEAR_STATE_DIR": "/state"}, clear=True):
            self.assertEqual(finetune_commands.default_trajectory_file(),
                             "/state/trajectories.jsonl")

        with patch.dict(os.environ, {"SPEAR_TRAJECTORY_FILE": "/named.jsonl"},
                        clear=True):
            self.assertEqual(finetune_commands.default_trajectory_file(),
                             "/named.jsonl")

    def test_with_nothing_set_the_command_asks_rather_than_assumes(self):
        import os
        from unittest.mock import patch
        import finetune_commands
        from training_controller import TrainingController

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = TrainingController(str(root / "store"), root / "audit")

            with patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(finetune_commands.default_trajectory_file())

                # Raised rather than returned: that is how this module reports
                # a malformed command, and rag_chat prints it.
                with self.assertRaises(finetune_commands.FinetuneCommandError) as raised:
                    finetune_commands.handle_finetune_command(
                        "/finetune ingest", controller)

        self.assertIn("usage: /finetune ingest <path>", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
