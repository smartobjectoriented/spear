import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from preference_export import main
from tests.test_agent_runtime import ScriptedBackend, make_context, text_turn
from tests.test_preference_dataset import bad_turn, good_turn
from tests.test_sft_dataset import episode
from tests.test_training_data import make_controller
from task_controller import TaskRequest, TaskStatus
from training_store import TrainingStore


class PreferenceExportTests(unittest.TestCase):
    @staticmethod
    def valid_pair_store(root):
        store = TrainingStore(root)
        store.save_episode(episode(
            [bad_turn()], episode_id="episode_bad", task_id="bad",
            eligibility="failure_trajectory",
        ))
        store.save_episode(episode(
            [good_turn()], episode_id="episode_good", task_id="good",
        ))
        return store

    def test_paired_unpaired_materialization_reports_and_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self.valid_pair_store(root / "source")
            self.assertEqual(main([
                "--source", str(store.root), "--output", str(root / "dry"),
                "--profile", "paired_dpo", "--dry-run",
            ]), 0)
            self.assertFalse((root / "dry").exists())
            self.assertEqual(main([
                "--source", str(store.root), "--output", str(root / "report"),
                "--profile", "unpaired_outcome", "--report-only",
            ]), 0)
            self.assertFalse((root / "report").exists())
            for profile, expected in (("paired_dpo", 1), ("unpaired_outcome", 2)):
                output = root / profile
                self.assertEqual(main([
                    "--source", str(store.root), "--output", str(output),
                    "--profile", profile,
                ]), 0)
                dataset = next(output.iterdir())
                manifest = json.loads((dataset / "dataset-manifest.json").read_text())
                rows = []
                for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
                    content = (dataset / name).read_bytes()
                    self.assertEqual(hashlib.sha256(content).hexdigest(),
                                     manifest["output_checksums"][name])
                    rows.extend(json.loads(line) for line in content.splitlines())
                self.assertEqual(len(rows), expected)
                if profile == "paired_dpo":
                    self.assertEqual(set(rows[0]), {"prompt", "chosen", "rejected", "tools"})
                else:
                    self.assertEqual({row["label"] for row in rows}, {True, False})
                    self.assertTrue(all(set(row) == {"prompt", "completion", "label", "tools"}
                                        for row in rows))
                self.assertTrue((dataset / "quality-report.json").is_file())
                self.assertTrue((dataset / "quality-report.md").is_file())

    def test_preference_derivation_failure_cannot_fail_task(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrainingStore(Path(directory) / "source")
            context = make_context(ScriptedBackend([text_turn("answer")]))
            with patch("preference_dataset.PreferenceDatasetBuilder.index_episode",
                       side_effect=OSError("index unavailable")):
                result = make_controller(context, store).run(TaskRequest(
                    "answer", context, (directory,), enable_planning=False,
                ))
            self.assertEqual(result.status, TaskStatus.COMPLETED)
            self.assertEqual(len(list(store.iterate_metadata())), 1)


if __name__ == "__main__":
    unittest.main()
