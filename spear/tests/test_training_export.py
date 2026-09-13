import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_sft_dataset import episode, turn
from tests.test_agent_runtime import ScriptedBackend, make_context, text_turn
from tests.test_training_data import make_controller
from task_controller import TaskRequest, TaskStatus
from training_export import main
from training_store import TrainingStore


class TrainingExportCLITests(unittest.TestCase):
    def test_dry_run_and_materialize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = TrainingStore(root / "source")
            store.save_episode(episode([
                turn(0, output="answer", labels=("final_success",)),
            ]))
            self.assertEqual(main([
                "--source", str(store.root), "--output", str(root / "dry"),
                "--dry-run", "--project", "proj",
            ]), 0)
            self.assertFalse((root / "dry").exists())
            self.assertEqual(main([
                "--source", str(store.root), "--output", str(root / "out"),
                "--profile", "final_response_only",
            ]), 0)
            datasets = list((root / "out").iterdir())
            self.assertEqual(len(datasets), 1)
            records = []
            for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
                records.extend(json.loads(line) for line in
                               (datasets[0] / name).read_text().splitlines())
            self.assertEqual(len(records), 1)
            self.assertEqual(set(records[0]), {"messages", "tools"})

    def test_post_task_derivation_failure_cannot_fail_task(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrainingStore(Path(directory) / "source")
            context = make_context(ScriptedBackend([text_turn("answer")]))
            with patch("sft_dataset.SFTDatasetBuilder.persist_episode_candidates",
                       side_effect=OSError("secondary storage failed")):
                result = make_controller(context, store).run(TaskRequest(
                    "answer", context, (directory,), enable_planning=False,
                ))
            self.assertEqual(result.status, TaskStatus.COMPLETED)
            self.assertEqual(len(list(store.iterate_metadata())), 1)


if __name__ == "__main__":
    unittest.main()
