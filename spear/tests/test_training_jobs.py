import json
import tempfile
import unittest
from pathlib import Path

from training_jobs import (
    TrainingJob, TrainingJobError, TrainingJobStatus, TrainingJobStore,
)


def job(job_id="ftjob_" + "a" * 24):
    return TrainingJob(
        job_id, "SFT", "training_bundle", "/bundle", "b" * 64, "c" * 64,
        "Qwen/Qwen3-Coder-Next", "deadbeef", {"train": "d" * 64},
        "fake", "fake", f"fake://jobs/{job_id}",
    )


class TrainingJobStoreTests(unittest.TestCase):
    def test_atomic_persistence_reload_and_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrainingJobStore(directory)
            value = store.create(job())
            self.assertEqual(store.load(value.job_id).bundle_id, "training_bundle")
            root = Path(directory) / value.job_id
            self.assertTrue((root / "job.json").is_file())
            self.assertTrue((root / "events.jsonl").is_file())
            self.assertEqual(len(TrainingJobStore(directory).list()), 1)

    def test_path_traversal_and_malformed_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrainingJobStore(directory)
            for value in ("../../x", "ftjob_bad", "ftjob_" + "g" * 24):
                with self.assertRaises(TrainingJobError): store.load(value)

    def test_valid_and_invalid_state_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrainingJobStore(directory); value = store.create(job())
            value = store.transition(value, TrainingJobStatus.PREFLIGHT)
            value = store.transition(value, TrainingJobStatus.STARTING)
            value = store.transition(value, TrainingJobStatus.RUNNING)
            value = store.transition(value, TrainingJobStatus.COMPLETED, exit_code=0)
            self.assertEqual(value.status, "COMPLETED")
            with self.assertRaises(TrainingJobError):
                store.transition(value, TrainingJobStatus.RUNNING)

    def test_event_journal_recovers_corrupt_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrainingJobStore(directory); value = store.create(job())
            value = store.transition(value, TrainingJobStatus.PREFLIGHT)
            (Path(directory) / value.job_id / "job.json").write_text("{")
            self.assertEqual(store.load(value.job_id).status, "PREFLIGHT")

    def test_unknown_schema_fields_rejected(self):
        raw = job().to_dict(); raw["secret_extension"] = "x"
        with self.assertRaises(TrainingJobError): TrainingJob.from_dict(raw)


if __name__ == "__main__":
    unittest.main()
