import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tests.test_sft_dataset import episode, turn
from tests.test_training_readiness import with_origin
from training_bundle import SourceModelProfile, TrainingBundleConfiguration
from training_controller import TrainingController, TrainingControllerError
from training_jobs import TrainingJobStatus
from tests.deployment_fixture import deployment
from training_launcher import (
    PreflightResult, StartResult, StatusResult, TrainingExecutionConfiguration,
)
from training_store import TrainingStore
from inference_service import FakeInferenceServiceController
from training_handoff import TrainingResourceHandoff


class FakeTrainingLauncher:
    launcher_type = "fake"
    def __init__(self):
        self.preflight_calls = 0; self.start_calls = 0; self.stop_calls = 0
        self.preflight_result = PreflightResult(True, "ok", "axolotl 1", {}, True)
        self.status_result = StatusResult(True)
        self.log_text = "step=2 loss=1.2\nHF_TOKEN=secret\nlast"
    def preflight(self, job, bundle, config_name):
        self.preflight_calls += 1; return self.preflight_result
    def start(self, job, bundle, config_name):
        self.start_calls += 1
        return StartResult(True, {"pid": 123, "pgid": 123, "marker": job.job_id}, "ok")
    def status(self, job): return self.status_result
    def stop(self, job):
        self.stop_calls += 1; return StatusResult(False, terminal=True)
    def logs(self, job, lines): return "\n".join(self.log_text.splitlines()[-lines:])
    def inspect(self, job): return {}


class TrainingControllerTests(unittest.TestCase):
    def controller(self, directory, *, positive=False, launcher=None, revision="d" * 40):
        store = TrainingStore(Path(directory) / "store")
        if positive:
            store.save_episode(episode([turn(0, output="done", labels=("final_success",))]))
        execution = deployment(source_model_revision=revision)
        return TrainingController(store, Path(directory) / "training", execution=execution,
                                  launcher=launcher or FakeTrainingLauncher())

    def test_empty_status_and_start_refusal_never_call_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = FakeTrainingLauncher(); control = self.controller(directory, launcher=launcher)
            report, jobs = control.status()
            self.assertEqual(report["sft"]["state"], "COLLECTING")
            result = control.start()
            self.assertFalse(result.ok); self.assertIn("not started", result.message)
            self.assertEqual(launcher.preflight_calls, 0); self.assertEqual(launcher.start_calls, 0)
            self.assertEqual(jobs, [])

    def test_empty_data_prevents_inference_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            service = FakeInferenceServiceController()
            control = self.controller(directory, launcher=FakeTrainingLauncher(),
                                      positive=False)
            control.resource_handoff = TrainingResourceHandoff(service)
            result = control.start()
            self.assertFalse(result.ok)
            self.assertEqual(service.stop_calls, 0)
            self.assertEqual(service.start_calls, 0)

    def test_prepare_is_local_and_default_start_is_sft(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = FakeTrainingLauncher(); control = self.controller(directory, positive=True, launcher=launcher)
            prepared = control.prepare()
            self.assertEqual(launcher.start_calls, 0); self.assertTrue(Path(prepared.bundle_path).is_dir())
            result = control.start()
            self.assertTrue(result.ok); self.assertEqual(result.job.method, "SFT")
            self.assertEqual(result.job.status, "RUNNING")

    def test_successful_training_handoff_restores_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = FakeTrainingLauncher()
            service = FakeInferenceServiceController()
            control = self.controller(directory, positive=True, launcher=launcher)
            control.resource_handoff = TrainingResourceHandoff(service)
            result = control.start()
            self.assertTrue(result.ok); self.assertEqual(service.stop_calls, 1)
            launcher.status_result = StatusResult(False, terminal=True, exit_code=0)
            completed = control.refresh(result.job)
            self.assertEqual(completed.status, TrainingJobStatus.COMPLETED.value)
            self.assertEqual(service.start_calls, 1)

    def test_preflight_failure_or_checksum_failure_prevents_train(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = FakeTrainingLauncher()
            launcher.preflight_result = PreflightResult(False, "bad preprocess")
            result = self.controller(directory, positive=True, launcher=launcher).start()
            self.assertFalse(result.ok); self.assertEqual(launcher.start_calls, 0)
            self.assertEqual(result.job.status, "FAILED")
        with tempfile.TemporaryDirectory() as directory:
            launcher = FakeTrainingLauncher()
            launcher.preflight_result = PreflightResult(True, "remote mismatch",
                                                         checksum_verified=False)
            result = self.controller(directory, positive=True, launcher=launcher).start()
            self.assertFalse(result.ok); self.assertEqual(launcher.start_calls, 0)

    def test_lineage_completion_failure_lost_and_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = FakeTrainingLauncher(); control = self.controller(directory, positive=True, launcher=launcher)
            running = control.start().job
            self.assertEqual(running.source_model_revision, "d" * 40)
            self.assertTrue(running.dataset_checksums); self.assertTrue(running.training_config_checksum)
            launcher.status_result = StatusResult(False, True, 0, completed_at="done")
            self.assertEqual(control.inspect(running.job_id).status, "COMPLETED")
        with tempfile.TemporaryDirectory() as directory:
            launcher = FakeTrainingLauncher(); control = self.controller(directory, positive=True, launcher=launcher)
            running = control.start().job
            launcher.status_result = StatusResult(False, True, 3)
            self.assertEqual(control.inspect(running.job_id).status, "FAILED")
        with tempfile.TemporaryDirectory() as directory:
            launcher = FakeTrainingLauncher(); control = self.controller(directory, positive=True, launcher=launcher)
            running = control.start().job
            launcher.status_result = StatusResult(False)
            self.assertEqual(control.inspect(running.job_id).status, "LOST")
        with tempfile.TemporaryDirectory() as directory:
            launcher = FakeTrainingLauncher(); control = self.controller(directory, positive=True, launcher=launcher)
            running = control.start().job
            self.assertEqual(control.stop(running.job_id).status, "STOPPED")
            self.assertEqual(launcher.stop_calls, 1)

    def test_missing_revision_and_gguf_are_hard_failures_even_for_force(self):
        with tempfile.TemporaryDirectory() as directory:
            control = self.controller(directory, positive=True, revision=None)
            with self.assertRaises(TrainingControllerError): control.start(force=True)
        with tempfile.TemporaryDirectory() as directory:
            store = TrainingStore(Path(directory) / "s")
            store.save_episode(episode([turn(0, output="done", labels=("final_success",))]))
            config = TrainingBundleConfiguration(model=replace(
                SourceModelProfile(), base_model="bad.gguf", revision="e" * 40))
            control = TrainingController(store, Path(directory) / "training",
                                         launcher=FakeTrainingLauncher(),
                                         bundle_configuration=config)
            with self.assertRaises(Exception): control.start(force=True)

    def test_duplicate_active_job_and_bounded_redacted_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = FakeTrainingLauncher(); control = self.controller(directory, positive=True, launcher=launcher)
            running = control.start().job
            with self.assertRaises(TrainingControllerError): control.start()
            value = control.logs(running.job_id, 2)
            self.assertNotIn("secret", value)
            with self.assertRaises(TrainingControllerError): control.logs(running.job_id, 501)

    def test_method_readiness_is_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            control = self.controller(directory, positive=True)
            self.assertFalse(control.start("kto").ok)
            self.assertFalse(control.start("dpo").ok)

    def test_real_ssh_start_requires_cached_remote_readiness(self):
        class SSHFake(FakeTrainingLauncher):
            launcher_type = "ssh"
        with tempfile.TemporaryDirectory() as directory:
            launcher = SSHFake()
            control = self.controller(directory, positive=True, launcher=launcher)
            result = control.start()
            self.assertFalse(result.ok)
            self.assertIn("remote host", result.message)
            self.assertEqual(launcher.preflight_calls, 0)

    def test_force_cannot_make_synthetic_or_empty_data_trainable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrainingStore(Path(directory) / "store")
            store.save_episode(with_origin(episode([
                turn(0, output="fake", labels=("final_success",)),
            ]), "SYNTHETIC"))
            launcher = FakeTrainingLauncher()
            control = TrainingController(store, Path(directory) / "training",
                execution=deployment(source_model_revision="d" * 40),
                launcher=launcher)
            self.assertFalse(control.start(force=True).ok)
            self.assertEqual(launcher.preflight_calls, 0)

    def test_launch_exception_is_durable_failure_and_bundle_stays_immutable(self):
        class BrokenStart(FakeTrainingLauncher):
            def start(self, job, bundle, config_name):
                self.start_calls += 1; raise OSError("transport failed")
        with tempfile.TemporaryDirectory() as directory:
            launcher = BrokenStart(); control = self.controller(directory, positive=True, launcher=launcher)
            before_episodes = list(control.training_store.iterate_metadata())
            result = control.start(); bundle = Path(result.bundle_path)
            frozen = (bundle / "checksums.sha256").read_bytes()
            self.assertFalse(result.ok); self.assertEqual(result.job.status, "FAILED")
            self.assertEqual(control.job_store.load(result.job.job_id).status, "FAILED")
            self.assertEqual((bundle / "checksums.sha256").read_bytes(), frozen)
            self.assertEqual(list(control.training_store.iterate_metadata()), before_episodes)


if __name__ == "__main__": unittest.main()
