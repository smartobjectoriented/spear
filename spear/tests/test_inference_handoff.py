import tempfile
import unittest
import subprocess
from pathlib import Path

from inference_service import FakeInferenceServiceController, GPUWorkloadOwner
from training_handoff import TrainingResourceHandoff, TrainingResourceLease
from inference_service import SSHInferenceServiceController
from tests.deployment_fixture import GPU_UUID, deployment
from training_launcher import TrainingExecutionConfiguration, TrainingLauncherError


class HandoffTests(unittest.TestCase):
    def test_service_identity_uses_fixed_status_template(self):
        class Runner:
            def __init__(self): self.calls=[]
            def run(self, argv, *, input_text=None, timeout=30):
                self.calls.append((argv,input_text))
                return subprocess.CompletedProcess(argv, 0,
                    "status=RUNNING\nhealth=READY\nidentity=verified\npgid=123\n", "")
        runner=Runner()
        cfg=deployment(inference_service_wrapper="/opt/serve.sh",
            inference_service_gpu_uuid=GPU_UUID)
        result=SSHInferenceServiceController(cfg, runner).status()
        self.assertTrue(result.identity_verified); self.assertIn("$3==b", runner.calls[0][1])

    def test_service_path_injection_rejected(self):
        with self.assertRaises(TrainingLauncherError):
            deployment(inference_service_wrapper="/opt/x;kill").validate()

    def test_successful_handoff_and_restore(self):
        with tempfile.TemporaryDirectory() as d:
            service = FakeInferenceServiceController()
            handoff = TrainingResourceHandoff(service, TrainingResourceLease(Path(d) / "lease"))
            acquired = handoff.acquire("job", GPU_UUID)
            self.assertTrue(acquired.ok); self.assertEqual(service.stop_calls, 1)
            restored = handoff.restore("job", {"stopped_by_job": True, "restart_required": True})
            self.assertEqual(service.start_calls, 1); self.assertEqual(restored["restart_result"], "READY")

    def test_other_workload_is_never_stopped(self):
        service = FakeInferenceServiceController(owner=GPUWorkloadOwner.BUSY_BY_OTHER_WORKLOAD)
        result = TrainingResourceHandoff(service).acquire("job", "GPU-x")
        self.assertFalse(result.ok); self.assertEqual(service.stop_calls, 0)

    def test_stop_failure_blocks_training(self):
        service = FakeInferenceServiceController(stop_ok=False)
        result = TrainingResourceHandoff(service).acquire("job", "GPU-x")
        self.assertFalse(result.ok); self.assertEqual(service.stop_calls, 1)

    def test_gpu_release_timeout_blocks_training(self):
        service = FakeInferenceServiceController(release_ok=False)
        result = TrainingResourceHandoff(service).acquire("job", "GPU-x")
        self.assertFalse(result.ok); self.assertEqual(service.stop_calls, 1)

    def test_already_stopped_service_is_not_restarted_by_restore(self):
        service = FakeInferenceServiceController(running=False)
        handoff = TrainingResourceHandoff(service)
        acquired = handoff.acquire("job", "GPU-x")
        self.assertTrue(acquired.ok); self.assertFalse(acquired.restart_required)
        restored = handoff.restore("job", {"stopped_by_job": False, "restart_required": False})
        self.assertFalse(restored["restart_attempted"]); self.assertEqual(service.start_calls, 0)

    def test_restart_failure_is_separate_from_training_result(self):
        service = FakeInferenceServiceController(start_ok=False)
        handoff = TrainingResourceHandoff(service)
        handoff.acquire("job", "GPU-x")
        restored = handoff.restore("job", {"stopped_by_job": True, "restart_required": True})
        self.assertEqual(restored["restart_result"], "FAILED")

    def test_lease_prevents_second_owner_and_releases(self):
        with tempfile.TemporaryDirectory() as d:
            lease = TrainingResourceLease(Path(d) / "lease")
            lease.acquire("one", "GPU-x")
            with self.assertRaises(Exception): lease.acquire("two", "GPU-x")
            lease.release("one")
            lease.acquire("two", "GPU-x")


if __name__ == "__main__": unittest.main()
