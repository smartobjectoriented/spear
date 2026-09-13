import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from remote_readiness import (
    RemoteReadinessState, SSHDiagnosis, SSHRemoteProbe, local_doctor,
    load_cached_remote_readiness, save_remote_readiness,
)
from tests.deployment_fixture import deployment
from training_launcher import TrainingExecutionConfiguration, TrainingLauncherError
from training_store import TrainingStore


class ProbeRunner:
    def __init__(self, results):
        self.results = list(results); self.calls = []

    def run(self, argv, *, input_text=None, timeout=30):
        self.calls.append((list(argv), input_text, timeout))
        value = self.results.pop(0)
        if isinstance(value, BaseException): raise value
        code, out, err = value
        return subprocess.CompletedProcess(argv, code, out, err)


def ok(output=""):
    return (0, output, "")


class RemoteReadinessTests(unittest.TestCase):
    def config(self, **changes):
        value = dict(axolotl_executable="axolotl", python_executable="python3",
                     minimum_free_vram_gib=None)
        value.update(changes)

        return deployment(**value)

    def test_ssh_authentication_failure_is_classified_without_secret_output(self):
        runner = ProbeRunner([ok("hostname gpu-host.example\n"),
                              (255, "", "operator@gpu-host.example: Permission denied (publickey,password).")])
        report = SSHRemoteProbe(self.config(), runner).inspect()
        self.assertEqual(report.state, RemoteReadinessState.UNKNOWN.value)
        self.assertIn("authentication_unavailable", report.reasons)
        self.assertNotIn("publickey", json.dumps(report.to_dict()))
        self.assertEqual(len(runner.calls), 2)

    def test_ssh_failure_classes(self):
        cases = (("Could not resolve hostname gpu-host.example", SSHDiagnosis.CONNECTIVITY_FAILURE),
                 ("Host key verification failed", SSHDiagnosis.HOST_KEY_FAILURE),
                 ("Connection timed out", SSHDiagnosis.TIMEOUT))
        for stderr, expected in cases:
            with self.subTest(stderr=stderr):
                runner = ProbeRunner([(255, "", stderr)])
                diagnosis, _ = SSHRemoteProbe(self.config(), runner).diagnose_ssh()
                self.assertEqual(diagnosis, expected)
        runner = ProbeRunner([(255, "", "Bad configuration option")])
        diagnosis, _ = SSHRemoteProbe(self.config(), runner).diagnose_ssh()
        self.assertEqual(diagnosis, SSHDiagnosis.ALIAS_MISSING)

    def test_remote_success_parses_environment_and_ready_state(self):
        output = ("hostname=gpu-host.example\npython=Python 3.12.2\n"
                  "axolotl_available=yes\n"
                  "axolotl_version=axolotl 0.16.1\ncuda=available\n"
                  "gpu=RTX 6000 Ada, 49152, 46000, 550.1\n"
                  "training_root=usable\ndisk_available_kib=80000\n"
                  "ram_kib=1000000\npackage=torch:2.12.0\n"
                  "package=transformers:5.8.0\npackage=peft:0.18.0\n"
                  "package=bitsandbytes:available\npackage=qwen3_next:available\n")
        runner = ProbeRunner([ok("host config"), ok(), ok(output)])
        report = SSHRemoteProbe(self.config(), runner).inspect()
        self.assertEqual(report.state, RemoteReadinessState.READY.value)
        self.assertEqual(report.checks["hostname"], "gpu-host.example")
        self.assertEqual(report.checks["package"]["torch"], "2.12.0")
        self.assertEqual(len(report.checks["gpu"]), 1)
        self.assertEqual(len(runner.calls), 3)
        self.assertIn("CUDA_VISIBLE_DEVICES", runner.calls[2][1])

    def test_missing_dependencies_and_low_python_are_blocked(self):
        output = ("python=Python 3.10.9\naxolotl_available=no\ncuda=unavailable\n"
                  "training_root=unavailable\npackage=torch:missing\n")
        runner = ProbeRunner([ok(), ok(), ok(output)])
        report = SSHRemoteProbe(self.config(minimum_free_vram_gib=48), runner).inspect()
        self.assertEqual(report.state, RemoteReadinessState.BLOCKED.value)
        self.assertIn("python_too_old", report.reasons)
        self.assertIn("axolotl_missing", report.reasons)
        self.assertIn("training_root_unavailable", report.reasons)

    def test_old_pytorch_is_blocked(self):
        output = ("python=Python 3.12.2\naxolotl_available=yes\ncuda=available\n"
                  "gpu=GPU, 49152, 46000, driver\ntraining_root=usable\n"
                  "package=torch:2.10.0\npackage=transformers:5\npackage=peft:1\n"
                  "package=bitsandbytes:available\n")
        report = SSHRemoteProbe(self.config(), ProbeRunner([ok(), ok(), ok(output)])).inspect()
        self.assertEqual(report.state, RemoteReadinessState.BLOCKED.value)
        self.assertIn("pytorch_too_old", report.reasons)

    def test_warning_for_insufficient_current_free_vram(self):
        output = ("python=Python 3.12.2\naxolotl_available=yes\ncuda=available\n"
                  "gpu=GPU, 49152, 32000, driver\ntraining_root=usable\n"
                  "package=torch:2.12\npackage=transformers:5\npackage=peft:1\n"
                  "package=bitsandbytes:available\n")
        runner = ProbeRunner([ok(), ok(), ok(output)])
        report = SSHRemoteProbe(self.config(minimum_free_vram_gib=40), runner).inspect()
        self.assertEqual(report.state, RemoteReadinessState.READY.value)
        self.assertIn("insufficient_current_free_vram", report.reasons)

    def test_total_capacity_ready_while_current_gpu_is_busy(self):
        output = ("python=Python 3.12.2\naxolotl_available=yes\naxolotl_version=axolotl 0.18\n"
                  "cuda=available\n"
                  "gpu=RTX PRO 6000, 97887, 82651, 14599, 0, 595.84\n"
                  "gpu_process=123, llama-server, 82628\ntraining_root=usable\n"
                  "package=torch:2.12\npackage=transformers:5\npackage=peft:1\n"
                  "package=bitsandbytes:available\n")
        report = SSHRemoteProbe(self.config(minimum_free_vram_gib=47),
                                ProbeRunner([ok(), ok(), ok(output)])).inspect()
        self.assertEqual(report.checks["hardware_capacity"], "READY")
        self.assertEqual(report.checks["current_gpu_availability"], "BUSY")
        self.assertEqual(report.state, RemoteReadinessState.READY.value)

    def test_total_capacity_insufficient_is_distinct_from_current_availability(self):
        output = ("python=Python 3.12.2\naxolotl_available=yes\ncuda=available\n"
                  "gpu=GPU, 32768, 1000, 31768, 0, driver\ntraining_root=usable\n"
                  "package=torch:2.12\npackage=transformers:5\npackage=peft:1\n"
                  "package=bitsandbytes:available\n")
        report = SSHRemoteProbe(self.config(minimum_free_vram_gib=47),
                                ProbeRunner([ok(), ok(), ok(output)])).inspect()
        self.assertEqual(report.checks["hardware_capacity"], "INSUFFICIENT")
        self.assertEqual(report.checks["current_gpu_availability"], "BUSY")

    def test_probe_rejects_injection_and_uses_fixed_templates(self):
        with self.assertRaises(TrainingLauncherError):
            SSHRemoteProbe(self.config(remote_root="/srv/root;touch"))
        runner = ProbeRunner([ok(), ok(), ok("python=Python 3.12\n")])
        SSHRemoteProbe(self.config(), runner).inspect()
        argv, script, _ = runner.calls[2]
        self.assertEqual(argv[:4], ["ssh", "-o", "BatchMode=yes", "-o"])
        self.assertIn("importlib", script)
        self.assertNotIn("$SHELL", script)

    def test_assigned_gpu_selector_is_the_only_gpu_queried(self):
        selector = "GPU-00000000-0000-0000-0000-000000000000"
        runner = ProbeRunner([ok(), ok(), ok("python=Python 3.12\n")])
        SSHRemoteProbe(self.config(cuda_visible_devices=selector), runner).inspect()
        script = runner.calls[2][1]
        self.assertIn('CUDA_VISIBLE_DEVICES="$visible"', script)
        self.assertIn('nvidia-smi --id="$visible"', script)
        with self.assertRaises(TrainingLauncherError):
            SSHRemoteProbe(self.config(cuda_visible_devices="GPU-other;kill"))

    def test_local_doctor_and_cache_are_separate_from_training_store_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrainingStore(Path(directory) / "store")
            local = local_doctor(store, Path(directory) / "training", None)
            self.assertEqual(local.checks["training_store"]["status"], "OK")
            self.assertIn("model_revision_missing", local.reasons)
            self.assertIsNone(load_cached_remote_readiness(store))
            runner = ProbeRunner([ok(), (255, "", "Permission denied")])
            report = SSHRemoteProbe(self.config(), runner).inspect()
            save_remote_readiness(store, report)
            cached = load_cached_remote_readiness(store)
            self.assertEqual(cached.state, RemoteReadinessState.UNKNOWN.value)
            self.assertEqual(sum(1 for _ in store.iterate_metadata()), 0)


if __name__ == "__main__":
    unittest.main()
