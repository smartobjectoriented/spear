import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.test_training_jobs import job
from tests.deployment_fixture import HOST, REMOTE_ROOT, deployment
from training_launcher import (
    SSHTrainingLauncher, TrainingExecutionConfiguration, TrainingLauncherError,
)


class RecordingRunner:
    def __init__(self, outputs=()): self.calls = []; self.outputs = list(outputs)
    def run(self, argv, *, input_text=None, timeout=30):
        self.calls.append((list(argv), input_text, timeout))
        code, stdout = self.outputs.pop(0) if self.outputs else (0, "")
        return subprocess.CompletedProcess(argv, code, stdout, "")


class SSHTrainingLauncherTests(unittest.TestCase):
    def test_configuration_rejects_host_root_executable_and_revision_injection(self):
        bad = (
            {"host": "gpu-host.example;evil"}, {"remote_root": "../../tmp"},
            {"remote_root": "/"}, {"remote_root": "/srv/train;evil"},
            {"remote_root": "/srv/train root"},
            {"axolotl_executable": "axolotl;rm"},
            {"source_model_revision": "main"},
            {"source_model_revision": "main; evil"},
        )
        for values in bad:
            with self.subTest(values=values), self.assertRaises(TrainingLauncherError):
                deployment(**values).validate()

    def test_an_unconfigured_deployment_says_so_instead_of_guessing(self):
        """These four used to default to one institute's machine and layout,
        so an operator who had configured nothing launched against somebody
        else's host. Missing configuration is now its own error, distinct from
        bad configuration, and names the field to set."""
        cases = (
            ({"host": ""}, "no SSH host"),
            ({"remote_root": ""}, "no remote training root"),
            ({"inference_service_binary": None}, "inference service binary"),
            ({"inference_service_model": None}, "inference service model"),
        )
        for change, expected in cases:
            with self.subTest(**change):
                with self.assertRaises(TrainingLauncherError) as raised:
                    deployment(**change).validate()

                self.assertIn(expected, str(raised.exception))

    def test_a_disabled_inference_service_needs_no_paths(self):
        """Requiring them everywhere would force an operator who manages the
        server by hand to invent two paths nothing reads."""
        deployment(inference_service_type="disabled",
                   inference_service_binary=None,
                   inference_service_model=None).validate()

    def test_remote_directory_is_confined_and_opaque(self):
        launcher = SSHTrainingLauncher(deployment())
        self.assertEqual(launcher.remote_directory("ftjob_" + "a" * 24),
                         REMOTE_ROOT + "/jobs/ftjob_" + "a" * 24)
        with self.assertRaises(TrainingLauncherError): launcher.remote_directory("../../x")

    def test_probe_uses_ssh_alias_fixed_script_and_no_shell_mode(self):
        runner = RecordingRunner([(0, "hostname=gpu-host.example\naxolotl_version=axolotl 0.12\n")])
        launcher = SSHTrainingLauncher(deployment(), runner)
        result = launcher.probe()
        self.assertTrue(result.success); self.assertIn("axolotl", result.axolotl_version)
        argv, script, _ = runner.calls[0]
        self.assertEqual(argv[:4], ["ssh", "--", HOST, "sh"])
        self.assertIn("command -v", script); self.assertNotIn("rm -rf", script)

    def test_stop_requires_stored_process_group_before_remote_command(self):
        launcher = SSHTrainingLauncher(deployment(), RecordingRunner())
        value = job(); value.remote_process_identity = {"pid": 12}
        with self.assertRaises(TrainingLauncherError): launcher.stop(value)
        self.assertEqual(launcher.runner.calls, [])

    def test_logs_are_bounded_at_transport(self):
        launcher = SSHTrainingLauncher(deployment(), RecordingRunner())
        with self.assertRaises(TrainingLauncherError): launcher.logs(job(), 501)

    def test_conservative_progress_parser(self):
        value = SSHTrainingLauncher._parse_metrics(
            "SPEAR_LOG:{'loss': 1.25, 'learning_rate': 2e-5, 'epoch': 0.5, 'step': 12}\n")
        self.assertEqual(value["step"], 12); self.assertEqual(value["loss"], 1.25)


if __name__ == "__main__": unittest.main()
