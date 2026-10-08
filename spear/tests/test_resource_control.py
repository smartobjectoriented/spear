"""Resource control: execution profiles, rlimits, cgroups and systemd scopes."""

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harness.sandbox

from harness.resource_control import (
    CgroupAvailability,
    CgroupLimits,
    DEFAULT_CGROUP_LIMITS,
    DEFAULT_RESOURCE_LIMITS,
    ExecutionProfile,
    ResourceLimits,
    SystemdScopeRunner,
)
from harness.tool_primitives import Capability, NetworkBackend, ToolResult
from harness.sandbox import BubblewrapSandbox
from harness.tool_runtime import CommandRunner
from harness.workspace import Workspace


class ExecutionProfileTests(unittest.TestCase):
    def test_profiles_are_pure_translations_of_granted_capabilities(self):
        readonly = ExecutionProfile.from_capabilities({Capability.FILESYSTEM_READ})
        self.assertTrue(readonly.workspace_read)
        self.assertFalse(readonly.workspace_write)
        self.assertFalse(readonly.shell_complex)
        self.assertFalse(readonly.network)

        writable = ExecutionProfile.from_capabilities({
            Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE,
        })
        self.assertTrue(writable.workspace_write)
        shell = ExecutionProfile.from_capabilities({
            Capability.FILESYSTEM_READ, Capability.SHELL_COMPLEX,
        })
        self.assertTrue(shell.shell_complex)

    def test_profiles_reject_inconsistent_sensitive_capabilities(self):
        with self.assertRaises(ValueError):
            ExecutionProfile.from_capabilities({Capability.WORKSPACE_WRITE})
        with self.assertRaises(ValueError):
            ExecutionProfile.from_capabilities({Capability.SSH})
        with self.assertRaises(ValueError):
            ExecutionProfile.from_capabilities({Capability.REMOTE_WRITE})

    def test_sensitive_profiles_are_representable_but_not_sandbox_implemented(self):
        network = ExecutionProfile.from_capabilities({Capability.NETWORK})
        gpu = ExecutionProfile.from_capabilities({Capability.GPU})
        container = ExecutionProfile.from_capabilities({Capability.CONTAINER_RUNTIME})
        secrets = ExecutionProfile.from_capabilities({Capability.SECRETS})
        self.assertTrue(network.network)
        self.assertTrue(gpu.gpu)
        self.assertTrue(container.container_runtime)
        self.assertTrue(secrets.secrets_allowed)


class ResourceLimitsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name) / "workspace"
        root.mkdir()
        self.workspace = Workspace.from_path(root)

    def tearDown(self):
        self.temp.cleanup()

    def test_invalid_limits_are_rejected_at_contract_construction(self):
        for kwargs in (
            {"nofile": 0},
            {"nofile": -1},
            {"core_bytes": -1},
            {"cpu_seconds": -1},
            {"file_size_bytes": -1},
            {"nofile": "4096"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ResourceLimits(**kwargs)

    def test_default_limits_only_enable_core_and_nofile(self):
        self.assertEqual(DEFAULT_RESOURCE_LIMITS.nofile, 4096)
        self.assertEqual(DEFAULT_RESOURCE_LIMITS.core_bytes, 0)
        self.assertIsNone(DEFAULT_RESOURCE_LIMITS.cpu_seconds)
        self.assertIsNone(DEFAULT_RESOURCE_LIMITS.file_size_bytes)

    def test_build_argv_wraps_command_with_prlimit(self):
        sandbox = BubblewrapSandbox()
        argv = sandbox.build_argv(
            self.workspace,
            ["/bin/sh", "-lc", "echo constrained"],
            resource_limits=ResourceLimits(
                nofile=4096,
                core_bytes=0,
                cpu_seconds=7,
                file_size_bytes=12345,
            ),
        )
        wrapper_at = argv.index("/usr/bin/prlimit")
        self.assertEqual(
            argv[wrapper_at:],
            [
                "/usr/bin/prlimit",
                "--nofile=4096:4096",
                "--core=0:0",
                "--cpu=7:7",
                "--fsize=12345:12345",
                "--",
                "/bin/sh", "-lc", "echo constrained",
            ],
        )

    def test_no_active_limit_omits_prlimit(self):
        argv = BubblewrapSandbox().build_argv(
            self.workspace, ["/bin/true"], resource_limits=ResourceLimits()
        )
        self.assertNotIn("/usr/bin/prlimit", argv)

    def test_active_limits_fail_closed_when_prlimit_is_unavailable(self):
        sandbox = BubblewrapSandbox(prlimit_binary="/definitely/missing/prlimit")
        with patch("harness.sandbox.subprocess.run") as run:
            result = sandbox.run(
                self.workspace, ["/bin/true"], resource_limits=DEFAULT_RESOURCE_LIMITS
            )
        self.assertEqual(result.status, "failed")
        self.assertIn("prlimit", result.summary)
        run.assert_not_called()

    def test_real_sandbox_applies_hard_limits_and_children_inherit(self):
        script = (
            "import json,resource,subprocess,sys\n"
            "child=subprocess.check_output([sys.executable, '-c', "
            "'import resource,json; print(json.dumps(resource.getrlimit(resource.RLIMIT_NOFILE)))'], text=True)\n"
            "raised=True\n"
            "try: resource.setrlimit(resource.RLIMIT_NOFILE, (8192,8192))\n"
            "except (ValueError,OSError): raised=False\n"
            "print(json.dumps({'nofile': resource.getrlimit(resource.RLIMIT_NOFILE), "
            "'core': resource.getrlimit(resource.RLIMIT_CORE), 'child': json.loads(child), "
            "'raise_allowed': raised}))\n"
        )
        result = BubblewrapSandbox().run(
            self.workspace,
            ["/usr/bin/python3", "-c", script],
            resource_limits=DEFAULT_RESOURCE_LIMITS,
        )
        self.assertTrue(result.ok, result.to_legacy_text())
        observed = json.loads(result.stdout)
        self.assertEqual(observed["nofile"], [4096, 4096])
        self.assertEqual(observed["core"], [0, 0])
        self.assertEqual(observed["child"], [4096, 4096])
        self.assertFalse(observed["raise_allowed"])


class CgroupLimitsContractTests(unittest.TestCase):
    """The dataclass is a pure, immutable contract: no systemd knowledge."""

    def test_an_empty_contract_is_inactive(self):
        self.assertFalse(CgroupLimits().active)
        self.assertEqual(CgroupLimits().systemd_properties(), [])
        self.assertEqual(CgroupLimits().required_controllers, frozenset())

    def test_production_defaults_are_active_and_calibrated(self):
        """The values calibrated in STEP 1d-3d, now in force."""
        self.assertTrue(DEFAULT_CGROUP_LIMITS.active)
        self.assertEqual(DEFAULT_CGROUP_LIMITS.memory_max_bytes, 2 * 1024 ** 3)
        self.assertEqual(DEFAULT_CGROUP_LIMITS.memory_swap_max_bytes, 0)
        self.assertEqual(DEFAULT_CGROUP_LIMITS.tasks_max, 256)
        self.assertEqual(DEFAULT_CGROUP_LIMITS.cpu_quota_percent, 800)
        self.assertEqual(DEFAULT_CGROUP_LIMITS.systemd_properties(), [
            "MemoryMax=2147483648",
            "MemorySwapMax=0",
            "TasksMax=256",
            "CPUQuota=800%",
        ])
        self.assertEqual(DEFAULT_CGROUP_LIMITS.required_controllers,
                         frozenset({"cpu", "memory", "pids"}))

    def test_production_defaults_keep_headroom_over_the_measured_peaks(self):
        """Guards against a future tightening below what real work needs.

        Measured peaks on the calibration workloads: 518 MiB and 47 tasks,
        both reached by ``make -j22``.
        """
        self.assertGreaterEqual(DEFAULT_CGROUP_LIMITS.memory_max_bytes,
                                3 * 518 * 1024 ** 2)
        self.assertGreaterEqual(DEFAULT_CGROUP_LIMITS.tasks_max, 3 * 47)

    def test_each_field_alone_activates_the_contract(self):
        for kwargs in (
            {"memory_max_bytes": 1},
            {"memory_swap_max_bytes": 0},
            {"tasks_max": 1},
            {"cpu_quota_percent": 1},
        ):
            with self.subTest(**kwargs):
                self.assertTrue(CgroupLimits(**kwargs).active)

    def test_valid_values_are_accepted(self):
        limits = CgroupLimits(memory_max_bytes=268435456, memory_swap_max_bytes=0,
                              tasks_max=64, cpu_quota_percent=200)
        self.assertTrue(limits.active)
        self.assertEqual(limits.memory_max_bytes, 268435456)
        self.assertEqual(limits.cpu_quota_percent, 200)

    def test_invalid_values_are_rejected(self):
        for kwargs in (
            {"memory_max_bytes": 0},
            {"memory_max_bytes": -1},
            {"memory_swap_max_bytes": -1},
            {"tasks_max": 0},
            {"tasks_max": -3},
            {"cpu_quota_percent": 0},
            {"cpu_quota_percent": -50},
            {"memory_max_bytes": 1.5},
            {"memory_max_bytes": "64M"},
            {"cpu_quota_percent": "200%"},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    CgroupLimits(**kwargs)

    def test_bool_is_explicitly_rejected(self):
        for field in ("memory_max_bytes", "memory_swap_max_bytes", "tasks_max",
                      "cpu_quota_percent"):
            for value in (True, False):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        CgroupLimits(**{field: value})

    def test_is_frozen(self):
        limits = CgroupLimits(tasks_max=8)
        with self.assertRaises(Exception):
            limits.tasks_max = 9

    def test_swap_is_not_implied_by_memory_max(self):
        """Coupling the two is a 1d-3d policy decision, not a dataclass rule."""
        limits = CgroupLimits(memory_max_bytes=1024)
        self.assertIsNone(limits.memory_swap_max_bytes)
        self.assertEqual(limits.systemd_properties(), ["MemoryMax=1024"])

    def test_systemd_properties_are_exact_and_stably_ordered(self):
        limits = CgroupLimits(memory_max_bytes=268435456, memory_swap_max_bytes=0,
                              tasks_max=64, cpu_quota_percent=200)
        self.assertEqual(limits.systemd_properties(), [
            "MemoryMax=268435456",
            "MemorySwapMax=0",
            "TasksMax=64",
            "CPUQuota=200%",
        ])

    def test_required_controllers_track_only_the_active_fields(self):
        self.assertEqual(CgroupLimits(memory_max_bytes=1).required_controllers,
                         frozenset({"memory"}))
        self.assertEqual(CgroupLimits(memory_swap_max_bytes=0).required_controllers,
                         frozenset({"memory"}))
        self.assertEqual(CgroupLimits(tasks_max=1).required_controllers,
                         frozenset({"pids"}))
        self.assertEqual(CgroupLimits(cpu_quota_percent=1).required_controllers,
                         frozenset({"cpu"}))
        self.assertEqual(
            CgroupLimits(memory_max_bytes=1, tasks_max=1, cpu_quota_percent=1)
            .required_controllers,
            frozenset({"memory", "pids", "cpu"}),
        )

    def test_no_out_of_scope_resource_knobs_are_introduced(self):
        rendered = " ".join(
            CgroupLimits(memory_max_bytes=1, memory_swap_max_bytes=0, tasks_max=1,
                         cpu_quota_percent=1).systemd_properties()
        )
        for forbidden in ("IO", "io.max", "cpuset", "AllowedCPUs", "hugetlb",
                          "LimitNPROC", "LimitAS"):
            self.assertNotIn(forbidden, rendered)


class SystemdScopeRunnerTests(unittest.TestCase):
    """The scope runner knows systemd and nothing else."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runtime_dir = Path(self.temp.name) / "runtime"
        self.runtime_dir.mkdir()
        self.bus = self.runtime_dir / "bus"
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.bind(str(self.bus))
        self.addCleanup(self._socket.close)
        self.addCleanup(self.temp.cleanup)

    def runner(self, **kwargs):
        kwargs.setdefault("runtime_dir", self.runtime_dir)
        return SystemdScopeRunner(**kwargs)

    def available_runner(self, **kwargs):
        runner = self.runner(**kwargs)
        runner._delegated_controllers = lambda: frozenset({"cpu", "memory", "pids"})
        return runner

    # -- unit names ---------------------------------------------------------

    def test_unit_name_is_unique_and_uses_safe_characters_only(self):
        names = {SystemdScopeRunner.unit_name() for _ in range(200)}
        self.assertEqual(len(names), 200)
        for name in names:
            self.assertRegex(name, r"^spear-tool-[0-9a-f]{32}\.scope$")

    def test_runtime_generated_names_carry_the_product_name(self):
        """The scope prefix and the two temporary-directory prefixes are the
        names this runtime leaves where an operator or another tool can see
        them -- `systemctl list-units`, a cgroup path, /tmp. They were
        renamed off the old product name together; nothing parses any of them
        back, so the only way a half-rename shows up is here.  The absence of
        the old prefixes is the public boundary scan's job, not this one's --
        asserting it here would only plant the stale literals it looks for."""

        self.assertEqual(SystemdScopeRunner.UNIT_PREFIX, "spear-tool-")
        self.assertTrue(SystemdScopeRunner.unit_name().startswith("spear-tool-"))

        source = Path(harness.sandbox.__file__).read_text()
        self.assertIn('prefix="spear-slirp-"', source)
        self.assertIn('prefix="spear-sandbox-"', source)

    def test_unit_name_never_embeds_caller_data(self):
        name = SystemdScopeRunner.unit_name()
        self.assertNotIn("/", name)
        self.assertNotIn(" ", name)

    # -- argv ---------------------------------------------------------------

    def test_inactive_limits_omit_the_wrapper_entirely(self):
        runner = self.available_runner()
        argv = ["/usr/bin/bwrap", "--die-with-parent", "/bin/true"]
        self.assertEqual(runner.wrap(argv, CgroupLimits(), unit="x.scope"), argv)

    def test_exact_systemd_run_argv(self):
        runner = self.available_runner()
        limits = CgroupLimits(memory_max_bytes=268435456, memory_swap_max_bytes=0,
                              tasks_max=64, cpu_quota_percent=200)
        unit = "spear-tool-00000000000000000000000000000000.scope"
        self.assertEqual(
            runner.wrap(["/usr/bin/bwrap", "--die-with-parent", "/bin/true"],
                        limits, unit=unit),
            [
                "/usr/bin/systemd-run", "--user", "--scope", "--quiet", "--collect",
                f"--unit={unit}",
                "-p", "MemoryMax=268435456",
                "-p", "MemorySwapMax=0",
                "-p", "TasksMax=64",
                "-p", "CPUQuota=200%",
                "--",
                "/usr/bin/bwrap", "--die-with-parent", "/bin/true",
            ],
        )

    def test_partial_limits_emit_only_the_requested_properties(self):
        runner = self.available_runner()
        argv = runner.wrap(["/usr/bin/bwrap"], CgroupLimits(tasks_max=8), unit="u.scope")
        self.assertIn("-p", argv)
        self.assertIn("TasksMax=8", argv)
        rendered = " ".join(argv)
        self.assertNotIn("MemoryMax", rendered)
        self.assertNotIn("MemorySwapMax", rendered)
        self.assertNotIn("CPUQuota", rendered)

    def test_wrap_never_uses_a_shell(self):
        runner = self.available_runner()
        argv = runner.wrap(["/usr/bin/bwrap"], CgroupLimits(tasks_max=8), unit="u.scope")
        self.assertNotIn("/bin/sh", argv)
        self.assertNotIn("-c", argv)

    def test_empty_inner_argv_is_rejected(self):
        with self.assertRaises(ValueError):
            self.available_runner().wrap([], CgroupLimits(tasks_max=8), unit="u.scope")

    # -- supervisor environment --------------------------------------------

    def test_supervisor_env_is_minimal_and_carries_no_secret(self):
        env = self.available_runner().supervisor_env()
        self.assertEqual(set(env), {"XDG_RUNTIME_DIR"})
        self.assertEqual(env["XDG_RUNTIME_DIR"], str(self.runtime_dir))

    # -- availability -------------------------------------------------------

    def test_inactive_limits_need_no_systemd_at_all(self):
        runner = self.runner(systemd_run_binary="/definitely/missing/systemd-run",
                             systemctl_binary="/definitely/missing/systemctl")
        self.assertIsNone(runner.availability_for(CgroupLimits()))

    def test_missing_systemd_run_fails_closed(self):
        runner = self.available_runner(systemd_run_binary="/definitely/missing/systemd-run")
        result = runner.availability_for(CgroupLimits(tasks_max=8))
        self.assertIsNotNone(result)
        self.assertEqual(result.status, "failed")
        self.assertIn("systemd-run", result.summary)
        self.assertEqual(runner.availability, CgroupAvailability.SYSTEMD_RUN_ABSENT)

    def test_missing_systemctl_fails_closed(self):
        runner = self.available_runner(systemctl_binary="/definitely/missing/systemctl")
        result = runner.availability_for(CgroupLimits(tasks_max=8))
        self.assertIsNotNone(result)
        self.assertEqual(result.status, "failed")
        self.assertEqual(runner.availability, CgroupAvailability.SYSTEMCTL_ABSENT)

    def test_missing_user_bus_fails_closed(self):
        self.bus.unlink()
        runner = self.available_runner()
        result = runner.availability_for(CgroupLimits(tasks_max=8))
        self.assertIsNotNone(result)
        self.assertEqual(result.status, "failed")
        self.assertEqual(runner.availability, CgroupAvailability.USER_BUS_UNAVAILABLE)

    def test_bus_that_is_not_a_socket_fails_closed(self):
        self.bus.unlink()
        self.bus.write_text("not a socket")
        runner = self.available_runner()
        result = runner.availability_for(CgroupLimits(tasks_max=8))
        self.assertIsNotNone(result)
        self.assertEqual(runner.availability, CgroupAvailability.USER_BUS_UNAVAILABLE)

    def test_missing_controller_fails_closed(self):
        runner = self.runner()
        runner._delegated_controllers = lambda: frozenset({"memory", "pids"})
        result = runner.availability_for(CgroupLimits(cpu_quota_percent=50))
        self.assertIsNotNone(result)
        self.assertEqual(result.status, "failed")
        self.assertIn("cpu", result.summary)
        self.assertEqual(runner.availability, CgroupAvailability.CONTROLLERS_UNAVAILABLE)

    def test_only_the_controllers_actually_needed_are_required(self):
        runner = self.runner()
        runner._delegated_controllers = lambda: frozenset({"memory"})
        self.assertIsNone(runner.availability_for(CgroupLimits(memory_max_bytes=1024)))
        self.assertIsNotNone(runner.availability_for(CgroupLimits(tasks_max=8)))

    def test_available_state_is_recorded(self):
        runner = self.available_runner()
        self.assertIsNone(runner.availability_for(CgroupLimits(tasks_max=8)))
        self.assertEqual(runner.availability, CgroupAvailability.AVAILABLE)

    def test_linger_is_never_enabled_automatically(self):
        runner = self.available_runner()
        with patch("harness.resource_control.subprocess.run") as run:
            runner.availability_for(CgroupLimits(tasks_max=8))
        for call in run.call_args_list:
            self.assertNotIn("loginctl", " ".join(call.args[0]))

    # -- termination --------------------------------------------------------

    def test_terminate_uses_systemctl_kill_with_a_bounded_timeout(self):
        runner = self.available_runner()
        with patch("harness.resource_control.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "", "")
            runner.terminate("spear-tool-abc.scope")
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], [
            "/usr/bin/systemctl", "--user", "kill", "--kill-whom=all",
            "--signal=KILL", "spear-tool-abc.scope",
        ])
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertIsNotNone(run.call_args.kwargs["timeout"])

    def test_terminate_never_walks_pid_trees_or_writes_cgroup_kill(self):
        runner = self.available_runner()
        with patch("harness.resource_control.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "", "")
            runner.terminate("spear-tool-abc.scope")
        rendered = " ".join(run.call_args.args[0])
        for forbidden in ("pkill", "pgrep", "cgroup.kill", "cgroup.procs"):
            self.assertNotIn(forbidden, rendered)

    def test_terminate_is_contained_when_systemctl_fails_or_hangs(self):
        runner = self.available_runner()
        for effect in (subprocess.TimeoutExpired("systemctl", 5),
                       OSError("boom"),
                       subprocess.CompletedProcess([], 1, "", "no such unit")):
            with self.subTest(effect=type(effect).__name__):
                with patch("harness.resource_control.subprocess.run") as run:
                    if isinstance(effect, subprocess.CompletedProcess):
                        run.return_value = effect
                    else:
                        run.side_effect = effect
                    self.assertIsInstance(runner.terminate("spear-tool-abc.scope"), bool)


class CgroupIntegrationContractTests(unittest.TestCase):
    """How BubblewrapSandbox / CommandRunner consume the contract (mocked)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.workspace = Workspace.from_path(self.workspace_root)
        self.runtime_dir = Path(self.temp.name) / "runtime"
        self.runtime_dir.mkdir()
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.bind(str(self.runtime_dir / "bus"))
        self.addCleanup(self._socket.close)
        self.addCleanup(self.temp.cleanup)
        self.limits = CgroupLimits(memory_max_bytes=268435456, memory_swap_max_bytes=0,
                                   tasks_max=64, cpu_quota_percent=100)

    def scope_runner(self, **kwargs):
        runner = SystemdScopeRunner(runtime_dir=self.runtime_dir, **kwargs)
        runner._delegated_controllers = lambda: frozenset({"cpu", "memory", "pids"})
        return runner

    def sandbox(self, **kwargs):
        kwargs.setdefault("scope_runner", self.scope_runner())
        return BubblewrapSandbox(**kwargs)

    # -- ownership ----------------------------------------------------------

    def test_command_runner_owns_the_active_production_contract(self):
        runner = CommandRunner()
        self.assertEqual(runner.cgroup_limits, DEFAULT_CGROUP_LIMITS)
        self.assertTrue(runner.cgroup_limits.active)

    def test_sandbox_without_an_explicit_contract_applies_none(self):
        """The sandbox is a mechanism; the contract belongs to CommandRunner.

        This is what keeps preflight and direct callers independent of a
        systemd user bus.
        """
        sandbox = self.sandbox()
        with patch.object(harness.sandbox.subprocess, "Popen") as popen:
            popen.return_value.communicate.return_value = ("", "")
            popen.return_value.returncode = 0
            sandbox.run(self.workspace, ["/bin/true"])
        self.assertEqual(popen.call_args.args[0][0], "bwrap")
        self.assertEqual(popen.call_args.kwargs["env"], {})

    def test_the_runner_path_is_scoped_with_the_production_values(self):
        sandbox = self.sandbox()
        runner = CommandRunner(sandbox=sandbox)
        with patch.object(harness.sandbox.subprocess, "Popen") as popen, \
             patch.object(sandbox, "ensure_available",
                          return_value=ToolResult("ok", "available")):
            popen.return_value.communicate.return_value = ("", "")
            popen.return_value.returncode = 0
            runner.run_sandboxed(self.workspace, ["/bin/true"])
        argv = popen.call_args.args[0]
        self.assertEqual(argv[0], "/usr/bin/systemd-run")
        for prop in ("MemoryMax=2147483648", "MemorySwapMax=0",
                     "TasksMax=256", "CPUQuota=800%"):
            self.assertIn(prop, argv)

    def test_command_runner_passes_its_cgroup_limits_to_the_sandbox(self):
        sandbox = MagicMock()
        sandbox.ensure_available.return_value = ToolResult("ok", "available")
        sandbox.run.return_value = ToolResult("ok", "done")
        runner = CommandRunner(sandbox=sandbox, cgroup_limits=self.limits)
        runner.run_sandboxed(self.workspace, ["/bin/true"])
        self.assertEqual(sandbox.run.call_args.kwargs["cgroup_limits"], self.limits)

    def test_execution_profile_is_untouched_by_cgroup_limits(self):
        profile = ExecutionProfile.from_capabilities({Capability.FILESYSTEM_READ})
        self.assertFalse(hasattr(profile, "cgroup_limits"))
        self.assertFalse(hasattr(profile, "memory_max_bytes"))

    # -- CLOSED route -------------------------------------------------------

    def test_closed_route_is_wrapped_when_limits_are_active(self):
        sandbox = self.sandbox()
        with patch.object(harness.sandbox.subprocess, "Popen") as popen:
            popen.return_value.communicate.return_value = ("", "")
            popen.return_value.returncode = 0
            sandbox.run(self.workspace, ["/bin/true"], cgroup_limits=self.limits)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[0], "/usr/bin/systemd-run")
        self.assertIn("--scope", argv)
        self.assertIn("--collect", argv)
        separator = argv.index("--")
        self.assertEqual(argv[separator + 1], "bwrap")

    def test_closed_route_is_not_wrapped_when_limits_are_inactive(self):
        sandbox = self.sandbox()
        with patch.object(harness.sandbox.subprocess, "Popen") as popen:
            popen.return_value.communicate.return_value = ("", "")
            popen.return_value.returncode = 0
            sandbox.run(self.workspace, ["/bin/true"], cgroup_limits=CgroupLimits())
        self.assertEqual(popen.call_args.args[0][0], "bwrap")

    def test_wrapper_order_is_systemd_run_then_bwrap_then_prlimit(self):
        sandbox = self.sandbox()
        with patch.object(harness.sandbox.subprocess, "Popen") as popen:
            popen.return_value.communicate.return_value = ("", "")
            popen.return_value.returncode = 0
            sandbox.run(self.workspace, ["/bin/true"], cgroup_limits=self.limits,
                        resource_limits=DEFAULT_RESOURCE_LIMITS)
        argv = popen.call_args.args[0]
        self.assertLess(argv.index("/usr/bin/systemd-run"), argv.index("bwrap"))
        self.assertLess(argv.index("bwrap"), argv.index("/usr/bin/prlimit"))
        # prlimit must stay inside bubblewrap, never wrap systemd-run.
        self.assertLess(argv.index("--die-with-parent"), argv.index("/usr/bin/prlimit"))

    def test_supervisor_env_is_used_for_systemd_run_but_never_for_bwrap(self):
        sandbox = self.sandbox()
        with patch.object(harness.sandbox.subprocess, "Popen") as popen:
            popen.return_value.communicate.return_value = ("", "")
            popen.return_value.returncode = 0
            sandbox.run(self.workspace, ["/bin/true"], cgroup_limits=self.limits)
        self.assertEqual(popen.call_args.kwargs["env"],
                         {"XDG_RUNTIME_DIR": str(self.runtime_dir)})
        # --clearenv remains the boundary for the sandboxed command itself.
        self.assertIn("--clearenv", popen.call_args.args[0])

        with patch.object(harness.sandbox.subprocess, "Popen") as popen:
            popen.return_value.communicate.return_value = ("", "")
            popen.return_value.returncode = 0
            sandbox.run(self.workspace, ["/bin/true"], cgroup_limits=CgroupLimits())
        self.assertEqual(popen.call_args.kwargs["env"], {})

    # -- fail closed --------------------------------------------------------

    def test_active_limits_fail_closed_when_systemd_run_is_absent(self):
        sandbox = self.sandbox(
            scope_runner=self.scope_runner(systemd_run_binary="/definitely/missing"))
        with patch.object(harness.sandbox.subprocess, "Popen") as popen, \
             patch.object(harness.sandbox.subprocess, "run") as run:
            result = sandbox.run(self.workspace, ["/bin/true"], cgroup_limits=self.limits)
        self.assertEqual(result.status, "failed")
        self.assertIn("resource control", result.summary.lower())
        popen.assert_not_called()
        run.assert_not_called()

    def test_active_limits_fail_closed_when_user_bus_is_absent(self):
        (self.runtime_dir / "bus").unlink()
        sandbox = self.sandbox()
        with patch.object(harness.sandbox.subprocess, "Popen") as popen:
            result = sandbox.run(self.workspace, ["/bin/true"], cgroup_limits=self.limits)
        self.assertEqual(result.status, "failed")
        popen.assert_not_called()

    def test_active_limits_fail_closed_when_a_controller_is_missing(self):
        runner = SystemdScopeRunner(runtime_dir=self.runtime_dir)
        runner._delegated_controllers = lambda: frozenset({"memory"})
        sandbox = self.sandbox(scope_runner=runner)
        with patch.object(harness.sandbox.subprocess, "Popen") as popen:
            result = sandbox.run(self.workspace, ["/bin/true"], cgroup_limits=self.limits)
        self.assertEqual(result.status, "failed")
        popen.assert_not_called()

    def test_there_is_no_unlimited_fallback(self):
        """A failed resource-control setup must never degrade to plain bwrap."""
        sandbox = self.sandbox(
            scope_runner=self.scope_runner(systemd_run_binary="/definitely/missing"))
        with patch.object(harness.sandbox.subprocess, "Popen") as popen, \
             patch.object(harness.sandbox.subprocess, "run") as run:
            result = sandbox.run(self.workspace, ["/bin/true"], cgroup_limits=self.limits)
        self.assertFalse(result.ok)
        popen.assert_not_called()
        run.assert_not_called()
        self.assertNotIn("prlimit", result.summary)

    # -- timeout ------------------------------------------------------------

    def test_timeout_invokes_scope_tree_cleanup_for_the_unit_it_created(self):
        runner = self.scope_runner()
        sandbox = self.sandbox(scope_runner=runner, timeout_seconds=1)
        with patch.object(harness.sandbox.subprocess, "Popen") as popen, \
             patch.object(runner, "terminate", return_value=True) as terminate:
            popen.return_value.communicate.side_effect = [
                subprocess.TimeoutExpired("bwrap", 1), ("", ""),
            ]
            popen.return_value.returncode = -9
            result = sandbox.run(self.workspace, ["/bin/sleep", "30"],
                                 cgroup_limits=self.limits)
        self.assertEqual(result.status, "timeout")
        terminate.assert_called_once()
        unit = terminate.call_args.args[0]
        self.assertRegex(unit, r"^spear-tool-[0-9a-f]{32}\.scope$")
        # The very unit that was spawned, not a glob or a discovered one.
        self.assertIn(f"--unit={unit}", popen.call_args.args[0])

    def test_timeout_without_limits_does_not_touch_systemd(self):
        runner = self.scope_runner()
        sandbox = self.sandbox(scope_runner=runner, timeout_seconds=1)
        with patch.object(harness.sandbox.subprocess, "Popen") as popen, \
             patch.object(runner, "terminate") as terminate:
            popen.return_value.communicate.side_effect = [
                subprocess.TimeoutExpired("bwrap", 1), ("", ""),
            ]
            popen.return_value.returncode = -9
            result = sandbox.run(self.workspace, ["/bin/sleep", "30"],
                                 cgroup_limits=CgroupLimits())
        self.assertEqual(result.status, "timeout")
        terminate.assert_not_called()

    # -- NETWORK route ------------------------------------------------------

    def test_network_route_wraps_bwrap_keeps_slirp_out_and_preserves_pass_fds(self):
        sandbox = self.sandbox()
        profile = ExecutionProfile.from_capabilities({Capability.NETWORK})
        spawned = []
        real_popen = subprocess.Popen

        def tracked(argv, *args, **kwargs):
            spawned.append((list(argv), kwargs.get("pass_fds"), kwargs.get("env")))
            return real_popen(argv, *args, **kwargs)

        with patch.object(harness.sandbox.subprocess, "Popen", side_effect=tracked):
            sandbox.run(self.workspace, ["/bin/true"], profile=profile,
                        backend=NetworkBackend.SLIRP4NETNS, cgroup_limits=self.limits)

        # The helper's option probe also goes through Popen; select by role
        # rather than by spawn order.
        scoped = [s for s in spawned if s[0][0] == "/usr/bin/systemd-run"]
        helpers = [s for s in spawned if "slirp4netns" in s[0][0] and "--help" not in s[0]]
        self.assertEqual(len(scoped), 1)
        bwrap_argv, bwrap_fds, bwrap_env = scoped[0]
        self.assertIn("bwrap", bwrap_argv)
        self.assertEqual(bwrap_env, {"XDG_RUNTIME_DIR": str(self.runtime_dir)})
        # The bwrap FD protocol is untouched by the wrapper.
        self.assertIsNotNone(bwrap_fds)
        self.assertEqual(len(bwrap_fds), 3)
        for flag in ("--info-fd", "--block-fd", "--sync-fd"):
            self.assertIn(flag, bwrap_argv)
            self.assertIn(int(bwrap_argv[bwrap_argv.index(flag) + 1]), bwrap_fds)

        if helpers:
            slirp_argv, slirp_fds, slirp_env = helpers[0]
            self.assertNotIn("systemd-run", slirp_argv[0])
            self.assertIn("slirp4netns", slirp_argv[0])
            self.assertEqual(slirp_env, {})

    def test_network_setup_failure_never_releases_the_sync_fd(self):
        """Scope creation failure must leave the COMMAND permanently blocked."""
        sandbox = self.sandbox()
        profile = ExecutionProfile.from_capabilities({Capability.NETWORK})
        marker = self.workspace_root / "command-ran"
        writes = []
        real_write = os.write

        def tracked_write(fd, data):
            writes.append((fd, data))
            return real_write(fd, data)

        # A systemd-run that always refuses: bwrap therefore never starts.
        runner = self.scope_runner(systemd_run_binary="/bin/false")
        sandbox = self.sandbox(scope_runner=runner)
        with patch.object(harness.sandbox.os, "write", side_effect=tracked_write):
            result = sandbox.run(
                self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
                profile=profile, backend=NetworkBackend.SLIRP4NETNS,
                cgroup_limits=self.limits)
        self.assertFalse(result.ok)
        self.assertFalse(marker.exists())
        self.assertEqual(writes, [], "no release write may follow a setup failure")

    def test_network_availability_failure_never_spawns_anything(self):
        (self.runtime_dir / "bus").unlink()
        sandbox = self.sandbox()
        profile = ExecutionProfile.from_capabilities({Capability.NETWORK})
        with patch.object(harness.sandbox.subprocess, "Popen") as popen:
            result = sandbox.run(self.workspace, ["/bin/true"], profile=profile,
                                 backend=NetworkBackend.SLIRP4NETNS,
                                 cgroup_limits=self.limits)
        self.assertFalse(result.ok)
        popen.assert_not_called()


@unittest.skipUnless(os.environ.get("SPEAR_TEST_CGROUP") == "1",
                     "set SPEAR_TEST_CGROUP=1 for opt-in real cgroup integration")
class OptInRealCgroupTests(unittest.TestCase):
    """Real transient scopes.  Requires a systemd user bus with delegation."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.workspace = Workspace.from_path(self.workspace_root)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.assert_no_residual_units)

    def assert_no_residual_units(self):
        listed = subprocess.run(
            ["/usr/bin/systemctl", "--user", "list-units", "--all", "--no-legend",
             "spear-tool-*"], capture_output=True, text=True, timeout=15)
        self.assertEqual(listed.stdout.strip(), "",
                         "residual spear-tool-*.scope unit(s) left behind")

    def sandbox(self, **kwargs):
        return BubblewrapSandbox(**kwargs)

    def observed_limits(self, limits):
        """Read the live cgroup files of the sandboxed process itself."""
        script = (
            "import pathlib\n"
            "rel = pathlib.Path('/proc/self/cgroup').read_text().strip().split(':')[-1]\n"
            "print(rel)\n"
        )
        result = self.sandbox().run(self.workspace, ["/usr/bin/python3", "-c", script],
                                    cgroup_limits=limits)
        self.assertTrue(result.ok, result.to_legacy_text())
        base = Path("/sys/fs/cgroup") / result.stdout.strip().lstrip("/")
        return base, result

    def test_A_to_D_cgroup_properties_are_actually_applied(self):
        limits = CgroupLimits(memory_max_bytes=134217728, memory_swap_max_bytes=0,
                              tasks_max=48, cpu_quota_percent=50)
        observed = {}
        script = (
            "import pathlib\n"
            "rel = pathlib.Path('/proc/self/cgroup').read_text().strip().split(':')[-1]\n"
            "print(rel)\n"
        )
        # The scope is gone once the command exits, so the values are read by a
        # helper that samples the live cgroup from the host side.
        sandbox = self.sandbox()
        unit_holder = {}
        real_popen = subprocess.Popen

        def sampling_popen(argv, *args, **kwargs):
            proc = real_popen(argv, *args, **kwargs)
            for flag in argv:
                if flag.startswith("--unit="):
                    unit_holder["unit"] = flag.split("=", 1)[1]
            base = Path("/sys/fs/cgroup/user.slice", f"user-{os.getuid()}.slice",
                        f"user@{os.getuid()}.service/app.slice",
                        unit_holder.get("unit", ""))
            for _ in range(300):
                try:
                    observed["memory.max"] = (base / "memory.max").read_text().strip()
                    observed["memory.swap.max"] = (base / "memory.swap.max").read_text().strip()
                    observed["pids.max"] = (base / "pids.max").read_text().strip()
                    observed["cpu.max"] = (base / "cpu.max").read_text().strip()
                    break
                except OSError:
                    time.sleep(0.02)
            return proc

        with patch.object(harness.sandbox.subprocess, "Popen", side_effect=sampling_popen):
            result = sandbox.run(self.workspace, ["/bin/sleep", "2"], cgroup_limits=limits)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertEqual(observed.get("memory.max"), "134217728")       # A
        self.assertEqual(observed.get("memory.swap.max"), "0")          # B
        self.assertEqual(observed.get("pids.max"), "48")                # C
        self.assertEqual(observed.get("cpu.max"), "50000 100000")       # D

    def test_E_memory_limit_kills_the_command_without_swap_escape(self):
        limits = CgroupLimits(memory_max_bytes=67108864, memory_swap_max_bytes=0)
        alloc = ("b = []\n"
                 "for _ in range(512): b.append(bytearray(1024 * 1024))\n"
                 "print('NO KILL')\n")
        result = self.sandbox(timeout_seconds=120).run(
            self.workspace, ["/usr/bin/python3", "-c", alloc], cgroup_limits=limits)
        self.assertFalse(result.ok)
        self.assertNotIn("NO KILL", result.stdout)

    def test_F_tasks_max_is_enforced_and_the_supervisor_survives(self):
        limits = CgroupLimits(tasks_max=12)
        script = "i=0; while [ $i -lt 40 ]; do /bin/sleep 5 & i=$((i+1)); done; echo done"
        result = self.sandbox(timeout_seconds=60).run(
            self.workspace, ["/bin/sh", "-c", script], cgroup_limits=limits)
        self.assertFalse(result.ok)
        self.assertIn("fork", result.stderr.lower())
        self.assertTrue(Path(f"/proc/{os.getpid()}").exists())

    def test_G_timeout_cleans_the_whole_scope_tree(self):
        limits = CgroupLimits(tasks_max=64)
        sandbox = self.sandbox(timeout_seconds=1)
        result = sandbox.run(self.workspace,
                             ["/bin/sh", "-c", "/bin/sleep 30 & /bin/sleep 30"],
                             cgroup_limits=limits)
        self.assertEqual(result.status, "timeout")
        time.sleep(0.5)
        survivors = subprocess.run(["/usr/bin/pgrep", "-c", "-f", "^/bin/sleep 30$"],
                                   capture_output=True, text=True)
        self.assertIn(survivors.stdout.strip(), ("", "0"),
                      "sandbox descendants survived the scope cleanup")

    @staticmethod
    def _comm(pid):
        try:
            return Path(f"/proc/{pid}/comm").read_text().strip()
        except OSError:
            return ""

    @staticmethod
    def _own_children():
        """Direct children of this process, without scanning all of /proc."""
        children = []
        for task in Path("/proc/self/task").iterdir():
            try:
                children.extend((task / "children").read_text().split())
            except OSError:
                continue
        return children

    def _watch_scope(self, unit, observed, stop):
        """Sample the live scope and the slirp helper from a side thread.

        Nothing here runs on the supervisor's thread, and the sampling is kept
        cheap on purpose.  bwrap moves its child into a second, nested user
        namespace a few milliseconds after writing its info-fd JSON, and once
        that happens slirp4netns can no longer setns() into the sandbox netns.
        Any contention inside that window breaks the network path outright
        (measured: a deliberate 20 ms delay fails 40 runs out of 40).
        """
        base = Path("/sys/fs/cgroup/user.slice", f"user-{os.getuid()}.slice",
                    f"user@{os.getuid()}.service/app.slice", unit)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and not stop.is_set():
            try:
                members = (base / "cgroup.procs").read_text().split()
            except OSError:
                members = []
            if members:
                sample = [(pid, self._comm(pid)) for pid in members]
                # Keep the most complete sample: early on, only bwrap and its
                # namespace child are in the scope; the command joins after the
                # release write.
                if len(sample) > len(observed.get("scope_members", ())):
                    observed["scope_members"] = sample
            for pid in self._own_children():
                if self._comm(pid) != "slirp4netns":
                    continue
                try:
                    observed["slirp_cgroup"] = (
                        Path(f"/proc/{pid}/cgroup").read_text().strip().split(":")[-1]
                    )
                except OSError:
                    continue
            seen_command = any(
                comm not in ("bwrap", "") for _, comm in observed.get("scope_members", ())
            )
            if seen_command and "slirp_cgroup" in observed:
                return
            stop.wait(0.01)

    def test_H_network_membership_bwrap_in_scope_slirp_and_supervisor_out(self):
        limits = CgroupLimits(memory_max_bytes=536870912, tasks_max=64)
        profile = ExecutionProfile.from_capabilities({Capability.NETWORK})
        real_unit_name = SystemdScopeRunner.unit_name
        observed = {}
        result = None
        stop = threading.Event()
        self.addCleanup(stop.set)

        def naming():
            unit = real_unit_name()
            # Started before the spawn, i.e. outside the critical window.
            threading.Thread(target=self._watch_scope, args=(unit, observed, stop),
                             daemon=True).start()
            return unit

        # The pre-existing slirp readiness race (see _watch_scope) is not this
        # test's subject; retry a bounded number of times so a rare miss is not
        # reported as a cgroup-membership failure.
        for _ in range(5):
            observed.clear()
            stop.clear()
            with patch.object(SystemdScopeRunner, "unit_name", staticmethod(naming)):
                result = self.sandbox().run(self.workspace, ["/bin/sleep", "1"],
                                            profile=profile,
                                            backend=NetworkBackend.SLIRP4NETNS,
                                            cgroup_limits=limits)
            stop.set()
            if result.ok:
                break
        self.assertTrue(
            result.ok,
            "network path failed in all attempts (pre-existing slirp setns race?): "
            + result.to_legacy_text())

        members = observed.get("scope_members") or []
        self.assertTrue(members, "the transient scope was never observed alive")
        comms = [comm for _, comm in members]
        pids = [pid for pid, _ in members]
        # bwrap, its namespace child and the sandboxed command's tree: the
        # contract is exactly bwrap + COMMAND + descendants.
        self.assertIn("bwrap", comms, members)
        self.assertIn("sleep", comms, members)
        self.assertNotIn("slirp4netns", comms, members)
        self.assertNotIn(str(os.getpid()), pids, "the supervisor joined the scope")
        self.assertIn("slirp_cgroup", observed, "slirp4netns was never observed")
        self.assertNotIn("spear-tool-", observed["slirp_cgroup"])
        supervisor = Path("/proc/self/cgroup").read_text().strip().split(":")[-1]
        self.assertNotIn("spear-tool-", supervisor)

    def test_I_exit_seven_and_oom_leave_no_residual_unit(self):
        seven = self.sandbox().run(self.workspace, ["/bin/sh", "-c", "exit 7"],
                                   cgroup_limits=CgroupLimits(tasks_max=32))
        self.assertEqual(seven.exit_code, 7)
        alloc = "b = []\nfor _ in range(512): b.append(bytearray(1024 * 1024))\n"
        oom = self.sandbox(timeout_seconds=120).run(
            self.workspace, ["/usr/bin/python3", "-c", alloc],
            cgroup_limits=CgroupLimits(memory_max_bytes=67108864, memory_swap_max_bytes=0))
        self.assertFalse(oom.ok)
        # assert_no_residual_units runs in cleanup.


class DelegatedResourceControlTests(unittest.TestCase):
    """Delegation must reach BOTH decisions, or it reaches neither usefully.

    Allowing the run while still wrapping it in a systemd-run that does not
    exist reported the sandbox unavailable for every command in the session --
    exactly the failure the delegation was added to prevent.
    """

    def setUp(self):
        self.runner = SystemdScopeRunner()
        self.limits = DEFAULT_CGROUP_LIMITS

    def test_delegation_allows_and_unwraps(self):
        with patch.dict(os.environ, {"SPEAR_RESOURCE_CONTROL": "delegated"}):
            self.assertTrue(self.runner.delegated())
            self.assertIsNone(self.runner.availability_for(self.limits))
            self.assertEqual(["/bin/true"],
                             self.runner.wrap(["/bin/true"], self.limits, unit="u"))

    def test_without_it_the_scope_is_still_required(self):
        env = {k: v for k, v in os.environ.items()
               if k != "SPEAR_RESOURCE_CONTROL"}
        with patch.dict(os.environ, env, clear=True):
            wrapped = self.runner.wrap(["/bin/true"], self.limits, unit="u")
            self.assertIn("systemd-run", wrapped[0])
            self.assertIn("--scope", wrapped)

    def test_inactive_limits_are_never_wrapped(self):
        empty = CgroupLimits()
        self.assertEqual(["/bin/true"],
                         self.runner.wrap(["/bin/true"], empty, unit="u"))


if __name__ == "__main__":
    unittest.main()
