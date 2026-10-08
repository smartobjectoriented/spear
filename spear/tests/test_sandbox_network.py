"""Network access from the sandbox: slirp4netns and the pinned namespaces."""

import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harness.sandbox

from harness.resource_control import ExecutionProfile
from harness.tool_primitives import Capability, ExecutionMode, NetworkBackend, ToolResult
from harness.sandbox import (
    CLONE_NEWNET,
    CLONE_NEWUSER,
    NS_GET_NSTYPE,
    NS_GET_USERNS,
    BubblewrapSandbox,
)
from harness.workspace import Workspace


class Slirp4netnsNetworkTests(unittest.TestCase):
    """Deterministic NETWORK prototype tests; Internet remains opt-in."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.workspace = Workspace.from_path(self.workspace_root)
        self.outside_root = Path(self.temp.name) / "outside"
        self.outside_root.mkdir()
        self.profile = ExecutionProfile.from_capabilities({Capability.NETWORK})

    def tearDown(self):
        self.temp.cleanup()

    def sandbox(self, **kwargs):
        kwargs.setdefault("network_ready_timeout_seconds", 2)
        return BubblewrapSandbox(**kwargs)

    def assert_no_test_process(self, marker):
        processes = subprocess.check_output(["ps", "-eo", "args="], text=True)
        self.assertNotIn(str(marker), processes)
        self.assertNotIn("/tmp/spear-slirp-", processes)

    def run_with_tracked_pidfd(self, argv, *, sandbox_kwargs=None):
        return self.sandbox(timeout_seconds=1, **(sandbox_kwargs or {})).run(
            self.workspace, argv, profile=self.profile,
            backend=NetworkBackend.SLIRP4NETNS,
        )

    def test_closed_backend_refuses_network_profile_and_slirp_keeps_unshare_net(self):
        sandbox = self.sandbox()
        with self.assertRaisesRegex(ValueError, "requires slirp4netns"):
            sandbox.build_argv(self.workspace, ["/bin/true"], profile=self.profile)
        argv = sandbox.build_argv(
            self.workspace, ["/bin/true"], profile=self.profile,
            backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertIn("--unshare-net", argv)
        self.assertNotIn("--share-net", argv)

    def test_network_namespace_has_slirp_topology_and_synthetic_dns(self):
        profile = ExecutionProfile.from_capabilities({
            Capability.NETWORK, Capability.SHELL_COMPLEX,
        })
        result = self.sandbox().run(
            self.workspace,
            ["/bin/sh", "-lc", "/usr/sbin/ip -brief addr; /usr/sbin/ip route; cat /etc/resolv.conf"],
            profile=profile,
            backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertIn("tap0", result.stdout)
        self.assertIn("10.0.2.100/24", result.stdout)
        self.assertIn("default via 10.0.2.2", result.stdout)
        self.assertIn("nameserver 10.0.2.3", result.stdout)

    def test_network_lifecycle_uses_and_closes_pidfd_on_success_and_timeout(self):
        success = self.run_with_tracked_pidfd(["/bin/true"])
        timed_out = self.run_with_tracked_pidfd(["/bin/sh", "-lc", "sleep 10"])
        self.assertTrue(success.ok, success.to_legacy_text())
        self.assertEqual(timed_out.status, "timeout")

    def test_network_uses_sync_fd_and_releases_once_after_ready(self):
        pipes = []
        writes = []
        bwrap_argvs = []
        real_pipe = os.pipe
        real_write = os.write
        real_popen = subprocess.Popen

        def tracked_pipe():
            pair = real_pipe()
            pipes.append(pair)
            return pair

        def tracked_write(fd, data):
            writes.append((fd, data))
            return real_write(fd, data)

        def tracked_popen(argv, *args, **kwargs):
            if argv and Path(argv[0]).name == "bwrap":
                bwrap_argvs.append(list(argv))
            return real_popen(argv, *args, **kwargs)

        with patch.object(harness.sandbox.os, "pipe", side_effect=tracked_pipe), \
             patch.object(harness.sandbox.os, "write", side_effect=tracked_write), \
             patch.object(harness.sandbox.subprocess, "Popen", side_effect=tracked_popen):
            result = self.sandbox().run(
                self.workspace, ["/bin/true"], profile=self.profile,
                backend=NetworkBackend.SLIRP4NETNS,
            )
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertGreaterEqual(len(pipes), 2)
        block_read, release_write = pipes[1]
        argv = bwrap_argvs[0]
        self.assertEqual(argv[argv.index("--block-fd") + 1], str(block_read))
        self.assertEqual(argv[argv.index("--sync-fd") + 1], str(release_write))
        self.assertEqual([(fd, data) for fd, data in writes if fd == release_write],
                         [(release_write, b"x")])

    def test_setup_failures_never_write_sync_release(self):
        marker = self.workspace_root / "unexpected-release"
        cases = (
            ("helper", {"slirp_binary": "/bin/false"}, None),
            ("ready-timeout", {"slirp_binary": "/bin/false"}, None),
            ("pidfd-open", {}, patch.object(harness.sandbox.os, "pidfd_open", side_effect=OSError("no"))),
            ("json", {}, patch.object(harness.sandbox.json, "loads", side_effect=RuntimeError("no"))),
        )
        for name, sandbox_kwargs, extra_patch in cases:
            with self.subTest(name=name):
                writes = []
                real_write = os.write
                def tracked_write(fd, data):
                    writes.append((fd, data))
                    return real_write(fd, data)
                with patch.object(harness.sandbox.os, "write", side_effect=tracked_write):
                    if extra_patch is None:
                        result = self.sandbox(**sandbox_kwargs).run(
                            self.workspace,
                            ["/bin/sh", "-lc", "touch /workspace/unexpected-release"],
                            profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
                        )
                    else:
                        with extra_patch:
                            result = self.sandbox(**sandbox_kwargs).run(
                                self.workspace,
                                ["/bin/sh", "-lc", "touch /workspace/unexpected-release"],
                                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
                            )
                self.assertFalse(result.ok)
                self.assertFalse(any(data == b"x" for _, data in writes))
                self.assertFalse(marker.exists())

    def test_pidfd_is_closed_after_helper_failure_and_python_exception(self):
        failed = self.run_with_tracked_pidfd(
            ["/bin/true"], sandbox_kwargs={"slirp_binary": "/bin/false"},
        )
        self.assertFalse(failed.ok)

        opened = []
        closed = []
        real_pidfd_open = os.pidfd_open
        real_close = os.close
        real_popen = subprocess.Popen

        def tracked_open(pid, flags=0):
            fd = real_pidfd_open(pid, flags)
            opened.append(fd)
            return fd

        def tracked_close(fd):
            closed.append(fd)
            return real_close(fd)

        calls = 0
        def fail_before_slirp(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("test orchestration exception")
            return real_popen(*args, **kwargs)

        # The helper's option probe is a preflight concern and spawns its own
        # process; prime it so the counter below tracks the orchestration only.
        sandbox = self.sandbox()
        self.assertIsNone(sandbox._slirp_pinned_namespace_status())

        with patch.object(harness.sandbox.os, "pidfd_open", side_effect=tracked_open), \
             patch.object(harness.sandbox.os, "close", side_effect=tracked_close), \
             patch.object(harness.sandbox.subprocess, "Popen", side_effect=fail_before_slirp):
            result = sandbox.run(
                self.workspace, ["/bin/true"], profile=self.profile,
                backend=NetworkBackend.SLIRP4NETNS,
            )
        self.assertFalse(result.ok)
        self.assertEqual(len(opened), 1)
        self.assertIn(opened[0], closed)
        self.assert_no_test_process(self.workspace_root)

    def test_pidfd_open_failure_fails_closed_before_command_release(self):
        marker = self.workspace_root / "pidfd-open-marker"
        with patch.object(harness.sandbox.os, "pidfd_open", side_effect=OSError("blocked")):
            result = self.sandbox().run(
                self.workspace, ["/bin/sh", "-lc", "touch /workspace/pidfd-open-marker"],
                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
            )
        self.assertFalse(result.ok)
        self.assertFalse(marker.exists())

    def test_missing_pidfd_support_fails_closed_before_command_release(self):
        marker = self.workspace_root / "pidfd-support-marker"
        with patch.object(harness.sandbox.os, "pidfd_open", None), \
             patch.object(harness.sandbox.signal, "pidfd_send_signal", None):
            result = self.sandbox().run(
                self.workspace, ["/bin/sh", "-lc", "touch /workspace/pidfd-support-marker"],
                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
            )
        self.assertFalse(result.ok)
        self.assertFalse(marker.exists())

    def test_network_preflight_requires_pidfd_support(self):
        with patch.object(harness.sandbox.os, "pidfd_open", None):
            result = self.sandbox().preflight_network(self.workspace)
        self.assertFalse(result.ok)
        self.assertIn("pidfd", result.summary)

    def test_pidfd_stop_handles_immediate_transient_and_already_dead_results(self):
        sandbox = self.sandbox()
        with patch.object(harness.sandbox.signal, "pidfd_send_signal") as send:
            self.assertTrue(sandbox._stop_namespace_child(123))
        send.assert_called_once_with(123, signal.SIGKILL)

        with patch.object(
            harness.sandbox.signal, "pidfd_send_signal", side_effect=[OSError("once"), None]
        ) as send:
            self.assertTrue(sandbox._stop_namespace_child(123))
        self.assertEqual(send.call_count, 2)

        with patch.object(
            harness.sandbox.signal, "pidfd_send_signal", side_effect=ProcessLookupError
        ):
            self.assertTrue(sandbox._stop_namespace_child(123))

    def test_wait_pidfd_exit_observes_exit_and_times_out_without_pid_lookup(self):
        process = subprocess.Popen(["/bin/sleep", "30"])
        pidfd = os.pidfd_open(process.pid)
        try:
            self.assertFalse(BubblewrapSandbox._wait_pidfd_exit(pidfd, 0.01))
            signal.pidfd_send_signal(pidfd, signal.SIGKILL)
            self.assertTrue(BubblewrapSandbox._wait_pidfd_exit(pidfd, 1))
        finally:
            process.wait()
            os.close(pidfd)

    def test_permanent_pidfd_signal_failure_closes_parent_writer_and_poisons_backend(self):
        marker = self.workspace_root / "pidfd-signal-marker"
        stalled_helper = Path(self.temp.name) / "stalled-slirp-pidfd"
        # Stands in for a capable helper: it must advertise pinned-namespace
        # attachment, otherwise the runtime fails closed before this scenario.
        stalled_helper.write_text(
            '#!/bin/sh\n'
            'case "$1" in --help) echo "--netns-type --userns-path"; exit 0;; esac\n'
            'exec /bin/sleep 30\n',
            encoding="utf-8",
        )
        stalled_helper.chmod(0o755)
        calls = []
        real_pidfd_open = os.pidfd_open
        real_pipe = os.pipe
        opened = []
        test_pidfds = []
        pipes = []
        safe_close_calls = []
        writes = []
        wait_calls = []
        supervisor_stops = []

        def tracked_open(pid, flags=0):
            fd = real_pidfd_open(pid, flags)
            opened.append(fd)
            test_pidfds.append(os.dup(fd))
            return fd

        def permanently_failing_signal(pidfd, signum, siginfo=None, flags=0):
            calls.append(("signal", pidfd))
            raise OSError("simulated permanent pidfd failure")

        def tracked_pipe():
            pair = real_pipe()
            pipes.append(pair)
            return pair

        original_wait = BubblewrapSandbox._wait_pidfd_exit
        original_stop = BubblewrapSandbox._stop_process
        original_safe_close = BubblewrapSandbox._safe_close

        def tracked_wait(pidfd, timeout):
            wait_calls.append((pidfd, timeout))
            return original_wait(pidfd, timeout)

        def tracked_stop(process, **kwargs):
            supervisor_stops.append(process)
            return original_stop(process, **kwargs)

        def tracked_safe_close(fd):
            safe_close_calls.append((opened[-1] if opened else None, fd))
            return original_safe_close(fd)

        real_write = os.write
        def tracked_write(fd, data):
            writes.append((fd, data))
            return real_write(fd, data)

        with patch.object(harness.sandbox.os, "pidfd_open", side_effect=tracked_open), \
             patch.object(harness.sandbox.signal, "pidfd_send_signal", side_effect=permanently_failing_signal), \
             patch.object(harness.sandbox.os, "pipe", side_effect=tracked_pipe), \
             patch.object(harness.sandbox.os, "write", side_effect=tracked_write), \
             patch.object(BubblewrapSandbox, "_safe_close", side_effect=tracked_safe_close), \
             patch.object(BubblewrapSandbox, "_wait_pidfd_exit", side_effect=tracked_wait), \
             patch.object(BubblewrapSandbox, "_stop_process", side_effect=tracked_stop):
            sandbox = self.sandbox(
                slirp_binary=str(stalled_helper), network_ready_timeout_seconds=0.1,
            )
            result = sandbox.run(
                self.workspace, ["/bin/sh", "-lc", "touch /workspace/pidfd-signal-marker"],
                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
            )
        self.assertFalse(result.ok)
        self.assertIn("could not verify child termination", result.summary)
        self.assertFalse(marker.exists())
        self.assertEqual(len(opened), 1)
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(calls and calls[0][0] == "signal")
        self.assertEqual(wait_calls[0][0], opened[0])
        self.assertTrue(supervisor_stops)
        self.assertGreaterEqual(len(pipes), 2)
        release_write = pipes[1][1]
        self.assertFalse(any(fd == release_write for fd, _ in writes))
        self.assertIn((opened[0], release_write), safe_close_calls)
        self.assertTrue(sandbox.network_containment_failed)
        retry = sandbox.run(
            self.workspace, ["/bin/true"], profile=self.profile,
            backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertFalse(retry.ok)
        self.assertIn("containment", retry.summary)
        closed = sandbox.run(self.workspace, ["/bin/true"])
        self.assertTrue(closed.ok, closed.to_legacy_text())

        # A test-only duplicate preserves identity after the runtime closes
        # its own pidfd; remove this explicitly identifiable blocked child.
        test_pidfd = test_pidfds[0]
        signal.pidfd_send_signal(test_pidfd, signal.SIGKILL)
        self.assertTrue(BubblewrapSandbox._wait_pidfd_exit(test_pidfd, 1))
        os.close(test_pidfd)
        self.assert_no_test_process(stalled_helper)

    def test_network_sandbox_cannot_reach_host_loopback_or_slirp_gateway(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            program = (
                "import socket; "
                f"a=socket.socket();a.settimeout(1);x=a.connect_ex(('127.0.0.1',{port}));a.close(); "
                f"b=socket.socket();b.settimeout(1);y=b.connect_ex(('10.0.2.2',{port}));b.close(); "
                "raise SystemExit(0 if x != 0 and y != 0 else 1)"
            )
            result = self.sandbox().run(
                self.workspace, ["/usr/bin/python3", "-c", program],
                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
            )
        finally:
            server.close()
        self.assertTrue(result.ok, result.to_legacy_text())

    def test_missing_or_failed_slirp_fails_closed_before_command_release(self):
        marker = self.workspace_root / "command-ran"
        missing = self.sandbox(slirp_binary="/definitely/missing/slirp4netns")
        result = missing.run(
            self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
            profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertFalse(result.ok)
        self.assertFalse(marker.exists())

        non_executable_helper = Path(self.temp.name) / "non-executable-slirp"
        non_executable_helper.write_text("not executable", encoding="utf-8")
        non_executable_helper.chmod(0o644)
        non_executable = self.sandbox(slirp_binary=str(non_executable_helper))
        result = non_executable.run(
            self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
            profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertFalse(result.ok)
        self.assertFalse(marker.exists())

    def test_bwrap_failure_and_slirp_ready_timeout_never_release_command(self):
        marker = self.workspace_root / "command-ran"
        # ``/bin/false`` passes the executable precheck but fails before it
        # emits Bubblewrap's namespace info, so slirp must never start.
        broken_bwrap = self.sandbox(binary="/bin/false")
        result = broken_bwrap.run(
            self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
            profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertFalse(result.ok)
        self.assertFalse(marker.exists())

        stalled_helper = Path(self.temp.name) / "stalled-slirp"
        # Stands in for a capable helper: it must advertise pinned-namespace
        # attachment, otherwise the runtime fails closed before this scenario.
        stalled_helper.write_text(
            '#!/bin/sh\n'
            'case "$1" in --help) echo "--netns-type --userns-path"; exit 0;; esac\n'
            'exec /bin/sleep 30\n',
            encoding="utf-8",
        )
        stalled_helper.chmod(0o755)
        stalled = self.sandbox(
            slirp_binary=str(stalled_helper), network_ready_timeout_seconds=0.1,
        )
        result = stalled.run(
            self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
            profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertFalse(result.ok)
        self.assertFalse(marker.exists())
        self.assert_no_test_process(stalled_helper)

    def test_python_orchestration_failure_cleans_bwrap_and_slirp(self):
        marker = self.workspace_root / "command-ran"
        with patch.object(harness.sandbox.json, "loads", side_effect=RuntimeError("test failure")):
            result = self.sandbox().run(
                self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
            )
        self.assertFalse(result.ok)
        self.assertFalse(marker.exists())
        self.assert_no_test_process(self.workspace_root)

        failed = self.sandbox(slirp_binary="/bin/false")
        result = failed.run(
            self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
            profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertFalse(result.ok)
        self.assertFalse(marker.exists())

    def test_network_lifecycle_cleans_helpers_on_exit_error_and_timeout(self):
        sandbox = self.sandbox(timeout_seconds=1)
        ok = sandbox.run(self.workspace, ["/bin/true"], profile=self.profile,
                         backend=NetworkBackend.SLIRP4NETNS)
        failed = sandbox.run(self.workspace, ["/bin/sh", "-lc", "exit 7"],
                             profile=self.profile, backend=NetworkBackend.SLIRP4NETNS)
        timed_out = sandbox.run(self.workspace, ["/bin/sh", "-lc", "sleep 10"],
                                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS)
        self.assertTrue(ok.ok, ok.to_legacy_text())
        self.assertFalse(failed.ok)
        self.assertEqual(failed.exit_code, 7)
        self.assertEqual(timed_out.status, "timeout")
        self.assert_no_test_process(self.workspace_root)

    @unittest.skipUnless(os.environ.get("SPEAR_TEST_NETWORK") == "1",
                         "set SPEAR_TEST_NETWORK=1 for opt-in outbound DNS integration")
    def test_opt_in_outbound_dns_and_https_tls(self):
        dns = self.sandbox().run(
            self.workspace, ["/usr/bin/getent", "hosts", "example.com"],
            profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertTrue(dns.ok, dns.to_legacy_text())
        https = self.sandbox().run(
            self.workspace,
            ["/usr/bin/curl", "--fail", "--silent", "--show-error", "https://example.com/"],
            profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertTrue(https.ok, https.to_legacy_text())

    def test_preflight_runs_a_real_mini_sandbox(self):
        sandbox = self.sandbox()
        result = sandbox.preflight(self.workspace)
        self.assertIsInstance(result, ToolResult)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertIn("sandbox", result.summary.lower())

    def test_workspace_is_visible_at_workspace_and_write_is_allowed(self):
        sandbox = self.sandbox()
        mount = sandbox.mount_root(self.workspace)
        result = sandbox.run(
            self.workspace,
            ["/bin/sh", "-lc", f"test \"$PWD\" = {mount} && touch writable.txt"]
        )
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertTrue((self.workspace_root / "writable.txt").exists())

    def test_write_outside_workspace_is_impossible(self):
        sandbox = self.sandbox()
        target = self.outside_root / "escaped.txt"
        result = sandbox.run(
            self.workspace, ["/bin/sh", "-lc", f"touch {target}"]
        )
        self.assertFalse(result.ok)
        self.assertFalse(target.exists())

    def test_symlink_to_host_outside_workspace_is_unusable(self):
        (self.workspace_root / "outside-link").symlink_to(self.outside_root,
                                                            target_is_directory=True)
        sandbox = self.sandbox()
        result = sandbox.run(
            self.workspace, ["/bin/sh", "-lc", "touch /workspace/outside-link/escaped.txt"]
        )
        self.assertFalse(result.ok)
        self.assertFalse((self.outside_root / "escaped.txt").exists())

    def test_home_and_sensitive_environment_are_not_inherited(self):
        host_home = self.outside_root / "host-home"
        host_home.mkdir()
        (host_home / ".ssh").mkdir()
        sandbox = self.sandbox()
        with patch.dict(os.environ, {"HOME": str(host_home), "TOKEN": "host-secret"}):
            result = sandbox.run(
                self.workspace,
                ["/bin/sh", "-lc", "test \"$HOME\" != '" + str(host_home)
                 + "' && test ! -e \"$HOME/.ssh\" && test -z \"${TOKEN:-}\""],
            )
        self.assertTrue(result.ok, result.to_legacy_text())

    def test_tmp_is_private(self):
        host_marker = Path(self.temp.name) / "host-tmp-marker"
        host_marker.write_text("host")
        sandbox = self.sandbox()
        result = sandbox.run(
            self.workspace,
            ["/bin/sh", "-lc", f"test ! -e {host_marker} && touch /tmp/sandbox-marker"],
        )
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertFalse((Path("/tmp") / "sandbox-marker").exists())

    def test_network_is_unavailable(self):
        sandbox = self.sandbox()
        argv = sandbox.build_argv(self.workspace, ["/bin/sh", "-lc", "true"])
        self.assertIn("--unshare-net", argv)
        result = sandbox.run(
            self.workspace,
            ["/bin/sh", "-lc", "if test -d /sys/class/net; then "
             "test \"$(find /sys/class/net -mindepth 1 -maxdepth 1 -printf '%f\\n' | sort)\" = lo; fi"],
        )
        self.assertTrue(result.ok, result.to_legacy_text())

    def test_auto_workspace_mutation_fails_closed_without_sandbox(self):
        sandbox = self.sandbox(binary="/definitely/missing/bwrap")
        from cli import session_workspace

        old_mode = session_workspace.EXECUTION_MODE
        old_sandbox = getattr(session_workspace.COMMAND_RUNNER, "sandbox", None)
        try:
            session_workspace.EXECUTION_MODE = ExecutionMode.AUTO
            session_workspace.COMMAND_RUNNER.sandbox = sandbox
            result = session_workspace.run_cmd("make test", need_confirm=False)
        finally:
            session_workspace.EXECUTION_MODE = old_mode
            session_workspace.COMMAND_RUNNER.sandbox = old_sandbox
        self.assertIn("sandbox", result.lower())
        self.assertIn("unavailable", result.lower())

    def test_auto_shell_complex_fails_closed_without_sandbox(self):
        sandbox = self.sandbox(binary="/definitely/missing/bwrap")
        from cli import session_workspace

        old_mode = session_workspace.EXECUTION_MODE
        old_sandbox = getattr(session_workspace.COMMAND_RUNNER, "sandbox", None)
        try:
            session_workspace.EXECUTION_MODE = ExecutionMode.AUTO
            session_workspace.COMMAND_RUNNER.sandbox = sandbox
            result = session_workspace.run_cmd("ls | head", need_confirm=False)
        finally:
            session_workspace.EXECUTION_MODE = old_mode
            session_workspace.COMMAND_RUNNER.sandbox = old_sandbox
        self.assertIn("sandbox", result.lower())
        self.assertIn("unavailable", result.lower())

    def test_ask_complex_command_keeps_explicit_confirmation_gate(self):
        sandbox = self.sandbox(binary="/definitely/missing/bwrap")
        from cli import session_workspace

        old_mode = session_workspace.EXECUTION_MODE
        old_sandbox = getattr(session_workspace.COMMAND_RUNNER, "sandbox", None)
        try:
            session_workspace.EXECUTION_MODE = ExecutionMode.ASK
            session_workspace.COMMAND_RUNNER.sandbox = sandbox
            with patch.object(session_workspace, "confirm", return_value=False) as confirm:
                result = session_workspace.run_cmd("ls | head", need_confirm=False)
        finally:
            session_workspace.EXECUTION_MODE = old_mode
            session_workspace.COMMAND_RUNNER.sandbox = old_sandbox
        confirm.assert_called_once()
        self.assertEqual(result, "CANCELLED")

    def test_ask_complex_command_fails_closed_after_approved_confirmation(self):
        sandbox = self.sandbox(binary="/definitely/missing/bwrap")
        from cli import session_workspace

        old_mode = session_workspace.EXECUTION_MODE
        old_sandbox = getattr(session_workspace.COMMAND_RUNNER, "sandbox", None)
        try:
            session_workspace.EXECUTION_MODE = ExecutionMode.ASK
            session_workspace.COMMAND_RUNNER.sandbox = sandbox
            with patch.object(session_workspace, "confirm", return_value=True) as confirm, \
                 patch.object(session_workspace.COMMAND_RUNNER, "run_simple") as simple, \
                 patch.object(session_workspace.COMMAND_RUNNER, "run_complex_approved") as complex_run:
                result = session_workspace.run_cmd("printf compatibility | cat", need_confirm=False)
        finally:
            session_workspace.EXECUTION_MODE = old_mode
            session_workspace.COMMAND_RUNNER.sandbox = old_sandbox
        confirm.assert_called_once()
        simple.assert_not_called()
        complex_run.assert_not_called()
        self.assertIn("sandbox unavailable", result)


class PinnedNamespaceAttachmentTests(unittest.TestCase):
    """Fast, always-on checks on argv construction and descriptor hygiene."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.workspace = Workspace.from_path(self.workspace_root)
        self.profile = ExecutionProfile.from_capabilities({Capability.NETWORK})
        self.addCleanup(self.temp.cleanup)

    def sandbox(self, **kwargs):
        kwargs.setdefault("network_ready_timeout_seconds", 2)
        return BubblewrapSandbox(**kwargs)

    def spawn_and_capture(self, argv, **run_kwargs):
        spawned = []
        real_popen = subprocess.Popen

        def tracked(command, *args, **kwargs):
            spawned.append((list(command), kwargs.get("pass_fds")))
            return real_popen(command, *args, **kwargs)

        with patch.object(harness.sandbox.subprocess, "Popen", side_effect=tracked):
            result = self.sandbox().run(self.workspace, argv, profile=self.profile,
                                        backend=NetworkBackend.SLIRP4NETNS, **run_kwargs)
        helper = [s for s in spawned
                  if "slirp4netns" in s[0][0] and "--help" not in s[0]]
        return result, helper[0] if helper else None

    def test_helper_receives_pinned_namespace_paths_not_a_pid(self):
        result, helper = self.spawn_and_capture(["/bin/true"])
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertIsNotNone(helper)
        helper_argv, pass_fds = helper
        self.assertIn("--netns-type=path", helper_argv)
        userns = [a for a in helper_argv if a.startswith("--userns-path=")]
        self.assertEqual(len(userns), 1)
        userns_fd = int(userns[0].rsplit("/", 1)[1])
        # Positional target is the pinned netns, never a bare PID.
        netns_arg = helper_argv[-2]
        self.assertTrue(netns_arg.startswith("/proc/self/fd/"), helper_argv)
        netns_fd = int(netns_arg.rsplit("/", 1)[1])
        # Both handles must be explicitly passed to the helper, and nothing else
        # beyond the ready/exit pipes.
        self.assertIn(userns_fd, pass_fds)
        self.assertIn(netns_fd, pass_fds)
        self.assertEqual(len(pass_fds), 4)
        # The protocol itself is unchanged.
        self.assertIn("--configure", helper_argv)
        self.assertIn("--disable-host-loopback", helper_argv)
        self.assertIn("--ready-fd", helper_argv)
        self.assertIn("--exit-fd", helper_argv)
        self.assertEqual(helper_argv[-1], "tap0")

    def test_command_never_inherits_the_namespace_handles(self):
        result = self.sandbox().run(
            self.workspace, ["/bin/ls", "-l", "/proc/self/fd"],
            profile=self.profile, backend=NetworkBackend.SLIRP4NETNS)
        self.assertTrue(result.ok, result.to_legacy_text())
        targets = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if "->" in parts:
                targets.append((parts[parts.index("->") - 1], parts[-1]))
        numbers = sorted(int(fd) for fd, _ in targets)
        # 0/1/2 plus the directory descriptor `ls` opens for itself.
        self.assertEqual(numbers, [0, 1, 2, 3], targets)
        rendered = " ".join(target for _, target in targets)
        self.assertNotIn("net:[", rendered)
        self.assertNotIn("user:[", rendered)

    def test_supervisor_leaks_no_descriptor_on_success_or_failure(self):
        def open_count():
            return len(os.listdir(f"/proc/{os.getpid()}/fd"))

        baseline = open_count()
        self.sandbox().run(self.workspace, ["/bin/true"], profile=self.profile,
                           backend=NetworkBackend.SLIRP4NETNS)
        self.assertLessEqual(open_count(), baseline)

        # Setup failure: bwrap never reports namespace info.
        self.sandbox(binary="/bin/false").run(
            self.workspace, ["/bin/true"], profile=self.profile,
            backend=NetworkBackend.SLIRP4NETNS)
        self.assertLessEqual(open_count(), baseline)

        # Python-level exception in the middle of the orchestration.
        with patch.object(harness.sandbox.json, "loads", side_effect=RuntimeError("boom")):
            self.sandbox().run(self.workspace, ["/bin/true"], profile=self.profile,
                               backend=NetworkBackend.SLIRP4NETNS)
        self.assertLessEqual(open_count(), baseline)

    def test_helper_without_pinned_namespace_support_fails_closed(self):
        incapable = Path(self.temp.name) / "old-slirp"
        incapable.write_text('#!/bin/sh\necho "--ready-fd --exit-fd"\nexit 0\n',
                             encoding="utf-8")
        incapable.chmod(0o755)
        marker = self.workspace_root / "command-ran"
        writes = []
        real_write = os.write

        def tracked_write(fd, data):
            writes.append((fd, data))
            return real_write(fd, data)

        with patch.object(harness.sandbox.os, "write", side_effect=tracked_write):
            result = self.sandbox(slirp_binary=str(incapable)).run(
                self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS)
        self.assertFalse(result.ok)
        self.assertIn("--netns-type", result.summary)
        self.assertFalse(marker.exists())
        self.assertEqual(writes, [], "no release may follow a fail-closed decision")

    def test_netns_mismatch_fails_closed_without_releasing_the_command(self):
        marker = self.workspace_root / "command-ran"
        writes = []
        real_write = os.write
        real_loads = json.loads

        def tampered(data, *args, **kwargs):
            info = real_loads(data, *args, **kwargs)
            if isinstance(info, dict) and "net-namespace" in info:
                info["net-namespace"] = info["net-namespace"] + 1
            return info

        def tracked_write(fd, data):
            writes.append((fd, data))
            return real_write(fd, data)

        with patch.object(harness.sandbox.json, "loads", side_effect=tampered), \
             patch.object(harness.sandbox.os, "write", side_effect=tracked_write):
            result = self.sandbox().run(
                self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS)
        self.assertFalse(result.ok)
        self.assertIn("does not match", result.summary)
        self.assertFalse(marker.exists())
        self.assertEqual(writes, [])

    def test_namespace_handles_are_type_checked_before_the_helper_runs(self):
        """Defence in depth: the pinned handles must be the expected ns types."""
        marker = self.workspace_root / "command-ran"
        writes = []
        real_write = os.write
        real_ioctl = fcntl.ioctl

        def wrong_type(fd, request, *args):
            if request == NS_GET_NSTYPE:
                return 0  # neither CLONE_NEWNET nor CLONE_NEWUSER
            return real_ioctl(fd, request, *args)

        def tracked_write(fd, data):
            writes.append((fd, data))
            return real_write(fd, data)

        with patch.object(harness.sandbox.fcntl, "ioctl", side_effect=wrong_type), \
             patch.object(harness.sandbox.os, "write", side_effect=tracked_write):
            result = self.sandbox().run(
                self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS)
        self.assertFalse(result.ok)
        self.assertIn("namespace", result.summary)
        self.assertFalse(marker.exists())
        self.assertEqual(writes, [])

    def test_real_namespace_handles_have_the_expected_types(self):
        observed = {}
        real_ioctl = fcntl.ioctl

        def watching(fd, request, *args):
            result = real_ioctl(fd, request, *args)
            if request == NS_GET_USERNS:
                observed["net"] = real_ioctl(fd, NS_GET_NSTYPE)
                observed["user"] = real_ioctl(result, NS_GET_NSTYPE)
            return result

        with patch.object(harness.sandbox.fcntl, "ioctl", side_effect=watching):
            result = self.sandbox().run(self.workspace, ["/bin/true"],
                                        profile=self.profile,
                                        backend=NetworkBackend.SLIRP4NETNS)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertEqual(observed.get("net"), CLONE_NEWNET)
        self.assertEqual(observed.get("user"), CLONE_NEWUSER)

    def test_pidfd_remains_the_process_identity_alongside_namespace_handles(self):
        opened = []
        real_pidfd_open = os.pidfd_open

        def tracked(pid, flags=0):
            opened.append(pid)
            return real_pidfd_open(pid, flags)

        with patch.object(harness.sandbox.os, "pidfd_open", side_effect=tracked):
            result = self.sandbox().run(self.workspace, ["/bin/true"],
                                        profile=self.profile,
                                        backend=NetworkBackend.SLIRP4NETNS)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertEqual(len(opened), 1, "the pidfd must still authenticate the child")


@unittest.skipUnless(os.environ.get("SPEAR_TEST_NETWORK_RACE") == "1",
                     "set SPEAR_TEST_NETWORK_RACE=1 for opt-in timing-race runs")
class OptInNetworkRaceTests(unittest.TestCase):
    """Repeated runs with a deliberate delay inside the formerly fatal window."""

    ITERATIONS = 40

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.workspace = Workspace.from_path(self.workspace_root)
        self.profile = ExecutionProfile.from_capabilities({Capability.NETWORK})
        self.addCleanup(self.temp.cleanup)

    def run_with_delay(self, delay, iterations=None):
        """Delay between pinning the namespaces and spawning the helper.

        ``os.pidfd_open`` is called immediately after the pin and immediately
        before the helper is spawned, so wrapping it injects the delay exactly
        where the PID-based attachment used to fail.
        """
        real_pidfd_open = os.pidfd_open

        def delayed(pid, flags=0):
            fd = real_pidfd_open(pid, flags)
            time.sleep(delay)
            return fd

        failures = []
        iterations = iterations or self.ITERATIONS
        sandbox = BubblewrapSandbox(network_ready_timeout_seconds=5)
        with patch.object(harness.sandbox.os, "pidfd_open", side_effect=delayed):
            for _ in range(iterations):
                result = sandbox.run(self.workspace, ["/bin/true"],
                                     profile=self.profile,
                                     backend=NetworkBackend.SLIRP4NETNS)
                if not result.ok:
                    failures.append(result.to_legacy_text()[:120])
        return failures

    def test_stable_with_20ms_delay(self):
        failures = self.run_with_delay(0.020)
        self.assertEqual(failures, [], f"{len(failures)}/{self.ITERATIONS} failed")

    def test_stable_with_50ms_delay(self):
        failures = self.run_with_delay(0.050)
        self.assertEqual(failures, [], f"{len(failures)}/{self.ITERATIONS} failed")

    def test_stable_with_50ms_delay_under_cpu_load(self):
        workers = [
            subprocess.Popen(
                [sys.executable, "-c",
                 "import time\nt=time.monotonic()\nwhile time.monotonic()-t<90: pass"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(4)
        ]
        try:
            failures = self.run_with_delay(0.050)
        finally:
            for worker in workers:
                worker.kill()
                worker.wait()
        self.assertEqual(failures, [], f"{len(failures)}/{self.ITERATIONS} failed")

    def test_characterization_old_pid_based_attachment_still_races(self):
        """Why the pinned form exists: the PID-based one fails deterministically.

        Exercises slirp4netns directly, not the runtime, which no longer offers
        the PID-based attachment.
        """
        sandbox = BubblewrapSandbox(network_ready_timeout_seconds=5)
        failures = 0
        attempts = 10
        for _ in range(attempts):
            with tempfile.TemporaryDirectory(prefix="spear-race-char-") as tmp:
                files = sandbox._network_files(Path(tmp))
                info_read, info_write = os.pipe()
                block_read, release_write = os.pipe()
                argv = sandbox.build_argv(
                    self.workspace, ["/bin/true"], profile=self.profile,
                    backend=NetworkBackend.SLIRP4NETNS, network_files=files,
                    info_fd=info_write, block_fd=block_read, sync_fd=release_write)
                bwrap = subprocess.Popen(
                    argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    shell=False, pass_fds=(info_write, block_read, release_write),
                    env={})
                os.close(info_write)
                os.close(block_read)
                info = json.loads(sandbox._wait_json_fd(info_read, 5).decode())
                child = info["child-pid"]
                time.sleep(0.020)
                ready_read, ready_write = os.pipe()
                exit_read, exit_write = os.pipe()
                helper = subprocess.run(
                    ["/usr/bin/slirp4netns", "--configure", "--disable-host-loopback",
                     "--ready-fd", str(ready_write), "--exit-fd", str(exit_read),
                     str(child), "tap0"],
                    capture_output=True, text=True, timeout=20,
                    pass_fds=(ready_write, exit_read), env={})
                if "setns" in helper.stderr:
                    failures += 1
                for fd in (info_read, release_write, ready_read, ready_write,
                           exit_read, exit_write):
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                for stream in (bwrap.stdout, bwrap.stderr):
                    stream.close()
                bwrap.kill()
                bwrap.wait()
                try:
                    os.kill(child, signal.SIGKILL)
                except OSError:
                    pass
        self.assertEqual(failures, attempts,
                         "the PID-based attachment no longer races; revisit the fix")


if __name__ == "__main__":
    unittest.main()
