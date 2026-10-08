"""The bubblewrap sandbox: mounts, profiles and what a command can reach."""

import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harness.sandbox

from harness.command_policy import CommandPolicy
from harness.resource_control import DEFAULT_RESOURCE_LIMITS, ExecutionProfile
from harness.tool_primitives import (
    Capability,
    ExecutionMode,
    NetworkBackend,
    ToolResult,
    shell_argv,
)
from harness.sandbox import BubblewrapSandbox, SandboxAvailability
from harness.tool_runtime import CommandRunner
from harness.workspace import SandboxSpec, Workspace


class MultiRootSandboxMountTests(unittest.TestCase):
    """Every declared root is mounted, with the same access as the primary."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.primary = base / "primary"; self.primary.mkdir()
        self.so3 = base / "so3"; self.so3.mkdir()
        self.lvgl = base / "lvgl"; self.lvgl.mkdir()
        self.ws = Workspace.from_path(self.primary, extra_roots=[self.so3, self.lvgl])
        self.sandbox = BubblewrapSandbox()

    def tearDown(self):
        self.temp.cleanup()

    def _mounts(self, profile):
        argv = self.sandbox.build_argv(self.ws, ["true"], profile)
        found = {}
        for index, token in enumerate(argv):
            if token in ("--bind", "--ro-bind") and index + 2 < len(argv):
                found[argv[index + 2]] = (token, argv[index + 1])
        return found

    def _expected(self, profile):
        """Host root → mount, as the sandbox resolves it for this profile."""
        mount = self.sandbox.mount_root(self.ws, profile)
        return self.ws.mount_map(
            mount, identity=mount != SandboxSpec().workspace_mount)

    def test_write_profile_binds_every_root_read_write(self):
        profile = ExecutionProfile.from_capabilities(
            [Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE])
        mounts = self._mounts(profile)
        for root in (self.primary, self.so3, self.lvgl):
            at = self._expected(profile)[str(root)]
            self.assertEqual(mounts[at], ("--bind", str(root)))

    def test_read_only_profile_binds_every_root_read_only(self):
        profile = ExecutionProfile.from_capabilities([Capability.FILESYSTEM_READ])
        mounts = self._mounts(profile)
        for root in (self.primary, self.so3, self.lvgl):
            at = self._expected(profile)[str(root)]
            self.assertEqual(mounts[at][0], "--ro-bind", at)

    def test_extra_roots_never_appear_without_read_access(self):
        # No filesystem:read at all: the sandbox gets an empty /workspace and
        # must not expose a secondary tree either.
        profile = ExecutionProfile.from_capabilities([])
        argv = self.sandbox.build_argv(self.ws, ["true"], profile)
        self.assertNotIn(str(self.so3), argv)
        self.assertNotIn("/workspaces/so3", argv)

    @patch.dict(os.environ, {"SPEAR_SANDBOX_IDENTITY_MOUNT": "0"})
    def test_mount_names_are_unique_when_basenames_collide(self):
        # Uniquifying is what the /workspaces/<name> scheme needs; identity
        # mounts are unique by construction, so this pins the legacy mount.
        other = Path(self.temp.name) / "nested"; (other / "so3").mkdir(parents=True)
        ws = Workspace.from_path(self.primary, extra_roots=[self.so3, other / "so3"])
        sandbox = BubblewrapSandbox()
        profile = ExecutionProfile.from_capabilities(
            [Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE])
        argv = sandbox.build_argv(ws, ["true"], profile)
        mounts = [argv[i + 2] for i, t in enumerate(argv)
                  if t in ("--bind", "--ro-bind") and argv[i + 2].startswith("/workspaces/")]
        self.assertEqual(len(mounts), len(set(mounts)), mounts)

    @patch.dict(os.environ, {"SPEAR_SANDBOX_IDENTITY_MOUNT": "0"})
    def test_single_root_argv_is_byte_identical_to_before(self):
        plain = Workspace.from_path(self.primary)
        profile = ExecutionProfile.from_capabilities(
            [Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE])
        argv = self.sandbox.build_argv(plain, ["true"], profile)
        self.assertNotIn("/workspaces", " ".join(argv))

    def test_mount_map_is_exposed_for_the_prompt(self):
        # The model must be told where each root appears, or it cannot use a
        # secondary tree from bash at all.
        self.assertEqual(self.ws.mount_map(), {
            str(self.primary): "/workspace",
            str(self.so3): "/workspaces/so3",
            str(self.lvgl): "/workspaces/lvgl",
        })


class WorkspaceMountProfileTests(unittest.TestCase):
    """Step 1d-1: Bubblewrap materializes only granted workspace access."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.workspace = Workspace.from_path(self.workspace_root)
        self.marker = self.workspace_root / "host-marker.txt"
        self.marker.write_text("host marker", encoding="utf-8")
        self.outside = Path(self.temp.name) / "outside.txt"
        self.outside.write_text("outside", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def profile(*capabilities):
        return ExecutionProfile.from_capabilities(capabilities)

    def run_shell(self, profile, script, *, backend=NetworkBackend.CLOSED):
        return BubblewrapSandbox().run(
            self.workspace,
            ["/bin/sh", "-lc", script],
            profile=profile,
            backend=backend,
            resource_limits=DEFAULT_RESOURCE_LIMITS,
        )

    def test_profile_without_filesystem_access_gets_private_empty_workspace(self):
        profile = self.profile()
        argv = BubblewrapSandbox().build_argv(
            self.workspace, ["/bin/true"], profile=profile
        )
        self.assertNotIn(str(self.workspace_root), argv)
        self.assertIn("--dir", argv)
        result = self.run_shell(
            profile,
            "test \"$PWD\" = /workspace && test ! -e host-marker.txt && touch local-created",
        )
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertFalse((self.workspace_root / "local-created").exists())

    def test_filesystem_read_profile_is_real_read_only_workspace(self):
        profile = self.profile(Capability.FILESYSTEM_READ)
        sandbox = BubblewrapSandbox()
        mount = sandbox.mount_root(self.workspace, profile)
        argv = sandbox.build_argv(self.workspace, ["/bin/true"], profile=profile)
        # Locate the workspace bind by destination: other read-only binds
        # (/usr, /etc/alternatives) precede it.
        bind_at = next(i for i, a in enumerate(argv)
                       if a == "--ro-bind" and argv[i + 2] == mount)
        self.assertEqual(argv[bind_at + 1:bind_at + 3], [str(self.workspace_root), mount])
        result = self.run_shell(
            profile,
            "test -r host-marker.txt && test \"$(cat host-marker.txt)\" = 'host marker' "
            "&& stat host-marker.txt >/dev/null "
            "&& ! touch created && ! printf changed > host-marker.txt "
            "&& ! rm host-marker.txt && ! mv host-marker.txt renamed "
            "&& ! mkdir created-dir && ! ln -s host-marker.txt created-link",
        )
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertEqual(self.marker.read_text(encoding="utf-8"), "host marker")
        self.assertFalse((self.workspace_root / "created").exists())

    def test_workspace_write_profile_keeps_real_read_write_workspace(self):
        profile = self.profile(Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE)
        sandbox = BubblewrapSandbox()
        mount = sandbox.mount_root(self.workspace, profile)
        argv = sandbox.build_argv(self.workspace, ["/bin/true"], profile=profile)
        bind_at = next(i for i, a in enumerate(argv)
                       if a == "--bind" and argv[i + 2] == mount)
        self.assertEqual(argv[bind_at + 1:bind_at + 3], [str(self.workspace_root), mount])
        result = self.run_shell(profile, "printf changed > host-marker.txt && touch created")
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertEqual(self.marker.read_text(encoding="utf-8"), "changed")
        self.assertTrue((self.workspace_root / "created").exists())

    def test_shell_complex_does_not_expand_workspace_mount_permissions(self):
        readonly = self.profile(Capability.FILESYSTEM_READ, Capability.SHELL_COMPLEX)
        writable = self.profile(
            Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE, Capability.SHELL_COMPLEX
        )
        self.assertTrue(self.run_shell(readonly, "cat host-marker.txt | cat && ! touch denied").ok)
        self.assertFalse((self.workspace_root / "denied").exists())
        self.assertTrue(self.run_shell(writable, "printf rw | cat > shell-created").ok)
        self.assertEqual((self.workspace_root / "shell-created").read_text(), "rw")

    def test_symlink_escapes_remain_unusable_in_read_only_and_read_write_profiles(self):
        (self.workspace_root / "outside-link").symlink_to(self.outside)
        readonly = self.profile(Capability.FILESYSTEM_READ)
        writable = self.profile(Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE)
        self.assertTrue(
            self.run_shell(readonly, "test ! -e outside-link && ! ln -s host-marker.txt new-link").ok
        )
        self.assertFalse((self.workspace_root / "new-link").exists())
        self.assertTrue(self.run_shell(writable, "test ! -e outside-link && ln -s /tmp/nope created-link").ok)
        self.assertTrue((self.workspace_root / "created-link").is_symlink())
        self.assertTrue(self.run_shell(writable, "test ! -e created-link").ok)

    def test_network_only_gets_private_workspace_and_slirp(self):
        profile = self.profile(Capability.NETWORK)
        result = self.run_shell(
            profile,
            "test ! -e host-marker.txt && touch local-created "
            "&& /usr/sbin/ip route | grep -q 'default via 10.0.2.2'",
            backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertFalse((self.workspace_root / "local-created").exists())

    def test_network_read_only_and_network_read_write_materialize_their_profiles(self):
        readonly = self.profile(Capability.FILESYSTEM_READ, Capability.NETWORK)
        writable = self.profile(
            Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE, Capability.NETWORK
        )
        readonly_result = self.run_shell(
            readonly,
            "test -r host-marker.txt && ! printf changed > host-marker.txt "
            "&& /usr/sbin/ip route | grep -q 'default via 10.0.2.2'",
            backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertTrue(readonly_result.ok, readonly_result.to_legacy_text())
        self.assertEqual(self.marker.read_text(encoding="utf-8"), "host marker")
        writable_result = self.run_shell(
            writable,
            "printf network-rw > host-marker.txt && /usr/sbin/ip route | grep -q 'default via 10.0.2.2'",
            backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertTrue(writable_result.ok, writable_result.to_legacy_text())
        self.assertEqual(self.marker.read_text(encoding="utf-8"), "network-rw")

    def test_profile_none_retains_legacy_read_write_mount_but_runner_uses_explicit_profile(self):
        sandbox_argv = BubblewrapSandbox()
        argv = sandbox_argv.build_argv(self.workspace, ["/bin/true"])
        mount = sandbox_argv.mount_root(self.workspace)
        bind_at = next(i for i, a in enumerate(argv)
                       if a == "--bind" and argv[i + 2] == mount)
        self.assertEqual(argv[bind_at + 1:bind_at + 3],
                         [str(self.workspace_root), mount])
        sandbox = MagicMock()
        sandbox.ensure_available.return_value = ToolResult("ok", "available")
        sandbox.run.return_value = ToolResult("ok", "done")
        runner = CommandRunner(sandbox=sandbox)
        runner.run_sandboxed(
            self.workspace,
            ["make"],
            self.profile(Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE),
        )
        self.assertIsNotNone(sandbox.run.call_args.kwargs["profile"])


class ReadOnlySandboxExecutionTests(unittest.TestCase):
    """Step 1d-2: authorized external reads are confined like every command."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        (self.workspace_root / "file.txt").write_text("needle\n", encoding="utf-8")
        self.outside = Path(self.temp.name) / "outside.txt"
        self.outside.write_text("outside marker", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(self.workspace_root)], check=True)
        self.workspace = Workspace.from_path(self.workspace_root)
        self.profile = ExecutionProfile.from_capabilities({Capability.FILESYSTEM_READ})

    def tearDown(self):
        self.temp.cleanup()

    def run_read_only(self, argv):
        return BubblewrapSandbox().run(
            self.workspace,
            argv,
            profile=self.profile,
            resource_limits=DEFAULT_RESOURCE_LIMITS,
        )

    def test_representative_read_only_commands_run_in_read_only_workspace(self):
        cases = {
            "pwd": ["/bin/pwd"],
            "ls": ["/bin/ls", "file.txt"],
            "cat": ["/bin/cat", "file.txt"],
            "grep": ["/bin/grep", "needle", "file.txt"],
            "find": ["/usr/bin/find", ".", "-maxdepth", "1", "-name", "file.txt"],
            "git status": ["/usr/bin/git", "status", "--porcelain"],
        }
        outputs = {}
        for name, argv in cases.items():
            with self.subTest(command=name):
                result = self.run_read_only(argv)
                self.assertTrue(result.ok, result.to_legacy_text())
                outputs[name] = result.stdout
        self.assertEqual(
            outputs["pwd"].strip(),
            BubblewrapSandbox().mount_root(self.workspace, self.profile))
        self.assertIn("file.txt", outputs["ls"])
        self.assertIn("needle", outputs["cat"])
        self.assertIn("needle", outputs["grep"])
        self.assertIn("file.txt", outputs["find"])

    @unittest.skipUnless(Path("/usr/bin/rg").is_file(),
                         "rg is not installed in the minimal /usr sandbox runtime")
    def test_rg_runs_when_installed_in_the_sandbox_runtime(self):
        result = self.run_read_only(["/usr/bin/rg", "needle", "file.txt"])
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertIn("needle", result.stdout)

    def test_read_only_sandbox_hides_host_and_enforces_limits(self):
        host_home = Path(self.temp.name) / "host-home"
        host_home.mkdir()
        (host_home / ".ssh").mkdir()
        (host_home / ".gitconfig").write_text("[user]\nname = host\n", encoding="utf-8")
        script = (
            "import os,resource,sys; "
            "checks=[os.environ.get('HOME') == '/home/sandbox', "
            "not os.path.exists('/home/sandbox/.ssh'), "
            "not os.path.exists('/home/sandbox/.gitconfig'), "
            f"not os.path.exists({str(self.outside)!r}), "
            "not os.path.exists('/opt/llm'), "
            # /etc/passwd IS exposed now, read-only: bitbake's is_local_uid()
            # opens it and dies without it. What must stay hidden is the
            # credential half of /etc, which the assertion below pins.
            "not os.path.exists('/etc/shadow'), "
            "not os.path.exists('/etc/gshadow'), "
            "not os.environ.get('SPEAR_TEST_SECRET'), "
            "resource.getrlimit(resource.RLIMIT_NOFILE) == (4096,4096), "
            "resource.getrlimit(resource.RLIMIT_CORE) == (0,0)]; "
            "sys.exit(0 if all(checks) else 1)"
        )
        with patch.dict(os.environ, {
            "HOME": str(host_home),
            "SPEAR_TEST_SECRET": "controlled-test-secret",
        }, clear=False):
            result = self.run_read_only(["/usr/bin/python3", "-c", script])
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertFalse(
            self.run_read_only(["/bin/cat", str(self.outside)]).ok,
            "an absolute host path must not be readable through the sandbox",
        )
        self.assertFalse(self.run_read_only(["/bin/sh", "-lc", "touch denied"]).ok)
        self.assertFalse((self.workspace_root / "denied").exists())


class Phase1bBubblewrapContractTests(unittest.TestCase):
    """Phase 1b contract tests. They intentionally fail until implemented."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.workspace = Workspace.from_path(self.workspace_root)
        self.outside_root = Path(self.temp.name) / "outside"
        self.outside_root.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def sandbox(self, *, binary="bwrap"):
        sandbox_type = getattr(harness.sandbox, "BubblewrapSandbox", None)
        self.assertIsNotNone(
            sandbox_type,
            "BubblewrapSandbox is not implemented yet (expected Phase 1b)",
        )
        return sandbox_type(binary=binary)

    def test_build_argv_exposes_workspace_only_at_its_mount(self):
        sandbox = self.sandbox()
        mount = sandbox.mount_root(self.workspace)
        argv = sandbox.build_argv(self.workspace, ["/bin/sh", "-lc", "pwd"])
        rendered = "\0".join(argv)
        bind_at = next(i for i, a in enumerate(argv)
                       if a == "--bind" and argv[i + 2] == mount)
        self.assertEqual(argv[bind_at + 1:bind_at + 3],
                         [str(self.workspace_root), mount])
        self.assertIn("--chdir", argv)
        chdir_at = argv.index("--chdir")
        # The cwd must be the mount, whatever the mount is: a --chdir that does
        # not match the bind lands the command outside the workspace.
        self.assertEqual(argv[chdir_at + 1], mount)
        self.assertNotIn("--ro-bind\0/\0/", rendered)

    def test_supported_profiles_preserve_phase_1b_confinement(self):
        for capabilities in (
            {Capability.FILESYSTEM_READ},
            {Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE},
            {Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE, Capability.SHELL_COMPLEX},
        ):
            with self.subTest(capabilities=capabilities):
                profile = ExecutionProfile.from_capabilities(capabilities)
                argv = self.sandbox().build_argv(self.workspace, ["/bin/true"], profile=profile)
                rendered = "\0".join(argv)
                self.assertIn("--unshare-net", argv)
                self.assertIn("--clearenv", argv)
                self.assertIn(self.sandbox().mount_root(self.workspace, profile), argv)
                self.assertIn("--tmpfs", argv)
                self.assertIn("/home/sandbox", argv)
                self.assertNotIn("/dev/nvidia0", rendered)
                self.assertNotIn("docker.sock", rendered)
                self.assertNotIn("SSH_AUTH_SOCK", rendered)

    def test_sensitive_profiles_fail_closed_before_bubblewrap_execution(self):
        for capabilities in (
            {Capability.GPU},
            {Capability.CONTAINER_RUNTIME},
            {Capability.SECRETS},
        ):
            with self.subTest(capabilities=capabilities):
                profile = ExecutionProfile.from_capabilities(capabilities)
                with self.assertRaisesRegex(ValueError, "capability/profile not implemented"):
                    self.sandbox().build_argv(self.workspace, ["/bin/true"], profile=profile)


class Phase1bBubblewrapAdversarialTests(unittest.TestCase):
    """Real escape attempts against the Phase 1b Bubblewrap boundary."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.workspace = Workspace.from_path(self.workspace_root)
        self.outside = Path(self.temp.name) / "host-outside"
        self.outside.mkdir()
        self.sandbox = BubblewrapSandbox()
        # Where the tree answers inside: its own host path, or /workspace under
        # the legacy mount. Asked of the sandbox so the test tracks the code.
        self.mount = self.sandbox.mount_root(self.workspace)

    def tearDown(self):
        self.temp.cleanup()

    def run_shell(self, script):
        return self.sandbox.run(self.workspace, ["/bin/sh", "-lc", script])

    def test_workspace_writable_but_host_filesystem_is_unreachable(self):
        marker = self.outside / "marker.txt"
        marker.write_text("unchanged")
        result = self.run_shell(
            f"printf inside > {self.mount}/inside.txt"
        )
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertEqual((self.workspace_root / "inside.txt").read_text(), "inside")

        # The operator's own home, not one particular person's: the point is
        # that the sandbox hides whatever home the caller has.
        home = str(Path.home())

        for target in (
            "/etc/spear-adversarial-marker",
            f"{home}/spear-adversarial-marker",
            str(marker),
            f"{self.mount}/../etc/spear-adversarial-marker",
            "/opt/llm/spear-adversarial-marker",
        ):
            attempted = self.run_shell(f"printf escaped > {target}")
            self.assertFalse(attempted.ok, f"unexpected host write: {target}")
        self.assertEqual(marker.read_text(), "unchanged")
        self.assertFalse(Path("/etc/spear-adversarial-marker").exists())
        self.assertFalse(Path(home, "spear-adversarial-marker").exists())
        self.assertTrue(self.run_shell(f"test ! -e {home}").ok)
        self.assertFalse(Path("/opt/llm/spear-adversarial-marker").exists())
        self.assertTrue(self.run_shell("test ! -e /opt/llm").ok)

    def test_preexisting_and_runtime_symlink_escapes_are_unusable(self):
        target = self.outside / "marker.txt"
        target.write_text("unchanged")
        directory = self.outside / "directory"
        directory.mkdir()
        (self.workspace_root / "file-link").symlink_to(target)
        (self.workspace_root / "dir-link").symlink_to(directory, target_is_directory=True)

        result = self.run_shell(
            f"test ! -e {self.mount}/file-link && test ! -e {self.mount}/dir-link "
            f"&& ln -s '{target}' {self.mount}/runtime-link "
            f"&& test ! -e {self.mount}/runtime-link"
        )
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertFalse(self.run_shell(f"printf x > {self.mount}/file-link").ok)
        self.assertFalse(self.run_shell(f"printf x > {self.mount}/dir-link/escaped").ok)
        self.assertFalse(self.run_shell(f"printf x > {self.mount}/runtime-link").ok)
        self.assertEqual(target.read_text(), "unchanged")
        self.assertFalse((directory / "escaped").exists())

    def test_environment_home_and_user_configuration_are_not_inherited(self):
        host_home = self.outside / "host-home"
        host_home.mkdir()
        (host_home / ".ssh").mkdir()
        (host_home / ".gitconfig").write_text("[user]\nname = host\n")
        secrets = {
            "OPENAI_API_KEY": "openai-test-secret",
            "ANTHROPIC_API_KEY": "anthropic-test-secret",
            "AWS_SECRET_ACCESS_KEY": "aws-test-secret",
            "SSH_AUTH_SOCK": "/tmp/host-agent.sock",
            "SPEAR_TEST_SECRET": "spear-test-secret",
            "HOME": str(host_home),
        }
        with patch.dict(os.environ, secrets, clear=False):
            result = self.run_shell(
                "test \"$HOME\" = /home/sandbox && test ! -e \"$HOME/.ssh\" "
                "&& test ! -e \"$HOME/.gitconfig\" "
                "&& test -z \"${OPENAI_API_KEY:-}${ANTHROPIC_API_KEY:-}\" "
                "&& test -z \"${AWS_SECRET_ACCESS_KEY:-}${SSH_AUTH_SOCK:-}${SPEAR_TEST_SECRET:-}\""
            )
        self.assertTrue(result.ok, result.to_legacy_text())

    def test_network_namespace_has_no_host_connection_or_inherited_sockets(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            argv = self.sandbox.build_argv(self.workspace, ["/bin/true"])
            self.assertIn("--unshare-net", argv)
            result = self.sandbox.run(
                self.workspace,
                ["/usr/bin/python3", "-c", (
                    "import socket,sys; "
                    f"s=socket.socket(); s.settimeout(1); "
                    f"sys.exit(0 if s.connect_ex(('127.0.0.1',{port})) != 0 else 1)"
                )],
            )
        finally:
            server.close()
        self.assertTrue(result.ok, result.to_legacy_text())
        interfaces = self.run_shell(
            "if test -d /sys/class/net; then "
            "test \"$(find /sys/class/net -mindepth 1 -maxdepth 1 -printf '%f\\n' | sort)\" = lo; fi "
            "&& test ! -e /sys/class/net/eth0"
        )
        self.assertTrue(interfaces.ok, interfaces.to_legacy_text())

    def test_proc_pid_namespace_and_child_lifecycle(self):
        result = self.run_shell(
            "test -e /proc/1 && test \"$$\" -gt 1 && test \"$$\" -le 3 "
            "&& test \"$(readlink /proc/1/exe)\" = /usr/bin/bwrap "
            "&& test ! -e /proc/999999 && ! kill -0 999999"
        )
        self.assertTrue(result.ok, result.to_legacy_text())

        before = subprocess.check_output(["ps", "-eo", "args="], text=True)
        child = self.run_shell("sleep 30 & echo child-started")
        self.assertTrue(child.ok, child.to_legacy_text())
        after = subprocess.check_output(["ps", "-eo", "args="], text=True)
        self.assertEqual(after.count("sleep 30"), before.count("sleep 30"))
        self.assertIn("--die-with-parent", self.sandbox.build_argv(self.workspace, ["/bin/true"]))

    def test_tmp_dev_and_complex_shell_escapes_remain_confined(self):
        host_tmp = self.outside / "host-tmp-marker"
        host_tmp.write_text("host")
        target = self.outside / "external.txt"
        target.write_text("unchanged")
        (self.workspace_root / "external-link").symlink_to(target)
        script = (
            "touch /tmp/sandbox-marker && test ! -e '" + str(host_tmp) + "' "
            "&& test ! -e /dev/nvidia0 && test ! -b /dev/sda "
            "&& target='" + str(target) + "'; "
            "(cd / && test ! -e \"$target\") | cat > /workspace/pipeline.txt; "
            "test ! -e \"$(readlink -f /workspace/external-link 2>/dev/null || true)\""
        )
        result = self.run_shell(script)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertFalse((Path("/tmp") / "sandbox-marker").exists())
        self.assertEqual(target.read_text(), "unchanged")

    def test_availability_states_and_cache_fail_closed_in_auto(self):
        unavailable = BubblewrapSandbox(binary="/definitely/missing/bwrap")
        self.assertEqual(unavailable.ensure_available(self.workspace).status, "failed")
        self.assertEqual(unavailable.availability, SandboxAvailability.ABSENT)
        not_executable = self.outside / "not-executable-bwrap"
        not_executable.write_text("not executable")
        not_executable.chmod(0o644)
        blocked = BubblewrapSandbox(binary=str(not_executable))
        self.assertEqual(blocked.ensure_available(self.workspace).status, "failed")
        self.assertEqual(blocked.availability, SandboxAvailability.INEXECUTABLE)

        refused = BubblewrapSandbox(binary="bwrap")
        with patch.object(refused, "preflight", return_value=ToolResult("failed", "refused")) as preflight:
            self.assertEqual(refused.ensure_available(self.workspace).status, "failed")
            self.assertEqual(refused.ensure_available(self.workspace).status, "failed")
        self.assertEqual(preflight.call_count, 1)
        self.assertEqual(refused.availability, SandboxAvailability.REFUSED)


class AlternativesMountTests(unittest.TestCase):
    """/etc/alternatives must be visible, and nothing else from /etc."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.workspace = Workspace.from_path(self.workspace_root)
        self.profile = ExecutionProfile.from_capabilities({
            Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE,
            Capability.SHELL_COMPLEX,
        })
        self.addCleanup(self.temp.cleanup)

    def test_build_argv_binds_the_alternatives_directory_read_only(self):
        argv = BubblewrapSandbox().build_argv(
            self.workspace, ["/bin/true"])
        self.assertIn("--dir", argv)
        rendered = "\0".join(argv)
        self.assertIn("--ro-bind\0/etc/alternatives\0/etc/alternatives", rendered)
        # Never a writable bind, and never the whole of /etc.
        self.assertNotIn("--bind\0/etc", rendered)
        self.assertNotIn("--ro-bind\0/etc\0/etc", rendered)

    def test_missing_alternatives_directory_is_simply_not_bound(self):
        spec = SandboxSpec(alternatives="/definitely/missing/alternatives")
        argv = BubblewrapSandbox(spec=spec).build_argv(
            self.workspace, ["/bin/true"])
        self.assertNotIn("/definitely/missing/alternatives", argv)
        self.assertIn("/etc", argv)          # the mount point still exists

    def test_etc_is_sealed_read_only(self):
        result = BubblewrapSandbox().run(
            self.workspace,
            ["/bin/sh", "-lc", "touch /etc/escape 2>&1; echo rc=$?; "
                               "mkdir /etc/d 2>&1; echo rc2=$?"],
            profile=self.profile)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertIn("Read-only file system", result.stdout)
        self.assertIn("rc=1", result.stdout)
        self.assertIn("rc2=1", result.stdout)

    def test_cc_resolves_inside_the_sandbox(self):
        result = BubblewrapSandbox().run(
            self.workspace, ["/bin/sh", "-lc", "command -v cc && cc --version | head -1"],
            profile=self.profile)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertIn("/usr/bin/cc", result.stdout)

    def test_make_uses_the_implicit_cc_rule(self):
        (self.workspace_root / "a.c").write_text(
            '#include <stdio.h>\nint main(void){puts("ok");return 0;}\n',
            encoding="utf-8")
        (self.workspace_root / "Makefile").write_text(
            "all: a\na: a.c\n\t$(CC) -O2 -o a a.c\n", encoding="utf-8")
        result = BubblewrapSandbox().run(
            self.workspace, ["/usr/bin/make"], profile=self.profile)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertTrue((self.workspace_root / "a").exists())

    def test_no_other_etc_content_is_exposed(self):
        """Exactly three entries, and the list is the point.

        /etc is a tmpfs so nothing outside the workspace is writable. What is
        bound into it is bound deliberately: `alternatives` because cc and awk
        are symlinks through it, and passwd/group because build tools resolve
        uids -- bitbake's is_local_uid() opens /etc/passwd and dies without it.
        Anything else appearing here is a leak, and shadow above all.
        """
        result = BubblewrapSandbox().run(
            self.workspace, ["/bin/sh", "-lc", "ls -A /etc"], profile=self.profile)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertEqual(sorted(result.stdout.split()),
                         ["alternatives", "group", "passwd"])

    def test_the_credential_half_of_etc_stays_out(self):
        result = BubblewrapSandbox().run(
            self.workspace,
            ["/bin/sh", "-lc", "cat /etc/shadow /etc/gshadow /etc/sudoers 2>&1; echo rc=$?"],
            profile=self.profile)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertNotIn("root:", result.stdout)
        self.assertIn("rc=1", result.stdout)

    def test_the_alternatives_bind_is_not_writable(self):
        result = BubblewrapSandbox().run(
            self.workspace,
            ["/bin/sh", "-lc", "touch /etc/alternatives/x 2>&1; echo rc=$?"],
            profile=self.profile)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertIn("rc=1", result.stdout)

    def test_network_profile_still_gets_its_resolver_files(self):
        argv = BubblewrapSandbox().build_argv(
            self.workspace, ["/bin/true"],
            profile=ExecutionProfile.from_capabilities({Capability.NETWORK}),
            backend=NetworkBackend.SLIRP4NETNS,
            network_files=(Path("/tmp/resolv.conf"), Path("/tmp/nsswitch.conf"), None))
        rendered = "\0".join(argv)
        self.assertIn("/etc/resolv.conf", rendered)
        self.assertIn("/etc/nsswitch.conf", rendered)
        self.assertIn("/etc/alternatives", rendered)
        # /etc is created exactly once even though two features want it, and
        # it is sealed after the last bind into it.
        self.assertEqual(
            [argv[i + 1] for i, a in enumerate(argv) if a == "--tmpfs"].count("/etc"), 1)
        self.assertEqual(
            [argv[i + 1] for i, a in enumerate(argv) if a == "--dir"].count("/etc"), 0)
        seal = argv.index("--remount-ro")
        self.assertEqual(argv[seal + 1], "/etc")
        last_etc_bind = max(i for i, a in enumerate(argv)
                            if a == "--ro-bind" and argv[i + 2].startswith("/etc"))
        self.assertGreater(seal, last_etc_bind)


class OutsideReadSandboxTests(unittest.TestCase):
    """The grant is a read-only mount, so the write boundary never moves."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.home())
        self.root = Path(self.temp.name) / "workspace"
        self.root.mkdir()
        self.outside = Path(self.temp.name) / "notes"
        self.outside.mkdir()
        (self.outside / "memory.md").write_text("remembered\n")
        self.workspace = Workspace.from_path(self.root)
        self.sandbox = BubblewrapSandbox()
        self.policy = CommandPolicy()
        self.policy.bind_workspace(self.workspace)

    def tearDown(self):
        self.temp.cleanup()

    def profile_for(self, command):
        assessment = self.policy.classify(command)
        outcome = self.policy.authorize(assessment, ExecutionMode.AUTO)
        self.assertTrue(outcome.allowed, assessment.reason)
        return assessment, ExecutionProfile.from_capabilities(
            outcome.granted_capabilities, assessment.host_read_paths)

    def test_the_outside_tree_is_visible_and_read_only(self):
        command = f"cat {self.outside}/memory.md"
        _, profile = self.profile_for(command)
        result = self.sandbox.run(
            self.workspace, shell_argv(command), profile=profile)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertIn("remembered", result.stdout)

        overwrite = self.sandbox.run(
            self.workspace,
            shell_argv(f"printf pwned > {self.outside}/memory.md"),
            profile=profile)
        self.assertFalse(overwrite.ok)
        self.assertEqual((self.outside / "memory.md").read_text(), "remembered\n")

    def test_a_tree_that_was_never_named_stays_invisible(self):
        other = Path(self.temp.name) / "unnamed"
        other.mkdir()
        (other / "secret.txt").write_text("secret\n")
        _, profile = self.profile_for(f"cat {self.outside}/memory.md")
        result = self.sandbox.run(
            self.workspace,
            shell_argv(f"test ! -e {other}/secret.txt"),
            profile=profile)
        self.assertTrue(result.ok, "an unnamed tree must not be exposed")

    def test_the_workspace_stays_writable_under_an_outside_read(self):
        # A build that also reads a reference tree: the outside bind must not
        # cost the workspace the write it was granted.
        profile = ExecutionProfile.from_capabilities(
            {Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE,
             Capability.SHELL_COMPLEX, Capability.HOST_READ},
            (str(self.outside),))
        mount = self.sandbox.mount_root(self.workspace, profile)
        result = self.sandbox.run(
            self.workspace,
            shell_argv(f"printf ok > {mount}/written.txt"),
            profile=profile)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertEqual((self.root / "written.txt").read_text(), "ok")


class ToolchainReachableInsideTheSandboxTests(unittest.TestCase):
    """The cross-compilers must be on PATH and their symlinks must resolve.

    ib.md tells the operator to symlink each toolchain's bin/ into
    /usr/local/bin. The sandbox put neither that directory on PATH nor
    /opt/toolchains in the mount set, so every symlink dangled and `make so3`
    died on "aarch64-none-elf-gcc: No such file or directory" with the compiler
    mounted and unreachable.
    """

    def test_usr_local_bin_is_on_the_sandbox_path(self):
        self.assertIn("/usr/local/bin", SandboxSpec().path.split(":"))

    def test_the_toolchain_root_is_bound_when_present(self):
        spec = SandboxSpec()
        self.assertIn("/opt/toolchains", spec.toolchain_dirs)
        if not Path("/opt/toolchains").is_dir():
            self.skipTest("no /opt/toolchains on this host")
        with tempfile.TemporaryDirectory() as tmp:
            argv = BubblewrapSandbox().build_argv(
                Workspace.from_path(tmp), ["/bin/true"])
        pairs = [(argv[i + 1], argv[i + 2]) for i, a in enumerate(argv)
                 if a == "--ro-bind" and i + 2 < len(argv)]
        self.assertIn(("/opt/toolchains", "/opt/toolchains"), pairs)


if __name__ == "__main__":
    unittest.main()
