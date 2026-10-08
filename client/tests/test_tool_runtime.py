"""The tool runtime: workspace, results, the command runner and the audit log."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from harness.command_policy import CommandPolicy
from harness.resource_control import DEFAULT_RESOURCE_LIMITS, ExecutionProfile
from harness.tool_primitives import (
    Capability,
    ExecutionMode,
    PathNotFoundError,
    PathPolicyError,
    PIPEFAIL_PRELUDE,
    SHELL_BINARY,
    ToolResult,
    shell_argv,
)
from harness.tool_runtime import AuditLogger, CommandRunner
from harness.workspace import Workspace


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "workspace"
        self.root.mkdir()
        self.outside = Path(self.temp.name) / "outside"
        self.outside.mkdir()
        (self.root / "inside.txt").write_text("inside")
        (self.outside / "secret.txt").write_text("outside")

    def tearDown(self):
        self.temp.cleanup()

    def test_relative_path_inside_workspace_resolves(self):
        workspace = Workspace.from_path(self.root)
        self.assertEqual(workspace.resolve("inside.txt"), self.root / "inside.txt")

    def test_parent_traversal_is_rejected(self):
        workspace = Workspace.from_path(self.root)
        with self.assertRaises(PathPolicyError):
            workspace.resolve("../outside/secret.txt")

    @patch.dict(os.environ, {"SPEAR_SANDBOX_IDENTITY_MOUNT": "0"})
    def test_absolute_path_requires_explicit_option_under_the_legacy_mount(self):
        workspace = Workspace.from_path(self.root)
        with self.assertRaises(PathPolicyError):
            workspace.resolve(self.root / "inside.txt")

    def test_host_path_is_accepted_when_it_is_the_sandbox_mount(self):
        """The gate distinguishes host naming from sandbox naming.

        Under the identity mount there is no distinction left: the sandbox
        binds the tree at its own host path and the prompt says so, exactly as
        /workspace was accepted for being the sandbox's own naming. Refusing it
        here would leave bash and edit_file disagreeing about the same path.
        """
        workspace = Workspace.from_path(self.root)
        self.assertEqual(workspace.resolve(self.root / "inside.txt"),
                         self.root / "inside.txt")
        # Containment is unaffected: outside stays outside.
        with self.assertRaises(PathPolicyError):
            workspace.resolve(self.outside / "secret.txt")

    def test_allowed_absolute_path_still_must_be_inside_workspace(self):
        workspace = Workspace.from_path(self.root, allow_absolute_paths=True)
        self.assertEqual(workspace.resolve(self.root / "inside.txt"), self.root / "inside.txt")
        with self.assertRaises(PathPolicyError):
            workspace.resolve(self.outside / "secret.txt")

    def test_symlink_escape_is_rejected_before_file_access(self):
        link = self.root / "outside-link"
        link.symlink_to(self.outside, target_is_directory=True)
        workspace = Workspace.from_path(self.root)
        with self.assertRaises(PathPolicyError):
            workspace.resolve("outside-link/secret.txt")

    def test_new_file_under_symlinked_parent_is_rejected(self):
        link = self.root / "outside-link"
        link.symlink_to(self.outside, target_is_directory=True)
        workspace = Workspace.from_path(self.root)
        with self.assertRaises(PathPolicyError):
            workspace.resolve("outside-link/new.txt")

    def test_required_missing_path_is_reported(self):
        workspace = Workspace.from_path(self.root)
        with self.assertRaises(PathNotFoundError):
            workspace.resolve("missing.txt", must_exist=True)


class ToolResultTests(unittest.TestCase):
    def test_legacy_rendering_is_compatible(self):
        self.assertEqual(ToolResult("ok", "done").to_legacy_text(), "OK: done")
        self.assertEqual(ToolResult("cancelled", "no").to_legacy_text(), "CANCELLED")
        self.assertEqual(ToolResult("denied", "blocked").to_legacy_text(), "ERROR: blocked")


class MultiRootWorkspaceTests(unittest.TestCase):
    """A workspace may declare extra roots beyond the launch directory."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.primary = base / "primary"; (self.primary / "sub").mkdir(parents=True)
        self.other = base / "other"; (self.other / "usr" / "src").mkdir(parents=True)
        self.outside = base / "outside"; self.outside.mkdir()
        (self.primary / "here.txt").write_text("here")
        (self.other / "usr" / "src" / "ping.c").write_text("int main(void){return 0;}")
        (self.outside / "secret.txt").write_text("secret")
        self.ws = Workspace.from_path(self.primary, extra_roots=[self.other])

    def tearDown(self):
        self.temp.cleanup()

    def test_relative_paths_still_resolve_against_the_primary_root_only(self):
        self.assertEqual(self.ws.resolve("here.txt"), self.primary / "here.txt")
        # A relative path never reaches a secondary root: that would make the
        # same string mean different files depending on the root list.
        with self.assertRaises(PathPolicyError):
            self.ws.resolve("usr/src/ping.c", must_exist=True)

    def test_absolute_paths_reach_declared_extra_roots(self):
        target = self.other / "usr" / "src" / "ping.c"
        self.assertEqual(self.ws.resolve(str(target), must_exist=True), target)
        # And a file that does not exist yet, inside an extra root, is legal:
        # creating is the point of declaring the root.
        self.assertEqual(self.ws.resolve(str(self.other / "usr" / "src" / "new.c")),
                         self.other / "usr" / "src" / "new.c")

    def test_undeclared_trees_remain_unreachable(self):
        for raw in (str(self.outside / "secret.txt"),
                    str(self.outside / "new.txt"),
                    "../outside/secret.txt",
                    str(self.other / ".." / "outside" / "secret.txt")):
            with self.assertRaises(PathPolicyError, msg=raw):
                self.ws.resolve(raw)

    @patch.dict(os.environ, {"SPEAR_SANDBOX_IDENTITY_MOUNT": "0"})
    def test_absolute_under_primary_still_needs_the_legacy_flag(self):
        # Unchanged behaviour: declaring extra roots must not silently start
        # accepting absolute paths into the launch directory. Pinned to the
        # legacy mount, where a host path is not also the sandbox's naming.
        with self.assertRaises(PathPolicyError):
            self.ws.resolve(str(self.primary / "here.txt"))
        opened = Workspace.from_path(self.primary, extra_roots=[self.other],
                                     allow_absolute_paths=True)
        self.assertEqual(opened.resolve(str(self.primary / "here.txt")),
                         self.primary / "here.txt")

    def test_symlink_escape_is_refused_from_every_root(self):
        (self.primary / "escape").symlink_to(self.outside, target_is_directory=True)
        (self.other / "escape").symlink_to(self.outside, target_is_directory=True)
        with self.assertRaises(PathPolicyError):
            self.ws.resolve("escape/secret.txt")
        with self.assertRaises(PathPolicyError):
            self.ws.resolve(str(self.other / "escape" / "secret.txt"))

    def test_roots_are_canonical_deduplicated_and_never_nested(self):
        # A nested or repeated root would mount the same tree twice and make
        # the label ambiguous; the primary always wins.
        ws = Workspace.from_path(
            self.primary,
            extra_roots=[self.other, self.other, self.primary / "sub", self.primary])
        self.assertEqual(ws.extra_roots, (self.other.resolve(),))
        self.assertEqual(ws.roots, (self.primary.resolve(), self.other.resolve()))

    def test_a_missing_extra_root_is_dropped_not_fatal(self):
        ws = Workspace.from_path(self.primary,
                                 extra_roots=[self.other, Path(self.temp.name) / "gone"])
        self.assertEqual(ws.extra_roots, (self.other.resolve(),))

    def test_labels_stay_unambiguous_across_roots(self):
        self.assertEqual(self.ws.relative(self.primary / "here.txt"), "here.txt")
        # A file in a secondary root is labelled by its root, never as a bare
        # relative path that would look like a primary-root file.
        self.assertEqual(self.ws.relative(self.other / "usr" / "src" / "ping.c"),
                         f"{self.other.name}:usr/src/ping.c")
        with self.assertRaises(PathPolicyError):
            self.ws.relative(self.outside / "secret.txt")

    def test_sandbox_mount_paths_are_accepted_by_the_file_tools(self):
        """One addressing scheme, not two.

        bash sees a secondary root at /workspaces/<name> while write_file acts
        on the host. Making the model juggle both is a guaranteed source of
        wrong paths, so a mount path resolves to its host root everywhere.
        """
        self.assertEqual(self.ws.resolve("/workspaces/other/usr/src/ping.c",
                                         must_exist=True),
                         self.other / "usr" / "src" / "ping.c")
        self.assertEqual(self.ws.resolve("/workspace/here.txt", must_exist=True),
                         self.primary / "here.txt")
        # Translation happens before containment, so traversal out of a mount
        # is still refused.
        with self.assertRaises(PathPolicyError):
            self.ws.resolve("/workspaces/other/../../outside/secret.txt")
        # An undeclared mount name is not a path into anything.
        with self.assertRaises(PathPolicyError):
            self.ws.resolve("/workspaces/nope/x.c")

    def test_single_root_workspace_is_unchanged(self):
        plain = Workspace.from_path(self.primary)
        self.assertEqual(plain.extra_roots, ())
        self.assertEqual(plain.roots, (self.primary.resolve(),))
        with self.assertRaises(PathPolicyError):
            plain.resolve(str(self.other / "usr" / "src" / "ping.c"))


class CommandRunnerTests(unittest.TestCase):
    def test_simple_command_uses_shell_false(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace.from_path(temporary)
            assessment = CommandPolicy().classify("pwd")
            runner = CommandRunner()
            with patch("harness.tool_runtime.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = "ok"
                run.return_value.stderr = ""
                result = runner.run_simple(assessment, workspace)
            self.assertTrue(result.ok)
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertEqual(run.call_args.args[0], ["pwd"])

    def test_network_preflight_cache_is_workspace_scoped_and_poison_aware(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = Workspace.from_path(root)
            second_root = root / "second"
            second_root.mkdir()
            second = Workspace.from_path(second_root)
            sandbox = MagicMock()
            sandbox.preflight_network.return_value = ToolResult("ok", "network ready")
            runner = CommandRunner(sandbox=sandbox)
            profile = ExecutionProfile.from_capabilities({Capability.NETWORK})
            self.assertTrue(runner.ensure_sandbox(first, profile).ok)
            self.assertTrue(runner.ensure_sandbox(first, profile).ok)
            self.assertTrue(runner.ensure_sandbox(second, profile).ok)
            self.assertEqual(sandbox.preflight_network.call_count, 2)
            sandbox.network_containment_failed = True
            blocked = runner.ensure_sandbox(second, profile)
            self.assertFalse(blocked.ok)
            self.assertIn("containment", blocked.summary)
            self.assertEqual(sandbox.preflight_network.call_count, 2)

    def test_network_preflight_cache_keeps_the_sandbox_object_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace.from_path(temporary)
            sandbox = MagicMock()
            sandbox.preflight_network.return_value = ToolResult("ok", "network ready")
            runner = CommandRunner(sandbox=sandbox)
            profile = ExecutionProfile.from_capabilities({Capability.NETWORK})
            self.assertTrue(runner.ensure_sandbox(workspace, profile).ok)
            self.assertIs(runner._network_preflight_sandbox, sandbox)
            self.assertEqual(runner._network_preflight_workspace, workspace.root)

    def test_sandboxed_runner_forwards_its_resource_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace.from_path(temporary)
            sandbox = MagicMock()
            runner = CommandRunner(sandbox=sandbox, resource_limits=DEFAULT_RESOURCE_LIMITS)
            availability = ToolResult("ok", "sandbox ready")
            sandbox.run.return_value = ToolResult("ok", "done")
            result = runner.run_sandboxed(
                workspace, ["/bin/true"], availability=availability
            )
            self.assertTrue(result.ok)
            self.assertEqual(
                sandbox.run.call_args.kwargs["resource_limits"], DEFAULT_RESOURCE_LIMITS
            )


class AuditLoggerTests(unittest.TestCase):
    def test_mutation_audit_never_records_contents_or_secrets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            root.mkdir()
            workspace = Workspace.from_path(root)
            log = Path(temporary) / "audit" / "events.jsonl"
            AuditLogger(log).record_mutation(
                action="write_file",
                mode=ExecutionMode.ASK,
                workspace=workspace,
                paths=[root / "src" / "main.py"],
                command="TOKEN=super-secret make test",
                approved=True,
                result=ToolResult("ok", "file written", changed_paths=("src/main.py",)),
            )
            event = json.loads(log.read_text().strip())
            self.assertEqual(event["paths"], ["src/main.py"])
            self.assertNotIn("super-secret", log.read_text())
            self.assertNotIn("content", event)
            self.assertEqual(event["command"], "make (1 args)")

    def test_audit_keeps_required_and_granted_capabilities_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            root.mkdir()
            log = Path(temporary) / "audit" / "events.jsonl"
            AuditLogger(log).record_mutation(
                action="command",
                mode=ExecutionMode.ASK,
                workspace=Workspace.from_path(root),
                approved=False,
                result=ToolResult("denied", "network denied"),
                required_capabilities=(Capability.FILESYSTEM_READ, Capability.NETWORK),
                granted_capabilities=(Capability.FILESYSTEM_READ,),
            )
            event = json.loads(log.read_text())
            self.assertEqual(event["required_capabilities"], ["filesystem:read", "network"])
            self.assertEqual(event["granted_capabilities"], ["filesystem:read"])


class PipefailShellTests(unittest.TestCase):
    """The shell must make pipeline failures visible without breaking dash."""

    def test_prelude_enables_pipefail_where_the_shell_supports_it(self):
        script = PIPEFAIL_PRELUDE + "false | tail -1; echo EXIT=$?"
        completed = subprocess.run([SHELL_BINARY, "-c", script],
                                   capture_output=True, text=True)
        expected = "EXIT=1" if SHELL_BINARY == "/bin/bash" else "EXIT=0"
        self.assertIn(expected, completed.stdout)

    @unittest.skipUnless(Path("/bin/dash").is_file(), "dash is not installed")
    def test_prelude_does_not_kill_a_shell_without_pipefail(self):
        # `set` is a special builtin: a bare `set -o pipefail` makes a
        # non-interactive dash EXIT, which would break every shell command on
        # a Debian-family server where /bin/sh is dash. The subshell probe is
        # what prevents that, so it is pinned here.
        completed = subprocess.run(
            ["/bin/dash", "-c", PIPEFAIL_PRELUDE + "echo alive"],
            capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("alive", completed.stdout)

    def test_shell_argv_carries_the_prelude_and_the_command(self):
        argv = shell_argv("make | tail -3")
        self.assertEqual(argv[0], SHELL_BINARY)
        self.assertEqual(argv[1], "-lc")
        self.assertTrue(argv[2].endswith("make | tail -3"))
        self.assertIn("pipefail", argv[2])
        self.assertEqual(shell_argv("x", login=False)[1], "-c")


if __name__ == "__main__":
    unittest.main()
