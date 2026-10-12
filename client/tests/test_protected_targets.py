"""A protected target stays protected whatever primitive reaches it.

The coding core's file tools, the legacy native tools and the terminal are
each driven against the same targets in a disposable tree. A case passes when
the target is unchanged: refused by the policy, or by the kernel when the
command ran in the sandbox with the target bound read-only. The controls
beside each class must stay writable through the same primitives.
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli import session_workspace, tool_routing
from harness import write_protection
from harness.tool_primitives import ExecutionMode
from harness.workspace import Workspace
from tests.test_spear_envelope import SANDBOX, Envelope


def fingerprint(path):
    """The bytes under `path` (a file, a directory, a link), or its absence."""
    path = Path(path)
    digest = hashlib.sha256()

    if path.is_symlink():
        return "link:" + os.readlink(path)

    if not path.exists():
        return "absent"

    for item in ([path] if path.is_file() else sorted(path.rglob("*"))):
        digest.update(str(item.relative_to(path.parent)).encode())

        if item.is_file() and not item.is_symlink():
            digest.update(item.read_bytes())

    return digest.hexdigest()


class Targets(Envelope):
    """The disposable tree, the core and the legacy tools against it."""

    OBJECTIVE = "Change the default hostname of the board configuration"

    def fill(self):
        super().fill()
        git = ["git", "-C", self.root, "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run(git + ["init", "-q"], check=True)
        subprocess.run(git + ["add", "-A"], check=True)
        subprocess.run(git + ["commit", "-qm", "base"], check=True)

        # A submodule-shaped checkout: sub2/.git is a FILE naming its gitdir.

        os.makedirs(f"{self.root}/.git/modules")
        subprocess.run(["git", "init", "-q", "--separate-git-dir",
                        f"{self.root}/.git/modules/sub2", f"{self.root}/sub2"], check=True)
        Path(self.root, "sub2/s.txt").write_text("s\n")
        write_protection.forget()

    def setUp(self):
        super().setUp()
        self.call = self.spear_session(self.OBJECTIVE)

    def git_object(self):
        return next(str(item.relative_to(self.root))
                    for item in sorted(Path(self.root, ".git/objects").glob("??/*")))

    def legacy(self, name, **arguments):
        """A native tool through the production router, as the legacy core runs it."""
        context = SimpleNamespace(
            task_id="task_legacy", trace=None, cancellation=None, work_phase=None,
            checkpoint_manager=None, checkpoint=None, role="main", read_only=False,
            advisory=False, standard_binding=None, execution_core="legacy",
            conversation=[], dropped_tool_results=frozenset(), session=None,
            working_state=SimpleNamespace(objective=self.OBJECTIVE))

        with patch.object(session_workspace, "PROJECT_ROOT", self.root), \
                patch.object(session_workspace, "WORKSPACE", Workspace.from_path(self.root)), \
                patch.object(session_workspace, "EXECUTION_MODE", ExecutionMode.AUTO), \
                redirect_stdout(io.StringIO()):
            return tool_routing.route_tool_envelope(name, dict(arguments), {},
                                                    agent_context=context).text

    def assertUnchanged(self, target, steps):
        """Run `steps` (core calls or legacy ones) and require `target` intact."""
        before = fingerprint(Path(self.root, target))
        texts = []

        for kind, name, arguments in steps:
            if kind == "legacy":
                texts.append(self.legacy(name, **arguments))
            else:
                texts.append(self.call(name, **arguments)[0])

        self.assertEqual(fingerprint(Path(self.root, target)), before, texts)

        return " | ".join(texts)


@unittest.skipUnless(SANDBOX, "needs the bubblewrap sandbox")
class GitMetadataIsNeverMutated(Targets):
    """No model mutation path changes a repository's own state."""

    TERMINAL = ("echo P > {t}", "echo P >> {t}", "echo P>>{t}", "cp a.txt {t}", "mv a.txt {t}",
                "rm -f {t}", "sed -i s/^/P/ {t}", "python3 -c \"open('{t}','a').write('P')\"",
                "cat > /tmp/p.py <<'EOF'\nopen('{T}','a').write('P')\nEOF\npython3 /tmp/p.py",
                "sh poke.sh", "chmod +x poke.sh; ./poke.sh", "make poke T={t}",
                "echo P >> sub/../{t}")

    def targets(self):
        return (".git/index", ".git/HEAD", ".git/config", self.git_object(), "sub2/.git",
                ".git/modules/sub2/HEAD")

    def test_the_file_tools_are_refused(self):
        for target in self.targets():
            for name, arguments in (("write_file", {"path": target, "content": "P\n"}),
                                    ("delete_file", {"path": target, "reason": "r"})):
                with self.subTest(target=target, tool=name):
                    text = self.assertUnchanged(target, [("core", name, arguments)])

                    self.assertIn("git metadata", text)

        for target in (".git/HEAD", ".git/config", "sub2/.git"):
            with self.subTest(target=target, tool="patch"):
                first = Path(self.root, target).read_text().splitlines()[0]
                text = self.assertUnchanged(target, [
                    ("core", "read_file", {"path": target}),
                    ("core", "patch", {"path": target, "old_string": first,
                                       "new_string": "P"})])

                self.assertIn("git metadata", text)

    def test_the_legacy_tools_are_refused(self):
        for target in (".git/HEAD", ".git/config", "sub2/.git", ".git/modules/sub2/HEAD"):
            first = Path(self.root, target).read_text().splitlines()[0]

            for name, arguments in (("edit_file", {"old_text": first, "new_text": "P"}),
                                    ("write_file", {"content": "P\n"}),
                                    ("append_file", {"content": "P"}),
                                    ("delete_file", {"reason": "r"})):
                with self.subTest(target=target, tool=name):
                    text = self.assertUnchanged(
                        target, [("legacy", name, {"path": target, **arguments})])

                    self.assertIn("git metadata", text)

        for name in ("append_file", "delete_file"):
            with self.subTest(target=".git/index", tool=name):
                self.assertUnchanged(".git/index", [
                    ("legacy", name, {"path": ".git/index", "content": "P", "reason": "r"})])

    def test_no_terminal_command_changes_it(self):
        Path(self.root, "Makefile").write_text("all:\n\t@echo built\npoke:\n\t@echo P >> $(T)\n")

        for target in self.targets():
            Path(self.root, "poke.sh").write_text(f"#!/bin/sh\necho P >> {self.root}/{target}\n")

            for route in self.TERMINAL:
                command = route.format(t=target, T=f"{self.root}/{target}")

                with self.subTest(target=target, command=command):
                    self.assertUnchanged(target, [("core", "terminal", {"command": command})])

    def test_git_commands_that_write_it_fail(self):
        for command in ("echo more >> a.txt; git add a.txt 2>&1",
                        "echo more >> a.txt; git -c user.name=t -c user.email=t@t commit -qam m 2>&1",
                        "echo more >> a.txt; git stash 2>&1",
                        "echo x > a.txt; git config core.foo bar 2>&1"):
            with self.subTest(command=command):
                self.assertUnchanged(".git", [("core", "terminal", {"command": command})])

    def test_ordinary_files_beside_it_stay_writable(self):
        for name, arguments in (("write_file", {"path": "new.c", "content": "int n;\n"}),
                                ("terminal", {"command": "echo more >> a.txt"}),
                                ("terminal", {"command": "echo s >> sub2/s.txt"}),
                                ("terminal", {"command": "cp a.txt b.txt"}),
                                ("write_file", {"path": ".gitignore", "content": "*.o\n"})):
            with self.subTest(tool=name, arguments=arguments):
                text, record = self.call(name, **arguments)

                self.assertFalse(record.refused, text)

        self.assertIn("more", Path(self.root, "a.txt").read_text())
        self.assertTrue(Path(self.root, "b.txt").is_file())
        self.assertIn("OK", self.legacy("append_file", path="a.txt", content="tail"))

    def test_read_only_git_still_works(self):
        Path(self.root, "a.txt").write_text("changed\n")

        for command in ("git status --short", "git log --oneline", "git diff",
                        "git show --stat HEAD", "cd sub2 && git status --short"):
            with self.subTest(command=command):
                text, record = self.call("terminal", command=command)

                self.assertFalse(record.refused, text)
                self.assertEqual(record.exit_code, 0, text)


if __name__ == "__main__":
    unittest.main()
