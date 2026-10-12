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
import shutil
import subprocess
import tempfile
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli import session_workspace, tool_routing
from context import request_scope
from harness import target_policy, write_protection
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


GEN_MARK = "/* DO NOT EDIT: generated by gen.sh */\nint x;\n"
KCONFIG = "#\n# Automatically generated file; DO NOT EDIT.\n# Linux/arm64 Kernel Configuration\n#\nCONFIG_A=y\n"
PATCH_WITH_MARK = ("--- a/configs/x_defconfig\n+++ b/configs/x_defconfig\n@@ -0,0 +1,3 @@\n"
                   "+#\n+# Automatically generated file; DO NOT EDIT.\n"
                   "+# BR2_PACKAGE_STRACE is not set\n")

# Each primitive is a list of core calls. {t} is the target as a relative
# path, {T} absolute, {d} its directory, {b} its name, {n} a fresh name.

PRIMITIVES = {
    "patch tool": [("read_file", {"path": "{t}"}),
                   ("patch", {"path": "{t}", "old_string": "{first}", "new_string": "P"})],
    "write_file": [("write_file", {"path": "{t}", "content": "P\n"})],
    "delete then recreate": [("delete_file", {"path": "{t}", "reason": "r"}),
                             ("write_file", {"path": "{t}", "content": "P\n"})],
    ">": [("terminal", {"command": "printf 'P\\n' > {t}"})],
    ">>": [("terminal", {"command": "echo P >> {t}"})],
    "x>>t": [("terminal", {"command": "echo P>>{t}"})],
    "tee": [("terminal", {"command": "echo P | tee -a {t}"})],
    "cp onto": [("terminal", {"command": "cp a.txt {t}"})],
    "cp into dir": [("terminal", {"command": "mkdir -p /tmp/{n} && cp {t} /tmp/{n}/ && "
                                             "echo P >> /tmp/{n}/{b} && cp /tmp/{n}/{b} {d}/"})],
    "mv": [("terminal", {"command": "mv a.txt {t}"})],
    "sed -i": [("terminal", {"command": "sed -i 's/^/P/' {t}"})],
    "perl -pi": [("terminal", {"command": "perl -pi -e 's/^/P/' {t}"})],
    "python3 -c": [("terminal", {"command": "python3 -c \"open('{t}','a').write('P')\""})],
    "heredoc + python": [("terminal", {"command": "cat > /tmp/{n}.py <<'EOF'\n"
                                                  "open('{T}','a').write('P')\nEOF\n"
                                                  "python3 /tmp/{n}.py"})],
    "python script": [("write_file", {"path": "{n}.py", "content": "open('{T}','a').write('P')\n"}),
                      ("terminal", {"command": "python3 {n}.py"})],
    "unittest helper": [("write_file", {"path": "tests/test_{n}.py", "content":
                         "import unittest\nclass T(unittest.TestCase):\n    def test_x(self):\n"
                         "        open('{T}','a').write('P')\n"}),
                        ("terminal", {"command": "python3 -m unittest tests.test_{n}"})],
    "patch -p1 <": [("write_file", {"path": "{n}.diff", "content": "{diff}"}),
                    ("terminal", {"command": "patch -p1 < {n}.diff"})],
    "git apply 2>&1": [("write_file", {"path": "{n}.diff", "content": "{diff}"}),
                       ("terminal", {"command": "git apply {n}.diff 2>&1"})],
    "executed script": [("read_file", {"path": "scripts/run.sh"}),
                        ("write_file", {"path": "scripts/run.sh",
                                        "content": "#!/bin/sh\necho P >> {T}\n"}),
                        ("terminal", {"command": "./scripts/run.sh"})],
    "make rule": [("terminal", {"command": "make poke T={t}"})],
    "symlink alias": [("terminal", {"command": "ln -s {T} {n}.x"}),
                      ("terminal", {"command": "echo P >> {n}.x"})],
    "cd + relative": [("terminal", {"command": "cd {d} && echo P >> {b}"})],
    "$VAR target": [("terminal", {"command": "F={t}; echo P >> $F"})],
}


@unittest.skipUnless(SANDBOX, "needs the bubblewrap sandbox")
class AProtectedTargetStaysProtected(Targets):
    """Changing the primitive never permits a mutation the policy refuses."""

    CLASSES = {"marker": ("gen/config.h", GEN_MARK),
               "kconfig output": ("linux/.config", KCONFIG),
               "generated dir": ("out/generated/table.h", "int table;\n"),
               "build/tmp": ("build/tmp/work/x/post.sh", "#!/bin/sh\n"),
               "snapshot": ("u-boot.back/board.c", "int board;\n")}

    def fill(self):
        super().fill()

        for path, text in list(self.CLASSES.values()) + [
                ("configs/x_defconfig.patch", PATCH_WITH_MARK),
                ("configs/x_defconfig", KCONFIG.replace("Linux/arm64 Kernel", "Buildroot")),
                ("scripts/run.sh", "#!/bin/sh\n")]:
            Path(self.root, path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.root, path).write_text(text)

        os.chmod(f"{self.root}/scripts/run.sh", 0o755)

        Path(self.root, "tests").mkdir()
        Path(self.root, "Makefile").write_text(
            "all:\n\t@echo built\npoke:\n\t@echo P >> $(T)\n"
            "regen:\n\t@printf '/* DO NOT EDIT */\\n' > gen/config.h\n"
            "\t@echo n > out/generated/new.h\n\t@echo n > build/tmp/work/x/new.sh\n")
        write_protection.forget()

    def run_primitive(self, target, primitive):
        """Run one primitive against `target` in a fresh session."""
        self.counter = getattr(self, "counter", 0) + 1
        lines = Path(self.root, target).read_text().splitlines()
        first = max(lines, key=len)
        diff = (f"--- a/{target}\n+++ b/{target}\n@@ -1,{len(lines)} +1,{len(lines)} @@\n"
                f"-{lines[0]}\n+P{self.counter}\n" + "".join(f" {line}\n" for line in lines[1:]))
        fields = {"t": target, "T": f"{self.root}/{target}", "d": os.path.dirname(target) or ".",
                  "b": os.path.basename(target), "n": f"h{self.counter}", "first": first,
                  "diff": diff}
        self.call = self.spear_session(self.OBJECTIVE)

        # Each route on its own: a file refused earlier in the session would
        # otherwise be protected by that refusal alone.

        target_policy.forget_refused()
        steps = [("core", name, {key: value.format(**fields) if isinstance(value, str) else value
                                 for key, value in arguments.items()})
                 for name, arguments in PRIMITIVES[primitive]]

        return steps

    def test_every_primitive_is_refused_for_every_class(self):
        for label, (target, _) in self.CLASSES.items():
            for primitive in PRIMITIVES:
                with self.subTest(target=label, primitive=primitive):
                    self.assertUnchanged(target, self.run_primitive(target, primitive))

    def test_every_terminal_primitive_is_refused_for_git_state(self):
        for target in (".git/HEAD", ".git/config", "sub2/.git", ".git/modules/sub2/HEAD"):
            for primitive, steps in PRIMITIVES.items():
                if any(name == "terminal" for name, _ in steps):
                    with self.subTest(target=target, primitive=primitive):
                        self.assertUnchanged(target, self.run_primitive(target, primitive))

    def test_a_new_file_in_a_protected_tree_is_refused(self):
        for target in ("out/generated/new.h", "build/tmp/work/x/new.sh", "u-boot.back/new.c"):
            for command in ("echo P > {t}", "touch {t}", "cp a.txt {d}/", "mkdir -p {t}",
                            "printf 'echo P > {T}\\n' > scripts/k.sh; sh scripts/k.sh",
                            "cat > scripts/w.sh <<'EOF'\necho P > {T}\nEOF\n"
                            "chmod +x scripts/w.sh; ./scripts/w.sh"):
                command = command.format(t=target, T=f"{self.root}/{target}",
                                         d=os.path.dirname(target))

                with self.subTest(target=target, command=command):
                    self.call = self.spear_session(self.OBJECTIVE)
                    self.assertUnchanged(target, [("core", "terminal", {"command": command})])

    def test_the_controls_stay_writable_through_the_same_primitives(self):
        for target in ("a.txt", "configs/x_defconfig.patch", "configs/x_defconfig"):
            for primitive in ("patch tool", ">>", "x>>t", "tee", "patch -p1 <", "executed script",
                              "$VAR target", "cd + relative", "make rule", "write_file"):
                with self.subTest(target=target, primitive=primitive):
                    before = Path(self.root, target).read_text()
                    texts = [self.call(name, **arguments)[0] for _, name, arguments
                             in self.run_primitive(target, primitive)]

                    self.assertNotEqual(Path(self.root, target).read_text(), before, texts)

    def test_the_controls_stay_writable_through_the_legacy_tools(self):
        for target, old in (("configs/x_defconfig.patch", "+# BR2_PACKAGE_STRACE is not set"),
                            ("configs/x_defconfig", "CONFIG_A=y")):
            with self.subTest(target=target):
                self.assertTrue(self.legacy("edit_file", path=target, old_text=old,
                                            new_text=old + "\n+# more").startswith("OK"))
                self.assertTrue(self.legacy("append_file", path=target,
                                            content="+BR2_X=y").startswith("OK"))

    def test_a_build_still_writes_its_output(self):
        text, record = self.call("terminal", command="make regen")

        self.assertFalse(record.refused, text)
        self.assertEqual(record.exit_code, 0, text)
        self.assertEqual(Path(self.root, "out/generated/new.h").read_text(), "n\n")
        self.assertEqual(Path(self.root, "build/tmp/work/x/new.sh").read_text(), "n\n")
        self.assertEqual(Path(self.root, "gen/config.h").read_text(), "/* DO NOT EDIT */\n")

    def test_the_legacy_tools_are_refused(self):
        for label, (target, _) in self.CLASSES.items():
            for name, arguments in (("write_file", {"content": "P\n"}),
                                    ("append_file", {"content": "P"}),
                                    ("delete_file", {"reason": "r"})):
                with self.subTest(target=label, tool=name):
                    self.assertUnchanged(target, [("legacy", name, {"path": target, **arguments})])


class ShellWriteTargets(unittest.TestCase):
    """The destinations the router judges, read the way the shell reads them."""

    def setUp(self):
        self.root = os.path.realpath(tempfile.mkdtemp())
        os.makedirs(f"{self.root}/gen")
        self.scope = request_scope.RequestScope(root=self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def targets(self, command):
        return [path[len(self.root) + 1:] if path.startswith(self.root) else path
                for path in request_scope.shell_write_targets(command, self.scope)]

    def test_a_redirection_written_against_its_word(self):
        self.assertEqual(self.targets("echo P>>gen/x.h"), ["gen/x.h"])
        self.assertEqual(self.targets("echo 'a>b' > gen/y.h"), ["gen/y.h"])
        self.assertEqual(self.targets("make 2>&1 | tail"), [])

    def test_a_variable_is_its_literal_value(self):
        self.assertEqual(self.targets("F=gen/x.h; echo P >> $F"), ["gen/x.h"])
        self.assertEqual(self.targets("export D=gen; echo P >> ${D}/x.h"), ["gen/x.h"])

    def test_a_copy_into_a_directory_writes_the_file_of_that_name(self):
        self.assertIn("gen/config.h", self.targets("cp /tmp/s/config.h gen/"))
        self.assertIn("gen/a.c", self.targets("mv a.c b.c gen"))


if __name__ == "__main__":
    unittest.main()
