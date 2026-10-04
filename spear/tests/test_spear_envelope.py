"""SPEAR's control plane around the frozen agent core: transparent when it
allows, firm when it refuses, and the only way through.

Every case runs the core's own dispatcher against two hosts on the same
disposable workspace, at the same path: a permissive reference host (the
Phase-7 benchmark host: plain `bash -c`, no policy) and SpearHost as
production builds it (rag_chat.coding_host: router, workspace policy, command
policy, sandbox, audit). An allowed call must read the same through both; a
refused one must say so in the core's own vocabulary and leave nothing
behind.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import completion
import rag_chat
from agent import dispatch, tools
from agent.host import CommandOutcome
from cancellation import NEVER_CANCELLED
from tool_runtime import ExecutionMode, Workspace, decode_command_output

NAMES = ("read_file", "search_files", "patch", "write_file", "delete_file", "terminal")
SANDBOX = shutil.which("bwrap") is not None


class ReferenceHost:
    """The Phase-7 benchmark host: no policy, commands run by plain bash."""

    collation_locale = "en_US.UTF-8"

    def __init__(self, root):
        self.workspace_root = root

    def authorize(self, name, arguments):
        return None

    def resolve_read(self, path):
        return os.path.realpath(os.path.join(self.workspace_root, path)), None

    resolve_write = resolve_workdir = resolve_read

    def write_file(self, path, content, *, action):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as handle:
            handle.write(content)

    def delete_file(self, path, reason):
        os.unlink(path)

    def run_command(self, command, script, *, timeout, output_chars):
        env = {key: value for key, value in os.environ.items()
               if key not in ("AI_AGENT", "HERMES_AGENT")}
        done = subprocess.run(["bash", "-c", script], cwd=self.workspace_root,
                              capture_output=True, timeout=timeout, env=env,
                              stdin=subprocess.DEVNULL)

        return CommandOutcome("ok", (decode_command_output(done.stdout)
                                     + decode_command_output(done.stderr))[:output_chars],
                              done.returncode)

    def after_tool(self, record):
        pass


class Envelope(unittest.TestCase):
    """A disposable workspace, the two hosts, and a way to call the core."""

    OBJECTIVE = "Fix board/virt64/post_image.sh so it also copies the dtb"

    def setUp(self):
        self.root = os.path.realpath(tempfile.mkdtemp(prefix="envelope-", dir=str(ROOT)))
        self.records = []
        self.fill()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def fill(self):
        for plat in ("virt64", "rpi4", "virt32"):
            os.makedirs(f"{self.root}/board/{plat}")
            Path(f"{self.root}/board/{plat}/post_image.sh").write_text("#!/bin/sh\n")

        Path(self.root, "board/virt64/initrd.cpio").write_bytes(b"070701" + b"0" * 64)
        os.makedirs(f"{self.root}/sub/deep")
        Path(self.root, "a.txt").write_text("alpha\nbeta\ngamma\n")
        Path(self.root, "sub/b.c").write_text("int main(void) { return 0; }\n")
        Path(self.root, "Makefile").write_text("all:\n\t@echo built\n")

    def spear(self, objective=None, **overrides):
        context = SimpleNamespace(
            task_id="task_envelope", cancellation=NEVER_CANCELLED, work_phase=None,
            checkpoint_manager=None, checkpoint=None, role="main", read_only=False,
            advisory=False, standard_binding=None, execution_core="coding",
            conversation=[], dropped_tool_results=frozenset(),
            working_state=SimpleNamespace(objective=objective or self.OBJECTIVE))

        for key, value in overrides.items():
            setattr(context, key, value)

        return rag_chat.coding_host(context, {}, self.records.append)

    def session(self, host):
        """A call function bound to `host` and a fresh core session."""
        state = tools.new_state()

        def call(name, **arguments):
            with patch.object(rag_chat, "PROJECT_ROOT", self.root), \
                    patch.object(rag_chat, "WORKSPACE", Workspace.from_path(self.root)), \
                    patch.object(rag_chat, "EXECUTION_MODE", ExecutionMode.AUTO), \
                    redirect_stdout(io.StringIO()):
                return dispatch.execute(host, state, f"c{len(self.records)}", name,
                                        arguments, NAMES)

        return call

    def spear_session(self, objective=None, **overrides):
        with patch.object(rag_chat, "PROJECT_ROOT", self.root), \
                patch.object(rag_chat, "WORKSPACE", Workspace.from_path(self.root)):
            return self.session(self.spear(objective, **overrides))

    def assertRefused(self, outcome):
        text, record = outcome
        self.assertTrue(record.refused, text)
        self.assertFalse(record.changed_paths)
        error = json.loads(text)["error"]
        self.assertTrue(error.startswith("refused: "), error)

        # The core has no edit_file, no scope_reason, and does have delete_file.
        for legacy in ("edit_file", "scope_reason", "old_text", "no tool can delete"):
            self.assertNotIn(legacy, error)

        return error


@unittest.skipUnless(SANDBOX, "needs the bubblewrap sandbox")
class AllowedCallsAreTransparent(Envelope):
    """Items 1-6, 10 and 14-15: what the model reads is the reference host's."""

    STEPS = [
        ("read_file", {"path": "a.txt"}),
        ("read_file", {"path": "a.txt", "offset": 2, "limit": 1}),
        ("read_file", {"path": "missing.txt"}),
        ("search_files", {"pattern": "beta"}),
        ("search_files", {"pattern": "*.c", "target": "files"}),
        ("search_files", {"pattern": "main", "output_mode": "count"}),
        ("patch", {"path": "board/virt64/post_image.sh", "old_string": "#!/bin/sh",
                   "new_string": "#!/bin/bash"}),
        ("patch", {"path": "a.txt", "old_string": "nothere", "new_string": "x"}),
        ("write_file", {"path": "board/virt64/notes/c.txt", "content": "c\n"}),
        ("delete_file", {"path": "board/virt64/notes/c.txt", "reason": "temporary"}),
        ("terminal", {"command": "pwd"}),
        ("terminal", {"command": "ls"}),
        ("terminal", {"command": "cat a.txt"}),
        ("terminal", {"command": "false"}),
        ("terminal", {"command": "ls nothere"}),
        ("terminal", {"command": "nosuchcmd"}),
        ("terminal", {"command": "cd sub && pwd"}),
        ("terminal", {"command": "export Q=7"}),
        ("terminal", {"command": "echo $Q; ls"}),
        ("terminal", {"command": "pwd", "workdir": "deep"}),
        ("terminal", {"command": "pwd", "workdir": "<ROOT>/board"}),
        ("terminal", {"command": "cd .."}),
        ("terminal", {"command": "make"}),
        ("terminal", {"command": "mkdir -p images && ln -sf ../board/virt64/initrd.cpio "
                                 "images/initrd.cpio && readlink images/initrd.cpio"}),
        ("terminal", {"command": "ln -sf ../board/virt64/initrd.cpio images/second"}),
        ("terminal", {"command": "sleep 1; echo slept"}),
    ]

    def run_steps(self, call):
        results = []

        for name, arguments in self.STEPS:
            arguments = {key: value.replace("<ROOT>", self.root) if isinstance(value, str)
                         else value for key, value in arguments.items()}
            text, record = call(name, **arguments)
            results.append((name, arguments, text, record.refused))

        return results

    def test_every_allowed_call_reads_as_through_the_reference_host(self):
        reference = self.run_steps(self.session(ReferenceHost(self.root)))
        shutil.rmtree(self.root)
        os.makedirs(self.root)
        self.fill()
        spear = self.run_steps(self.spear_session())

        for (name, arguments, expected, _), (_, _, actual, refused) in zip(reference, spear):
            with self.subTest(tool=name, arguments=arguments):
                self.assertFalse(refused, actual)
                self.assertEqual(actual, expected)


@unittest.skipUnless(SANDBOX, "needs the bubblewrap sandbox")
class TheShellHasNoMoreReachThanPatch(Envelope):
    """Items 7-8 and the shell replay: the named target, and not its siblings."""

    def test_a_shell_write_to_the_named_target_runs(self):
        call = self.spear_session()
        text, record = call("terminal", command="echo cp >> board/virt64/post_image.sh")

        self.assertFalse(record.refused, text)
        self.assertIn("cp", Path(self.root, "board/virt64/post_image.sh").read_text())

    def test_a_sibling_is_refused_however_the_path_is_reached(self):
        call = self.spear_session()
        cases = [
            ("patch", {"path": "board/rpi4/post_image.sh", "old_string": "#!/bin/sh",
                       "new_string": "#!/bin/bash"}),
            ("write_file", {"path": "board/rpi4/post_image.sh", "content": "x\n"}),
            ("terminal", {"command": "echo x >> board/rpi4/post_image.sh"}),
            ("terminal", {"command": "cd board && echo x >> rpi4/post_image.sh"}),
            ("terminal", {"command": "echo x >> rpi4/post_image.sh",
                          "workdir": f"{self.root}/board"}),
            ("terminal", {"command": "echo x >> post_image.sh", "workdir": "board/rpi4"}),
            ("terminal", {"command": "cp board/virt64/post_image.sh board/rpi4/"}),
            ("terminal", {"command": "mv board/virt64/post_image.sh board/rpi4/post_image.sh"}),
            ("terminal", {"command": "tee board/rpi4/post_image.sh < a.txt"}),
            ("terminal", {"command": "sed -i s/sh/bash/ board/rpi4/post_image.sh"}),
            ("terminal", {"command": "dd if=a.txt of=board/rpi4/post_image.sh"}),
            ("terminal", {"command": "touch board/rpi4/new"}),
            ("terminal", {"command": "python3 -c \"open('board/rpi4/post_image.sh','w')\""}),
            ("terminal", {"command": "find board -name post_image.sh | xargs sed -i s/a/b/"}),
            ("terminal", {"command": "for f in board/*/post_image.sh; do echo x >> $f; done"}),
        ]

        for name, arguments in cases:
            with self.subTest(tool=name, arguments=arguments):
                self.assertRefused(call(name, **arguments))

        self.assertEqual(Path(self.root, "board/rpi4/post_image.sh").read_text(), "#!/bin/sh\n")

    def test_a_cd_earlier_in_the_session_does_not_open_a_sibling(self):
        call = self.spear_session()
        text, record = call("terminal", command="cd board")

        self.assertFalse(record.refused, text)
        self.assertRefused(call("terminal", command="echo x >> rpi4/post_image.sh"))
        self.assertEqual(Path(self.root, "board/rpi4/post_image.sh").read_text(), "#!/bin/sh\n")

    def test_the_whole_family_opens_when_the_user_asks_for_it(self):
        call = self.spear_session("Fix post_image.sh on all platforms")
        text, record = call("terminal", command="echo x >> board/rpi4/post_image.sh")

        self.assertFalse(record.refused, text)


@unittest.skipUnless(SANDBOX, "needs the bubblewrap sandbox")
class NothingLeavesTheWorkspace(Envelope):
    """Items 9, 11 and 12: no write, workdir or symlink reaches outside."""

    def setUp(self):
        super().setUp()
        self.outside = tempfile.mkdtemp(prefix="envelope-outside-")

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.outside, ignore_errors=True)

    def test_direct_writes_outside_are_refused(self):
        call = self.spear_session()

        self.assertRefused(call("write_file", path=f"{self.outside}/x.txt", content="x"))
        self.assertRefused(call("patch", path="/etc/hostname", old_string="a", new_string="b"))
        self.assertRefused(call("delete_file", path=f"{self.outside}/x.txt", reason="x"))
        self.assertEqual(os.listdir(self.outside), [])

    def test_a_workdir_outside_is_refused_and_inside_runs(self):
        call = self.spear_session()

        for workdir in ("/", self.outside, "../.."):
            with self.subTest(workdir=workdir):
                text, record = call("terminal", command="pwd", workdir=workdir)
                self.assertTrue(record.refused, text)
                self.assertEqual(json.loads(text)["status"], "blocked")

        text, _ = call("terminal", command="pwd", workdir="board/virt64")
        self.assertEqual(json.loads(text)["output"], f"{self.root}/board/virt64")

    def test_a_symlink_out_of_the_workspace_is_not_a_way_out(self):
        os.symlink(self.outside, os.path.join(self.root, "escape"))
        call = self.spear_session("write escape/x.txt")

        self.assertRefused(call("write_file", path="escape/x.txt", content="x"))
        call("terminal", command="echo x > escape/x.txt")
        call("terminal", command="cd escape && touch y")
        self.assertRefused(call("terminal", command="pwd", workdir="escape"))
        self.assertEqual(os.listdir(self.outside), [])


@unittest.skipUnless(SANDBOX, "needs the bubblewrap sandbox")
class AnAdvisoryTurnChangesNothing(Envelope):
    """Item 13: a question about a change cannot make it, by any tool."""

    def test_every_way_to_write_is_refused(self):
        call = self.spear_session("Could the dtb be copied in board/virt64/post_image.sh?",
                                  read_only=True, advisory=True)

        for name, arguments in (
                ("patch", {"path": "board/virt64/post_image.sh", "old_string": "#!/bin/sh",
                           "new_string": "#!/bin/bash"}),
                ("write_file", {"path": "board/virt64/new.txt", "content": "x"}),
                ("delete_file", {"path": "a.txt", "reason": "x"})):
            with self.subTest(tool=name):
                self.assertRefused(call(name, **arguments))

        call("terminal", command="echo x >> board/virt64/post_image.sh")
        call("terminal", command="mkdir -p made")

        self.assertEqual(Path(self.root, "board/virt64/post_image.sh").read_text(), "#!/bin/sh\n")
        self.assertTrue(Path(self.root, "a.txt").exists())
        self.assertFalse(Path(self.root, "made").exists())

        text, record = call("read_file", path="board/virt64/post_image.sh")
        self.assertFalse(record.refused, text)


@unittest.skipUnless(SANDBOX, "needs the bubblewrap sandbox")
class TheRecordIsTheEvidence(Envelope):
    """Items 16-17: refusals are recorded, never counted, and the verdict
    rewrites a claim the record does not support."""

    def test_a_refusal_is_recorded_and_is_neither_a_change_nor_a_check(self):
        call = self.spear_session()
        call("patch", path="board/rpi4/post_image.sh", old_string="#!/bin/sh",
             new_string="#!/bin/bash")
        call("terminal", command="cd board && echo x >> rpi4/post_image.sh")
        evidence = [completion.evidence(item, lambda path: path) for item in self.records]

        self.assertEqual([item.refused for item in self.records], [True, True])
        self.assertEqual(completion.decide(evidence).state, "NO_CHANGE")

    def test_a_change_with_no_check_is_unverified_and_the_claim_is_qualified(self):
        call = self.spear_session()
        call("patch", path="board/virt64/post_image.sh", old_string="#!/bin/sh",
             new_string="#!/bin/bash")
        evidence = [completion.evidence(item, lambda path: path) for item in self.records]
        verdict = completion.decide(evidence)
        answer = completion.qualify("Done. It now survives a clean build.", verdict)

        self.assertEqual(verdict.state, "UNVERIFIED")
        self.assertTrue(answer.startswith("**UNVERIFIED**"))
        self.assertIn("It now survives a clean build. *(not verified)*", answer)

    def test_a_change_the_turn_then_ran_is_verified_and_untouched(self):
        call = self.spear_session()
        call("patch", path="board/virt64/post_image.sh", old_string="#!/bin/sh",
             new_string="#!/bin/bash")
        call("terminal", command="make")
        evidence = [completion.evidence(item, lambda path: path) for item in self.records]
        verdict = completion.decide(evidence)

        self.assertEqual(verdict.state, "VERIFIED")
        self.assertEqual(completion.qualify("Done.", verdict), "Done.")


class TheVocabularyTableMatchesItsSources(unittest.TestCase):
    """Every legacy phrase the core's refusals are translated from still exists
    where SPEAR writes it -- a reworded source must not silently stop being
    translated."""

    def test_each_phrase_is_still_in_the_source(self):
        import control_plane

        sources = "".join(Path(ROOT, name).read_text()
                          for name in ("request_scope.py", "tool_runtime.py"))
        sources = sources.replace('"\n', "").replace('f"', "").replace('    "', "")
        sources = " ".join(sources.split())

        for pattern, _ in control_plane.CORE_VOCABULARY:
            literal = pattern.pattern.replace("\\", "")
            probe = literal.split(".*?")[0].strip()[:40]

            with self.subTest(phrase=probe):
                self.assertIn(probe, sources)


if __name__ == "__main__":
    unittest.main()
