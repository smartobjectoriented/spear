"""What a prompt of Claude Code is given: the text of a SPEAR turn, by a hook."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli import claude_context  # noqa: E402
from context import workspace_knowledge  # noqa: E402


class FakeIndex:
    default = "alpha"

    def __init__(self, context="# File: src/a.c\nint a;"):
        self.asked, self._context = [], context

    def names(self):
        return ["alpha"]

    def context(self, query, name):
        self.asked.append((query, name))

        return self._context


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "knowledge.sqlite3")
        self.projects = {"alpha": {"path": self.dir}}
        patch = mock.patch.dict(os.environ, {"SPEAR_KNOWLEDGE_DB": self.db})
        patch.start()
        self.addCleanup(patch.stop)

    def record(self, workspace_id, statement):
        store = workspace_knowledge.KnowledgeStore(self.db)
        store.add(workspace_id, kind=workspace_knowledge.Kind.PROJECT_FACT, subject="modem",
                  statement=statement, provenance=workspace_knowledge.Provenance.USER_CONFIRMED)
        store.close()


class WhatAPromptIsShown(Base):
    def test_the_passages_retrieved_for_it_and_the_workspaces_knowledge(self):
        self.record("alpha", "The modem is on UART2.")
        index = FakeIndex()

        text = claude_context.compose(index, "where is a?", "alpha", self.dir, self.projects)

        self.assertEqual(index.asked, [("where is a?", "alpha")])
        self.assertIn(claude_context.RETRIEVED_HEADER, text)
        self.assertIn("# File: src/a.c", text)
        self.assertIn("## Workspace knowledge", text)
        self.assertIn("The modem is on UART2.", text)

    def test_nothing_from_a_corpus_that_has_no_index(self):
        index = FakeIndex()

        self.assertEqual(
            claude_context.compose(index, "q", "beta", self.dir, {"beta": {"path": self.dir}}),
            "")
        self.assertEqual(index.asked, [])

    def test_a_tree_nobody_registered_is_known_by_where_it_is(self):
        root = os.path.realpath(self.dir)
        self.record(f"adhoc:{root}", "Built with ninja.")

        text = claude_context.compose(None, "how is it built?", None, self.dir, {})

        self.assertIn("Built with ninja.", text)
        self.assertNotIn(claude_context.RETRIEVED_HEADER, text)

    def test_a_search_that_fails_leaves_the_knowledge(self):
        self.record("alpha", "The modem is on UART2.")
        index = FakeIndex()
        index.context = mock.Mock(side_effect=RuntimeError("index gone"))

        with mock.patch("sys.stderr", io.StringIO()):
            text = claude_context.compose(index, "q", "alpha", self.dir, self.projects)

        self.assertIn("The modem is on UART2.", text)


class TheHook(Base):
    def run_hook(self, argv, event):
        out = io.StringIO()
        code = claude_context.hook(argv, io.StringIO(json.dumps(event)), out)

        return code, out.getvalue()

    def test_it_asks_the_sessions_server_and_prints_its_answer(self):
        path = os.path.join(self.dir, "socket")
        index = FakeIndex()
        listener = claude_context.serve(index, path, self.projects)
        self.addCleanup(listener.close)

        code, text = self.run_hook(["--socket", path, "--corpus", "alpha"],
                                   {"prompt": "where is a?", "cwd": self.dir})

        self.assertEqual(code, 0)
        self.assertIn("# File: src/a.c", text)
        self.assertIn(("where is a?", "alpha"), index.asked)
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)

    def test_without_a_server_it_does_the_work_itself(self):
        with mock.patch.object(claude_context, "_alone", return_value="the context") as alone:
            code, text = self.run_hook(["--socket", os.path.join(self.dir, "nobody"),
                                        "--corpus", "alpha"],
                                       {"prompt": "q", "cwd": self.dir})

        self.assertEqual((code, text), (0, "the context\n"))
        alone.assert_called_once_with("q", "alpha", self.dir)

    def test_it_never_fails_a_prompt(self):
        with mock.patch.object(claude_context, "_alone", side_effect=RuntimeError("broken")), \
                mock.patch("sys.stderr", io.StringIO()):
            self.assertEqual(self.run_hook([], {"prompt": "q", "cwd": self.dir}), (0, ""))

        self.assertEqual(self.run_hook([], {"prompt": "  "}), (0, ""))
        self.assertEqual(claude_context.hook([], io.StringIO("not json"), io.StringIO()), 0)


class TheKnowledgeCommand(Base):
    def test_it_reads_and_proposes_for_the_tree_it_is_run_in(self):
        root = os.path.realpath(self.dir)
        self.record(f"adhoc:{root}", "Built with ninja.")
        out = io.StringIO()

        with mock.patch("os.getcwd", return_value=root), \
                mock.patch("cli.corpus_registry.load_projects", return_value={}):
            listed = claude_context.knowledge_command(["list"], out)
            proposed = claude_context.knowledge_command(
                ["propose", json.dumps({"subject": "flash", "statement": "Flashed over USB."})],
                out)

        self.assertEqual((listed, proposed), (0, 0))
        self.assertIn("Built with ninja.", out.getvalue())
        self.assertIn("proposed", out.getvalue())


if __name__ == "__main__":
    unittest.main()
