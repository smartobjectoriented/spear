"""Every tool the model is offered runs, and every argument it is offered is read.

Phase 6 measured the alternative: the core was handed Hermes' whole tool
list, and three sessions ended because the model called a tool the core does
not implement (execute_code, web_search) or relied on an argument the core
silently dropped (terminal's workdir). SPEAR's own coding view had the same
defect: it offered search_corpus, which the core answers with "does not
exist".
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import dispatch, loop, tools
from tests.test_wrapper_parity import BashHost


def production_views():
    """The coding views SPEAR builds, for every role and turn shape."""
    sys.argv = sys.argv[:1]
    from cli import tool_routing
    from runtime.agent_roles import AgentRole
    from harness.tool_exposure import ToolExposurePolicy

    policy = ToolExposurePolicy()

    for role in (AgentRole.MAIN, AgentRole.EXPLORER, AgentRole.REVIEWER):
        for objective in ("fix the build so the symlink survives a clean",
                          "which file defines add, without editing files"):
            for retrieval in (True, False):
                yield policy.select(tool_routing.TOOL_REGISTRY, role, objective=objective,
                                    toolset="coding", retrieval_available=retrieval)


def definitions(view):
    return [{"type": "function", "function": {"name": item.name,
                                              "description": item.description,
                                              "parameters": dict(item.input_schema)}}
            for item in view.definitions]


class Recording(dict):
    """The arguments of one call, noting which ones the handler reads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = set()

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)


class TheProductionSurfaceIsExecutable(unittest.TestCase):
    def test_every_production_coding_view_passes_the_contract(self):
        for view in production_views():
            with self.subTest(role=view.role, names=view.names):
                dispatch.check_contract(definitions(view))

    def test_each_offered_schema_declares_exactly_what_its_handler_reads(self):
        for view in production_views():
            for item in view.definitions:
                with self.subTest(tool=item.name):
                    self.assertEqual(set(item.input_schema.get("properties", {})),
                                     set(dispatch.ARGUMENTS[item.name]))


class TheContractRefusesWhatCannotRun(unittest.TestCase):
    def schema(self, name, *arguments):
        return {"type": "function", "function": {"name": name, "description": "",
                "parameters": {"type": "object",
                               "properties": {a: {"type": "string"} for a in arguments}}}}

    def test_a_tool_without_a_handler(self):
        with self.assertRaisesRegex(dispatch.ToolContractError, "execute_code: no handler"):
            dispatch.check_contract([self.schema("terminal", "command"),
                                     self.schema("execute_code", "code")])

    def test_an_argument_no_handler_reads(self):
        with self.assertRaisesRegex(dispatch.ToolContractError,
                                    "terminal: arguments no handler reads: background, pty"):
            dispatch.check_contract([self.schema("terminal", "command", "background", "pty")])

    def test_the_loop_refuses_before_the_first_request(self):
        asked = []

        with self.assertRaises(dispatch.ToolContractError):
            loop.run(model=lambda *a: asked.append(a), host=None, system="", history=[],
                     request="hi", tool_definitions=[self.schema("web_search", "query")],
                     max_iterations=1, max_tool_calls=1, context_window=0)

        self.assertEqual(asked, [])


class EachHandlerReadsEveryDeclaredArgument(unittest.TestCase):
    """Called with every argument it declares, each handler reads them all."""

    CALLS = {
        "read_file": {"path": "a.txt", "offset": 1, "limit": 10},
        "search_files": {"pattern": "x", "target": "content", "path": ".", "file_glob": "*.txt",
                         "limit": 5, "offset": 0, "output_mode": "content", "context": 0},
        "patch": {"path": "a.txt", "old_string": "x", "new_string": "y", "replace_all": False},
        "write_file": {"path": "b.txt", "content": "z\n"},
        "delete_file": {"path": "c.txt", "reason": "obsolete"},
        "terminal": {"command": "pwd", "timeout": 5, "workdir": "sub"},
    }

    def setUp(self):
        self.root = os.path.realpath(tempfile.mkdtemp())
        os.makedirs(os.path.join(self.root, "sub"))
        for name in ("a.txt", "c.txt"):
            Path(self.root, name).write_text("x\n")
        self.host = BashHost(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_handlers(self):
        self.assertEqual(set(self.CALLS), set(dispatch.HANDLERS))

        for name, arguments in self.CALLS.items():
            with self.subTest(tool=name):
                self.assertEqual(set(arguments), set(dispatch.ARGUMENTS[name]))
                recording = Recording(arguments)
                dispatch.HANDLERS[name](self.host, tools.new_state(), "c", recording)
                self.assertEqual(set(arguments) - recording.read, set())


if __name__ == "__main__":
    unittest.main()
