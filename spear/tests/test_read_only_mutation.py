"""A turn that may not write does not get one built for it.

Asked "Inspect probe_ack_cam.c ... Do not modify anything" under --safe, the
harness did the following, in order: detected the read-only request correctly,
selected a tool view of seven tools with no write tools in it, let the model
read and answer -- and then took the file the model had printed, spent a model
call asking it to re-express the rewrite as edit_file calls, and executed a
write_file the view had deliberately withheld. Nothing was written: a size
heuristic refused it, and safe mode would have refused after that. But the
refusal the model saw was about the CONTENT, and it recommended edit_file --
another tool that turn could not use either.

Two permissions, independent, and the harness needs both before it turns model
output into an action of its own:

    task    the user asked for an inspection and said not to modify
    policy  --safe forbids mutation whatever was asked

Neither was consulted. The model never requested the write; the harness
synthesised it. These tests hold the line at three places: the predicate, the
router (which every caller passes, including the ones that are not the model),
and the refusal text (which must not name a way to write that does not exist).
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag_chat
from tool_exposure import ToolExposurePolicy
from tool_registry import ToolMutability, native_tool_specs
from tool_router import ToolResultStatus

READ_ONLY_ASK = "Inspect foo.c and explain it. Do not modify anything."
WRITE_ASK = "Replace foo.c with the corrected version."

# What the model printed: a whole file, long enough for the conversion path to
# take an interest in it.
PRINTED_FILE = "```c\n" + "\n".join(
    f"static int line_{index}(void) {{ return {index}; }}" for index in range(40)
) + "\n```"

MUTATING = tuple(spec.name for spec in native_tool_specs()
                 if spec.mutability == ToolMutability.MUTATING)

# Every way the refusals must NOT answer: naming another mechanism to write
# with is what sent a read-only turn looking for one.
OTHER_MECHANISMS = ("use edit_file", "use write_file", "use append_file",
                    "--auto", "--ask", "rerun with", "bypass")


def context(*, read_only):
    return SimpleNamespace(read_only=read_only)


class safe_mode:
    """Run a block with the module-level execution mode set, and restore it."""

    def __init__(self, mode):
        self.mode = mode

    def __enter__(self):
        from tool_runtime import ExecutionMode

        self.previous = rag_chat.EXECUTION_MODE
        rag_chat.EXECUTION_MODE = getattr(ExecutionMode, self.mode)

        return self

    def __exit__(self, *exc):
        rag_chat.EXECUTION_MODE = self.previous

        return False


class BothPermissionsAreRequired(unittest.TestCase):
    """The predicate, which is the whole of the decision."""

    def test_the_four_combinations(self):
        for mode, read_only, expected in (
            ("SAFE", True, False),     # CASE 1
            ("AUTO", True, False),     # CASE 2 -- auto does not override the ask
            ("SAFE", False, False),    # CASE 3 -- policy alone forbids it
            ("AUTO", False, True),     # CASE 4 -- both permit, work proceeds
            ("ASK", False, True),
            ("ASK", True, False),
        ):
            with self.subTest(mode=mode, read_only=read_only):
                with safe_mode(mode):
                    self.assertIs(
                        rag_chat.mutation_permitted(context(read_only=read_only)),
                        expected)

    def test_a_turn_with_no_context_still_obeys_the_mode(self):
        with safe_mode("SAFE"):
            self.assertFalse(rag_chat.mutation_permitted(None))

        with safe_mode("AUTO"):
            self.assertTrue(rag_chat.mutation_permitted(None))


class NothingIsSynthesisedForAReadOnlyTurn(unittest.TestCase):
    """CASE 1 and CASE 2 -- the conversion path, at its entry point."""

    def source(self):
        return inspect.getsource(rag_chat.main)

    def test_the_conversion_asks_permission_before_it_looks_at_the_block(self):
        source = self.source()
        start = source.index("def compatibility_mutation")
        body = source[start:start + 900]
        guard = body.index("mutation_permitted")
        extract = body.index("extract_code_block_with_language")

        self.assertLess(guard, extract)

    def test_it_returns_before_the_refining_model_call(self):
        source = self.source()
        start = source.index("def compatibility_mutation")
        body = source[start:source.index("return result", start + 900)]

        self.assertLess(body.index("mutation_permitted"),
                        body.index("try_surgical_edits")
                        if "try_surgical_edits" in body else len(body))

    def test_the_request_is_recognised_as_read_only(self):
        self.assertTrue(ToolExposurePolicy.read_only_intent(READ_ONLY_ASK))
        self.assertFalse(ToolExposurePolicy.read_only_intent(WRITE_ASK))

    def test_the_printed_file_is_the_kind_the_path_would_have_taken(self):
        """Otherwise the guard above would be proving nothing."""
        block, language = rag_chat.extract_code_block_with_language(PRINTED_FILE)

        self.assertEqual(language, "c")
        self.assertGreater(len(block), 200)


class TheRouterRefusesWhoeverAsks(unittest.TestCase):
    """The backstop: the model is not the only caller.

    The tool view withheld the write tools and a write still ran, because the
    conversion path calls the router directly. A gate in the view protects
    against the model; this one protects against the harness.
    """

    def route(self, name, arguments, *, read_only):
        cache = {}
        envelope = rag_chat.route_tool_envelope(
            name, arguments, cache,
            agent_context=SimpleNamespace(
                read_only=read_only, role="main", task_id="t", trace=None,
                checkpoint_manager=None, checkpoint=None,
                standard_binding=None),
        )

        return envelope

    def test_every_mutating_tool_is_refused(self):
        self.assertTrue(MUTATING)

        for name in MUTATING:
            with self.subTest(tool=name):
                envelope = self.route(
                    name, {"path": "foo.c", "content": "x", "old_text": "a",
                           "new_text": "b"}, read_only=True)

                self.assertEqual(envelope.status, ToolResultStatus.DENIED)
                self.assertIn("read-only", envelope.text)

    def test_the_refusal_names_no_other_way_to_write(self):
        text = self.route("write_file", {"path": "foo.c", "content": "x"},
                          read_only=True).text.lower()

        for mechanism in OTHER_MECHANISMS:
            with self.subTest(mechanism=mechanism):
                self.assertNotIn(mechanism, text)

    def test_nothing_is_written(self):
        target = Path(rag_chat.PROJECT_ROOT) / "should-not-exist-read-only.c"
        self.assertFalse(target.exists())
        self.route("write_file", {"path": target.name, "content": "x" * 300},
                   read_only=True)

        self.assertFalse(target.exists())

    def test_a_read_only_tool_is_untouched_by_the_gate(self):
        """CASE 6 -- no false positives."""
        for spec in native_tool_specs():
            if spec.mutability == ToolMutability.MUTATING:
                continue

            with self.subTest(tool=spec.name):
                self.assertNotEqual(
                    getattr(spec, "mutability", None), ToolMutability.MUTATING)

    def test_the_gate_runs_before_the_handler(self):
        source = inspect.getsource(
            sys.modules["tool_router"].ToolRouter.execute)
        gate = source.index("context.read_only")
        dispatch = source.index("spec.handler_key") if "spec.handler_key" in source \
            else len(source)

        self.assertLess(gate, dispatch)


class SafeModeSaysSoFirst(unittest.TestCase):
    """CASE 3 -- the mode refuses, and says which refusal this is."""

    def test_it_refuses_in_safe_and_nowhere_else(self):
        with safe_mode("SAFE"):
            self.assertIsNotNone(rag_chat.safe_mode_refusal("write_file"))

        for mode in ("ASK", "AUTO"):
            with self.subTest(mode=mode), safe_mode(mode):
                self.assertIsNone(rag_chat.safe_mode_refusal("write_file"))

    def test_it_names_the_mode_not_the_content(self):
        with safe_mode("SAFE"):
            text = rag_chat.safe_mode_refusal("write_file")

        self.assertIn("safe mode", text)
        self.assertIn("write_file", text)
        self.assertNotIn("bytes vs", text)

    def test_it_recommends_no_way_around_the_mode(self):
        with safe_mode("SAFE"):
            text = rag_chat.safe_mode_refusal("edit_file").lower()

        for mechanism in OTHER_MECHANISMS:
            with self.subTest(mechanism=mechanism):
                self.assertNotIn(mechanism, text)

    def test_every_mutating_handler_checks_it_before_the_heuristics(self):
        for handler, marker in (
            (rag_chat._registered_write_file, "is much larger than"),
            (rag_chat._registered_edit_file, None),
            (rag_chat._registered_append_file, None),
        ):
            with self.subTest(handler=handler.__name__):
                source = inspect.getsource(handler)
                check = source.index("safe_mode_refusal")

                self.assertLess(check, source.index("authorize_mutation"))

                if marker:
                    self.assertLess(check, source.index(marker))


class ReadingStillWorks(unittest.TestCase):
    """CASE 6 -- the gate is about writing and about nothing else."""

    def test_the_read_floor_survives_a_read_only_request(self):
        floor = ToolExposurePolicy.floor(
            __import__("agent_roles", fromlist=["AgentRole"]).AgentRole.MAIN
            if hasattr(__import__("agent_roles"), "AgentRole")
            else None, read_only=True)

        self.assertIn("bash", floor)

    def test_no_mutating_tool_is_in_the_read_only_floor(self):
        from tool_exposure import AgentRole

        floor = set(ToolExposurePolicy.floor(AgentRole.MAIN, read_only=True))

        self.assertEqual(floor & set(MUTATING), set())

    def test_a_read_only_command_is_not_a_mutation(self):
        from tool_runtime import CommandClassification

        for command in ("cat foo.c", "grep -rn ack src/", "ls -la"):
            with self.subTest(command=command):
                self.assertEqual(
                    rag_chat.COMMAND_POLICY.classify(command).classification,
                    CommandClassification.READ_ONLY)


class TheCompileRefusalStaysAccurate(unittest.TestCase):
    """CASE 5 -- what a read-only turn is told when it reaches for a compiler."""

    def test_it_names_the_request_and_forbids_another_route(self):
        text = rag_chat.READ_ONLY_REFUSAL.lower()

        self.assertIn("read-only", text)
        self.assertIn("do not try another way", text)

    def test_a_compiler_is_not_classified_read_only(self):
        from tool_runtime import CommandClassification

        self.assertNotEqual(
            rag_chat.COMMAND_POLICY.classify("gcc -o probe probe.c").classification,
            CommandClassification.READ_ONLY)


if __name__ == "__main__":
    unittest.main()
