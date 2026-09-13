"""A call the router refuses is one the operator can see was refused.

The gates that refuse before dispatch -- role, task scope, execution mode,
repeat suppression -- are the ones whose refusal nobody watching can see. The
handler renders the call, and the handler never runs: watched live under
`--safe`, a session asked for edit_file twice, was refused twice, and the
terminal went straight from one model sentence to the next. The model was
told in its tool result; the person was not told at all.

So one line per refused call, from the layer that owns the terminal, keyed on
the category the router stamped. A handler that fails has already printed its
own result and must not gain a second line -- which is why this keys on the
router's own category names rather than on failure in general.

The trace side needed nothing: `_early_failure` has always passed
`error_category` to the event, and every refusal in the router goes through
it, repeat suppression included. These tests hold that so it stays true.
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag_chat
from tool_runtime import ExecutionMode
from tracing import EventType, TraceEmitter

# Nothing a refusal is allowed to suggest. A line that names one of these
# reads as a menu, and the turn goes shopping.
FORBIDDEN = ("--auto", "--ask", "--safe", "rerun", "instead", "another tool",
             "edit_file for", "bypass", "enable", "disable")


class Recorder(TraceEmitter):
    """A trace that keeps its events, so the failure can be read back."""

    def __init__(self):
        super().__init__()
        self.events = []

    def emit(self, event_type, task_id, **kwargs):
        self.events.append((event_type, kwargs))

        return super().emit(event_type, task_id, **kwargs)

    def failures(self):
        return [kwargs for event_type, kwargs in self.events
                if event_type == EventType.TOOL_CALL_FAILED]


def call(name, arguments, *, mode=ExecutionMode.SAFE, read_only=False,
         cache=None):
    """One tool call through the production routing path, output captured."""
    trace = Recorder()
    context = SimpleNamespace(
        read_only=read_only, role="main", task_id="task_visibility",
        trace=trace, checkpoint_manager=None, checkpoint=None,
        standard_binding=None)
    screen = io.StringIO()

    with patch.object(rag_chat, "EXECUTION_MODE", mode), redirect_stdout(screen):
        envelope = rag_chat.route_tool_envelope(
            name, arguments, {} if cache is None else cache,
            trace=trace, agent_context=context)

    return envelope, screen.getvalue(), trace


def refusal_lines(screen):
    return [line for line in screen.splitlines() if "refused:" in line]


class AModeRefusalIsVisible(unittest.TestCase):
    """CASE 1 -- a tool excluded from safe mode."""

    def setUp(self):
        self.envelope, self.screen, self.trace = call(
            "write_file", {"path": "a.txt", "content": "x"},
            mode=ExecutionMode.SAFE)

    def test_the_handler_was_not_reached(self):
        self.assertEqual(self.envelope.error_category, "execution_mode_denied")
        self.assertFalse(Path("a.txt").exists())

    def test_exactly_one_refusal_line_is_printed(self):
        self.assertEqual(len(refusal_lines(self.screen)), 1)

    def test_it_names_the_tool_and_the_reason(self):
        line = refusal_lines(self.screen)[0]

        self.assertIn("write_file refused", line)
        self.assertIn("unavailable in safe mode", line)

    def test_it_suggests_nothing(self):
        line = refusal_lines(self.screen)[0].lower()

        for phrase in FORBIDDEN:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, line)

    def test_the_trace_failure_carries_the_category(self):
        failures = self.trace.failures()

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["error_category"], "execution_mode_denied")


class AReadOnlyRefusalIsVisible(unittest.TestCase):
    """CASE 2 -- auto mode, read-only task, a mutating tool asked for."""

    def setUp(self):
        self.envelope, self.screen, self.trace = call(
            "edit_file", {"path": "a.txt", "old_text": "a", "new_text": "b"},
            mode=ExecutionMode.AUTO, read_only=True)

    def test_the_handler_was_not_reached(self):
        self.assertEqual(self.envelope.error_category, "read_only_task")

    def test_exactly_one_refusal_line_naming_the_task(self):
        lines = refusal_lines(self.screen)

        self.assertEqual(len(lines), 1)
        self.assertIn("edit_file refused", lines[0])
        self.assertIn("the task is read-only", lines[0])

    def test_the_trace_failure_carries_the_category(self):
        self.assertEqual(self.trace.failures()[0]["error_category"],
                         "read_only_task")

    def test_the_mode_did_not_decide_this_one(self):
        """AUTO permits the tool; the task does not. The line must say so."""
        self.assertNotIn("mode", refusal_lines(self.screen)[0])


class AHandlerFailureKeepsItsOwnOutput(unittest.TestCase):
    """CASE 3 -- no second line for a failure the handler already rendered."""

    def setUp(self):
        self.envelope, self.screen, self.trace = call(
            "write_file", {"path": "/etc/definitely-not-here.txt",
                           "content": "x"},
            mode=ExecutionMode.AUTO)

    def test_the_handler_reported_it(self):
        self.assertIn("ERROR", self.screen)
        self.assertIn("path escapes the workspace", self.screen)

    def test_no_router_refusal_line_was_added(self):
        self.assertEqual(refusal_lines(self.screen), [])

    def test_its_category_is_the_generic_one(self):
        """Which is exactly why the notice keys on the router's own names."""
        self.assertNotIn(self.envelope.error_category,
                         rag_chat._ROUTER_REFUSAL_NOTICES)


class SuccessIsUnchanged(unittest.TestCase):
    """CASE 4 and CASE 6 -- nothing new for a call that runs."""

    def test_a_read_only_tool_prints_no_refusal(self):
        envelope, screen, trace = call(
            "search_history", {"query": "acknowledge"}, mode=ExecutionMode.SAFE)

        self.assertEqual(refusal_lines(screen), [])

    def test_a_read_only_tool_is_not_refused_in_safe_mode(self):
        envelope, _, _ = call("search_history", {"query": "x"},
                              mode=ExecutionMode.SAFE)

        self.assertNotEqual(envelope.error_category, "execution_mode_denied")

    def test_the_notice_is_silent_for_anything_it_does_not_know(self):
        for category in (None, "", "failed", "timeout", "cancelled"):
            with self.subTest(category=category):
                screen = io.StringIO()

                with redirect_stdout(screen):
                    printed = rag_chat.announce_router_refusal(
                        "probe", SimpleNamespace(error_category=category))

                self.assertFalse(printed)
                self.assertEqual(screen.getvalue(), "")


class ACancellationIsNotAModeRefusal(unittest.TestCase):
    """CASE 5 -- the operator declined; the mode permitted it."""

    def test_a_declined_mutation_is_reported_as_cancelled(self):
        with patch.object(rag_chat, "confirm", return_value=False):
            envelope, screen, _ = call(
                "write_file", {"path": "declined.txt", "content": "x"},
                mode=ExecutionMode.ASK)

        self.assertNotEqual(envelope.error_category, "execution_mode_denied")
        self.assertEqual(refusal_lines(screen), [])

    def test_the_notice_has_no_entry_for_an_ordinary_cancellation(self):
        self.assertNotIn("cancelled", rag_chat._ROUTER_REFUSAL_NOTICES)


class TheTraceAlreadyCarriedTheCategory(unittest.TestCase):
    """CASE 7 -- additive, and true before this change as well as after.

    `_early_failure` has always passed `error_category`, and every refusal in
    the router is built by it -- repeat suppression calls it too. No event
    schema changed here; these tests exist so that stays so.
    """

    #: Pre-dispatch outcomes that are NOT policy refusals: the call was
    #: malformed, not forbidden. They stay silent on purpose -- "refused:
    #: unknown tool" for a name the model mistyped is noise, and the model
    #: corrects it from the tool result on the next round.
    NOT_REFUSALS = {"unknown_tool", "invalid_arguments"}

    def test_nothing_fails_early_without_going_through_that_function(self):
        """Two emissions: the dispatch span, and `_early_failure`.

        A third would be a refusal whose category nothing stamps, which is
        the shape of the gap this is meant to keep closed.
        """
        source = Path(ROOT, "tool_router.py").read_text(encoding="utf-8")

        self.assertEqual(source.count("EventType.TOOL_CALL_FAILED"), 2)

    def test_every_refusal_category_is_either_announced_or_excluded(self):
        import re

        source = Path(ROOT, "tool_router.py").read_text(encoding="utf-8")
        categories = set(re.findall(
            r"ToolResultStatus\.[A-Z_]+, \"([a-z_]+)\"", source))

        self.assertTrue(categories)

        for category in categories:
            with self.subTest(category=category):
                self.assertTrue(
                    category in rag_chat._ROUTER_REFUSAL_NOTICES
                    or category in self.NOT_REFUSALS,
                    f"{category} is neither announced nor deliberately silent")

    def test_the_repeat_suppression_path_uses_it_too(self):
        import inspect

        import tool_router

        source = inspect.getsource(tool_router.ToolRouter._suppressed_repeat)

        self.assertIn("self._early_failure(", source)

    def test_that_function_stamps_the_category_on_the_event(self):
        import inspect

        import tool_router

        source = inspect.getsource(tool_router.ToolRouter._early_failure)
        failed = source[source.index("EventType.TOOL_CALL_FAILED"):]

        self.assertIn("error_category=category", failed)

    def test_the_field_is_optional_on_the_event(self):
        """A reader that does not expect it is unaffected."""
        from tracing import TraceEvent

        field = TraceEvent.__dataclass_fields__["error_category"]

        self.assertIsNone(field.default)

    def test_a_successful_call_leaves_it_unset(self):
        _, _, trace = call("search_history", {"query": "x"},
                           mode=ExecutionMode.SAFE)

        self.assertEqual(trace.failures(), [])


if __name__ == "__main__":
    unittest.main()
