"""A tool runs in the modes its own spec declares, and the router says so.

`ToolSpec.execution_modes` has been carried on every spec since the registry
was written -- write_file declares ("ask", "auto"), bash declares all three --
and nothing read it. Safe mode was enforced instead inside each mutating
handler, one local check per tool, which is a rule a new tool can forget.
`save_skill` already had: MUTATING, no declaration, refused in safe mode only
because its handler happened to call authorize_mutation.

So the declaration is enforced once, at the routing boundary, in the order the
layers actually mean:

    role -> task scope -> execution mode -> tool authorization -> execution

Three properties are load-bearing and each has a test here:

  * the gate reads the DECLARATION, not the mutability. A read-only tool may
    restrict itself and is held to it; a mutating tool that declares nothing
    is not refused by this gate.
  * an empty declaration stays unrestricted. Every read-only spec carries one,
    so reading it as "denied everywhere" would turn off retrieval in every
    session.
  * the read-only task gate still runs first and is independent of the mode.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cancellation import NEVER_CANCELLED
from tool_registry import (
    ToolCategory, ToolMutability, ToolRegistry, ToolSpec,
)
from tool_router import (
    ToolExecutionContext, ToolHandlerResult, ToolResultStatus, ToolRouter,
)
from tracing import TraceEmitter

SCHEMA = {"type": "object", "properties": {"value": {"type": "string"}}}


class Handler:
    """Records whether it was reached at all -- which is the whole question."""

    def __init__(self):
        self.calls = []

    def __call__(self, context, arguments):
        self.calls.append(dict(arguments))

        return ToolHandlerResult("OK: ran")


def spec(name, *, modes=(), mutability=ToolMutability.MUTATING):
    return ToolSpec(name, f"probe tool {name}", SCHEMA,
                    ToolCategory.OTHER, mutability,
                    execution_modes=modes, handler_key=name)


def route(name, *, mode="", read_only=False, modes=(),
          mutability=ToolMutability.MUTATING):
    registry = ToolRegistry()
    handler = Handler()
    registry.register(spec(name, modes=modes, mutability=mutability), handler)
    context = ToolExecutionContext(
        task_id="task_modegate", trace=TraceEmitter(), cache={},
        cancellation=NEVER_CANCELLED, read_only=read_only,
        execution_mode=mode)
    envelope = ToolRouter(registry).execute(
        context, "call_1", name, {"value": "x"})

    return envelope, handler


class TheDeclarationIsHonoured(unittest.TestCase):
    """CASE 1, CASE 2, CASE 6 -- the invariant itself."""

    def test_a_permitted_mode_reaches_the_handler(self):
        envelope, handler = route("probe", mode="auto", modes=("ask", "auto"))

        self.assertEqual(envelope.status, ToolResultStatus.OK)
        self.assertEqual(len(handler.calls), 1)

    def test_a_prohibited_mode_never_reaches_it(self):
        envelope, handler = route("probe", mode="safe", modes=("ask", "auto"))

        self.assertEqual(envelope.status, ToolResultStatus.DENIED)
        self.assertEqual(handler.calls, [])

    def test_ask_is_a_permitted_mode_and_the_router_decides_nothing_else(self):
        """CASE 3 -- the gate lets ASK through; approval is not its business."""
        envelope, handler = route("probe", mode="ask", modes=("ask", "auto"))

        self.assertEqual(envelope.status, ToolResultStatus.OK)
        self.assertEqual(len(handler.calls), 1)

    def test_auto_runs_the_tool(self):
        """CASE 4."""
        envelope, handler = route("probe", mode="auto", modes=("ask", "auto"))

        self.assertEqual(envelope.status, ToolResultStatus.OK)
        self.assertEqual(len(handler.calls), 1)

    def test_a_write_task_in_safe_mode_is_refused_by_the_mode(self):
        """CASE 6 -- the task permits writing; the mode does not."""
        envelope, handler = route("probe", mode="safe", read_only=False,
                                  modes=("ask", "auto"))

        self.assertEqual(envelope.status, ToolResultStatus.DENIED)
        self.assertEqual(handler.calls, [])
        self.assertIn("unavailable in safe mode", envelope.text)


class ItReadsTheDeclarationNotTheMutability(unittest.TestCase):
    """CASE 7 -- the proof that this is not "mutating means unsafe" again."""

    def test_a_read_only_tool_may_restrict_itself_and_is_held_to_it(self):
        envelope, handler = route("reader", mode="safe", modes=("auto",),
                                  mutability=ToolMutability.READ_ONLY)

        self.assertEqual(envelope.status, ToolResultStatus.DENIED)
        self.assertEqual(handler.calls, [])

    def test_the_same_read_only_tool_runs_in_the_mode_it_declares(self):
        envelope, handler = route("reader", mode="auto", modes=("auto",),
                                  mutability=ToolMutability.READ_ONLY)

        self.assertEqual(envelope.status, ToolResultStatus.OK)
        self.assertEqual(len(handler.calls), 1)

    def test_a_non_read_only_tool_declaring_nothing_is_not_refused_here(self):
        """The gate enforces a declaration; it does not invent one.

        Shown with a CONDITIONAL spec, which is what bash is: not read-only,
        legitimately undeclared, and the gate lets it through in every mode
        because that is what an empty declaration means.

        Deliberately not shown with a MUTATING spec. The gate would treat it
        the same way -- it reads the declaration and nothing else -- but a
        mutating tool that declares no modes is a configuration mistake in
        its own right, and a test is no place to enshrine one. Refusing it
        belongs to the layer that builds specs, not to this one.
        """
        envelope, handler = route("undeclared", mode="safe", modes=(),
                                  mutability=ToolMutability.CONDITIONAL)

        self.assertEqual(envelope.status, ToolResultStatus.OK)
        self.assertEqual(len(handler.calls), 1)


class AnAbsentDeclarationIsUnrestricted(unittest.TestCase):
    """CASE 8 -- the compatibility semantic, read off the registry."""

    def test_every_read_only_native_spec_relies_on_it(self):
        import rag_chat

        unrestricted = [item.name for item in rag_chat.TOOL_REGISTRY.list_specs()
                        if item.mutability == ToolMutability.READ_ONLY
                        and not item.execution_modes]

        self.assertTrue(unrestricted)

    def test_such_a_tool_runs_in_every_mode(self):
        for mode in ("safe", "ask", "auto"):
            with self.subTest(mode=mode):
                envelope, handler = route(
                    "reader", mode=mode, modes=(),
                    mutability=ToolMutability.READ_ONLY)

                self.assertEqual(envelope.status, ToolResultStatus.OK)
                self.assertEqual(len(handler.calls), 1)

    def test_a_caller_that_states_no_mode_is_unaffected(self):
        """The gate can only refuse a mode it was told about."""
        envelope, handler = route("probe", mode="", modes=("ask", "auto"))

        self.assertEqual(envelope.status, ToolResultStatus.OK)
        self.assertEqual(len(handler.calls), 1)


class TheTaskScopeGateStillWins(unittest.TestCase):
    """CASE 5 -- the two permissions stay independent, and scope runs first."""

    def test_a_read_only_task_refuses_a_tool_the_mode_would_allow(self):
        envelope, handler = route("probe", mode="auto", read_only=True,
                                  modes=("ask", "auto"))

        self.assertEqual(envelope.status, ToolResultStatus.DENIED)
        self.assertEqual(handler.calls, [])
        self.assertIn("read-only", envelope.text)

    def test_the_scope_refusal_is_the_one_reported(self):
        """Both gates would refuse; the task is the more specific fact."""
        envelope, _ = route("probe", mode="safe", read_only=True,
                            modes=("ask", "auto"))

        self.assertIn("read-only", envelope.text)
        self.assertNotIn("unavailable in safe mode", envelope.text)


class TheRefusalIsNotActionable(unittest.TestCase):
    """It states the fact and names no way around it."""

    FORBIDDEN = ("--auto", "--ask", "rerun", "instead", "use edit_file",
                 "bypass", "enable")

    def test_it_names_the_tool_and_the_mode(self):
        envelope, _ = route("probe", mode="safe", modes=("ask", "auto"))

        self.assertEqual(envelope.text,
                         "ERROR: 'probe' is unavailable in safe mode.")

    def test_it_recommends_nothing(self):
        text = route("probe", mode="safe", modes=("ask", "auto"))[0].text.lower()

        for phrase in self.FORBIDDEN:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, text)

    def test_it_is_audited_as_a_policy_denial(self):
        envelope, _ = route("probe", mode="safe", modes=("ask", "auto"))

        self.assertEqual(envelope.status, ToolResultStatus.DENIED)
        self.assertEqual(envelope.error_category, "execution_mode_denied")


class EveryMutatingToolDeclaresItsModes(unittest.TestCase):
    """The hazard this exists to remove, as a test rather than a convention.

    A new mutating tool that forgets its declaration used to be protected
    only by whatever its own handler remembered to check. It now fails here
    instead, which is the point at which it is cheap to fix.
    """

    def test_no_mutating_tool_permits_safe_mode(self):
        import rag_chat

        for item in rag_chat.TOOL_REGISTRY.list_specs():
            if item.mutability != ToolMutability.MUTATING:
                continue

            with self.subTest(tool=item.name):
                self.assertTrue(item.execution_modes,
                                f"{item.name} declares no execution_modes")
                self.assertNotIn("safe", item.execution_modes)

    def test_the_session_carries_its_mode_to_the_router(self):
        import inspect

        import rag_chat

        source = inspect.getsource(rag_chat.route_tool_envelope)

        self.assertIn("execution_mode=str(EXECUTION_MODE)", source)

    def test_only_the_router_reaches_a_handler(self):
        """CASE 9 -- no production path dispatches around the gate."""
        import subprocess

        found = subprocess.run(
            ["grep", "-rn", r"\.handler(", "--include=*.py", str(ROOT)],
            capture_output=True, text=True).stdout.splitlines()
        production = [line for line in found
                      if "/lib/" not in line and "/tests/" not in line
                      and "/eval/" not in line]

        self.assertEqual(len(production), 1, production)
        self.assertIn("tool_router.py", production[0])


if __name__ == "__main__":
    unittest.main()
