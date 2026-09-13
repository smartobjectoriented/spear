"""A tool's declaration is checked where it is written, not where it is used.

The router enforces `execution_modes` generically: a spec runs in the modes it
names, and a spec that names none is unrestricted. That rule is right for the
nine read-only tools, which all name none -- and it is a trap for a tool that
writes, because unrestricted includes safe mode, where nothing may be written.

`save_skill` was MUTATING and named no modes. Measured before this check
existed: the spec was accepted, and routed in safe mode its handler RAN. It
was saved only by remembering, inside itself, to ask for authorization -- the
per-tool discipline the central gate exists to replace.

Two mistakes are possible and both are silent:

    execution_modes=()                  a mutating tool, unrestricted
    execution_modes=("asks", "auto")    denies the mode it meant to allow

Neither is a narrower permission; both are a different one. So they are
refused at construction, which is the moment someone is looking at the
declaration. This is configuration validation and nothing else: the router
goes on enforcing what a spec declares, for every tool, without consulting
mutability -- `test_execution_mode_gate` holds that.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tool_registry import (
    VALID_EXECUTION_MODES, ToolCategory, ToolMutability, ToolSpec,
    native_tool_specs,
)

SCHEMA = {"type": "object", "properties": {"value": {"type": "string"}}}


def build(*, mutability=ToolMutability.MUTATING, modes=(), name="probe"):
    return ToolSpec(name, "a probe tool", SCHEMA, ToolCategory.OTHER,
                    mutability, execution_modes=modes, handler_key=name)


class AnUnknownModeIsRefused(unittest.TestCase):
    """CASE 1 -- a typo changes behaviour, so it is not allowed to be quiet."""

    def test_a_misspelled_mode_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as caught:
            build(modes=("asks", "auto"))

        self.assertIn("unknown execution mode", str(caught.exception))
        self.assertIn("asks", str(caught.exception))

    def test_the_error_names_the_tool_and_the_choices(self):
        with self.assertRaises(ValueError) as caught:
            build(modes=("ask", "AUTO"), name="loud")

        message = str(caught.exception)

        self.assertIn("loud", message)
        self.assertIn("'ask', 'auto', 'safe'", message)

    def test_it_applies_to_a_read_only_tool_too(self):
        """The vocabulary is the vocabulary, whatever the tool does."""
        with self.assertRaises(ValueError):
            build(mutability=ToolMutability.READ_ONLY, modes=("often",))

    def test_the_vocabulary_is_exactly_the_execution_modes(self):
        from tool_runtime import ExecutionMode

        self.assertEqual(VALID_EXECUTION_MODES,
                         {str(mode) for mode in ExecutionMode})


class AMutatingToolStatesItsModes(unittest.TestCase):
    """CASE 2 and CASE 3 -- empty means unrestricted, so it is not allowed."""

    def test_an_empty_declaration_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            build(modes=())

        message = str(caught.exception)

        self.assertIn("must declare execution_modes", message)
        self.assertIn("unrestricted", message)

    def test_declaring_safe_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            build(modes=("safe", "ask", "auto"))

        self.assertIn("cannot declare 'safe'", str(caught.exception))

    def test_safe_alone_is_rejected(self):
        with self.assertRaises(ValueError):
            build(modes=("safe",))

    def test_the_ordinary_declaration_is_accepted(self):
        """CASE 4."""
        spec = build(modes=("ask", "auto"))

        self.assertEqual(spec.execution_modes, ("ask", "auto"))

    def test_auto_only_is_accepted(self):
        self.assertEqual(build(modes=("auto",)).execution_modes, ("auto",))


class TheOtherMutabilitiesKeepTheirContract(unittest.TestCase):
    """CASE 5 and CASE 6 -- an omitted declaration stays legitimate."""

    def test_a_read_only_tool_may_declare_nothing(self):
        spec = build(mutability=ToolMutability.READ_ONLY, modes=())

        self.assertEqual(spec.execution_modes, ())

    def test_a_conditional_tool_may_declare_nothing(self):
        spec = build(mutability=ToolMutability.CONDITIONAL, modes=())

        self.assertEqual(spec.execution_modes, ())

    def test_a_conditional_tool_may_declare_safe(self):
        """bash does, and runs in safe mode with its capabilities withdrawn."""
        spec = build(mutability=ToolMutability.CONDITIONAL,
                     modes=("safe", "ask", "auto"))

        self.assertEqual(spec.execution_modes, ("safe", "ask", "auto"))

    def test_a_read_only_tool_may_restrict_itself(self):
        self.assertEqual(
            build(mutability=ToolMutability.READ_ONLY,
                  modes=("auto",)).execution_modes, ("auto",))


class TheRealRegistryPasses(unittest.TestCase):
    """CASE 7 -- every production spec, built and checked.

    Importing the registry constructs them all, so a future tool that forgets
    its declaration does not reach a review: the suite stops.
    """

    def specs(self):
        import rag_chat

        return list(rag_chat.TOOL_REGISTRY.list_specs())

    def test_every_native_spec_constructs(self):
        self.assertTrue(native_tool_specs())

    def test_the_whole_registry_is_present(self):
        self.assertTrue(self.specs())

    def test_every_declared_mode_is_a_real_mode(self):
        for spec in self.specs():
            with self.subTest(tool=spec.name):
                self.assertTrue(
                    set(spec.execution_modes) <= VALID_EXECUTION_MODES)

    def test_every_mutating_spec_declares_modes_without_safe(self):
        mutating = [spec for spec in self.specs()
                    if spec.mutability == ToolMutability.MUTATING]

        self.assertTrue(mutating)

        for spec in mutating:
            with self.subTest(tool=spec.name):
                self.assertTrue(spec.execution_modes)
                self.assertNotIn("safe", spec.execution_modes)

    def test_the_read_only_specs_still_declare_nothing(self):
        """The compatibility contract, still relied upon in production."""
        undeclared = [spec.name for spec in self.specs()
                      if spec.mutability == ToolMutability.READ_ONLY
                      and not spec.execution_modes]

        self.assertTrue(undeclared)


class TheCheckLivesInOnePlace(unittest.TestCase):
    """One boundary, not a rule repeated until one copy drifts."""

    def test_the_spec_validates_itself(self):
        import inspect

        import tool_registry

        source = inspect.getsource(tool_registry.ToolSpec.__post_init__)

        self.assertIn("VALID_EXECUTION_MODES", source)
        self.assertIn("ToolMutability.MUTATING", source)

    def test_the_registry_adds_no_second_copy(self):
        import inspect

        import tool_registry

        source = inspect.getsource(tool_registry.ToolRegistry.register)

        self.assertNotIn("execution_modes", source)

    def test_the_router_still_decides_on_the_declaration_alone(self):
        """The runtime must not have become "mutating means unsafe"."""
        import inspect

        import tool_router

        source = inspect.getsource(tool_router.ToolRouter.execute)
        gate = source[source.index("spec.execution_modes"):]
        gate = gate[:gate.index("_early_failure") + 200]

        self.assertNotIn("MUTATING", gate)


if __name__ == "__main__":
    unittest.main()
