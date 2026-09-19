import unittest

from tool_registry import (
    ToolCategory, ToolMutability, ToolRegistry, ToolSpec, native_tool_specs,
)


def spec(name="probe", *, roles=frozenset({"main"}), visible=True):
    return ToolSpec(
        name, "Probe a value",
        {"type": "object", "properties": {"value": {"type": "string"}},
         "required": ["value"]},
        ToolCategory.OTHER, ToolMutability.READ_ONLY,
        roles=roles, model_visible=visible, handler_key=name,
    )


class ToolRegistryTests(unittest.TestCase):
    def test_registration_and_handler_association(self):
        registry = ToolRegistry()
        handler = lambda *_: "ok"
        registry.register(spec(), handler)
        self.assertEqual(registry.get("probe").name, "probe")
        self.assertIs(registry.handler("probe"), handler)

    def test_duplicate_registration_is_rejected(self):
        registry = ToolRegistry()
        registry.register(spec(), lambda *_: "ok")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            registry.register(spec(), lambda *_: "again")

    def test_listing_is_deterministic_registration_order(self):
        registry = ToolRegistry()
        for name in ("zeta", "alpha", "middle"):
            registry.register(spec(name), lambda *_: "ok")
        self.assertEqual(
            [item.name for item in registry.list_specs()],
            ["zeta", "alpha", "middle"],
        )

    def test_schema_exposure_is_provider_neutral_and_openai_compatible(self):
        registry = ToolRegistry()
        registry.register(spec(), lambda *_: "ok")
        definition = registry.definitions_for_model()[0]
        self.assertEqual(definition.name, "probe")
        self.assertEqual(definition.input_schema["required"], ["value"])
        self.assertEqual(
            registry.openai_definitions_for_model()[0]["function"]["name"],
            "probe",
        )

    def test_unknown_tool_and_invalid_schema(self):
        registry = ToolRegistry()
        with self.assertRaises(KeyError):
            registry.get("missing")
        with self.assertRaises(ValueError):
            ToolSpec("bad", "bad", {"type": "string"})
        with self.assertRaises(ValueError):
            ToolSpec("bad", "bad", {
                "type": "object", "properties": {}, "required": ["missing"],
            })

    def test_subset_role_and_visibility_filtering(self):
        registry = ToolRegistry()
        registry.register(spec("main"), lambda *_: "ok")
        registry.register(spec(
            "read_only", roles=frozenset({"main", "explore"})), lambda *_: "ok")
        registry.register(spec("hidden", visible=False), lambda *_: "ok")
        self.assertEqual(
            [item.name for item in registry.list_specs(role="explore")],
            ["read_only"],
        )
        self.assertEqual(
            [item.name for item in registry.list_specs(names={"main"})], ["main"],
        )
        self.assertNotIn("hidden", [item.name for item in registry.definitions_for_model()])

    def test_mutability_category_and_capability_metadata(self):
        bash = native_tool_specs()[0]
        self.assertEqual(bash.category, ToolCategory.COMMAND)
        self.assertEqual(bash.mutability, ToolMutability.CONDITIONAL)
        self.assertIn("dynamic_command_policy", bash.required_capabilities)

    def test_current_native_set_and_order_are_preserved(self):
        self.assertEqual([item.name for item in native_tool_specs()], [
            "bash", "edit_file", "write_file", "append_file", "delete_file",
            "remember", "plan_change", "search_corpus", "search_internet",
            "fetch_url",
        ])

    def test_non_command_requires_handler_but_command_is_router_owned(self):
        registry = ToolRegistry()
        with self.assertRaisesRegex(ValueError, "requires a handler"):
            registry.register(spec(), None)
        command = ToolSpec(
            "run", "Run securely", {"type": "object", "properties": {
                "command": {"type": "string"}}, "required": ["command"]},
            ToolCategory.COMMAND, ToolMutability.CONDITIONAL,
        )
        registry.register(command, None)
        self.assertIsNone(registry.handler("run"))


class BashDescriptionTests(unittest.TestCase):
    """How to run tests belongs in the tool description, before the first
    attempt -- not only in the refusal that follows it."""

    def test_the_bash_description_states_the_runnable_test_command(self):
        bash = next(item for item in native_tool_specs() if item.name == "bash")

        self.assertIn("python3 -m unittest discover -s tests", bash.description)
        self.assertIn("REFUSED", bash.description)
        self.assertIn("python3 -c", bash.description)

        # Alone: a `||` chain with `python3 -c` in it refuses the whole
        # command, so a fallback is not a safety net, it is the failure.

        self.assertIn("ALONE", bash.description)
        self.assertIn("fallback", bash.description)


if __name__ == "__main__":
    unittest.main()
