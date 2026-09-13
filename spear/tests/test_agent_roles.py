import unittest

from agent_roles import AgentRole, AgentRoleSpec, explorer_role
from tool_registry import native_tool_specs, ToolRegistry, ToolSpec
from tool_router import ToolExecutionContext, ToolResultStatus, ToolRouter
from tracing import TraceEmitter
from tool_runtime import CommandPolicy, ExecutionMode


def registry():
    result = ToolRegistry()
    for spec in native_tool_specs():
        result.register(spec, None if spec.name == "bash" else lambda *_: "OK")
    for name in ("save_skill", "checkpoint", "rollback"):
        result.register(
            ToolSpec(
                name, "Main-only capability",
                {"type": "object", "properties": {}},
            ),
            lambda *_: "OK",
        )
    return result


class AgentRoleTests(unittest.TestCase):
    def test_explorer_role_is_bounded_and_read_only(self):
        role = explorer_role(max_model_turns=5, max_tool_calls=9,
                             context_limit=2048)
        self.assertEqual(role.role, AgentRole.EXPLORER)
        self.assertEqual(role.execution_mode, "safe")
        self.assertEqual((role.max_model_turns, role.max_tool_calls), (5, 9))
        self.assertFalse(role.web_access)
        self.assertFalse(role.can_write_memory)
        self.assertFalse(role.checkpoint_access)
        self.assertFalse(role.recursive_delegation)
        role.validate_registry(registry())

    def test_registry_explorer_view_is_minimal(self):
        names = {spec.name for spec in registry().list_specs(role="explorer")}
        self.assertEqual(names, {"bash", "search_corpus"})
        definitions = registry().definitions_for_model(role="explorer")
        self.assertEqual({tool.name for tool in definitions}, names)

    def test_explorer_cannot_be_configured_with_sensitive_access(self):
        for values in (
            {"can_write_memory": True}, {"web_access": True},
            {"checkpoint_access": True}, {"recursive_delegation": True},
        ):
            with self.assertRaises(ValueError):
                AgentRoleSpec(
                    AgentRole.EXPLORER, "read", frozenset({"bash"}),
                    "safe", 2, 2, 1024, **values,
                )

    def test_optional_model_override_is_provider_neutral_metadata(self):
        role = explorer_role(model_override="local-small")
        self.assertEqual(role.model_override, "local-small")

    def test_router_structurally_denies_main_only_tools(self):
        tools = registry()
        router = ToolRouter(tools)
        context = ToolExecutionContext(
            "task_explorer", TraceEmitter(), {}, role="explorer",
            command_executor=lambda _: "OK",
        )
        attempts = {
            "edit_file": {"path": "a", "old_text": "x", "new_text": "y"},
            "write_file": {"path": "a", "content": "x"},
            "append_file": {"path": "a", "content": "x"},
            "remember": {"note": "x"},
            "search_internet": {"query": "x"},
            "save_skill": {}, "checkpoint": {}, "rollback": {},
        }
        for name, arguments in attempts.items():
            with self.subTest(name=name):
                result = router.execute(context, "call_" + name, name, arguments)
                self.assertEqual(result.status, ToolResultStatus.DENIED)
                self.assertFalse(result.success)

    def test_explorer_command_requires_real_command_boundary(self):
        result = ToolRouter(registry()).execute(
            ToolExecutionContext("task_explorer", TraceEmitter(), {},
                                 role="explorer"),
            "call", "bash", {"command": "printf pwned > file"},
        )
        self.assertEqual(result.status, ToolResultStatus.DENIED)
        self.assertIn("boundary unavailable", result.text)

    def test_safe_command_policy_rejects_mutating_shell_command(self):
        policy = CommandPolicy()
        assessment = policy.classify("printf pwned > file")
        authorization = policy.authorize(assessment, ExecutionMode.SAFE)
        self.assertIsNotNone(authorization.result)
        self.assertEqual(authorization.result.status, "denied")


if __name__ == "__main__":
    unittest.main()
