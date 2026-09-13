import threading
import tempfile
import unittest
from unittest.mock import Mock

from cancellation import (
    CancellationScope, CancellationSource, OperationCancelled,
)
from tool_registry import ToolCategory, ToolMutability, ToolRegistry, ToolSpec
from tool_router import ToolExecutionContext, ToolResultStatus, ToolRouter
from tracing import TraceEmitter
from tool_runtime import CommandRunner, ToolResult, Workspace


class CancellationTests(unittest.TestCase):
    def test_cancel_is_idempotent_and_retains_first_reason(self):
        source = CancellationSource()
        self.assertTrue(source.cancel("stop", CancellationScope.OPERATION))
        self.assertFalse(source.cancel("later", CancellationScope.TASK))
        self.assertEqual(source.token.reason, "stop")
        self.assertEqual(source.token.scope, CancellationScope.OPERATION)

    def test_cooperative_check_raises_typed_exception(self):
        source = CancellationSource()
        source.cancel("user interrupt")
        with self.assertRaises(OperationCancelled) as raised:
            source.token.raise_if_cancelled()
        self.assertEqual(raised.exception.reason, "user interrupt")

    def test_router_cancels_before_handler(self):
        called = []
        registry = ToolRegistry()
        registry.register(ToolSpec(
            "probe", "probe", {"type": "object"}, ToolCategory.OTHER,
            ToolMutability.READ_ONLY, handler_key="probe",
        ), lambda *_: called.append(True) or "OK")
        source = CancellationSource(); source.cancel()
        result = ToolRouter(registry).execute(
            ToolExecutionContext("task", TraceEmitter(), {}, cancellation=source.token),
            "call", "probe", {},
        )
        self.assertEqual(result.status, ToolResultStatus.CANCELLED)
        self.assertFalse(called)

    def test_router_observes_cancellation_during_fake_tool(self):
        registry = ToolRegistry(); source = CancellationSource()
        def handler(context, _arguments):
            source.cancel("during tool")
            return "should not become success"
        registry.register(ToolSpec(
            "probe", "probe", {"type": "object"}, ToolCategory.OTHER,
            ToolMutability.READ_ONLY, handler_key="probe",
        ), handler)
        result = ToolRouter(registry).execute(
            ToolExecutionContext("task", TraceEmitter(), {}, cancellation=source.token),
            "call", "probe", {},
        )
        self.assertEqual(result.status, ToolResultStatus.CANCELLED)
        self.assertFalse(result.success)

    def test_cancelled_command_does_not_enter_sandbox(self):
        source = CancellationSource(); source.cancel("stop command")
        sandbox = Mock()
        with tempfile.TemporaryDirectory() as root:
            result = CommandRunner(sandbox=sandbox).run_sandboxed(
                Workspace(root), ["/bin/true"], cancellation=source.token,
            )
        self.assertEqual(result.status, "cancelled")
        sandbox.ensure_available.assert_not_called()
        sandbox.run.assert_not_called()
