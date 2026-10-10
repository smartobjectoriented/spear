"""A failed project validation gets one repair pass, then is judged again.

An implementation turn that ends with the project's own build or tests
failing on the final state is given that failure -- the command, its origin,
its exit status, its output and the source epoch -- for exactly one more pass
of the coding core. Whatever that pass leaves is validated again; a failure
after it, or a validation that could not run, is UNVERIFIED.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evidence import project_build
from runtime import agent_core_turn
from runtime.agent_context import RuntimeTerminalReason
from runtime.agent_runtime import AgentRuntime
from runtime.tracing import EventType
from tests.test_agent_loop import Host, call, turn
from tests.test_core_runtime import TOOLS, Backend, _Recording
from tests.test_agent_runtime import make_context

ROOT = Path(__file__).resolve().parent.parent

FAILURE = "a.sh:2: error: still prints alpha\n(exit status 2)"


class Recording(Backend):
    """The scripted backend, keeping every request it was sent."""

    def __init__(self, raw_turns):
        super().__init__(raw_turns)
        self.requests = []

    def complete_messages(self, messages, tools, *, max_tokens, on_token=None):
        self.requests.append(messages)
        return super().complete_messages(messages, tools, max_tokens=max_tokens,
                                         on_token=on_token)


class Verifier:
    """The project's validation: passes once a.sh prints `wanted`."""

    def __init__(self, root, wanted="gamma", otherwise=("failed", FAILURE)):
        self.root, self.wanted, self.otherwise, self.calls = root, wanted, otherwise, []

    def __call__(self, command):
        self.calls.append(command)

        if self.wanted and self.wanted in Path(self.root, "a.sh").read_text():
            return "passed", ""

        return self.otherwise


class RefusesSecondWrite(Host):
    """Writes are refused once the first one has landed."""

    def authorize(self, name, arguments):
        if name == "patch" and "beta" in Path(self.workspace_root, "a.sh").read_text():
            return "refused: not in scope"

        return None


class ValidationRepair(unittest.TestCase):
    def run_turn(self, raw_turns, *, verifier=None, host_class=Host, rounds=10, actions=10,
                 mixed=False):
        self.root = tempfile.mkdtemp()
        Path(self.root, "a.sh").write_text("#!/bin/sh\necho alpha\n")
        Path(self.root, "a.sh").chmod(0o755)
        self.backend = Recording(raw_turns)
        context = make_context(self.backend, rounds=rounds, actions=actions)
        context.execution_core = "coding"
        context.tools = TOOLS
        context.project_root = self.root
        context.project_commands = project_build.declared({"build_commands": ["make check"]})
        verifier = verifier or Verifier(None)
        verifier.root = self.root
        context.project_verifier = verifier
        host = host_class(self.root)
        context.coding_host = lambda ctx, cache, record: _Recording(host, record)
        self.events = []
        emit = context.trace.emit
        context.trace.emit = lambda kind, *args, **kwargs: (
            self.events.append((kind, kwargs.get("metadata"))), emit(kind, *args, **kwargs))[1]

        if mixed:
            context.mixed_record = SimpleNamespace(packet=None)

        return AgentRuntime().run(context), context

    def repairs(self):
        return [meta for kind, meta in self.events
                if kind == EventType.VALIDATION_REPAIR_STARTED]

    @staticmethod
    def first_change():
        return turn("", call("c1", "patch", path="a.sh", old_string="alpha", new_string="beta"))

    def test_a_passing_validation_gets_no_repair(self):
        verifier = Verifier(None, wanted="beta")
        result, context = self.run_turn(
            [self.first_change(), turn("Done.")], verifier=verifier)

        self.assertEqual(context.core_verdict.state, "VERIFIED")
        self.assertEqual(self.repairs(), [])
        self.assertEqual(result.final_response, "Done.")

    def test_a_repair_that_fixes_the_code_is_verified_on_the_new_state(self):
        result, context = self.run_turn([
            self.first_change(), turn("Done."),
            turn("", call("c2", "patch", path="a.sh", old_string="beta", new_string="gamma")),
            turn("Fixed: it prints gamma.")])

        self.assertEqual(len(self.repairs()), 1)
        self.assertEqual(context.core_verdict.state, "VERIFIED")
        self.assertEqual(result.final_response, "Fixed: it prints gamma.")
        self.assertEqual(len(context.project_verifier.calls), 2)

        request = self.backend.requests[2][-1]["content"]

        for expected in ("validation failed on the current state", "build (configured",
                         "command: make check", "exit status: 2", "source epoch:",
                         "still prints alpha", "one repair pass"):
            with self.subTest(expected=expected):
                self.assertIn(expected, request)

        self.assertNotIn("wrong", request.lower())

    def test_a_repair_that_still_fails_is_unverified_and_not_repaired_again(self):
        verifier = Verifier(None, wanted="")
        result, context = self.run_turn([
            self.first_change(), turn("Done."),
            turn("", call("c2", "patch", path="a.sh", old_string="beta", new_string="delta")),
            turn("Fixed.")], verifier=verifier)

        self.assertEqual(len(self.repairs()), 1)
        self.assertEqual(context.core_verdict.state, "UNVERIFIED")
        self.assertTrue(result.final_response.startswith("**UNVERIFIED**"))
        self.assertEqual(len(verifier.calls), 2)
        self.assertEqual(self.backend.raw, [])

    def test_old_evidence_cannot_verify_the_state_a_repair_wrote(self):
        """The first pass ran its own check; the repair changed the file after
        it, and the project validation of the new state could not run."""
        verifier = Verifier(None, wanted="")
        verifier.otherwise = ("failed", FAILURE)
        sequence = iter([("failed", FAILURE), ("not_run", "no sandbox")])
        verifier.__class__ = type("Sequenced", (Verifier,), {
            "__call__": lambda self, command: (self.calls.append(command), next(sequence))[1]})
        result, context = self.run_turn([
            self.first_change(), turn("", call("c2", "terminal", command="./a.sh")),
            turn("Done, it prints beta."),
            turn("", call("c3", "patch", path="a.sh", old_string="beta", new_string="gamma")),
            turn("Fixed.")], verifier=verifier)

        self.assertEqual(len(self.repairs()), 1)
        self.assertEqual(len(verifier.calls), 2)
        self.assertEqual(context.core_verdict.state, "UNVERIFIED")

    def test_a_repair_that_changes_nothing_ends_there(self):
        verifier = Verifier(None, wanted="")
        result, context = self.run_turn([
            self.first_change(), turn("Done."),
            turn("I cannot see what to change.")], verifier=verifier)

        self.assertEqual(len(self.repairs()), 1)
        self.assertEqual(len(verifier.calls), 1)
        self.assertEqual(context.core_verdict.state, "UNVERIFIED")
        self.assertEqual(self.backend.raw, [])

        finished = [meta for kind, meta in self.events
                    if kind == EventType.VALIDATION_REPAIR_FINISHED]
        self.assertFalse(finished[0]["mutated"])

    def test_a_backgrounded_check_in_the_repair_verifies_nothing(self):
        verifier = Verifier(None, wanted="")
        result, context = self.run_turn([
            self.first_change(), turn("Done."),
            turn("", call("c2", "patch", path="a.sh", old_string="beta", new_string="gamma")),
            turn("", call("c3", "terminal", command="./a.sh &")),
            turn("Fixed and checked.")], verifier=verifier)

        self.assertEqual(context.core_verdict.state, "UNVERIFIED")

    def test_a_validation_that_did_not_run_is_not_repaired(self):
        verifier = Verifier(None, wanted="", otherwise=("not_run", "no sandbox"))
        result, context = self.run_turn([self.first_change(), turn("Done.")],
                                        verifier=verifier)

        self.assertEqual(self.repairs(), [])
        self.assertEqual(context.core_verdict.state, "UNVERIFIED")

    def test_a_refusal_loop_in_the_repair_is_still_stopped(self):
        verifier = Verifier(None, wanted="")
        retry = turn("", call("c9", "patch", path="a.sh", old_string="beta", new_string="x"))
        result, context = self.run_turn(
            [self.first_change(), turn("Done.")] + [retry] * 12,
            verifier=verifier, host_class=RefusesSecondWrite)

        self.assertEqual(len(self.repairs()), 1)
        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.STALLED)
        self.assertEqual(context.core_verdict.state, "UNVERIFIED")
        self.assertTrue(any(kind == EventType.REPEATED_REFUSAL_STOPPED
                            for kind, _ in self.events))

    def test_a_mixed_implementation_pass_is_left_to_its_orchestrator(self):
        result, context = self.run_turn([self.first_change(), turn("Done.")], mixed=True)

        self.assertEqual(self.repairs(), [])
        self.assertEqual(context.core_verdict.state, "UNVERIFIED")

    def test_a_turn_with_no_budget_left_is_not_resumed(self):
        result, context = self.run_turn([self.first_change(), turn("Done.")], rounds=2)

        self.assertEqual(self.repairs(), [])

    def test_the_request_bounds_the_output_it_carries(self):
        failed = SimpleNamespace(validation_kind="test", origin="probed", command="make test",
                                 evidence="x" * 9000 + "\n(exit status 1)")
        request = agent_core_turn.repair_request(failed, 4)

        self.assertIn("exit status: 1", request)
        self.assertIn("probed from the tree", request)
        self.assertLess(len(request), agent_core_turn.REPAIR_OUTPUT_CHARS + 800)


class TheCodingCoreIsUnchanged(unittest.TestCase):
    """The repair pass is SPEAR's; the core it runs is the one v0.3.0 shipped."""

    def test_the_core_tree_is_the_released_one(self):
        def tree(revision):
            done = subprocess.run(["git", "rev-parse", revision], cwd=ROOT,
                                  capture_output=True, text=True)

            return done.stdout.strip() if done.returncode == 0 else ""

        released = tree("v0.3.0:spear/agent")
        current = tree("HEAD:client/agent")

        if not released or not current:
            self.skipTest("not a git checkout carrying the v0.3.0 tag")

        dirty = subprocess.run(["git", "status", "--porcelain", "--", "agent"], cwd=ROOT,
                               capture_output=True, text=True).stdout

        self.assertEqual(current, released)
        self.assertEqual(dirty, "")


if __name__ == "__main__":
    unittest.main()
