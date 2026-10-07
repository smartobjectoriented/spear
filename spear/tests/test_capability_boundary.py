"""The gateway's boundary fails closed.

A terminal command addressed to the gateway is answered by it -- a result or
a refusal -- and never reaches a shell, whatever is wrong with it. A command
that only mentions the gateway's name is an ordinary command. Arguments are
data: split into words and parsed as JSON, never expanded.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import capabilities as cap
import capability_gateway as gw
import context_selection as cs
import mcp_provider
from tests.test_capabilities import SERVER, config, gateway
from tracing import EventType

OWNER = "spear-capability invoke tracker/lookup_component '{\"name\": \"uart\"}'"


class Recognition(unittest.TestCase):
    RECOGNISED = ("spear-capability list", "  spear-capability\tlist",
                  "cd x && spear-capability list", "x=1; spear-capability list",
                  "ls | spear-capability list", "echo $(spear-capability list)",
                  "echo `spear-capability list`", "ls\nspear-capability list",
                  "(spear-capability list)")
    ORDINARY = ("grep -rn spear-capability README.md", "./spear-capability list",
                "/usr/bin/spear-capability list", "spear-capabilty list",
                "spear-capability-x list", "spear‐capability list",
                "echo 'see the spear-capability docs'", "make spear-capability.o")

    def test_where_a_shell_would_run_it(self):
        for command in self.RECOGNISED:
            with self.subTest(command=command):
                self.assertTrue(gw.is_capability_command(command))

    def test_anywhere_else_it_is_text(self):
        for command in self.ORDINARY:
            with self.subTest(command=command):
                self.assertFalse(gw.is_capability_command(command))


class FailsClosed(unittest.TestCase):
    """Every way a recognised command can go wrong, through SpearHost."""

    def host(self, door):
        from control_plane import SpearHost

        self.ran = []
        return SpearHost(workspace_root=tempfile.mkdtemp(), authorize=lambda *a: "refused",
                         resolve=lambda path, purpose: path, write=lambda *a: None,
                         delete=lambda *a: None, run=lambda *a: self.ran.append(a),
                         record=lambda record: None, capabilities=door)

    def answered(self, door, command):
        from agent import tools

        host = self.host(door)
        self.assertIsNone(host.authorize("terminal", {"command": command}))
        outcome = host.run_command(command, "script", timeout=10, output_chars=10_000)
        text, code, _ = tools.terminal_from_outcome(
            command, outcome, {"cwd": None, "env": "", "last": None}, host.workspace_root, 10)
        self.assertEqual(self.ran, [], "a shell was asked to run it")

        return code, json.loads(text)["output"]

    def test_each_failure_is_an_answer_not_a_shell_command(self):
        door, _, _, registry = gateway(config(write="refuse"))
        self.addCleanup(registry.close)

        for command, expected in (
                ("spear-capability", "usage"),
                ("spear-capability frobnicate", "usage"),
                ("spear-capability list; rm -rf /tmp/x", "on its own"),
                ("spear-capability list | sh", "on its own"),
                ("spear-capability list > /tmp/out", "on its own"),
                ("spear-capability invoke 'unterminated", "No closing quotation"),
                ("spear-capability list\nrm -rf /tmp/x", "one line"),
                ("spear-capability invoke tracker/nothing '{}'", "no such capability"),
                ("spear-capability invoke tracker/lookup_component '{\"name\": 3}'",
                 "must be string"),
                ("spear-capability invoke tracker/create_note '{\"title\": \"t\"}'",
                 "refused"),
                ("x=1; spear-capability list", "on its own")):
            with self.subTest(command=command):
                code, output = self.answered(door, command)

                self.assertNotEqual(code, 0)
                self.assertIn(expected, output)

    def test_a_provider_that_fails_or_stalls(self):
        for flags, expected in ((("--exit-on-call",), "the provider failed"),
                                (("--slow", "2"), "within"),
                                (("--malformed-call",), "not a list"),
                                (("--forget", "lookup_component"), "unknown tool")):
            with self.subTest(flags=flags):
                door, _, _, registry = gateway(config("p", *flags, timeout=0.5))
                self.addCleanup(registry.close)
                code, output = self.answered(door, OWNER.replace("tracker/", "p/"))

                self.assertEqual(code, 1)
                self.assertIn(expected, output)

    def test_an_unexpected_error_inside_the_gateway(self):
        door, _, _, registry = gateway(config())
        self.addCleanup(registry.close)

        with mock.patch.object(door, "run", side_effect=RuntimeError("boom")):
            code, output = self.answered(door, "spear-capability list")

        self.assertEqual(code, 1)
        self.assertIn("boom", output)

    def test_no_gateway_still_no_shell(self):
        code, output = self.answered(None, OWNER)

        self.assertEqual(code, 1)
        self.assertIn("No external capability", output)

    def test_an_ordinary_command_keeps_the_ordinary_path(self):
        door, _, _, registry = gateway(config())
        self.addCleanup(registry.close)
        host = self.host(door)

        # Judged by the command policy (here a stub that refuses everything),
        # not answered by the gateway.
        self.assertIn("refused", host.authorize(
            "terminal", {"command": "grep spear-capability README.md"}))


class ArgumentsAreData(unittest.TestCase):
    def setUp(self):
        self.door, _, _, registry = gateway(config())
        self.addCleanup(registry.close)

    def owner_of(self, name):
        payload = json.dumps({"name": name}).replace("'", "'\\''")
        return self.door.run(f"spear-capability invoke tracker/lookup_component '{payload}'")

    def test_shell_syntax_in_an_argument_reaches_the_provider_as_text(self):
        for name in ("$(rm -rf /tmp/x)", "`id`", "a; rm -rf /tmp/x", "*.c", "$HOME",
                     "a | b && c > d"):
            with self.subTest(name=name):
                code, output = self.owner_of(name)

                self.assertEqual(code, 0)
                self.assertIn(f"{name}: owner team-a", output)

    def test_shell_syntax_outside_quotes_is_refused(self):
        code, output = self.door.run("spear-capability invoke tracker/lookup_component "
                                     "$(cat args.json)")

        self.assertEqual(code, 2)

    def test_structure_and_size(self):
        for raw, expected in (('{"name": {"nested": 1}}', "must be string"),
                              ('{"name": ["a"]}', "must be string"),
                              ('{"name": null}', "must be string"),
                              ('{"name": true}', "must be string")):
            with self.subTest(raw=raw):
                code, output = self.door.run(
                    f"spear-capability invoke tracker/lookup_component '{raw}'")

                self.assertEqual(code, 2)
                self.assertIn(expected, output)

        huge = "x" * (gw.MAX_COMMAND_CHARS + 1)
        self.assertIn("longer than", self.door.run(f"spear-capability invoke a '{huge}'")[1])

    def test_an_invalid_call_never_reaches_the_provider(self):
        provider = self.door.registry.provider("tracker")

        with mock.patch.object(provider, "invoke", side_effect=AssertionError("called")):
            for command in ("spear-capability invoke tracker/lookup_component '{}'",
                            "spear-capability invoke tracker/lookup_component '{\"x\": 1}'",
                            "spear-capability invoke tracker/lookup_component 'nope'"):
                with self.subTest(command=command):
                    self.assertEqual(self.door.run(command)[0], 2)


class Audit(unittest.TestCase):
    def outcomes(self, trace):
        return [metadata.get("outcome") for kind, metadata in trace.events
                if kind == EventType.EXTERNAL_CAPABILITY_FAILED]

    def test_every_attempt_is_recorded_by_what_became_of_it(self):
        door, _, trace, registry = gateway(config(write="refuse"))
        self.addCleanup(registry.close)

        for command in ("spear-capability list | sh",
                        "spear-capability invoke tracker/nothing '{}'",
                        "spear-capability invoke tracker/lookup_component '{}'",
                        "spear-capability invoke tracker/create_note '{\"title\": \"t\"}'",
                        OWNER):
            door.run(command)

        self.assertEqual(self.outcomes(trace), [gw.SYNTAX, gw.UNKNOWN, gw.VALIDATION,
                                                gw.REFUSED])
        self.assertIn(EventType.EXTERNAL_CAPABILITY_INVOKED, trace.kinds())

    def test_a_large_family_records_an_undescribed_call(self):
        door, _, trace, registry = gateway(config("big", "--extra", "37"))
        self.addCleanup(registry.close)
        door.run(OWNER.replace("tracker/", "big/"))

        self.assertEqual(self.outcomes(trace), [gw.NOT_DESCRIBED])

    def test_provider_failures_are_told_apart(self):
        for flags, outcome in ((("--slow", "2"), gw.TIMEOUT),
                               (("--exit-on-call",), gw.UNAVAILABLE),
                               (("--malformed-call",), gw.PROVIDER_ERROR)):
            with self.subTest(flags=flags):
                door, _, trace, registry = gateway(config("p", *flags, timeout=0.5))
                self.addCleanup(registry.close)
                door.run(OWNER.replace("tracker/", "p/"))

                self.assertEqual(self.outcomes(trace), [outcome])

    def test_no_command_or_environment_in_the_record(self):
        door, _, trace, registry = gateway(cap.ProviderConfig(
            id="ghost", transport="stdio", command=("/srv/private/bin/server", "--token=s3cret"),
            env=(("API_TOKEN", "s3cret"),), scope=("generic",)))
        self.addCleanup(registry.close)
        recorded = json.dumps([metadata for _, metadata in trace.events])

        self.assertIn(gw.UNAVAILABLE, recorded)
        self.assertNotIn("s3cret", recorded)
        self.assertNotIn("/srv/private", recorded)


class ProviderLifecycle(unittest.TestCase):
    def test_a_server_that_exits_after_initializing(self):
        door, _, trace, registry = gateway(config("gone", "--exit-on-list"))
        self.addCleanup(registry.close)

        self.assertEqual(door.items, {})
        self.assertIn("gone", door.failed)
        self.assertEqual(door.run("spear-capability list")[0], 1)

    def test_a_listing_that_fails(self):
        door, _, _, registry = gateway(config("broken", "--list-error"), config())
        self.addCleanup(registry.close)

        self.assertIn("listing is broken", door.failed["broken"])
        self.assertEqual(sorted({item.provider for item in door.items.values()}), ["tracker"])


class ProtocolVersion(unittest.TestCase):
    def test_a_server_speaking_another_version_is_refused(self):
        provider = mcp_provider.MCPProvider(config("old", "--version", "2024-11-05"))
        self.addCleanup(provider.close)

        with self.assertRaises(cap.CapabilityProtocolError) as raised:
            provider.list()

        self.assertIn("2024-11-05", str(raised.exception))
        self.assertIsNone(provider._process)

    def test_the_version_is_sent_deliberately(self):
        self.assertEqual(mcp_provider.PROTOCOL_VERSION, "2025-06-18")


class LegacyRuntime(unittest.TestCase):
    """A standard-bound change that is not about the standard runs on the
    legacy runtime, which has no gateway: kept as it is, and said so."""

    def test_no_shell_runs_a_gateway_command_there(self):
        import rag_chat

        result = rag_chat.run_cmd_result("spear-capability list", need_confirm=False)

        self.assertEqual(result.status, "denied")
        self.assertIn("not available on this runtime", result.summary)

    def test_the_admission_is_recorded_not_dropped(self):
        import rag_chat
        from tests.test_capabilities import Recorder

        root = tempfile.mkdtemp()
        file = Path(tempfile.mkdtemp(), "capabilities.json")
        file.write_text(json.dumps({"providers": [
            {"id": "tracker", "command": [sys.executable, SERVER], "scope": "global"}]}))
        trace = Recorder()
        rag_chat._CAPABILITY_REGISTRY.clear()
        self.addCleanup(rag_chat._CAPABILITY_REGISTRY.clear)

        with mock.patch.multiple(
                rag_chat, PROJECT=f"workspace:{root}", PROJECT_ROOT=root, CORPUS_ROOT=root,
                SKILLS_DIR=tempfile.mkdtemp(), RULES_DIR=tempfile.mkdtemp(),
                LEARNED_RULES_FILE="/nonexistent", CTX_LIMIT=200_000,
                CAPABILITIES_FILE=str(file)), \
                mock.patch.object(rag_chat, "load_projects", lambda: {}):
            _, _, rendered = rag_chat.select_turn_context(
                user_input="Rename cnt to count in main.c.", turn_scope="IMPLEMENTATION",
                binding={"standard_id": "SYNTH-1", "revision": "1"}, write=True,
                project_spec={}, project_commands=None, history_text="", memories="",
                skills=[], retrieval="", system_instructions="", system_source="",
                tool_guide="", working_directory="", trace=trace)

        chosen = rendered[cs.IMPLEMENTATION]
        self.assertIsNone(chosen["gateway"])
        self.assertEqual(chosen["capabilities_unsupported"], ("tracker",))
        self.assertIn("UNSUPPORTED", [metadata.get("mode") for kind, metadata in trace.events
                                      if kind == EventType.CAPABILITY_FAMILY_SELECTED])


if __name__ == "__main__":
    unittest.main()
