import fcntl
import contextlib
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tool_runtime

from tool_runtime import (
    AuditLogger,
    AuthorizationResult,
    Capability,
    CapabilityPolicy,
    CgroupAvailability,
    CgroupLimits,
    CommandClassification,
    CommandPolicy,
    CommandRunner,
    DEFAULT_CAPABILITY_POLICY,
    DEFAULT_CGROUP_LIMITS,
    DEFAULT_RESOURCE_LIMITS,
    ExecutionMode,
    ExecutionProfile,
    NetworkBackend,
    PathNotFoundError,
    PathPolicyError,
    ResourceLimits,
    SystemdScopeRunner,
    ToolResult,
    ToolPolicy,
    Workspace,
)


class RagChatWorkspaceIntegrationTests(unittest.TestCase):
    """Step 2 coverage: filesystem helpers must use the Workspace boundary."""

    @classmethod
    def setUpClass(cls):
        import rag_chat

        cls.rag_chat = rag_chat

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "workspace"
        self.root.mkdir()
        self.outside = Path(self.temp.name) / "outside"
        self.outside.mkdir()
        (self.root / "inside.txt").write_text("inside")
        (self.outside / "secret.txt").write_text("outside")
        self.old_project_root = self.rag_chat.PROJECT_ROOT
        self.old_workspace = self.rag_chat.WORKSPACE
        self.old_execution_mode = self.rag_chat.EXECUTION_MODE
        self.old_audit_logger = getattr(self.rag_chat, "AUDIT_LOGGER", None)
        self.audit_log = Path(self.temp.name) / "audit.jsonl"
        self.rag_chat.PROJECT_ROOT = str(self.root)
        self.rag_chat.WORKSPACE = Workspace.from_path(self.root)
        if self.old_audit_logger is not None:
            self.rag_chat.AUDIT_LOGGER = AuditLogger(self.audit_log)

    def tearDown(self):
        self.rag_chat.PROJECT_ROOT = self.old_project_root
        self.rag_chat.WORKSPACE = self.old_workspace
        self.rag_chat.EXECUTION_MODE = self.old_execution_mode
        if self.old_audit_logger is not None:
            self.rag_chat.AUDIT_LOGGER = self.old_audit_logger
        self.temp.cleanup()

    def test_evidence_visibility_follows_the_conversation_not_the_turn(self):
        """What the next request will actually carry, not what once happened.

        The router asks this before telling a model its observation is
        "already above in this conversation". After a compaction rewrote the
        conversation, that sentence is false and the only move it leaves is
        another spelling of the same read.
        """

        from model_backend import ConversationMessage, TextBlock, ToolResultBlock

        class Context:
            conversation = [
                ConversationMessage("assistant", (TextBlock("reading it"),)),
                ConversationMessage("user", (ToolResultBlock("call-1", "def add"),)),
            ]

        context = Context()
        self.assertTrue(self.rag_chat._evidence_in_context(context, "call-1"))
        self.assertFalse(self.rag_chat._evidence_in_context(context, "call-2"))

        # Compaction replaces the exchange with a summary.

        context.conversation = [
            ConversationMessage("user", (TextBlock("[earlier work summarised]"),)),
        ]
        self.assertFalse(self.rag_chat._evidence_in_context(context, "call-1"))

        # An empty result block is not evidence either.

        context.conversation = [
            ConversationMessage("user", (ToolResultBlock("call-1", ""),)),
        ]
        self.assertFalse(self.rag_chat._evidence_in_context(context, "call-1"))

    def test_a_read_only_refusal_is_about_scope_not_permissions(self):
        """"Command denied" invites the next spelling of the same intent.

        A run asked only which file defines `add` tried `sed -i`, was told it
        lacked permission, and spent thirty-five more steps finding other
        ways. The refusal has to say what it is FOR, and what to do instead.
        """

        self.assertIn("read-only task", self.rag_chat.READ_ONLY_REFUSAL)
        self.assertIn("Do not try another way to modify or test",
                      self.rag_chat.READ_ONLY_REFUSAL)
        self.assertIn("Answer the original question",
                      self.rag_chat.READ_ONLY_REFUSAL)

        # Counted by category, not by signature: every spelling is the same
        # violation of the same scope.

        source = Path(self.rag_chat.__file__).read_text()
        self.assertIn("READ_ONLY_VIOLATIONS", source)
        self.assertIn("EventType.READ_ONLY_VIOLATION", source)
        self.assertIn("read_only_violations", source)

    def test_read_rejects_traversal_and_symlink_escapes(self):
        (self.root / "outside-link").symlink_to(self.outside, target_is_directory=True)
        self.assertIn("path escapes the workspace", self.rag_chat.read_file("../outside/secret.txt"))
        self.assertIn("path escapes the workspace",
                      self.rag_chat.read_file("outside-link/secret.txt"))

    def test_file_lookup_does_not_fall_back_after_rejected_path(self):
        self.assertIsNone(self.rag_chat.find_file("../outside/secret.txt"))

    def test_working_directory_note_matches_what_bash_actually_sees(self):
        """The note must name the sandbox mount, not only the host path.

        Announcing only the host directory sent a real session chasing
        `ls /opt/llm/spear` from inside the sandbox, where that path does
        not exist and `pwd` prints the mount instead.
        """
        source = Path(self.rag_chat.__file__).read_text()
        note = source.split("\n        mount = sandbox_mount()", 1)[1].split(
            "## Retrieved Context", 1)[0]
        # The note interpolates the computed mount rather than a literal.
        self.assertIn("{mount}", note)
        self.assertIn("PROJECT_ROOT", note)
        for expected in ("RELATIVE", "pwd", "parent"):
            self.assertIn(expected, note, f"note should mention {expected}")
        # The mount is asked of the sandbox, never restated by hand: the note
        # and the bind must not be able to drift apart.
        self.assertEqual(
            self.rag_chat.sandbox_mount(),
            tool_runtime.BubblewrapSandbox().mount_root(self.rag_chat.WORKSPACE))
        self.assertNotIn('"/workspace"', note)

    def test_corpus_mention_is_reported_but_never_acted_on(self):
        so3 = Path(self.temp.name) / "so3"
        (so3 / "usr").mkdir(parents=True)
        deeper = Path(self.temp.name) / "micropython-so3"
        deeper.mkdir()
        projects = {
            "so3": {"path": str(so3), "kind": "generic"},
            "micropython-so3": {"path": str(deeper), "kind": "generic"},
            "src": {"path": str(so3), "kind": "generic"},
            "gone": {"path": str(Path(self.temp.name) / "absent"), "kind": "generic"},
        }
        hint = self.rag_chat.corpus_mention_hint

        try:
            self.rag_chat._HINTED_CORPORA.clear()
            # The case that cost a turn: a registered name in the question
            # while the tools are elsewhere.
            message = hint("generate a ping.c to run in so3", projects, "spear")
            self.assertIsNotNone(message)
            self.assertIn("so3", message)
            self.assertIn(str(so3), message)

            # Once per name per session — informs, does not nag.
            self.assertIsNone(hint("still about so3", projects, "spear"))

            # The longest registered name wins over a substring of it.
            self.rag_chat._HINTED_CORPORA.clear()
            message = hint("port micropython-so3 please", projects, "spear")
            self.assertIn("micropython-so3", message)

            # Never fires for the corpus already in use.
            self.rag_chat._HINTED_CORPORA.clear()
            self.assertIsNone(hint("something about so3", projects, "so3"))

            # Ordinary directory words and path fragments are not mentions.
            self.assertIsNone(hint("look in src for the parser", projects, "spear"))
            self.assertIsNone(hint("open so3/usr/main.c", projects, "spear"))
            # Substrings of longer words are not mentions either.
            self.assertIsNone(hint("the so3xyz variant", projects, "spear"))
            # A registry entry whose tree has gone is not offered.
            self.assertIsNone(hint("what about gone", projects, "spear"))

            # Above all: reporting must not move the workspace.
            before = (self.rag_chat.PROJECT_ROOT, self.rag_chat.WORKSPACE.root)
            self.rag_chat._HINTED_CORPORA.clear()
            hint("build ping.c for so3", projects, "spear")
            self.assertEqual((self.rag_chat.PROJECT_ROOT,
                              self.rag_chat.WORKSPACE.root), before)
        finally:
            self.rag_chat._HINTED_CORPORA.clear()

    def test_file_lookup_walks_the_workspace_for_a_bare_basename(self):
        # The rejected-path test above returns before the tree walk, which is
        # how a missing `pathlib` import survived in that walk until a real
        # session hit it.  This drives the walk itself.
        nested = self.root / "sub" / "deeper"
        nested.mkdir(parents=True)
        (nested / "ping.c").write_text("int main(void) { return 0; }")
        found = self.rag_chat.find_file("ping.c")
        self.assertIsNotNone(found)
        self.assertEqual(Path(found).name, "ping.c")
        self.assertTrue(Path(found).is_file())
        # A basename that exists nowhere still resolves to None, not an error.
        self.assertIsNone(self.rag_chat.find_file("absent-from-the-tree.c"))

    def test_write_file_rejects_absolute_and_outside_paths_before_writing(self):
        # In a session that MAY write: the path diagnostic belongs to the
        # tool's own validation, and the execution-mode gate now runs ahead
        # of the handler, so in safe mode the mode is reported instead. The
        # file is not written either way -- the case below holds that.
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK

        with patch.object(self.rag_chat, "confirm", return_value=True):
            result = self.rag_chat.execute_tool(
                "write_file", {"path": str(self.outside / "new.txt"), "content": "x"}, {}
            )
        # The reason is now the accurate one: this path is outside every
        # declared root, which is a stronger statement than "it is absolute".
        # An absolute path INSIDE the primary root still reports the form —
        # see MultiRootWorkspaceTests.
        self.assertIn("path escapes the workspace", result)
        self.assertFalse((self.outside / "new.txt").exists())
        event = json.loads(self.audit_log.read_text().strip())
        self.assertEqual((event["action"], event["status"]), ("write_file", "invalid_path"))

    def test_an_escaping_path_is_written_in_no_mode_at_all(self):
        """The safety property, which does not depend on which gate spoke."""
        for mode in (ExecutionMode.SAFE, ExecutionMode.ASK, ExecutionMode.AUTO):
            with self.subTest(mode=mode):
                self.rag_chat.EXECUTION_MODE = mode
                target = self.outside / f"new-{mode}.txt"

                with patch.object(self.rag_chat, "confirm", return_value=True):
                    result = self.rag_chat.execute_tool(
                        "write_file", {"path": str(target), "content": "x"}, {})

                self.assertTrue(result.startswith("ERROR"))
                self.assertFalse(target.exists())

    def test_write_file_creates_missing_parent_directories(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK
        with patch.object(self.rag_chat, "confirm", return_value=True):
            result = self.rag_chat.execute_tool(
                "write_file", {"path": "usr/src/ping.c", "content": "int main(void){}"}, {})
        # Refusing `usr/src/` while allowing `usr/src/ping.c` was arbitrary:
        # the containment check already covers the resolved path.
        self.assertIn("OK", result)
        self.assertTrue((self.root / "usr" / "src" / "ping.c").is_file())

    def test_write_file_creates_nothing_when_authorization_is_refused(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK
        with patch.object(self.rag_chat, "confirm", return_value=False):
            self.rag_chat.execute_tool(
                "write_file", {"path": "denied/deep/x.c", "content": "x"}, {})
        # Creating directories IS a mutation: it must happen after the gate,
        # never as a side effect of preparing the write.
        self.assertFalse((self.root / "denied").exists())

    def test_write_file_creates_no_directory_outside_a_root(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK
        with patch.object(self.rag_chat, "confirm", return_value=True):
            result = self.rag_chat.execute_tool(
                "write_file",
                {"path": str(self.outside / "deep" / "x.c"), "content": "x"}, {})
        self.assertIn("escapes the workspace", result)
        self.assertFalse((self.outside / "deep").exists())

    def test_edit_and_append_still_require_an_existing_file(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK
        with patch.object(self.rag_chat, "confirm", return_value=True):
            for tool in ("edit_file", "append_file"):
                result = self.rag_chat.execute_tool(
                    tool, {"path": "absent/deep/x.c", "content": "x",
                           "old_text": "a", "new_text": "b"}, {})
                self.assertIn("ERROR", result, tool)
        self.assertFalse((self.root / "absent").exists())

    def test_write_file_stays_inside_workspace(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK
        with patch.object(self.rag_chat, "confirm", return_value=True):
            result = self.rag_chat.execute_tool(
                "write_file", {"path": "created.txt", "content": "inside"}, {}
            )
        self.assertTrue(result.startswith("OK:"))
        self.assertEqual((self.root / "created.txt").read_text(), "inside\n")

    def test_safe_mode_denies_mutation_before_confirmation(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.SAFE
        with patch.object(self.rag_chat, "confirm") as confirm:
            result = self.rag_chat.execute_tool(
                "write_file", {"path": "blocked.txt", "content": "x"}, {}
            )
        # The refusal names the mode and offers no other way to write. It is
        # now the router's, made from the tool's own execution_modes before
        # the handler is reached at all; the handler keeps its own check for
        # callers that do not cross the router.
        self.assertIn("safe mode", result)
        self.assertNotIn("use edit_file", result)
        self.assertNotIn("--auto", result)
        confirm.assert_not_called()
        self.assertFalse((self.root / "blocked.txt").exists())

    def test_ask_mode_requires_confirmation_for_mutation(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK
        with patch.object(self.rag_chat, "confirm", return_value=False):
            result = self.rag_chat.execute_tool(
                "write_file", {"path": "not-approved.txt", "content": "x"}, {}
            )
        self.assertEqual(result, "CANCELLED")
        self.assertFalse((self.root / "not-approved.txt").exists())

    def test_auto_mode_allows_workspace_mutation(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
        with patch.object(self.rag_chat, "confirm") as confirm:
            result = self.rag_chat.execute_tool(
                "write_file", {"path": "auto.txt", "content": "x"}, {}
            )
        self.assertTrue(result.startswith("OK:"))
        confirm.assert_not_called()
        self.assertEqual((self.root / "auto.txt").read_text(), "x\n")

    def test_mutating_tool_actions_are_audited_without_file_contents(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
        result = self.rag_chat.execute_tool(
            "write_file", {"path": "audited.txt", "content": "TOP SECRET CONTENT"}, {}
        )
        self.assertTrue(result.startswith("OK:"))
        event = json.loads(self.audit_log.read_text().strip())
        self.assertEqual(event["action"], "write_file")
        self.assertEqual(event["paths"], ["audited.txt"])
        self.assertEqual(event["status"], "ok")
        self.assertNotIn("TOP SECRET CONTENT", self.audit_log.read_text())

    def test_denied_mutation_attempt_is_audited(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.SAFE
        self.rag_chat.execute_tool(
            "write_file", {"path": "denied.txt", "content": "x"}, {}
        )
        event = json.loads(self.audit_log.read_text().strip())
        self.assertEqual(event["action"], "write_file")
        self.assertEqual(event["status"], "denied")
        self.assertFalse(event["approved"])

    def test_safe_mode_allows_simple_read_only_command_without_shell(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.SAFE
        with patch.object(self.rag_chat.COMMAND_RUNNER, "run_simple",
                          return_value=ToolResult("ok", "command completed", stdout="ok")) as run:
            result = self.rag_chat.run_cmd("pwd", need_confirm=False)
        # pwd really runs in the sandbox: it prints the mount, which is now the
        # workspace's own host path.
        self.assertEqual(result, self.rag_chat.sandbox_mount() + "\n")
        run.assert_not_called()

    def test_safe_mode_rejects_mutation_but_allows_read_only_pipelines(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.SAFE
        self.assertIn("workspace_mutating commands are disabled in safe mode",
                      self.rag_chat.run_cmd("make test", need_confirm=False))
        # A pipeline that writes is refused on the capability, not the class.
        self.assertIn("required capabilities are not allowed",
                      self.rag_chat.run_cmd("echo x > f", need_confirm=False))
        # A read-only pipeline is authorised: it reaches execution instead of
        # being turned away by the policy. Whether the sandbox is available in
        # this environment is a separate concern, so assert on the refusal
        # messages being gone rather than on the output.
        outcome = self.rag_chat.run_cmd("ls | head", need_confirm=False)
        self.assertNotIn("disabled in safe mode", outcome)
        self.assertNotIn("required capabilities are not allowed", outcome)

    def test_ask_mode_fails_closed_for_complex_command_without_sandbox(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK
        old_sandbox = self.rag_chat.COMMAND_RUNNER.sandbox
        self.rag_chat.COMMAND_RUNNER.sandbox = tool_runtime.BubblewrapSandbox(
            binary="/definitely/missing/bwrap"
        )
        try:
            with patch.object(self.rag_chat, "confirm", return_value=True), \
                 patch.object(self.rag_chat.COMMAND_RUNNER, "run_simple") as simple, \
                 patch.object(self.rag_chat.COMMAND_RUNNER, "run_complex_approved") as complex_run:
                result = self.rag_chat.run_cmd("ls | head", need_confirm=False)
            self.assertIn("sandbox unavailable", result)
            simple.assert_not_called()
            complex_run.assert_not_called()
        finally:
            self.rag_chat.COMMAND_RUNNER.sandbox = old_sandbox

    def test_auto_mode_fails_closed_for_complex_commands_without_sandbox(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
        old_sandbox = self.rag_chat.COMMAND_RUNNER.sandbox
        self.rag_chat.COMMAND_RUNNER.sandbox = tool_runtime.BubblewrapSandbox(
            binary="/definitely/missing/bwrap"
        )
        try:
            self.assertIn("sandbox unavailable",
                          self.rag_chat.run_cmd("ls | head", need_confirm=False))
        finally:
            self.rag_chat.COMMAND_RUNNER.sandbox = old_sandbox


class RagChatCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rag_chat

        cls.rag_chat = rag_chat

    def test_openai_tool_schema_is_preserved(self):
        tools = self.rag_chat.TOOLS
        self.assertEqual([tool["function"]["name"] for tool in tools], [
            "bash", "edit_file", "write_file", "append_file", "delete_file",
            "remember", "search_corpus", "search_internet", "fetch_url",
        ])
        self.assertTrue(all(tool["type"] == "function" for tool in tools))
        self.assertEqual(tools[0]["function"]["parameters"]["required"], ["command"])

    def test_chat_completion_keeps_provider_contract(self):
        class FakeCompletions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return []

        completions = FakeCompletions()
        client = type("Client", (), {
            "chat": type("Chat", (), {"completions": completions})()
        })()
        text, calls = self.rag_chat.chat_once(
            client, [{"role": "user", "content": "hello"}], use_tools=True
        )
        self.assertEqual((text, calls), ("", []))
        self.assertTrue(completions.kwargs["stream"])
        self.assertEqual(completions.kwargs["tools"], self.rag_chat.TOOLS)
        self.assertEqual(completions.kwargs["extra_body"]["chat_template_kwargs"],
                         {"enable_thinking": False})
        self.assertEqual(completions.kwargs["model"], "qwen3")

    def test_rag_and_agent_loop_constants_are_unchanged(self):
        # Anchored on the module's own directory, not an absolute literal: the
        # literal went stale when the tree was relocated and startup broke.
        app_dir = os.path.dirname(os.path.realpath(self.rag_chat.__file__))
        self.assertEqual(self.rag_chat.APP_DIR, app_dir)
        self.assertEqual(self.rag_chat.ROOT_DIR, os.path.dirname(app_dir))
        self.assertEqual(self.rag_chat.DB_PATH, os.path.join(app_dir, "chromadb"))
        self.assertNotIn("/opt/llm/spear/spear",
                         Path(self.rag_chat.__file__).read_text())
        self.assertEqual(self.rag_chat.TOP_K, 12)
        self.assertEqual(self.rag_chat.MAX_CONTEXT_CHARS, 12000)
        self.assertEqual(self.rag_chat.MAX_HISTORY, 80)
        self.assertEqual(self.rag_chat.HISTORY_INJECT, 40)
        # The agent-loop budget is deliberately NOT pinned to a literal here.
        # It used to assert "MAX_TOOL_ROUNDS = 8", which froze a number without
        # testing anything: 8 rounds cut off legitimate investigation — finding
        # how a project builds costs more than eight commands — and the turn
        # ended mid-work. What matters is that a budget exists, is generous
        # enough to finish a real task, and can be tuned per run.
        source = Path(self.rag_chat.__file__).read_text()
        self.assertIn("SPEAR_MAX_TOOL_ROUNDS", source)
        self.assertIn("SPEAR_MAX_COMMANDS", source)
        for var, floor in (("SPEAR_MAX_TOOL_ROUNDS", 20),
                           ("SPEAR_MAX_COMMANDS", 40)):
            m = re.search(rf'{var}", "(\d+)"', source)
            self.assertIsNotNone(m, f"{var} has no default")
            self.assertGreaterEqual(int(m.group(1)), floor,
                                    f"{var} is too small to finish a real task")

    def test_absolute_path_flag_is_workspace_limited(self):
        source = Path(self.rag_chat.__file__).read_text()
        self.assertIn('allow_absolute_paths="--allow-absolute-paths" in sys.argv[1:]',
                      source)

    def test_banner_names_the_backend_for_all_three(self):
        label = self.rag_chat.backend_label
        self.assertEqual(label("anthropic", "http://127.0.0.1:8080/v1"), "anthropic API")
        self.assertEqual(label("openai-compatible", "http://127.0.0.1:8081/v1"),
                         "remote/pod vLLM")
        # Generic: a port number cannot name a host. This used to assert
        # "reds-server (RTX PRO 6000)" and kept claiming it after reds.conf
        # was repointed at another machine.
        self.assertEqual(label("openai-compatible", "http://127.0.0.1:8082/v1"),
                         "reds tunnel")
        self.assertEqual(label("openai-compatible", "http://127.0.0.1:8080/v1"),
                         "local llama-server")
        # The provider decides before the endpoint: an Anthropic run that
        # inherited a stale SPEAR_API_BASE must not be labelled remote.
        self.assertEqual(label("anthropic", "http://127.0.0.1:8081/v1"), "anthropic API")

    def test_banner_prefers_the_label_the_launcher_supplies(self):
        """The launcher knows which host it tunnelled to; rag_chat does not."""
        prev = os.environ.get("SPEAR_BACKEND_LABEL")
        os.environ["SPEAR_BACKEND_LABEL"] = "gpu-host.example (tunnel :8082)"
        try:
            self.assertEqual(
                self.rag_chat.backend_label("openai-compatible",
                                            "http://127.0.0.1:8082/v1"),
                "gpu-host.example (tunnel :8082)")
        finally:
            os.environ.pop("SPEAR_BACKEND_LABEL", None)
            if prev is not None:
                os.environ["SPEAR_BACKEND_LABEL"] = prev

    def test_an_empty_label_falls_back_to_the_port_heuristic(self):
        prev = os.environ.get("SPEAR_BACKEND_LABEL")
        os.environ["SPEAR_BACKEND_LABEL"] = "   "
        try:
            self.assertEqual(
                self.rag_chat.backend_label("openai-compatible",
                                            "http://127.0.0.1:8080/v1"),
                "local llama-server")
        finally:
            os.environ.pop("SPEAR_BACKEND_LABEL", None)
            if prev is not None:
                os.environ["SPEAR_BACKEND_LABEL"] = prev

    def test_backend_picker_prefers_flags_then_memory_then_prompt(self):
        import backend_select

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(backend_select, "STATE_FILE", os.path.join(tmp, "b.conf")):
                # An explicit flag is never second-guessed by a prompt.
                for argv, expected in (
                    # The third element says whether the choice is worth
                    # remembering. A flag never is: it speaks for this run.
                    (["--remote"], ("remote", "", False)),
                    (["--reds"], ("reds", "", False)),
                    (["--pod"], ("remote", "", False)),
                    (["--local"], ("local", "", False)),
                    (["--provider", "anthropic"], ("anthropic", "", False)),
                    (["--provider=anthropic", "--model=claude-opus-5"],
                     ("anthropic", "claude-opus-5", False)),
                ):
                    self.assertEqual(
                        backend_select.choose(
                            argv, isatty=True,
                            reader=lambda _: (_ for _ in ()).throw(
                                AssertionError("must not prompt"))),
                        expected, argv)

                # First run with no state: local, and no prompt off a TTY.
                self.assertEqual(backend_select.choose([], isatty=False), ("local", "", False))

                # The remembered choice comes back preselected, and Enter takes it.
                backend_select.save_choice("anthropic", "claude-opus-5")
                self.assertEqual(backend_select.load_last_choice(),
                                 ("anthropic", "claude-opus-5"))
                self.assertEqual(backend_select.choose([], isatty=False),
                                 ("anthropic", "claude-opus-5", False))
                out = io.StringIO()
                self.assertEqual(
                    backend_select.choose([], isatty=True, reader=lambda _: "", out=out),
                    ("anthropic", "claude-opus-5", True))
                menu = out.getvalue()
                self.assertIn("last used", menu.splitlines()[1])
                for name in backend_select.BACKENDS:
                    self.assertIn(name, menu)

                # Picking another entry by number drops the remembered model.
                out = io.StringIO()
                chosen = backend_select.choose([], isatty=True,
                                               reader=lambda _: "2", out=out)
                self.assertIn(chosen[0], ("local", "remote"))
                self.assertEqual(chosen[1], "")

                # An interrupted or unreadable answer keeps the last choice.
                def _eof(_):
                    raise EOFError

                self.assertEqual(
                    backend_select.choose([], isatty=True, reader=_eof,
                                          out=io.StringIO()),
                    ("anthropic", "claude-opus-5", True))

                # The contract with the launcher, which captures stdout and
                # nothing else: the token goes there, everything a human reads
                # goes to `out`. input()'s own prompt argument breaks it --
                # input() writes the prompt to stdout -- and the launcher then
                # captured "  reds", matched no case, and silently started the
                # local server instead of the chosen one.
                menu = io.StringIO()
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    backend_select.choose([], isatty=True,
                                          reader=lambda _: "1", out=menu)
                self.assertEqual(captured.getvalue(), "",
                                 "stdout carries the token and nothing else")
                self.assertIn("Backend:", menu.getvalue())

                # The rule the whole flag exists for: one `spear-chat --local`
                # must not become the default for every later launch. main()
                # persists only what the menu returned.
                backend_select.save_choice("reds")
                backend_select.main(["--local"])
                self.assertEqual(backend_select.load_last_choice(), ("reds", ""),
                                 "an explicit flag must not rewrite the default")
                backend_select.main([])          # off a TTY: reuse, do not rewrite
                self.assertEqual(backend_select.load_last_choice(), ("reds", ""))

                # Listing options must not probe: no network, no SSH.
                with patch.object(backend_select, "_read_conf", return_value=""):
                    for name in backend_select.BACKENDS:
                        self.assertIsInstance(backend_select.describe(name), str)

                # Every backend must be reachable by a flag, listed by the
                # picker, documented in --help, and given its own tunnel port —
                # a new one that lands in only some of those is half-wired.
                help_text = self.rag_chat.HELP_TEXT
                for name in backend_select.BACKENDS:
                    self.assertIn(name, help_text, f"{name} missing from --help")
                launcher = Path(backend_select.APP_DIR, "spear-chat.sh").read_text()
                selectors = {"local": "--local", "remote": "--remote",
                             "reds": "--reds", "anthropic": "--provider anthropic"}
                self.assertEqual(set(selectors), set(backend_select.BACKENDS),
                                 "a backend was added without a selector")
                for name, flag in selectors.items():
                    self.assertIn(flag.split()[0], launcher,
                                  f"{name}: {flag} unknown to the launcher")
                ports = re.findall(r"LPORT=(\d+)", launcher)
                self.assertEqual(len(ports), len(set(ports)),
                                 f"tunnel ports collide: {ports}")

    def test_help_documents_every_flag_the_cli_actually_parses(self):
        import inspect
        import re as _re

        help_text = self.rag_chat.HELP_TEXT
        # Derive the flags from the parsers themselves rather than restating a
        # list here: a hand-written help drifts silently, and the whole point
        # of adding it was that the flags were undiscoverable.
        parsed = set()
        for fn in (self.rag_chat.execution_mode_from_argv,
                   self.rag_chat.capability_policy_from_argv,
                   self.rag_chat.model_provider_from_argv):
            parsed |= set(_re.findall(r'"(--?[a-z][a-z-]*)"', inspect.getsource(fn)))
        self.assertIn("--auto", parsed)  # the derivation itself must work
        undocumented = sorted(f for f in parsed if f not in help_text)
        self.assertEqual(undocumented, [], f"flags missing from --help: {undocumented}")

        # The corpus selectors live in startup resolution, same requirement.
        for flag in ("--corpus", "--project", "--checkout", "--here", "--help"):
            self.assertIn(flag, help_text)

        # --help must answer without resolving a corpus or touching a server.
        with patch.object(self.rag_chat.sys, "argv", ["rag_chat.py", "--help"]), \
             patch.object(self.rag_chat, "banner_art",
                          side_effect=AssertionError("must not start a session")), \
             patch.object(self.rag_chat, "set_project",
                          side_effect=AssertionError("must not resolve a corpus")), \
             patch.object(self.rag_chat, "print_help") as printed:
            self.rag_chat.main()
        printed.assert_called_once_with()

    def test_banner_names_the_real_mode_and_the_flag_that_changes_it(self):
        # The banner is where a user learns which mode they are in; it used to
        # branch on the legacy BYPASS_PERMISSIONS boolean and announce SAFE —
        # the default — as "ask before each edit/run", a mode SAFE does not
        # have.  Each line must also name the flags that reach the other modes.
        for mode, expected, flags in (
            (ExecutionMode.SAFE, "safe (default)", ("--ask", "--auto")),
            (ExecutionMode.ASK, "ask before each edit/run", ("--safe", "--auto")),
            (ExecutionMode.AUTO, "without asking", ("--ask", "--safe",
                                                    "--no-network")),
        ):
            row = self.rag_chat.permissions_row(mode)
            self.assertIn(expected, row, f"{mode}: {row}")
            for flag in flags:
                self.assertIn(flag, row, f"{mode} lacks {flag}: {row}")
        # SAFE must never claim it will ask; ASK must never claim it is silent.
        self.assertNotIn("ask before each",
                         self.rag_chat.permissions_row(ExecutionMode.SAFE))
        self.assertNotIn("(default)",
                         self.rag_chat.permissions_row(ExecutionMode.AUTO))

    def test_session_settings_are_reachable_as_flags(self):
        """They were environment variables only. Nothing in --help mentioned
        them, and configuring one run meant `VAR=x VAR2=y spear-chat` — neither
        discoverable nor reviewable next to the other flags."""
        import os

        environment = {}
        with unittest.mock.patch.dict(os.environ, environment, clear=False):
            left = self.rag_chat.apply_env_options(
                ["--safe", "--ctx", "65536", "--standard-embed-remote",
                 "gpu@example.invalid", "--trace", "--corpus", "so3"])

            # the settings are consumed; everything else is untouched
            self.assertEqual(["--safe", "--corpus", "so3"], left)
            self.assertEqual("65536", os.environ["SPEAR_CTX"])
            self.assertEqual("gpu@example.invalid",
                             os.environ["SPEAR_STANDARD_EMBED_REMOTE"])
            self.assertEqual("1", os.environ["SPEAR_TRACE"])

    def test_a_settings_flag_without_a_value_is_refused(self):
        with self.assertRaises(SystemExit) as raised:
            self.rag_chat.apply_env_options(["--ctx"])

        self.assertIn("--ctx", str(raised.exception))

    def test_every_settings_flag_is_documented_by_its_own_table(self):
        """The help text is rendered FROM the parser's table, so a flag added
        without a line in --help cannot happen."""
        rendered = self.rag_chat._env_option_lines()

        for flag, (variable, _) in self.rag_chat.ENV_OPTIONS.items():
            self.assertIn(flag, rendered)
            self.assertIn(variable, rendered)

    def test_legacy_mode_flags_and_no_network_restriction(self):
        for flag, mode in (("--safe", ExecutionMode.SAFE), ("--ask", ExecutionMode.ASK),
                           ("--confirm", ExecutionMode.ASK), ("--auto", ExecutionMode.AUTO),
                           ("--yolo", ExecutionMode.AUTO)):
            self.assertEqual(self.rag_chat.execution_mode_from_argv([flag]), mode)
        policy = self.rag_chat.capability_policy_from_argv(["--ask", "--no-network"])
        self.assertNotIn(Capability.NETWORK, policy.ask)
        # --no-network has to reach AUTO too, now that AUTO has the network:
        # stripping it from ASK alone would leave the flag doing nothing in
        # the one mode where it is not confirmed command by command.
        self.assertNotIn(Capability.NETWORK, policy.auto)
        self.assertIn(Capability.NETWORK,
                      self.rag_chat.capability_policy_from_argv(["--ask"]).ask)
        self.assertIn(Capability.NETWORK,
                      self.rag_chat.capability_policy_from_argv([]).auto)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "workspace"
        self.root.mkdir()
        self.outside = Path(self.temp.name) / "outside"
        self.outside.mkdir()
        (self.root / "inside.txt").write_text("inside")
        (self.outside / "secret.txt").write_text("outside")

    def tearDown(self):
        self.temp.cleanup()

    def test_relative_path_inside_workspace_resolves(self):
        workspace = Workspace.from_path(self.root)
        self.assertEqual(workspace.resolve("inside.txt"), self.root / "inside.txt")

    def test_parent_traversal_is_rejected(self):
        workspace = Workspace.from_path(self.root)
        with self.assertRaises(PathPolicyError):
            workspace.resolve("../outside/secret.txt")

    @patch.dict(os.environ, {"SPEAR_SANDBOX_IDENTITY_MOUNT": "0"})
    def test_absolute_path_requires_explicit_option_under_the_legacy_mount(self):
        workspace = Workspace.from_path(self.root)
        with self.assertRaises(PathPolicyError):
            workspace.resolve(self.root / "inside.txt")

    def test_host_path_is_accepted_when_it_is_the_sandbox_mount(self):
        """The gate distinguishes host naming from sandbox naming.

        Under the identity mount there is no distinction left: the sandbox
        binds the tree at its own host path and the prompt says so, exactly as
        /workspace was accepted for being the sandbox's own naming. Refusing it
        here would leave bash and edit_file disagreeing about the same path.
        """
        workspace = Workspace.from_path(self.root)
        self.assertEqual(workspace.resolve(self.root / "inside.txt"),
                         self.root / "inside.txt")
        # Containment is unaffected: outside stays outside.
        with self.assertRaises(PathPolicyError):
            workspace.resolve(self.outside / "secret.txt")

    def test_allowed_absolute_path_still_must_be_inside_workspace(self):
        workspace = Workspace.from_path(self.root, allow_absolute_paths=True)
        self.assertEqual(workspace.resolve(self.root / "inside.txt"), self.root / "inside.txt")
        with self.assertRaises(PathPolicyError):
            workspace.resolve(self.outside / "secret.txt")

    def test_symlink_escape_is_rejected_before_file_access(self):
        link = self.root / "outside-link"
        link.symlink_to(self.outside, target_is_directory=True)
        workspace = Workspace.from_path(self.root)
        with self.assertRaises(PathPolicyError):
            workspace.resolve("outside-link/secret.txt")

    def test_new_file_under_symlinked_parent_is_rejected(self):
        link = self.root / "outside-link"
        link.symlink_to(self.outside, target_is_directory=True)
        workspace = Workspace.from_path(self.root)
        with self.assertRaises(PathPolicyError):
            workspace.resolve("outside-link/new.txt")

    def test_required_missing_path_is_reported(self):
        workspace = Workspace.from_path(self.root)
        with self.assertRaises(PathNotFoundError):
            workspace.resolve("missing.txt", must_exist=True)


class ToolResultTests(unittest.TestCase):
    def test_legacy_rendering_is_compatible(self):
        self.assertEqual(ToolResult("ok", "done").to_legacy_text(), "OK: done")
        self.assertEqual(ToolResult("cancelled", "no").to_legacy_text(), "CANCELLED")
        self.assertEqual(ToolResult("denied", "blocked").to_legacy_text(), "ERROR: blocked")


class ExecutionProfileTests(unittest.TestCase):
    def test_profiles_are_pure_translations_of_granted_capabilities(self):
        readonly = ExecutionProfile.from_capabilities({Capability.FILESYSTEM_READ})
        self.assertTrue(readonly.workspace_read)
        self.assertFalse(readonly.workspace_write)
        self.assertFalse(readonly.shell_complex)
        self.assertFalse(readonly.network)

        writable = ExecutionProfile.from_capabilities({
            Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE,
        })
        self.assertTrue(writable.workspace_write)
        shell = ExecutionProfile.from_capabilities({
            Capability.FILESYSTEM_READ, Capability.SHELL_COMPLEX,
        })
        self.assertTrue(shell.shell_complex)

    def test_profiles_reject_inconsistent_sensitive_capabilities(self):
        with self.assertRaises(ValueError):
            ExecutionProfile.from_capabilities({Capability.WORKSPACE_WRITE})
        with self.assertRaises(ValueError):
            ExecutionProfile.from_capabilities({Capability.SSH})
        with self.assertRaises(ValueError):
            ExecutionProfile.from_capabilities({Capability.REMOTE_WRITE})

    def test_sensitive_profiles_are_representable_but_not_sandbox_implemented(self):
        network = ExecutionProfile.from_capabilities({Capability.NETWORK})
        gpu = ExecutionProfile.from_capabilities({Capability.GPU})
        container = ExecutionProfile.from_capabilities({Capability.CONTAINER_RUNTIME})
        secrets = ExecutionProfile.from_capabilities({Capability.SECRETS})
        self.assertTrue(network.network)
        self.assertTrue(gpu.gpu)
        self.assertTrue(container.container_runtime)
        self.assertTrue(secrets.secrets_allowed)


class MultiRootWorkspaceTests(unittest.TestCase):
    """A workspace may declare extra roots beyond the launch directory."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.primary = base / "primary"; (self.primary / "sub").mkdir(parents=True)
        self.other = base / "other"; (self.other / "usr" / "src").mkdir(parents=True)
        self.outside = base / "outside"; self.outside.mkdir()
        (self.primary / "here.txt").write_text("here")
        (self.other / "usr" / "src" / "ping.c").write_text("int main(void){return 0;}")
        (self.outside / "secret.txt").write_text("secret")
        self.ws = Workspace.from_path(self.primary, extra_roots=[self.other])

    def tearDown(self):
        self.temp.cleanup()

    def test_relative_paths_still_resolve_against_the_primary_root_only(self):
        self.assertEqual(self.ws.resolve("here.txt"), self.primary / "here.txt")
        # A relative path never reaches a secondary root: that would make the
        # same string mean different files depending on the root list.
        with self.assertRaises(PathPolicyError):
            self.ws.resolve("usr/src/ping.c", must_exist=True)

    def test_absolute_paths_reach_declared_extra_roots(self):
        target = self.other / "usr" / "src" / "ping.c"
        self.assertEqual(self.ws.resolve(str(target), must_exist=True), target)
        # And a file that does not exist yet, inside an extra root, is legal:
        # creating is the point of declaring the root.
        self.assertEqual(self.ws.resolve(str(self.other / "usr" / "src" / "new.c")),
                         self.other / "usr" / "src" / "new.c")

    def test_undeclared_trees_remain_unreachable(self):
        for raw in (str(self.outside / "secret.txt"),
                    str(self.outside / "new.txt"),
                    "../outside/secret.txt",
                    str(self.other / ".." / "outside" / "secret.txt")):
            with self.assertRaises(PathPolicyError, msg=raw):
                self.ws.resolve(raw)

    @patch.dict(os.environ, {"SPEAR_SANDBOX_IDENTITY_MOUNT": "0"})
    def test_absolute_under_primary_still_needs_the_legacy_flag(self):
        # Unchanged behaviour: declaring extra roots must not silently start
        # accepting absolute paths into the launch directory. Pinned to the
        # legacy mount, where a host path is not also the sandbox's naming.
        with self.assertRaises(PathPolicyError):
            self.ws.resolve(str(self.primary / "here.txt"))
        opened = Workspace.from_path(self.primary, extra_roots=[self.other],
                                     allow_absolute_paths=True)
        self.assertEqual(opened.resolve(str(self.primary / "here.txt")),
                         self.primary / "here.txt")

    def test_symlink_escape_is_refused_from_every_root(self):
        (self.primary / "escape").symlink_to(self.outside, target_is_directory=True)
        (self.other / "escape").symlink_to(self.outside, target_is_directory=True)
        with self.assertRaises(PathPolicyError):
            self.ws.resolve("escape/secret.txt")
        with self.assertRaises(PathPolicyError):
            self.ws.resolve(str(self.other / "escape" / "secret.txt"))

    def test_roots_are_canonical_deduplicated_and_never_nested(self):
        # A nested or repeated root would mount the same tree twice and make
        # the label ambiguous; the primary always wins.
        ws = Workspace.from_path(
            self.primary,
            extra_roots=[self.other, self.other, self.primary / "sub", self.primary])
        self.assertEqual(ws.extra_roots, (self.other.resolve(),))
        self.assertEqual(ws.roots, (self.primary.resolve(), self.other.resolve()))

    def test_a_missing_extra_root_is_dropped_not_fatal(self):
        ws = Workspace.from_path(self.primary,
                                 extra_roots=[self.other, Path(self.temp.name) / "gone"])
        self.assertEqual(ws.extra_roots, (self.other.resolve(),))

    def test_labels_stay_unambiguous_across_roots(self):
        self.assertEqual(self.ws.relative(self.primary / "here.txt"), "here.txt")
        # A file in a secondary root is labelled by its root, never as a bare
        # relative path that would look like a primary-root file.
        self.assertEqual(self.ws.relative(self.other / "usr" / "src" / "ping.c"),
                         f"{self.other.name}:usr/src/ping.c")
        with self.assertRaises(PathPolicyError):
            self.ws.relative(self.outside / "secret.txt")

    def test_sandbox_mount_paths_are_accepted_by_the_file_tools(self):
        """One addressing scheme, not two.

        bash sees a secondary root at /workspaces/<name> while write_file acts
        on the host. Making the model juggle both is a guaranteed source of
        wrong paths, so a mount path resolves to its host root everywhere.
        """
        self.assertEqual(self.ws.resolve("/workspaces/other/usr/src/ping.c",
                                         must_exist=True),
                         self.other / "usr" / "src" / "ping.c")
        self.assertEqual(self.ws.resolve("/workspace/here.txt", must_exist=True),
                         self.primary / "here.txt")
        # Translation happens before containment, so traversal out of a mount
        # is still refused.
        with self.assertRaises(PathPolicyError):
            self.ws.resolve("/workspaces/other/../../outside/secret.txt")
        # An undeclared mount name is not a path into anything.
        with self.assertRaises(PathPolicyError):
            self.ws.resolve("/workspaces/nope/x.c")

    def test_single_root_workspace_is_unchanged(self):
        plain = Workspace.from_path(self.primary)
        self.assertEqual(plain.extra_roots, ())
        self.assertEqual(plain.roots, (self.primary.resolve(),))
        with self.assertRaises(PathPolicyError):
            plain.resolve(str(self.other / "usr" / "src" / "ping.c"))


class SafeModeReadOnlyShellTests(unittest.TestCase):
    """SAFE may run a read-only pipeline: the MOUNT is what makes it safe."""

    def setUp(self):
        self.policy = CommandPolicy()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "ws"; self.root.mkdir()
        (self.root / "a.txt").write_text("alpha\nbeta\n")

    def tearDown(self):
        self.temp.cleanup()

    def _granted(self, command, mode):
        assessment = self.policy.classify(command)
        policy = DEFAULT_CAPABILITY_POLICY
        return assessment.required_capabilities <= policy.for_mode(mode)

    def test_safe_grants_shell_complex(self):
        self.assertIn(Capability.SHELL_COMPLEX,
                      DEFAULT_CAPABILITY_POLICY.for_mode(ExecutionMode.SAFE))
        # …and still refuses the capability that actually writes.
        self.assertNotIn(Capability.WORKSPACE_WRITE,
                         DEFAULT_CAPABILITY_POLICY.for_mode(ExecutionMode.SAFE))
        self.assertNotIn(Capability.NETWORK,
                         DEFAULT_CAPABILITY_POLICY.for_mode(ExecutionMode.SAFE))

    def test_read_only_pipelines_are_allowed_in_safe(self):
        for command in ("grep -n beta a.txt | head -3",
                        'find . -name "*.txt" -type f 2>/dev/null | head',
                        "cat a.txt | wc -l"):
            self.assertTrue(self._granted(command, ExecutionMode.SAFE), command)

    def test_writing_pipelines_are_still_refused_in_safe(self):
        for command in ("echo x > a.txt",
                        "cat a.txt | tee copy.txt",
                        "make 2>&1 | tail",
                        "curl https://example.invalid | head"):
            self.assertFalse(self._granted(command, ExecutionMode.SAFE), command)

    def test_every_mutating_classification_declares_the_write_capability(self):
        """The invariant that lets the capability check replace the class gate.

        If a WORKSPACE_MUTATING command could ever omit workspace:write, SAFE
        would authorise it the moment the classification gate stopped being the
        thing that refused it.
        """
        for command in ("make", "cmake .", "ninja", "pytest",
                        "curl -o out https://example.invalid",
                        "wget https://example.invalid"):
            assessment = self.policy.classify(command)
            if assessment.classification == CommandClassification.WORKSPACE_MUTATING:
                self.assertIn(Capability.WORKSPACE_WRITE,
                              assessment.required_capabilities, command)
                self.assertFalse(self._granted(command, ExecutionMode.SAFE), command)

    def test_the_mount_is_read_only_in_safe_not_just_the_policy(self):
        """The guarantee must not rest on the classifier being exhaustive."""
        profile = ExecutionProfile.from_capabilities(
            DEFAULT_CAPABILITY_POLICY.for_mode(ExecutionMode.SAFE))
        self.assertFalse(profile.workspace_write)
        sandbox = tool_runtime.BubblewrapSandbox()
        workspace = Workspace.from_path(self.root)
        mount = sandbox.mount_root(workspace, profile)
        argv = sandbox.build_argv(workspace, ["true"], profile)
        binds = [argv[i] for i, t in enumerate(argv)
                 if t in ("--bind", "--ro-bind") and argv[i + 2] == mount]
        self.assertEqual(binds, ["--ro-bind"])


class MultiRootSandboxMountTests(unittest.TestCase):
    """Every declared root is mounted, with the same access as the primary."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.primary = base / "primary"; self.primary.mkdir()
        self.so3 = base / "so3"; self.so3.mkdir()
        self.lvgl = base / "lvgl"; self.lvgl.mkdir()
        self.ws = Workspace.from_path(self.primary, extra_roots=[self.so3, self.lvgl])
        self.sandbox = tool_runtime.BubblewrapSandbox()

    def tearDown(self):
        self.temp.cleanup()

    def _mounts(self, profile):
        argv = self.sandbox.build_argv(self.ws, ["true"], profile)
        found = {}
        for index, token in enumerate(argv):
            if token in ("--bind", "--ro-bind") and index + 2 < len(argv):
                found[argv[index + 2]] = (token, argv[index + 1])
        return found

    def _expected(self, profile):
        """Host root → mount, as the sandbox resolves it for this profile."""
        mount = self.sandbox.mount_root(self.ws, profile)
        return self.ws.mount_map(
            mount, identity=mount != tool_runtime.SandboxSpec().workspace_mount)

    def test_write_profile_binds_every_root_read_write(self):
        profile = ExecutionProfile.from_capabilities(
            [Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE])
        mounts = self._mounts(profile)
        for root in (self.primary, self.so3, self.lvgl):
            at = self._expected(profile)[str(root)]
            self.assertEqual(mounts[at], ("--bind", str(root)))

    def test_read_only_profile_binds_every_root_read_only(self):
        profile = ExecutionProfile.from_capabilities([Capability.FILESYSTEM_READ])
        mounts = self._mounts(profile)
        for root in (self.primary, self.so3, self.lvgl):
            at = self._expected(profile)[str(root)]
            self.assertEqual(mounts[at][0], "--ro-bind", at)

    def test_extra_roots_never_appear_without_read_access(self):
        # No filesystem:read at all: the sandbox gets an empty /workspace and
        # must not expose a secondary tree either.
        profile = ExecutionProfile.from_capabilities([])
        argv = self.sandbox.build_argv(self.ws, ["true"], profile)
        self.assertNotIn(str(self.so3), argv)
        self.assertNotIn("/workspaces/so3", argv)

    @patch.dict(os.environ, {"SPEAR_SANDBOX_IDENTITY_MOUNT": "0"})
    def test_mount_names_are_unique_when_basenames_collide(self):
        # Uniquifying is what the /workspaces/<name> scheme needs; identity
        # mounts are unique by construction, so this pins the legacy mount.
        other = Path(self.temp.name) / "nested"; (other / "so3").mkdir(parents=True)
        ws = Workspace.from_path(self.primary, extra_roots=[self.so3, other / "so3"])
        sandbox = tool_runtime.BubblewrapSandbox()
        profile = ExecutionProfile.from_capabilities(
            [Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE])
        argv = sandbox.build_argv(ws, ["true"], profile)
        mounts = [argv[i + 2] for i, t in enumerate(argv)
                  if t in ("--bind", "--ro-bind") and argv[i + 2].startswith("/workspaces/")]
        self.assertEqual(len(mounts), len(set(mounts)), mounts)

    @patch.dict(os.environ, {"SPEAR_SANDBOX_IDENTITY_MOUNT": "0"})
    def test_single_root_argv_is_byte_identical_to_before(self):
        plain = Workspace.from_path(self.primary)
        profile = ExecutionProfile.from_capabilities(
            [Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE])
        argv = self.sandbox.build_argv(plain, ["true"], profile)
        self.assertNotIn("/workspaces", " ".join(argv))

    def test_mount_map_is_exposed_for_the_prompt(self):
        # The model must be told where each root appears, or it cannot use a
        # secondary tree from bash at all.
        self.assertEqual(self.ws.mount_map(), {
            str(self.primary): "/workspace",
            str(self.so3): "/workspaces/so3",
            str(self.lvgl): "/workspaces/lvgl",
        })


class CommandPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = CommandPolicy()

    def test_classifies_read_only_and_workspace_build_commands(self):
        self.assertEqual(self.policy.classify("rg symbol src").classification,
                         CommandClassification.READ_ONLY)
        self.assertEqual(self.policy.classify("make test").classification,
                         CommandClassification.WORKSPACE_MUTATING)
        self.assertEqual(self.policy.classify("cmake --build build").classification,
                         CommandClassification.WORKSPACE_MUTATING)
        self.assertEqual(self.policy.classify("pytest tests").classification,
                         CommandClassification.WORKSPACE_MUTATING)

    def test_a_read_only_turn_cannot_write_through_bash(self):
        """Dropping the write TOOLS is half of a prohibition.

        bash writes too — `sed -i`, `cp`, `mv`, a redirection — so a turn told
        not to change anything runs it under SAFE, which grants no
        workspace:write and mounts every root read-only.
        """

        safe = self.policy.available_capabilities(ExecutionMode.SAFE)
        self.assertNotIn(Capability.WORKSPACE_WRITE, safe)

        for command in ("sed -i 's/a/b/' main.py", "cp main.py main.bak",
                        "mv main.py other.py", "echo x > main.py",
                        "tee main.py"):
            with self.subTest(command=command):
                assessment = self.policy.classify(command)
                refused = (assessment.classification
                           in {CommandClassification.DANGEROUS,
                               CommandClassification.SHELL_COMPLEX}
                           or Capability.WORKSPACE_WRITE
                           in assessment.required_capabilities)
                self.assertTrue(refused, assessment.classification)
                self.assertIsNotNone(
                    self.policy.authorize(assessment, ExecutionMode.SAFE).result,
                    command)

        # And reading is untouched: that is the whole point of the turn.

        for command in ("cat main.py", "grep -rn add .", "sed -n '1,20p' main.py"):
            with self.subTest(command=command):
                assessment = self.policy.classify(command)
                self.assertIsNone(
                    self.policy.authorize(assessment, ExecutionMode.SAFE).result,
                    command)

    def test_python_module_test_runners_are_runnable(self):
        """`python -m pytest` is the spelling models use, and it runs pytest.

        Refusing it cost the control-edit benchmark its first verification:
        the run was denied, and the turn finished without any test having run.
        Everything else about the interpreter stays refused.
        """

        for command in ("python3 -m pytest -q", "python -m pytest",
                        "python3 -m unittest discover"):
            with self.subTest(command=command):
                assessment = self.policy.classify(command)
                self.assertEqual(assessment.classification,
                                 CommandClassification.WORKSPACE_MUTATING)
                self.assertIn(Capability.WORKSPACE_WRITE,
                              assessment.required_capabilities)

        for command in ("python3 main.py", "python3 -m pip install x",
                        "python3 -m http.server", "/usr/bin/python3 -m pytest"):
            with self.subTest(command=command):
                self.assertEqual(self.policy.classify(command).classification,
                                 CommandClassification.DANGEROUS)

    def test_every_interpreter_refusal_names_the_command_that_works_first(self):
        """The way out comes before the reason, on every path that refuses.

        A run asked to "run the test" tried `python3 -c "..."` twice. The
        quoted parentheses made it shell-complex, so it was refused by the
        PIPELINE classifier, whose message named no alternative at all -- and
        the single-command message that did name one buried it behind the
        reason. The run never ran a test.
        """

        for command in ('python3 -c "print(1)"', "python3 -c 'import main'",
                        "python3 main.py", "python3 tests/test_main.py",
                        'echo x | python3 -c "y"', "python -c 'x'"):
            with self.subTest(command=command):
                reason = self.policy.classify(command).reason
                self.assertEqual(self.policy.classify(command).classification,
                                 CommandClassification.DANGEROUS)
                self.assertTrue(
                    reason.startswith("run tests with `python3 -m unittest "
                                      "discover -s tests`"), reason)
                self.assertIn("`python3 -c` and running a script directly are "
                              "refused", reason)

    def test_the_generic_refusal_leads_with_what_can_run(self):
        reason = self.policy.classify("frobnicate --all").reason
        self.assertTrue(reason.startswith("build and test with"), reason)
        self.assertIn("python3 -m unittest discover -s tests", reason)
        self.assertIn("'frobnicate' is none of those", reason)

    def test_classifies_shell_and_dangerous_commands(self):
        self.assertEqual(self.policy.classify("rg symbol src | head").classification,
                         CommandClassification.SHELL_COMPLEX)
        self.assertEqual(self.policy.classify("rm -rf build").classification,
                         CommandClassification.DANGEROUS)
        self.assertEqual(self.policy.classify("git reset --hard").classification,
                         CommandClassification.DANGEROUS)

    def test_redirection_target_is_a_file_not_a_command(self):
        """`2>/dev/null` must not make a command dangerous.

        The stage splitter treated the token after a redirection operator as a
        new command stage, so `/dev/null` hit the "argv[0] contains /" rule and
        every command carrying the most common idiom in shell was refused as a
        "dangerous shell stage" — including `ls 2>/dev/null`.
        """
        for command in ('find . -name "ping.c" -type f 2>/dev/null | head -20',
                        "ls 2>/dev/null",
                        "echo hi 2>/dev/null",
                        'grep -r so3 . --include="*.md" 2>/dev/null | head',
                        "make 2>&1 | tail -20",
                        "cat notes.txt > copy.txt",
                        "sort < input.txt"):
            assessment = self.policy.classify(command)
            self.assertNotEqual(assessment.classification,
                                CommandClassification.DANGEROUS,
                                f"{command!r}: {assessment.reason}")

        # The guard the bug was accidentally providing must survive on its own:
        # a redirection escaping the workspace stays dangerous, and only the
        # harmless /dev sinks are exempt.
        for command in ("echo pwned > /etc/spear-marker",
                        "echo pwned >> /usr/bin/spear-marker",
                        "cat secrets > ../outside.txt",
                        "cat secrets > /home/other/loot"):
            assessment = self.policy.classify(command)
            self.assertEqual(assessment.classification,
                             CommandClassification.DANGEROUS,
                             f"{command!r} must stay dangerous")

        # A write redirection still declares that it mutates the workspace.
        self.assertIn(Capability.WORKSPACE_WRITE,
                      self.policy.classify("echo hi > note.txt").required_capabilities)
        # Reading a pipeline stage after a redirect is still analysed: a real
        # command behind the redirect keeps its own capabilities.
        self.assertIn(Capability.NETWORK,
                      self.policy.classify("curl https://example.invalid 2>/dev/null")
                      .required_capabilities)

    def test_sandbox_mount_paths_are_not_escapes(self):
        """Paths under the sandbox mounts are inside, by construction.

        The blanket "argv contains an absolute path" refusal predates multiple
        roots. It left the feature half-wired: write_file could reach a second
        root while bash could not.
        """
        for command in ("ls /workspaces/so3/usr/src",
                        "cat /workspace/notes.txt",
                        "grep -r main /workspaces/lvgl",
                        "echo ok > /workspaces/so3/usr/src/ping.c",
                        "grep main /workspace/a.c /workspaces/so3/b.c"):
            assessment = self.policy.classify(command)
            self.assertNotEqual(assessment.classification,
                                CommandClassification.DANGEROUS,
                                f"{command!r}: {assessment.reason}")

        # WRITING outside stays refused, and a mount name is not a licence to
        # traverse out of it. Reading outside is a separate grant — see
        # OutsideReadPolicyTests.
        for command in ("echo pwned > /workspace/../etc/x",
                        "echo pwned > /etc/x",
                        "cat ../outside/secret"):
            self.assertEqual(self.policy.classify(command).classification,
                             CommandClassification.DANGEROUS, command)

    def test_assessments_declare_required_capabilities(self):
        cases = {
            "pwd": {Capability.FILESYSTEM_READ},
            "git status": {Capability.FILESYSTEM_READ},
            "git diff": {Capability.FILESYSTEM_READ},
            "make": {Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE},
            "pytest": {Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE},
            "curl https://example.invalid": {Capability.NETWORK},
            "git clone https://example.invalid/repo": {
                Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE, Capability.NETWORK,
            },
            "git fetch": {
                Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE, Capability.NETWORK,
            },
            "git fetch origin": {
                Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE, Capability.NETWORK,
            },
            "git push": {
                Capability.FILESYSTEM_READ, Capability.NETWORK, Capability.REMOTE_WRITE,
            },
            "wget https://example.invalid": {Capability.NETWORK, Capability.WORKSPACE_WRITE},
            "curl -o result https://example.invalid": {
                Capability.NETWORK, Capability.WORKSPACE_WRITE,
            },
            "curl --output result https://example.invalid": {
                Capability.NETWORK, Capability.WORKSPACE_WRITE,
            },
            "curl -O https://example.invalid": {Capability.NETWORK, Capability.WORKSPACE_WRITE},
            "curl --remote-name https://example.invalid": {
                Capability.NETWORK, Capability.WORKSPACE_WRITE,
            },
            "ssh host": {Capability.SSH, Capability.NETWORK, Capability.REMOTE_WRITE},
            "ssh host command": {Capability.SSH, Capability.NETWORK, Capability.REMOTE_WRITE},
            "scp file host:": {
                Capability.SSH, Capability.NETWORK, Capability.FILESYSTEM_READ,
                Capability.REMOTE_WRITE,
            },
            "scp host:path local-file": {
                Capability.SSH, Capability.NETWORK, Capability.WORKSPACE_WRITE,
            },
            "scp host:path .": {Capability.SSH, Capability.NETWORK, Capability.WORKSPACE_WRITE},
            "docker ps": {Capability.CONTAINER_RUNTIME},
            "podman ps": {Capability.CONTAINER_RUNTIME},
            "nvidia-smi": {Capability.GPU},
            "ls | head": {Capability.FILESYSTEM_READ, Capability.SHELL_COMPLEX},
            # filesystem:read is unconditional on a pipeline now: without it
            # the sandbox mounts an empty directory instead of the trees.
            "curl https://example.invalid | cat": {
                Capability.NETWORK, Capability.SHELL_COMPLEX,
                Capability.FILESYSTEM_READ},
            "make && pytest": {
                Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE,
                Capability.SHELL_COMPLEX,
            },
            "curl -o result https://example.invalid | cat": {
                Capability.NETWORK, Capability.WORKSPACE_WRITE,
                Capability.SHELL_COMPLEX, Capability.FILESYSTEM_READ,
            },
            "git push && git status": {
                Capability.FILESYSTEM_READ, Capability.NETWORK,
                Capability.REMOTE_WRITE, Capability.SHELL_COMPLEX,
            },
            "curl -X POST https://example.invalid | cat": {
                Capability.NETWORK, Capability.REMOTE_WRITE,
                Capability.SHELL_COMPLEX, Capability.FILESYSTEM_READ,
            },
            "scp file host:path && echo done": {
                Capability.FILESYSTEM_READ, Capability.SSH, Capability.NETWORK,
                Capability.REMOTE_WRITE, Capability.SHELL_COMPLEX,
            },
            "wget https://example.invalid && cat file": {
                Capability.NETWORK, Capability.WORKSPACE_WRITE,
                Capability.FILESYSTEM_READ, Capability.SHELL_COMPLEX,
            },
            "git fetch && git status": {
                Capability.NETWORK, Capability.WORKSPACE_WRITE,
                Capability.FILESYSTEM_READ, Capability.SHELL_COMPLEX,
            },
            "printf 'ok\\n' > result.txt": {
                Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE,
                Capability.SHELL_COMPLEX,
            },
        }
        for command, expected in cases.items():
            with self.subTest(command=command):
                assessment = self.policy.classify(command)
                self.assertEqual(assessment.required_capabilities, frozenset(expected))
        for command in ("git status", "git diff"):
            self.assertNotIn(Capability.NETWORK, self.policy.classify(command).required_capabilities)

    def test_remote_mutation_capability_for_curl(self):
        for command in (
            "curl -X POST https://example.invalid",
            "curl --request DELETE https://example.invalid",
            "curl -d x=1 https://example.invalid",
            "curl --data x=1 https://example.invalid",
            "curl -F file=@x https://example.invalid",
        ):
            with self.subTest(command=command):
                capabilities = self.policy.classify(command).required_capabilities
                self.assertIn(Capability.NETWORK, capabilities)
                self.assertIn(Capability.REMOTE_WRITE, capabilities)
        output = self.policy.classify("curl -o result -X POST https://example.invalid")
        self.assertEqual(output.required_capabilities, frozenset({
            Capability.NETWORK, Capability.WORKSPACE_WRITE, Capability.REMOTE_WRITE,
        }))
        for command in (
            "curl https://example.invalid",
            "curl -I https://example.invalid",
            "curl --head https://example.invalid",
            "git fetch",
            "scp host:file .",
        ):
            with self.subTest(command=command):
                self.assertNotIn(Capability.REMOTE_WRITE,
                                 self.policy.classify(command).required_capabilities)

    def test_sensitive_capabilities_are_classified_without_becoming_dangerous(self):
        for command in ("curl https://example.invalid", "ssh host", "docker ps", "nvidia-smi"):
            with self.subTest(command=command):
                self.assertNotEqual(self.policy.classify(command).classification,
                                    CommandClassification.DANGEROUS)

    def test_sensitive_capabilities_are_refused_in_every_mode(self):
        assessments = [
            self.policy.classify("git push"),
            self.policy.classify("ssh host"),
            self.policy.classify("docker ps"),
            self.policy.classify("nvidia-smi"),
        ]
        for mode in ExecutionMode:
            for assessment in assessments:
                with self.subTest(mode=mode, command=assessment.command):
                    result = self.policy.authorize(assessment, mode, lambda _: True)
                    self.assertIsNotNone(result.result)
                self.assertEqual(result.result.status, "denied")

    def test_ask_network_is_confirmable_but_remote_write_is_denied_first(self):
        curl = self.policy.classify("curl https://example.invalid")
        fetch = self.policy.classify("git fetch origin")
        post = self.policy.classify("curl -X POST https://example.invalid")
        push = self.policy.classify("git push")
        ssh = self.policy.classify("ssh example.invalid")
        status = self.policy.classify("git status")

        self.assertTrue(self.policy.authorize(curl, ExecutionMode.ASK, lambda _: True).allowed)
        self.assertEqual(self.policy.authorize(curl, ExecutionMode.ASK, lambda _: False).result.status,
                         "cancelled")
        self.assertTrue(self.policy.authorize(fetch, ExecutionMode.ASK, lambda _: True).allowed)
        for assessment in (post, push, ssh):
            with self.subTest(command=assessment.command):
                approve = MagicMock(return_value=True)
                result = self.policy.authorize(assessment, ExecutionMode.ASK, approve)
                self.assertEqual(result.result.status, "denied")
                approve.assert_not_called()
        self.assertTrue(self.policy.authorize(status, ExecutionMode.ASK, lambda _: False).allowed)
        approve = MagicMock(return_value=True)
        result = self.policy.authorize(curl, ExecutionMode.SAFE, approve)
        self.assertEqual(result.result.status, "denied")
        approve.assert_not_called()
        # AUTO grants it and asks nothing — that is what AUTO means. It used
        # to deny, which made `-y` the mode that still refused curl.
        approve = MagicMock(return_value=True)
        self.assertTrue(self.policy.authorize(curl, ExecutionMode.AUTO,
                                              approve).allowed)
        approve.assert_not_called()

    def test_modes_enforce_expected_command_policy(self):
        readonly = self.policy.classify("ls")
        mutate = self.policy.classify("ninja")
        complex_command = self.policy.classify("ls | head")
        self.assertTrue(self.policy.authorize(readonly, ExecutionMode.SAFE).allowed)
        self.assertEqual(self.policy.authorize(mutate, ExecutionMode.SAFE).result.status, "denied")
        # A read-only pipeline is now allowed in SAFE — the mount keeps it
        # read-only. A pipeline that writes still fails the capability check.
        self.assertTrue(self.policy.authorize(complex_command, ExecutionMode.SAFE).allowed)
        self.assertEqual(
            self.policy.authorize(self.policy.classify("echo x > f"),
                                  ExecutionMode.SAFE).result.status, "denied")
        self.assertTrue(self.policy.authorize(mutate, ExecutionMode.AUTO).allowed)
        # Authorization is distinct from execution: AUTO shell-complex still
        # requires Bubblewrap and is fail-closed by the integration route.
        self.assertTrue(self.policy.authorize(complex_command, ExecutionMode.AUTO).allowed)
        self.assertEqual(self.policy.authorize(mutate, ExecutionMode.ASK, lambda _: False).result.status,
                         "cancelled")
        self.assertTrue(self.policy.authorize(mutate, ExecutionMode.ASK, lambda _: True).allowed)

    def test_capability_policy_default_and_network_restriction(self):
        # SAFE now also grants shell:complex — read-only pipelines. What keeps
        # SAFE read-only is the mount, not the classifier; see
        # SafeModeReadOnlyShellTests.
        # host:read is granted in SAFE too: looking at a tree outside the
        # declared roots is reading, which is what SAFE is for, and the paths
        # are bind-mounted read-only in every mode.
        self.assertEqual(DEFAULT_CAPABILITY_POLICY.for_mode(ExecutionMode.SAFE),
                         frozenset({Capability.FILESYSTEM_READ,
                                    Capability.HOST_READ,
                                    Capability.SHELL_COMPLEX}))
        self.assertIn(Capability.NETWORK, DEFAULT_CAPABILITY_POLICY.for_mode(ExecutionMode.ASK))
        # AUTO is ASK without the prompt, so it cannot grant less than ASK.
        # It used to withhold the network, which made `-y` — reached for to
        # stop being blocked — the mode that still refused curl.
        self.assertIn(Capability.NETWORK,
                      DEFAULT_CAPABILITY_POLICY.for_mode(ExecutionMode.AUTO))
        restricted = CommandPolicy(DEFAULT_CAPABILITY_POLICY.without_network())
        approve = MagicMock(return_value=True)
        outcome = restricted.authorize(
            restricted.classify("curl https://example.invalid"), ExecutionMode.ASK, approve
        )
        self.assertFalse(outcome.allowed)
        self.assertEqual(outcome.result.status, "denied")
        approve.assert_not_called()

    def test_a_capability_refusal_names_the_mode_that_grants_it(self):
        """A refusal that only states the rule is a dead end: told `network`
        was not allowed and nothing else, the model ran six more web searches
        and told the user to open a browser."""
        policy = CommandPolicy(DEFAULT_CAPABILITY_POLICY)
        outcome = policy.authorize(
            policy.classify("curl https://example.invalid"), ExecutionMode.SAFE)
        message = str(outcome.result)
        self.assertIn("--ask", message)
        self.assertIn("--auto", message)
        # …and the way that needs no capability at all.
        self.assertIn("fetch_url", message)

    def test_a_refusal_no_mode_can_lift_says_so(self):
        offline = CommandPolicy(DEFAULT_CAPABILITY_POLICY.without_network())
        outcome = offline.authorize(
            offline.classify("curl https://example.invalid"), ExecutionMode.AUTO)
        message = str(outcome.result)
        self.assertIn("No mode grants", message)
        self.assertNotIn("--ask", message)
        # …and it does not send the model to fetch_url either: --no-network
        # takes the web tools out of the session, so that would be a dead end
        # of a different kind.
        self.assertNotIn("fetch_url", message)

    def test_authorization_result_is_the_granted_capability_source(self):
        assessment = self.policy.classify("git fetch origin")
        outcome = self.policy.authorize(assessment, ExecutionMode.ASK, lambda _: True)
        self.assertIsInstance(outcome, AuthorizationResult)
        self.assertTrue(outcome.allowed)
        self.assertEqual(outcome.granted_capabilities, assessment.required_capabilities)

    def test_heredoc_body_is_data_not_path_operands(self):
        """A file's CONTENT must not be read as arguments of the command.

        Writing a documentation chapter with `cat > doc/source/ls.rst <<EOF`
        was refused with "/ contains the sandbox's own mounts and cannot be
        exposed": the chapter contained the shell-prompt example line `/ % ls`,
        and shlex handed that bare `/` over as a path operand of `cat`. The
        refusal named a path the request never contained.
        """
        command = ("cat > /tmp/ls.rst << 'EOF'\n"
                   ".. _ls:\n\nExamples\n========\n\n::\n\n"
                   "   / % ls\n   / % ls -l /etc\n"
                   "EOF")
        assessment = self.policy.classify(command)
        self.assertNotEqual(assessment.classification,
                            CommandClassification.DANGEROUS)
        self.assertNotIn("sandbox's own mounts", assessment.reason)

    def test_a_heredoc_that_feeds_an_interpreter_is_still_scanned(self):
        """Stripping the body must not become a way to smuggle commands.

        `bash <<EOF` and `cat <<EOF | bash` both execute their body, so there
        the body is code and stays visible to the danger scan.
        """
        for command in ("bash << 'EOF'\nrm -rf /\nEOF",
                        "cat << 'EOF' | bash\nrm -rf /\nEOF"):
            with self.subTest(command=command.splitlines()[0]):
                self.assertEqual(self.policy.classify(command).classification,
                                 CommandClassification.DANGEROUS)

    def test_the_heredoc_target_path_is_still_checked(self):
        """Only the body goes; the redirection target is still an operand."""
        command = ("cat > /etc/shadow << 'EOF'\nharmless text\nEOF")
        assessment = self.policy.classify(command)
        self.assertIn("/etc/shadow", assessment.command)


class ToolPolicyTests(unittest.TestCase):
    def test_safe_ask_and_auto_mutation_behavior(self):
        self.assertEqual(ToolPolicy(ExecutionMode.SAFE).authorize_mutation("Write file").status,
                         "denied")
        self.assertEqual(ToolPolicy(ExecutionMode.ASK, lambda _: False)
                         .authorize_mutation("Write file").status, "cancelled")
        self.assertIsNone(ToolPolicy(ExecutionMode.ASK, lambda _: True)
                          .authorize_mutation("Write file"))
        self.assertIsNone(ToolPolicy(ExecutionMode.AUTO).authorize_mutation("Write file"))


class CommandRunnerTests(unittest.TestCase):
    def test_simple_command_uses_shell_false(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace.from_path(temporary)
            assessment = CommandPolicy().classify("pwd")
            runner = CommandRunner()
            with patch("tool_runtime.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = "ok"
                run.return_value.stderr = ""
                result = runner.run_simple(assessment, workspace)
            self.assertTrue(result.ok)
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertEqual(run.call_args.args[0], ["pwd"])

    def test_network_preflight_cache_is_workspace_scoped_and_poison_aware(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = Workspace.from_path(root)
            second_root = root / "second"
            second_root.mkdir()
            second = Workspace.from_path(second_root)
            sandbox = MagicMock()
            sandbox.preflight_network.return_value = ToolResult("ok", "network ready")
            runner = CommandRunner(sandbox=sandbox)
            profile = ExecutionProfile.from_capabilities({Capability.NETWORK})
            self.assertTrue(runner.ensure_sandbox(first, profile).ok)
            self.assertTrue(runner.ensure_sandbox(first, profile).ok)
            self.assertTrue(runner.ensure_sandbox(second, profile).ok)
            self.assertEqual(sandbox.preflight_network.call_count, 2)
            sandbox.network_containment_failed = True
            blocked = runner.ensure_sandbox(second, profile)
            self.assertFalse(blocked.ok)
            self.assertIn("containment", blocked.summary)
            self.assertEqual(sandbox.preflight_network.call_count, 2)

    def test_network_preflight_cache_keeps_the_sandbox_object_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace.from_path(temporary)
            sandbox = MagicMock()
            sandbox.preflight_network.return_value = ToolResult("ok", "network ready")
            runner = CommandRunner(sandbox=sandbox)
            profile = ExecutionProfile.from_capabilities({Capability.NETWORK})
            self.assertTrue(runner.ensure_sandbox(workspace, profile).ok)
            self.assertIs(runner._network_preflight_sandbox, sandbox)
            self.assertEqual(runner._network_preflight_workspace, workspace.root)

    def test_sandboxed_runner_forwards_its_resource_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace.from_path(temporary)
            sandbox = MagicMock()
            runner = CommandRunner(sandbox=sandbox, resource_limits=DEFAULT_RESOURCE_LIMITS)
            availability = ToolResult("ok", "sandbox ready")
            sandbox.run.return_value = ToolResult("ok", "done")
            result = runner.run_sandboxed(
                workspace, ["/bin/true"], availability=availability
            )
            self.assertTrue(result.ok)
            self.assertEqual(
                sandbox.run.call_args.kwargs["resource_limits"], DEFAULT_RESOURCE_LIMITS
            )


class ResourceLimitsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name) / "workspace"
        root.mkdir()
        self.workspace = Workspace.from_path(root)

    def tearDown(self):
        self.temp.cleanup()

    def test_invalid_limits_are_rejected_at_contract_construction(self):
        for kwargs in (
            {"nofile": 0},
            {"nofile": -1},
            {"core_bytes": -1},
            {"cpu_seconds": -1},
            {"file_size_bytes": -1},
            {"nofile": "4096"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ResourceLimits(**kwargs)

    def test_default_limits_only_enable_core_and_nofile(self):
        self.assertEqual(DEFAULT_RESOURCE_LIMITS.nofile, 4096)
        self.assertEqual(DEFAULT_RESOURCE_LIMITS.core_bytes, 0)
        self.assertIsNone(DEFAULT_RESOURCE_LIMITS.cpu_seconds)
        self.assertIsNone(DEFAULT_RESOURCE_LIMITS.file_size_bytes)

    def test_build_argv_wraps_command_with_prlimit(self):
        sandbox = tool_runtime.BubblewrapSandbox()
        argv = sandbox.build_argv(
            self.workspace,
            ["/bin/sh", "-lc", "echo constrained"],
            resource_limits=ResourceLimits(
                nofile=4096,
                core_bytes=0,
                cpu_seconds=7,
                file_size_bytes=12345,
            ),
        )
        wrapper_at = argv.index("/usr/bin/prlimit")
        self.assertEqual(
            argv[wrapper_at:],
            [
                "/usr/bin/prlimit",
                "--nofile=4096:4096",
                "--core=0:0",
                "--cpu=7:7",
                "--fsize=12345:12345",
                "--",
                "/bin/sh", "-lc", "echo constrained",
            ],
        )

    def test_no_active_limit_omits_prlimit(self):
        argv = tool_runtime.BubblewrapSandbox().build_argv(
            self.workspace, ["/bin/true"], resource_limits=ResourceLimits()
        )
        self.assertNotIn("/usr/bin/prlimit", argv)

    def test_active_limits_fail_closed_when_prlimit_is_unavailable(self):
        sandbox = tool_runtime.BubblewrapSandbox(prlimit_binary="/definitely/missing/prlimit")
        with patch("tool_runtime.subprocess.run") as run:
            result = sandbox.run(
                self.workspace, ["/bin/true"], resource_limits=DEFAULT_RESOURCE_LIMITS
            )
        self.assertEqual(result.status, "failed")
        self.assertIn("prlimit", result.summary)
        run.assert_not_called()

    def test_real_sandbox_applies_hard_limits_and_children_inherit(self):
        script = (
            "import json,resource,subprocess,sys\n"
            "child=subprocess.check_output([sys.executable, '-c', "
            "'import resource,json; print(json.dumps(resource.getrlimit(resource.RLIMIT_NOFILE)))'], text=True)\n"
            "raised=True\n"
            "try: resource.setrlimit(resource.RLIMIT_NOFILE, (8192,8192))\n"
            "except (ValueError,OSError): raised=False\n"
            "print(json.dumps({'nofile': resource.getrlimit(resource.RLIMIT_NOFILE), "
            "'core': resource.getrlimit(resource.RLIMIT_CORE), 'child': json.loads(child), "
            "'raise_allowed': raised}))\n"
        )
        result = tool_runtime.BubblewrapSandbox().run(
            self.workspace,
            ["/usr/bin/python3", "-c", script],
            resource_limits=DEFAULT_RESOURCE_LIMITS,
        )
        self.assertTrue(result.ok, result.to_legacy_text())
        observed = json.loads(result.stdout)
        self.assertEqual(observed["nofile"], [4096, 4096])
        self.assertEqual(observed["core"], [0, 0])
        self.assertEqual(observed["child"], [4096, 4096])
        self.assertFalse(observed["raise_allowed"])


class WorkspaceMountProfileTests(unittest.TestCase):
    """Step 1d-1: Bubblewrap materializes only granted workspace access."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.workspace = Workspace.from_path(self.workspace_root)
        self.marker = self.workspace_root / "host-marker.txt"
        self.marker.write_text("host marker", encoding="utf-8")
        self.outside = Path(self.temp.name) / "outside.txt"
        self.outside.write_text("outside", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def profile(*capabilities):
        return ExecutionProfile.from_capabilities(capabilities)

    def run_shell(self, profile, script, *, backend=NetworkBackend.CLOSED):
        return tool_runtime.BubblewrapSandbox().run(
            self.workspace,
            ["/bin/sh", "-lc", script],
            profile=profile,
            backend=backend,
            resource_limits=DEFAULT_RESOURCE_LIMITS,
        )

    def test_profile_without_filesystem_access_gets_private_empty_workspace(self):
        profile = self.profile()
        argv = tool_runtime.BubblewrapSandbox().build_argv(
            self.workspace, ["/bin/true"], profile=profile
        )
        self.assertNotIn(str(self.workspace_root), argv)
        self.assertIn("--dir", argv)
        result = self.run_shell(
            profile,
            "test \"$PWD\" = /workspace && test ! -e host-marker.txt && touch local-created",
        )
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertFalse((self.workspace_root / "local-created").exists())

    def test_filesystem_read_profile_is_real_read_only_workspace(self):
        profile = self.profile(Capability.FILESYSTEM_READ)
        sandbox = tool_runtime.BubblewrapSandbox()
        mount = sandbox.mount_root(self.workspace, profile)
        argv = sandbox.build_argv(self.workspace, ["/bin/true"], profile=profile)
        # Locate the workspace bind by destination: other read-only binds
        # (/usr, /etc/alternatives) precede it.
        bind_at = next(i for i, a in enumerate(argv)
                       if a == "--ro-bind" and argv[i + 2] == mount)
        self.assertEqual(argv[bind_at + 1:bind_at + 3], [str(self.workspace_root), mount])
        result = self.run_shell(
            profile,
            "test -r host-marker.txt && test \"$(cat host-marker.txt)\" = 'host marker' "
            "&& stat host-marker.txt >/dev/null "
            "&& ! touch created && ! printf changed > host-marker.txt "
            "&& ! rm host-marker.txt && ! mv host-marker.txt renamed "
            "&& ! mkdir created-dir && ! ln -s host-marker.txt created-link",
        )
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertEqual(self.marker.read_text(encoding="utf-8"), "host marker")
        self.assertFalse((self.workspace_root / "created").exists())

    def test_workspace_write_profile_keeps_real_read_write_workspace(self):
        profile = self.profile(Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE)
        sandbox = tool_runtime.BubblewrapSandbox()
        mount = sandbox.mount_root(self.workspace, profile)
        argv = sandbox.build_argv(self.workspace, ["/bin/true"], profile=profile)
        bind_at = next(i for i, a in enumerate(argv)
                       if a == "--bind" and argv[i + 2] == mount)
        self.assertEqual(argv[bind_at + 1:bind_at + 3], [str(self.workspace_root), mount])
        result = self.run_shell(profile, "printf changed > host-marker.txt && touch created")
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertEqual(self.marker.read_text(encoding="utf-8"), "changed")
        self.assertTrue((self.workspace_root / "created").exists())

    def test_shell_complex_does_not_expand_workspace_mount_permissions(self):
        readonly = self.profile(Capability.FILESYSTEM_READ, Capability.SHELL_COMPLEX)
        writable = self.profile(
            Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE, Capability.SHELL_COMPLEX
        )
        self.assertTrue(self.run_shell(readonly, "cat host-marker.txt | cat && ! touch denied").ok)
        self.assertFalse((self.workspace_root / "denied").exists())
        self.assertTrue(self.run_shell(writable, "printf rw | cat > shell-created").ok)
        self.assertEqual((self.workspace_root / "shell-created").read_text(), "rw")

    def test_symlink_escapes_remain_unusable_in_read_only_and_read_write_profiles(self):
        (self.workspace_root / "outside-link").symlink_to(self.outside)
        readonly = self.profile(Capability.FILESYSTEM_READ)
        writable = self.profile(Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE)
        self.assertTrue(
            self.run_shell(readonly, "test ! -e outside-link && ! ln -s host-marker.txt new-link").ok
        )
        self.assertFalse((self.workspace_root / "new-link").exists())
        self.assertTrue(self.run_shell(writable, "test ! -e outside-link && ln -s /tmp/nope created-link").ok)
        self.assertTrue((self.workspace_root / "created-link").is_symlink())
        self.assertTrue(self.run_shell(writable, "test ! -e created-link").ok)

    def test_network_only_gets_private_workspace_and_slirp(self):
        profile = self.profile(Capability.NETWORK)
        result = self.run_shell(
            profile,
            "test ! -e host-marker.txt && touch local-created "
            "&& /usr/sbin/ip route | grep -q 'default via 10.0.2.2'",
            backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertFalse((self.workspace_root / "local-created").exists())

    def test_network_read_only_and_network_read_write_materialize_their_profiles(self):
        readonly = self.profile(Capability.FILESYSTEM_READ, Capability.NETWORK)
        writable = self.profile(
            Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE, Capability.NETWORK
        )
        readonly_result = self.run_shell(
            readonly,
            "test -r host-marker.txt && ! printf changed > host-marker.txt "
            "&& /usr/sbin/ip route | grep -q 'default via 10.0.2.2'",
            backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertTrue(readonly_result.ok, readonly_result.to_legacy_text())
        self.assertEqual(self.marker.read_text(encoding="utf-8"), "host marker")
        writable_result = self.run_shell(
            writable,
            "printf network-rw > host-marker.txt && /usr/sbin/ip route | grep -q 'default via 10.0.2.2'",
            backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertTrue(writable_result.ok, writable_result.to_legacy_text())
        self.assertEqual(self.marker.read_text(encoding="utf-8"), "network-rw")

    def test_profile_none_retains_legacy_read_write_mount_but_runner_uses_explicit_profile(self):
        sandbox_argv = tool_runtime.BubblewrapSandbox()
        argv = sandbox_argv.build_argv(self.workspace, ["/bin/true"])
        mount = sandbox_argv.mount_root(self.workspace)
        bind_at = next(i for i, a in enumerate(argv)
                       if a == "--bind" and argv[i + 2] == mount)
        self.assertEqual(argv[bind_at + 1:bind_at + 3],
                         [str(self.workspace_root), mount])
        sandbox = MagicMock()
        sandbox.ensure_available.return_value = ToolResult("ok", "available")
        sandbox.run.return_value = ToolResult("ok", "done")
        runner = CommandRunner(sandbox=sandbox)
        runner.run_sandboxed(
            self.workspace,
            ["make"],
            self.profile(Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE),
        )
        self.assertIsNotNone(sandbox.run.call_args.kwargs["profile"])


class ReadOnlySandboxExecutionTests(unittest.TestCase):
    """Step 1d-2: authorized external reads are confined like every command."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        (self.workspace_root / "file.txt").write_text("needle\n", encoding="utf-8")
        self.outside = Path(self.temp.name) / "outside.txt"
        self.outside.write_text("outside marker", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(self.workspace_root)], check=True)
        self.workspace = Workspace.from_path(self.workspace_root)
        self.profile = ExecutionProfile.from_capabilities({Capability.FILESYSTEM_READ})

    def tearDown(self):
        self.temp.cleanup()

    def run_read_only(self, argv):
        return tool_runtime.BubblewrapSandbox().run(
            self.workspace,
            argv,
            profile=self.profile,
            resource_limits=DEFAULT_RESOURCE_LIMITS,
        )

    def test_representative_read_only_commands_run_in_read_only_workspace(self):
        cases = {
            "pwd": ["/bin/pwd"],
            "ls": ["/bin/ls", "file.txt"],
            "cat": ["/bin/cat", "file.txt"],
            "grep": ["/bin/grep", "needle", "file.txt"],
            "find": ["/usr/bin/find", ".", "-maxdepth", "1", "-name", "file.txt"],
            "git status": ["/usr/bin/git", "status", "--porcelain"],
        }
        outputs = {}
        for name, argv in cases.items():
            with self.subTest(command=name):
                result = self.run_read_only(argv)
                self.assertTrue(result.ok, result.to_legacy_text())
                outputs[name] = result.stdout
        self.assertEqual(
            outputs["pwd"].strip(),
            tool_runtime.BubblewrapSandbox().mount_root(self.workspace, self.profile))
        self.assertIn("file.txt", outputs["ls"])
        self.assertIn("needle", outputs["cat"])
        self.assertIn("needle", outputs["grep"])
        self.assertIn("file.txt", outputs["find"])

    @unittest.skipUnless(Path("/usr/bin/rg").is_file(),
                         "rg is not installed in the minimal /usr sandbox runtime")
    def test_rg_runs_when_installed_in_the_sandbox_runtime(self):
        result = self.run_read_only(["/usr/bin/rg", "needle", "file.txt"])
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertIn("needle", result.stdout)

    def test_read_only_sandbox_hides_host_and_enforces_limits(self):
        host_home = Path(self.temp.name) / "host-home"
        host_home.mkdir()
        (host_home / ".ssh").mkdir()
        (host_home / ".gitconfig").write_text("[user]\nname = host\n", encoding="utf-8")
        script = (
            "import os,resource,sys; "
            "checks=[os.environ.get('HOME') == '/home/sandbox', "
            "not os.path.exists('/home/sandbox/.ssh'), "
            "not os.path.exists('/home/sandbox/.gitconfig'), "
            f"not os.path.exists({str(self.outside)!r}), "
            "not os.path.exists('/opt/llm'), "
            # /etc/passwd IS exposed now, read-only: bitbake's is_local_uid()
            # opens it and dies without it. What must stay hidden is the
            # credential half of /etc, which the assertion below pins.
            "not os.path.exists('/etc/shadow'), "
            "not os.path.exists('/etc/gshadow'), "
            "not os.environ.get('SPEAR_TEST_SECRET'), "
            "resource.getrlimit(resource.RLIMIT_NOFILE) == (4096,4096), "
            "resource.getrlimit(resource.RLIMIT_CORE) == (0,0)]; "
            "sys.exit(0 if all(checks) else 1)"
        )
        with patch.dict(os.environ, {
            "HOME": str(host_home),
            "SPEAR_TEST_SECRET": "controlled-test-secret",
        }, clear=False):
            result = self.run_read_only(["/usr/bin/python3", "-c", script])
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertFalse(
            self.run_read_only(["/bin/cat", str(self.outside)]).ok,
            "an absolute host path must not be readable through the sandbox",
        )
        self.assertFalse(self.run_read_only(["/bin/sh", "-lc", "touch denied"]).ok)
        self.assertFalse((self.workspace_root / "denied").exists())


class AuditLoggerTests(unittest.TestCase):
    def test_mutation_audit_never_records_contents_or_secrets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            root.mkdir()
            workspace = Workspace.from_path(root)
            log = Path(temporary) / "audit" / "events.jsonl"
            AuditLogger(log).record_mutation(
                action="write_file",
                mode=ExecutionMode.ASK,
                workspace=workspace,
                paths=[root / "src" / "main.py"],
                command="TOKEN=super-secret make test",
                approved=True,
                result=ToolResult("ok", "file written", changed_paths=("src/main.py",)),
            )
            event = json.loads(log.read_text().strip())
            self.assertEqual(event["paths"], ["src/main.py"])
            self.assertNotIn("super-secret", log.read_text())
            self.assertNotIn("content", event)
            self.assertEqual(event["command"], "make (1 args)")

    def test_audit_keeps_required_and_granted_capabilities_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            root.mkdir()
            log = Path(temporary) / "audit" / "events.jsonl"
            AuditLogger(log).record_mutation(
                action="command",
                mode=ExecutionMode.ASK,
                workspace=Workspace.from_path(root),
                approved=False,
                result=ToolResult("denied", "network denied"),
                required_capabilities=(Capability.FILESYSTEM_READ, Capability.NETWORK),
                granted_capabilities=(Capability.FILESYSTEM_READ,),
            )
            event = json.loads(log.read_text())
            self.assertEqual(event["required_capabilities"], ["filesystem:read", "network"])
            self.assertEqual(event["granted_capabilities"], ["filesystem:read"])


class Phase1bBubblewrapContractTests(unittest.TestCase):
    """Phase 1b contract tests. They intentionally fail until implemented."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.workspace = Workspace.from_path(self.workspace_root)
        self.outside_root = Path(self.temp.name) / "outside"
        self.outside_root.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def sandbox(self, *, binary="bwrap"):
        sandbox_type = getattr(tool_runtime, "BubblewrapSandbox", None)
        self.assertIsNotNone(
            sandbox_type,
            "BubblewrapSandbox is not implemented yet (expected Phase 1b)",
        )
        return sandbox_type(binary=binary)

    def test_build_argv_exposes_workspace_only_at_its_mount(self):
        sandbox = self.sandbox()
        mount = sandbox.mount_root(self.workspace)
        argv = sandbox.build_argv(self.workspace, ["/bin/sh", "-lc", "pwd"])
        rendered = "\0".join(argv)
        bind_at = next(i for i, a in enumerate(argv)
                       if a == "--bind" and argv[i + 2] == mount)
        self.assertEqual(argv[bind_at + 1:bind_at + 3],
                         [str(self.workspace_root), mount])
        self.assertIn("--chdir", argv)
        chdir_at = argv.index("--chdir")
        # The cwd must be the mount, whatever the mount is: a --chdir that does
        # not match the bind lands the command outside the workspace.
        self.assertEqual(argv[chdir_at + 1], mount)
        self.assertNotIn("--ro-bind\0/\0/", rendered)

    def test_supported_profiles_preserve_phase_1b_confinement(self):
        for capabilities in (
            {Capability.FILESYSTEM_READ},
            {Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE},
            {Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE, Capability.SHELL_COMPLEX},
        ):
            with self.subTest(capabilities=capabilities):
                profile = ExecutionProfile.from_capabilities(capabilities)
                argv = self.sandbox().build_argv(self.workspace, ["/bin/true"], profile=profile)
                rendered = "\0".join(argv)
                self.assertIn("--unshare-net", argv)
                self.assertIn("--clearenv", argv)
                self.assertIn(self.sandbox().mount_root(self.workspace, profile), argv)
                self.assertIn("--tmpfs", argv)
                self.assertIn("/home/sandbox", argv)
                self.assertNotIn("/dev/nvidia0", rendered)
                self.assertNotIn("docker.sock", rendered)
                self.assertNotIn("SSH_AUTH_SOCK", rendered)

    def test_sensitive_profiles_fail_closed_before_bubblewrap_execution(self):
        for capabilities in (
            {Capability.GPU},
            {Capability.CONTAINER_RUNTIME},
            {Capability.SECRETS},
        ):
            with self.subTest(capabilities=capabilities):
                profile = ExecutionProfile.from_capabilities(capabilities)
                with self.assertRaisesRegex(ValueError, "capability/profile not implemented"):
                    self.sandbox().build_argv(self.workspace, ["/bin/true"], profile=profile)


class Slirp4netnsNetworkTests(unittest.TestCase):
    """Deterministic NETWORK prototype tests; Internet remains opt-in."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.workspace = Workspace.from_path(self.workspace_root)
        self.outside_root = Path(self.temp.name) / "outside"
        self.outside_root.mkdir()
        self.profile = ExecutionProfile.from_capabilities({Capability.NETWORK})

    def tearDown(self):
        self.temp.cleanup()

    def sandbox(self, **kwargs):
        kwargs.setdefault("network_ready_timeout_seconds", 2)
        return tool_runtime.BubblewrapSandbox(**kwargs)

    def assert_no_test_process(self, marker):
        processes = subprocess.check_output(["ps", "-eo", "args="], text=True)
        self.assertNotIn(str(marker), processes)
        self.assertNotIn("/tmp/edgem-slirp-", processes)

    def run_with_tracked_pidfd(self, argv, *, sandbox_kwargs=None):
        return self.sandbox(timeout_seconds=1, **(sandbox_kwargs or {})).run(
            self.workspace, argv, profile=self.profile,
            backend=NetworkBackend.SLIRP4NETNS,
        )

    def test_closed_backend_refuses_network_profile_and_slirp_keeps_unshare_net(self):
        sandbox = self.sandbox()
        with self.assertRaisesRegex(ValueError, "requires slirp4netns"):
            sandbox.build_argv(self.workspace, ["/bin/true"], profile=self.profile)
        argv = sandbox.build_argv(
            self.workspace, ["/bin/true"], profile=self.profile,
            backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertIn("--unshare-net", argv)
        self.assertNotIn("--share-net", argv)

    def test_network_namespace_has_slirp_topology_and_synthetic_dns(self):
        profile = ExecutionProfile.from_capabilities({
            Capability.NETWORK, Capability.SHELL_COMPLEX,
        })
        result = self.sandbox().run(
            self.workspace,
            ["/bin/sh", "-lc", "/usr/sbin/ip -brief addr; /usr/sbin/ip route; cat /etc/resolv.conf"],
            profile=profile,
            backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertIn("tap0", result.stdout)
        self.assertIn("10.0.2.100/24", result.stdout)
        self.assertIn("default via 10.0.2.2", result.stdout)
        self.assertIn("nameserver 10.0.2.3", result.stdout)

    def test_network_lifecycle_uses_and_closes_pidfd_on_success_and_timeout(self):
        success = self.run_with_tracked_pidfd(["/bin/true"])
        timed_out = self.run_with_tracked_pidfd(["/bin/sh", "-lc", "sleep 10"])
        self.assertTrue(success.ok, success.to_legacy_text())
        self.assertEqual(timed_out.status, "timeout")

    def test_network_uses_sync_fd_and_releases_once_after_ready(self):
        pipes = []
        writes = []
        bwrap_argvs = []
        real_pipe = os.pipe
        real_write = os.write
        real_popen = subprocess.Popen

        def tracked_pipe():
            pair = real_pipe()
            pipes.append(pair)
            return pair

        def tracked_write(fd, data):
            writes.append((fd, data))
            return real_write(fd, data)

        def tracked_popen(argv, *args, **kwargs):
            if argv and Path(argv[0]).name == "bwrap":
                bwrap_argvs.append(list(argv))
            return real_popen(argv, *args, **kwargs)

        with patch.object(tool_runtime.os, "pipe", side_effect=tracked_pipe), \
             patch.object(tool_runtime.os, "write", side_effect=tracked_write), \
             patch.object(tool_runtime.subprocess, "Popen", side_effect=tracked_popen):
            result = self.sandbox().run(
                self.workspace, ["/bin/true"], profile=self.profile,
                backend=NetworkBackend.SLIRP4NETNS,
            )
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertGreaterEqual(len(pipes), 2)
        block_read, release_write = pipes[1]
        argv = bwrap_argvs[0]
        self.assertEqual(argv[argv.index("--block-fd") + 1], str(block_read))
        self.assertEqual(argv[argv.index("--sync-fd") + 1], str(release_write))
        self.assertEqual([(fd, data) for fd, data in writes if fd == release_write],
                         [(release_write, b"x")])

    def test_setup_failures_never_write_sync_release(self):
        marker = self.workspace_root / "unexpected-release"
        cases = (
            ("helper", {"slirp_binary": "/bin/false"}, None),
            ("ready-timeout", {"slirp_binary": "/bin/false"}, None),
            ("pidfd-open", {}, patch.object(tool_runtime.os, "pidfd_open", side_effect=OSError("no"))),
            ("json", {}, patch.object(tool_runtime.json, "loads", side_effect=RuntimeError("no"))),
        )
        for name, sandbox_kwargs, extra_patch in cases:
            with self.subTest(name=name):
                writes = []
                real_write = os.write
                def tracked_write(fd, data):
                    writes.append((fd, data))
                    return real_write(fd, data)
                with patch.object(tool_runtime.os, "write", side_effect=tracked_write):
                    if extra_patch is None:
                        result = self.sandbox(**sandbox_kwargs).run(
                            self.workspace,
                            ["/bin/sh", "-lc", "touch /workspace/unexpected-release"],
                            profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
                        )
                    else:
                        with extra_patch:
                            result = self.sandbox(**sandbox_kwargs).run(
                                self.workspace,
                                ["/bin/sh", "-lc", "touch /workspace/unexpected-release"],
                                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
                            )
                self.assertFalse(result.ok)
                self.assertFalse(any(data == b"x" for _, data in writes))
                self.assertFalse(marker.exists())

    def test_pidfd_is_closed_after_helper_failure_and_python_exception(self):
        failed = self.run_with_tracked_pidfd(
            ["/bin/true"], sandbox_kwargs={"slirp_binary": "/bin/false"},
        )
        self.assertFalse(failed.ok)

        opened = []
        closed = []
        real_pidfd_open = os.pidfd_open
        real_close = os.close
        real_popen = subprocess.Popen

        def tracked_open(pid, flags=0):
            fd = real_pidfd_open(pid, flags)
            opened.append(fd)
            return fd

        def tracked_close(fd):
            closed.append(fd)
            return real_close(fd)

        calls = 0
        def fail_before_slirp(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("test orchestration exception")
            return real_popen(*args, **kwargs)

        # The helper's option probe is a preflight concern and spawns its own
        # process; prime it so the counter below tracks the orchestration only.
        sandbox = self.sandbox()
        self.assertIsNone(sandbox._slirp_pinned_namespace_status())

        with patch.object(tool_runtime.os, "pidfd_open", side_effect=tracked_open), \
             patch.object(tool_runtime.os, "close", side_effect=tracked_close), \
             patch.object(tool_runtime.subprocess, "Popen", side_effect=fail_before_slirp):
            result = sandbox.run(
                self.workspace, ["/bin/true"], profile=self.profile,
                backend=NetworkBackend.SLIRP4NETNS,
            )
        self.assertFalse(result.ok)
        self.assertEqual(len(opened), 1)
        self.assertIn(opened[0], closed)
        self.assert_no_test_process(self.workspace_root)

    def test_pidfd_open_failure_fails_closed_before_command_release(self):
        marker = self.workspace_root / "pidfd-open-marker"
        with patch.object(tool_runtime.os, "pidfd_open", side_effect=OSError("blocked")):
            result = self.sandbox().run(
                self.workspace, ["/bin/sh", "-lc", "touch /workspace/pidfd-open-marker"],
                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
            )
        self.assertFalse(result.ok)
        self.assertFalse(marker.exists())

    def test_missing_pidfd_support_fails_closed_before_command_release(self):
        marker = self.workspace_root / "pidfd-support-marker"
        with patch.object(tool_runtime.os, "pidfd_open", None), \
             patch.object(tool_runtime.signal, "pidfd_send_signal", None):
            result = self.sandbox().run(
                self.workspace, ["/bin/sh", "-lc", "touch /workspace/pidfd-support-marker"],
                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
            )
        self.assertFalse(result.ok)
        self.assertFalse(marker.exists())

    def test_network_preflight_requires_pidfd_support(self):
        with patch.object(tool_runtime.os, "pidfd_open", None):
            result = self.sandbox().preflight_network(self.workspace)
        self.assertFalse(result.ok)
        self.assertIn("pidfd", result.summary)

    def test_pidfd_stop_handles_immediate_transient_and_already_dead_results(self):
        sandbox = self.sandbox()
        with patch.object(tool_runtime.signal, "pidfd_send_signal") as send:
            self.assertTrue(sandbox._stop_namespace_child(123))
        send.assert_called_once_with(123, signal.SIGKILL)

        with patch.object(
            tool_runtime.signal, "pidfd_send_signal", side_effect=[OSError("once"), None]
        ) as send:
            self.assertTrue(sandbox._stop_namespace_child(123))
        self.assertEqual(send.call_count, 2)

        with patch.object(
            tool_runtime.signal, "pidfd_send_signal", side_effect=ProcessLookupError
        ):
            self.assertTrue(sandbox._stop_namespace_child(123))

    def test_wait_pidfd_exit_observes_exit_and_times_out_without_pid_lookup(self):
        process = subprocess.Popen(["/bin/sleep", "30"])
        pidfd = os.pidfd_open(process.pid)
        try:
            self.assertFalse(tool_runtime.BubblewrapSandbox._wait_pidfd_exit(pidfd, 0.01))
            signal.pidfd_send_signal(pidfd, signal.SIGKILL)
            self.assertTrue(tool_runtime.BubblewrapSandbox._wait_pidfd_exit(pidfd, 1))
        finally:
            process.wait()
            os.close(pidfd)

    def test_permanent_pidfd_signal_failure_closes_parent_writer_and_poisons_backend(self):
        marker = self.workspace_root / "pidfd-signal-marker"
        stalled_helper = Path(self.temp.name) / "stalled-slirp-pidfd"
        # Stands in for a capable helper: it must advertise pinned-namespace
        # attachment, otherwise the runtime fails closed before this scenario.
        stalled_helper.write_text(
            '#!/bin/sh\n'
            'case "$1" in --help) echo "--netns-type --userns-path"; exit 0;; esac\n'
            'exec /bin/sleep 30\n',
            encoding="utf-8",
        )
        stalled_helper.chmod(0o755)
        calls = []
        real_pidfd_open = os.pidfd_open
        real_pipe = os.pipe
        opened = []
        test_pidfds = []
        pipes = []
        safe_close_calls = []
        writes = []
        wait_calls = []
        supervisor_stops = []

        def tracked_open(pid, flags=0):
            fd = real_pidfd_open(pid, flags)
            opened.append(fd)
            test_pidfds.append(os.dup(fd))
            return fd

        def permanently_failing_signal(pidfd, signum, siginfo=None, flags=0):
            calls.append(("signal", pidfd))
            raise OSError("simulated permanent pidfd failure")

        def tracked_pipe():
            pair = real_pipe()
            pipes.append(pair)
            return pair

        original_wait = tool_runtime.BubblewrapSandbox._wait_pidfd_exit
        original_stop = tool_runtime.BubblewrapSandbox._stop_process
        original_safe_close = tool_runtime.BubblewrapSandbox._safe_close

        def tracked_wait(pidfd, timeout):
            wait_calls.append((pidfd, timeout))
            return original_wait(pidfd, timeout)

        def tracked_stop(process, **kwargs):
            supervisor_stops.append(process)
            return original_stop(process, **kwargs)

        def tracked_safe_close(fd):
            safe_close_calls.append((opened[-1] if opened else None, fd))
            return original_safe_close(fd)

        real_write = os.write
        def tracked_write(fd, data):
            writes.append((fd, data))
            return real_write(fd, data)

        with patch.object(tool_runtime.os, "pidfd_open", side_effect=tracked_open), \
             patch.object(tool_runtime.signal, "pidfd_send_signal", side_effect=permanently_failing_signal), \
             patch.object(tool_runtime.os, "pipe", side_effect=tracked_pipe), \
             patch.object(tool_runtime.os, "write", side_effect=tracked_write), \
             patch.object(tool_runtime.BubblewrapSandbox, "_safe_close", side_effect=tracked_safe_close), \
             patch.object(tool_runtime.BubblewrapSandbox, "_wait_pidfd_exit", side_effect=tracked_wait), \
             patch.object(tool_runtime.BubblewrapSandbox, "_stop_process", side_effect=tracked_stop):
            sandbox = self.sandbox(
                slirp_binary=str(stalled_helper), network_ready_timeout_seconds=0.1,
            )
            result = sandbox.run(
                self.workspace, ["/bin/sh", "-lc", "touch /workspace/pidfd-signal-marker"],
                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
            )
        self.assertFalse(result.ok)
        self.assertIn("could not verify child termination", result.summary)
        self.assertFalse(marker.exists())
        self.assertEqual(len(opened), 1)
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(calls and calls[0][0] == "signal")
        self.assertEqual(wait_calls[0][0], opened[0])
        self.assertTrue(supervisor_stops)
        self.assertGreaterEqual(len(pipes), 2)
        release_write = pipes[1][1]
        self.assertFalse(any(fd == release_write for fd, _ in writes))
        self.assertIn((opened[0], release_write), safe_close_calls)
        self.assertTrue(sandbox.network_containment_failed)
        retry = sandbox.run(
            self.workspace, ["/bin/true"], profile=self.profile,
            backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertFalse(retry.ok)
        self.assertIn("containment", retry.summary)
        closed = sandbox.run(self.workspace, ["/bin/true"])
        self.assertTrue(closed.ok, closed.to_legacy_text())

        # A test-only duplicate preserves identity after the runtime closes
        # its own pidfd; remove this explicitly identifiable blocked child.
        test_pidfd = test_pidfds[0]
        signal.pidfd_send_signal(test_pidfd, signal.SIGKILL)
        self.assertTrue(tool_runtime.BubblewrapSandbox._wait_pidfd_exit(test_pidfd, 1))
        os.close(test_pidfd)
        self.assert_no_test_process(stalled_helper)

    def test_network_sandbox_cannot_reach_host_loopback_or_slirp_gateway(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            program = (
                "import socket; "
                f"a=socket.socket();a.settimeout(1);x=a.connect_ex(('127.0.0.1',{port}));a.close(); "
                f"b=socket.socket();b.settimeout(1);y=b.connect_ex(('10.0.2.2',{port}));b.close(); "
                "raise SystemExit(0 if x != 0 and y != 0 else 1)"
            )
            result = self.sandbox().run(
                self.workspace, ["/usr/bin/python3", "-c", program],
                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
            )
        finally:
            server.close()
        self.assertTrue(result.ok, result.to_legacy_text())

    def test_missing_or_failed_slirp_fails_closed_before_command_release(self):
        marker = self.workspace_root / "command-ran"
        missing = self.sandbox(slirp_binary="/definitely/missing/slirp4netns")
        result = missing.run(
            self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
            profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertFalse(result.ok)
        self.assertFalse(marker.exists())

        non_executable_helper = Path(self.temp.name) / "non-executable-slirp"
        non_executable_helper.write_text("not executable", encoding="utf-8")
        non_executable_helper.chmod(0o644)
        non_executable = self.sandbox(slirp_binary=str(non_executable_helper))
        result = non_executable.run(
            self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
            profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertFalse(result.ok)
        self.assertFalse(marker.exists())

    def test_bwrap_failure_and_slirp_ready_timeout_never_release_command(self):
        marker = self.workspace_root / "command-ran"
        # ``/bin/false`` passes the executable precheck but fails before it
        # emits Bubblewrap's namespace info, so slirp must never start.
        broken_bwrap = self.sandbox(binary="/bin/false")
        result = broken_bwrap.run(
            self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
            profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertFalse(result.ok)
        self.assertFalse(marker.exists())

        stalled_helper = Path(self.temp.name) / "stalled-slirp"
        # Stands in for a capable helper: it must advertise pinned-namespace
        # attachment, otherwise the runtime fails closed before this scenario.
        stalled_helper.write_text(
            '#!/bin/sh\n'
            'case "$1" in --help) echo "--netns-type --userns-path"; exit 0;; esac\n'
            'exec /bin/sleep 30\n',
            encoding="utf-8",
        )
        stalled_helper.chmod(0o755)
        stalled = self.sandbox(
            slirp_binary=str(stalled_helper), network_ready_timeout_seconds=0.1,
        )
        result = stalled.run(
            self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
            profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertFalse(result.ok)
        self.assertFalse(marker.exists())
        self.assert_no_test_process(stalled_helper)

    def test_python_orchestration_failure_cleans_bwrap_and_slirp(self):
        marker = self.workspace_root / "command-ran"
        with patch.object(tool_runtime.json, "loads", side_effect=RuntimeError("test failure")):
            result = self.sandbox().run(
                self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
            )
        self.assertFalse(result.ok)
        self.assertFalse(marker.exists())
        self.assert_no_test_process(self.workspace_root)

        failed = self.sandbox(slirp_binary="/bin/false")
        result = failed.run(
            self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
            profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertFalse(result.ok)
        self.assertFalse(marker.exists())

    def test_network_lifecycle_cleans_helpers_on_exit_error_and_timeout(self):
        sandbox = self.sandbox(timeout_seconds=1)
        ok = sandbox.run(self.workspace, ["/bin/true"], profile=self.profile,
                         backend=NetworkBackend.SLIRP4NETNS)
        failed = sandbox.run(self.workspace, ["/bin/sh", "-lc", "exit 7"],
                             profile=self.profile, backend=NetworkBackend.SLIRP4NETNS)
        timed_out = sandbox.run(self.workspace, ["/bin/sh", "-lc", "sleep 10"],
                                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS)
        self.assertTrue(ok.ok, ok.to_legacy_text())
        self.assertFalse(failed.ok)
        self.assertEqual(failed.exit_code, 7)
        self.assertEqual(timed_out.status, "timeout")
        self.assert_no_test_process(self.workspace_root)

    @unittest.skipUnless(os.environ.get("SPEAR_TEST_NETWORK") == "1",
                         "set SPEAR_TEST_NETWORK=1 for opt-in outbound DNS integration")
    def test_opt_in_outbound_dns_and_https_tls(self):
        dns = self.sandbox().run(
            self.workspace, ["/usr/bin/getent", "hosts", "example.com"],
            profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertTrue(dns.ok, dns.to_legacy_text())
        https = self.sandbox().run(
            self.workspace,
            ["/usr/bin/curl", "--fail", "--silent", "--show-error", "https://example.com/"],
            profile=self.profile, backend=NetworkBackend.SLIRP4NETNS,
        )
        self.assertTrue(https.ok, https.to_legacy_text())

    def test_preflight_runs_a_real_mini_sandbox(self):
        sandbox = self.sandbox()
        result = sandbox.preflight(self.workspace)
        self.assertIsInstance(result, ToolResult)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertIn("sandbox", result.summary.lower())

    def test_workspace_is_visible_at_workspace_and_write_is_allowed(self):
        sandbox = self.sandbox()
        mount = sandbox.mount_root(self.workspace)
        result = sandbox.run(
            self.workspace,
            ["/bin/sh", "-lc", f"test \"$PWD\" = {mount} && touch writable.txt"]
        )
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertTrue((self.workspace_root / "writable.txt").exists())

    def test_write_outside_workspace_is_impossible(self):
        sandbox = self.sandbox()
        target = self.outside_root / "escaped.txt"
        result = sandbox.run(
            self.workspace, ["/bin/sh", "-lc", f"touch {target}"]
        )
        self.assertFalse(result.ok)
        self.assertFalse(target.exists())

    def test_symlink_to_host_outside_workspace_is_unusable(self):
        (self.workspace_root / "outside-link").symlink_to(self.outside_root,
                                                            target_is_directory=True)
        sandbox = self.sandbox()
        result = sandbox.run(
            self.workspace, ["/bin/sh", "-lc", "touch /workspace/outside-link/escaped.txt"]
        )
        self.assertFalse(result.ok)
        self.assertFalse((self.outside_root / "escaped.txt").exists())

    def test_home_and_sensitive_environment_are_not_inherited(self):
        host_home = self.outside_root / "host-home"
        host_home.mkdir()
        (host_home / ".ssh").mkdir()
        sandbox = self.sandbox()
        with patch.dict(os.environ, {"HOME": str(host_home), "TOKEN": "host-secret"}):
            result = sandbox.run(
                self.workspace,
                ["/bin/sh", "-lc", "test \"$HOME\" != '" + str(host_home)
                 + "' && test ! -e \"$HOME/.ssh\" && test -z \"${TOKEN:-}\""],
            )
        self.assertTrue(result.ok, result.to_legacy_text())

    def test_tmp_is_private(self):
        host_marker = Path(self.temp.name) / "host-tmp-marker"
        host_marker.write_text("host")
        sandbox = self.sandbox()
        result = sandbox.run(
            self.workspace,
            ["/bin/sh", "-lc", f"test ! -e {host_marker} && touch /tmp/sandbox-marker"],
        )
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertFalse((Path("/tmp") / "sandbox-marker").exists())

    def test_network_is_unavailable(self):
        sandbox = self.sandbox()
        argv = sandbox.build_argv(self.workspace, ["/bin/sh", "-lc", "true"])
        self.assertIn("--unshare-net", argv)
        result = sandbox.run(
            self.workspace,
            ["/bin/sh", "-lc", "if test -d /sys/class/net; then "
             "test \"$(find /sys/class/net -mindepth 1 -maxdepth 1 -printf '%f\\n' | sort)\" = lo; fi"],
        )
        self.assertTrue(result.ok, result.to_legacy_text())

    def test_auto_workspace_mutation_fails_closed_without_sandbox(self):
        sandbox = self.sandbox(binary="/definitely/missing/bwrap")
        import rag_chat

        old_mode = rag_chat.EXECUTION_MODE
        old_sandbox = getattr(rag_chat.COMMAND_RUNNER, "sandbox", None)
        try:
            rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
            rag_chat.COMMAND_RUNNER.sandbox = sandbox
            result = rag_chat.run_cmd("make test", need_confirm=False)
        finally:
            rag_chat.EXECUTION_MODE = old_mode
            rag_chat.COMMAND_RUNNER.sandbox = old_sandbox
        self.assertIn("sandbox", result.lower())
        self.assertIn("unavailable", result.lower())

    def test_auto_shell_complex_fails_closed_without_sandbox(self):
        sandbox = self.sandbox(binary="/definitely/missing/bwrap")
        import rag_chat

        old_mode = rag_chat.EXECUTION_MODE
        old_sandbox = getattr(rag_chat.COMMAND_RUNNER, "sandbox", None)
        try:
            rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
            rag_chat.COMMAND_RUNNER.sandbox = sandbox
            result = rag_chat.run_cmd("ls | head", need_confirm=False)
        finally:
            rag_chat.EXECUTION_MODE = old_mode
            rag_chat.COMMAND_RUNNER.sandbox = old_sandbox
        self.assertIn("sandbox", result.lower())
        self.assertIn("unavailable", result.lower())

    def test_ask_complex_command_keeps_explicit_confirmation_gate(self):
        sandbox = self.sandbox(binary="/definitely/missing/bwrap")
        import rag_chat

        old_mode = rag_chat.EXECUTION_MODE
        old_sandbox = getattr(rag_chat.COMMAND_RUNNER, "sandbox", None)
        try:
            rag_chat.EXECUTION_MODE = ExecutionMode.ASK
            rag_chat.COMMAND_RUNNER.sandbox = sandbox
            with patch.object(rag_chat, "confirm", return_value=False) as confirm:
                result = rag_chat.run_cmd("ls | head", need_confirm=False)
        finally:
            rag_chat.EXECUTION_MODE = old_mode
            rag_chat.COMMAND_RUNNER.sandbox = old_sandbox
        confirm.assert_called_once()
        self.assertEqual(result, "CANCELLED")

    def test_ask_complex_command_fails_closed_after_approved_confirmation(self):
        sandbox = self.sandbox(binary="/definitely/missing/bwrap")
        import rag_chat

        old_mode = rag_chat.EXECUTION_MODE
        old_sandbox = getattr(rag_chat.COMMAND_RUNNER, "sandbox", None)
        try:
            rag_chat.EXECUTION_MODE = ExecutionMode.ASK
            rag_chat.COMMAND_RUNNER.sandbox = sandbox
            with patch.object(rag_chat, "confirm", return_value=True) as confirm, \
                 patch.object(rag_chat.COMMAND_RUNNER, "run_simple") as simple, \
                 patch.object(rag_chat.COMMAND_RUNNER, "run_complex_approved") as complex_run:
                result = rag_chat.run_cmd("printf compatibility | cat", need_confirm=False)
        finally:
            rag_chat.EXECUTION_MODE = old_mode
            rag_chat.COMMAND_RUNNER.sandbox = old_sandbox
        confirm.assert_called_once()
        simple.assert_not_called()
        complex_run.assert_not_called()
        self.assertIn("sandbox unavailable", result)


class Phase1bBubblewrapAdversarialTests(unittest.TestCase):
    """Real escape attempts against the Phase 1b Bubblewrap boundary."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.workspace = Workspace.from_path(self.workspace_root)
        self.outside = Path(self.temp.name) / "host-outside"
        self.outside.mkdir()
        self.sandbox = tool_runtime.BubblewrapSandbox()
        # Where the tree answers inside: its own host path, or /workspace under
        # the legacy mount. Asked of the sandbox so the test tracks the code.
        self.mount = self.sandbox.mount_root(self.workspace)

    def tearDown(self):
        self.temp.cleanup()

    def run_shell(self, script):
        return self.sandbox.run(self.workspace, ["/bin/sh", "-lc", script])

    def test_workspace_writable_but_host_filesystem_is_unreachable(self):
        marker = self.outside / "marker.txt"
        marker.write_text("unchanged")
        result = self.run_shell(
            f"printf inside > {self.mount}/inside.txt"
        )
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertEqual((self.workspace_root / "inside.txt").read_text(), "inside")

        # The operator's own home, not one particular person's: the point is
        # that the sandbox hides whatever home the caller has.
        home = str(Path.home())

        for target in (
            "/etc/spear-adversarial-marker",
            f"{home}/spear-adversarial-marker",
            str(marker),
            f"{self.mount}/../etc/spear-adversarial-marker",
            "/opt/llm/spear-adversarial-marker",
        ):
            attempted = self.run_shell(f"printf escaped > {target}")
            self.assertFalse(attempted.ok, f"unexpected host write: {target}")
        self.assertEqual(marker.read_text(), "unchanged")
        self.assertFalse(Path("/etc/spear-adversarial-marker").exists())
        self.assertFalse(Path(home, "spear-adversarial-marker").exists())
        self.assertTrue(self.run_shell(f"test ! -e {home}").ok)
        self.assertFalse(Path("/opt/llm/spear-adversarial-marker").exists())
        self.assertTrue(self.run_shell("test ! -e /opt/llm").ok)

    def test_preexisting_and_runtime_symlink_escapes_are_unusable(self):
        target = self.outside / "marker.txt"
        target.write_text("unchanged")
        directory = self.outside / "directory"
        directory.mkdir()
        (self.workspace_root / "file-link").symlink_to(target)
        (self.workspace_root / "dir-link").symlink_to(directory, target_is_directory=True)

        result = self.run_shell(
            f"test ! -e {self.mount}/file-link && test ! -e {self.mount}/dir-link "
            f"&& ln -s '{target}' {self.mount}/runtime-link "
            f"&& test ! -e {self.mount}/runtime-link"
        )
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertFalse(self.run_shell(f"printf x > {self.mount}/file-link").ok)
        self.assertFalse(self.run_shell(f"printf x > {self.mount}/dir-link/escaped").ok)
        self.assertFalse(self.run_shell(f"printf x > {self.mount}/runtime-link").ok)
        self.assertEqual(target.read_text(), "unchanged")
        self.assertFalse((directory / "escaped").exists())

    def test_environment_home_and_user_configuration_are_not_inherited(self):
        host_home = self.outside / "host-home"
        host_home.mkdir()
        (host_home / ".ssh").mkdir()
        (host_home / ".gitconfig").write_text("[user]\nname = host\n")
        secrets = {
            "OPENAI_API_KEY": "openai-test-secret",
            "ANTHROPIC_API_KEY": "anthropic-test-secret",
            "AWS_SECRET_ACCESS_KEY": "aws-test-secret",
            "SSH_AUTH_SOCK": "/tmp/host-agent.sock",
            "SPEAR_TEST_SECRET": "spear-test-secret",
            "HOME": str(host_home),
        }
        with patch.dict(os.environ, secrets, clear=False):
            result = self.run_shell(
                "test \"$HOME\" = /home/sandbox && test ! -e \"$HOME/.ssh\" "
                "&& test ! -e \"$HOME/.gitconfig\" "
                "&& test -z \"${OPENAI_API_KEY:-}${ANTHROPIC_API_KEY:-}\" "
                "&& test -z \"${AWS_SECRET_ACCESS_KEY:-}${SSH_AUTH_SOCK:-}${SPEAR_TEST_SECRET:-}\""
            )
        self.assertTrue(result.ok, result.to_legacy_text())

    def test_network_namespace_has_no_host_connection_or_inherited_sockets(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            argv = self.sandbox.build_argv(self.workspace, ["/bin/true"])
            self.assertIn("--unshare-net", argv)
            result = self.sandbox.run(
                self.workspace,
                ["/usr/bin/python3", "-c", (
                    "import socket,sys; "
                    f"s=socket.socket(); s.settimeout(1); "
                    f"sys.exit(0 if s.connect_ex(('127.0.0.1',{port})) != 0 else 1)"
                )],
            )
        finally:
            server.close()
        self.assertTrue(result.ok, result.to_legacy_text())
        interfaces = self.run_shell(
            "if test -d /sys/class/net; then "
            "test \"$(find /sys/class/net -mindepth 1 -maxdepth 1 -printf '%f\\n' | sort)\" = lo; fi "
            "&& test ! -e /sys/class/net/eth0"
        )
        self.assertTrue(interfaces.ok, interfaces.to_legacy_text())

    def test_proc_pid_namespace_and_child_lifecycle(self):
        result = self.run_shell(
            "test -e /proc/1 && test \"$$\" -gt 1 && test \"$$\" -le 3 "
            "&& test \"$(readlink /proc/1/exe)\" = /usr/bin/bwrap "
            "&& test ! -e /proc/999999 && ! kill -0 999999"
        )
        self.assertTrue(result.ok, result.to_legacy_text())

        before = subprocess.check_output(["ps", "-eo", "args="], text=True)
        child = self.run_shell("sleep 30 & echo child-started")
        self.assertTrue(child.ok, child.to_legacy_text())
        after = subprocess.check_output(["ps", "-eo", "args="], text=True)
        self.assertEqual(after.count("sleep 30"), before.count("sleep 30"))
        self.assertIn("--die-with-parent", self.sandbox.build_argv(self.workspace, ["/bin/true"]))

    def test_tmp_dev_and_complex_shell_escapes_remain_confined(self):
        host_tmp = self.outside / "host-tmp-marker"
        host_tmp.write_text("host")
        target = self.outside / "external.txt"
        target.write_text("unchanged")
        (self.workspace_root / "external-link").symlink_to(target)
        script = (
            "touch /tmp/sandbox-marker && test ! -e '" + str(host_tmp) + "' "
            "&& test ! -e /dev/nvidia0 && test ! -b /dev/sda "
            "&& target='" + str(target) + "'; "
            "(cd / && test ! -e \"$target\") | cat > /workspace/pipeline.txt; "
            "test ! -e \"$(readlink -f /workspace/external-link 2>/dev/null || true)\""
        )
        result = self.run_shell(script)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertFalse((Path("/tmp") / "sandbox-marker").exists())
        self.assertEqual(target.read_text(), "unchanged")

    def test_availability_states_and_cache_fail_closed_in_auto(self):
        unavailable = tool_runtime.BubblewrapSandbox(binary="/definitely/missing/bwrap")
        self.assertEqual(unavailable.ensure_available(self.workspace).status, "failed")
        self.assertEqual(unavailable.availability, tool_runtime.SandboxAvailability.ABSENT)
        not_executable = self.outside / "not-executable-bwrap"
        not_executable.write_text("not executable")
        not_executable.chmod(0o644)
        blocked = tool_runtime.BubblewrapSandbox(binary=str(not_executable))
        self.assertEqual(blocked.ensure_available(self.workspace).status, "failed")
        self.assertEqual(blocked.availability, tool_runtime.SandboxAvailability.INEXECUTABLE)

        refused = tool_runtime.BubblewrapSandbox(binary="bwrap")
        with patch.object(refused, "preflight", return_value=ToolResult("failed", "refused")) as preflight:
            self.assertEqual(refused.ensure_available(self.workspace).status, "failed")
            self.assertEqual(refused.ensure_available(self.workspace).status, "failed")
        self.assertEqual(preflight.call_count, 1)
        self.assertEqual(refused.availability, tool_runtime.SandboxAvailability.REFUSED)


class Phase1bRagChatRoutingTests(unittest.TestCase):
    """Step 1b-3: classify in rag_chat, execute through CommandRunner."""

    @classmethod
    def setUpClass(cls):
        import rag_chat

        cls.rag_chat = rag_chat

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.old_mode = self.rag_chat.EXECUTION_MODE
        self.old_workspace = self.rag_chat.WORKSPACE
        self.old_project_root = self.rag_chat.PROJECT_ROOT
        self.old_sandbox = self.rag_chat.COMMAND_RUNNER.sandbox
        self.old_audit_logger = self.rag_chat.AUDIT_LOGGER
        self.rag_chat.WORKSPACE = Workspace.from_path(self.workspace_root)
        self.rag_chat.PROJECT_ROOT = str(self.workspace_root)
        self.rag_chat.AUDIT_LOGGER = AuditLogger(Path(self.temp.name) / "audit.jsonl")

    def tearDown(self):
        self.rag_chat.EXECUTION_MODE = self.old_mode
        self.rag_chat.WORKSPACE = self.old_workspace
        self.rag_chat.PROJECT_ROOT = self.old_project_root
        self.rag_chat.COMMAND_RUNNER.sandbox = self.old_sandbox
        self.rag_chat.AUDIT_LOGGER = self.old_audit_logger
        self.temp.cleanup()

    def sandbox_available(self):
        sandbox = MagicMock()
        sandbox.ensure_available.return_value = ToolResult("ok", "sandbox available")
        sandbox.preflight_network.return_value = ToolResult("ok", "network sandbox available")
        sandbox.run.return_value = ToolResult("ok", "command completed", stdout="sandboxed")
        return sandbox

    def test_read_only_commands_route_only_through_sandbox_in_every_mode(self):
        for mode in (ExecutionMode.SAFE, ExecutionMode.ASK, ExecutionMode.AUTO):
            with self.subTest(mode=mode):
                self.rag_chat.EXECUTION_MODE = mode
                sandbox = self.sandbox_available()
                self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
                with patch.object(self.rag_chat, "confirm") as confirm, \
                     patch.object(self.rag_chat.COMMAND_RUNNER, "run_simple") as simple, \
                     patch.object(self.rag_chat.COMMAND_RUNNER, "run_complex_approved") as complex_run:
                    result = self.rag_chat.run_cmd("git status", need_confirm=False)
                self.assertEqual(result, "sandboxed")
                confirm.assert_not_called()
                simple.assert_not_called()
                complex_run.assert_not_called()
                sandbox.ensure_available.assert_called_once_with(self.rag_chat.WORKSPACE)
                sandbox.run.assert_called_once_with(
                    self.rag_chat.WORKSPACE,
                    ["git", "status"],
                    profile=ANY,
                    resource_limits=DEFAULT_RESOURCE_LIMITS,
                    cgroup_limits=DEFAULT_CGROUP_LIMITS,
                )
                profile = sandbox.run.call_args.kwargs["profile"]
                self.assertTrue(profile.workspace_read)
                self.assertFalse(profile.workspace_write)
                self.assertFalse(profile.network)

    def test_read_only_fails_closed_without_bubblewrap_and_never_runs_on_host(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.SAFE
        sandbox = MagicMock()
        sandbox.ensure_available.return_value = ToolResult("failed", "sandbox unavailable")
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.rag_chat.COMMAND_RUNNER, "run_simple") as simple, \
             patch.object(self.rag_chat.COMMAND_RUNNER, "run_complex_approved") as complex_run:
            result = self.rag_chat.run_cmd("pwd", need_confirm=False)
        self.assertIn("sandbox unavailable", result)
        simple.assert_not_called()
        complex_run.assert_not_called()
        sandbox.run.assert_not_called()

    def test_auto_workspace_mutation_uses_available_sandbox(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
        sandbox = self.sandbox_available()
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox

        result = self.rag_chat.run_cmd("make test", need_confirm=False)

        self.assertEqual(result, "sandboxed")
        sandbox.ensure_available.assert_called_once_with(self.rag_chat.WORKSPACE)
        sandbox.run.assert_called_once_with(
            self.rag_chat.WORKSPACE, ["make", "test"], profile=ANY,
            resource_limits=DEFAULT_RESOURCE_LIMITS,
            cgroup_limits=DEFAULT_CGROUP_LIMITS,
        )
        self.assertEqual(sandbox.run.call_args.kwargs["profile"].capabilities, frozenset({
            Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE,
        }))

    def test_auto_shell_complex_uses_shell_inside_available_sandbox(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
        sandbox = self.sandbox_available()
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox

        result = self.rag_chat.run_cmd("printf sandboxed | cat", need_confirm=False)

        self.assertEqual(result, "sandboxed")
        sandbox.run.assert_called_once_with(
            self.rag_chat.WORKSPACE,
            tool_runtime.shell_argv("printf sandboxed | cat"),
            profile=ANY,
            resource_limits=DEFAULT_RESOURCE_LIMITS,
            cgroup_limits=DEFAULT_CGROUP_LIMITS,
        )

    def test_ask_workspace_mutation_uses_sandbox_after_confirmation(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK
        sandbox = self.sandbox_available()
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.rag_chat, "confirm", return_value=True) as confirm:
            result = self.rag_chat.run_cmd("pytest tests", need_confirm=False)

        self.assertEqual(result, "sandboxed")
        confirm.assert_called_once()
        sandbox.run.assert_called_once_with(
            self.rag_chat.WORKSPACE, ["pytest", "tests"], profile=ANY,
            resource_limits=DEFAULT_RESOURCE_LIMITS,
            cgroup_limits=DEFAULT_CGROUP_LIMITS,
        )

    def test_ask_shell_complex_uses_sandbox_after_confirmation(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK
        sandbox = self.sandbox_available()
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.rag_chat, "confirm", return_value=True) as confirm:
            result = self.rag_chat.run_cmd("printf sandboxed | cat", need_confirm=False)

        self.assertEqual(result, "sandboxed")
        confirm.assert_called_once()
        sandbox.run.assert_called_once_with(
            self.rag_chat.WORKSPACE,
            tool_runtime.shell_argv("printf sandboxed | cat"),
            profile=ANY,
            resource_limits=DEFAULT_RESOURCE_LIMITS,
            cgroup_limits=DEFAULT_CGROUP_LIMITS,
        )

    def test_a_failing_pipeline_stage_is_not_reported_as_success(self):
        """`make | tail` must not exit 0 when make fails.

        A pipeline reports its LAST stage's status, so truncating a build log
        hides the build's own failure. The model then reads exit 0 and reports
        a success that never happened -- the concrete way a fabricated
        conclusion survives a harness that already prints exit codes.
        """
        self.rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
        sandbox = self.sandbox_available()
        sandbox.run.return_value = ToolResult(
            "failed", "command failed", stdout="last lines", exit_code=2)
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
        result = self.rag_chat.run_cmd("make 2>&1 | tail -3", need_confirm=False)
        self.assertIn("(exit 2)", result)

    def test_sigpipe_from_head_is_not_reported_as_a_failure(self):
        """`find . | head -5` is not a failed command.

        pipefail surfaces the SIGPIPE that head causes upstream, which is the
        POINT of head, not an error. Reporting 141 would send the model
        debugging one of its most common idioms.
        """
        self.rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
        sandbox = self.sandbox_available()
        sandbox.run.return_value = ToolResult(
            "failed", "command failed", stdout="a.c\nb.c", exit_code=141)
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
        result = self.rag_chat.run_cmd("find . -name '*.c' | head -5",
                                       need_confirm=False)
        self.assertNotIn("exit", result)
        self.assertIn("a.c", result)

    def test_ask_fails_closed_when_sandbox_is_unavailable(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK
        sandbox = MagicMock()
        sandbox.ensure_available.return_value = ToolResult("failed", "sandbox unavailable")
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.rag_chat, "confirm", return_value=True), \
             patch.object(self.rag_chat.COMMAND_RUNNER, "run_simple") as simple, \
             patch.object(self.rag_chat.COMMAND_RUNNER, "run_complex_approved",
                          return_value=ToolResult("ok", "done", stdout="compatibility")) as complex_run:
            mutate = self.rag_chat.run_cmd("make test", need_confirm=False)
            complex_result = self.rag_chat.run_cmd("printf compatibility | cat", need_confirm=False)

        self.assertIn("sandbox unavailable", mutate)
        self.assertIn("sandbox unavailable", complex_result)
        simple.assert_not_called()
        complex_run.assert_not_called()

    def test_ask_declined_command_does_not_start_sandbox_or_runner(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK
        sandbox = self.sandbox_available()
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.rag_chat, "confirm", return_value=False), \
             patch.object(self.rag_chat.COMMAND_RUNNER, "run_simple") as simple, \
             patch.object(self.rag_chat.COMMAND_RUNNER, "run_complex_approved") as complex_run:
            mutate = self.rag_chat.run_cmd("make test", need_confirm=False)
            complex_result = self.rag_chat.run_cmd("printf x | cat", need_confirm=False)

        self.assertEqual(mutate, "CANCELLED")
        self.assertEqual(complex_result, "CANCELLED")
        sandbox.ensure_available.assert_not_called()
        sandbox.run.assert_not_called()
        simple.assert_not_called()
        complex_run.assert_not_called()

    def test_auto_unavailable_sandbox_never_uses_non_sandboxed_runner(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
        sandbox = MagicMock()
        sandbox.ensure_available.return_value = ToolResult("failed", "sandbox unavailable")
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.rag_chat.COMMAND_RUNNER, "run_simple") as simple, \
             patch.object(self.rag_chat.COMMAND_RUNNER, "run_complex_approved") as complex_run:
            mutate = self.rag_chat.run_cmd("make test", need_confirm=False)
            complex_result = self.rag_chat.run_cmd("printf x | cat", need_confirm=False)

        self.assertIn("sandbox unavailable", mutate)
        self.assertIn("sandbox unavailable", complex_result)
        simple.assert_not_called()
        complex_run.assert_not_called()

    def test_auto_fails_closed_for_absent_inexecutable_and_refused_sandboxes(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
        non_executable = Path(self.temp.name) / "not-executable-bwrap"
        non_executable.write_text("not executable")
        non_executable.chmod(0o644)
        refused = tool_runtime.BubblewrapSandbox(binary="bwrap")
        refused.availability = tool_runtime.SandboxAvailability.REFUSED
        sandboxes = (
            tool_runtime.BubblewrapSandbox(binary="/definitely/missing/bwrap"),
            tool_runtime.BubblewrapSandbox(binary=str(non_executable)),
            refused,
        )
        for sandbox in sandboxes:
            self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
            with patch.object(self.rag_chat.COMMAND_RUNNER, "run_simple") as simple, \
                 patch.object(self.rag_chat.COMMAND_RUNNER, "run_complex_approved") as complex_run:
                result = self.rag_chat.run_cmd("make test", need_confirm=False)
            self.assertIn("sandbox unavailable", result)
            simple.assert_not_called()
            complex_run.assert_not_called()

    def test_dangerous_command_never_reaches_bubblewrap(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
        sandbox = self.sandbox_available()
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox

        result = self.rag_chat.run_cmd("rm -rf build", need_confirm=False)

        self.assertIn("command denied", result)
        sandbox.ensure_available.assert_not_called()
        sandbox.run.assert_not_called()

    def test_network_is_refused_before_execution_in_safe(self):
        """SAFE stops it before bubblewrap is even asked to exist, and the
        refusal names both ways forward: the flags, and fetch_url."""
        self.rag_chat.EXECUTION_MODE = ExecutionMode.SAFE
        sandbox = self.sandbox_available()
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.rag_chat.COMMAND_RUNNER, "run_simple") as simple, \
             patch.object(self.rag_chat.COMMAND_RUNNER, "run_complex_approved") as complex_run:
            result = self.rag_chat.run_cmd("curl https://example.invalid",
                                           need_confirm=False)
        self.assertIn("network", result)
        self.assertIn("fetch_url", result)
        sandbox.ensure_available.assert_not_called()
        sandbox.run.assert_not_called()
        simple.assert_not_called()
        complex_run.assert_not_called()

    def test_auto_network_reaches_the_slirp_sandbox_without_a_prompt(self):
        """AUTO grants the network now; it must still go through the network
        sandbox, and only through it."""
        self.rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
        sandbox = self.sandbox_available()
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.rag_chat, "confirm") as confirm:
            result = self.rag_chat.run_cmd("curl https://example.invalid",
                                           need_confirm=False)
        self.assertEqual(result, "sandboxed")
        confirm.assert_not_called()
        sandbox.preflight_network.assert_called_once_with(self.rag_chat.WORKSPACE)

    def test_ask_network_routes_only_to_slirp_sandbox_after_confirmation(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK
        sandbox = self.sandbox_available()
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.rag_chat, "confirm", return_value=True) as confirm, \
             patch.object(self.rag_chat.COMMAND_RUNNER, "run_simple") as simple, \
             patch.object(self.rag_chat.COMMAND_RUNNER, "run_complex_approved") as complex_run:
            result = self.rag_chat.run_cmd("curl https://example.invalid", need_confirm=False)
        self.assertEqual(result, "sandboxed")
        confirm.assert_called_once()
        sandbox.preflight_network.assert_called_once_with(self.rag_chat.WORKSPACE)
        sandbox.run.assert_called_once_with(
            self.rag_chat.WORKSPACE, ["curl", "https://example.invalid"],
            profile=ANY, backend=NetworkBackend.SLIRP4NETNS,
            resource_limits=DEFAULT_RESOURCE_LIMITS,
            cgroup_limits=DEFAULT_CGROUP_LIMITS,
        )
        self.assertTrue(sandbox.run.call_args.kwargs["profile"].network)
        simple.assert_not_called()
        complex_run.assert_not_called()

    def test_ask_network_declined_starts_no_sandbox_or_preflight(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK
        sandbox = self.sandbox_available()
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.rag_chat, "confirm", return_value=False) as confirm:
            result = self.rag_chat.run_cmd("curl https://example.invalid", need_confirm=False)
        self.assertEqual(result, "CANCELLED")
        confirm.assert_called_once()
        sandbox.preflight_network.assert_not_called()
        sandbox.run.assert_not_called()

    def test_ask_network_fails_closed_for_preflight_or_containment_failure(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK
        for summary in ("slirp unavailable", "network containment failure"):
            with self.subTest(summary=summary):
                sandbox = self.sandbox_available()
                sandbox.preflight_network.return_value = ToolResult("failed", summary)
                self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
                with patch.object(self.rag_chat, "confirm", return_value=True), \
                     patch.object(self.rag_chat.COMMAND_RUNNER, "run_simple") as simple:
                    result = self.rag_chat.run_cmd("curl https://example.invalid", need_confirm=False)
                self.assertIn(summary, result)
                sandbox.run.assert_not_called()
                simple.assert_not_called()

    def test_ask_remote_write_and_ssh_are_denied_without_confirmation(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK
        sandbox = self.sandbox_available()
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
        for command in (
            "curl -X POST https://example.invalid", "git push", "ssh example.invalid",
        ):
            with self.subTest(command=command), patch.object(self.rag_chat, "confirm") as confirm:
                result = self.rag_chat.run_cmd(command, need_confirm=False)
            self.assertIn("required capabilities", result)
            confirm.assert_not_called()
        sandbox.preflight_network.assert_not_called()
        sandbox.run.assert_not_called()

    def test_ask_network_shell_uses_single_confirmation_and_slirp(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK
        sandbox = self.sandbox_available()
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.rag_chat, "confirm", return_value=True) as confirm:
            result = self.rag_chat.run_cmd("curl https://example.invalid | head", need_confirm=False)
        self.assertEqual(result, "sandboxed")
        confirm.assert_called_once()
        sandbox.run.assert_called_once_with(
            self.rag_chat.WORKSPACE,
            tool_runtime.shell_argv("curl https://example.invalid | head"),
            profile=ANY, backend=NetworkBackend.SLIRP4NETNS,
            resource_limits=DEFAULT_RESOURCE_LIMITS,
            cgroup_limits=DEFAULT_CGROUP_LIMITS,
        )
        profile = sandbox.run.call_args.kwargs["profile"]
        self.assertTrue(profile.network)
        self.assertTrue(profile.shell_complex)

    def test_ask_network_preflight_is_cached_and_audited(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK
        sandbox = self.sandbox_available()
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.rag_chat, "confirm", return_value=True):
            self.rag_chat.run_cmd("curl https://example.invalid", need_confirm=False)
            self.rag_chat.run_cmd("curl https://example.invalid", need_confirm=False)
        self.assertEqual(sandbox.preflight_network.call_count, 1)
        event = json.loads((Path(self.temp.name) / "audit.jsonl").read_text().splitlines()[0])
        self.assertEqual(event["required_capabilities"], ["network"])
        self.assertEqual(event["granted_capabilities"], ["network"])
        self.assertTrue(event["execution_profile"]["network"])
        self.assertTrue(event["sandboxed"])

    @unittest.skipUnless(os.environ.get("SPEAR_TEST_NETWORK") == "1",
                         "set SPEAR_TEST_NETWORK=1 for opt-in end-to-end routing")
    def test_opt_in_ask_network_routes_through_slirp(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.ASK
        self.rag_chat.COMMAND_RUNNER.sandbox = tool_runtime.BubblewrapSandbox()
        with patch.object(self.rag_chat, "confirm", return_value=True):
            result = self.rag_chat.run_cmd(
                "curl --fail --silent --show-error https://example.com/", need_confirm=False
            )
        self.assertNotIn("ERROR:", result)

    def test_command_audit_records_capability_names_only(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
        sandbox = self.sandbox_available()
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox

        self.rag_chat.run_cmd("make test", need_confirm=False)

        event = json.loads((Path(self.temp.name) / "audit.jsonl").read_text())
        self.assertEqual(event["required_capabilities"], [
            "filesystem:read", "workspace:write",
        ])
        self.assertEqual(event["granted_capabilities"], [
            "filesystem:read", "workspace:write",
        ])

    def test_sandbox_preflight_is_cached_between_sandboxed_commands(self):
        self.rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
        sandbox = tool_runtime.BubblewrapSandbox(binary="bwrap")
        self.rag_chat.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(sandbox, "preflight",
                          return_value=ToolResult("ok", "sandbox preflight completed")) as preflight, \
             patch.object(sandbox, "run",
                          return_value=ToolResult("ok", "done", stdout="sandboxed")):
            self.rag_chat.run_cmd("make test", need_confirm=False)
            self.rag_chat.run_cmd("pytest tests", need_confirm=False)

        self.assertEqual(preflight.call_count, 1)


# ---------------------------------------------------------------------------
# Pinned-namespace slirp attachment: removing the info-fd -> helper-spawn race
# ---------------------------------------------------------------------------


class AlternativesMountTests(unittest.TestCase):
    """/etc/alternatives must be visible, and nothing else from /etc."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.workspace = Workspace.from_path(self.workspace_root)
        self.profile = ExecutionProfile.from_capabilities({
            Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE,
            Capability.SHELL_COMPLEX,
        })
        self.addCleanup(self.temp.cleanup)

    def test_build_argv_binds_the_alternatives_directory_read_only(self):
        argv = tool_runtime.BubblewrapSandbox().build_argv(
            self.workspace, ["/bin/true"])
        self.assertIn("--dir", argv)
        rendered = "\0".join(argv)
        self.assertIn("--ro-bind\0/etc/alternatives\0/etc/alternatives", rendered)
        # Never a writable bind, and never the whole of /etc.
        self.assertNotIn("--bind\0/etc", rendered)
        self.assertNotIn("--ro-bind\0/etc\0/etc", rendered)

    def test_missing_alternatives_directory_is_simply_not_bound(self):
        spec = tool_runtime.SandboxSpec(alternatives="/definitely/missing/alternatives")
        argv = tool_runtime.BubblewrapSandbox(spec=spec).build_argv(
            self.workspace, ["/bin/true"])
        self.assertNotIn("/definitely/missing/alternatives", argv)
        self.assertIn("/etc", argv)          # the mount point still exists

    def test_etc_is_sealed_read_only(self):
        result = tool_runtime.BubblewrapSandbox().run(
            self.workspace,
            ["/bin/sh", "-lc", "touch /etc/escape 2>&1; echo rc=$?; "
                               "mkdir /etc/d 2>&1; echo rc2=$?"],
            profile=self.profile)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertIn("Read-only file system", result.stdout)
        self.assertIn("rc=1", result.stdout)
        self.assertIn("rc2=1", result.stdout)

    def test_cc_resolves_inside_the_sandbox(self):
        result = tool_runtime.BubblewrapSandbox().run(
            self.workspace, ["/bin/sh", "-lc", "command -v cc && cc --version | head -1"],
            profile=self.profile)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertIn("/usr/bin/cc", result.stdout)

    def test_make_uses_the_implicit_cc_rule(self):
        (self.workspace_root / "a.c").write_text(
            '#include <stdio.h>\nint main(void){puts("ok");return 0;}\n',
            encoding="utf-8")
        (self.workspace_root / "Makefile").write_text(
            "all: a\na: a.c\n\t$(CC) -O2 -o a a.c\n", encoding="utf-8")
        result = tool_runtime.BubblewrapSandbox().run(
            self.workspace, ["/usr/bin/make"], profile=self.profile)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertTrue((self.workspace_root / "a").exists())

    def test_no_other_etc_content_is_exposed(self):
        """Exactly three entries, and the list is the point.

        /etc is a tmpfs so nothing outside the workspace is writable. What is
        bound into it is bound deliberately: `alternatives` because cc and awk
        are symlinks through it, and passwd/group because build tools resolve
        uids -- bitbake's is_local_uid() opens /etc/passwd and dies without it.
        Anything else appearing here is a leak, and shadow above all.
        """
        result = tool_runtime.BubblewrapSandbox().run(
            self.workspace, ["/bin/sh", "-lc", "ls -A /etc"], profile=self.profile)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertEqual(sorted(result.stdout.split()),
                         ["alternatives", "group", "passwd"])

    def test_the_credential_half_of_etc_stays_out(self):
        result = tool_runtime.BubblewrapSandbox().run(
            self.workspace,
            ["/bin/sh", "-lc", "cat /etc/shadow /etc/gshadow /etc/sudoers 2>&1; echo rc=$?"],
            profile=self.profile)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertNotIn("root:", result.stdout)
        self.assertIn("rc=1", result.stdout)

    def test_the_alternatives_bind_is_not_writable(self):
        result = tool_runtime.BubblewrapSandbox().run(
            self.workspace,
            ["/bin/sh", "-lc", "touch /etc/alternatives/x 2>&1; echo rc=$?"],
            profile=self.profile)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertIn("rc=1", result.stdout)

    def test_network_profile_still_gets_its_resolver_files(self):
        argv = tool_runtime.BubblewrapSandbox().build_argv(
            self.workspace, ["/bin/true"],
            profile=ExecutionProfile.from_capabilities({Capability.NETWORK}),
            backend=NetworkBackend.SLIRP4NETNS,
            network_files=(Path("/tmp/resolv.conf"), Path("/tmp/nsswitch.conf"), None))
        rendered = "\0".join(argv)
        self.assertIn("/etc/resolv.conf", rendered)
        self.assertIn("/etc/nsswitch.conf", rendered)
        self.assertIn("/etc/alternatives", rendered)
        # /etc is created exactly once even though two features want it, and
        # it is sealed after the last bind into it.
        self.assertEqual(
            [argv[i + 1] for i, a in enumerate(argv) if a == "--tmpfs"].count("/etc"), 1)
        self.assertEqual(
            [argv[i + 1] for i, a in enumerate(argv) if a == "--dir"].count("/etc"), 0)
        seal = argv.index("--remount-ro")
        self.assertEqual(argv[seal + 1], "/etc")
        last_etc_bind = max(i for i, a in enumerate(argv)
                            if a == "--ro-bind" and argv[i + 2].startswith("/etc"))
        self.assertGreater(seal, last_etc_bind)


class PinnedNamespaceAttachmentTests(unittest.TestCase):
    """Fast, always-on checks on argv construction and descriptor hygiene."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.workspace = Workspace.from_path(self.workspace_root)
        self.profile = ExecutionProfile.from_capabilities({Capability.NETWORK})
        self.addCleanup(self.temp.cleanup)

    def sandbox(self, **kwargs):
        kwargs.setdefault("network_ready_timeout_seconds", 2)
        return tool_runtime.BubblewrapSandbox(**kwargs)

    def spawn_and_capture(self, argv, **run_kwargs):
        spawned = []
        real_popen = subprocess.Popen

        def tracked(command, *args, **kwargs):
            spawned.append((list(command), kwargs.get("pass_fds")))
            return real_popen(command, *args, **kwargs)

        with patch.object(tool_runtime.subprocess, "Popen", side_effect=tracked):
            result = self.sandbox().run(self.workspace, argv, profile=self.profile,
                                        backend=NetworkBackend.SLIRP4NETNS, **run_kwargs)
        helper = [s for s in spawned
                  if "slirp4netns" in s[0][0] and "--help" not in s[0]]
        return result, helper[0] if helper else None

    def test_helper_receives_pinned_namespace_paths_not_a_pid(self):
        result, helper = self.spawn_and_capture(["/bin/true"])
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertIsNotNone(helper)
        helper_argv, pass_fds = helper
        self.assertIn("--netns-type=path", helper_argv)
        userns = [a for a in helper_argv if a.startswith("--userns-path=")]
        self.assertEqual(len(userns), 1)
        userns_fd = int(userns[0].rsplit("/", 1)[1])
        # Positional target is the pinned netns, never a bare PID.
        netns_arg = helper_argv[-2]
        self.assertTrue(netns_arg.startswith("/proc/self/fd/"), helper_argv)
        netns_fd = int(netns_arg.rsplit("/", 1)[1])
        # Both handles must be explicitly passed to the helper, and nothing else
        # beyond the ready/exit pipes.
        self.assertIn(userns_fd, pass_fds)
        self.assertIn(netns_fd, pass_fds)
        self.assertEqual(len(pass_fds), 4)
        # The protocol itself is unchanged.
        self.assertIn("--configure", helper_argv)
        self.assertIn("--disable-host-loopback", helper_argv)
        self.assertIn("--ready-fd", helper_argv)
        self.assertIn("--exit-fd", helper_argv)
        self.assertEqual(helper_argv[-1], "tap0")

    def test_command_never_inherits_the_namespace_handles(self):
        result = self.sandbox().run(
            self.workspace, ["/bin/ls", "-l", "/proc/self/fd"],
            profile=self.profile, backend=NetworkBackend.SLIRP4NETNS)
        self.assertTrue(result.ok, result.to_legacy_text())
        targets = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if "->" in parts:
                targets.append((parts[parts.index("->") - 1], parts[-1]))
        numbers = sorted(int(fd) for fd, _ in targets)
        # 0/1/2 plus the directory descriptor `ls` opens for itself.
        self.assertEqual(numbers, [0, 1, 2, 3], targets)
        rendered = " ".join(target for _, target in targets)
        self.assertNotIn("net:[", rendered)
        self.assertNotIn("user:[", rendered)

    def test_supervisor_leaks_no_descriptor_on_success_or_failure(self):
        def open_count():
            return len(os.listdir(f"/proc/{os.getpid()}/fd"))

        baseline = open_count()
        self.sandbox().run(self.workspace, ["/bin/true"], profile=self.profile,
                           backend=NetworkBackend.SLIRP4NETNS)
        self.assertLessEqual(open_count(), baseline)

        # Setup failure: bwrap never reports namespace info.
        self.sandbox(binary="/bin/false").run(
            self.workspace, ["/bin/true"], profile=self.profile,
            backend=NetworkBackend.SLIRP4NETNS)
        self.assertLessEqual(open_count(), baseline)

        # Python-level exception in the middle of the orchestration.
        with patch.object(tool_runtime.json, "loads", side_effect=RuntimeError("boom")):
            self.sandbox().run(self.workspace, ["/bin/true"], profile=self.profile,
                               backend=NetworkBackend.SLIRP4NETNS)
        self.assertLessEqual(open_count(), baseline)

    def test_helper_without_pinned_namespace_support_fails_closed(self):
        incapable = Path(self.temp.name) / "old-slirp"
        incapable.write_text('#!/bin/sh\necho "--ready-fd --exit-fd"\nexit 0\n',
                             encoding="utf-8")
        incapable.chmod(0o755)
        marker = self.workspace_root / "command-ran"
        writes = []
        real_write = os.write

        def tracked_write(fd, data):
            writes.append((fd, data))
            return real_write(fd, data)

        with patch.object(tool_runtime.os, "write", side_effect=tracked_write):
            result = self.sandbox(slirp_binary=str(incapable)).run(
                self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS)
        self.assertFalse(result.ok)
        self.assertIn("--netns-type", result.summary)
        self.assertFalse(marker.exists())
        self.assertEqual(writes, [], "no release may follow a fail-closed decision")

    def test_netns_mismatch_fails_closed_without_releasing_the_command(self):
        marker = self.workspace_root / "command-ran"
        writes = []
        real_write = os.write
        real_loads = json.loads

        def tampered(data, *args, **kwargs):
            info = real_loads(data, *args, **kwargs)
            if isinstance(info, dict) and "net-namespace" in info:
                info["net-namespace"] = info["net-namespace"] + 1
            return info

        def tracked_write(fd, data):
            writes.append((fd, data))
            return real_write(fd, data)

        with patch.object(tool_runtime.json, "loads", side_effect=tampered), \
             patch.object(tool_runtime.os, "write", side_effect=tracked_write):
            result = self.sandbox().run(
                self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS)
        self.assertFalse(result.ok)
        self.assertIn("does not match", result.summary)
        self.assertFalse(marker.exists())
        self.assertEqual(writes, [])

    def test_namespace_handles_are_type_checked_before_the_helper_runs(self):
        """Defence in depth: the pinned handles must be the expected ns types."""
        marker = self.workspace_root / "command-ran"
        writes = []
        real_write = os.write
        real_ioctl = fcntl.ioctl

        def wrong_type(fd, request, *args):
            if request == tool_runtime.NS_GET_NSTYPE:
                return 0  # neither CLONE_NEWNET nor CLONE_NEWUSER
            return real_ioctl(fd, request, *args)

        def tracked_write(fd, data):
            writes.append((fd, data))
            return real_write(fd, data)

        with patch.object(tool_runtime.fcntl, "ioctl", side_effect=wrong_type), \
             patch.object(tool_runtime.os, "write", side_effect=tracked_write):
            result = self.sandbox().run(
                self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
                profile=self.profile, backend=NetworkBackend.SLIRP4NETNS)
        self.assertFalse(result.ok)
        self.assertIn("namespace", result.summary)
        self.assertFalse(marker.exists())
        self.assertEqual(writes, [])

    def test_real_namespace_handles_have_the_expected_types(self):
        observed = {}
        real_ioctl = fcntl.ioctl

        def watching(fd, request, *args):
            result = real_ioctl(fd, request, *args)
            if request == tool_runtime.NS_GET_USERNS:
                observed["net"] = real_ioctl(fd, tool_runtime.NS_GET_NSTYPE)
                observed["user"] = real_ioctl(result, tool_runtime.NS_GET_NSTYPE)
            return result

        with patch.object(tool_runtime.fcntl, "ioctl", side_effect=watching):
            result = self.sandbox().run(self.workspace, ["/bin/true"],
                                        profile=self.profile,
                                        backend=NetworkBackend.SLIRP4NETNS)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertEqual(observed.get("net"), tool_runtime.CLONE_NEWNET)
        self.assertEqual(observed.get("user"), tool_runtime.CLONE_NEWUSER)

    def test_pidfd_remains_the_process_identity_alongside_namespace_handles(self):
        opened = []
        real_pidfd_open = os.pidfd_open

        def tracked(pid, flags=0):
            opened.append(pid)
            return real_pidfd_open(pid, flags)

        with patch.object(tool_runtime.os, "pidfd_open", side_effect=tracked):
            result = self.sandbox().run(self.workspace, ["/bin/true"],
                                        profile=self.profile,
                                        backend=NetworkBackend.SLIRP4NETNS)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertEqual(len(opened), 1, "the pidfd must still authenticate the child")


@unittest.skipUnless(os.environ.get("SPEAR_TEST_NETWORK_RACE") == "1",
                     "set SPEAR_TEST_NETWORK_RACE=1 for opt-in timing-race runs")
class OptInNetworkRaceTests(unittest.TestCase):
    """Repeated runs with a deliberate delay inside the formerly fatal window."""

    ITERATIONS = 40

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.workspace = Workspace.from_path(self.workspace_root)
        self.profile = ExecutionProfile.from_capabilities({Capability.NETWORK})
        self.addCleanup(self.temp.cleanup)

    def run_with_delay(self, delay, iterations=None):
        """Delay between pinning the namespaces and spawning the helper.

        ``os.pidfd_open`` is called immediately after the pin and immediately
        before the helper is spawned, so wrapping it injects the delay exactly
        where the PID-based attachment used to fail.
        """
        real_pidfd_open = os.pidfd_open

        def delayed(pid, flags=0):
            fd = real_pidfd_open(pid, flags)
            time.sleep(delay)
            return fd

        failures = []
        iterations = iterations or self.ITERATIONS
        sandbox = tool_runtime.BubblewrapSandbox(network_ready_timeout_seconds=5)
        with patch.object(tool_runtime.os, "pidfd_open", side_effect=delayed):
            for _ in range(iterations):
                result = sandbox.run(self.workspace, ["/bin/true"],
                                     profile=self.profile,
                                     backend=NetworkBackend.SLIRP4NETNS)
                if not result.ok:
                    failures.append(result.to_legacy_text()[:120])
        return failures

    def test_stable_with_20ms_delay(self):
        failures = self.run_with_delay(0.020)
        self.assertEqual(failures, [], f"{len(failures)}/{self.ITERATIONS} failed")

    def test_stable_with_50ms_delay(self):
        failures = self.run_with_delay(0.050)
        self.assertEqual(failures, [], f"{len(failures)}/{self.ITERATIONS} failed")

    def test_stable_with_50ms_delay_under_cpu_load(self):
        workers = [
            subprocess.Popen(
                [sys.executable, "-c",
                 "import time\nt=time.monotonic()\nwhile time.monotonic()-t<90: pass"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(4)
        ]
        try:
            failures = self.run_with_delay(0.050)
        finally:
            for worker in workers:
                worker.kill()
                worker.wait()
        self.assertEqual(failures, [], f"{len(failures)}/{self.ITERATIONS} failed")

    def test_characterization_old_pid_based_attachment_still_races(self):
        """Why the pinned form exists: the PID-based one fails deterministically.

        Exercises slirp4netns directly, not the runtime, which no longer offers
        the PID-based attachment.
        """
        sandbox = tool_runtime.BubblewrapSandbox(network_ready_timeout_seconds=5)
        failures = 0
        attempts = 10
        for _ in range(attempts):
            with tempfile.TemporaryDirectory(prefix="spear-race-char-") as tmp:
                files = sandbox._network_files(Path(tmp))
                info_read, info_write = os.pipe()
                block_read, release_write = os.pipe()
                argv = sandbox.build_argv(
                    self.workspace, ["/bin/true"], profile=self.profile,
                    backend=NetworkBackend.SLIRP4NETNS, network_files=files,
                    info_fd=info_write, block_fd=block_read, sync_fd=release_write)
                bwrap = subprocess.Popen(
                    argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    shell=False, pass_fds=(info_write, block_read, release_write),
                    env={})
                os.close(info_write)
                os.close(block_read)
                info = json.loads(sandbox._wait_json_fd(info_read, 5).decode())
                child = info["child-pid"]
                time.sleep(0.020)
                ready_read, ready_write = os.pipe()
                exit_read, exit_write = os.pipe()
                helper = subprocess.run(
                    ["/usr/bin/slirp4netns", "--configure", "--disable-host-loopback",
                     "--ready-fd", str(ready_write), "--exit-fd", str(exit_read),
                     str(child), "tap0"],
                    capture_output=True, text=True, timeout=20,
                    pass_fds=(ready_write, exit_read), env={})
                if "setns" in helper.stderr:
                    failures += 1
                for fd in (info_read, release_write, ready_read, ready_write,
                           exit_read, exit_write):
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                for stream in (bwrap.stdout, bwrap.stderr):
                    stream.close()
                bwrap.kill()
                bwrap.wait()
                try:
                    os.kill(child, signal.SIGKILL)
                except OSError:
                    pass
        self.assertEqual(failures, attempts,
                         "the PID-based attachment no longer races; revisit the fix")


# ---------------------------------------------------------------------------
# STEP 1d-3c: cgroup v2 resource control through a transient systemd user scope
# ---------------------------------------------------------------------------


class CgroupLimitsContractTests(unittest.TestCase):
    """The dataclass is a pure, immutable contract: no systemd knowledge."""

    def test_an_empty_contract_is_inactive(self):
        self.assertFalse(CgroupLimits().active)
        self.assertEqual(CgroupLimits().systemd_properties(), [])
        self.assertEqual(CgroupLimits().required_controllers, frozenset())

    def test_production_defaults_are_active_and_calibrated(self):
        """The values calibrated in STEP 1d-3d, now in force."""
        self.assertTrue(DEFAULT_CGROUP_LIMITS.active)
        self.assertEqual(DEFAULT_CGROUP_LIMITS.memory_max_bytes, 2 * 1024 ** 3)
        self.assertEqual(DEFAULT_CGROUP_LIMITS.memory_swap_max_bytes, 0)
        self.assertEqual(DEFAULT_CGROUP_LIMITS.tasks_max, 256)
        self.assertEqual(DEFAULT_CGROUP_LIMITS.cpu_quota_percent, 800)
        self.assertEqual(DEFAULT_CGROUP_LIMITS.systemd_properties(), [
            "MemoryMax=2147483648",
            "MemorySwapMax=0",
            "TasksMax=256",
            "CPUQuota=800%",
        ])
        self.assertEqual(DEFAULT_CGROUP_LIMITS.required_controllers,
                         frozenset({"cpu", "memory", "pids"}))

    def test_production_defaults_keep_headroom_over_the_measured_peaks(self):
        """Guards against a future tightening below what real work needs.

        Measured peaks on the calibration workloads: 518 MiB and 47 tasks,
        both reached by ``make -j22``.
        """
        self.assertGreaterEqual(DEFAULT_CGROUP_LIMITS.memory_max_bytes,
                                3 * 518 * 1024 ** 2)
        self.assertGreaterEqual(DEFAULT_CGROUP_LIMITS.tasks_max, 3 * 47)

    def test_each_field_alone_activates_the_contract(self):
        for kwargs in (
            {"memory_max_bytes": 1},
            {"memory_swap_max_bytes": 0},
            {"tasks_max": 1},
            {"cpu_quota_percent": 1},
        ):
            with self.subTest(**kwargs):
                self.assertTrue(CgroupLimits(**kwargs).active)

    def test_valid_values_are_accepted(self):
        limits = CgroupLimits(memory_max_bytes=268435456, memory_swap_max_bytes=0,
                              tasks_max=64, cpu_quota_percent=200)
        self.assertTrue(limits.active)
        self.assertEqual(limits.memory_max_bytes, 268435456)
        self.assertEqual(limits.cpu_quota_percent, 200)

    def test_invalid_values_are_rejected(self):
        for kwargs in (
            {"memory_max_bytes": 0},
            {"memory_max_bytes": -1},
            {"memory_swap_max_bytes": -1},
            {"tasks_max": 0},
            {"tasks_max": -3},
            {"cpu_quota_percent": 0},
            {"cpu_quota_percent": -50},
            {"memory_max_bytes": 1.5},
            {"memory_max_bytes": "64M"},
            {"cpu_quota_percent": "200%"},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    CgroupLimits(**kwargs)

    def test_bool_is_explicitly_rejected(self):
        for field in ("memory_max_bytes", "memory_swap_max_bytes", "tasks_max",
                      "cpu_quota_percent"):
            for value in (True, False):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        CgroupLimits(**{field: value})

    def test_is_frozen(self):
        limits = CgroupLimits(tasks_max=8)
        with self.assertRaises(Exception):
            limits.tasks_max = 9

    def test_swap_is_not_implied_by_memory_max(self):
        """Coupling the two is a 1d-3d policy decision, not a dataclass rule."""
        limits = CgroupLimits(memory_max_bytes=1024)
        self.assertIsNone(limits.memory_swap_max_bytes)
        self.assertEqual(limits.systemd_properties(), ["MemoryMax=1024"])

    def test_systemd_properties_are_exact_and_stably_ordered(self):
        limits = CgroupLimits(memory_max_bytes=268435456, memory_swap_max_bytes=0,
                              tasks_max=64, cpu_quota_percent=200)
        self.assertEqual(limits.systemd_properties(), [
            "MemoryMax=268435456",
            "MemorySwapMax=0",
            "TasksMax=64",
            "CPUQuota=200%",
        ])

    def test_required_controllers_track_only_the_active_fields(self):
        self.assertEqual(CgroupLimits(memory_max_bytes=1).required_controllers,
                         frozenset({"memory"}))
        self.assertEqual(CgroupLimits(memory_swap_max_bytes=0).required_controllers,
                         frozenset({"memory"}))
        self.assertEqual(CgroupLimits(tasks_max=1).required_controllers,
                         frozenset({"pids"}))
        self.assertEqual(CgroupLimits(cpu_quota_percent=1).required_controllers,
                         frozenset({"cpu"}))
        self.assertEqual(
            CgroupLimits(memory_max_bytes=1, tasks_max=1, cpu_quota_percent=1)
            .required_controllers,
            frozenset({"memory", "pids", "cpu"}),
        )

    def test_no_out_of_scope_resource_knobs_are_introduced(self):
        rendered = " ".join(
            CgroupLimits(memory_max_bytes=1, memory_swap_max_bytes=0, tasks_max=1,
                         cpu_quota_percent=1).systemd_properties()
        )
        for forbidden in ("IO", "io.max", "cpuset", "AllowedCPUs", "hugetlb",
                          "LimitNPROC", "LimitAS"):
            self.assertNotIn(forbidden, rendered)


class SystemdScopeRunnerTests(unittest.TestCase):
    """The scope runner knows systemd and nothing else."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runtime_dir = Path(self.temp.name) / "runtime"
        self.runtime_dir.mkdir()
        self.bus = self.runtime_dir / "bus"
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.bind(str(self.bus))
        self.addCleanup(self._socket.close)
        self.addCleanup(self.temp.cleanup)

    def runner(self, **kwargs):
        kwargs.setdefault("runtime_dir", self.runtime_dir)
        return SystemdScopeRunner(**kwargs)

    def available_runner(self, **kwargs):
        runner = self.runner(**kwargs)
        runner._delegated_controllers = lambda: frozenset({"cpu", "memory", "pids"})
        return runner

    # -- unit names ---------------------------------------------------------

    def test_unit_name_is_unique_and_uses_safe_characters_only(self):
        names = {SystemdScopeRunner.unit_name() for _ in range(200)}
        self.assertEqual(len(names), 200)
        for name in names:
            self.assertRegex(name, r"^edgem-tool-[0-9a-f]{32}\.scope$")

    def test_unit_name_never_embeds_caller_data(self):
        name = SystemdScopeRunner.unit_name()
        self.assertNotIn("/", name)
        self.assertNotIn(" ", name)

    # -- argv ---------------------------------------------------------------

    def test_inactive_limits_omit_the_wrapper_entirely(self):
        runner = self.available_runner()
        argv = ["/usr/bin/bwrap", "--die-with-parent", "/bin/true"]
        self.assertEqual(runner.wrap(argv, CgroupLimits(), unit="x.scope"), argv)

    def test_exact_systemd_run_argv(self):
        runner = self.available_runner()
        limits = CgroupLimits(memory_max_bytes=268435456, memory_swap_max_bytes=0,
                              tasks_max=64, cpu_quota_percent=200)
        unit = "edgem-tool-00000000000000000000000000000000.scope"
        self.assertEqual(
            runner.wrap(["/usr/bin/bwrap", "--die-with-parent", "/bin/true"],
                        limits, unit=unit),
            [
                "/usr/bin/systemd-run", "--user", "--scope", "--quiet", "--collect",
                f"--unit={unit}",
                "-p", "MemoryMax=268435456",
                "-p", "MemorySwapMax=0",
                "-p", "TasksMax=64",
                "-p", "CPUQuota=200%",
                "--",
                "/usr/bin/bwrap", "--die-with-parent", "/bin/true",
            ],
        )

    def test_partial_limits_emit_only_the_requested_properties(self):
        runner = self.available_runner()
        argv = runner.wrap(["/usr/bin/bwrap"], CgroupLimits(tasks_max=8), unit="u.scope")
        self.assertIn("-p", argv)
        self.assertIn("TasksMax=8", argv)
        rendered = " ".join(argv)
        self.assertNotIn("MemoryMax", rendered)
        self.assertNotIn("MemorySwapMax", rendered)
        self.assertNotIn("CPUQuota", rendered)

    def test_wrap_never_uses_a_shell(self):
        runner = self.available_runner()
        argv = runner.wrap(["/usr/bin/bwrap"], CgroupLimits(tasks_max=8), unit="u.scope")
        self.assertNotIn("/bin/sh", argv)
        self.assertNotIn("-c", argv)

    def test_empty_inner_argv_is_rejected(self):
        with self.assertRaises(ValueError):
            self.available_runner().wrap([], CgroupLimits(tasks_max=8), unit="u.scope")

    # -- supervisor environment --------------------------------------------

    def test_supervisor_env_is_minimal_and_carries_no_secret(self):
        env = self.available_runner().supervisor_env()
        self.assertEqual(set(env), {"XDG_RUNTIME_DIR"})
        self.assertEqual(env["XDG_RUNTIME_DIR"], str(self.runtime_dir))

    # -- availability -------------------------------------------------------

    def test_inactive_limits_need_no_systemd_at_all(self):
        runner = self.runner(systemd_run_binary="/definitely/missing/systemd-run",
                             systemctl_binary="/definitely/missing/systemctl")
        self.assertIsNone(runner.availability_for(CgroupLimits()))

    def test_missing_systemd_run_fails_closed(self):
        runner = self.available_runner(systemd_run_binary="/definitely/missing/systemd-run")
        result = runner.availability_for(CgroupLimits(tasks_max=8))
        self.assertIsNotNone(result)
        self.assertEqual(result.status, "failed")
        self.assertIn("systemd-run", result.summary)
        self.assertEqual(runner.availability, CgroupAvailability.SYSTEMD_RUN_ABSENT)

    def test_missing_systemctl_fails_closed(self):
        runner = self.available_runner(systemctl_binary="/definitely/missing/systemctl")
        result = runner.availability_for(CgroupLimits(tasks_max=8))
        self.assertIsNotNone(result)
        self.assertEqual(result.status, "failed")
        self.assertEqual(runner.availability, CgroupAvailability.SYSTEMCTL_ABSENT)

    def test_missing_user_bus_fails_closed(self):
        self.bus.unlink()
        runner = self.available_runner()
        result = runner.availability_for(CgroupLimits(tasks_max=8))
        self.assertIsNotNone(result)
        self.assertEqual(result.status, "failed")
        self.assertEqual(runner.availability, CgroupAvailability.USER_BUS_UNAVAILABLE)

    def test_bus_that_is_not_a_socket_fails_closed(self):
        self.bus.unlink()
        self.bus.write_text("not a socket")
        runner = self.available_runner()
        result = runner.availability_for(CgroupLimits(tasks_max=8))
        self.assertIsNotNone(result)
        self.assertEqual(runner.availability, CgroupAvailability.USER_BUS_UNAVAILABLE)

    def test_missing_controller_fails_closed(self):
        runner = self.runner()
        runner._delegated_controllers = lambda: frozenset({"memory", "pids"})
        result = runner.availability_for(CgroupLimits(cpu_quota_percent=50))
        self.assertIsNotNone(result)
        self.assertEqual(result.status, "failed")
        self.assertIn("cpu", result.summary)
        self.assertEqual(runner.availability, CgroupAvailability.CONTROLLERS_UNAVAILABLE)

    def test_only_the_controllers_actually_needed_are_required(self):
        runner = self.runner()
        runner._delegated_controllers = lambda: frozenset({"memory"})
        self.assertIsNone(runner.availability_for(CgroupLimits(memory_max_bytes=1024)))
        self.assertIsNotNone(runner.availability_for(CgroupLimits(tasks_max=8)))

    def test_available_state_is_recorded(self):
        runner = self.available_runner()
        self.assertIsNone(runner.availability_for(CgroupLimits(tasks_max=8)))
        self.assertEqual(runner.availability, CgroupAvailability.AVAILABLE)

    def test_linger_is_never_enabled_automatically(self):
        runner = self.available_runner()
        with patch("tool_runtime.subprocess.run") as run:
            runner.availability_for(CgroupLimits(tasks_max=8))
        for call in run.call_args_list:
            self.assertNotIn("loginctl", " ".join(call.args[0]))

    # -- termination --------------------------------------------------------

    def test_terminate_uses_systemctl_kill_with_a_bounded_timeout(self):
        runner = self.available_runner()
        with patch("tool_runtime.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "", "")
            runner.terminate("edgem-tool-abc.scope")
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], [
            "/usr/bin/systemctl", "--user", "kill", "--kill-whom=all",
            "--signal=KILL", "edgem-tool-abc.scope",
        ])
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertIsNotNone(run.call_args.kwargs["timeout"])

    def test_terminate_never_walks_pid_trees_or_writes_cgroup_kill(self):
        runner = self.available_runner()
        with patch("tool_runtime.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "", "")
            runner.terminate("edgem-tool-abc.scope")
        rendered = " ".join(run.call_args.args[0])
        for forbidden in ("pkill", "pgrep", "cgroup.kill", "cgroup.procs"):
            self.assertNotIn(forbidden, rendered)

    def test_terminate_is_contained_when_systemctl_fails_or_hangs(self):
        runner = self.available_runner()
        for effect in (subprocess.TimeoutExpired("systemctl", 5),
                       OSError("boom"),
                       subprocess.CompletedProcess([], 1, "", "no such unit")):
            with self.subTest(effect=type(effect).__name__):
                with patch("tool_runtime.subprocess.run") as run:
                    if isinstance(effect, subprocess.CompletedProcess):
                        run.return_value = effect
                    else:
                        run.side_effect = effect
                    self.assertIsInstance(runner.terminate("edgem-tool-abc.scope"), bool)


class CgroupIntegrationContractTests(unittest.TestCase):
    """How BubblewrapSandbox / CommandRunner consume the contract (mocked)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.workspace = Workspace.from_path(self.workspace_root)
        self.runtime_dir = Path(self.temp.name) / "runtime"
        self.runtime_dir.mkdir()
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.bind(str(self.runtime_dir / "bus"))
        self.addCleanup(self._socket.close)
        self.addCleanup(self.temp.cleanup)
        self.limits = CgroupLimits(memory_max_bytes=268435456, memory_swap_max_bytes=0,
                                   tasks_max=64, cpu_quota_percent=100)

    def scope_runner(self, **kwargs):
        runner = SystemdScopeRunner(runtime_dir=self.runtime_dir, **kwargs)
        runner._delegated_controllers = lambda: frozenset({"cpu", "memory", "pids"})
        return runner

    def sandbox(self, **kwargs):
        kwargs.setdefault("scope_runner", self.scope_runner())
        return tool_runtime.BubblewrapSandbox(**kwargs)

    # -- ownership ----------------------------------------------------------

    def test_command_runner_owns_the_active_production_contract(self):
        runner = CommandRunner()
        self.assertEqual(runner.cgroup_limits, DEFAULT_CGROUP_LIMITS)
        self.assertTrue(runner.cgroup_limits.active)

    def test_sandbox_without_an_explicit_contract_applies_none(self):
        """The sandbox is a mechanism; the contract belongs to CommandRunner.

        This is what keeps preflight and direct callers independent of a
        systemd user bus.
        """
        sandbox = self.sandbox()
        with patch.object(tool_runtime.subprocess, "Popen") as popen:
            popen.return_value.communicate.return_value = ("", "")
            popen.return_value.returncode = 0
            sandbox.run(self.workspace, ["/bin/true"])
        self.assertEqual(popen.call_args.args[0][0], "bwrap")
        self.assertEqual(popen.call_args.kwargs["env"], {})

    def test_the_runner_path_is_scoped_with_the_production_values(self):
        sandbox = self.sandbox()
        runner = CommandRunner(sandbox=sandbox)
        with patch.object(tool_runtime.subprocess, "Popen") as popen, \
             patch.object(sandbox, "ensure_available",
                          return_value=ToolResult("ok", "available")):
            popen.return_value.communicate.return_value = ("", "")
            popen.return_value.returncode = 0
            runner.run_sandboxed(self.workspace, ["/bin/true"])
        argv = popen.call_args.args[0]
        self.assertEqual(argv[0], "/usr/bin/systemd-run")
        for prop in ("MemoryMax=2147483648", "MemorySwapMax=0",
                     "TasksMax=256", "CPUQuota=800%"):
            self.assertIn(prop, argv)

    def test_command_runner_passes_its_cgroup_limits_to_the_sandbox(self):
        sandbox = MagicMock()
        sandbox.ensure_available.return_value = ToolResult("ok", "available")
        sandbox.run.return_value = ToolResult("ok", "done")
        runner = CommandRunner(sandbox=sandbox, cgroup_limits=self.limits)
        runner.run_sandboxed(self.workspace, ["/bin/true"])
        self.assertEqual(sandbox.run.call_args.kwargs["cgroup_limits"], self.limits)

    def test_execution_profile_is_untouched_by_cgroup_limits(self):
        profile = ExecutionProfile.from_capabilities({Capability.FILESYSTEM_READ})
        self.assertFalse(hasattr(profile, "cgroup_limits"))
        self.assertFalse(hasattr(profile, "memory_max_bytes"))

    # -- CLOSED route -------------------------------------------------------

    def test_closed_route_is_wrapped_when_limits_are_active(self):
        sandbox = self.sandbox()
        with patch.object(tool_runtime.subprocess, "Popen") as popen:
            popen.return_value.communicate.return_value = ("", "")
            popen.return_value.returncode = 0
            sandbox.run(self.workspace, ["/bin/true"], cgroup_limits=self.limits)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[0], "/usr/bin/systemd-run")
        self.assertIn("--scope", argv)
        self.assertIn("--collect", argv)
        separator = argv.index("--")
        self.assertEqual(argv[separator + 1], "bwrap")

    def test_closed_route_is_not_wrapped_when_limits_are_inactive(self):
        sandbox = self.sandbox()
        with patch.object(tool_runtime.subprocess, "Popen") as popen:
            popen.return_value.communicate.return_value = ("", "")
            popen.return_value.returncode = 0
            sandbox.run(self.workspace, ["/bin/true"], cgroup_limits=CgroupLimits())
        self.assertEqual(popen.call_args.args[0][0], "bwrap")

    def test_wrapper_order_is_systemd_run_then_bwrap_then_prlimit(self):
        sandbox = self.sandbox()
        with patch.object(tool_runtime.subprocess, "Popen") as popen:
            popen.return_value.communicate.return_value = ("", "")
            popen.return_value.returncode = 0
            sandbox.run(self.workspace, ["/bin/true"], cgroup_limits=self.limits,
                        resource_limits=DEFAULT_RESOURCE_LIMITS)
        argv = popen.call_args.args[0]
        self.assertLess(argv.index("/usr/bin/systemd-run"), argv.index("bwrap"))
        self.assertLess(argv.index("bwrap"), argv.index("/usr/bin/prlimit"))
        # prlimit must stay inside bubblewrap, never wrap systemd-run.
        self.assertLess(argv.index("--die-with-parent"), argv.index("/usr/bin/prlimit"))

    def test_supervisor_env_is_used_for_systemd_run_but_never_for_bwrap(self):
        sandbox = self.sandbox()
        with patch.object(tool_runtime.subprocess, "Popen") as popen:
            popen.return_value.communicate.return_value = ("", "")
            popen.return_value.returncode = 0
            sandbox.run(self.workspace, ["/bin/true"], cgroup_limits=self.limits)
        self.assertEqual(popen.call_args.kwargs["env"],
                         {"XDG_RUNTIME_DIR": str(self.runtime_dir)})
        # --clearenv remains the boundary for the sandboxed command itself.
        self.assertIn("--clearenv", popen.call_args.args[0])

        with patch.object(tool_runtime.subprocess, "Popen") as popen:
            popen.return_value.communicate.return_value = ("", "")
            popen.return_value.returncode = 0
            sandbox.run(self.workspace, ["/bin/true"], cgroup_limits=CgroupLimits())
        self.assertEqual(popen.call_args.kwargs["env"], {})

    # -- fail closed --------------------------------------------------------

    def test_active_limits_fail_closed_when_systemd_run_is_absent(self):
        sandbox = self.sandbox(
            scope_runner=self.scope_runner(systemd_run_binary="/definitely/missing"))
        with patch.object(tool_runtime.subprocess, "Popen") as popen, \
             patch.object(tool_runtime.subprocess, "run") as run:
            result = sandbox.run(self.workspace, ["/bin/true"], cgroup_limits=self.limits)
        self.assertEqual(result.status, "failed")
        self.assertIn("resource control", result.summary.lower())
        popen.assert_not_called()
        run.assert_not_called()

    def test_active_limits_fail_closed_when_user_bus_is_absent(self):
        (self.runtime_dir / "bus").unlink()
        sandbox = self.sandbox()
        with patch.object(tool_runtime.subprocess, "Popen") as popen:
            result = sandbox.run(self.workspace, ["/bin/true"], cgroup_limits=self.limits)
        self.assertEqual(result.status, "failed")
        popen.assert_not_called()

    def test_active_limits_fail_closed_when_a_controller_is_missing(self):
        runner = SystemdScopeRunner(runtime_dir=self.runtime_dir)
        runner._delegated_controllers = lambda: frozenset({"memory"})
        sandbox = self.sandbox(scope_runner=runner)
        with patch.object(tool_runtime.subprocess, "Popen") as popen:
            result = sandbox.run(self.workspace, ["/bin/true"], cgroup_limits=self.limits)
        self.assertEqual(result.status, "failed")
        popen.assert_not_called()

    def test_there_is_no_unlimited_fallback(self):
        """A failed resource-control setup must never degrade to plain bwrap."""
        sandbox = self.sandbox(
            scope_runner=self.scope_runner(systemd_run_binary="/definitely/missing"))
        with patch.object(tool_runtime.subprocess, "Popen") as popen, \
             patch.object(tool_runtime.subprocess, "run") as run:
            result = sandbox.run(self.workspace, ["/bin/true"], cgroup_limits=self.limits)
        self.assertFalse(result.ok)
        popen.assert_not_called()
        run.assert_not_called()
        self.assertNotIn("prlimit", result.summary)

    # -- timeout ------------------------------------------------------------

    def test_timeout_invokes_scope_tree_cleanup_for_the_unit_it_created(self):
        runner = self.scope_runner()
        sandbox = self.sandbox(scope_runner=runner, timeout_seconds=1)
        with patch.object(tool_runtime.subprocess, "Popen") as popen, \
             patch.object(runner, "terminate", return_value=True) as terminate:
            popen.return_value.communicate.side_effect = [
                subprocess.TimeoutExpired("bwrap", 1), ("", ""),
            ]
            popen.return_value.returncode = -9
            result = sandbox.run(self.workspace, ["/bin/sleep", "30"],
                                 cgroup_limits=self.limits)
        self.assertEqual(result.status, "timeout")
        terminate.assert_called_once()
        unit = terminate.call_args.args[0]
        self.assertRegex(unit, r"^edgem-tool-[0-9a-f]{32}\.scope$")
        # The very unit that was spawned, not a glob or a discovered one.
        self.assertIn(f"--unit={unit}", popen.call_args.args[0])

    def test_timeout_without_limits_does_not_touch_systemd(self):
        runner = self.scope_runner()
        sandbox = self.sandbox(scope_runner=runner, timeout_seconds=1)
        with patch.object(tool_runtime.subprocess, "Popen") as popen, \
             patch.object(runner, "terminate") as terminate:
            popen.return_value.communicate.side_effect = [
                subprocess.TimeoutExpired("bwrap", 1), ("", ""),
            ]
            popen.return_value.returncode = -9
            result = sandbox.run(self.workspace, ["/bin/sleep", "30"],
                                 cgroup_limits=CgroupLimits())
        self.assertEqual(result.status, "timeout")
        terminate.assert_not_called()

    # -- NETWORK route ------------------------------------------------------

    def test_network_route_wraps_bwrap_keeps_slirp_out_and_preserves_pass_fds(self):
        sandbox = self.sandbox()
        profile = ExecutionProfile.from_capabilities({Capability.NETWORK})
        spawned = []
        real_popen = subprocess.Popen

        def tracked(argv, *args, **kwargs):
            spawned.append((list(argv), kwargs.get("pass_fds"), kwargs.get("env")))
            return real_popen(argv, *args, **kwargs)

        with patch.object(tool_runtime.subprocess, "Popen", side_effect=tracked):
            sandbox.run(self.workspace, ["/bin/true"], profile=profile,
                        backend=NetworkBackend.SLIRP4NETNS, cgroup_limits=self.limits)

        # The helper's option probe also goes through Popen; select by role
        # rather than by spawn order.
        scoped = [s for s in spawned if s[0][0] == "/usr/bin/systemd-run"]
        helpers = [s for s in spawned if "slirp4netns" in s[0][0] and "--help" not in s[0]]
        self.assertEqual(len(scoped), 1)
        bwrap_argv, bwrap_fds, bwrap_env = scoped[0]
        self.assertIn("bwrap", bwrap_argv)
        self.assertEqual(bwrap_env, {"XDG_RUNTIME_DIR": str(self.runtime_dir)})
        # The bwrap FD protocol is untouched by the wrapper.
        self.assertIsNotNone(bwrap_fds)
        self.assertEqual(len(bwrap_fds), 3)
        for flag in ("--info-fd", "--block-fd", "--sync-fd"):
            self.assertIn(flag, bwrap_argv)
            self.assertIn(int(bwrap_argv[bwrap_argv.index(flag) + 1]), bwrap_fds)

        if helpers:
            slirp_argv, slirp_fds, slirp_env = helpers[0]
            self.assertNotIn("systemd-run", slirp_argv[0])
            self.assertIn("slirp4netns", slirp_argv[0])
            self.assertEqual(slirp_env, {})

    def test_network_setup_failure_never_releases_the_sync_fd(self):
        """Scope creation failure must leave the COMMAND permanently blocked."""
        sandbox = self.sandbox()
        profile = ExecutionProfile.from_capabilities({Capability.NETWORK})
        marker = self.workspace_root / "command-ran"
        writes = []
        real_write = os.write

        def tracked_write(fd, data):
            writes.append((fd, data))
            return real_write(fd, data)

        # A systemd-run that always refuses: bwrap therefore never starts.
        runner = self.scope_runner(systemd_run_binary="/bin/false")
        sandbox = self.sandbox(scope_runner=runner)
        with patch.object(tool_runtime.os, "write", side_effect=tracked_write):
            result = sandbox.run(
                self.workspace, ["/bin/sh", "-lc", "touch /workspace/command-ran"],
                profile=profile, backend=NetworkBackend.SLIRP4NETNS,
                cgroup_limits=self.limits)
        self.assertFalse(result.ok)
        self.assertFalse(marker.exists())
        self.assertEqual(writes, [], "no release write may follow a setup failure")

    def test_network_availability_failure_never_spawns_anything(self):
        (self.runtime_dir / "bus").unlink()
        sandbox = self.sandbox()
        profile = ExecutionProfile.from_capabilities({Capability.NETWORK})
        with patch.object(tool_runtime.subprocess, "Popen") as popen:
            result = sandbox.run(self.workspace, ["/bin/true"], profile=profile,
                                 backend=NetworkBackend.SLIRP4NETNS,
                                 cgroup_limits=self.limits)
        self.assertFalse(result.ok)
        popen.assert_not_called()


@unittest.skipUnless(os.environ.get("SPEAR_TEST_CGROUP") == "1",
                     "set SPEAR_TEST_CGROUP=1 for opt-in real cgroup integration")
class OptInRealCgroupTests(unittest.TestCase):
    """Real transient scopes.  Requires a systemd user bus with delegation."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.workspace = Workspace.from_path(self.workspace_root)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.assert_no_residual_units)

    def assert_no_residual_units(self):
        listed = subprocess.run(
            ["/usr/bin/systemctl", "--user", "list-units", "--all", "--no-legend",
             "edgem-tool-*"], capture_output=True, text=True, timeout=15)
        self.assertEqual(listed.stdout.strip(), "",
                         "residual edgem-tool-*.scope unit(s) left behind")

    def sandbox(self, **kwargs):
        return tool_runtime.BubblewrapSandbox(**kwargs)

    def observed_limits(self, limits):
        """Read the live cgroup files of the sandboxed process itself."""
        script = (
            "import pathlib\n"
            "rel = pathlib.Path('/proc/self/cgroup').read_text().strip().split(':')[-1]\n"
            "print(rel)\n"
        )
        result = self.sandbox().run(self.workspace, ["/usr/bin/python3", "-c", script],
                                    cgroup_limits=limits)
        self.assertTrue(result.ok, result.to_legacy_text())
        base = Path("/sys/fs/cgroup") / result.stdout.strip().lstrip("/")
        return base, result

    def test_A_to_D_cgroup_properties_are_actually_applied(self):
        limits = CgroupLimits(memory_max_bytes=134217728, memory_swap_max_bytes=0,
                              tasks_max=48, cpu_quota_percent=50)
        observed = {}
        script = (
            "import pathlib\n"
            "rel = pathlib.Path('/proc/self/cgroup').read_text().strip().split(':')[-1]\n"
            "print(rel)\n"
        )
        # The scope is gone once the command exits, so the values are read by a
        # helper that samples the live cgroup from the host side.
        sandbox = self.sandbox()
        unit_holder = {}
        real_popen = subprocess.Popen

        def sampling_popen(argv, *args, **kwargs):
            proc = real_popen(argv, *args, **kwargs)
            for flag in argv:
                if flag.startswith("--unit="):
                    unit_holder["unit"] = flag.split("=", 1)[1]
            base = Path("/sys/fs/cgroup/user.slice", f"user-{os.getuid()}.slice",
                        f"user@{os.getuid()}.service/app.slice",
                        unit_holder.get("unit", ""))
            for _ in range(300):
                try:
                    observed["memory.max"] = (base / "memory.max").read_text().strip()
                    observed["memory.swap.max"] = (base / "memory.swap.max").read_text().strip()
                    observed["pids.max"] = (base / "pids.max").read_text().strip()
                    observed["cpu.max"] = (base / "cpu.max").read_text().strip()
                    break
                except OSError:
                    time.sleep(0.02)
            return proc

        with patch.object(tool_runtime.subprocess, "Popen", side_effect=sampling_popen):
            result = sandbox.run(self.workspace, ["/bin/sleep", "2"], cgroup_limits=limits)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertEqual(observed.get("memory.max"), "134217728")       # A
        self.assertEqual(observed.get("memory.swap.max"), "0")          # B
        self.assertEqual(observed.get("pids.max"), "48")                # C
        self.assertEqual(observed.get("cpu.max"), "50000 100000")       # D

    def test_E_memory_limit_kills_the_command_without_swap_escape(self):
        limits = CgroupLimits(memory_max_bytes=67108864, memory_swap_max_bytes=0)
        alloc = ("b = []\n"
                 "for _ in range(512): b.append(bytearray(1024 * 1024))\n"
                 "print('NO KILL')\n")
        result = self.sandbox(timeout_seconds=120).run(
            self.workspace, ["/usr/bin/python3", "-c", alloc], cgroup_limits=limits)
        self.assertFalse(result.ok)
        self.assertNotIn("NO KILL", result.stdout)

    def test_F_tasks_max_is_enforced_and_the_supervisor_survives(self):
        limits = CgroupLimits(tasks_max=12)
        script = "i=0; while [ $i -lt 40 ]; do /bin/sleep 5 & i=$((i+1)); done; echo done"
        result = self.sandbox(timeout_seconds=60).run(
            self.workspace, ["/bin/sh", "-c", script], cgroup_limits=limits)
        self.assertFalse(result.ok)
        self.assertIn("fork", result.stderr.lower())
        self.assertTrue(Path(f"/proc/{os.getpid()}").exists())

    def test_G_timeout_cleans_the_whole_scope_tree(self):
        limits = CgroupLimits(tasks_max=64)
        sandbox = self.sandbox(timeout_seconds=1)
        result = sandbox.run(self.workspace,
                             ["/bin/sh", "-c", "/bin/sleep 30 & /bin/sleep 30"],
                             cgroup_limits=limits)
        self.assertEqual(result.status, "timeout")
        time.sleep(0.5)
        survivors = subprocess.run(["/usr/bin/pgrep", "-c", "-f", "^/bin/sleep 30$"],
                                   capture_output=True, text=True)
        self.assertIn(survivors.stdout.strip(), ("", "0"),
                      "sandbox descendants survived the scope cleanup")

    @staticmethod
    def _comm(pid):
        try:
            return Path(f"/proc/{pid}/comm").read_text().strip()
        except OSError:
            return ""

    @staticmethod
    def _own_children():
        """Direct children of this process, without scanning all of /proc."""
        children = []
        for task in Path("/proc/self/task").iterdir():
            try:
                children.extend((task / "children").read_text().split())
            except OSError:
                continue
        return children

    def _watch_scope(self, unit, observed, stop):
        """Sample the live scope and the slirp helper from a side thread.

        Nothing here runs on the supervisor's thread, and the sampling is kept
        cheap on purpose.  bwrap moves its child into a second, nested user
        namespace a few milliseconds after writing its info-fd JSON, and once
        that happens slirp4netns can no longer setns() into the sandbox netns.
        Any contention inside that window breaks the network path outright
        (measured: a deliberate 20 ms delay fails 40 runs out of 40).
        """
        base = Path("/sys/fs/cgroup/user.slice", f"user-{os.getuid()}.slice",
                    f"user@{os.getuid()}.service/app.slice", unit)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and not stop.is_set():
            try:
                members = (base / "cgroup.procs").read_text().split()
            except OSError:
                members = []
            if members:
                sample = [(pid, self._comm(pid)) for pid in members]
                # Keep the most complete sample: early on, only bwrap and its
                # namespace child are in the scope; the command joins after the
                # release write.
                if len(sample) > len(observed.get("scope_members", ())):
                    observed["scope_members"] = sample
            for pid in self._own_children():
                if self._comm(pid) != "slirp4netns":
                    continue
                try:
                    observed["slirp_cgroup"] = (
                        Path(f"/proc/{pid}/cgroup").read_text().strip().split(":")[-1]
                    )
                except OSError:
                    continue
            seen_command = any(
                comm not in ("bwrap", "") for _, comm in observed.get("scope_members", ())
            )
            if seen_command and "slirp_cgroup" in observed:
                return
            stop.wait(0.01)

    def test_H_network_membership_bwrap_in_scope_slirp_and_supervisor_out(self):
        limits = CgroupLimits(memory_max_bytes=536870912, tasks_max=64)
        profile = ExecutionProfile.from_capabilities({Capability.NETWORK})
        real_unit_name = SystemdScopeRunner.unit_name
        observed = {}
        result = None
        stop = threading.Event()
        self.addCleanup(stop.set)

        def naming():
            unit = real_unit_name()
            # Started before the spawn, i.e. outside the critical window.
            threading.Thread(target=self._watch_scope, args=(unit, observed, stop),
                             daemon=True).start()
            return unit

        # The pre-existing slirp readiness race (see _watch_scope) is not this
        # test's subject; retry a bounded number of times so a rare miss is not
        # reported as a cgroup-membership failure.
        for _ in range(5):
            observed.clear()
            stop.clear()
            with patch.object(SystemdScopeRunner, "unit_name", staticmethod(naming)):
                result = self.sandbox().run(self.workspace, ["/bin/sleep", "1"],
                                            profile=profile,
                                            backend=NetworkBackend.SLIRP4NETNS,
                                            cgroup_limits=limits)
            stop.set()
            if result.ok:
                break
        self.assertTrue(
            result.ok,
            "network path failed in all attempts (pre-existing slirp setns race?): "
            + result.to_legacy_text())

        members = observed.get("scope_members") or []
        self.assertTrue(members, "the transient scope was never observed alive")
        comms = [comm for _, comm in members]
        pids = [pid for pid, _ in members]
        # bwrap, its namespace child and the sandboxed command's tree: the
        # contract is exactly bwrap + COMMAND + descendants.
        self.assertIn("bwrap", comms, members)
        self.assertIn("sleep", comms, members)
        self.assertNotIn("slirp4netns", comms, members)
        self.assertNotIn(str(os.getpid()), pids, "the supervisor joined the scope")
        self.assertIn("slirp_cgroup", observed, "slirp4netns was never observed")
        self.assertNotIn("edgem-tool-", observed["slirp_cgroup"])
        supervisor = Path("/proc/self/cgroup").read_text().strip().split(":")[-1]
        self.assertNotIn("edgem-tool-", supervisor)

    def test_I_exit_seven_and_oom_leave_no_residual_unit(self):
        seven = self.sandbox().run(self.workspace, ["/bin/sh", "-c", "exit 7"],
                                   cgroup_limits=CgroupLimits(tasks_max=32))
        self.assertEqual(seven.exit_code, 7)
        alloc = "b = []\nfor _ in range(512): b.append(bytearray(1024 * 1024))\n"
        oom = self.sandbox(timeout_seconds=120).run(
            self.workspace, ["/usr/bin/python3", "-c", alloc],
            cgroup_limits=CgroupLimits(memory_max_bytes=67108864, memory_swap_max_bytes=0))
        self.assertFalse(oom.ok)
        # assert_no_residual_units runs in cleanup.


if __name__ == "__main__":
    unittest.main()


class PipefailShellTests(unittest.TestCase):
    """The shell must make pipeline failures visible without breaking dash."""

    def test_prelude_enables_pipefail_where_the_shell_supports_it(self):
        script = tool_runtime.PIPEFAIL_PRELUDE + "false | tail -1; echo EXIT=$?"
        completed = subprocess.run([tool_runtime.SHELL_BINARY, "-c", script],
                                   capture_output=True, text=True)
        expected = "EXIT=1" if tool_runtime.SHELL_BINARY == "/bin/bash" else "EXIT=0"
        self.assertIn(expected, completed.stdout)

    @unittest.skipUnless(Path("/bin/dash").is_file(), "dash is not installed")
    def test_prelude_does_not_kill_a_shell_without_pipefail(self):
        # `set` is a special builtin: a bare `set -o pipefail` makes a
        # non-interactive dash EXIT, which would break every shell command on
        # a Debian-family server where /bin/sh is dash. The subshell probe is
        # what prevents that, so it is pinned here.
        completed = subprocess.run(
            ["/bin/dash", "-c", tool_runtime.PIPEFAIL_PRELUDE + "echo alive"],
            capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("alive", completed.stdout)

    def test_shell_argv_carries_the_prelude_and_the_command(self):
        argv = tool_runtime.shell_argv("make | tail -3")
        self.assertEqual(argv[0], tool_runtime.SHELL_BINARY)
        self.assertEqual(argv[1], "-lc")
        self.assertTrue(argv[2].endswith("make | tail -3"))
        self.assertIn("pipefail", argv[2])
        self.assertEqual(tool_runtime.shell_argv("x", login=False)[1], "-c")


class SedCommandPolicyTests(unittest.TestCase):
    """sed reads like cat but can also write, so it is allowlisted narrowly."""

    def setUp(self):
        self.policy = CommandPolicy()

    def test_reading_forms_are_read_only(self):
        for command in ("sed -n '85,130p' os_desktop.cmake",
                        "sed -n '/BEGIN/,/END/p' f.txt",
                        "sed -e /pattern/d f.txt",
                        "sed 's/a/b/' f.txt"):
            with self.subTest(command=command):
                self.assertEqual(self.policy.classify(command).classification,
                                 CommandClassification.READ_ONLY)

    def test_a_sed_script_is_not_treated_as_a_path(self):
        """`/BEGIN/,/END/p` starts with a slash and names no file.

        Checking it as a path refused ordinary sed with "path argument may
        escape the workspace" -- a reason the model cannot act on, because the
        premise is false.
        """
        assessment = self.policy.classify("sed -n '/BEGIN/,/END/p' f.txt")
        self.assertNotEqual(assessment.reason, "path argument may escape the workspace")

    def test_in_place_editing_is_refused_in_every_spelling(self):
        for command in ("sed -i s/a/b/ f.txt", "sed -i.bak s/a/b/ f.txt",
                        "sed --in-place s/a/b/ f.txt",
                        "sed --in-place=.bak s/a/b/ f.txt",
                        "sed -ni p f.txt", "sed -n -i p f.txt"):
            with self.subTest(command=command):
                assessment = self.policy.classify(command)
                self.assertEqual(assessment.classification,
                                 CommandClassification.DANGEROUS)
                self.assertIn("edit_file", assessment.reason)

    def test_file_operands_are_still_checked_for_escapes(self):
        # Only the script is exempt from the path check, not the operands --
        # and -f takes a script FILE, which is a real path. An absolute operand
        # outside the roots is now a read grant; a RELATIVE `..` still is not,
        # because the classifier cannot know which cwd it resolves against.
        assessment = self.policy.classify("sed -n p ../outside/secret.txt")
        self.assertEqual(assessment.classification, CommandClassification.DANGEROUS)
        self.assertIn("relative path leaves the workspace", assessment.reason)

        for command in ("sed -n p /etc/passwd", "sed -f /etc/evil.sed f.txt"):
            with self.subTest(command=command):
                assessment = self.policy.classify(command)
                self.assertEqual(assessment.classification,
                                 CommandClassification.READ_ONLY)

    def test_other_binaries_keep_checking_every_argument(self):
        assessment = self.policy.classify("cat /etc/passwd")
        self.assertEqual(assessment.classification, CommandClassification.READ_ONLY)
        self.assertIn(Capability.HOST_READ, assessment.required_capabilities)


class ShowDiffTests(unittest.TestCase):
    """In auto mode the diff is the only review the user gets."""

    @classmethod
    def setUpClass(cls):
        import rag_chat
        cls.rag_chat = rag_chat

    def render(self, old, new):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.rag_chat.show_diff(old, new)
        return re.sub(r"\x1b\[[0-9;]*m", "", buffer.getvalue())

    INCLUDES = ("#include <stdio.h>\n#include <dirent.h>\n#include <stdlib.h>\n"
                "#include <string.h>\n#include <unistd.h>\n#include <sys/stat.h>")

    def test_an_edit_below_the_unchanged_head_is_visible(self):
        """The failure this replaced: six identical lines, then the change.

        Showing the head of each side rendered both as the same six #includes,
        so an added include was invisible and the user read "OK: updated" under
        a diff showing nothing.
        """
        out = self.render(self.INCLUDES,
                          self.INCLUDES + '\n#include "libxml2/triostr.h"')
        self.assertIn('+ #include "libxml2/triostr.h"', out)
        self.assertNotIn("- #include <stdio.h>", out)

    def test_a_no_op_edit_says_so_instead_of_drawing_a_change(self):
        out = self.render(self.INCLUDES, self.INCLUDES)
        self.assertIn("no change", out)
        self.assertNotIn("+ #include", out)

    def test_a_modified_line_shows_both_sides(self):
        out = self.render("a\nb\nc", "a\nB\nc")
        self.assertIn("- b", out)
        self.assertIn("+ B", out)
        self.assertNotIn("a", out.replace("no change", ""))

    def test_a_large_change_is_truncated_and_says_how_much(self):
        out = self.render("\n".join(f"old{i}" for i in range(20)),
                          "\n".join(f"new{i}" for i in range(20)))
        self.assertIn("more changed lines", out)


class FeedbackLoopPolicyTests(unittest.TestCase):
    """A model that cannot run its code cannot check it."""

    def setUp(self):
        self.policy = CommandPolicy()

    def test_a_compiler_is_workspace_mutating_like_make(self):
        for command in ("gcc -o /tmp/t so3/usr/src/ls.c", "cc -o /tmp/t x.c",
                        "g++ -o /tmp/t x.cc"):
            with self.subTest(command=command):
                self.assertEqual(self.policy.classify(command).classification,
                                 CommandClassification.WORKSPACE_MUTATING)

    def test_a_program_built_in_the_sandbox_may_be_run(self):
        """Refusing it denied the only way to TEST a change.

        `make` is already allowed and a Makefile runs anything, so the binary
        the model just compiled is no wider an exposure -- same sandbox, same
        confinement, no network.
        """
        for command in ("/tmp/t 'ts*'", "./t 'ts*'"):
            with self.subTest(command=command):
                self.assertEqual(self.policy.classify(command).classification,
                                 CommandClassification.WORKSPACE_MUTATING)

    def test_a_project_entry_point_may_live_in_a_subdirectory(self):
        """`./scripts/build.sh` is the command the model is meant to run.

        Relative paths with a subdirectory used to be refused outright, which
        refused every Infrabase front-end script. `..` is still caught, and the
        confinement is the sandbox: a program reachable by a relative path
        inside it is inside a mounted tree by construction.
        """
        assessment = self.policy.classify("./scripts/build.sh usr-so3")
        self.assertEqual(assessment.classification,
                         CommandClassification.WORKSPACE_MUTATING)
        self.assertIn(Capability.WORKSPACE_WRITE, assessment.required_capabilities)

    def test_programs_outside_the_sandbox_are_still_refused(self):
        for command in ("/bin/sh -c evil", "../outside/prog", "/etc/passwd",
                        "./../a/prog", "/usr/bin/env sh"):
            with self.subTest(command=command):
                assessment = self.policy.classify(command)
                self.assertEqual(assessment.classification,
                                 CommandClassification.DANGEROUS, command)

    def test_running_a_binary_still_needs_workspace_write(self):
        # SAFE mode grants no workspace:write, so the loop stays unavailable
        # there rather than being classified read-only by accident.
        assessment = self.policy.classify("/tmp/t 'ts*'")
        self.assertIn(Capability.WORKSPACE_WRITE, assessment.required_capabilities)


class TurnEvidenceTests(unittest.TestCase):
    """The conclusion is grounded in the tool log, not in intentions."""

    @classmethod
    def setUpClass(cls):
        import rag_chat
        cls.rag_chat = rag_chat

    def test_no_tool_activity_adds_nothing(self):
        self.assertEqual(self.rag_chat.turn_evidence([]), "")

    def test_a_turn_that_changed_no_file_says_so(self):
        """The fabricated-repair case: a claimed fix, an untouched file."""
        note = self.rag_chat.turn_evidence(['bash {"command": "cat x"}\nsome output'])
        self.assertIn("files changed: none", note)

    def test_failed_commands_are_counted(self):
        note = self.rag_chat.turn_evidence([
            'bash {"command": "make"}\nbuilt',
            'bash {"command": "make bad"}\nError 2\n(exit 2)',
        ])
        self.assertIn("2 command(s) run", note)
        self.assertIn("1 of them exited non-zero", note)

    def test_applied_edits_are_named_and_rejected_ones_are_not(self):
        note = self.rag_chat.turn_evidence([
            'edit_file {"path": "a.c"}\nOK: a.c updated',
            'edit_file {"path": "b.c"}\nERROR: file not found: b.c',
        ])
        self.assertIn("a.c", note)
        self.assertNotIn("b.c", note)


class UnverifiedChangeTests(unittest.TestCase):
    """A change nobody ran is a claim, not a result."""

    @classmethod
    def setUpClass(cls):
        import rag_chat
        cls.rag_chat = rag_chat

    EDIT = 'edit_file {"path": "a.c"}\nOK: a.c updated'
    RUN = 'bash {"command": "make"}\nbuilt'

    def test_running_after_the_change_clears_it(self):
        self.assertFalse(self.rag_chat.unverified_change([self.EDIT, self.RUN]))

    def test_running_BEFORE_the_change_does_not(self):
        """Order is the whole point: a build before the edit tested the edit
        that was not yet made."""
        self.assertTrue(self.rag_chat.unverified_change([self.RUN, self.EDIT]))

    def test_a_second_change_reopens_it(self):
        self.assertTrue(self.rag_chat.unverified_change(
            [self.EDIT, self.RUN, self.EDIT]))

    def test_a_question_answered_from_reads_is_not_caught(self):
        # Nothing was changed, so there is nothing to verify and the gate must
        # stay out of the way of informational turns.
        self.assertFalse(self.rag_chat.unverified_change(
            ['bash {"command": "cat a.c"}\nsome text']))

    def test_a_rejected_edit_is_not_a_change(self):
        self.assertFalse(self.rag_chat.unverified_change(
            ['edit_file {"path": "a.c"}\nERROR: file not found: a.c']))

    def test_the_demand_names_the_files_and_asks_for_failing_cases(self):
        demand = self.rag_chat.verify_demand([self.EDIT])
        self.assertIn("a.c", demand)
        self.assertIn("could reasonably fail", demand)


class TrajectoryRecordingTests(unittest.TestCase):
    """Rated trajectories are the training data a fine-tune would need."""

    @classmethod
    def setUpClass(cls):
        import rag_chat
        cls.rag_chat = rag_chat

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "trajectories.jsonl"
        self._saved = self.rag_chat.TRAJECTORY_FILE
        self.rag_chat.TRAJECTORY_FILE = str(self.path)

    def tearDown(self):
        self.rag_chat.TRAJECTORY_FILE = self._saved
        self.temp.cleanup()

    STEPS = [{"tool": "bash", "arguments": {"command": "make"}, "result": "built"}]

    def test_a_trajectory_keeps_the_tool_steps_not_just_the_prose(self):
        """A question/answer pair cannot teach behaviour.

        What needs training is running the change and judging the output, and
        neither is visible in the final prose.
        """
        self.rag_chat.save_trajectory("q", self.STEPS, "a", "pass", "bench")
        sample = json.loads(self.path.read_text().strip())
        self.assertEqual(sample["steps"], self.STEPS)
        self.assertEqual(sample["verdict"], "pass")
        self.assertEqual(sample["source"], "bench")

    def test_failures_are_recorded_too_and_labelled(self):
        # A dataset of successes alone cannot teach what to stop doing, and
        # filtering later is free while re-running a session is not.
        self.rag_chat.save_trajectory("q", self.STEPS, "a", "fail", "bench")
        self.assertEqual(json.loads(self.path.read_text())["verdict"], "fail")

    def test_samples_accumulate_one_per_line(self):
        self.rag_chat.save_trajectory("q1", self.STEPS, "a", "pass", "bench")
        count = self.rag_chat.save_trajectory("q2", self.STEPS, "a", "fail", "bench")
        self.assertEqual(count, 2)
        self.assertEqual(len(self.path.read_text().strip().split("\n")), 2)

    def test_no_declared_bench_gives_no_verdict_rather_than_a_pass(self):
        with patch.object(self.rag_chat, "project_bench", return_value=None):
            self.assertIsNone(self.rag_chat.run_project_bench())

    def test_a_turn_no_bench_judged_is_kept_as_unrated_not_dropped(self):
        """Recording is wider than judging, on purpose.

        Gating the record on a verdict meant a project without a bench
        recorded nothing: one project declares one, so the dataset held a
        single trajectory. "unrated" says the verdict is missing, which a
        trainer can filter on; dropping the turn says nothing happened.
        """
        self.rag_chat.save_trajectory("q", self.STEPS, "a", "unrated", "answer")
        sample = json.loads(self.path.read_text().strip())
        self.assertEqual((sample["verdict"], sample["source"]),
                         ("unrated", "answer"))
        self.assertEqual(sample["steps"], self.STEPS)

    def test_every_recorded_source_stays_distinguishable(self):
        for verdict, source in (("pass", "bench"), ("unrated", "change"),
                                ("unrated", "answer"), ("pass", "user")):
            self.rag_chat.save_trajectory("q", self.STEPS, "a", verdict, source)
        rows = [json.loads(l) for l in self.path.read_text().strip().split("\n")]
        self.assertEqual([(r["verdict"], r["source"]) for r in rows],
                         [("pass", "bench"), ("unrated", "change"),
                          ("unrated", "answer"), ("pass", "user")])


class BenchLocationTests(unittest.TestCase):
    """The bench belongs to the harness, not to the tree it judges."""

    @classmethod
    def setUpClass(cls):
        import rag_chat
        cls.rag_chat = rag_chat

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self._saved = self.rag_chat.BENCH_DIR
        self.rag_chat.BENCH_DIR = self.temp.name

    def tearDown(self):
        self.rag_chat.BENCH_DIR = self._saved
        self.temp.cleanup()

    def test_a_bare_name_resolves_inside_the_harness(self):
        script = Path(self.temp.name) / "proj.sh"
        script.write_text("#!/bin/sh\nexit 0\n")
        with patch.object(self.rag_chat, "project_bench", return_value="proj.sh"):
            self.assertEqual(self.rag_chat.bench_command(), str(script))

    def test_an_unknown_name_is_passed_through_as_a_command(self):
        # Escape hatch: a project that wants to own its bench still can.
        with patch.object(self.rag_chat, "project_bench",
                          return_value="./scripts/mine.sh"):
            self.assertEqual(self.rag_chat.bench_command(), "./scripts/mine.sh")

    def test_no_declared_bench_resolves_to_nothing(self):
        with patch.object(self.rag_chat, "project_bench", return_value=None):
            self.assertIsNone(self.rag_chat.bench_command())


class SessionTmpdirTests(unittest.TestCase):
    """One /tmp per session, so a check can span two commands."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "workspace"
        self.root.mkdir()
        self.workspace = Workspace.from_path(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_a_file_written_in_tmp_survives_to_the_next_command(self):
        """The reason for the change: compile in one call, run in the next.

        With a per-command tmpfs the binary was gone before it could be run,
        forcing every verification into one unwieldy command line.
        """
        sandbox = tool_runtime.BubblewrapSandbox()
        first = sandbox.run(self.workspace,
                            tool_runtime.shell_argv("echo kept > /tmp/marker"))
        self.assertTrue(first.ok, first.to_legacy_text())
        second = sandbox.run(self.workspace,
                             tool_runtime.shell_argv("cat /tmp/marker"))
        self.assertIn("kept", second.stdout)

    def test_two_sessions_do_not_share_their_tmp(self):
        tool_runtime.BubblewrapSandbox().run(
            self.workspace, tool_runtime.shell_argv("echo x > /tmp/leak"))
        other = tool_runtime.BubblewrapSandbox().run(
            self.workspace,
            tool_runtime.shell_argv("test -e /tmp/leak && echo SHARED || echo private"))
        self.assertIn("private", other.stdout)

    def test_the_host_tmp_is_never_written_to_directly(self):
        sandbox = tool_runtime.BubblewrapSandbox()
        sandbox.run(self.workspace,
                    tool_runtime.shell_argv("touch /tmp/spear-host-leak-marker"))
        self.assertFalse(Path("/tmp/spear-host-leak-marker").exists())

    @patch.dict(os.environ, {"SPEAR_SANDBOX_EPHEMERAL_TMP": "1"})
    def test_the_per_command_tmpfs_can_be_restored(self):
        argv = tool_runtime.BubblewrapSandbox().build_argv(
            self.workspace, ["/bin/true"])
        index = argv.index("--tmpfs")
        self.assertIn("/tmp", [argv[i + 1] for i, t in enumerate(argv)
                               if t == "--tmpfs"])
        self.assertIsNotNone(index)


class OutsideReadPolicyTests(unittest.TestCase):
    """Reading a tree no declared root contains.

    The refusal this replaces ended a real session: asked to look at the notes
    in /opt/llm/claude, the assistant was told "path argument may escape the
    workspace", re-ran the same command, then handed the question back to the
    user. Looking at a file is not changing it, and the two are now separate
    grants.
    """

    def setUp(self):
        # NOT under /tmp: the sandbox's own tmpfs lives there, so a path in it
        # already counts as inside and would not exercise an outside read.
        self.temp = tempfile.TemporaryDirectory(dir=Path.home())
        self.root = Path(self.temp.name) / "workspace"
        self.root.mkdir()
        self.outside = Path(self.temp.name) / "notes"
        self.outside.mkdir()
        (self.outside / "memory.md").write_text("remembered\n")
        self.workspace = Workspace.from_path(self.root)
        self.policy = CommandPolicy()
        self.policy.bind_workspace(self.workspace)

    def tearDown(self):
        self.temp.cleanup()

    def test_a_read_only_command_may_name_an_outside_tree(self):
        assessment = self.policy.classify(f"ls -la {self.outside}")
        self.assertEqual(assessment.classification, CommandClassification.READ_ONLY)
        self.assertIn(Capability.HOST_READ, assessment.required_capabilities)
        self.assertEqual(assessment.host_read_paths, (str(self.outside),))

    def test_the_grant_survives_a_pipeline(self):
        # The shell branch used to check nothing at all: the command ran with
        # the tree unmounted and reported "No such file or directory" about a
        # file that plainly exists -- worse than either allowing or refusing.
        assessment = self.policy.classify(
            f"cat {self.outside}/memory.md | head -3")
        self.assertEqual(assessment.classification, CommandClassification.SHELL_COMPLEX)
        self.assertIn(Capability.HOST_READ, assessment.required_capabilities)
        self.assertIn(str(self.outside / "memory.md"), assessment.host_read_paths)

    def test_an_input_redirection_reads_and_an_output_redirection_does_not(self):
        readable = self.policy.classify(f"cat < {self.outside}/memory.md")
        self.assertNotEqual(readable.classification, CommandClassification.DANGEROUS)
        self.assertIn(str(self.outside / "memory.md"), readable.host_read_paths)

        writable = self.policy.classify(f"echo x > {self.outside}/memory.md")
        self.assertEqual(writable.classification, CommandClassification.DANGEROUS)

    def test_a_missing_outside_path_binds_nothing_and_is_not_refused(self):
        # ENOENT from the command is the truthful answer and the one the model
        # can act on; a policy error about a file that was never there is not.
        assessment = self.policy.classify(f"cat {self.outside}/absent.md")
        self.assertEqual(assessment.classification, CommandClassification.READ_ONLY)
        self.assertEqual(assessment.host_read_paths, ())

    def test_credential_stores_stay_refused(self):
        for path in ("/home/someone/.ssh/id_ed25519", "/home/someone/.aws/credentials",
                     "/etc/shadow", "/home/someone/.netrc"):
            with self.subTest(path=path):
                assessment = self.policy.classify(f"cat {path}")
                self.assertEqual(assessment.classification,
                                 CommandClassification.DANGEROUS)
                self.assertIn("credentials", assessment.reason)

    def test_a_sandbox_mount_point_cannot_be_bound_over(self):
        # Binding /home would land on top of the sandbox's own $HOME.
        for path in ("/", "/home", "/usr", "/tmp"):
            with self.subTest(path=path):
                assessment = self.policy.classify(f"ls {path}")
                self.assertEqual(assessment.classification,
                                 CommandClassification.DANGEROUS)
                self.assertIn("subdirectory", assessment.reason)

    def test_a_glob_mounts_the_directory_the_shell_will_expand_in(self):
        # The pattern names nothing on the host; the directory holding the
        # matches is what has to be there when the sandbox's shell expands it.
        assessment = self.policy.classify(f"grep -l x {self.outside}/*.md")
        self.assertEqual(assessment.host_read_paths, (str(self.outside),))

    def test_a_symlinked_tree_is_bound_where_the_command_spells_it(self):
        # /opt/llm/claude is a symlink; binding only its target answers ENOENT
        # for the exact path the user named.
        link = Path(self.temp.name) / "link-to-notes"
        link.symlink_to(self.outside, target_is_directory=True)
        assessment = self.policy.classify(f"ls -la {link}/")
        self.assertEqual(assessment.host_read_paths, (str(link),))

    def test_a_symlink_is_not_a_way_round_the_checks(self):
        link = Path(self.temp.name) / "innocent"
        link.symlink_to("/etc/shadow")
        assessment = self.policy.classify(f"cat {link}")
        self.assertEqual(assessment.classification, CommandClassification.DANGEROUS)
        self.assertIn("credentials", assessment.reason)

    def test_a_relative_traversal_is_refused_and_says_what_to_do(self):
        assessment = self.policy.classify("cat ../notes/memory.md")
        self.assertEqual(assessment.classification, CommandClassification.DANGEROUS)
        self.assertIn("absolute path", assessment.reason)
        self.assertIn(str(self.root), assessment.reason)

    def test_every_mode_grants_the_read_and_ask_confirms_it(self):
        assessment = self.policy.classify(f"ls {self.outside}")
        for mode in (ExecutionMode.SAFE, ExecutionMode.AUTO):
            with self.subTest(mode=mode):
                self.assertTrue(self.policy.authorize(assessment, mode).allowed)
        refused = self.policy.authorize(assessment, ExecutionMode.ASK, lambda _: False)
        self.assertFalse(refused.allowed)
        self.assertTrue(
            self.policy.authorize(assessment, ExecutionMode.ASK, lambda _: True).allowed)

    def test_a_refusal_names_the_trees_that_are_reachable(self):
        reason = self.policy.classify("cat /etc/shadow").reason
        self.assertIn(str(self.root), reason)
        self.assertIn("READ-ONLY", reason)


class OutsideReadSandboxTests(unittest.TestCase):
    """The grant is a read-only mount, so the write boundary never moves."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.home())
        self.root = Path(self.temp.name) / "workspace"
        self.root.mkdir()
        self.outside = Path(self.temp.name) / "notes"
        self.outside.mkdir()
        (self.outside / "memory.md").write_text("remembered\n")
        self.workspace = Workspace.from_path(self.root)
        self.sandbox = tool_runtime.BubblewrapSandbox()
        self.policy = CommandPolicy()
        self.policy.bind_workspace(self.workspace)

    def tearDown(self):
        self.temp.cleanup()

    def profile_for(self, command):
        assessment = self.policy.classify(command)
        outcome = self.policy.authorize(assessment, ExecutionMode.AUTO)
        self.assertTrue(outcome.allowed, assessment.reason)
        return assessment, ExecutionProfile.from_capabilities(
            outcome.granted_capabilities, assessment.host_read_paths)

    def test_the_outside_tree_is_visible_and_read_only(self):
        command = f"cat {self.outside}/memory.md"
        _, profile = self.profile_for(command)
        result = self.sandbox.run(
            self.workspace, tool_runtime.shell_argv(command), profile=profile)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertIn("remembered", result.stdout)

        overwrite = self.sandbox.run(
            self.workspace,
            tool_runtime.shell_argv(f"printf pwned > {self.outside}/memory.md"),
            profile=profile)
        self.assertFalse(overwrite.ok)
        self.assertEqual((self.outside / "memory.md").read_text(), "remembered\n")

    def test_a_tree_that_was_never_named_stays_invisible(self):
        other = Path(self.temp.name) / "unnamed"
        other.mkdir()
        (other / "secret.txt").write_text("secret\n")
        _, profile = self.profile_for(f"cat {self.outside}/memory.md")
        result = self.sandbox.run(
            self.workspace,
            tool_runtime.shell_argv(f"test ! -e {other}/secret.txt"),
            profile=profile)
        self.assertTrue(result.ok, "an unnamed tree must not be exposed")

    def test_the_workspace_stays_writable_under_an_outside_read(self):
        # A build that also reads a reference tree: the outside bind must not
        # cost the workspace the write it was granted.
        profile = ExecutionProfile.from_capabilities(
            {Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE,
             Capability.SHELL_COMPLEX, Capability.HOST_READ},
            (str(self.outside),))
        mount = self.sandbox.mount_root(self.workspace, profile)
        result = self.sandbox.run(
            self.workspace,
            tool_runtime.shell_argv(f"printf ok > {mount}/written.txt"),
            profile=profile)
        self.assertTrue(result.ok, result.to_legacy_text())
        self.assertEqual((self.root / "written.txt").read_text(), "ok")


class SandboxDownStopsBlindEditsTests(unittest.TestCase):
    """A sandbox that is unavailable does not come back mid-turn.

    Observed in a container where bwrap could not mount /proc: eight commands,
    eight "sandbox unavailable" refusals, then nine edits to a source file made
    from retrieved context alone -- the model could not read the file it was
    changing, let alone compile it. The result was plausible-looking nonsense
    (SOL_SOCKET redefined as 0xffff, Linux headers added to an SO3 userspace
    program) that the user then had to revert.
    """

    @classmethod
    def setUpClass(cls):
        import rag_chat
        cls.rag_chat = rag_chat

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "f.c").write_text("int main(void) { return 0; }\n")
        self._ws = self.rag_chat.WORKSPACE
        self.rag_chat.WORKSPACE = Workspace.from_path(self.root)
        self._mode = self.rag_chat.EXECUTION_MODE
        self.rag_chat.EXECUTION_MODE = ExecutionMode.AUTO

    def tearDown(self):
        self.rag_chat.WORKSPACE = self._ws
        self.rag_chat.EXECUTION_MODE = self._mode
        self.temp.cleanup()

    def test_an_unavailable_sandbox_is_remembered_for_the_turn(self):
        # Patched at run_cmd_result and not at run_cmd: the router calls the
        # structured boundary directly, and run_cmd is now only the renderer
        # around it. The claim under test is unchanged -- a command that
        # reports the sandbox missing makes the turn remember it.
        cache = {}
        down = tool_runtime.ToolResult(
            status="failed", summary="bubblewrap sandbox unavailable",
            stderr="bubblewrap sandbox unavailable", exit_code=1)
        with patch.object(self.rag_chat, "run_cmd_result", return_value=down):
            self.rag_chat.execute_tool("bash", {"command": "ls"}, cache)
        self.assertTrue(cache.get(self.rag_chat.SANDBOX_DOWN))

    def test_edits_are_refused_once_it_is_down(self):
        cache = {self.rag_chat.SANDBOX_DOWN: True}
        before = (self.root / "f.c").read_text()
        for tool, args in (
            ("edit_file", {"path": "f.c", "old_text": "return 0",
                           "new_text": "return 1"}),
            ("write_file", {"path": "f.c", "content": "wiped"}),
            ("append_file", {"path": "f.c", "content": "// more"}),
        ):
            with self.subTest(tool=tool):
                result = self.rag_chat.execute_tool(tool, args, cache)
                self.assertIn("REFUSED", result)
                # The refusal is the point: the file must be untouched.
                self.assertEqual(before, (self.root / "f.c").read_text())

    def test_a_working_sandbox_leaves_edits_alone(self):
        cache = {}
        result = self.rag_chat.execute_tool(
            "edit_file", {"path": "f.c", "old_text": "return 0",
                          "new_text": "return 1"}, cache)
        self.assertNotIn("REFUSED", result)
        self.assertIn("return 1", (self.root / "f.c").read_text())


class LearnedRulesAreGlobalTests(unittest.TestCase):
    """`/recall` writes where every corpus will see it.

    `/remember` lands in memories-<corpus>.md, so a rule that holds everywhere
    -- "an existing copyright header is never rewritten" -- was invisible in
    every other tree. This is the global counterpart, and it lives under
    STATE_DIR rather than in rules.d/ because rules.d/ ships inside the image.
    """

    @classmethod
    def setUpClass(cls):
        import rag_chat
        cls.rag_chat = rag_chat

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self._file = self.rag_chat.LEARNED_RULES_FILE
        self.rag_chat.LEARNED_RULES_FILE = str(
            Path(self.temp.name) / "rules-learned.md")

        # A shipped rule of this test's own. rules.d/ ships empty -- a rule
        # describes one organisation's code, so none belongs to the platform
        # -- and the ordering below needs something to be ordered after.

        self._rules_dir = self.rag_chat.RULES_DIR
        shipped = Path(self.temp.name) / "rules.d"
        shipped.mkdir()
        (shipped / "20-conventions.md").write_text(
            "## Conventions\n\nLeave a blank line after a comment block.\n",
            encoding="utf-8")
        self.rag_chat.RULES_DIR = str(shipped)

    def tearDown(self):
        self.rag_chat.LEARNED_RULES_FILE = self._file
        self.rag_chat.RULES_DIR = self._rules_dir
        self.temp.cleanup()

    def test_a_recalled_rule_is_injected_after_the_shipped_ones(self):
        self.rag_chat.save_learned_rule("never rewrite an existing header")
        self.rag_chat.save_learned_rule("- prefer build.sh over make")
        body = Path(self.rag_chat.LEARNED_RULES_FILE).read_text()
        self.assertEqual(len(body.strip().splitlines()), 2)
        # The leading dash the user typed is not doubled.
        self.assertNotIn("- - ", body)

        rules = self.rag_chat.load_rules()
        self.assertIn("## Rule: learned", rules)
        self.assertIn("never rewrite an existing header", rules)
        # After the shipped ones, so a later line can qualify an earlier one.
        self.assertGreater(rules.index("## Rule: learned"),
                           rules.index("## Rule: conventions"))


class RepeatedCorpusSearchIsCachedTests(unittest.TestCase):
    """The corpus does not change mid-turn, so neither does the answer.

    Observed in a real session: the model asked the corpus the same question
    twice in a row and got the same file back, spending a round on it. bash has
    had a per-turn cache for exactly this; the retrieval tools did not.
    """

    @classmethod
    def setUpClass(cls):
        import rag_chat
        cls.rag_chat = rag_chat

    def _context(self, cache):
        return SimpleNamespace(cache=cache, action_id="a1", role="main",
                               cancellation=None)

    def test_the_second_identical_search_is_served_from_the_turn_cache(self):
        cache = {}
        calls = []

        def fake_search(query):
            calls.append(query)
            return f"# File: so3/usr/src/more.c\n{query}"

        with patch.object(self.rag_chat, "search_corpus", fake_search):
            first = self.rag_chat._registered_search_corpus(
                self._context(cache), {"query": "how does more read stdin"})
            second = self.rag_chat._registered_search_corpus(
                self._context(cache), {"query": "how does more read stdin"})
            other = self.rag_chat._registered_search_corpus(
                self._context(cache), {"query": "something else"})

        self.assertEqual(calls, ["how does more read stdin", "something else"],
                         "the repeated query must not reach the corpus twice")
        self.assertIn("more.c", first.text)
        self.assertIn("ALREADY SEARCHED", second.text)
        self.assertIn("more.c", second.text, "the cached answer is still given")
        self.assertNotIn("ALREADY SEARCHED", other.text)


class DelegatedResourceControlTests(unittest.TestCase):
    """Delegation must reach BOTH decisions, or it reaches neither usefully.

    Allowing the run while still wrapping it in a systemd-run that does not
    exist reported the sandbox unavailable for every command in the session --
    exactly the failure the delegation was added to prevent.
    """

    def setUp(self):
        self.runner = tool_runtime.SystemdScopeRunner()
        self.limits = tool_runtime.DEFAULT_CGROUP_LIMITS

    def test_delegation_allows_and_unwraps(self):
        with patch.dict(os.environ, {"SPEAR_RESOURCE_CONTROL": "delegated"}):
            self.assertTrue(self.runner.delegated())
            self.assertIsNone(self.runner.availability_for(self.limits))
            self.assertEqual(["/bin/true"],
                             self.runner.wrap(["/bin/true"], self.limits, unit="u"))

    def test_without_it_the_scope_is_still_required(self):
        env = {k: v for k, v in os.environ.items()
               if k != "SPEAR_RESOURCE_CONTROL"}
        with patch.dict(os.environ, env, clear=True):
            wrapped = self.runner.wrap(["/bin/true"], self.limits, unit="u")
            self.assertIn("systemd-run", wrapped[0])
            self.assertIn("--scope", wrapped)

    def test_inactive_limits_are_never_wrapped(self):
        empty = tool_runtime.CgroupLimits()
        self.assertEqual(["/bin/true"],
                         self.runner.wrap(["/bin/true"], empty, unit="u"))


class ToolchainReachableInsideTheSandboxTests(unittest.TestCase):
    """The cross-compilers must be on PATH and their symlinks must resolve.

    ib.md tells the operator to symlink each toolchain's bin/ into
    /usr/local/bin. The sandbox put neither that directory on PATH nor
    /opt/toolchains in the mount set, so every symlink dangled and `make so3`
    died on "aarch64-none-elf-gcc: No such file or directory" with the compiler
    mounted and unreachable.
    """

    def test_usr_local_bin_is_on_the_sandbox_path(self):
        self.assertIn("/usr/local/bin", tool_runtime.SandboxSpec().path.split(":"))

    def test_the_toolchain_root_is_bound_when_present(self):
        spec = tool_runtime.SandboxSpec()
        self.assertIn("/opt/toolchains", spec.toolchain_dirs)
        if not Path("/opt/toolchains").is_dir():
            self.skipTest("no /opt/toolchains on this host")
        with tempfile.TemporaryDirectory() as tmp:
            argv = tool_runtime.BubblewrapSandbox().build_argv(
                Workspace.from_path(tmp), ["/bin/true"])
        pairs = [(argv[i + 1], argv[i + 2]) for i, a in enumerate(argv)
                 if a == "--ro-bind" and i + 2 < len(argv)]
        self.assertIn(("/opt/toolchains", "/opt/toolchains"), pairs)


class CorpusRulesTravelWithTheHarnessTests(unittest.TestCase):
    """An orientation map must reach a colleague who only clones spear.

    They used to live in the tree they describe. In so3 a blanket `.*` line in
    .gitignore swallowed the file, so a fresh clone gave the assistant no idea
    where anything lived -- and the map is the highest-leverage context there
    is: a stale path in it sent the model looking for a build script that had
    been deleted months earlier.
    """

    @classmethod
    def setUpClass(cls):
        import rag_chat
        cls.rag_chat = rag_chat

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.saved = (self.rag_chat.PROJECT, self.rag_chat.PROJECT_ROOT,
                      self.rag_chat.CORPUS_ROOT, self.rag_chat.SHIPPED_CORPUS_RULES)
        self.shipped = self.root / "shipped"
        self.shipped.mkdir()
        (self.shipped / "demo.md").write_text("SHIPPED MAP\n")
        self.rag_chat.SHIPPED_CORPUS_RULES = str(self.shipped)
        self.rag_chat.PROJECT = "demo"
        self.tree = self.root / "tree"
        self.tree.mkdir()
        self.rag_chat.PROJECT_ROOT = self.rag_chat.CORPUS_ROOT = str(self.tree)

    def tearDown(self):
        (self.rag_chat.PROJECT, self.rag_chat.PROJECT_ROOT,
         self.rag_chat.CORPUS_ROOT, self.rag_chat.SHIPPED_CORPUS_RULES) = self.saved
        self.temp.cleanup()

    def test_the_shipped_map_is_used_when_the_tree_has_none(self):
        self.assertIn("SHIPPED MAP", self.rag_chat.load_corpus_rules())

    def test_a_tree_that_has_its_own_map_wins(self):
        """A tree someone else owns may carry one, and theirs beats ours."""
        (self.tree / ".edgem-rules.md").write_text("TREE MAP\n")
        rules = self.rag_chat.load_corpus_rules()
        self.assertIn("TREE MAP", rules)
        self.assertNotIn("SHIPPED MAP", rules)

    def test_a_corpus_with_no_map_anywhere_invents_nothing(self):
        self.rag_chat.PROJECT = "unknown-corpus"
        self.assertEqual("", self.rag_chat.load_corpus_rules())

