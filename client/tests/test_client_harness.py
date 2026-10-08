"""The chat client on the harness: workspace, compatibility and routing."""

import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from harness.resource_control import DEFAULT_CGROUP_LIMITS, DEFAULT_RESOURCE_LIMITS
from harness.tool_primitives import (
    Capability,
    ExecutionMode,
    NetworkBackend,
    ToolResult,
    shell_argv,
)
from harness.sandbox import BubblewrapSandbox, SandboxAvailability
from harness.tool_runtime import AuditLogger
from harness.workspace import Workspace


class RagChatWorkspaceIntegrationTests(unittest.TestCase):
    """Step 2 coverage: filesystem helpers must use the Workspace boundary."""

    @classmethod
    def setUpClass(cls):
        from cli import (
            operator_input, rag_chat, session_workspace,
            tool_handlers, tool_routing,
        )

        cls.rag_chat = rag_chat
        cls.operator_input = operator_input
        cls.session_workspace = session_workspace
        cls.tool_handlers = tool_handlers
        cls.tool_routing = tool_routing

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "workspace"
        self.root.mkdir()
        self.outside = Path(self.temp.name) / "outside"
        self.outside.mkdir()
        (self.root / "inside.txt").write_text("inside")
        (self.outside / "secret.txt").write_text("outside")
        self.old_project_root = self.session_workspace.PROJECT_ROOT
        self.old_workspace = self.session_workspace.WORKSPACE
        self.old_execution_mode = self.session_workspace.EXECUTION_MODE
        self.old_audit_logger = getattr(self.session_workspace, "AUDIT_LOGGER", None)
        self.audit_log = Path(self.temp.name) / "audit.jsonl"
        self.session_workspace.PROJECT_ROOT = str(self.root)
        self.session_workspace.WORKSPACE = Workspace.from_path(self.root)
        if self.old_audit_logger is not None:
            self.session_workspace.AUDIT_LOGGER = AuditLogger(self.audit_log)

    def tearDown(self):
        self.session_workspace.PROJECT_ROOT = self.old_project_root
        self.session_workspace.WORKSPACE = self.old_workspace
        self.session_workspace.EXECUTION_MODE = self.old_execution_mode
        if self.old_audit_logger is not None:
            self.session_workspace.AUDIT_LOGGER = self.old_audit_logger
        self.temp.cleanup()

    def test_evidence_visibility_follows_the_conversation_not_the_turn(self):
        """What the next request will actually carry, not what once happened.

        The router asks this before telling a model its observation is
        "already above in this conversation". After a compaction rewrote the
        conversation, that sentence is false and the only move it leaves is
        another spelling of the same read.
        """

        from models.model_backend import ConversationMessage, TextBlock, ToolResultBlock

        class Context:
            conversation = [
                ConversationMessage("assistant", (TextBlock("reading it"),)),
                ConversationMessage("user", (ToolResultBlock("call-1", "def add"),)),
            ]

        context = Context()
        self.assertTrue(self.tool_routing._evidence_in_context(context, "call-1"))
        self.assertFalse(self.tool_routing._evidence_in_context(context, "call-2"))

        # Compaction replaces the exchange with a summary.

        context.conversation = [
            ConversationMessage("user", (TextBlock("[earlier work summarised]"),)),
        ]
        self.assertFalse(self.tool_routing._evidence_in_context(context, "call-1"))

        # An empty result block is not evidence either.

        context.conversation = [
            ConversationMessage("user", (ToolResultBlock("call-1", ""),)),
        ]
        self.assertFalse(self.tool_routing._evidence_in_context(context, "call-1"))

    def test_a_read_only_refusal_is_about_scope_not_permissions(self):
        """"Command denied" invites the next spelling of the same intent.

        A run asked only which file defines `add` tried `sed -i`, was told it
        lacked permission, and spent thirty-five more steps finding other
        ways. The refusal has to say what it is FOR, and what to do instead.
        """

        self.assertIn("read-only task", self.tool_handlers.READ_ONLY_REFUSAL)
        self.assertIn("Do not try another way to modify or test",
                      self.tool_handlers.READ_ONLY_REFUSAL)
        self.assertIn("Answer the original question",
                      self.tool_handlers.READ_ONLY_REFUSAL)

        # Counted by category, not by signature: every spelling is the same
        # violation of the same scope.

        source = Path(self.tool_handlers.__file__).read_text()
        self.assertIn("READ_ONLY_VIOLATIONS", source)
        self.assertIn("EventType.READ_ONLY_VIOLATION", source)
        self.assertIn("read_only_violations", source)

    def test_read_rejects_traversal_and_symlink_escapes(self):
        (self.root / "outside-link").symlink_to(self.outside, target_is_directory=True)
        self.assertIn("path escapes the workspace", self.session_workspace.read_file("../outside/secret.txt"))
        self.assertIn("path escapes the workspace",
                      self.session_workspace.read_file("outside-link/secret.txt"))

    def test_file_lookup_does_not_fall_back_after_rejected_path(self):
        self.assertIsNone(self.session_workspace.find_file("../outside/secret.txt"))

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
            self.session_workspace.sandbox_mount(),
            BubblewrapSandbox().mount_root(self.session_workspace.WORKSPACE))
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
        hint = self.operator_input.corpus_mention_hint

        try:
            self.operator_input._HINTED_CORPORA.clear()
            # The case that cost a turn: a registered name in the question
            # while the tools are elsewhere.
            message = hint("generate a ping.c to run in so3", projects, "spear")
            self.assertIsNotNone(message)
            self.assertIn("so3", message)
            self.assertIn(str(so3), message)

            # Once per name per session — informs, does not nag.
            self.assertIsNone(hint("still about so3", projects, "spear"))

            # The longest registered name wins over a substring of it.
            self.operator_input._HINTED_CORPORA.clear()
            message = hint("port micropython-so3 please", projects, "spear")
            self.assertIn("micropython-so3", message)

            # Never fires for the corpus already in use.
            self.operator_input._HINTED_CORPORA.clear()
            self.assertIsNone(hint("something about so3", projects, "so3"))

            # Ordinary directory words and path fragments are not mentions.
            self.assertIsNone(hint("look in src for the parser", projects, "spear"))
            self.assertIsNone(hint("open so3/usr/main.c", projects, "spear"))
            # Substrings of longer words are not mentions either.
            self.assertIsNone(hint("the so3xyz variant", projects, "spear"))
            # A registry entry whose tree has gone is not offered.
            self.assertIsNone(hint("what about gone", projects, "spear"))

            # Above all: reporting must not move the workspace.
            before = (self.session_workspace.PROJECT_ROOT, self.session_workspace.WORKSPACE.root)
            self.operator_input._HINTED_CORPORA.clear()
            hint("build ping.c for so3", projects, "spear")
            self.assertEqual((self.session_workspace.PROJECT_ROOT,
                              self.session_workspace.WORKSPACE.root), before)
        finally:
            self.operator_input._HINTED_CORPORA.clear()

    def test_file_lookup_walks_the_workspace_for_a_bare_basename(self):
        # The rejected-path test above returns before the tree walk, which is
        # how a missing `pathlib` import survived in that walk until a real
        # session hit it.  This drives the walk itself.
        nested = self.root / "sub" / "deeper"
        nested.mkdir(parents=True)
        (nested / "ping.c").write_text("int main(void) { return 0; }")
        found = self.session_workspace.find_file("ping.c")
        self.assertIsNotNone(found)
        self.assertEqual(Path(found).name, "ping.c")
        self.assertTrue(Path(found).is_file())
        # A basename that exists nowhere still resolves to None, not an error.
        self.assertIsNone(self.session_workspace.find_file("absent-from-the-tree.c"))

    def test_write_file_rejects_absolute_and_outside_paths_before_writing(self):
        # In a session that MAY write: the path diagnostic belongs to the
        # tool's own validation, and the execution-mode gate now runs ahead
        # of the handler, so in safe mode the mode is reported instead. The
        # file is not written either way -- the case below holds that.
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK

        with patch.object(self.session_workspace, "confirm", return_value=True):
            result = self.tool_routing.execute_tool(
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
                self.session_workspace.EXECUTION_MODE = mode
                target = self.outside / f"new-{mode}.txt"

                with patch.object(self.session_workspace, "confirm", return_value=True):
                    result = self.tool_routing.execute_tool(
                        "write_file", {"path": str(target), "content": "x"}, {})

                self.assertTrue(result.startswith("ERROR"))
                self.assertFalse(target.exists())

    def test_write_file_creates_missing_parent_directories(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK
        with patch.object(self.session_workspace, "confirm", return_value=True):
            result = self.tool_routing.execute_tool(
                "write_file", {"path": "usr/src/ping.c", "content": "int main(void){}"}, {})
        # Refusing `usr/src/` while allowing `usr/src/ping.c` was arbitrary:
        # the containment check already covers the resolved path.
        self.assertIn("OK", result)
        self.assertTrue((self.root / "usr" / "src" / "ping.c").is_file())

    def test_write_file_creates_nothing_when_authorization_is_refused(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK
        with patch.object(self.session_workspace, "confirm", return_value=False):
            self.tool_routing.execute_tool(
                "write_file", {"path": "denied/deep/x.c", "content": "x"}, {})
        # Creating directories IS a mutation: it must happen after the gate,
        # never as a side effect of preparing the write.
        self.assertFalse((self.root / "denied").exists())

    def test_write_file_creates_no_directory_outside_a_root(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK
        with patch.object(self.session_workspace, "confirm", return_value=True):
            result = self.tool_routing.execute_tool(
                "write_file",
                {"path": str(self.outside / "deep" / "x.c"), "content": "x"}, {})
        self.assertIn("escapes the workspace", result)
        self.assertFalse((self.outside / "deep").exists())

    def test_edit_and_append_still_require_an_existing_file(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK
        with patch.object(self.session_workspace, "confirm", return_value=True):
            for tool in ("edit_file", "append_file"):
                result = self.tool_routing.execute_tool(
                    tool, {"path": "absent/deep/x.c", "content": "x",
                           "old_text": "a", "new_text": "b"}, {})
                self.assertIn("ERROR", result, tool)
        self.assertFalse((self.root / "absent").exists())

    def test_write_file_stays_inside_workspace(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK
        with patch.object(self.session_workspace, "confirm", return_value=True):
            result = self.tool_routing.execute_tool(
                "write_file", {"path": "created.txt", "content": "inside"}, {}
            )
        self.assertTrue(result.startswith("OK:"))
        self.assertEqual((self.root / "created.txt").read_text(), "inside\n")

    def test_safe_mode_denies_mutation_before_confirmation(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.SAFE
        with patch.object(self.session_workspace, "confirm") as confirm:
            result = self.tool_routing.execute_tool(
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
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK
        with patch.object(self.session_workspace, "confirm", return_value=False):
            result = self.tool_routing.execute_tool(
                "write_file", {"path": "not-approved.txt", "content": "x"}, {}
            )
        self.assertEqual(result, "CANCELLED")
        self.assertFalse((self.root / "not-approved.txt").exists())

    def test_auto_mode_allows_workspace_mutation(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.AUTO
        with patch.object(self.session_workspace, "confirm") as confirm:
            result = self.tool_routing.execute_tool(
                "write_file", {"path": "auto.txt", "content": "x"}, {}
            )
        self.assertTrue(result.startswith("OK:"))
        confirm.assert_not_called()
        self.assertEqual((self.root / "auto.txt").read_text(), "x\n")

    def test_mutating_tool_actions_are_audited_without_file_contents(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.AUTO
        result = self.tool_routing.execute_tool(
            "write_file", {"path": "audited.txt", "content": "TOP SECRET CONTENT"}, {}
        )
        self.assertTrue(result.startswith("OK:"))
        event = json.loads(self.audit_log.read_text().strip())
        self.assertEqual(event["action"], "write_file")
        self.assertEqual(event["paths"], ["audited.txt"])
        self.assertEqual(event["status"], "ok")
        self.assertNotIn("TOP SECRET CONTENT", self.audit_log.read_text())

    def test_denied_mutation_attempt_is_audited(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.SAFE
        self.tool_routing.execute_tool(
            "write_file", {"path": "denied.txt", "content": "x"}, {}
        )
        event = json.loads(self.audit_log.read_text().strip())
        self.assertEqual(event["action"], "write_file")
        self.assertEqual(event["status"], "denied")
        self.assertFalse(event["approved"])

    def test_safe_mode_allows_simple_read_only_command_without_shell(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.SAFE
        with patch.object(self.session_workspace.COMMAND_RUNNER, "run_simple",
                          return_value=ToolResult("ok", "command completed", stdout="ok")) as run:
            result = self.session_workspace.run_cmd("pwd", need_confirm=False)
        # pwd really runs in the sandbox: it prints the mount, which is now the
        # workspace's own host path.
        self.assertEqual(result, self.session_workspace.sandbox_mount() + "\n")
        run.assert_not_called()

    def test_safe_mode_rejects_mutation_but_allows_read_only_pipelines(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.SAFE
        self.assertIn("workspace_mutating commands are disabled in safe mode",
                      self.session_workspace.run_cmd("make test", need_confirm=False))
        # A pipeline that writes is refused on the capability, not the class.
        self.assertIn("required capabilities are not allowed",
                      self.session_workspace.run_cmd("echo x > f", need_confirm=False))
        # A read-only pipeline is authorised: it reaches execution instead of
        # being turned away by the policy. Whether the sandbox is available in
        # this environment is a separate concern, so assert on the refusal
        # messages being gone rather than on the output.
        outcome = self.session_workspace.run_cmd("ls | head", need_confirm=False)
        self.assertNotIn("disabled in safe mode", outcome)
        self.assertNotIn("required capabilities are not allowed", outcome)

    def test_ask_mode_fails_closed_for_complex_command_without_sandbox(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK
        old_sandbox = self.session_workspace.COMMAND_RUNNER.sandbox
        self.session_workspace.COMMAND_RUNNER.sandbox = BubblewrapSandbox(
            binary="/definitely/missing/bwrap"
        )
        try:
            with patch.object(self.session_workspace, "confirm", return_value=True), \
                 patch.object(self.session_workspace.COMMAND_RUNNER, "run_simple") as simple, \
                 patch.object(self.session_workspace.COMMAND_RUNNER, "run_complex_approved") as complex_run:
                result = self.session_workspace.run_cmd("ls | head", need_confirm=False)
            self.assertIn("sandbox unavailable", result)
            simple.assert_not_called()
            complex_run.assert_not_called()
        finally:
            self.session_workspace.COMMAND_RUNNER.sandbox = old_sandbox

    def test_auto_mode_fails_closed_for_complex_commands_without_sandbox(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.AUTO
        old_sandbox = self.session_workspace.COMMAND_RUNNER.sandbox
        self.session_workspace.COMMAND_RUNNER.sandbox = BubblewrapSandbox(
            binary="/definitely/missing/bwrap"
        )
        try:
            self.assertIn("sandbox unavailable",
                          self.session_workspace.run_cmd("ls | head", need_confirm=False))
        finally:
            self.session_workspace.COMMAND_RUNNER.sandbox = old_sandbox


class RagChatCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cli import (
            chat_settings, corpus_search, model_io, operator_input,
            rag_chat, session_history, session_workspace,
            startup_banner, terminal_ui, tool_routing,
        )

        cls.rag_chat = rag_chat
        cls.chat_settings = chat_settings
        cls.corpus_search = corpus_search
        cls.model_io = model_io
        cls.operator_input = operator_input
        cls.session_history = session_history
        cls.session_workspace = session_workspace
        cls.startup_banner = startup_banner
        cls.terminal_ui = terminal_ui
        cls.tool_routing = tool_routing

    def test_openai_tool_schema_is_preserved(self):
        tools = self.tool_routing.TOOLS
        self.assertEqual([tool["function"]["name"] for tool in tools], [
            "bash", "edit_file", "write_file", "append_file", "delete_file",
            "remember", "plan_change", "search_corpus", "search_internet",
            "fetch_url",
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
        text, calls = self.model_io.chat_once(
            client, [{"role": "user", "content": "hello"}], use_tools=True
        )
        self.assertEqual((text, calls), ("", []))
        self.assertTrue(completions.kwargs["stream"])
        self.assertEqual(completions.kwargs["tools"], self.tool_routing.TOOLS)
        self.assertEqual(completions.kwargs["extra_body"]["chat_template_kwargs"],
                         {"enable_thinking": False})
        self.assertEqual(completions.kwargs["model"], "qwen3")

    def test_rag_and_agent_loop_constants_are_unchanged(self):
        # Anchored on the module's own location, not an absolute literal: the
        # literal went stale when the tree was relocated and startup broke.
        app_dir = os.path.dirname(os.path.dirname(os.path.realpath(self.rag_chat.__file__)))
        self.assertEqual(self.chat_settings.APP_DIR, app_dir)
        self.assertEqual(self.chat_settings.ROOT_DIR, os.path.dirname(app_dir))
        # The production default is the checkout's own index; a test run that
        # names none is given a temporary one instead, never the real index.
        client = "".join(path.read_text() for path in
                         sorted(Path(self.rag_chat.__file__).parent.glob("*.py")))
        self.assertIn('f"{APP_DIR}/chromadb"', client)
        if not os.environ.get("SPEAR_DB_PATH"):
            self.assertFalse(self.chat_settings.DB_PATH.startswith(app_dir))
        self.assertNotIn("/opt/llm/spear/client", client)
        self.assertEqual(self.corpus_search.TOP_K, 12)
        self.assertEqual(self.corpus_search.MAX_CONTEXT_CHARS, 12000)
        self.assertEqual(self.session_history.MAX_HISTORY, 80)
        self.assertEqual(self.session_history.HISTORY_INJECT, 40)
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
        source = Path(self.session_workspace.__file__).read_text()
        self.assertIn('allow_absolute_paths="--allow-absolute-paths" in sys.argv[1:]',
                      source)

    def test_banner_names_the_backend_for_all_three(self):
        label = self.startup_banner.backend_label
        self.assertEqual(label("anthropic", "http://127.0.0.1:8080/v1"), "anthropic API")
        self.assertEqual(label("openai-compatible", "http://127.0.0.1:8081/v1"),
                         "remote/pod vLLM")
        # Generic: a port number cannot name a host. This used to assert
        # a hard-coded host name and kept claiming it after reds.conf
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
                self.startup_banner.backend_label("openai-compatible",
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
                self.startup_banner.backend_label("openai-compatible",
                                                  "http://127.0.0.1:8080/v1"),
                "local llama-server")
        finally:
            os.environ.pop("SPEAR_BACKEND_LABEL", None)
            if prev is not None:
                os.environ["SPEAR_BACKEND_LABEL"] = prev

    def test_backend_picker_prefers_flags_then_memory_then_prompt(self):
        from cli import backend_select

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
                help_text = self.startup_banner.HELP_TEXT
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

        help_text = self.startup_banner.HELP_TEXT
        # Derive the flags from the parsers themselves rather than restating a
        # list here: a hand-written help drifts silently, and the whole point
        # of adding it was that the flags were undiscoverable.
        parsed = set()
        for fn in (self.session_workspace.execution_mode_from_argv,
                   self.session_workspace.capability_policy_from_argv,
                   self.model_io.model_provider_from_argv):
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
            row = self.startup_banner.permissions_row(mode)
            self.assertIn(expected, row, f"{mode}: {row}")
            for flag in flags:
                self.assertIn(flag, row, f"{mode} lacks {flag}: {row}")
        # SAFE must never claim it will ask; ASK must never claim it is silent.
        self.assertNotIn("ask before each",
                         self.startup_banner.permissions_row(ExecutionMode.SAFE))
        self.assertNotIn("(default)",
                         self.startup_banner.permissions_row(ExecutionMode.AUTO))

    def test_tab_completes_commands_and_nothing_else(self):
        complete = self.operator_input.completion_candidates
        self.assertEqual(complete("/st", "/st"), ["/standard"])
        self.assertIn("/search", complete("/", "/"))
        # A question to the model is never completed.
        self.assertEqual(complete("what does /st", "/st"), [])
        self.assertEqual(complete("/search reg", "reg"), [])

    def test_tab_opens_a_lone_directory_instead_of_closing_it(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            os.mkdir(os.path.join(directory, "standards"))
            pdf = os.path.join(directory, "standards", "manual.pdf")
            open(pdf, "w").close()
            typed = os.path.join(directory, "stan")

            self.assertEqual(self.operator_input.completion_candidates(
                "/standard ingest " + typed, typed), [pdf])

    def test_a_long_command_shows_its_phases_on_the_spinner(self):
        import io
        from unittest import mock

        class Holder:
            label = ""

        spinner = Holder()
        out = io.StringIO()

        with mock.patch("sys.stdout", out), \
                mock.patch.object(self.rag_chat.time, "time",
                                  side_effect=[0, 10, 20, 30]):
            progress = self.terminal_ui.SpinnerProgress(spinner)
            progress("extracting text")
            progress("writing corpus", 0, 100)
            progress("writing corpus", 25, 100)
            self.assertIn("[2] writing corpus 25% (25/100)", spinner.label)
            self.assertIn("~30s left", spinner.label)
            progress.finish()

        lines = out.getvalue()
        self.assertIn("[1] extracting text", lines)
        self.assertIn("[2] writing corpus", lines)

    def test_banner_names_where_rules_skills_and_benches_come_from(self):
        # A missing deployment env file used to fall back to the in-tree
        # directories in silence; the banner is where that must show.
        import os
        import tempfile

        app_dir = self.chat_settings.APP_DIR
        in_tree = {name: os.path.join(app_dir, name)
                   for name in ("rules", "skills", "benches")}
        row = self.startup_banner.content_row(in_tree)
        self.assertIn("in-tree", row)
        self.assertNotIn("missing", row)

        with tempfile.TemporaryDirectory() as directory:
            external = {name: os.path.join(directory, name)
                        for name in ("rules", "skills", "benches")}
            for path in external.values():
                os.mkdir(path)

            row = self.startup_banner.content_row(external)
            self.assertIn(directory, row)
            self.assertIn("rules, skills, benches", row)
            self.assertNotIn("in-tree", row)
            self.assertNotIn("missing", row)

            os.rmdir(external["benches"])
            self.assertIn("missing: benches", self.startup_banner.content_row(external))

    def test_session_settings_are_reachable_as_flags(self):
        """They were environment variables only. Nothing in --help mentioned
        them, and configuring one run meant `VAR=x VAR2=y spear-chat` — neither
        discoverable nor reviewable next to the other flags."""
        import os

        environment = {}
        with unittest.mock.patch.dict(os.environ, environment, clear=False):
            left = self.chat_settings.apply_env_options(
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
            self.chat_settings.apply_env_options(["--ctx"])

        self.assertIn("--ctx", str(raised.exception))

    def test_every_settings_flag_is_documented_by_its_own_table(self):
        """The help text is rendered FROM the parser's table, so a flag added
        without a line in --help cannot happen."""
        rendered = self.startup_banner._env_option_lines()

        for flag, (variable, _) in self.chat_settings.ENV_OPTIONS.items():
            self.assertIn(flag, rendered)
            self.assertIn(variable, rendered)

    def test_legacy_mode_flags_and_no_network_restriction(self):
        for flag, mode in (("--safe", ExecutionMode.SAFE), ("--ask", ExecutionMode.ASK),
                           ("--confirm", ExecutionMode.ASK), ("--auto", ExecutionMode.AUTO),
                           ("--yolo", ExecutionMode.AUTO)):
            self.assertEqual(self.session_workspace.execution_mode_from_argv([flag]), mode)
        policy = self.session_workspace.capability_policy_from_argv(["--ask", "--no-network"])
        self.assertNotIn(Capability.NETWORK, policy.ask)
        # --no-network has to reach AUTO too, now that AUTO has the network:
        # stripping it from ASK alone would leave the flag doing nothing in
        # the one mode where it is not confirmed command by command.
        self.assertNotIn(Capability.NETWORK, policy.auto)
        self.assertIn(Capability.NETWORK,
                      self.session_workspace.capability_policy_from_argv(["--ask"]).ask)
        self.assertIn(Capability.NETWORK,
                      self.session_workspace.capability_policy_from_argv([]).auto)


class Phase1bRagChatRoutingTests(unittest.TestCase):
    """Step 1b-3: classify in rag_chat, execute through CommandRunner."""

    @classmethod
    def setUpClass(cls):
        from cli import session_workspace

        cls.session_workspace = session_workspace

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp.name) / "workspace"
        self.workspace_root.mkdir()
        self.old_mode = self.session_workspace.EXECUTION_MODE
        self.old_workspace = self.session_workspace.WORKSPACE
        self.old_project_root = self.session_workspace.PROJECT_ROOT
        self.old_sandbox = self.session_workspace.COMMAND_RUNNER.sandbox
        self.old_audit_logger = self.session_workspace.AUDIT_LOGGER
        self.session_workspace.WORKSPACE = Workspace.from_path(self.workspace_root)
        self.session_workspace.PROJECT_ROOT = str(self.workspace_root)
        self.session_workspace.AUDIT_LOGGER = AuditLogger(Path(self.temp.name) / "audit.jsonl")

    def tearDown(self):
        self.session_workspace.EXECUTION_MODE = self.old_mode
        self.session_workspace.WORKSPACE = self.old_workspace
        self.session_workspace.PROJECT_ROOT = self.old_project_root
        self.session_workspace.COMMAND_RUNNER.sandbox = self.old_sandbox
        self.session_workspace.AUDIT_LOGGER = self.old_audit_logger
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
                self.session_workspace.EXECUTION_MODE = mode
                sandbox = self.sandbox_available()
                self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
                with patch.object(self.session_workspace, "confirm") as confirm, \
                     patch.object(self.session_workspace.COMMAND_RUNNER, "run_simple") as simple, \
                     patch.object(self.session_workspace.COMMAND_RUNNER, "run_complex_approved") as complex_run:
                    result = self.session_workspace.run_cmd("git status", need_confirm=False)
                self.assertEqual(result, "sandboxed")
                confirm.assert_not_called()
                simple.assert_not_called()
                complex_run.assert_not_called()
                sandbox.ensure_available.assert_called_once_with(self.session_workspace.WORKSPACE)
                sandbox.run.assert_called_once_with(
                    self.session_workspace.WORKSPACE,
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
        self.session_workspace.EXECUTION_MODE = ExecutionMode.SAFE
        sandbox = MagicMock()
        sandbox.ensure_available.return_value = ToolResult("failed", "sandbox unavailable")
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.session_workspace.COMMAND_RUNNER, "run_simple") as simple, \
             patch.object(self.session_workspace.COMMAND_RUNNER, "run_complex_approved") as complex_run:
            result = self.session_workspace.run_cmd("pwd", need_confirm=False)
        self.assertIn("sandbox unavailable", result)
        simple.assert_not_called()
        complex_run.assert_not_called()
        sandbox.run.assert_not_called()

    def test_auto_workspace_mutation_uses_available_sandbox(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.AUTO
        sandbox = self.sandbox_available()
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox

        result = self.session_workspace.run_cmd("make test", need_confirm=False)

        self.assertEqual(result, "sandboxed")
        sandbox.ensure_available.assert_called_once_with(self.session_workspace.WORKSPACE)
        sandbox.run.assert_called_once_with(
            self.session_workspace.WORKSPACE, ["make", "test"], profile=ANY,
            resource_limits=DEFAULT_RESOURCE_LIMITS,
            cgroup_limits=DEFAULT_CGROUP_LIMITS,
        )
        self.assertEqual(sandbox.run.call_args.kwargs["profile"].capabilities, frozenset({
            Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE,
        }))

    def test_auto_shell_complex_uses_shell_inside_available_sandbox(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.AUTO
        sandbox = self.sandbox_available()
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox

        result = self.session_workspace.run_cmd("printf sandboxed | cat", need_confirm=False)

        self.assertEqual(result, "sandboxed")
        sandbox.run.assert_called_once_with(
            self.session_workspace.WORKSPACE,
            shell_argv("printf sandboxed | cat"),
            profile=ANY,
            resource_limits=DEFAULT_RESOURCE_LIMITS,
            cgroup_limits=DEFAULT_CGROUP_LIMITS,
        )

    def test_ask_workspace_mutation_uses_sandbox_after_confirmation(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK
        sandbox = self.sandbox_available()
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.session_workspace, "confirm", return_value=True) as confirm:
            result = self.session_workspace.run_cmd("pytest tests", need_confirm=False)

        self.assertEqual(result, "sandboxed")
        confirm.assert_called_once()
        sandbox.run.assert_called_once_with(
            self.session_workspace.WORKSPACE, ["pytest", "tests"], profile=ANY,
            resource_limits=DEFAULT_RESOURCE_LIMITS,
            cgroup_limits=DEFAULT_CGROUP_LIMITS,
        )

    def test_ask_shell_complex_uses_sandbox_after_confirmation(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK
        sandbox = self.sandbox_available()
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.session_workspace, "confirm", return_value=True) as confirm:
            result = self.session_workspace.run_cmd("printf sandboxed | cat", need_confirm=False)

        self.assertEqual(result, "sandboxed")
        confirm.assert_called_once()
        sandbox.run.assert_called_once_with(
            self.session_workspace.WORKSPACE,
            shell_argv("printf sandboxed | cat"),
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
        self.session_workspace.EXECUTION_MODE = ExecutionMode.AUTO
        sandbox = self.sandbox_available()
        sandbox.run.return_value = ToolResult(
            "failed", "command failed", stdout="last lines", exit_code=2)
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
        result = self.session_workspace.run_cmd("make 2>&1 | tail -3", need_confirm=False)
        self.assertIn("(exit 2)", result)

    def test_sigpipe_from_head_is_not_reported_as_a_failure(self):
        """`find . | head -5` is not a failed command.

        pipefail surfaces the SIGPIPE that head causes upstream, which is the
        POINT of head, not an error. Reporting 141 would send the model
        debugging one of its most common idioms.
        """
        self.session_workspace.EXECUTION_MODE = ExecutionMode.AUTO
        sandbox = self.sandbox_available()
        sandbox.run.return_value = ToolResult(
            "failed", "command failed", stdout="a.c\nb.c", exit_code=141)
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
        result = self.session_workspace.run_cmd("find . -name '*.c' | head -5",
                                                need_confirm=False)
        self.assertNotIn("exit", result)
        self.assertIn("a.c", result)

    def test_ask_fails_closed_when_sandbox_is_unavailable(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK
        sandbox = MagicMock()
        sandbox.ensure_available.return_value = ToolResult("failed", "sandbox unavailable")
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.session_workspace, "confirm", return_value=True), \
             patch.object(self.session_workspace.COMMAND_RUNNER, "run_simple") as simple, \
             patch.object(self.session_workspace.COMMAND_RUNNER, "run_complex_approved",
                          return_value=ToolResult("ok", "done", stdout="compatibility")) as complex_run:
            mutate = self.session_workspace.run_cmd("make test", need_confirm=False)
            complex_result = self.session_workspace.run_cmd("printf compatibility | cat", need_confirm=False)

        self.assertIn("sandbox unavailable", mutate)
        self.assertIn("sandbox unavailable", complex_result)
        simple.assert_not_called()
        complex_run.assert_not_called()

    def test_ask_declined_command_does_not_start_sandbox_or_runner(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK
        sandbox = self.sandbox_available()
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.session_workspace, "confirm", return_value=False), \
             patch.object(self.session_workspace.COMMAND_RUNNER, "run_simple") as simple, \
             patch.object(self.session_workspace.COMMAND_RUNNER, "run_complex_approved") as complex_run:
            mutate = self.session_workspace.run_cmd("make test", need_confirm=False)
            complex_result = self.session_workspace.run_cmd("printf x | cat", need_confirm=False)

        self.assertEqual(mutate, "CANCELLED")
        self.assertEqual(complex_result, "CANCELLED")
        sandbox.ensure_available.assert_not_called()
        sandbox.run.assert_not_called()
        simple.assert_not_called()
        complex_run.assert_not_called()

    def test_auto_unavailable_sandbox_never_uses_non_sandboxed_runner(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.AUTO
        sandbox = MagicMock()
        sandbox.ensure_available.return_value = ToolResult("failed", "sandbox unavailable")
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.session_workspace.COMMAND_RUNNER, "run_simple") as simple, \
             patch.object(self.session_workspace.COMMAND_RUNNER, "run_complex_approved") as complex_run:
            mutate = self.session_workspace.run_cmd("make test", need_confirm=False)
            complex_result = self.session_workspace.run_cmd("printf x | cat", need_confirm=False)

        self.assertIn("sandbox unavailable", mutate)
        self.assertIn("sandbox unavailable", complex_result)
        simple.assert_not_called()
        complex_run.assert_not_called()

    def test_auto_fails_closed_for_absent_inexecutable_and_refused_sandboxes(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.AUTO
        non_executable = Path(self.temp.name) / "not-executable-bwrap"
        non_executable.write_text("not executable")
        non_executable.chmod(0o644)
        refused = BubblewrapSandbox(binary="bwrap")
        refused.availability = SandboxAvailability.REFUSED
        sandboxes = (
            BubblewrapSandbox(binary="/definitely/missing/bwrap"),
            BubblewrapSandbox(binary=str(non_executable)),
            refused,
        )
        for sandbox in sandboxes:
            self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
            with patch.object(self.session_workspace.COMMAND_RUNNER, "run_simple") as simple, \
                 patch.object(self.session_workspace.COMMAND_RUNNER, "run_complex_approved") as complex_run:
                result = self.session_workspace.run_cmd("make test", need_confirm=False)
            self.assertIn("sandbox unavailable", result)
            simple.assert_not_called()
            complex_run.assert_not_called()

    def test_dangerous_command_never_reaches_bubblewrap(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.AUTO
        sandbox = self.sandbox_available()
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox

        result = self.session_workspace.run_cmd("rm -rf build", need_confirm=False)

        self.assertIn("command denied", result)
        sandbox.ensure_available.assert_not_called()
        sandbox.run.assert_not_called()

    def test_network_is_refused_before_execution_in_safe(self):
        """SAFE stops it before bubblewrap is even asked to exist, and the
        refusal names both ways forward: the flags, and fetch_url."""
        self.session_workspace.EXECUTION_MODE = ExecutionMode.SAFE
        sandbox = self.sandbox_available()
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.session_workspace.COMMAND_RUNNER, "run_simple") as simple, \
             patch.object(self.session_workspace.COMMAND_RUNNER, "run_complex_approved") as complex_run:
            result = self.session_workspace.run_cmd("curl https://example.invalid",
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
        self.session_workspace.EXECUTION_MODE = ExecutionMode.AUTO
        sandbox = self.sandbox_available()
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.session_workspace, "confirm") as confirm:
            result = self.session_workspace.run_cmd("curl https://example.invalid",
                                                    need_confirm=False)
        self.assertEqual(result, "sandboxed")
        confirm.assert_not_called()
        sandbox.preflight_network.assert_called_once_with(self.session_workspace.WORKSPACE)

    def test_ask_network_routes_only_to_slirp_sandbox_after_confirmation(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK
        sandbox = self.sandbox_available()
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.session_workspace, "confirm", return_value=True) as confirm, \
             patch.object(self.session_workspace.COMMAND_RUNNER, "run_simple") as simple, \
             patch.object(self.session_workspace.COMMAND_RUNNER, "run_complex_approved") as complex_run:
            result = self.session_workspace.run_cmd("curl https://example.invalid", need_confirm=False)
        self.assertEqual(result, "sandboxed")
        confirm.assert_called_once()
        sandbox.preflight_network.assert_called_once_with(self.session_workspace.WORKSPACE)
        sandbox.run.assert_called_once_with(
            self.session_workspace.WORKSPACE, ["curl", "https://example.invalid"],
            profile=ANY, backend=NetworkBackend.SLIRP4NETNS,
            resource_limits=DEFAULT_RESOURCE_LIMITS,
            cgroup_limits=DEFAULT_CGROUP_LIMITS,
        )
        self.assertTrue(sandbox.run.call_args.kwargs["profile"].network)
        simple.assert_not_called()
        complex_run.assert_not_called()

    def test_ask_network_declined_starts_no_sandbox_or_preflight(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK
        sandbox = self.sandbox_available()
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.session_workspace, "confirm", return_value=False) as confirm:
            result = self.session_workspace.run_cmd("curl https://example.invalid", need_confirm=False)
        self.assertEqual(result, "CANCELLED")
        confirm.assert_called_once()
        sandbox.preflight_network.assert_not_called()
        sandbox.run.assert_not_called()

    def test_ask_network_fails_closed_for_preflight_or_containment_failure(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK
        for summary in ("slirp unavailable", "network containment failure"):
            with self.subTest(summary=summary):
                sandbox = self.sandbox_available()
                sandbox.preflight_network.return_value = ToolResult("failed", summary)
                self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
                with patch.object(self.session_workspace, "confirm", return_value=True), \
                     patch.object(self.session_workspace.COMMAND_RUNNER, "run_simple") as simple:
                    result = self.session_workspace.run_cmd("curl https://example.invalid", need_confirm=False)
                self.assertIn(summary, result)
                sandbox.run.assert_not_called()
                simple.assert_not_called()

    def test_ask_remote_write_and_ssh_are_denied_without_confirmation(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK
        sandbox = self.sandbox_available()
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
        for command in (
            "curl -X POST https://example.invalid", "git push", "ssh example.invalid",
        ):
            with self.subTest(command=command), patch.object(self.session_workspace, "confirm") as confirm:
                result = self.session_workspace.run_cmd(command, need_confirm=False)
            self.assertIn("required capabilities", result)
            confirm.assert_not_called()
        sandbox.preflight_network.assert_not_called()
        sandbox.run.assert_not_called()

    def test_ask_network_shell_uses_single_confirmation_and_slirp(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK
        sandbox = self.sandbox_available()
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.session_workspace, "confirm", return_value=True) as confirm:
            result = self.session_workspace.run_cmd("curl https://example.invalid | head", need_confirm=False)
        self.assertEqual(result, "sandboxed")
        confirm.assert_called_once()
        sandbox.run.assert_called_once_with(
            self.session_workspace.WORKSPACE,
            shell_argv("curl https://example.invalid | head"),
            profile=ANY, backend=NetworkBackend.SLIRP4NETNS,
            resource_limits=DEFAULT_RESOURCE_LIMITS,
            cgroup_limits=DEFAULT_CGROUP_LIMITS,
        )
        profile = sandbox.run.call_args.kwargs["profile"]
        self.assertTrue(profile.network)
        self.assertTrue(profile.shell_complex)

    def test_ask_network_preflight_is_cached_and_audited(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK
        sandbox = self.sandbox_available()
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(self.session_workspace, "confirm", return_value=True):
            self.session_workspace.run_cmd("curl https://example.invalid", need_confirm=False)
            self.session_workspace.run_cmd("curl https://example.invalid", need_confirm=False)
        self.assertEqual(sandbox.preflight_network.call_count, 1)
        event = json.loads((Path(self.temp.name) / "audit.jsonl").read_text().splitlines()[0])
        self.assertEqual(event["required_capabilities"], ["network"])
        self.assertEqual(event["granted_capabilities"], ["network"])
        self.assertTrue(event["execution_profile"]["network"])
        self.assertTrue(event["sandboxed"])

    @unittest.skipUnless(os.environ.get("SPEAR_TEST_NETWORK") == "1",
                         "set SPEAR_TEST_NETWORK=1 for opt-in end-to-end routing")
    def test_opt_in_ask_network_routes_through_slirp(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.ASK
        self.session_workspace.COMMAND_RUNNER.sandbox = BubblewrapSandbox()
        with patch.object(self.session_workspace, "confirm", return_value=True):
            result = self.session_workspace.run_cmd(
                "curl --fail --silent --show-error https://example.com/", need_confirm=False
            )
        self.assertNotIn("ERROR:", result)

    def test_command_audit_records_capability_names_only(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.AUTO
        sandbox = self.sandbox_available()
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox

        self.session_workspace.run_cmd("make test", need_confirm=False)

        event = json.loads((Path(self.temp.name) / "audit.jsonl").read_text())
        self.assertEqual(event["required_capabilities"], [
            "filesystem:read", "workspace:write",
        ])
        self.assertEqual(event["granted_capabilities"], [
            "filesystem:read", "workspace:write",
        ])

    def test_sandbox_preflight_is_cached_between_sandboxed_commands(self):
        self.session_workspace.EXECUTION_MODE = ExecutionMode.AUTO
        sandbox = BubblewrapSandbox(binary="bwrap")
        self.session_workspace.COMMAND_RUNNER.sandbox = sandbox
        with patch.object(sandbox, "preflight",
                          return_value=ToolResult("ok", "sandbox preflight completed")) as preflight, \
             patch.object(sandbox, "run",
                          return_value=ToolResult("ok", "done", stdout="sandboxed")):
            self.session_workspace.run_cmd("make test", need_confirm=False)
            self.session_workspace.run_cmd("pytest tests", need_confirm=False)

        self.assertEqual(preflight.call_count, 1)


if __name__ == "__main__":
    unittest.main()
