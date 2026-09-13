import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import model_backend
from model_backend import (
    AnthropicBackend,
    ModelBackendConfigurationError,
    ConversationMessage,
    ModelToolCall,
    ModelTurn,
    OpenAICompatibleBackend,
    StopReason,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
    canonical_tools_from_openai,
)


def chunk(*, content=None, calls=(), finish_reason=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=content, tool_calls=calls),
        finish_reason=finish_reason,
    )])


def tool_delta(index, *, call_id=None, name=None, arguments=None):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=call_id, function=function)


class FakeCompletions:
    def __init__(self, streams):
        self.streams = list(streams)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.streams.pop(0)


class FakeClient:
    def __init__(self, streams):
        self.chat = SimpleNamespace(completions=FakeCompletions(streams))


class FakeAnthropicStream:
    def __init__(self, text_deltas, final_message):
        self.text_stream = iter(text_deltas)
        self.final_message = final_message
        self.entered = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def get_final_message(self):
        return self.final_message


class FakeAnthropicMessages:
    def __init__(self, streams):
        self.streams = list(streams)
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return self.streams.pop(0)


class FakeAnthropicClient:
    def __init__(self, streams):
        self.messages = FakeAnthropicMessages(streams)


class FakeModelBackend:
    """Test-only provider replacement; it never executes an SPEAR tool."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.conversations = []

    def complete(self, *, system, conversation, tools, use_tools, on_token=None):
        self.conversations.append(tuple(conversation))
        return self.turns.pop(0)


class OpenAICompatibleBackendTests(unittest.TestCase):
    def setUp(self):
        self.tool = ToolDefinition(
            name="read_file", description="Read a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        self.conversation = [ConversationMessage("user", (TextBlock("Read README"),))]

    def test_request_preserves_current_qwen_openai_contract(self):
        client = FakeClient([[chunk(content="done", finish_reason="stop")]])
        backend = OpenAICompatibleBackend(client, model="qwen3", temperature=0.25, max_tokens=8192)
        turn = backend.complete(
            system="system", conversation=self.conversation, tools=[self.tool], use_tools=True
        )
        self.assertEqual(turn.text, "done")
        self.assertEqual(turn.stop_reason, StopReason.END_TURN)
        kwargs = client.chat.completions.calls[0]
        self.assertEqual(kwargs["model"], "qwen3")
        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["temperature"], 0.25)
        self.assertEqual(kwargs["top_p"], 0.8)
        self.assertEqual(kwargs["max_tokens"], 8192)
        self.assertEqual(kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"], False)
        self.assertEqual(kwargs["extra_body"]["top_k"], 20)
        self.assertEqual(kwargs["extra_body"]["repeat_penalty"], 1.05)
        self.assertEqual(kwargs["extra_body"]["repetition_penalty"], 1.05)
        self.assertEqual(kwargs["tools"][0]["function"]["parameters"], self.tool.input_schema)

    def test_multiple_streamed_tool_calls_are_typed_and_ordered(self):
        client = FakeClient([[
            chunk(calls=(tool_delta(0, call_id="a", name="read_file", arguments='{"path":'),)),
            chunk(calls=(tool_delta(0, arguments='"README.md"}'), tool_delta(
                1, call_id="b", name="write_file", arguments='{"path":"result.txt"}'
            )), finish_reason="tool_calls"),
        ]])
        turn = OpenAICompatibleBackend(client).complete(
            system="s", conversation=self.conversation, tools=[self.tool], use_tools=True
        )
        self.assertEqual(turn.stop_reason, StopReason.TOOL_USE)
        self.assertEqual(turn.tool_calls, (
            ModelToolCall("a", "read_file", {"path": "README.md"}),
            ModelToolCall("b", "write_file", {"path": "result.txt"}),
        ))

    def test_invalid_json_is_controlled_and_never_becomes_a_tool_call(self):
        client = FakeClient([[chunk(calls=(tool_delta(
            0, call_id="a", name="bash", arguments='{"command":'
        ),), finish_reason="tool_calls")]])
        turn = OpenAICompatibleBackend(client).complete(
            system="s", conversation=self.conversation, tools=[self.tool], use_tools=True
        )
        self.assertEqual(turn.stop_reason, StopReason.ERROR)
        self.assertEqual(turn.tool_calls, ())
        self.assertIn("invalid JSON", turn.error)

    def test_qwen_leaked_xml_and_json_recovery_stays_in_openai_backend(self):
        client = FakeClient([[chunk(content=(
            '<tool_call><function=read_file><parameter=path>README.md</parameter>'
            '</function></tool_call>{"name":"write_file","arguments":{"path":"result.txt"}}'
        ), finish_reason="stop")]])
        turn = OpenAICompatibleBackend(client).complete(
            system="s", conversation=self.conversation, tools=[self.tool], use_tools=True
        )
        self.assertEqual(turn.text, "")
        self.assertEqual([call.name for call in turn.tool_calls], ["read_file", "write_file"])
        self.assertEqual(turn.tool_calls[0].arguments, {"path": "README.md"})

    def test_a_leaked_normative_call_is_recovered_like_any_other(self):
        """The dotted names were the only ones \\w+ could not recover.

        Verbatim from the turn that ended on one: twelve rounds of reading,
        a plan, and a leaked standard.fetch that read as prose. The runtime
        took the plan for a conclusion and the turn wrote nothing.
        """
        client = FakeClient([[chunk(content=(
            "I'll implement the change by adding the CmdAckKind enum.\n\n"
            "<function=standard.fetch>\n"
            "<parameter=source_id>\nstd-37d3d6aafd5ea8a36f0b072c6d02830e\n</parameter>\n"
            "<parameter=include_parent>\ntrue\n</parameter>\n"
            "</function>\n</tool_call>"
        ), finish_reason="stop")]])
        turn = OpenAICompatibleBackend(client).complete(
            system="s", conversation=self.conversation, tools=[self.tool], use_tools=True
        )
        self.assertEqual([call.name for call in turn.tool_calls],
                         ["standard.fetch"])
        self.assertEqual(turn.tool_calls[0].arguments,
                         {"source_id": "std-37d3d6aafd5ea8a36f0b072c6d02830e",
                          "include_parent": "true"})
        self.assertNotIn("<function=", turn.text)

    def test_a_leaked_json_call_with_a_dotted_name_is_recovered(self):
        client = FakeClient([[chunk(content=(
            '{"name":"standard.search","arguments":{"query":"ACK"}}'
        ), finish_reason="stop")]])
        turn = OpenAICompatibleBackend(client).complete(
            system="s", conversation=self.conversation, tools=[self.tool], use_tools=True
        )
        self.assertEqual([call.name for call in turn.tool_calls],
                         ["standard.search"])

    def test_canonical_tool_result_maps_to_openai_tool_message(self):
        client = FakeClient([[chunk(content="final", finish_reason="stop")]])
        conversation = [
            ConversationMessage("user", (TextBlock("do it"),)),
            ConversationMessage("assistant", (ToolUseBlock("call_1", "read_file", {"path": "README.md"}),)),
            ConversationMessage("user", (ToolResultBlock("call_1", "contents", False),)),
        ]
        OpenAICompatibleBackend(client).complete(
            system="s", conversation=conversation, tools=[self.tool], use_tools=True
        )
        messages = client.chat.completions.calls[0]["messages"]
        self.assertEqual(messages[-1], {"role": "tool", "tool_call_id": "call_1", "content": "contents"})

    def test_existing_openai_tool_definitions_are_not_duplicated(self):
        source = [{"type": "function", "function": {
            "name": "bash", "description": "run", "parameters": {"type": "object"},
        }}]
        self.assertEqual(canonical_tools_from_openai(source)[0], ToolDefinition(
            "bash", "run", {"type": "object"}
        ))

    def test_fake_backend_four_turn_scenario_uses_spear_tools_and_policy(self):
        import rag_chat
        from tool_runtime import ExecutionMode, Workspace

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README.md").write_text("TODO: write result\n", encoding="utf-8")
            old_workspace = rag_chat.WORKSPACE
            old_root = rag_chat.PROJECT_ROOT
            old_mode = rag_chat.EXECUTION_MODE
            try:
                rag_chat.WORKSPACE = Workspace.from_path(root)
                rag_chat.PROJECT_ROOT = str(root)
                rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
                backend = FakeModelBackend([
                    # read README, search TODO, create result, then answer
                    SimpleNamespace(tool_calls=(ModelToolCall(
                        "read", "read_file", {"path": "README.md"}),), text="", stop_reason=StopReason.TOOL_USE),
                    SimpleNamespace(tool_calls=(ModelToolCall(
                        "search", "bash", {"command": "grep -n TODO README.md"}),), text="", stop_reason=StopReason.TOOL_USE),
                    SimpleNamespace(tool_calls=(ModelToolCall(
                        "write", "write_file", {"path": "result.txt", "content": "TODO found\n"}),), text="", stop_reason=StopReason.TOOL_USE),
                    SimpleNamespace(tool_calls=(), text="Created result.txt from the TODO.", stop_reason=StopReason.END_TURN),
                ])
                conversation = [ConversationMessage("user", (TextBlock("process README"),))]
                final = ""
                for _ in range(8):
                    turn = backend.complete(system="s", conversation=conversation, tools=(), use_tools=True)
                    if not turn.tool_calls:
                        final = turn.text
                        break
                    conversation.append(ConversationMessage("assistant", tuple(
                        ToolUseBlock(call.id, call.name, call.arguments) for call in turn.tool_calls
                    )))
                    results = []
                    for call in turn.tool_calls:
                        result = rag_chat.execute_tool(call.name, dict(call.arguments), {})
                        results.append(ToolResultBlock(call.id, result, result.startswith("ERROR:")))
                    conversation.append(ConversationMessage("user", tuple(results)))
            finally:
                rag_chat.WORKSPACE = old_workspace
                rag_chat.PROJECT_ROOT = old_root
                rag_chat.EXECUTION_MODE = old_mode
            self.assertEqual(final, "Created result.txt from the TODO.")
            self.assertEqual((root / "result.txt").read_text(encoding="utf-8"), "TODO found\n")
            self.assertEqual(len(backend.conversations), 4)
            self.assertIsInstance(backend.conversations[1][-1].content[0], ToolResultBlock)


class AnthropicBackendTests(unittest.TestCase):
    def setUp(self):
        self.tool = ToolDefinition(
            name="bash", description="Run a command",
            input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
        )
        self.conversation = [ConversationMessage("user", (TextBlock("Read README"),))]

    @staticmethod
    def message(*, content=(), stop_reason="end_turn", usage=None):
        return SimpleNamespace(content=list(content), stop_reason=stop_reason, usage=usage)

    def backend(self, final_message, *, deltas=()):
        stream = FakeAnthropicStream(deltas, final_message)
        client = FakeAnthropicClient([stream])
        return AnthropicBackend(client=client, model="claude-sonnet-5"), client, stream

    def test_a_dotted_tool_name_is_translated_at_the_wire_and_back(self):
        """Anthropic refuses `standard.search`; the loop must still see it.

        The 400 this prevents came back before a single token, so an
        otherwise healthy provider answered no turn at all.
        """
        normative = ToolDefinition(
            name="standard.search", description="Search the bound standard",
            input_schema={"type": "object", "properties": {}},
        )
        final = self.message(
            content=(SimpleNamespace(type="tool_use", id="c1",
                                     name="standard_search",
                                     input={"query": "8.4.1.1"}),),
            stop_reason="tool_use",
        )
        backend, client, _ = self.backend(final)
        turn = backend.complete(system="s", conversation=self.conversation,
                                tools=[normative], use_tools=True)
        sent = client.messages.calls[0]["tools"][0]["name"]

        self.assertEqual(sent, "standard_search")
        self.assertRegex(sent, r"^[a-zA-Z0-9_-]{1,128}$")
        self.assertEqual([call.name for call in turn.tool_calls],
                         ["standard.search"])

    def test_history_replays_a_dotted_call_under_its_wire_name(self):
        """The assistant's own earlier block is validated by the same rule."""
        conversation = [
            ConversationMessage("assistant", (ToolUseBlock(
                "c1", "standard.search", {"query": "x"}),)),
            ConversationMessage("user", (ToolResultBlock("c1", "8.4.1.1-3"),)),
        ]
        backend, client, _ = self.backend(self.message(
            content=(SimpleNamespace(type="text", text="ok"),)))
        backend.complete(system="s", conversation=conversation,
                         tools=[self.tool], use_tools=True)
        block = client.messages.calls[0]["messages"][0]["content"][0]

        self.assertEqual(block["name"], "standard_search")

    def test_empty_text_blocks_never_reach_the_wire(self):
        """A tool-only assistant turn carries an empty text block.

        Anthropic answers that with a 400 before reading anything else, so
        the turn is lost for a block that says nothing.
        """
        conversation = [
            ConversationMessage("assistant", (
                TextBlock(""), ToolUseBlock("c1", "bash", {"command": "ls"}))),
            ConversationMessage("user", (ToolResultBlock("c1", "README.md"),)),
            ConversationMessage("assistant", (TextBlock(""),)),
        ]
        backend, client, _ = self.backend(self.message(
            content=(SimpleNamespace(type="text", text="ok"),)))
        backend.complete(system="s", conversation=conversation,
                         tools=[self.tool], use_tools=True)
        sent = client.messages.calls[0]["messages"]

        self.assertEqual(len(sent), 2)
        self.assertEqual([block["type"] for block in sent[0]["content"]],
                         ["tool_use"])

        for message in sent:
            self.assertTrue(message["content"])

            for block in message["content"]:
                self.assertNotEqual(block.get("text", "unset"), "")

    def test_maps_system_tools_text_streaming_stop_and_usage(self):
        final = self.message(
            content=(SimpleNamespace(type="text", text="done"),),
            usage=SimpleNamespace(input_tokens=12, output_tokens=4),
        )
        backend, client, stream = self.backend(final, deltas=("do", "ne"))
        ticks = []
        turn = backend.complete(system="system", conversation=self.conversation,
                                tools=[self.tool], use_tools=True,
                                on_token=lambda: ticks.append(1))
        self.assertTrue(stream.entered)
        self.assertEqual(turn.text, "done")
        self.assertEqual(turn.stop_reason, StopReason.END_TURN)
        self.assertEqual(turn.usage, {"input_tokens": 12, "output_tokens": 4})
        self.assertEqual(len(ticks), 2)
        kwargs = client.messages.calls[0]
        self.assertEqual(kwargs["model"], "claude-sonnet-5")
        self.assertEqual(kwargs["system"], "system")
        self.assertEqual(kwargs["thinking"], {"type": "disabled"})
        self.assertEqual(kwargs["tools"], [{
            "name": "bash", "description": "Run a command", "input_schema": self.tool.input_schema,
        }])
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": [
            {"type": "text", "text": "Read README"},
        ]}])

    def test_maps_multiple_tool_use_blocks_in_order(self):
        final = self.message(content=(
            SimpleNamespace(type="text", text="I will inspect."),
            SimpleNamespace(type="tool_use", id="a", name="bash", input={"command": "cat README.md"}),
            SimpleNamespace(type="tool_use", id="b", name="bash", input={"command": "grep TODO README.md"}),
        ), stop_reason="tool_use")
        backend, _, _ = self.backend(final)
        turn = backend.complete(system="s", conversation=self.conversation, tools=[self.tool], use_tools=True)
        self.assertEqual(turn.text, "I will inspect.")
        self.assertEqual(turn.stop_reason, StopReason.TOOL_USE)
        self.assertEqual(turn.tool_calls, (
            ModelToolCall("a", "bash", {"command": "cat README.md"}),
            ModelToolCall("b", "bash", {"command": "grep TODO README.md"}),
        ))

    def test_maps_tool_results_success_and_error_without_reordering(self):
        final = self.message(content=(SimpleNamespace(type="text", text="final"),))
        backend, client, _ = self.backend(final)
        conversation = [
            ConversationMessage("user", (TextBlock("do it"),)),
            ConversationMessage("assistant", (
                TextBlock("working"),
                ToolUseBlock("a", "bash", {"command": "one"}),
                ToolUseBlock("b", "bash", {"command": "two"}),
            )),
            ConversationMessage("user", (
                ToolResultBlock("a", "ok", False), ToolResultBlock("b", "failed", True),
            )),
        ]
        backend.complete(system="s", conversation=conversation, tools=[self.tool], use_tools=True)
        messages = client.messages.calls[0]["messages"]
        self.assertEqual(messages[-2]["content"], [
            {"type": "text", "text": "working"},
            {"type": "tool_use", "id": "a", "name": "bash", "input": {"command": "one"}},
            {"type": "tool_use", "id": "b", "name": "bash", "input": {"command": "two"}},
        ])
        self.assertEqual(messages[-1], {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "ok"},
            {"type": "tool_result", "tool_use_id": "b", "content": "failed", "is_error": True},
        ]})

    def test_rejects_malformed_blocks_and_maps_non_normal_stop_reasons(self):
        for block, reason, expected in (
            (SimpleNamespace(type="tool_use", id="a", name="bash", input="not-a-dict"), "tool_use", StopReason.ERROR),
            (SimpleNamespace(type="thinking", thinking="hidden"), "end_turn", StopReason.ERROR),
            (SimpleNamespace(type="text", text="partial"), "max_tokens", StopReason.MAX_TOKENS),
            (SimpleNamespace(type="text", text="pause"), "pause_turn", StopReason.ERROR),
            (SimpleNamespace(type="text", text="refusal"), "refusal", StopReason.REFUSAL),
        ):
            with self.subTest(reason=reason):
                backend, _, _ = self.backend(self.message(content=(block,), stop_reason=reason))
                turn = backend.complete(system="s", conversation=self.conversation, tools=[self.tool], use_tools=True)
                self.assertEqual(turn.stop_reason, expected)
                if expected == StopReason.ERROR:
                    self.assertEqual(turn.tool_calls, ())
                    self.assertTrue(turn.error)

    def test_stop_reason_must_agree_with_complete_tool_calls(self):
        tool = SimpleNamespace(type="tool_use", id="a", name="bash", input={"command": "touch sentinel"})
        cases = (
            ((), "tool_use", "no tool calls"),
            ((tool,), "end_turn", "tool calls"),
            ((tool,), "max_tokens", "tool calls"),
            ((tool,), "refusal", "tool calls"),
        )
        for blocks, reason, message in cases:
            with self.subTest(reason=reason):
                backend, _, _ = self.backend(self.message(content=blocks, stop_reason=reason))
                turn = backend.complete(system="s", conversation=self.conversation, tools=[self.tool], use_tools=True)
                self.assertEqual(turn.stop_reason, StopReason.ERROR)
                self.assertEqual(turn.tool_calls, ())
                self.assertIn(message, turn.error)

    def multi_turn_backend(self, calls):
        """One backend across `calls` turns -- the span a cache lives in."""

        streams = [FakeAnthropicStream((), self.message(
            content=(SimpleNamespace(type="text", text="ok"),))) for _ in range(calls)]
        client = FakeAnthropicClient(streams)

        return AnthropicBackend(client=client, model="claude-sonnet-5"), client

    def send(self, backend, system, *, tools=None, use_tools=True):
        return backend.complete(system=system, conversation=self.conversation,
                                tools=self.tool_block() if tools is None else tools,
                                use_tools=use_tools)

    def tool_block(self, count=40):
        """Enough schemas to clear the minimum cacheable prefix."""

        return [ToolDefinition(
            name=f"tool_{index}", description="A tool with a description long "
            "enough that forty of them are worth a breakpoint.",
            input_schema={"type": "object", "properties": {
                "argument": {"type": "string", "description": "An argument."}}},
        ) for index in range(count)]

    def test_a_first_call_is_sent_exactly_as_before(self):
        # Nothing has repeated yet, so nothing is marked and `system` keeps
        # the plain string shape the adapter always sent.
        backend, client = self.multi_turn_backend(1)
        self.send(backend, "rules\n" * 2000)
        kwargs = client.messages.calls[0]
        self.assertEqual(kwargs["system"], "rules\n" * 2000)
        self.assertNotIn("cache_control", json.dumps(kwargs["tools"]))

    def test_a_repeated_prefix_is_marked_and_the_volatile_tail_is_not(self):
        stable = "system rules\n" * 800
        backend, client = self.multi_turn_backend(2)
        self.send(backend, stable + "working state: round 1\n")
        self.send(backend, stable + "working state: round 2\n")
        blocks = client.messages.calls[1]["system"]
        self.assertEqual([block["text"] for block in blocks],
                         [stable, "working state: round 2\n"])
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", blocks[1])
        # The two blocks still say exactly what the one string said.
        self.assertEqual("".join(block["text"] for block in blocks),
                         stable + "working state: round 2\n")

    def test_repeated_tool_schemas_are_marked_once_at_the_end(self):
        backend, client = self.multi_turn_backend(2)
        self.send(backend, "s")
        self.send(backend, "s")
        tools = client.messages.calls[1]["tools"]
        self.assertNotIn("cache_control", json.dumps(tools[:-1]))
        self.assertEqual(tools[-1]["cache_control"], {"type": "ephemeral"})
        # A marked payload must not become the baseline: a third identical
        # call has to match too.
        backend.client.messages.streams.append(FakeAnthropicStream(
            (), self.message(content=(SimpleNamespace(type="text", text="ok"),))))
        self.send(backend, "s")
        self.assertEqual(client.messages.calls[2]["tools"][-1]["cache_control"],
                         {"type": "ephemeral"})

    def test_a_changed_toolset_drops_the_mark(self):
        backend, client = self.multi_turn_backend(2)
        self.send(backend, "s")
        self.send(backend, "s", tools=self.tool_block(39))
        self.assertNotIn("cache_control", json.dumps(client.messages.calls[1]["tools"]))

    def test_a_short_or_absent_shared_prefix_is_left_unmarked(self):
        for first, second in (("short\n", "short\nmore\n"),
                              ("a" * 9000 + "\n", "b" * 9000 + "\n")):
            with self.subTest(first=first[:6]):
                backend, client = self.multi_turn_backend(2)
                self.send(backend, first)
                self.send(backend, second)
                self.assertEqual(client.messages.calls[1]["system"], second)

    def test_the_boundary_falls_on_a_line_and_never_mid_sentence(self):
        stable = "rule line\n" * 900
        backend, client = self.multi_turn_backend(2)
        self.send(backend, stable + "state alpha")
        self.send(backend, stable + "state beta")
        cached = client.messages.calls[1]["system"][0]["text"]
        # "state " is common to both tails and must not be cached with them.
        self.assertTrue(cached.endswith("\n"))
        self.assertEqual(cached, stable)

    def test_cache_counters_are_reported_when_the_provider_sends_them(self):
        final = self.message(
            content=(SimpleNamespace(type="text", text="ok"),),
            usage=SimpleNamespace(input_tokens=12, output_tokens=4,
                                  cache_creation_input_tokens=0,
                                  cache_read_input_tokens=5100),
        )
        backend, _, _ = self.backend(final)
        turn = backend.complete(system="s", conversation=self.conversation,
                                tools=[self.tool], use_tools=True)
        self.assertEqual(turn.usage, {
            "input_tokens": 12, "output_tokens": 4,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 5100,
        })

    def test_backend_errors_are_controlled(self):
        client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("rate limited"))))
        turn = AnthropicBackend(client=client, model="claude-sonnet-5").complete(
            system="s", conversation=self.conversation, tools=[self.tool], use_tools=True
        )
        self.assertEqual(turn.stop_reason, StopReason.ERROR)
        self.assertIn("Anthropic backend error", turn.error)

    def test_missing_key_and_missing_optional_package_fail_without_network_call(self):
        # No env key AND no resolvable session: the only case that must fail.
        with patch.dict("os.environ", {}, clear=True), \
             patch.object(model_backend, "anthropic_credentials_available",
                          return_value=False):
            with self.assertRaisesRegex(ModelBackendConfigurationError, "ANTHROPIC_API_KEY"):
                AnthropicBackend(model="claude-sonnet-5")
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "controlled-secret"}, clear=True), \
             patch.object(model_backend.importlib, "import_module", side_effect=ImportError):
            with self.assertRaisesRegex(ModelBackendConfigurationError, "anthropic"):
                AnthropicBackend(model="claude-sonnet-5")
        # The OpenAI-compatible path remains constructible independently of Anthropic.
        self.assertIsInstance(OpenAICompatibleBackend(FakeClient([])), OpenAICompatibleBackend)

    def test_credential_detection_delegates_to_the_sdk_chain(self):
        # An env key short-circuits without importing anything.
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k"}, clear=True), \
             patch.object(model_backend.importlib, "import_module",
                          side_effect=AssertionError("must not be consulted")):
            self.assertTrue(model_backend.anthropic_credentials_available())
        # Otherwise the SDK's own resolution chain is the authority, both ways.
        minted = SimpleNamespace(provider=lambda: "token")
        for resolved, expected in ((minted, True), (None, False)):
            chain = SimpleNamespace(default_credentials=lambda r=resolved: r)
            with patch.dict("os.environ", {}, clear=True), \
                 patch.object(model_backend.importlib, "import_module",
                              return_value=chain):
                self.assertIs(model_backend.anthropic_credentials_available(), expected)

        # A profile that resolves but cannot mint (`ant auth logout` leaves the
        # config behind) must read as unavailable, not as a usable session —
        # otherwise the sign-in offer is skipped and the first request fails.
        def _cannot_mint():
            raise RuntimeError("Credentials file not found")

        chain = SimpleNamespace(
            default_credentials=lambda: SimpleNamespace(provider=_cannot_mint))
        with patch.dict("os.environ", {}, clear=True), \
             patch.object(model_backend.importlib, "import_module", return_value=chain):
            self.assertFalse(model_backend.anthropic_credentials_available())
        # A missing SDK is a negative answer, never an exception.
        with patch.dict("os.environ", {}, clear=True), \
             patch.object(model_backend.importlib, "import_module",
                          side_effect=ImportError):
            self.assertFalse(model_backend.anthropic_credentials_available())

    def test_session_without_env_key_constructs_a_bare_client(self):
        built = {}

        class FakeAnthropic:
            def __init__(self, **kwargs):
                built.update(kwargs)

        module = SimpleNamespace(Anthropic=FakeAnthropic)
        # A resolvable session (ant auth login / federation) and no env key:
        # the client must be built bare so the SDK owns refresh.
        with patch.dict("os.environ", {}, clear=True), \
             patch.object(model_backend.importlib, "import_module", return_value=module), \
             patch.object(model_backend, "anthropic_credentials_available",
                          return_value=True):
            AnthropicBackend(model="claude-sonnet-5")
        self.assertEqual(built, {})
        # An explicit key still wins and is passed through unchanged.
        built.clear()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "controlled-secret"}, clear=True), \
             patch.object(model_backend.importlib, "import_module", return_value=module):
            AnthropicBackend(model="claude-sonnet-5")
        self.assertEqual(built, {"api_key": "controlled-secret"})

    def test_login_is_offered_never_automatic_and_fails_closed(self):
        import rag_chat

        # Already authenticated: no prompt, no subprocess.
        with patch.object(rag_chat, "anthropic_credentials_available", return_value=True), \
             patch.object(rag_chat.subprocess, "run",
                          side_effect=AssertionError("must not spawn")):
            rag_chat.offer_anthropic_login()

        # No terminal: fail closed with instructions rather than hang on input().
        with patch.object(rag_chat, "anthropic_credentials_available", return_value=False), \
             patch.object(rag_chat.sys.stdin, "isatty", return_value=False), \
             patch.object(rag_chat.subprocess, "run",
                          side_effect=AssertionError("must not spawn")):
            with self.assertRaisesRegex(ModelBackendConfigurationError, "no terminal"):
                rag_chat.offer_anthropic_login()

        # Declined at the prompt: nothing is spawned.
        with patch.object(rag_chat, "anthropic_credentials_available", return_value=False), \
             patch.object(rag_chat.sys.stdin, "isatty", return_value=True), \
             patch("builtins.input", return_value="2"), \
             patch.object(rag_chat.subprocess, "run",
                          side_effect=AssertionError("must not spawn")):
            with self.assertRaisesRegex(ModelBackendConfigurationError, "declined"):
                rag_chat.offer_anthropic_login()

        # Accepted but the CLI is absent: the message carries the install hint.
        with patch.object(rag_chat, "anthropic_credentials_available", return_value=False), \
             patch.object(rag_chat.sys.stdin, "isatty", return_value=True), \
             patch("builtins.input", return_value=""), \
             patch.object(rag_chat.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ModelBackendConfigurationError, "releases"):
                rag_chat.offer_anthropic_login()

        # Accepted, CLI ran, but no credential resulted: still fail closed.
        with patch.object(rag_chat, "anthropic_credentials_available", return_value=False), \
             patch.object(rag_chat.sys.stdin, "isatty", return_value=True), \
             patch("builtins.input", return_value=""), \
             patch.object(rag_chat.shutil, "which", return_value="/usr/bin/ant"), \
             patch.object(rag_chat.subprocess, "run") as run:
            with self.assertRaisesRegex(ModelBackendConfigurationError, "did not complete"):
                rag_chat.offer_anthropic_login()
        self.assertEqual(run.call_args.args[0], ["/usr/bin/ant", "auth", "login"])

        # Happy path: the login runs unsandboxed and the offer returns.
        answers = iter([False, True])
        with patch.object(rag_chat, "anthropic_credentials_available",
                          side_effect=lambda: next(answers)), \
             patch.object(rag_chat.sys.stdin, "isatty", return_value=True), \
             patch("builtins.input", return_value=""), \
             patch.object(rag_chat.shutil, "which", return_value="/usr/bin/ant"), \
             patch.object(rag_chat.subprocess, "run") as run:
            rag_chat.offer_anthropic_login()
        self.assertEqual(run.call_args.args[0], ["/usr/bin/ant", "auth", "login"])
        self.assertNotIn("bwrap", str(run.call_args))

    def test_rag_cli_keeps_qwen_default_and_selects_anthropic_explicitly(self):
        import rag_chat

        self.assertEqual(rag_chat.model_provider_from_argv([]), ("openai-compatible", None))
        self.assertEqual(rag_chat.model_provider_from_argv(
            ["--provider", "anthropic", "--model", "claude-sonnet-5"]
        ), ("anthropic", "claude-sonnet-5"))
        with self.assertRaisesRegex(ModelBackendConfigurationError, "--provider"):
            rag_chat.model_provider_from_argv(["--provider", "unsupported"])

    def test_agent_loop_gate_executes_only_valid_tool_use_turns(self):
        import rag_chat
        from tool_runtime import ExecutionMode, Workspace

        sentinel = ModelToolCall("write", "write_file", {
            "path": "sentinel.txt", "content": "must not exist",
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_workspace, old_root, old_mode = (
                rag_chat.WORKSPACE, rag_chat.PROJECT_ROOT, rag_chat.EXECUTION_MODE,
            )
            try:
                rag_chat.WORKSPACE = Workspace.from_path(root)
                rag_chat.PROJECT_ROOT = str(root)
                rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
                valid = FakeModelBackend([ModelTurn("", (sentinel,), StopReason.TOOL_USE)])
                turn = valid.complete(system="s", conversation=(), tools=(), use_tools=True)
                self.assertEqual(rag_chat.validate_agent_turn(turn), "tool_use")
                rag_chat.execute_tool(sentinel.name, dict(sentinel.arguments), {})
                self.assertTrue((root / "sentinel.txt").is_file())
                (root / "sentinel.txt").unlink()

                for status in (StopReason.MAX_TOKENS, StopReason.ERROR,
                               StopReason.END_TURN, StopReason.REFUSAL):
                    with self.subTest(status=status):
                        invalid = FakeModelBackend([ModelTurn("", (sentinel,), status, error="provider failure")])
                        turn = invalid.complete(system="s", conversation=(), tools=(), use_tools=True)
                        with self.assertRaisesRegex(RuntimeError, "tool_calls"):
                            rag_chat.validate_agent_turn(turn)
                        self.assertFalse((root / "sentinel.txt").exists())
            finally:
                rag_chat.WORKSPACE, rag_chat.PROJECT_ROOT, rag_chat.EXECUTION_MODE = (
                    old_workspace, old_root, old_mode,
                )


if __name__ == "__main__":
    unittest.main()
