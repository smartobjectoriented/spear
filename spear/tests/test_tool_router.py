import json
import tempfile
import unittest
from pathlib import Path

from agent_runtime import AgentContext, AgentRuntime
from compaction import CompactionPolicy
from context_engine import ContextEngine, ContextItem, ContextLayer
from hooks import HookManager
from model_backend import (
    ConversationMessage, ModelToolCall, ModelTurn, StopReason, TextBlock,
    ToolResultBlock,
)
from result_store import ResultStore
from tool_registry import (
    ToolCategory, ToolMutability, ToolRegistry, ToolResultPolicy, ToolSpec,
)
from tool_router import (
    MAX_FAILED_REPEATS, ToolExecutionContext, ToolHandlerResult,
    ToolResultStatus, ToolRouter, _REPEAT_KEY, action_signature,
    invalidates_reads,
)
from tracing import EventType, TraceEmitter
from working_state import WorkingState


class MemoryRecorder:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


def spec(
    name="probe", *, category=ToolCategory.OTHER,
    mutability=ToolMutability.READ_ONLY, policy=None, modes=None,
):
    # A mutating spec states the modes it runs in, as every real one does:
    # an empty declaration means unrestricted at the router, which for a
    # tool that writes would include safe mode. ToolSpec refuses it, so the
    # fixture says what the tools it stands in for say.

    if modes is None:
        modes = (("ask", "auto")
                 if mutability == ToolMutability.MUTATING else ())

    return ToolSpec(
        name, "Probe", {"type": "object", "properties": {
            "value": {"type": "string"}}, "required": ["value"]},
        category, mutability, execution_modes=modes,
        result_policy=policy or ToolResultPolicy(), handler_key=name,
    )


class RecordingHook:
    def __init__(self, *, fail=False):
        self.events = []
        self.fail = fail

    def before_tool(self, event):
        self.events.append(event.phase)
        if self.fail:
            raise RuntimeError("observer broke")

    def after_tool(self, event):
        self.events.append(event.phase)

    def on_tool_failure(self, event):
        self.events.append(event.phase)


class ToolRouterTests(unittest.TestCase):
    def setUp(self):
        self.recorder = MemoryRecorder()
        self.trace = TraceEmitter(self.recorder)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = ResultStore(self.temporary.name)

    def context(self, *, command_executor=None):
        return ToolExecutionContext(
            "task", self.trace, {}, self.store,
            command_executor=command_executor,
        )

    def router(self, handler=lambda *_: "OK: value", *, tool_spec=None, hooks=None):
        registry = ToolRegistry()
        registered = tool_spec or spec()
        registry.register(registered, handler)
        return ToolRouter(registry, hooks=hooks)

    def test_successful_non_mutating_tool(self):
        result = self.router().execute(
            self.context(), "call", "probe", {"value": "x"})
        self.assertTrue(result.success)
        self.assertEqual(result.status, ToolResultStatus.OK)
        self.assertFalse(result.mutation)

    def test_successful_mutating_tool_and_affected_path(self):
        handler = lambda *_: ToolHandlerResult(
            "OK: changed", mutation=True, affected_paths=("src/a.py",),
        )
        result = self.router(handler, tool_spec=spec(
            mutability=ToolMutability.MUTATING,
        )).execute(self.context(), "call", "probe", {"value": "x"})
        self.assertTrue(result.mutation)
        self.assertEqual(result.affected_paths, ("src/a.py",))

    def test_argument_validation_and_unknown_tool(self):
        router = self.router()
        invalid = router.execute(self.context(), "call", "probe", {})
        malformed = router.execute(self.context(), "bad", "probe", None)
        unknown = router.execute(self.context(), "call", "missing", {})
        self.assertEqual(invalid.status, ToolResultStatus.INVALID_ARGUMENTS)
        self.assertEqual(malformed.status, ToolResultStatus.INVALID_ARGUMENTS)
        self.assertEqual(unknown.status, ToolResultStatus.UNKNOWN_TOOL)
        self.assertFalse(invalid.success)
        self.assertFalse(unknown.success)

    def test_handler_exception_is_structured_failure(self):
        def broken(*_):
            raise ValueError("broken handler")
        result = self.router(broken).execute(
            self.context(), "call", "probe", {"value": "x"})
        self.assertEqual(result.status, ToolResultStatus.FAILED)
        self.assertEqual(result.error_category, "ValueError")
        self.assertIn("broken handler", result.model_content)

    def test_command_fails_closed_without_security_boundary(self):
        registry = ToolRegistry()
        command = ToolSpec(
            "run", "Run", {"type": "object", "properties": {
                "command": {"type": "string"}}, "required": ["command"]},
            ToolCategory.COMMAND, ToolMutability.CONDITIONAL,
        )
        registry.register(command, None)
        result = ToolRouter(registry).execute(
            self.context(), "call", "run", {"command": "pwd"})
        self.assertEqual(result.status, ToolResultStatus.DENIED)
        self.assertIn("boundary unavailable", result.text)

    def test_command_nonzero_and_timeout_remain_structured(self):
        registry = ToolRegistry()
        command = ToolSpec(
            "run", "Run", {"type": "object", "properties": {
                "command": {"type": "string"}}, "required": ["command"]},
            ToolCategory.COMMAND, ToolMutability.CONDITIONAL,
        )
        registry.register(command, None)
        router = ToolRouter(registry)
        failed = router.execute(self.context(command_executor=lambda _: ToolHandlerResult(
            "build failed\n(exit 2)", ToolResultStatus.FAILED, exit_code=2,
            error_category="command_nonzero", error_summary="build failed",
        )), "one", "run", {"command": "make"})
        timeout = router.execute(self.context(command_executor=lambda _: ToolHandlerResult(
            "ERROR: timed out", ToolResultStatus.TIMEOUT,
            error_category="timeout", error_summary="timed out",
        )), "two", "run", {"command": "find ."})
        self.assertEqual(failed.exit_code, 2)
        self.assertEqual(timeout.status, ToolResultStatus.TIMEOUT)

    def test_model_facing_rendering_only_bounds_large_output(self):
        policy = ToolResultPolicy(50, True, 10)
        result = self.router(
            lambda *_: "x" * 500, tool_spec=spec(policy=policy),
        ).execute(self.context(), "call", "probe", {"value": "x"})
        self.assertEqual(result.text, "x" * 500)
        self.assertLess(len(result.model_content), 500)
        self.assertIsNotNone(result.result_reference)

    def test_tracing_has_one_authoritative_lifecycle_pair_and_safe_metadata(self):
        self.router().execute(
            self.context(), "call", "probe", {"value": "secret"})
        lifecycle = [event for event in self.recorder.events if event.event_type in {
            EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_FINISHED,
            EventType.TOOL_CALL_FAILED,
        }]
        self.assertEqual([event.event_type for event in lifecycle], [
            EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_FINISHED,
        ])
        encoded = json.dumps([event.to_dict() for event in lifecycle])
        self.assertNotIn("secret", encoded)
        self.assertIn("output_bytes", encoded)

    def test_hooks_run_in_order_and_failure_is_observer_only(self):
        good = RecordingHook()
        broken = RecordingHook(fail=True)
        manager = HookManager((broken, good))
        result = self.router(hooks=manager).execute(
            self.context(), "call", "probe", {"value": "x"})
        self.assertTrue(result.success)
        self.assertEqual(good.events, ["before_tool", "after_tool"])
        self.assertEqual(len(manager.failures), 1)

    def test_failure_hook_and_sequential_router_reuse(self):
        hook = RecordingHook()
        router = self.router(
            lambda *_: ToolHandlerResult(
                "ERROR: denied", ToolResultStatus.DENIED,
                error_category="denied", error_summary="denied",
            ), hooks=HookManager((hook,)),
        )
        first = router.execute(self.context(), "one", "probe", {"value": "x"})
        second = router.execute(self.context(), "two", "probe", {"value": "y"})
        self.assertFalse(first.success)
        self.assertNotEqual(first.action_id, second.action_id)
        self.assertEqual(hook.events.count("on_tool_failure"), 2)

    def test_registered_command_handler_cannot_bypass_router_boundary(self):
        invoked = []
        registry = ToolRegistry()
        command = ToolSpec(
            "run", "Run", {"type": "object", "properties": {
                "command": {"type": "string"}}, "required": ["command"]},
            ToolCategory.COMMAND, ToolMutability.CONDITIONAL,
        )
        registry.register(command, lambda *_: invoked.append("unsafe") or "unsafe")
        secured = []
        result = ToolRouter(registry).execute(
            self.context(command_executor=lambda command: secured.append(command) or "safe"),
            "call", "run", {"command": "pwd"},
        )
        self.assertEqual(result.text, "safe")
        self.assertEqual(secured, ["pwd"])
        self.assertEqual(invoked, [])

    def test_router_has_no_cli_or_concrete_tool_dependency(self):
        source = Path(__file__).resolve().parents[1].joinpath("tool_router.py").read_text()
        self.assertNotIn("import rag_chat", source)
        self.assertNotIn("print(", source)


class RepeatedFailedActionTests(unittest.TestCase):
    """The same failing call, over and over, until the budget ran out.

    A run made this call twelve times, verbatim, and nothing stopped it:
    fetch_url({"url": "https://example.com/main.py", "save_as": "main.py"}).
    Dedup existed only for bash, and only for commands that had succeeded.
    """

    ARGUMENTS = {"url": "https://example.com/main.py", "save_as": "main.py"}

    def router(self):
        registry = ToolRegistry()
        self.calls = []

        def handler(context, arguments):
            self.calls.append(dict(arguments))

            return ToolHandlerResult(
                "ERROR: fetch failed", ToolResultStatus.FAILED,
                error_category="fetch_error", error_summary="fetch failed")

        registry.register(
            ToolSpec("fetch_url", "Fetch one URL",
                     {"type": "object",
                      "properties": {"url": {"type": "string"},
                                     "save_as": {"type": "string"}},
                      "required": ["url"]},
                     ToolCategory.WEB, ToolMutability.READ_ONLY),
            handler)

        return ToolRouter(registry)

    def context(self, recorder):
        return ToolExecutionContext(
            task_id="task_repeat", trace=TraceEmitter(recorder), cache={})

    def test_only_two_identical_failures_reach_the_tool(self):
        router = self.router()
        recorder = MemoryRecorder()
        context = self.context(recorder)
        envelopes = [router.execute(context, f"call{index}", "fetch_url",
                                    dict(self.ARGUMENTS)) for index in range(3)]

        self.assertEqual(len(self.calls), 2, "the third must not run")
        self.assertTrue(all(not item.success for item in envelopes))
        self.assertEqual(envelopes[2].status, ToolResultStatus.DENIED)
        self.assertTrue(envelopes[2].metadata["suppressed_repeat"])
        self.assertEqual(envelopes[2].metadata["repeat_count"], 3)

        # And it says so, in a result the model can act on.

        self.assertIn("already failed", envelopes[2].text)
        self.assertIn("edit_file", envelopes[2].text)

        repeats = [item for item in recorder.events
                   if item.event_type == EventType.REPEATED_ACTION_DETECTED]
        self.assertEqual(len(repeats), 1)
        self.assertEqual(repeats[0].metadata["kind"], "failed_action_repeated")
        self.assertNotIn("main.py", json.dumps(
            [item.to_dict() for item in recorder.events]))

    def test_asking_a_fourth_time_is_counted_for_the_caller_to_stop_on(self):
        router = self.router()
        context = self.context(MemoryRecorder())

        for index in range(4):
            envelope = router.execute(context, f"call{index}", "fetch_url",
                                      dict(self.ARGUMENTS))

        self.assertEqual(len(self.calls), 2)
        self.assertGreater(envelope.metadata["repeat_count"], MAX_FAILED_REPEATS + 1)

    def test_a_different_argument_is_a_different_action(self):
        router = self.router()
        context = self.context(MemoryRecorder())

        for index in range(3):
            router.execute(context, f"call{index}", "fetch_url",
                           {"url": f"https://example.com/{index}.py"})

        self.assertEqual(len(self.calls), 3)

    def test_argument_order_does_not_disguise_a_repeat(self):
        router = self.router()
        context = self.context(MemoryRecorder())
        router.execute(context, "a", "fetch_url",
                       {"url": "https://x/y", "save_as": "y"})
        router.execute(context, "b", "fetch_url",
                       {"save_as": "y", "url": "https://x/y"})
        envelope = router.execute(context, "c", "fetch_url",
                                  {"url": "https://x/y", "save_as": "y"})

        self.assertEqual(len(self.calls), 2)
        self.assertTrue(envelope.metadata["suppressed_repeat"])

    def test_a_call_that_works_clears_its_own_history(self):
        registry = ToolRegistry()
        outcomes = ["ERROR: nope", "ERROR: nope", "OK: done", "ERROR: nope"]
        seen = []

        def handler(context, arguments):
            seen.append(dict(arguments))
            text = outcomes.pop(0)

            return ToolHandlerResult(
                text, ToolResultStatus.FAILED if text.startswith("ERROR")
                else ToolResultStatus.OK)

        registry.register(
            ToolSpec("probe", "Probe", {"type": "object", "properties": {}},
                     ToolCategory.RETRIEVAL, ToolMutability.READ_ONLY),
            handler)
        router = ToolRouter(registry)
        context = self.context(MemoryRecorder())

        for index in range(2):
            router.execute(context, f"f{index}", "probe", {})

        self.assertEqual(len(seen), 2)

        # Suppressed, then the counter is cleared by a success... which this
        # third call never gets to have. The success has to come from a call
        # that RAN, so the history is cleared by hand for the next stretch.

        context.cache[_REPEAT_KEY]["failed"].clear()
        router.execute(context, "ok", "probe", {})
        self.assertEqual(len(seen), 3)
        router.execute(context, "again", "probe", {})
        self.assertEqual(len(seen), 4, "a success resets the count")



class CachedActionTests(unittest.TestCase):
    """A cache hit is an action that obtained nothing.

    A read-only turn ran `cat main.py` once and was handed it back from cache
    six more times: nineteen model calls, eighteen tool calls, 316k context
    tokens, and the correct answer arriving only as the budget ran out. The
    breaker counted failures; a call that succeeds and then says nothing new
    is the same loop with a better status code.
    """

    CONTENT = "def add(a, b):\n    return a - b\n"
    NEW_CONTENT = "def add(a, b):\n    return a + b   # half-written\n"

    def setUp(self):
        self.registry = ToolRegistry()
        self.ran = []
        self.registry.register(
            ToolSpec("bash", "Run", {"type": "object", "properties": {
                "command": {"type": "string"}}, "required": ["command"]},
                ToolCategory.COMMAND, ToolMutability.CONDITIONAL),
            None)
        self.registry.register(
            ToolSpec("edit_file", "Edit", {"type": "object", "properties": {
                "path": {"type": "string"}}, "required": ["path"]},
                ToolCategory.FILE_WRITE, ToolMutability.MUTATING,
                execution_modes=("ask", "auto")),
            self.edit)
        self.router = ToolRouter(self.registry)
        self.recorder = MemoryRecorder()
        self.denied = set()
        self.failing = {}
        self.content = self.CONTENT
        self.context = ToolExecutionContext(
            task_id="task_cache", trace=TraceEmitter(self.recorder), cache={},
            command_executor=self.command)

    def edit(self, context, arguments):
        self.ran.append(("edit_file", arguments["path"]))

        # What rag_chat does on every mutation: the read cache is now stale.

        for key in [key for key in self.context.cache if isinstance(key, str)]:
            del self.context.cache[key]

        return ToolHandlerResult("OK: updated", mutation=True,
                                 affected_paths=(arguments["path"],))

    def command(self, command):
        """The real shape of rag_chat's boundary, cache and all."""

        if command in self.denied:
            # Refused before it ran. It wrote nothing, so it forgets nothing.

            return ToolHandlerResult(
                "ERROR: command denied", ToolResultStatus.DENIED,
                error_category="permission_denied")

        if command in self.failing:
            self.ran.append(("bash", command))
            unmeasured = self.failing[command]
            metadata = ({"potentially_mutating": True} if unmeasured
                        else {"mutation_count": 0})

            # rag_chat drops its read cache under exactly the predicate the
            # router uses for its generation, so the fake does the same.

            if invalidates_reads(ToolResultStatus.FAILED, False, metadata):
                for key in [key for key in self.context.cache
                            if isinstance(key, str)]:
                    del self.context.cache[key]

                self.content = self.NEW_CONTENT

            return ToolHandlerResult("ERROR: exit 1", ToolResultStatus.FAILED,
                                     exit_code=1, metadata=metadata)

        if command in self.context.cache:
            return ToolHandlerResult(
                "(ALREADY EXECUTED this turn — its output is already above in "
                "this conversation.)\n", ToolResultStatus.CACHED)

        # Only here: this is the command actually running.

        self.ran.append(("bash", command))
        self.context.cache[command] = self.content

        return ToolHandlerResult(self.content)

    def call(self, index, command="cat main.py"):
        return self.router.execute(self.context, f"c{index}", "bash",
                                   {"command": command})

    def refuse(self, command):
        """A command the policy turned down: it never ran, so it wrote nothing."""

        self.denied.add(command)

        return self.router.execute(self.context, f"d{command}", "bash",
                                   {"command": command})

    def fail_command(self, command, *, wrote=True):
        """A command that ran and came back non-zero.

        ``wrote`` says whether anything measured what it changed: bash cannot,
        so it declares itself potentially mutating; a handler that can says
        ``mutation_count`` and is believed.
        """

        self.failing[command] = wrote

        return self.router.execute(self.context, f"f{command}", "bash",
                                   {"command": command})

    def visible(self, marker):
        return marker not in self.compacted

    def test_evidence_lost_to_compaction_is_restored_not_pointed_at(self):
        """"Already above in this conversation" is only true while it is.

        Compaction rewrites the conversation and the composer decides what
        each request carries, so an observation the harness once delivered
        can be somewhere the model cannot read. Telling it to look there
        leaves it one move: another spelling of the same read, which is the
        guard rail defeated rather than respected.
        """

        self.compacted = set()
        self.context.evidence_available = self.visible

        first = self.call(0)
        self.assertIn(self.CONTENT, first.model_content)

        # Still visible: the repeat carries the pointer and nothing else.

        second = self.call(1)
        self.assertEqual(second.status, ToolResultStatus.CACHED)
        self.assertNotIn(self.CONTENT, second.model_content)
        self.assertFalse(second.metadata.get("rehydrated_evidence"))

        # Compaction takes the original result block out of the context.

        self.compacted.add("c0")

        third = self.call(2)

        self.assertNotEqual(third.status, ToolResultStatus.DENIED)
        self.assertIn(self.CONTENT, third.model_content,
                      "the model must get the content back, not a pointer")
        self.assertNotIn("already above", third.model_content)
        self.assertTrue(third.metadata["rehydrated_evidence"])

        # Restoring lost evidence is not progress: the repetition still
        # counts, and the counter was not reset by it.

        self.assertEqual(third.status, ToolResultStatus.CACHED)
        ledger = self.context.cache[_REPEAT_KEY]
        self.assertEqual(
            ledger["cached"][action_signature(
                "bash", {"command": "cat main.py"}, ledger["generation"])], 2)

        repeats = [item for item in self.recorder.events
                   if item.event_type == EventType.REPEATED_ACTION_DETECTED]
        self.assertEqual(len(repeats), 2)

    def test_a_refusal_is_never_issued_about_evidence_that_is_gone(self):
        """The suppression says the observation is in the context. It must be."""

        self.compacted = set()
        self.context.evidence_available = self.visible

        self.call(0)
        self.call(1)
        self.compacted.add("c0")
        third = self.call(2)

        self.assertNotEqual(third.status, ToolResultStatus.DENIED)
        self.assertNotIn("already present in the context", third.text)

        # Back in the context, so the next identical call is refused again.

        self.compacted.clear()
        fourth = self.call(3)
        self.assertEqual(fourth.status, ToolResultStatus.DENIED)
        self.assertIn("already present in the context", fourth.text)

    def test_a_cached_answer_is_re_offered_once_and_then_refused(self):
        first = self.call(0)
        second = self.call(1)
        third = self.call(2)

        self.assertTrue(first.success)
        self.assertEqual(second.status, ToolResultStatus.CACHED)
        self.assertEqual(third.status, ToolResultStatus.DENIED)
        self.assertTrue(third.metadata["suppressed_repeat"])
        self.assertEqual(third.metadata["repeat_kind"], "cached_action_repeated")
        self.assertFalse(third.metadata["terminal_repeat"])
        self.assertIn("already present in the context", third.text)
        self.assertIn("no state has changed", third.text)
        self.assertIn("answer the user", third.text)

        # The tool ran twice; and while the original result is still in the
        # active context, the file's content is put in front of the model
        # exactly once. That is the property -- not "once per turn": evidence
        # compaction has taken away is restored, and
        # test_evidence_lost_to_compaction_is_restored_not_pointed_at is where
        # that case lives.

        self.assertEqual(self.ran, [("bash", "cat main.py")],
                         "the command itself ran once")
        injected = [item.model_content for item in (first, second, third)]
        self.assertEqual(sum(self.CONTENT in text for text in injected), 1)

        kinds = [item.metadata.get("kind") for item in self.recorder.events
                 if item.event_type == EventType.REPEATED_ACTION_DETECTED]
        self.assertEqual(kinds, ["cached_tool_call", "cached_action_repeated"])

    def test_asking_a_fourth_time_is_marked_terminal(self):
        for index in range(3):
            self.call(index)

        fourth = self.call(3)

        self.assertTrue(fourth.metadata["terminal_repeat"])
        self.assertEqual(fourth.metadata["repeat_kind"], "cached_action_repeated")

    def test_reading_a_file_again_after_writing_it_is_allowed(self):
        self.call(0)
        self.call(1)
        self.router.execute(self.context, "edit", "edit_file", {"path": "main.py"})

        # A new generation, so the read is a new action; the caller dropped
        # its stale cache, so it runs for real and returns the new text.

        again = self.call(2)

        self.assertTrue(again.success)
        self.assertNotEqual(again.status, ToolResultStatus.DENIED)
        self.assertIn(self.CONTENT, again.model_content)
        self.assertEqual(self.ran.count(("bash", "cat main.py")), 2)

    def test_the_ledger_survives_everything_that_is_not_a_write(self):
        """The observed control-doc run, interleaved exactly as it was.

        `cat main.py` at steps 5, 11, 27 and 35; `grep`, `sed -n` and `cat -n`
        between them; a refused `sed -i` at step 4. Every one of those `cat`s
        came back `ok` instead of `cached`, so nothing ever counted them:
        thirty-nine model calls, 797k cumulative tokens, to answer which file
        defines `add`. Not being consecutive is not progress.
        """

        self.call(0)                                   # cat main.py  -> ran
        self.call(1, command="grep -n add main.py")    # another read
        self.refuse("sed -i 's/a/b/' main.py")         # refused: wrote nothing
        second = self.call(2)                          # cat main.py  -> cached
        self.call(3, command="sed -n '1,3p' main.py")  # another read
        third = self.call(4)                           # cat main.py  -> refused

        self.assertEqual(second.status, ToolResultStatus.CACHED)
        self.assertEqual(third.status, ToolResultStatus.DENIED)
        self.assertEqual(third.metadata["repeat_kind"], "cached_action_repeated")

        # The real executor saw `cat main.py` exactly once.

        self.assertEqual(self.ran.count(("bash", "cat main.py")), 1)

    def generation(self):
        return self.context.cache[_REPEAT_KEY]["generation"]

    def test_a_half_written_file_is_a_new_generation_whatever_the_exit_code(self):
        """The exit code is not the question; what was written is.

        The two rules drifted: the read cache dropped its reads for any
        executed non-read-only command while the generation moved only on
        success. A command that wrote half a file and exited 1 left the cache
        empty and the generation unchanged -- so the fresh read that followed
        carried the same registry key as the stale one, and could be served
        from it or refused as a repetition of it.
        """

        first = self.call(0)
        self.assertIn(self.CONTENT, first.model_content)
        self.assertEqual(self.generation(), 0)

        # It ran, it wrote, it exited 1.

        self.fail_command("sed -i 's/-/+/' main.py && false")

        self.assertEqual(self.generation(), 1)

        again = self.call(1)

        self.assertTrue(again.success)
        self.assertNotEqual(again.status, ToolResultStatus.CACHED)
        self.assertNotEqual(again.status, ToolResultStatus.DENIED)
        self.assertFalse(again.metadata.get("suppressed_repeat"))
        self.assertIn(self.NEW_CONTENT, again.model_content)
        self.assertEqual(self.ran.count(("bash", "cat main.py")), 2)

    def test_a_failure_that_measured_no_change_keeps_the_generation(self):
        """`mutation_count == 0` is a measurement, and it is believed."""

        self.call(0)
        self.fail_command("check-nothing", wrote=False)

        self.assertEqual(self.generation(), 0)

        # The earlier read is still usable: same generation, same identity.

        again = self.call(1)
        self.assertEqual(again.status, ToolResultStatus.CACHED)
        self.assertEqual(self.ran.count(("bash", "cat main.py")), 1)

    def test_a_refused_command_invalidates_nothing(self):
        """It never ran, so `mutation_count` is zero and nothing is stale."""

        self.call(0)
        self.refuse("sed -i 's/a/b/' main.py")

        self.assertEqual(self.generation(), 0)
        self.assertEqual(self.call(1).status, ToolResultStatus.CACHED)
        self.assertEqual(self.ran.count(("bash", "cat main.py")), 1)

    def test_the_cache_and_the_ledger_never_disagree(self):
        """One predicate decides both, so there is no state where they differ."""

        for status, mutation, metadata in (
            (ToolResultStatus.DENIED, False, {"potentially_mutating": True}),
            (ToolResultStatus.CANCELLED, False, {"potentially_mutating": True}),
            (ToolResultStatus.FAILED, False, {"potentially_mutating": True}),
            (ToolResultStatus.FAILED, False, {"mutation_count": 0}),
            (ToolResultStatus.FAILED, False, {"mutation_count": 3}),
            (ToolResultStatus.OK, True, {}),
            (ToolResultStatus.OK, False, {"mutation_count": 0}),
            (ToolResultStatus.CACHED, False, {}),
        ):
            with self.subTest(status=status, metadata=metadata):
                decided = invalidates_reads(status, mutation, metadata)
                self.assertIs(decided,
                              invalidates_reads(status, mutation, metadata))
                self.assertIsInstance(decided, bool)

        self.assertFalse(invalidates_reads(
            ToolResultStatus.DENIED, True, {"mutation_count": 9}),
            "a call that never ran wrote nothing, whatever it claims")

    def test_a_different_file_is_a_different_action(self):
        self.call(0)
        self.call(1)
        other = self.call(2, command="cat other.py")

        self.assertTrue(other.success)
        self.assertNotEqual(other.status, ToolResultStatus.DENIED)


class ScriptedBackend:
    model = "fake-local"
    max_tokens = 64

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.turns.pop(0)


class ToolRouterRuntimeIntegrationTests(unittest.TestCase):
    def test_runtime_uses_structured_results_for_state_context_and_storage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = ToolRegistry()
            registry.register(spec("read"), lambda *_: ToolHandlerResult(
                "contents", read_paths=("input.txt",),
            ))
            registry.register(spec(
                "write", mutability=ToolMutability.MUTATING,
            ), lambda *_: (
                root.joinpath("out.txt").write_text("value")
                and ToolHandlerResult(
                    "OK: wrote", mutation=True, affected_paths=("out.txt",),
                    metadata={"created_paths": ("out.txt",)},
                )
            ))
            registry.register(ToolSpec(
                "run", "Run", {"type": "object", "properties": {
                    "command": {"type": "string"}}, "required": ["command"]},
                ToolCategory.COMMAND, ToolMutability.CONDITIONAL,
            ), None)
            registry.register(spec("fail"), lambda *_: ToolHandlerResult(
                "ERROR: expected", ToolResultStatus.FAILED,
                error_category="expected", error_summary="expected",
            ))
            registry.register(spec(
                "large", policy=ToolResultPolicy(40, True, 10),
            ), lambda *_: "large-output-" * 100)
            recorder = MemoryRecorder()
            trace = TraceEmitter(recorder)
            store = ResultStore(root / "results")
            router = ToolRouter(registry)
            secure_commands = []

            def execute(agent_context, call_id, name, arguments, cache):
                execution = ToolExecutionContext(
                    agent_context.task_id, trace, cache, store,
                    command_executor=lambda command: (
                        secure_commands.append(command) or ToolHandlerResult("command ok")
                    ),
                )
                return router.execute(execution, call_id, name, arguments)

            turns = [
                ModelTurn("", (ModelToolCall("r", "read", {"value": "x"}),), StopReason.TOOL_USE),
                ModelTurn("", (ModelToolCall("w", "write", {"value": "x"}),), StopReason.TOOL_USE),
                ModelTurn("", (ModelToolCall("c", "run", {"command": "pwd"}),), StopReason.TOOL_USE),
                ModelTurn("", (ModelToolCall("f", "fail", {"value": "x"}),), StopReason.TOOL_USE),
                ModelTurn("", (ModelToolCall("l", "large", {"value": "x"}),), StopReason.TOOL_USE),
                ModelTurn("done", (), StopReason.END_TURN),
            ]
            state = WorkingState.start("task", "exercise tools", max_model_rounds=8,
                                       max_tool_actions=8)
            context = AgentContext(
                state, ScriptedBackend(turns), ContextEngine(), trace, "system",
                (ContextItem("system", ContextLayer.SYSTEM_RULES, "test", "system",
                             protected=True, inclusion_reason="test"),),
                [ConversationMessage("user", (TextBlock("go"),))],
                registry.definitions_for_model(), execute, 8, 8, 8192,
                output_reserve=64,
                compaction_policy=CompactionPolicy(minimum_compactable_tokens=10000),
            )
            result = AgentRuntime().run(context)
            self.assertEqual(result.final_response, "done")
            self.assertEqual(state.files_read, {"input.txt"})
            self.assertEqual(state.created_files, {"out.txt"})
            self.assertEqual(secure_commands, ["pwd"])
            self.assertEqual(len(state.unresolved_failures), 1)
            large = next(step for step in result.trajectory if step["tool"] == "large")
            self.assertTrue(store.exists(large["result_reference"]))
            large_block = next(
                block for message in context.conversation for block in message.content
                if isinstance(block, ToolResultBlock) and block.tool_call_id == "l"
            )
            self.assertIn("full result reference", large_block.content)
            self.assertTrue(root.joinpath("out.txt").is_file())


class ProductionCommandBoundaryTests(unittest.TestCase):
    def test_registered_bash_still_crosses_command_runner_and_sandbox_preflight(self):
        from unittest.mock import patch
        import rag_chat
        from tool_runtime import ExecutionMode, ToolResult, Workspace

        with tempfile.TemporaryDirectory() as temporary:
            old_workspace = rag_chat.WORKSPACE
            old_root = rag_chat.PROJECT_ROOT
            old_mode = rag_chat.EXECUTION_MODE
            try:
                rag_chat.WORKSPACE = Workspace.from_path(temporary)
                rag_chat.PROJECT_ROOT = temporary
                rag_chat.EXECUTION_MODE = ExecutionMode.SAFE
                with patch.object(
                    rag_chat.COMMAND_RUNNER, "ensure_sandbox",
                    return_value=ToolResult("ok", "sandbox available"),
                ) as preflight, patch.object(
                    rag_chat.COMMAND_RUNNER, "run_sandboxed",
                    return_value=ToolResult("ok", "done", stdout="secured\n"),
                ) as run:
                    envelope = rag_chat.route_tool_envelope(
                        "bash", {"command": "pwd"}, {}, task_id="security",
                        trace=TraceEmitter(), tool_call_id="call",
                    )
            finally:
                rag_chat.WORKSPACE = old_workspace
                rag_chat.PROJECT_ROOT = old_root
                rag_chat.EXECUTION_MODE = old_mode
        self.assertTrue(envelope.success)
        self.assertEqual(envelope.text, "secured\n")
        self.assertEqual(envelope.stdout, "secured\n")
        self.assertEqual(envelope.stderr, "")
        preflight.assert_called_once()
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
