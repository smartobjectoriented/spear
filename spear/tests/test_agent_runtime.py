import json
import time
import tempfile
import unittest
from pathlib import Path

from agent_runtime import (
    AgentContext, AgentRuntime, RuntimeTerminalReason, _asked,
    _record_project_verification, conclude_demand, is_write_request,
    may_demand_write, wants_write,
)
from compaction import (
    CompactionMode, CompactionPolicy, CompactionRequest, CompactionService,
)
from budgets import BudgetKind, BudgetLimit, BudgetManager
from checkpoint import CheckpointManager, RollbackStatus
from context_engine import (
    ContextEngine, ContextItem, ContextLayer, ContextRequest, Freshness,
)
from model_backend import (
    ConversationMessage, ModelToolCall, ModelTurn, StopReason, TextBlock,
    ToolDefinition,
)
import project_build
from tracing import EventType, TraceEmitter
from session_store import (
    FileSessionStore, SessionConfiguration, SessionHandle, new_session_id,
)
from tool_router import ToolResultEnvelope, ToolResultStatus
from working_state import (
    ActionKind, StateEventType, StateSource, TerminalStatus, WorkingState,
)
from verification import CompletionVerificationStatus


class MemoryRecorder:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


class ScriptedBackend:
    model = "scripted-local"
    max_tokens = 64

    def __init__(self, turns, *, compaction_output=None, compaction_error=None):
        self.turns = list(turns)
        self.calls = []
        self.compaction_output = compaction_output or json.dumps(
            {"narrative_summary": "Earlier context summarized."}
        )
        self.compaction_error = compaction_error

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if "compact agent context" in kwargs["system"]:
            if self.compaction_error:
                raise self.compaction_error
            return ModelTurn(self.compaction_output, (), StopReason.END_TURN)
        if not self.turns:
            raise AssertionError("unexpected primary model call")
        turn = self.turns.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        return turn


class CountingContextEngine(ContextEngine):
    def __init__(self):
        super().__init__()
        self.compose_count = 0

    def compose(self, request):
        self.compose_count += 1
        return super().compose(request)


class RecordingObserver:
    class Activity:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return None

    def __init__(self):
        self.text = []
        self.notices = []
        self.activities = []

    def model_activity(self, label):
        self.activities.append(label)

        return self.Activity()

    def intermediate_text(self, text):
        self.text.append(text)

    def notice(self, kind, metadata):
        self.notices.append((kind, dict(metadata)))


class GroundedToolExecutor:
    def __init__(self, outputs=None):
        self.outputs = list(outputs or [])
        self.calls = []

    def __call__(self, context, tool_call_id, name, arguments, cache):
        self.calls.append((context.task_id, name, dict(arguments)))
        key = arguments.get("command") if name == "bash" else None
        cached = name == "bash" and key in cache
        if cached:
            result = "(ALREADY EXECUTED this turn — same result repeated below.)\n" + cache[key]
        else:
            result = self.outputs.pop(0) if self.outputs else "OK: done"
            if name == "bash":
                cache[key] = result
        action_id = f"tool_{len(self.calls)}_{context.task_id}"
        failed = result.startswith("ERROR") or "(exit 2)" in result
        status = (ToolResultStatus.CACHED if cached else
                  ToolResultStatus.FAILED if failed else ToolResultStatus.OK)
        affected = ((str(arguments.get("path")),)
                    if name in {"edit_file", "write_file", "append_file"}
                    and not failed else ())
        return ToolResultEnvelope(
            tool_call_id, action_id, name, not failed, status, result, result,
            "command" if name == "bash" else "file_write", 0.001,
            len(result.encode()), exit_code=2 if "(exit 2)" in result else None,
            mutation=bool(affected), affected_paths=affected,
            error_category=("command_nonzero" if failed and name == "bash"
                            else "tool_error" if failed else None),
            error_summary=result.splitlines()[0] if failed else None,
        )


def text_turn(text="done", status=StopReason.END_TURN):
    return ModelTurn(text, (), status)


def tool_turn(call_id, name, **arguments):
    return ModelTurn(
        "", (ModelToolCall(call_id, name, arguments),), StopReason.TOOL_USE,
    )


def base_items(state, *, extra=()):
    return (
        ContextItem(
            "system", ContextLayer.SYSTEM_RULES, "test", "SYSTEM",
            priority=100, freshness=Freshness.CURRENT, protected=True,
            inclusion_reason="test system",
        ),
    ) + tuple(extra)


def make_context(
    backend, *, task_id="task_runtime", executor=None, rounds=5, actions=10,
    engine=None, trace=None, conversation=None, context_limit=4096,
    policy=None, extra_items=(), observer=None,
):
    state = WorkingState.start(
        task_id, "Complete the scripted task",
        max_model_rounds=rounds, max_tool_actions=actions,
    )
    return AgentContext(
        working_state=state,
        backend=backend,
        context_engine=engine or ContextEngine(),
        trace=trace or TraceEmitter(),
        system_prompt="SYSTEM",
        context_items=base_items(state, extra=extra_items),
        conversation=list(conversation or (
            ConversationMessage("user", (TextBlock("do it"),)),
        )),
        tools=(ToolDefinition("bash", "run", {"type": "object"}),),
        tool_executor=executor or GroundedToolExecutor(),
        max_model_rounds=rounds,
        max_tool_actions=actions,
        context_limit=context_limit,
        output_reserve=64,
        safety_margin=16,
        compaction_policy=policy or CompactionPolicy(
            pressure_threshold=.95, minimum_compactable_tokens=10000,
        ),
        observer=observer or RecordingObserver(),
    )

class DenyingToolExecutor(GroundedToolExecutor):
    """A sandbox that refuses the calls named in `deny`."""

    def __init__(self, deny=(), outputs=None):
        super().__init__(outputs)
        self.deny = set(deny)

    def __call__(self, context, tool_call_id, name, arguments, cache):
        if arguments.get("command") in self.deny:
            self.calls.append((context.task_id, name, dict(arguments)))

            return ToolResultEnvelope(
                tool_call_id, f"tool_{len(self.calls)}_{context.task_id}", name,
                False, ToolResultStatus.DENIED,
                "ERROR: command denied: not allowlisted",
                "ERROR: command denied: not allowlisted",
                "command", 0.001, 40,
            )

        return super().__call__(context, tool_call_id, name, arguments, cache)



class ABudgetMustNotLeaveTheTreeBroken(unittest.TestCase):
    """The third exit the build gate could not reach.

    A master run made the right edit at minute nineteen of thirty, left a
    symbol undefined, and the budget ended the turn with ten minutes to
    spare and nothing asking it to finish.
    """

    def context(self, backend):
        context = make_context(backend, executor=GroundedToolExecutor(),
                               rounds=10, actions=20)
        context.project_commands = project_build.ProjectCommands("false", "",
                                                                 "test")
        context.project_root = "."
        # Room for one edit, then spent -- a budget that stops before any
        # work happened cannot demonstrate a tree left broken by the work.
        context.budget_manager = BudgetManager("task", {
            BudgetKind.TOOL_CALLS: BudgetLimit(1),
        })

        return context

    def test_the_repair_is_asked_for_outside_the_spent_budget(self):
        backend = ScriptedBackend([
            tool_turn("one", "edit_file", path="a.c", old_text="x",
                      new_text="y"),
            tool_turn("two", "edit_file", path="a.c", old_text="y",
                      new_text="z"),
            text_turn("done"),
        ])
        observer = RecordingObserver()
        context = self.context(backend)
        context.observer = observer
        AgentRuntime().run(context)
        after = [metadata.get("after") for kind, metadata in observer.notices
                 if kind == "project_build_failed"]

        self.assertIn("budget", after,
                      "a spent budget must still ask for the repair")

    def test_a_tree_that_builds_is_not_asked_for_anything(self):
        backend = ScriptedBackend([
            tool_turn("one", "edit_file", path="a.c", old_text="x",
                      new_text="y"),
            text_turn("done"),
        ])
        observer = RecordingObserver()
        context = self.context(backend)
        context.observer = observer
        context.project_commands = project_build.ProjectCommands("true", "true",
                                                                 "test")
        AgentRuntime().run(context)

        self.assertNotIn("project_build_failed",
                         [kind for kind, _ in observer.notices])


class RoomToWorkIsNotRoomToRead(unittest.TestCase):
    """A bigger budget must not buy more reading before the work starts."""

    def test_the_reading_allowance_is_capped_not_merely_scaled(self):
        """Measured: the ceiling rose to 500, the nudge moved to a hundred
        rounds, and a master run read for thirty minutes and wrote nothing."""
        import agent_runtime

        self.assertEqual(agent_runtime._investigation_ceiling(60), 12)
        self.assertEqual(agent_runtime._investigation_ceiling(120), 24)
        self.assertEqual(agent_runtime._investigation_ceiling(500),
                         agent_runtime.INVESTIGATION_CEILING)
        self.assertLess(agent_runtime._investigation_ceiling(5000), 100)


class ASpentBudgetIsNotAModelError(unittest.TestCase):
    """Retrying a stop is not a retry.

    BudgetExceeded is a RuntimeError, so the model-failure handler took it
    for one and retried -- against a budget that raises on every charge.
    A turn ran an hour past its wall clock.
    """

    def test_a_wall_clock_stop_ends_the_turn_rather_than_retrying(self):
        backend = ScriptedBackend([tool_turn(str(i), "bash", command=f"echo {i}")
                                   for i in range(6)] + [text_turn("done")] * 3)
        context = make_context(backend, executor=GroundedToolExecutor(),
                               rounds=10, actions=20)
        context.budget_manager = BudgetManager("task", {
            BudgetKind.WALL_TIME_MS: BudgetLimit(1),
            BudgetKind.TOOL_CALLS: BudgetLimit(50),
        })
        time.sleep(0.01)
        result = AgentRuntime().run(context)

        self.assertTrue(result.budget_exhausted)
        self.assertLessEqual(len(backend.calls), 3,
                             "a spent budget must not be retried into a loop")


class TheVerbsThatMeanChangeTheCode(unittest.TestCase):
    """"can you adapt the code accordingly" was not read as a write request.

    Every write-side gate stayed silent for a whole turn because of it: the
    model looped on reads and nothing asked it for the edit.
    """

    WRITES = ("can you adapt the code accordingly",
              "adapte le code en consequence",
              "please adjust the encoder",
              "rework the ack path",
              "can you validate and make changes in the code accordingly")

    ASKS = ("How should the ACK be managed?",
            "what does the standard say about warnings?",
            "explain the CAM field")

    def test_the_write_verbs_are_recognised(self):
        for text in self.WRITES:
            with self.subTest(text=text):
                self.assertTrue(is_write_request(text))

    def test_a_question_is_still_a_question(self):
        for text in self.ASKS:
            with self.subTest(text=text):
                self.assertFalse(is_write_request(text))


class EveryGateReachableFromEveryExit(unittest.TestCase):
    """Four times now a gate has turned out to be unreachable from one exit.

    A turn wrote the one field the change needs, stalled, and ended: the
    tree compiled so the build gate was silent, something had been written
    so the write redirect was silent, and the clause redirect lived only at
    the answer boundary, which a stall never reaches.

    Every turn here is an implementation turn, asked for in the user's own
    words: the clauses a session established are the obligations of a turn
    that was asked to change the code, and of no other. `test_task_scope`
    holds the other half.
    """

    ASKED = [ConversationMessage(
        "user", (TextBlock("implement the acknowledgement rules in a.c"),))]

    def test_a_stall_after_writing_still_asks_for_the_clauses(self):
        backend = ScriptedBackend([
            tool_turn("one", "edit_file", path="a.c", old_text="x",
                      new_text="y"),
        ] + [tool_turn(str(i), "bash", command="grep -n x a.c")
             for i in range(2, 8)] + [text_turn("done")] * 4)
        observer = RecordingObserver()
        context = make_context(backend, executor=GroundedToolExecutor(),
                               observer=observer, rounds=14, actions=20,
                               conversation=list(self.ASKED))
        context.prior_clauses = ("8.4.1.1-2", "8.4.1.1-3", "8.3.1.4-1")
        AgentRuntime().run(context)
        after = [metadata.get("after") for kind, metadata in observer.notices
                 if kind == "clauses_unaddressed"]

        self.assertIn("stall", after)

    def test_a_turn_that_accounted_for_them_is_left_alone(self):
        # Distinct calls, so the turn reaches its answer instead of
        # stalling: a turn that loops without saying anything HAS not
        # accounted for the clauses, and being asked is right.
        backend = ScriptedBackend(
            [tool_turn(str(i), "bash", command=f"grep -n x{i} a.c")
             for i in range(3)]
            # Cited the way the harness asks for it: the marker is what
            # tells a clause from a version number.
            + [text_turn("src/a.c:12 now satisfies Rule 8.4.1.1-2, and "
                         "src/a.c:20 satisfies Rule 8.4.1.1-3.")] * 4)
        observer = RecordingObserver()
        context = make_context(backend, executor=GroundedToolExecutor(),
                               observer=observer, rounds=12, actions=20,
                               conversation=list(self.ASKED))
        context.prior_clauses = ("8.4.1.1-2", "8.4.1.1-3")
        AgentRuntime().run(context)

        self.assertNotIn("clauses_unaddressed",
                         [kind for kind, _ in observer.notices])


class ALoopThatAnnouncesItselfMustAlsoEnd(unittest.TestCase):
    """Saying "you have seen this" is not the same as stopping.

    Watched live: the notice fired three times and the turn ran on to a
    hundred and twenty-seven calls, because each repeat named a path in its
    arguments and so looked like new ground to the progress monitor.
    """

    WALL = "the same answer\n" * 40

    def test_repeated_results_stall_the_turn_instead_of_feeding_it(self):
        backend = ScriptedBackend(
            [tool_turn(str(i), "bash", command=f"grep -rn ack --include='*.{i}'")
             for i in range(20)] + [text_turn("done")] * 3)
        observer = RecordingObserver()
        result = AgentRuntime().run(make_context(
            backend, executor=GroundedToolExecutor([self.WALL] * 20),
            observer=observer, rounds=24, actions=24))

        self.assertLess(len(backend.calls), 12,
                        "a turn fed the same answer must not run to the ceiling")
        self.assertIn("identical_result",
                      [kind for kind, _ in observer.notices])

    def test_the_third_identical_call_is_refused_outright(self):
        import agent_runtime
        from tool_router import ToolResultEnvelope, ToolResultStatus

        envelope = ToolResultEnvelope(
            "c1", "a1", "bash", True, ToolResultStatus.OK,
            self.WALL, self.WALL, "command", 0.0, len(self.WALL))
        seen = {}

        for _ in range(agent_runtime._REPEAT_REFUSE):
            count = agent_runtime._repeated_result(seen, envelope)

        refused = agent_runtime._with_repeat_note(envelope, count)

        self.assertIn("REFUSED", refused.model_content)


class RepetitionJudgedByTheAnswer(unittest.TestCase):
    """Twelve greps differing only in their --include glob, same wall of
    matches each time, and nothing counted them as repeats."""

    WALL = "match line\n" * 40

    def run_with(self, commands, outputs):
        backend = ScriptedBackend(
            [tool_turn(str(i), "bash", command=c) for i, c in enumerate(commands)]
            + [text_turn("done")] * 3)
        observer = RecordingObserver()
        executor = GroundedToolExecutor(outputs)
        AgentRuntime().run(make_context(
            backend, executor=executor, observer=observer,
            rounds=len(commands) + 4, actions=len(commands) + 4))

        return observer, backend

    def test_different_arguments_with_the_same_answer_count_as_repeats(self):
        observer, _ = self.run_with(
            [f"grep -rn ack --include='*.{ext}'" for ext in "chxyz"],
            [self.WALL] * 5)

        self.assertIn("identical_result",
                      [kind for kind, _ in observer.notices])

    def test_a_short_result_is_not_an_identity(self):
        """"0" and "" repeat for perfectly good reasons."""
        observer, _ = self.run_with([f"wc -l f{i}.c" for i in range(5)],
                                    ["0"] * 5)

        self.assertNotIn("identical_result",
                         [kind for kind, _ in observer.notices])

    def test_a_repeated_failure_is_never_muffled(self):
        """The repair loop depends on hearing the same error again."""
        observer, _ = self.run_with([f"make target{i}" for i in range(5)],
                                    ["ERROR: " + self.WALL] * 5)

        self.assertNotIn("identical_result",
                         [kind for kind, _ in observer.notices])


class AskingMoreThanOnce(unittest.TestCase):
    """One redirect is measurably not enough, and it is not unbounded either."""

    def turn(self, rounds=30):
        backend = ScriptedBackend(
            [tool_turn(str(i), "bash", command="grep -n x a.c")
             for i in range(40)] + [text_turn("stopping")] * 4)
        observer = RecordingObserver()
        conversation = [ConversationMessage("user", (TextBlock(
            "can you validate and make changes in the code accordingly"),))]
        AgentRuntime().run(make_context(
            backend, executor=GroundedToolExecutor(), observer=observer,
            conversation=conversation, rounds=rounds, actions=60))

        return [kind for kind, _ in observer.notices
                if kind == "write_request_unanswered"]

    def test_a_turn_that_keeps_not_writing_is_asked_again(self):
        """Measured: asked once, read for twenty more rounds, ended empty."""

        self.assertGreater(len(self.turn()), 1)

    def test_preambles_do_not_burn_every_redirect_in_three_rounds(self):
        """Measured: two exploratory calls, then all three asks, then a stall.

        The turn had read nothing when the last of them fired.
        """
        backend = ScriptedBackend(
            [tool_turn("one", "bash", command="find . -name '*.c'")]
            + [text_turn("Let me examine the source files.")] * 10
            + [text_turn("stopping")] * 4)
        observer = RecordingObserver()
        conversation = [ConversationMessage("user", (TextBlock(
            "can you validate and make changes in the code accordingly"),))]
        AgentRuntime().run(make_context(
            backend, executor=GroundedToolExecutor(), observer=observer,
            conversation=conversation, rounds=8, actions=20))
        asked = [kind for kind, _ in observer.notices
                 if kind == "write_request_unanswered"]

        self.assertLessEqual(len(asked), 1,
                             "eight rounds is room for one ask, not three")

    def test_the_demanded_round_is_offered_writing_tools_only(self):
        backend = ScriptedBackend(
            [tool_turn(str(i), "bash", command="grep -n x a.c")
             for i in range(40)] + [text_turn("stopping")] * 4)
        conversation = [ConversationMessage("user", (TextBlock(
            "can you validate and make changes in the code accordingly"),))]
        context = make_context(backend, executor=GroundedToolExecutor(),
                               conversation=conversation, rounds=30, actions=60)
        context.tools = (ToolDefinition("bash", "run", {"type": "object"}),
                         ToolDefinition("edit_file", "edit", {"type": "object"}))
        AgentRuntime().run(context)
        offered = [tuple(sorted(tool.name for tool in call["tools"]))
                   for call in backend.calls]

        self.assertIn(("edit_file",), offered,
                      "the round after the demand offers writing only")

    def test_and_not_forever(self):
        import agent_runtime

        self.assertLessEqual(len(self.turn()),
                             agent_runtime.WRITE_REDIRECT_LIMIT)


class AStallOnABrokenTree(unittest.TestCase):
    """The two other build gates are unreachable from a stall."""

    def test_a_stalled_turn_that_broke_the_tree_is_asked_to_repair_it(self):
        """Measured: header changed, definition left behind, turn stalled.

        The tree no longer compiled, the turn was never told, and the bench
        found out afterwards.
        """
        repeated = dict(path="a.c", old_text="x", new_text="y")
        backend = ScriptedBackend([
            tool_turn("one", "edit_file", **repeated),
            # The same call over and over is what the progress monitor calls
            # a stall.
            tool_turn("two", "bash", command="grep -n x a.c"),
            tool_turn("three", "bash", command="grep -n x a.c"),
            tool_turn("four", "bash", command="grep -n x a.c"),
            tool_turn("five", "bash", command="grep -n x a.c"),
            tool_turn("six", "bash", command="grep -n x a.c"),
            text_turn("stopping"),
            text_turn("stopping"),
        ])
        observer = RecordingObserver()
        context = make_context(backend, executor=GroundedToolExecutor(),
                               observer=observer, rounds=12, actions=20)
        context.project_commands = project_build.ProjectCommands("false", "",
                                                                 "test")
        context.project_root = "."
        AgentRuntime().run(context)
        after_stall = [metadata.get("after")
                       for kind, metadata in observer.notices
                       if kind == "project_build_failed"]

        self.assertIn("stall", after_stall,
                      "a stall on a broken tree must ask for the repair")


class AfterABrokenBuild(unittest.TestCase):
    """What the harness controls is what it OFFERS."""

    def test_the_round_after_a_broken_build_offers_only_writing_tools(self):
        """Three demands, three greps, no edit — measured on a live run.

        Telling a model "fix it now, do not read anything else first" is a
        request. Removing the reading tools for that one round is not.
        """
        backend = ScriptedBackend([
            tool_turn("one", "edit_file", path="a.c", old_text="x",
                      new_text="y"),
            tool_turn("two", "bash", command="grep -n x a.c"),
            text_turn("done"),
            # The rounds the build demand buys: without them the scripted
            # model runs out before the narrowed round ever happens.
            tool_turn("three", "edit_file", path="a.c", old_text="y",
                      new_text="z"),
            text_turn("fixed"),
            text_turn("fixed"),
        ])
        executor = GroundedToolExecutor()
        context = make_context(backend, executor=executor, rounds=6, actions=10)
        context.project_commands = project_build.ProjectCommands("false", "",
                                                                 "test")
        context.project_root = "."
        context.tools = (ToolDefinition("bash", "run", {"type": "object"}),
                         ToolDefinition("edit_file", "edit", {"type": "object"}))
        AgentRuntime().run(context)
        offered = [tuple(sorted(tool.name for tool in call["tools"]))
                   for call in backend.calls]

        self.assertEqual(offered[0], ("bash", "edit_file"),
                         "the first round sees everything")
        self.assertIn(("edit_file",), offered[1:],
                      "the round after the failed build sees writing only")

    def test_the_narrowing_is_re_armed_never_sticky(self):
        backend = ScriptedBackend([
            tool_turn("one", "edit_file", path="a.c", old_text="x",
                      new_text="y"),
            tool_turn("two", "edit_file", path="a.c", old_text="y",
                      new_text="z"),
            text_turn("done"),
            tool_turn("three", "edit_file", path="a.c", old_text="z",
                      new_text="w"),
            text_turn("fixed"),
            text_turn("fixed"),
        ])
        context = make_context(backend, executor=GroundedToolExecutor(),
                               rounds=8, actions=12)
        context.project_commands = project_build.ProjectCommands("false", "",
                                                                 "test")
        context.project_root = "."
        context.tools = (ToolDefinition("bash", "run", {"type": "object"}),
                         ToolDefinition("edit_file", "edit", {"type": "object"}))
        observer = RecordingObserver()
        context.observer = observer
        AgentRuntime().run(context)
        offered = [tuple(sorted(tool.name for tool in call["tools"]))
                   for call in backend.calls]
        narrowed = [names for names in offered if names == ("edit_file",)]
        demands = [kind for kind, _ in observer.notices
                   if kind == "project_build_failed"]

        # A build that keeps failing narrows the round after each failure,
        # which is right. What must never happen is a narrowing that outlives
        # the demand that armed it.
        self.assertTrue(narrowed, "a failed build narrows the next round")
        self.assertLessEqual(len(narrowed), len(demands),
                             "each narrowed round answers one demand")


class SilentWork(unittest.TestCase):
    """Work the operator waits for has to be work the operator can see."""

    def context(self, backend):
        context = make_context(backend, rounds=4)
        # Pressure low enough that compaction is attempted every round, and
        # no tail held back -- otherwise a short scripted conversation is all
        # tail, nothing is eligible, and the service declines. It used to
        # decline INSIDE the spinner, which is what made that invisible.
        context.compaction_policy = CompactionPolicy(
            pressure_threshold=.01, minimum_compactable_tokens=1,
            recent_tail_groups=0)

        return context

    def test_compacting_runs_behind_a_labelled_activity(self):
        """Announced when it happens -- there has to be something to compact."""

        backend = ScriptedBackend([
            tool_turn("one", "bash", command="pwd"),
            tool_turn("two", "bash", command="ls"),
            text_turn("done"),
        ])
        observer = RecordingObserver()
        context = self.context(backend)
        context.observer = observer
        AgentRuntime().run(context)

        self.assertIn("Compacting context…", list(observer.activities),
                      "a model call the turn waits for must be announced")

    def test_nothing_is_announced_when_there_is_no_pressure_to_relieve(self):
        """The label has to mean the work, not the permission to do it.

        The cooldown says compaction is ALLOWED now; it does not say the
        context needs it. The spinner was opened first and the service
        declined inside it, in under a millisecond and with no model call.
        `_last_compaction_round` starts at -99, so round one always entered
        and then one round in four: a real session with 41 rounds and 0.42 of
        its window in use showed "Compacting context…" about eleven times and
        compacted nothing at all.
        """

        backend = ScriptedBackend([
            tool_turn("one", "bash", command="pwd"),
            tool_turn("two", "bash", command="ls"),
            text_turn("done"),
        ])
        observer = RecordingObserver()
        context = make_context(backend, rounds=6)
        context.observer = observer

        # The production threshold against a window nothing is filling.

        context.compaction_policy = CompactionPolicy(pressure_threshold=0.90)
        result = AgentRuntime().run(context)

        self.assertEqual(result.final_response, "done")
        self.assertNotIn("Compacting context…", list(observer.activities))
        self.assertIsNone(context.compaction_artifact)

        # And the round marker is untouched, so the first real pressure is
        # not made to wait out a cooldown nothing earned.

        self.assertIsNone(getattr(context, "_last_compaction_round", None))

    def test_the_label_and_the_work_come_from_one_decision(self):
        """`will_compact` is what `compact` asks itself, not a second guess."""

        service = CompactionService(
            policy=CompactionPolicy(pressure_threshold=.01,
                                    minimum_compactable_tokens=1))
        state = WorkingState.start("task_compaction", "probe")
        empty = CompactionRequest(
            working_state=state,
            context_request=ContextRequest(items=(), context_limit=4096),
            trigger_reason="probe")

        self.assertFalse(service.will_compact(empty))
        self.assertFalse(service.compact(empty).succeeded)

    def test_a_tool_call_runs_behind_the_same_indicator_as_a_model_call(self):
        """The operator waits for a tool exactly as for a model call.

        Only the model call had an indicator, so a running tool left the
        finished call's last frame on screen, unmoving. Every standard.*
        tool prints nothing while it works.
        """

        backend = ScriptedBackend([
            tool_turn("one", "bash", command="pwd"),
            text_turn("done"),
        ])
        observer = RecordingObserver()
        context = make_context(backend, observer=observer, rounds=4)
        AgentRuntime().run(context)
        labels = list(observer.activities)

        self.assertIn("bash…", labels)

        # And it is announced while the tool runs, not after: the model call
        # that asked for it comes first, the tool second.

        self.assertLess(labels.index("Thinking…"), labels.index("bash…"))

    def test_every_tool_is_announced_by_its_own_name(self):
        backend = ScriptedBackend([
            tool_turn("one", "bash", command="pwd"),
            tool_turn("two", "edit_file", path="a.py", old_text="a", new_text="b"),
            text_turn("done"),
        ])
        observer = RecordingObserver()
        context = make_context(backend, executor=GroundedToolExecutor(
            ["OK: pwd", "OK: edit"]), observer=observer, rounds=6)
        AgentRuntime().run(context)

        self.assertIn("bash…", observer.activities)
        self.assertIn("edit_file…", observer.activities)

    def test_a_failing_compaction_is_reported_once_not_every_round(self):
        """One trace held twenty-one failures in a row, none of them shown."""
        backend = ScriptedBackend([
            tool_turn("one", "bash", command="pwd"),
            tool_turn("two", "bash", command="ls"),
            text_turn("done"),
        ])
        observer = RecordingObserver()
        context = self.context(backend)
        context.observer = observer
        context.compaction_policy = CompactionPolicy(
            pressure_threshold=.01, minimum_compactable_tokens=1, max_retries=0)
        AgentRuntime().run(context)
        reported = [kind for kind, _ in observer.notices
                    if kind == "compaction_failed"]

        self.assertLessEqual(len(reported), 1,
                             "announced once per turn, never per round")


class TheProjectsOwnVerdict(unittest.TestCase):
    """A writing turn is built and tested; that outcome is a label.

    Filing it as "unrated" beside turns nothing ever checked threw away the
    only verdict the harness produces without anyone typing /good.
    """

    def result(self, *, commands, outputs=None, calls=None, verifier=...):
        backend = ScriptedBackend([
            tool_turn("one", "edit_file", path="a.c", old_text="x",
                      new_text="y"),
            text_turn("done"),
        ] if calls is None else calls)
        context = make_context(backend, executor=GroundedToolExecutor(outputs),
                               rounds=6, actions=10)
        context.project_commands = commands
        context.project_root = "."

        # A confined runner, stubbed: "true" passes, "false" fails. There is
        # no host fallback, so a context with no verifier runs nothing.

        if verifier is ...:
            verifier = lambda command: (
                ("passed", "") if command == "true" else ("failed", "exit 1"))

        context.project_verifier = verifier

        return AgentRuntime().run(context)

    def test_a_green_build_labels_the_turn_pass(self):
        commands = project_build.ProjectCommands("true", "true", "test")

        self.assertIs(self.result(commands=commands).project_build_ok, True)

    def test_a_failing_build_labels_the_turn_fail(self):
        commands = project_build.ProjectCommands("false", "", "test")

        self.assertIs(self.result(commands=commands).project_build_ok, False)

    def test_a_tree_with_no_build_is_not_judged(self):
        """Not judged is not judged wrong, and the corpus must tell them apart."""

        self.assertIsNone(self.result(commands=None).project_build_ok)

    def test_a_turn_that_wrote_nothing_is_not_judged(self):
        commands = project_build.ProjectCommands("true", "true", "test")
        result = self.result(commands=commands,
                             calls=[text_turn("nothing to change")])

        self.assertIsNone(result.project_build_ok)

    def test_a_verification_that_could_not_run_leaves_the_turn_unjudged(self):
        """Fail closed, and say which failure it was.

        Without a confined runner nothing is run at all -- never on the host
        as a fallback -- and a turn nothing could check is not a turn that
        failed its build.
        """

        commands = project_build.ProjectCommands("true", "true", "test")

        self.assertIsNone(self.result(commands=commands, verifier=None).project_build_ok)
        self.assertIsNone(self.result(
            commands=commands,
            verifier=lambda command: ("not_run", "sandbox unavailable"),
        ).project_build_ok)


class ProjectVerificationCountsTests(unittest.TestCase):
    """The harness running the project's own tests is a verification.

    It is part of what the complete agent does, so it settles the turn. A run
    that has just been shown a green suite must not be told to run the same
    suite again to prove what the harness already observed.
    """

    def run_turn(self, commands, verifier):
        backend = ScriptedBackend([
            tool_turn("one", "edit_file", path="a.py", old_text="x", new_text="y"),
            text_turn("done"), text_turn("done"), text_turn("done"),
        ])
        recorder = MemoryRecorder()
        context = make_context(backend, executor=GroundedToolExecutor(None),
                               rounds=6, actions=10, trace=TraceEmitter(recorder))
        context.project_commands = commands
        context.project_root = "."
        context.project_verifier = verifier
        result = AgentRuntime().run(context)

        return context, result, recorder

    def test_a_configured_suite_that_passed_verifies_the_turn(self):
        context, result, recorder = self.run_turn(
            project_build.ProjectCommands("", "run-tests", "projects.json"),
            lambda command: ("passed", ""))
        evaluation = context.verification_policy.evaluate_completion(
            result.working_state)

        self.assertEqual(evaluation.status, CompletionVerificationStatus.VERIFIED)
        passed = next(item for item in recorder.events
                      if item.event_type == EventType.VERIFICATION_PASSED)
        self.assertEqual(passed.metadata["coverage"], "full")
        self.assertTrue(passed.metadata["proves_change"])
        self.assertEqual(passed.metadata["mutation_generation"],
                         result.working_state.mutation_generation)

    def test_a_probed_command_builds_the_tree_without_certifying_the_turn(self):
        """`make` found by looking is a guess, not a contract."""

        context, result, recorder = self.run_turn(
            project_build.ProjectCommands("make", "", "make"),
            lambda command: ("passed", ""))

        self.assertEqual(
            context.verification_policy.evaluate_completion(
                result.working_state).status,
            CompletionVerificationStatus.UNVERIFIED)
        self.assertNotIn(EventType.VERIFICATION_PASSED,
                         [item.event_type for item in recorder.events])

    def test_the_configured_suite_runs_before_any_retry_is_decided(self):
        """The order the v4 run got wrong.

        It fixed main.py, ran a targeted test, and was then told to verify:
        verification_started(kind=verification_nudge) and
        retry(reason=unverified_change) were emitted, and the deterministic
        build/full/passed evidence only appeared after the extra actions. Nine
        model calls to learn what the harness could observe for free. The
        suite is configured and deterministic, so it runs first, and the
        demand is only made if its answer is not enough.
        """

        backend = ScriptedBackend([
            tool_turn("one", "edit_file", path="a.py", old_text="x", new_text="y"),
            text_turn("fixed it"),
        ])
        recorder = MemoryRecorder()
        context = make_context(backend, executor=GroundedToolExecutor(None),
                               rounds=6, actions=10, trace=TraceEmitter(recorder))
        context.project_commands = project_build.ProjectCommands(
            "", "run-tests", "projects.json")
        context.project_root = "."
        runs = []
        context.project_verifier = lambda command: (
            runs.append(command), ("passed", ""))[1]
        result = AgentRuntime().run(context)

        self.assertEqual(result.final_response, "fixed it")
        self.assertEqual(runs, ["run-tests"], "one execution per generation")

        kinds = [(item.event_type, (item.metadata or {}).get("kind"),
                  (item.metadata or {}).get("reason")) for item in recorder.events]

        self.assertNotIn((EventType.VERIFICATION_STARTED, "verification_nudge", None),
                         kinds)
        self.assertNotIn((EventType.RETRY, None, "unverified_change"), kinds)

        # And it is recorded before anything could have decided to retry.

        order = [item.event_type for item in recorder.events]
        self.assertIn(EventType.VERIFICATION_PASSED, order)
        self.assertNotIn(EventType.RETRY, order)
        self.assertEqual(
            context.verification_policy.evaluate_completion(
                result.working_state).status,
            CompletionVerificationStatus.VERIFIED)

    def test_an_unconfigured_tree_still_asks_the_model_to_verify(self):
        """Nothing deterministic to run, so the demand is the only route."""

        backend = ScriptedBackend([
            tool_turn("one", "edit_file", path="a.py", old_text="x", new_text="y"),
            text_turn("fixed it"), text_turn("fixed it"), text_turn("fixed it"),
        ])
        recorder = MemoryRecorder()
        context = make_context(backend, executor=GroundedToolExecutor(None),
                               rounds=6, actions=10, trace=TraceEmitter(recorder))
        context.project_commands = project_build.ProjectCommands("make", "", "make")
        context.project_root = "."
        context.project_verifier = lambda command: ("passed", "")
        AgentRuntime().run(context)

        self.assertIn("verification_nudge",
                      [(item.metadata or {}).get("kind") for item in recorder.events])

    def recorded(self, status, source="projects.json"):
        """What one project verification of that status leaves behind."""

        context = make_context(ScriptedBackend([text_turn("x")]))
        context.apply_state_event(
            StateEventType.ACTION_SUCCEEDED, source=StateSource.TOOL_RUNTIME,
            action_id="edit1", kind=ActionKind.TOOL.value, name="edit_file",
            observed_status="ok")
        context.apply_state_event(
            StateEventType.FILE_MODIFIED, source=StateSource.TOOL_RUNTIME,
            path="a.py", action_id="edit1")
        _record_project_verification(
            context, project_build.ProjectCommands("", "run-tests", source),
            (("run-tests", status, ""),))

        return context.verification_policy.evaluate_completion(
            context.working_state)

    def test_a_suite_that_could_not_run_does_not_verify_anything(self):
        self.assertEqual(self.recorded("not_run").status,
                         CompletionVerificationStatus.UNVERIFIED)

    def test_a_failing_suite_is_recorded_as_a_failure(self):
        self.assertEqual(self.recorded("failed").status,
                         CompletionVerificationStatus.FAILED)

    def test_a_passing_suite_verifies_the_current_generation(self):
        self.assertEqual(self.recorded("passed").status,
                         CompletionVerificationStatus.VERIFIED)

    def test_a_probed_command_records_nothing_at_all(self):
        self.assertEqual(self.recorded("passed", source="make").status,
                         CompletionVerificationStatus.UNVERIFIED)


class ReadOnlyIntentTests(unittest.TestCase):
    """A function's NAME is not an instruction to write it.

    `_WRITE_REQUEST_RE` carries `\badd\b`, so "Answer which file defines add
    without editing files" was classified as a request to change the code.
    Every gate that asks then spent rounds demanding an edit the user had
    forbidden: thirty-nine model calls to answer a question the turn had
    already answered.
    """

    ASKED = "Answer which file defines add without editing files."

    def run_turn(self, *, read_only, asked=None, turns=None, executor=None):
        backend = ScriptedBackend(turns or [
            tool_turn("look", "bash", command="grep -rn 'def add' ."),
            text_turn("The file defining add is main.py."),
        ])
        recorder = MemoryRecorder()
        observer = RecordingObserver()
        context = make_context(
            backend, executor=executor or GroundedToolExecutor(
                ["main.py:1:def add(a, b):"]),
            rounds=8, actions=10, trace=TraceEmitter(recorder),
            observer=observer,
            conversation=[ConversationMessage(
                "user", (TextBlock(asked or self.ASKED),))])
        context.read_only = read_only

        return backend, recorder, observer, AgentRuntime().run(context)

    def test_the_answer_is_handed_back_with_no_demand_to_write(self):
        backend, recorder, observer, result = self.run_turn(read_only=True)

        self.assertEqual(result.final_response, "The file defining add is main.py.")
        self.assertEqual(len(backend.calls), 2, "no extra model call")

        kinds = [kind for kind, _ in observer.notices]
        self.assertNotIn("write_request_unanswered", kinds)
        self.assertNotIn("announced_change_not_made", kinds)

        reasons = [(item.metadata or {}).get("reason") for item in recorder.events
                   if item.event_type == EventType.RETRY]
        self.assertNotIn("write_request_unanswered", reasons)

        asked = "".join(
            block.text for message in backend.calls[-1]["conversation"]
            for block in message.content if isinstance(block, TextBlock))
        self.assertNotIn("this turn is not finished until a file changes", asked)
        self.assertNotIn("Make the edit now", asked)

    def test_read_only_beats_a_decision_already_cached_as_a_write(self):
        """The prohibition is checked before the cache, and rewrites it."""

        context = make_context(ScriptedBackend([text_turn("x")]))
        context._write_request = True
        context.read_only = True

        self.assertFalse(wants_write(context, self.ASKED))
        self.assertIs(context._write_request, False)
        self.assertFalse(may_demand_write(context))

        # And without the prohibition the same question still reads as a
        # write, which is the bug -- untouched, because the fix is the
        # prohibition, not a new exception for one verb.

        plain = make_context(ScriptedBackend([text_turn("x")]))
        self.assertTrue(is_write_request(self.ASKED))
        self.assertTrue(wants_write(plain, self.ASKED))

    def test_conclude_demand_consumes_the_decision_it_is_given(self):
        self.assertIn("not finished until a file changes",
                      conclude_demand(self.ASKED))
        self.assertNotIn("not finished until a file changes",
                         conclude_demand(self.ASKED, False))
        self.assertIn("not finished until a file changes",
                      conclude_demand("explain the architecture", True))

    def test_a_real_write_request_is_still_sent_back_to_the_code(self):
        """No global prohibition, so every gate works exactly as before."""

        backend, recorder, observer, result = self.run_turn(
            read_only=False,
            asked="Add a multiply function and its test.",
            turns=[tool_turn("look", "bash", command="cat service.py"),
                   text_turn("Here is the change I would make."),
                   text_turn("Here is the change I would make."),
                   text_turn("done")])

        kinds = [kind for kind, _ in observer.notices]
        self.assertIn("write_request_unanswered", kinds)

        reasons = [(item.metadata or {}).get("reason") for item in recorder.events
                   if item.event_type == EventType.RETRY]
        self.assertIn("write_request_unanswered", reasons)

    def test_a_read_only_turn_never_narrows_the_tools_to_the_writing_ones(self):
        source = Path(__file__).resolve().parents[1] / "agent_runtime.py"
        text = source.read_text()

        for block in text.split("only_tools = _WRITE_TOOLS")[:-1]:
            self.assertIn("may_demand_write(context)", block[-4000:],
                          "a write window opened without asking the scope")


class WhoIsAskingTests(unittest.TestCase):
    """The harness speaks in the user role; it must not hear itself."""

    def test_a_harness_nudge_is_not_the_users_request(self):
        conversation = [
            ConversationMessage("user", (TextBlock(
                "How should the ACK be managed in VITA 49.2?"),)),
            ConversationMessage("assistant", (TextBlock("Rule 8.4.1.1-2 ..."),)),
            ConversationMessage("user", (TextBlock("Conclude now and fix it."),),
                                authored_by="harness"),
        ]

        self.assertEqual(_asked(conversation),
                         "How should the ACK be managed in VITA 49.2?")
        self.assertFalse(is_write_request(_asked(conversation)))

    def test_the_conclude_nudge_would_read_as_a_write_request(self):
        """Why the field exists: the nudge's own words match the detector.

        Read back as the request, it turned a question into a demand for an
        edit, and the turn ended arguing instead of answering.
        """
        nudge = conclude_demand("How should the ACK be managed?")

        self.assertTrue(is_write_request(nudge))

    def test_an_unmarked_message_is_still_the_operator(self):
        conversation = [ConversationMessage("user", (TextBlock("fix it"),))]

        self.assertEqual(_asked(conversation), "fix it")


class AgentRuntimeCoreTests(unittest.TestCase):
    def setUp(self):
        self.runtime = AgentRuntime()

    def test_simple_assistant_response_and_terminal_result(self):
        context = make_context(ScriptedBackend([text_turn("answer")]))
        result = self.runtime.run(context)
        self.assertEqual(result.final_response, "answer")
        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.COMPLETED)
        self.assertEqual(result.terminal_status, TerminalStatus.COMPLETED)
        self.assertIs(result.working_state, context.working_state)

    def test_generation_aware_verification_repair_sequence(self):
        backend = ScriptedBackend([
            tool_turn("edit1", "edit_file", path="a.py", old_text="a", new_text="b"),
            tool_turn("pwd", "bash", command="pwd"),
            tool_turn("test1", "bash", command="pytest"),
            tool_turn("edit2", "edit_file", path="a.py", old_text="b", new_text="c"),
            tool_turn("test2", "bash", command="pytest -q"),
            tool_turn("edit3", "edit_file", path="a.py", old_text="c", new_text="d"),
            tool_turn("test3", "bash", command="pytest tests"),
            text_turn("repaired"),
        ])
        executor = GroundedToolExecutor([
            "OK: edit", "OK: pwd", "OK: tests", "OK: edit",
            "ERROR: tests failed (exit 2)", "OK: repair", "OK: tests",
        ])
        recorder = MemoryRecorder()
        context = make_context(
            backend, executor=executor, rounds=10, actions=12,
            trace=TraceEmitter(recorder),
        )
        result = self.runtime.run(context)
        self.assertEqual(result.final_response, "repaired")
        self.assertEqual(result.working_state.mutation_generation, 3)
        self.assertEqual(context.verification_policy.evaluate_completion(
            result.working_state).status, CompletionVerificationStatus.VERIFIED)
        self.assertTrue(any(item.outcome.value == "failed"
                            for item in result.working_state.verifications))
        pwd = next(item for item in result.working_state.verifications
                   if item.command == "pwd")
        self.assertEqual(pwd.coverage, "none")
        event_types = [item.event_type for item in recorder.events]
        self.assertIn(EventType.VERIFICATION_INVALIDATED, event_types)
        self.assertIn(EventType.VERIFICATION_FAILED, event_types)
        self.assertIn(EventType.VERIFICATION_PASSED, event_types)

    def test_unclassifiable_command_is_traced_as_classified_not_passed(self):
        """`verification_passed` must mean a verification passed.

        An unrecognised command that exits zero used to emit it, carrying
        category=unknown and coverage=unknown. The completion gate already
        ignored such evidence; every reader of the trace did not.
        """

        backend = ScriptedBackend([
            tool_turn("edit1", "edit_file", path="a.py", old_text="a", new_text="b"),
            tool_turn("odd", "bash", command="frobnicate --all"),
            text_turn("done"),
        ])
        executor = GroundedToolExecutor(["OK: edit", "OK: frobnicated"])
        recorder = MemoryRecorder()
        context = make_context(backend, executor=executor, rounds=6, actions=8,
                               trace=TraceEmitter(recorder))
        self.runtime.run(context)
        event_types = [item.event_type for item in recorder.events]
        self.assertNotIn(EventType.VERIFICATION_PASSED, event_types)
        self.assertIn(EventType.VERIFICATION_CLASSIFIED, event_types)
        classified = next(item for item in recorder.events
                          if item.event_type == EventType.VERIFICATION_CLASSIFIED)
        self.assertEqual(classified.metadata["category"], "unknown")
        self.assertFalse(classified.metadata["proves_change"])

    def test_programmatic_rollback_updates_working_state_and_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            target = workspace / "a.txt"
            target.write_text("original")
            context = make_context(ScriptedBackend([text_turn("done")]))
            store = FileSessionStore(root / "sessions")
            session_id = new_session_id()
            context.session = SessionHandle(store, session_id,
                SessionConfiguration(str(workspace), "test", "safe"))
            manager = CheckpointManager(root / "checkpoints", workspace)
            checkpoint = manager.begin_checkpoint(context.task_id, session_id)
            context.checkpoint_manager, context.checkpoint = manager, checkpoint
            context.apply_state_event(
                StateEventType.CHECKPOINT_RECORDED,
                checkpoint_id=checkpoint.checkpoint_id, status="active",
            )
            manager.capture(checkpoint, target)
            target.write_text("task change")
            manager.record_mutation(checkpoint, target, "edit_rollback")
            result = context.rollback_checkpoint()
            self.assertEqual(result.status, RollbackStatus.SUCCESS)
            self.assertEqual(target.read_text(), "original")
            self.assertEqual(context.working_state.checkpoint_status, "rolled_back")
            loaded = store.load_snapshot(session_id)
            self.assertEqual(loaded.checkpoint_status, "rolled_back")
    def test_one_successful_tool_call(self):
        executor = GroundedToolExecutor(["OK: listed"])
        backend = ScriptedBackend([tool_turn("one", "bash", command="ls"),
                                   text_turn("found it")])
        result = self.runtime.run(make_context(backend, executor=executor))
        self.assertEqual(result.final_response, "found it")
        self.assertEqual(len(result.tool_log), 1)
        self.assertEqual(result.working_state.tool_actions, 1)
        self.assertIsInstance(backend.calls[-1]["conversation"][-1].content[0],
                              object)

    def test_multiple_sequential_tool_calls(self):
        backend = ScriptedBackend([
            tool_turn("one", "bash", command="pwd"),
            tool_turn("two", "bash", command="ls"),
            text_turn("done"),
        ])
        result = self.runtime.run(make_context(backend))
        self.assertEqual(len(result.tool_log), 2)
        self.assertEqual(result.rounds_consumed, 3)

    def test_tool_failure_is_returned_to_model(self):
        executor = GroundedToolExecutor(["ERROR: denied"])
        backend = ScriptedBackend([
            tool_turn("bad", "bash", command="forbidden"), text_turn("adjusted"),
        ])
        result = self.runtime.run(make_context(backend, executor=executor))
        second_request = backend.calls[-1]["conversation"]
        result_block = second_request[-1].content[0]
        self.assertTrue(result_block.is_error)
        self.assertEqual(result.final_response, "adjusted")
        self.assertEqual(len(result.working_state.unresolved_failures), 1)

    def test_failed_mutation_envelope_does_not_claim_a_modification(self):
        executor = GroundedToolExecutor(["ERROR: denied"])
        backend = ScriptedBackend([
            tool_turn("bad", "write_file", path="not-written.py"),
            text_turn("not changed"),
        ])
        result = self.runtime.run(make_context(backend, executor=executor))
        self.assertFalse(result.did_modify)
        self.assertEqual(result.working_state.modified_files, set())

    def test_nonzero_command_outcome_is_grounded_failure(self):
        executor = GroundedToolExecutor(["compiler failed\n\n(exit 2)"])
        backend = ScriptedBackend([
            tool_turn("build", "bash", command="make"), text_turn("failed"),
        ])
        result = self.runtime.run(make_context(backend, executor=executor))
        action = next(item for item in result.working_state.actions
                      if item.kind == ActionKind.COMMAND)
        self.assertEqual(action.exit_code, 2)
        self.assertEqual(action.status.value, "failed")

    def test_malformed_turn_fails_with_normalized_reason(self):
        malformed = ModelTurn("", (), StopReason.TOOL_USE)
        result = self.runtime.run(make_context(ScriptedBackend([malformed])))
        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.INVALID_TURN)
        self.assertEqual(result.terminal_status, TerminalStatus.FAILED)
        self.assertIn("without tool_calls", result.error_summary)

    def test_preamble_retry_path(self):
        backend = ScriptedBackend([text_turn("Let me read that first."),
                                   text_turn("actual answer")])
        result = self.runtime.run(make_context(backend))
        self.assertEqual(result.final_response, "actual answer")
        self.assertEqual(result.working_state.retry_count, 1)
        self.assertIn("Do it now", backend.calls[-1]["conversation"][-1].content[0].text)

    def test_repeated_command_cache_and_anti_flailing_nudge(self):
        backend = ScriptedBackend([
            tool_turn("one", "bash", command="pwd"),
            tool_turn("two", "bash", command="pwd"),
            tool_turn("three", "bash", command="pwd"),
            text_turn("conclusion"),
        ])
        observer = RecordingObserver()
        result = self.runtime.run(make_context(backend, observer=observer, rounds=6))
        self.assertEqual(result.working_state.repeated_action_count, 2)
        self.assertIn("investigation_nudge", [kind for kind, _ in observer.notices])
        self.assertEqual(result.final_response, "conclusion")

    def test_a_terminal_repeat_ends_the_turn_instead_of_the_budget(self):
        """The router has told the model twice; the loop stops here.

        A read-only turn ran `cat main.py` once and was handed it back six
        more times, finishing at nineteen model calls with 316k tokens of
        context. Once the same observation has been refused as already
        present, asking for it again is not a step towards an answer.
        """

        class Repeating:
            """Serves once, then caches, then reports terminal repetition."""

            def __init__(self):
                self.seen = 0

            def __call__(self, context, tool_call_id, name, arguments, cache):
                self.seen += 1
                status, metadata = ToolResultStatus.OK, {}

                if self.seen == 2:
                    status = ToolResultStatus.CACHED
                elif self.seen >= 3:
                    status = ToolResultStatus.DENIED
                    metadata = {"suppressed_repeat": True,
                                "repeat_kind": "cached_action_repeated",
                                "repeat_count": self.seen,
                                "terminal_repeat": self.seen > 3}

                return ToolResultEnvelope(
                    tool_call_id, f"a{self.seen}", name,
                    status is ToolResultStatus.OK, status, "out", "out",
                    "command", 0.001, 3, metadata=metadata)

        backend = ScriptedBackend([
            tool_turn(str(index), "bash", command="cat main.py")
            for index in range(5)
        ] + [text_turn("main.py defines add")])
        recorder = MemoryRecorder()
        result = self.runtime.run(make_context(
            backend, executor=Repeating(), rounds=8, actions=10,
            trace=TraceEmitter(recorder)))

        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.STALLED)
        kinds = [(item.metadata or {}).get("kind") for item in recorder.events
                 if item.event_type == EventType.STALLED_DETECTED]
        self.assertIn("cached_action_repeated", kinds)

        # Four tool calls, not the whole budget.

        self.assertLessEqual(result.tool_actions, 4)

    def test_a_second_scope_violation_closes_the_tool_window(self):
        """One refusal names the scope; a search for another syntax ends it.

        Asked only which file defines `add`, a run tried `sed -i` at step
        four, then kept looking for a spelling that would get through --
        thirty-nine model calls to answer a question it could already answer.
        """

        class Violating:
            def __init__(self):
                self.attempts = 0

            def __call__(self, context, tool_call_id, name, arguments, cache):
                self.attempts += 1

                return ToolResultEnvelope(
                    tool_call_id, f"v{self.attempts}", name, False,
                    ToolResultStatus.DENIED,
                    "ERROR: command denied\nThe user explicitly requested a "
                    "read-only task.", "ERROR: command denied", "command",
                    0.001, 20,
                    metadata={"read_only_violations": self.attempts})

        backend = ScriptedBackend([
            tool_turn("one", "bash", command="sed -i 's/a/b/' main.py"),
            tool_turn("two", "bash", command="python3 -c \"open('main.py','w')\""),
            text_turn("main.py defines add"),
        ])
        observer = RecordingObserver()
        result = self.runtime.run(make_context(
            backend, executor=Violating(), observer=observer,
            rounds=8, actions=10))

        self.assertEqual(result.final_response, "main.py defines add")
        self.assertIn("read_only_violation", [kind for kind, _ in observer.notices])

        # The final request was made with the tool window shut.

        self.assertFalse(backend.calls[-1]["use_tools"])

    def test_concluding_does_not_excuse_an_empty_diff_on_a_write_request(self):
        """"Proposed fix (not yet applied)" is not an answer to "make changes".

        Two master runs ended exactly there: correct diagnosis, correct
        plan, nothing written in sixteen minutes. The nudge had fired, the
        turn concluded, and concluding used to suppress the one message
        that sends it back to the code.
        """
        backend = ScriptedBackend([
            tool_turn("one", "bash", command="pwd"),
            tool_turn("two", "bash", command="pwd"),
            tool_turn("three", "bash", command="pwd"),     # nudge fires here
            tool_turn("four", "bash", command="grep x"),   # the round it allows
            # By here the tool window has shut: concluding is true, and the
            # redirect must reopen it or the edit it demands is impossible.
            text_turn("Proposed fix (not yet applied) to command_wire.c"),
            tool_turn("six", "edit_file", path="command_wire.c",
                      old_text="x", new_text="y"),
            text_turn("done"),
        ])
        executor = GroundedToolExecutor()
        observer = RecordingObserver()
        conversation = [ConversationMessage("user", (TextBlock(
            "can you validate and make changes in the code accordingly"),))]
        self.runtime.run(make_context(
            backend, executor=executor, observer=observer, conversation=conversation,
            rounds=10, actions=20))
        written = [name for _, name, _ in executor.calls if name == "edit_file"]

        offered = [call["use_tools"] for call in backend.calls]

        self.assertIn("write_request_unanswered",
                      [kind for kind, _ in observer.notices])
        self.assertFalse(offered[4], "the window had shut before the redirect")
        self.assertTrue(offered[5], "the redirect reopens it, or asks the "
                                    "impossible")
        self.assertEqual(len(written), 1,
                         "the turn is sent back to the code, and writes")

    def test_a_turn_that_did_work_never_hands_back_nothing(self):
        """Three live runs rendered a blank answer over a full tool log."""
        backend = ScriptedBackend([
            tool_turn("one", "bash", command="pwd"),
            text_turn(""),
            text_turn(""),
            text_turn(""),
        ])
        observer = RecordingObserver()
        result = self.runtime.run(make_context(backend, observer=observer,
                                               rounds=6))

        self.assertTrue(result.final_response.strip())
        self.assertIn("without reaching a conclusion", result.final_response)
        self.assertIn("empty_final_synthesis",
                      [kind for kind, _ in observer.notices])

    def test_an_edit_after_the_nudge_reopens_the_window(self):
        """"You can make more small edits after this one" has to be true.

        A five-file change arrived as one edit -- the field the rest of the
        change needs -- and the window shut on the next round, leaving the
        tree half-changed and the turn reporting an answer.
        """
        backend = ScriptedBackend([
            tool_turn("one", "bash", command="pwd"),
            tool_turn("two", "bash", command="pwd"),
            tool_turn("three", "bash", command="pwd"),       # nudge fires here
            tool_turn("four", "edit_file", path="a.c", old_text="x",
                      new_text="y"),                          # the turn writes
            tool_turn("five", "edit_file", path="b.c", old_text="p",
                      new_text="q"),                          # must still be offered
            text_turn("done"),
        ])
        executor = GroundedToolExecutor()
        self.runtime.run(make_context(backend, executor=executor,
                                      rounds=10, actions=20))
        offered = [call["use_tools"] for call in backend.calls]
        written = [name for _, name, _ in executor.calls if name == "edit_file"]

        self.assertEqual(offered[:5], [True] * 5,
                         "a writing turn keeps its tools")
        self.assertEqual(len(written), 2,
                         "the second edit must still be reachable")

    def test_a_refused_round_does_not_spend_the_one_the_nudge_allows(self):
        """The sandbox refusing a call is not the turn using its round.

        Twice on a live run the model answered the nudge with a no-op the
        allowlist denied -- `true`, then `echo skip` -- and the tool window
        shut on the next round. The turn was then asked to conclude with
        nothing done, and the write it had been asked for never happened.
        """
        backend = ScriptedBackend([
            tool_turn("one", "bash", command="pwd"),
            tool_turn("two", "bash", command="pwd"),
            tool_turn("three", "bash", command="pwd"),      # nudge fires here
            tool_turn("four", "bash", command="true"),      # refused: not a round
            tool_turn("five", "bash", command="ls"),        # still offered tools
            text_turn("conclusion"),
        ])
        executor = DenyingToolExecutor(deny={"true"})
        result = self.runtime.run(make_context(
            backend, executor=executor, rounds=8))
        offered = [call["use_tools"] for call in backend.calls]

        self.assertEqual(offered[:5], [True] * 5,
                         "the refused round is given back, not charged")
        self.assertEqual(result.final_response, "conclusion")

    def test_the_nudge_costs_one_round_then_forces_a_conclusion(self):
        """A model that ignores "conclude now" gets one round, not eight.

        Observed in a real session: after the nudge the model ran another
        eight investigative commands, because the message was advisory and
        nothing enforced it. The round after the nudge still carries tools --
        the same message asks for one small edit -- and the one after that is
        final.
        """
        backend = ScriptedBackend([
            tool_turn("one", "bash", command="pwd"),
            tool_turn("two", "bash", command="pwd"),
            tool_turn("three", "bash", command="pwd"),      # nudge fires here
            tool_turn("four", "bash", command="ls"),        # the one allowed round
            tool_turn("five", "bash", command="grep -r x"),  # must not happen
            text_turn("conclusion"),
        ])
        observer = RecordingObserver()
        result = self.runtime.run(make_context(backend, observer=observer, rounds=8))
        self.assertIn("investigation_nudge", [kind for kind, _ in observer.notices])
        # What the harness controls is what it OFFERS. Rounds 1-3 investigate,
        # round 4 is the one the nudge allows for a small edit, and round 5 is
        # called with no tools at all -- which is what "no further commands"
        # has to mean to be true.
        offered = [call["use_tools"] for call in backend.calls]
        self.assertEqual(offered[:4], [True] * 4,
                         "three investigating rounds, then the one the nudge allows")
        self.assertTrue(offered[4:] and not any(offered[4:]),
                        "after that round the tool window stays shut, including "
                        "the final summarising call")
        self.assertEqual(result.final_response, "conclusion")
        # And the turn ENDS, it does not fail. Routing the nudge through the
        # budget branch marked the task budget_exhausted, so the CLI printed a
        # failure and dropped the answer -- after the edit the nudge had asked
        # for was already on disk. Being told to wrap up is not running out.
        self.assertFalse(result.budget_exhausted,
                         "concluding is not an exhausted budget")
        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.COMPLETED)
        # A model that emits a tool call anyway, with none offered, is still
        # executed once: this asserts the boundary rather than pretending the
        # runtime refuses it. The scripted backend does exactly that, which is
        # why the count is five and not four.
        commands = [item for item in result.tool_log if item.startswith("bash")]
        self.assertEqual(len(commands), 5)

    def test_a_message_less_exception_still_names_itself(self):
        """"runtime failure" and nothing else is not a diagnosis.

        A real turn died on an exception whose str() was empty. The class was
        recorded, the summary was not, and the CLI shows the summary -- so the
        user got two words and the session record stopped at the last model
        turn. Tracing is off by default, so that string was the only evidence
        that could have existed.
        """
        class Speechless(RuntimeError):
            pass

        result = self.runtime.run(make_context(ScriptedBackend([Speechless()])))
        self.assertEqual(result.error_category, "Speechless")
        self.assertTrue(result.error_summary,
                        "an empty exception message must not empty the summary")
        self.assertIn("Speechless", result.error_summary)

    def test_model_exception_is_model_failure(self):
        result = self.runtime.run(make_context(
            ScriptedBackend([RuntimeError("offline")])
        ))
        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.MODEL_FAILURE)
        self.assertEqual(result.error_category, "RuntimeError")
        self.assertEqual(result.working_state.terminal_status, TerminalStatus.FAILED)

    def test_round_budget_forces_no_tools(self):
        backend = ScriptedBackend([text_turn("final")])
        result = self.runtime.run(make_context(backend, rounds=1))
        self.assertFalse(backend.calls[0]["use_tools"])
        self.assertEqual(result.terminal_reason,
                         RuntimeTerminalReason.ROUND_BUDGET_EXHAUSTED)
        self.assertTrue(result.budget_exhausted)

    def test_tool_budget_forces_next_round_final(self):
        backend = ScriptedBackend([
            tool_turn("one", "bash", command="ls"), text_turn("final"),
        ])
        result = self.runtime.run(make_context(backend, actions=1))
        self.assertEqual([call["use_tools"] for call in backend.calls], [True, False])
        self.assertEqual(result.terminal_reason,
                         RuntimeTerminalReason.TOOL_BUDGET_EXHAUSTED)

    def test_forced_final_synthesis_preserves_observed_work(self):
        backend = ScriptedBackend([
            tool_turn("one", "bash", command="ls"),
            text_turn(""),
            text_turn("synthesized"),
        ])
        result = self.runtime.run(make_context(backend, rounds=2))
        self.assertEqual(result.final_response, "synthesized")
        self.assertEqual(len(backend.calls), 3)
        self.assertFalse(backend.calls[-1]["use_tools"])

    def test_empty_forced_synthesis_degrades_to_evidence(self):
        backend = ScriptedBackend([
            tool_turn("one", "bash", command="ls"), text_turn(""), text_turn(""),
        ])
        result = self.runtime.run(make_context(backend, rounds=2))
        self.assertIn("without reaching a conclusion", result.final_response)
        self.assertIn("OK: done", result.final_response)

    def test_verification_reprompt(self):
        executor = GroundedToolExecutor(["OK: changed", "OK: tests"])
        backend = ScriptedBackend([
            tool_turn("edit", "edit_file", path="x.py"),
            text_turn("implemented"),
            tool_turn("verify", "bash", command="pytest"),
            text_turn("verified"),
        ])
        result = self.runtime.run(make_context(backend, executor=executor, rounds=6))
        self.assertEqual(result.final_response, "verified")
        self.assertEqual(result.working_state.verification_reprompt_count, 1)
        self.assertEqual(result.working_state.retry_count, 1)

    def test_working_state_transitions_are_grounded(self):
        executor = GroundedToolExecutor(["OK: changed"])
        result = self.runtime.run(make_context(
            ScriptedBackend([tool_turn("edit", "edit_file", path="real.py"),
                             text_turn("I changed invented.py")]),
            executor=executor,
        ))
        self.assertEqual(result.working_state.modified_files, {"real.py"})
        self.assertNotIn("invented.py", result.working_state.relevant_files)

    def test_context_engine_runs_before_every_primary_call(self):
        engine = CountingContextEngine()
        backend = ScriptedBackend([
            tool_turn("one", "bash", command="ls"), text_turn("done"),
        ])
        self.runtime.run(make_context(backend, engine=engine))
        self.assertEqual(engine.compose_count, 2)

    def test_task_interruption(self):
        result = self.runtime.run(make_context(
            ScriptedBackend([KeyboardInterrupt()])
        ))
        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.INTERRUPTED)
        self.assertEqual(result.terminal_status, TerminalStatus.INTERRUPTED)

    def test_refusal_is_typed_terminal_result(self):
        result = self.runtime.run(make_context(
            ScriptedBackend([text_turn("cannot", StopReason.REFUSAL)])
        ))
        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.REFUSAL)
        self.assertEqual(result.final_response, "cannot")

    def test_deferred_completion_supports_controller_compatibility_step(self):
        context = make_context(ScriptedBackend([text_turn("done")]))
        result = self.runtime.run(context, defer_completion=True)
        self.assertEqual(context.working_state.terminal_status, TerminalStatus.RUNNING)
        self.runtime.complete(context, result)
        self.assertEqual(result.terminal_status, TerminalStatus.COMPLETED)
        self.assertFalse(result.completion_deferred)


class ContextOwnershipTests(unittest.TestCase):
    def test_two_contexts_have_isolated_working_state_and_task_ids(self):
        runtime = AgentRuntime()
        first = make_context(ScriptedBackend([text_turn("one")]), task_id="one")
        second = make_context(ScriptedBackend([text_turn("two")]), task_id="two")
        one = runtime.run(first)
        two = runtime.run(second)
        self.assertNotEqual(one.task_id, two.task_id)
        self.assertIsNot(one.working_state, two.working_state)
        self.assertEqual(one.final_response, "one")
        self.assertEqual(two.final_response, "two")

    def test_runtime_object_is_reusable_without_state_leak(self):
        runtime = AgentRuntime()
        first = runtime.run(make_context(
            ScriptedBackend([tool_turn("x", "bash", command="pwd"), text_turn("a")]),
            task_id="reuse_a",
        ))
        second = runtime.run(make_context(
            ScriptedBackend([text_turn("b")]), task_id="reuse_b",
        ))
        self.assertEqual(len(first.tool_log), 1)
        self.assertEqual(second.tool_log, ())
        self.assertEqual(second.working_state.tool_actions, 0)

    def test_compaction_artifact_is_context_owned_and_isolated(self):
        old_messages = tuple(
            ConversationMessage("user" if index % 2 == 0 else "assistant",
                                (TextBlock((f"old {index} ") * 80),))
            for index in range(9)
        )
        policy = CompactionPolicy(
            mode=CompactionMode.AUTOMATIC, pressure_threshold=.2,
            recent_tail_groups=1, minimum_compactable_tokens=1,
            max_summary_chars=512, max_retries=0,
        )
        first = make_context(
            ScriptedBackend([text_turn("one")]), task_id="compact_one",
            conversation=old_messages, context_limit=1800, policy=policy,
        )
        second = make_context(
            ScriptedBackend([text_turn("two")]), task_id="compact_two",
            policy=policy,
        )
        runtime = AgentRuntime()
        runtime.run(first)
        runtime.run(second)
        self.assertIsNotNone(first.compaction_artifact)
        self.assertIsNone(second.compaction_artifact)
        self.assertEqual(first.compaction_artifact.structured_state.task_id,
                         "compact_one")

    def test_failed_compaction_preserves_previous_artifact(self):
        state = WorkingState.start("artifact", "Preserve artifact")
        policy = CompactionPolicy(
            mode=CompactionMode.MANUAL, recent_tail_groups=1,
            minimum_compactable_tokens=1, max_retries=0,
        )
        items = tuple(ContextItem(
            f"conversation:{index}", ContextLayer.RECENT_CONVERSATION,
            "test", "old " * 100, freshness=Freshness.STALE,
            inclusion_reason="old", eviction_group=f"g:{index}",
        ) for index in range(5))
        service = CompactionService(policy=policy)
        artifact = service.compact(CompactionRequest(
            state,
            __import__("context_engine").ContextRequest(
                items, context_limit=2000, output_reserve=0, safety_margin=0,
            ),
        )).artifact
        backend = ScriptedBackend(
            [text_turn("done")], compaction_output="malformed",
        )
        context = make_context(
            backend, task_id="artifact", conversation=tuple(
                ConversationMessage("user" if i % 2 == 0 else "assistant",
                                    (TextBlock("new " * 100),))
                for i in range(9)
            ), context_limit=1800, policy=CompactionPolicy(
                pressure_threshold=.2, recent_tail_groups=1,
                minimum_compactable_tokens=1, max_retries=0,
            ),
        )
        context.working_state = state
        context.compaction_artifact = artifact
        AgentRuntime().run(context)
        self.assertIs(context.compaction_artifact, artifact)

    def test_automatic_compaction_populates_summary_layer(self):
        messages = tuple(
            ConversationMessage("user" if i % 2 == 0 else "assistant",
                                (TextBlock("history " * 100),))
            for i in range(9)
        )
        backend = ScriptedBackend([text_turn("done")])
        context = make_context(
            backend, conversation=messages, context_limit=1800,
            policy=CompactionPolicy(
                pressure_threshold=.2, recent_tail_groups=1,
                minimum_compactable_tokens=1, max_retries=0,
            ),
        )
        AgentRuntime().run(context)
        self.assertIsNotNone(context.compaction_artifact)
        primary = [call for call in backend.calls
                   if "compact agent context" not in call["system"]][-1]
        self.assertIn("Conversation continuity", primary["system"])

    def test_trace_events_share_explicit_task_id_and_model_purpose(self):
        recorder = MemoryRecorder()
        context = make_context(
            ScriptedBackend([text_turn("done")]), task_id="trace_task",
            trace=TraceEmitter(recorder),
        )
        AgentRuntime().run(context)
        self.assertTrue(recorder.events)
        self.assertEqual({event.task_id for event in recorder.events}, {"trace_task"})
        started = next(event for event in recorder.events
                       if event.event_type == EventType.MODEL_CALL_STARTED)
        self.assertEqual(started.metadata["purpose"], "primary_agent")

    def test_runtime_has_no_transitional_global_dependency(self):
        source = Path(__file__).resolve().parents[1].joinpath(
            "agent_runtime.py"
        ).read_text()
        self.assertNotIn("CURRENT_WORKING_STATE", source)
        self.assertNotIn("CURRENT_COMPACTION_ARTIFACT", source)
        self.assertNotIn("import rag_chat", source)


class ProductionWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rag_chat
        cls.rag_chat = rag_chat

    def test_real_transitional_tool_boundary_runs_without_cli_or_globals(self):
        from tool_runtime import AuditLogger, ExecutionMode, Workspace

        backend = ScriptedBackend([
            tool_turn("write", "write_file", path="created.txt", content="value"),
            text_turn("implemented"),
            tool_turn("verify", "bash", command="test -f created.txt"),
            text_turn("created"),
        ])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = {
                "WORKSPACE": self.rag_chat.WORKSPACE,
                "PROJECT_ROOT": self.rag_chat.PROJECT_ROOT,
                "EXECUTION_MODE": self.rag_chat.EXECUTION_MODE,
                "AUDIT_LOGGER": self.rag_chat.AUDIT_LOGGER,
            }
            self.rag_chat.WORKSPACE = Workspace.from_path(root)
            self.rag_chat.PROJECT_ROOT = str(root)
            self.rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
            self.rag_chat.AUDIT_LOGGER = AuditLogger(root / "audit.jsonl")
            context = make_context(backend, task_id="production_adapter")
            context.tool_executor = lambda ctx, call_id, name, args, cache: self.rag_chat.route_tool_envelope(
                name, dict(args), cache, task_id=ctx.task_id,
                trace=ctx.trace, tool_call_id=call_id,
            )
            try:
                result = AgentRuntime().run(context)
            finally:
                for name, value in old.items():
                    setattr(self.rag_chat, name, value)
            # The turn wrote a file and ran nothing after it, so the
            # answer carries the unverified-write note. What this test
            # pins is the model's own text reaching the result.
            self.assertTrue(result.final_response.startswith("created"))
            self.assertIn("created.txt changed", result.final_response)
            self.assertEqual((root / "created.txt").read_text(), "value\n")
            self.assertEqual(result.working_state.created_files, {"created.txt"})


if __name__ == "__main__":
    unittest.main()
