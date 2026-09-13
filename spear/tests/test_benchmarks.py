import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from benchmarks.acceptance import ORACLE_DIR, run_acceptance_tests
from benchmarks.config import CONFIGURATIONS, get_configuration
from benchmarks.reporting import aggregate, write_report
from benchmarks.runner import campaign_db, main, run_one
from benchmarks.scoring import score_task, summarize_trace
from benchmarks.tasks import (
    TASKS, create_fixture, create_memory_fixture, fixture_version, get_tasks,
)
from agent_roles import AgentRole
from memory_store import MarkdownMemoryStore
import project_build
from tool_exposure import ToolExposurePolicy
from tool_registry import ToolRegistry, native_tool_specs
from tool_runtime import (
    CommandClassification, CommandPolicy, ToolResult, Workspace,
)


# What a run that tested its own change leaves in the trace. Tests asserting
# a full pass pass this, so the agent-verification oracle is exercised rather
# than accidentally disabled.
VERIFIED = {"runtime_verifications": 1, "runtime_verification_categories": ["unit_test"]}


class BenchmarkInfrastructureTests(unittest.TestCase):
    def test_task_corpus_is_diverse_and_fixture_isolated(self):
        self.assertGreaterEqual(len(TASKS), 15)
        self.assertGreaterEqual(len({task.task_class for task in TASKS}), 6)
        with tempfile.TemporaryDirectory() as directory:
            task = get_tasks(("control-edit",))[0]
            initial = create_fixture(task, directory)
            self.assertTrue((Path(directory) / "main.py").exists())
            self.assertEqual(fixture_version(task), fixture_version(task))
            self.assertTrue(initial)

    def test_scoring_is_objective_and_detects_forbidden_changes(self):
        task = get_tasks(("control-edit",))[0]
        with tempfile.TemporaryDirectory() as directory:
            initial = create_fixture(task, directory)
            path = Path(directory) / "main.py"
            path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            self.assertTrue(score_task(task, directory, initial, VERIFIED).task_success)
            (Path(directory) / "unrelated.py").write_text("bad\n", encoding="utf-8")
            self.assertFalse(score_task(task, directory, initial, VERIFIED).task_success)

    def test_every_task_declares_an_oracle(self):
        for task in TASKS:
            with self.subTest(task=task.task_id):
                self.assertTrue(task.oracles, "task can decide nothing")

    def test_no_task_passes_on_its_own_untouched_fixture(self):
        """The property that makes the suite a measurement.

        Four tasks used to declare neither expected content nor required
        reads, so score_task had nothing to check and returned success for
        every run of them -- including a run that did nothing at all. This
        asserts the opposite for all of them: a fixture nobody has worked on
        fails its own task.
        """

        for task in TASKS:
            with self.subTest(task=task.task_id), tempfile.TemporaryDirectory() as directory:
                initial = create_fixture(task, directory)
                score = score_task(task, directory, initial, {"read_paths": []})
                self.assertFalse(score.task_success, f"{task.task_id} passes untouched")
                self.assertTrue(score.failure_reason)

    def test_a_task_with_no_oracle_cannot_be_scored_a_success(self):
        empty = replace(get_tasks(("control-edit",))[0], expected={},
                        acceptance={}, forbidden_content={}, required_reads=(),
                        expected_answer=(), forbidden_answer=(),
                        requires_agent_verification=False)
        with tempfile.TemporaryDirectory() as directory:
            initial = create_fixture(empty, directory)
            score = score_task(empty, directory, initial)
            self.assertFalse(score.task_success)
            self.assertIn("no oracle", score.failure_reason)

    def test_run_state_is_isolated_from_the_operators_own(self):
        """Nothing a run accumulates may land in the real checkout.

        History, trajectories, sessions, checkpoints and the training corpus
        all hang off SPEAR_STATE_DIR. Left at its default they went into the
        operator's tree, so a benchmark campaign wrote benchmark trajectories
        into the corpus of the next fine-tune.
        """

        task = get_tasks(("control-symbol",))[0]
        with tempfile.TemporaryDirectory() as directory, patch(
                "benchmarks.runner.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = ""
            run.return_value.stderr = ""
            row = run_one(task, get_configuration("full"), Path(directory), 1,
                          real=True, timeout=1)
            child = next(item for item in run.call_args_list if "env" in item.kwargs)
            state = Path(child.kwargs["env"]["SPEAR_STATE_DIR"])
            self.assertTrue(state.is_dir())
            self.assertTrue(state.is_relative_to(Path(directory).resolve()))
            self.assertIn(row["run_id"], state.parts)
            answer = Path(child.kwargs["env"]["SPEAR_BENCH_ANSWER_FILE"])
            self.assertTrue(answer.parent.samefile(directory))

    def test_every_fixture_test_runs_with_an_allowlisted_command(self):
        """The visible tests must be runnable by something the harness allows.

        They were bare pytest functions and pytest is not a dependency here:
        told to run the test, a run reached for `python3 -c`, which is
        correctly refused, and finished having verified nothing. unittest is
        in the standard library and `python -m unittest` is allowlisted.
        """

        policy = CommandPolicy()
        command = "python3 -m unittest discover"
        self.assertEqual(policy.classify(command).classification,
                         CommandClassification.WORKSPACE_MUTATING)
        self.assertEqual(policy.classify('python3 -c "import main"').classification,
                         CommandClassification.DANGEROUS)

        for task in TASKS:
            if not any(name.startswith("tests/") for name in task.files):
                continue

            with self.subTest(task=task.task_id), tempfile.TemporaryDirectory() as directory:
                create_fixture(task, directory)
                self.assertIn("tests/__init__.py", task.files,
                              "unittest discovery collects nothing without it")
                result = subprocess.run(
                    [sys.executable, "-m", "unittest", "discover"], cwd=directory,
                    capture_output=True, text=True, timeout=60)
                self.assertNotIn("NO TESTS RAN", result.stderr)
                self.assertNotIn("Ran 0 tests", result.stderr)

    def test_a_local_task_is_offered_the_tools_it_needs(self):
        """The reported failure, at the task that produced it.

        "Find the failing assertion in the long evidence file and correct
        main.py" came back with four tool schemas and no edit_file, so the
        run reached for the network tool it HAD been given and called
        fetch_url(save_as="main.py") to write a local file.
        """

        registry = ToolRegistry()

        for item in native_tool_specs():
            registry.register(item, lambda *args, **kwargs: "OK")

        for task_id in ("context-evidence", "control-edit"):
            task = get_tasks((task_id,))[0]
            view = ToolExposurePolicy().select(
                registry, AgentRole.MAIN, objective=task.objective,
                web_enabled=True)

            with self.subTest(task=task_id):
                for name in ("bash", "edit_file", "write_file", "search_corpus"):
                    self.assertIn(name, view.names, task.objective)

                self.assertNotIn("fetch_url", view.names, task.objective)

        # control-doc is the exception, and on purpose: it forbids editing.

        doc = get_tasks(("control-doc",))[0]
        view = ToolExposurePolicy().select(
            registry, AgentRole.MAIN, objective=doc.objective, web_enabled=True)
        self.assertIn("bash", view.names)
        self.assertNotIn("edit_file", view.names)
        self.assertNotIn("fetch_url", view.names)

    def test_control_doc_is_the_only_read_only_task(self):
        """Its objective forbids editing; no other task's does.

        The fixture stays as it is: control-doc keeps `forbidden_paths`, so a
        run that edits anyway fails it, and that failure is the scope
        violation being measured.
        """

        read_only = [task.task_id for task in TASKS
                     if ToolExposurePolicy.read_only_intent(task.objective)]

        self.assertEqual(read_only, ["control-doc"])

        registry = ToolRegistry()

        for item in native_tool_specs():
            registry.register(item, lambda *args, **kwargs: "OK")

        task = get_tasks(("control-doc",))[0]
        view = ToolExposurePolicy().select(
            registry, AgentRole.MAIN, objective=task.objective, web_enabled=True)

        self.assertTrue(view.read_only)
        self.assertIn("bash", view.names)
        self.assertIn("search_corpus", view.names)
        self.assertNotIn("edit_file", view.names)
        self.assertEqual(task.forbidden_paths, ("main.py", "tests/test_main.py"))

    def test_a_run_may_write_to_its_fixture_and_nowhere_else(self):
        """--single-root, whatever the operator's corpus registry holds.

        By default the child declares every registered corpus as an additional
        writable root and says so in its prompt: a benchmark run was offering
        the model twenty-odd real repositories to edit.
        """

        task = get_tasks(("control-symbol",))[0]
        with tempfile.TemporaryDirectory() as directory, patch(
                "benchmarks.runner.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = ""
            run.return_value.stderr = ""
            run_one(task, get_configuration("full"), Path(directory), 1,
                    real=True, timeout=1)
            child = next(item for item in run.call_args_list if "env" in item.kwargs)
            self.assertIn("--single-root", child.args[0])
            self.assertNotIn("--allow-absolute-paths", child.args[0])

    def test_the_fixtures_test_command_is_inferred_only_for_the_benchmark(self):
        """The harness may infer it here; a real tree says how it is tested."""

        task = get_tasks(("control-edit",))[0]
        with tempfile.TemporaryDirectory() as directory, patch(
                "benchmarks.runner.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = ""
            run.return_value.stderr = ""
            run_one(task, get_configuration("full"), Path(directory), 1,
                    real=True, timeout=1)
            child = next(item for item in run.call_args_list if "env" in item.kwargs)
            self.assertEqual(child.kwargs["env"]["SPEAR_INFER_TEST_COMMAND"], "1")
            self.assertNotIn("SPEAR_PROJECT_VERIFY_ON_HOST", child.kwargs["env"])

        with tempfile.TemporaryDirectory() as directory:
            create_fixture(task, directory)
            self.assertEqual(project_build.probe(directory).test, "")
            self.assertEqual(
                project_build.probe(directory, infer_unittest=True).test,
                "python3 -m unittest discover -s tests")

    def test_the_campaign_indexes_into_its_own_chroma_database(self):
        """Never the production one.

        Every fixture is a fresh temporary directory, so its collection name
        is unique: a campaign pointed at the real database leaves one dead
        collection in it per task per repetition.
        """

        tasks = get_tasks(("control-symbol", "control-doc"))
        with tempfile.TemporaryDirectory() as directory, patch(
                "benchmarks.runner.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = ""
            run.return_value.stderr = ""
            seen = set()

            for task in tasks:
                row = run_one(task, get_configuration("full"), Path(directory), 1,
                              real=True, timeout=1)
                seen.add(row["db_reference"])

            database = Path(seen.pop())
            self.assertFalse(seen, "the campaign must share one database")
            self.assertTrue(database.is_dir())
            self.assertEqual(database, campaign_db(Path(directory)))
            self.assertTrue(database.is_relative_to(Path(directory).resolve()))

            for child in (item for item in run.call_args_list if "env" in item.kwargs):
                self.assertEqual(child.kwargs["env"]["SPEAR_DB_PATH"], str(database))

    def test_campaign_description_records_the_stores_it_used(self):
        with tempfile.TemporaryDirectory() as directory:
            main(["--smoke", "--task", "control-symbol", "--output", directory])
            campaign = json.loads(
                (Path(directory) / "campaign.json").read_text(encoding="utf-8"))
            self.assertEqual(campaign["chroma_db"], str(campaign_db(Path(directory))))
            self.assertEqual(campaign["state_root"],
                             str(Path(directory).resolve() / "state"))
            self.assertEqual(campaign["tasks"], ["control-symbol"])
            self.assertNotIn("prompt", json.dumps(campaign).lower())

    def test_acceptance_tests_are_executed_not_pattern_matched(self):
        """The expected fragment can be present in code that does not work."""

        task = get_tasks(("control-edit",))[0]
        with tempfile.TemporaryDirectory() as directory:
            initial = create_fixture(task, directory)
            (Path(directory) / "main.py").write_text(
                "def add(a, b):\n    return a - b  # return a + b\n", encoding="utf-8")
            score = score_task(task, directory, initial)
            self.assertTrue(score.expected_files_ok)
            self.assertFalse(score.acceptance_tests_passed)
            self.assertFalse(score.task_success)
            self.assertEqual(score.failure_reason, "acceptance tests failed")

            (Path(directory) / "main.py").write_text(
                "def add(a, b):\n    return a + b\n", encoding="utf-8")
            score = score_task(task, directory, initial, VERIFIED)
            self.assertTrue(score.acceptance_tests_passed)
            self.assertTrue(score.task_success)

    def test_acceptance_run_leaves_the_scored_workspace_untouched(self):
        task = get_tasks(("control-edit",))[0]
        with tempfile.TemporaryDirectory() as directory:
            initial = create_fixture(task, directory)
            (Path(directory) / "main.py").write_text(
                "def add(a, b):\n    return a + b\n", encoding="utf-8")
            score = score_task(task, directory, initial)
            self.assertEqual(score.changed_files, ("main.py",))
            self.assertFalse(list(Path(directory).rglob("__pycache__")))

    def test_the_oracle_is_not_the_test_the_model_can_edit(self):
        """Emptying the fixture's own test must change nothing about the score.

        Running the tests the model may have modified is not an oracle: the
        cheapest way to make a suite pass is to delete it. The task's own
        tests decide, and the run never sees them.
        """

        task = get_tasks(("control-edit",))[0]
        self.assertNotIn("tests/test_main.py", task.acceptance)

        with tempfile.TemporaryDirectory() as directory:
            initial = create_fixture(task, directory)
            (Path(directory) / "tests" / "test_main.py").write_text("", encoding="utf-8")

            # Emptied tests, unfixed code: still a failure.

            self.assertFalse(score_task(task, directory, initial, VERIFIED).task_success)

            (Path(directory) / "main.py").write_text(
                "def add(a, b):\n    return a + b\n", encoding="utf-8")
            score = score_task(task, directory, initial, VERIFIED)
            self.assertTrue(score.acceptance_tests_passed)
            self.assertTrue(score.task_success)

    def test_a_test_that_passes_by_asserting_nothing_is_not_a_pass(self):
        task = replace(get_tasks(("control-edit",))[0],
                       acceptance={"test_empty.py": "# nothing here\n"})
        with tempfile.TemporaryDirectory() as directory:
            initial = create_fixture(task, directory)
            (Path(directory) / "main.py").write_text(
                "def add(a, b):\n    return a + b\n", encoding="utf-8")
            score = score_task(task, directory, initial)
            self.assertFalse(score.task_success)
            self.assertEqual(score.acceptance_reports[0]["status"], "empty")

    def test_a_correct_file_the_run_never_tested_is_not_a_task_success(self):
        """The observed control-edit run, reproduced.

        Qwen fixed main.py, tried `python3 -c` twice, was refused both times,
        never ran unittest, and ended on "UNVERIFIED — main.py changed without
        sufficient current verification". The hidden oracle was green and the
        benchmark reported success=true, 100%. The oracle proves the file is
        right. It says nothing about whether the run did what the task asked,
        and the task asked, in those words, to run the test.
        """

        task = get_tasks(("control-edit",))[0]
        self.assertTrue(task.requires_agent_verification)
        self.assertIn("Run the test", task.objective)

        with tempfile.TemporaryDirectory() as directory:
            initial = create_fixture(task, directory)
            (Path(directory) / "main.py").write_text(
                "def add(a, b):\n    return a + b\n", encoding="utf-8")

            # The trace of that run: two refused commands, no verification.

            trace = {"read_paths": ["main.py"], "runtime_verifications": 0,
                     "tool_calls": 4, "verifications_passed": 0}
            score = score_task(task, directory, initial, trace)
            self.assertTrue(score.functionally_correct)
            self.assertTrue(score.acceptance_tests_passed)
            self.assertFalse(score.runtime_verified)
            self.assertFalse(score.task_success)
            self.assertIn("no verification covering it", score.failure_reason)

            # The same workspace, by a run that did run the tests.

            tested = score_task(task, directory, initial,
                                dict(trace, runtime_verifications=1))
            self.assertTrue(tested.runtime_verified)
            self.assertTrue(tested.task_success)

    def test_only_a_covering_verification_counts(self):
        """Inconclusive does not count, and partial does not either."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("\n".join(json.dumps(event) for event in [
                # An unrelated compile: classified, never passed.
                {"event_type": "verification_classified",
                 "metadata": {"category": "unrelated", "coverage": "none",
                              "proves_change": False}},
                # A narrowed test run: real, but it does not cover the change.
                {"event_type": "verification_passed",
                 "metadata": {"category": "unit_test", "coverage": "partial",
                              "proves_change": True, "project_bench": False}},
            ]) + "\n", encoding="utf-8")
            self.assertEqual(summarize_trace(path)["runtime_verifications"], 0)

            path.write_text(path.read_text(encoding="utf-8") + json.dumps(
                {"event_type": "verification_passed",
                 "metadata": {"category": "unit_test", "coverage": "full",
                              "proves_change": True, "project_bench": False}})
                + "\n", encoding="utf-8")
            metrics = summarize_trace(path)
            self.assertEqual(metrics["runtime_verifications"], 1)
            self.assertEqual(metrics["runtime_verification_categories"], ["unit_test"])

    def test_a_cached_repeat_counts_as_a_repeated_action(self):
        """A call answered from cache obtained nothing, whatever its status."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("\n".join(json.dumps(event) for event in [
                {"event_type": "tool_call_started", "action_id": "a1"},
                {"event_type": "tool_call_finished", "action_id": "a1"},
                {"event_type": "repeated_action_detected", "action_id": "a2",
                 "metadata": {"kind": "cached_tool_call"}},
                {"event_type": "repeated_action_detected", "action_id": "a3",
                 "metadata": {"kind": "cached_action_repeated",
                              "terminal_repeat": False}},
            ]) + "\n", encoding="utf-8")

            self.assertEqual(summarize_trace(path)["repeated_actions"], 2)

    def test_the_harness_running_the_suite_counts_as_the_run_verifying(self):
        """It is part of what the complete agent does.

        A run shown a green suite by the harness must not be made to run the
        same suite again to prove what was already observed. The benchmark's
        own hidden oracle is the thing that does not count -- and it cannot:
        it runs out here, after the fact, and never reaches the trace.
        """

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(json.dumps(
                {"event_type": "verification_passed",
                 "metadata": {"category": "integration_test", "coverage": "full",
                              "proves_change": True, "project_bench": True}})
                + "\n", encoding="utf-8")
            metrics = summarize_trace(path)
            self.assertEqual(metrics["runtime_verifications"], 1)

            task = get_tasks(("control-edit",))[0]

            with tempfile.TemporaryDirectory() as workspace:
                initial = create_fixture(task, workspace)
                (Path(workspace) / "main.py").write_text(
                    "def add(a, b):\n    return a + b\n", encoding="utf-8")
                score = score_task(task, workspace, initial, metrics)
                self.assertTrue(score.runtime_verified)
                self.assertTrue(score.task_success)

    def test_the_manifest_names_task_success_explicitly(self):
        task = get_tasks(("control-symbol",))[0]
        with tempfile.TemporaryDirectory() as directory:
            row = run_one(task, get_configuration("full"), Path(directory), 1,
                          real=False, timeout=1)

        self.assertIn("task_success", row)
        self.assertEqual(row["task_success"], row["success"])
        self.assertIn("functionally_correct", row)

    def test_the_report_separates_the_two_verdicts(self):
        rows = [{"configuration": {"name": "full"}, "task_class": "A-control",
                 "success": False, "functionally_correct": True,
                 "duration_seconds": 2.0}]
        summary = aggregate(rows)
        self.assertEqual(summary["success_rate"], 0.0)
        self.assertEqual(summary["functional_rate"], 1.0)

        with tempfile.TemporaryDirectory() as directory:
            write_report(rows, directory)
            text = (Path(directory) / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Task successes: 0 (0.0%)", text)
            self.assertIn("Functionally correct: 1 (100.0%)", text)

    def test_no_oracle_test_is_ever_written_into_a_fixture(self):
        for task in TASKS:
            with self.subTest(task=task.task_id):
                self.assertFalse(set(task.acceptance) & set(task.files))

                with tempfile.TemporaryDirectory() as directory:
                    create_fixture(task, directory)
                    present = {path.name for path in Path(directory).rglob("*")}
                    self.assertFalse(present & set(task.acceptance))
                    self.assertFalse((Path(directory) / ORACLE_DIR).exists())

    def test_oracle_tests_run_under_the_harness_confinement(self):
        """The code under test was written by the model; it runs contained.

        The marker below is the discriminating part: Bubblewrap starts the
        command with --clearenv, so an environment that still carries the
        scorer's variables is an environment the sandbox never built. Network
        closure is asserted beside it, because the point of running the tests
        under the harness contract is that a test the model wrote cannot
        reach out of the machine.
        """

        contained = replace(get_tasks(("control-edit",))[0], acceptance={
            "test_confinement.py": (
                "import os\nimport socket\n\n\n"
                "def test_the_scorer_environment_did_not_come_along():\n"
                "    assert os.environ.get('SPEAR_ACCEPTANCE_PROBE') is None\n\n\n"
                "def test_network_is_closed():\n"
                "    try:\n"
                "        socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
                "    except OSError:\n"
                "        return\n"
                "    raise AssertionError('network reachable inside the sandbox')\n"),
        })
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, {"SPEAR_ACCEPTANCE_PROBE": "1"}):
            initial = create_fixture(contained, directory)
            (Path(directory) / "main.py").write_text(
                "def add(a, b):\n    return a + b\n", encoding="utf-8")
            score = score_task(contained, directory, initial)
            self.assertTrue(score.acceptance_tests_passed, score.acceptance_reports)

    def test_an_unavailable_sandbox_is_not_a_pass(self):
        class Refusing:
            def run_sandboxed(self, *args, **kwargs):
                return ToolResult("failed", "bubblewrap sandbox unavailable")

        task = get_tasks(("control-edit",))[0]
        with tempfile.TemporaryDirectory() as directory:
            create_fixture(task, directory)
            passed, reports = run_acceptance_tests(
                directory, dict(task.acceptance), runner=Refusing())
            self.assertFalse(passed)
            self.assertEqual(reports[0]["status"], "not_run")

    def test_informative_tasks_are_scored_on_the_answer(self):
        """Reading the right files is not the same as answering correctly."""

        task = get_tasks(("call-chain",))[0]
        reads = {"read_paths": list(task.required_reads)}
        with tempfile.TemporaryDirectory() as directory:
            initial = create_fixture(task, directory)

            for answer, ok in (
                (None, False),
                ("The persistence function is handle() in service.py.", False),
                ("storage.py defines save(), which is the persistence function.", True),
                ("STORAGE.PY defines SAVE()", True),
                ("save() in storage.py, see notes.txt", False),
            ):
                with self.subTest(answer=answer):
                    score = score_task(task, directory, initial, reads, answer=answer)
                    self.assertEqual(score.task_success, ok)
                    self.assertEqual(score.answer_ok, ok)

    def test_memory_fixture_is_written_outside_the_workspace(self):
        task = get_tasks(("memory-preference",))[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            memories = Path(directory) / "memories.md"
            create_fixture(task, root)
            self.assertEqual(create_memory_fixture(task, memories), len(task.memories))
            self.assertFalse(list(root.rglob("*.md")))

            # The superseded preference stays in the file and is never offered:
            # that supersession is exactly what the task measures.

            offered = MarkdownMemoryStore(memories).select("output format")
            contents = [item.content for item in offered.records]
            self.assertIn("Output format for this project is JSON.", contents)
            self.assertNotIn("Output format for this project is XML.", contents)

    def test_repeated_actions_counts_repetition_not_action_lifecycle(self):
        """One action emits many events; that is not a repeat.

        Counting events that shared an action id made every ordinary action
        look like a repetition of itself, so the counter measured how
        talkative the trace was.
        """

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("\n".join([
                json.dumps({"event_type": "tool_call_started", "action_id": "a1"}),
                json.dumps({"event_type": "tool_call_finished", "action_id": "a1"}),
                json.dumps({"event_type": "file_read", "action_id": "a1",
                            "metadata": {"path": "main.py"}}),
                json.dumps({"event_type": "tool_call_started", "action_id": "a2"}),
                json.dumps({"event_type": "tool_call_finished", "action_id": "a2"}),
            ]) + "\n", encoding="utf-8")
            metrics = summarize_trace(path)
            self.assertEqual(metrics["repeated_actions"], 0)
            self.assertEqual(metrics["distinct_actions"], 2)

            path.write_text(path.read_text(encoding="utf-8") + json.dumps(
                {"event_type": "repeated_action_detected", "action_id": "a3",
                 "metadata": {"kind": "cached_tool_call"}}) + "\n", encoding="utf-8")
            self.assertEqual(summarize_trace(path)["repeated_actions"], 1)

    def test_inconclusive_verifications_are_counted_apart_from_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("\n".join([
                json.dumps({"event_type": "verification_passed",
                            "metadata": {"category": "unit_test"}}),
                json.dumps({"event_type": "verification_classified",
                            "metadata": {"category": "unknown"}}),
            ]) + "\n", encoding="utf-8")
            metrics = summarize_trace(path)
            self.assertEqual(metrics["verifications_passed"], 1)
            self.assertEqual(metrics["verifications_inconclusive"], 1)

    def test_ablation_configurations_are_explicit(self):
        self.assertEqual(set(CONFIGURATIONS), {"full", "production-default", "no-explorer", "no-reviewer",
                                                "no-planning", "no-compaction", "full-tools",
                                                "no-memory", "no-repair"})
        self.assertFalse(get_configuration("no-explorer").explorer)
        self.assertTrue(get_configuration("full").role_aware_tools)

    def test_trace_metrics_are_derived_without_prompt_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("\n".join([
                json.dumps({"event_type": "model_call_finished", "input_tokens": 10, "output_tokens": 3}),
                json.dumps({"event_type": "file_read", "metadata": {"path": "a.py"}}),
                json.dumps({"event_type": "file_modified", "metadata": {"path": "a.py"}}),
            ]) + "\n", encoding="utf-8")
            metrics = summarize_trace(path)
            self.assertEqual(metrics["model_calls"], 1)
            self.assertEqual(metrics["distinct_files_read"], 1)
            self.assertNotIn("prompt", metrics)

    def test_manifest_aggregation_and_report(self):
        rows = [{"configuration": {"name": "full"}, "task_class": "A-control",
                 "success": True, "duration_seconds": 2.0},
                {"configuration": {"name": "full"}, "task_class": "A-control",
                 "success": False, "duration_seconds": 4.0}]
        summary = aggregate(rows)
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["successes"], 1)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_report(rows, directory)
            self.assertTrue(all(path.exists() for path in paths))

    def test_task_selection(self):
        self.assertEqual([task.task_id for task in get_tasks(("control-symbol",))], ["control-symbol"])
        self.assertEqual(get_tasks(("missing",)), ())

    def test_runner_smoke_produces_safe_machine_manifest_row(self):
        task = get_tasks(("control-symbol",))[0]
        with tempfile.TemporaryDirectory() as directory:
            row = run_one(task, get_configuration("full"), Path(directory), 1,
                          real=False, timeout=1)
            self.assertFalse(row["real_model"])
            self.assertIn("fixture_version", row)
            self.assertNotIn("prompt", json.dumps(row).lower())
            self.assertFalse(list(Path(directory).glob("*.trace.jsonl")))

    def test_real_runner_marks_captured_data_as_benchmark(self):
        task = get_tasks(("control-symbol",))[0]
        with tempfile.TemporaryDirectory() as directory, patch(
                "benchmarks.runner.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = ""
            run.return_value.stderr = ""
            run_one(task, get_configuration("full"), Path(directory), 1,
                    real=True, timeout=1)
            child = next(item for item in run.call_args_list if "env" in item.kwargs)
            self.assertEqual(child.kwargs["env"]["SPEAR_TRAINING_DATA_ORIGIN"],
                             "BENCHMARK")


class ProjectVerifierConfinementTests(unittest.TestCase):
    """No host fallback, ever.

    A sandbox that cannot be established is a verification that did not
    happen. Degrading it into an unconfined run of code the model has just
    written is the one outcome worse than not verifying at all, so it is the
    operator's choice to make in advance, in writing, and never a fallback.
    """

    @classmethod
    def setUpClass(cls):
        import rag_chat

        cls.rag_chat = rag_chat

    def setUp(self):
        self.saved = (self.rag_chat.WORKSPACE, self.rag_chat.PROJECT_ROOT,
                      self.rag_chat.PROJECT_VERIFY_ON_HOST)

    def tearDown(self):
        (self.rag_chat.WORKSPACE, self.rag_chat.PROJECT_ROOT,
         self.rag_chat.PROJECT_VERIFY_ON_HOST) = self.saved

    def test_no_workspace_means_not_run_never_the_host(self):
        self.rag_chat.WORKSPACE = None
        status, detail = self.rag_chat.verify_project_command("echo never")
        self.assertEqual(status, "not_run")
        self.assertIn("SPEAR_PROJECT_VERIFY_ON_HOST", detail)

    def test_an_unavailable_sandbox_means_not_run(self):
        class Refusing:
            def ensure_sandbox(self, workspace, profile=None):
                return ToolResult("failed", "bubblewrap sandbox unavailable")

            def run_sandboxed(self, *args, **kwargs):
                raise AssertionError("must not run without a sandbox")

        with tempfile.TemporaryDirectory() as directory, patch.object(
                self.rag_chat, "PROJECT_VERIFY_RUNNER", Refusing()):
            self.rag_chat.WORKSPACE = Workspace.from_path(directory)
            status, detail = self.rag_chat.verify_project_command("echo never")

        self.assertEqual(status, "not_run")
        self.assertIn("bubblewrap sandbox unavailable", detail)

    def test_the_host_is_reachable_only_by_explicit_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            self.rag_chat.WORKSPACE = None
            self.rag_chat.PROJECT_ROOT = directory
            self.rag_chat.PROJECT_VERIFY_ON_HOST = True
            self.assertEqual(
                self.rag_chat.verify_project_command("true")[0], "passed")
            self.assertEqual(
                self.rag_chat.verify_project_command("false")[0], "failed")

    def test_a_confined_run_reports_pass_and_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            (root / "tests" / "__init__.py").write_text("", encoding="utf-8")
            (root / "main.py").write_text(
                "def add(a, b):\n    return a + b\n", encoding="utf-8")
            (root / "tests" / "test_main.py").write_text(
                "import unittest\nfrom main import add\n\n\n"
                "class T(unittest.TestCase):\n    def test_add(self):\n"
                "        self.assertEqual(add(2, 3), 5)\n", encoding="utf-8")
            self.rag_chat.WORKSPACE = Workspace.from_path(root)
            self.rag_chat.PROJECT_ROOT = str(root)
            command = project_build.probe(str(root), infer_unittest=True).test
            self.assertEqual(self.rag_chat.verify_project_command(command),
                             ("passed", ""))

            (root / "main.py").write_text(
                "def add(a, b):\n    return a - b   # broken\n", encoding="utf-8")
            status, detail = self.rag_chat.verify_project_command(command)
            self.assertEqual(status, "failed")
            self.assertIn("FAIL", detail)


class SingleRootBoundaryTests(unittest.TestCase):
    """The property `--single-root` buys, asserted where it is decided.

    The runner passing the flag is one half; this is the other: with it, the
    corpus registry contributes nothing writable and the prompt says nothing
    about other trees, however many the operator has declared.
    """

    _GLOBALS = ("PROJECT", "PROJECT_ROOT", "CORPUS_ROOT", "PROJECT_KIND",
                "WORKSPACE", "COLLECTION_NAME", "HISTORY_FILE", "MEMORIES_FILE",
                "PROJECT_SPEC")

    @classmethod
    def setUpClass(cls):
        import rag_chat

        cls.rag_chat = rag_chat

    def setUp(self):
        self.saved = {name: getattr(self.rag_chat, name, None)
                      for name in self._GLOBALS}
        self.temp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(self.rag_chat, name, value)

        if self.saved["WORKSPACE"] is not None:
            self.rag_chat.COMMAND_POLICY.bind_workspace(self.saved["WORKSPACE"])

        os.chdir(self.cwd)
        self.temp.cleanup()

    def _bind(self, argv):
        root = Path(self.temp.name) / "fixture"
        root.mkdir()
        registry = {name: {"name": name, "path": str(Path(self.temp.name) / name),
                           "kind": "generic"}
                    for name in ("so3", "lvgl", "infrabase", "spear")}

        for spec in registry.values():
            Path(spec["path"]).mkdir()

        os.chdir(root)

        with patch.object(sys, "argv", ["rag_chat.py", *argv]), patch.object(
                self.rag_chat, "load_projects", lambda: registry):
            self.rag_chat.set_project({"name": "fixture", "path": str(root),
                                       "kind": "generic"})

        return self.rag_chat.WORKSPACE

    def test_the_registry_is_writable_without_the_flag(self):
        workspace = self._bind(["--auto", "--here"])
        self.assertEqual(len(workspace.extra_roots), 4)
        self.assertIn("writable", self.rag_chat.extra_roots_note())

    def test_the_registry_contributes_nothing_with_the_flag(self):
        workspace = self._bind(["--auto", "--here", "--single-root"])
        self.assertEqual(workspace.extra_roots, ())
        self.assertEqual(self.rag_chat.extra_roots_note(), "")
        self.assertFalse(workspace.allow_absolute_paths)

        # And the boundary, not only the prompt: a declared corpus is no
        # longer reachable by the file tools.

        outside = Path(self.temp.name) / "so3" / "kernel.c"
        outside.write_text("int main(void) { return 0; }\n", encoding="utf-8")
        self.assertIn("path escapes the workspace",
                      self.rag_chat.read_file(str(outside)))


if __name__ == "__main__":
    unittest.main()
