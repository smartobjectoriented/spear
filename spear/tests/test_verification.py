import unittest

from tool_router import ToolResultEnvelope, ToolResultStatus
from verification import (
    CompletionVerificationStatus, VerificationCategory, VerificationCoverage,
    VerificationHints, VerificationPolicy,
)
from working_state import (
    ActionKind, StateEvent, StateEventType, StateSource, VerificationOutcome,
    WorkingState,
)


def mutate(state, action_id="edit", path="a.py"):
    state.apply(StateEvent.create(
        StateEventType.ACTION_SUCCEEDED, state.task_id, StateSource.TOOL_RUNTIME,
        action_id=action_id, kind=ActionKind.TOOL.value, name="edit_file",
        observed_status="ok",
    ))
    state.apply(StateEvent.create(
        StateEventType.FILE_MODIFIED, state.task_id, StateSource.TOOL_RUNTIME,
        path=path, action_id=action_id,
    ))


def envelope(command, success=True, action_id="cmd"):
    status = ToolResultStatus.OK if success else ToolResultStatus.FAILED
    return ToolResultEnvelope(
        action_id, action_id, "bash", success, status, "output", "output",
        "command", 0.1, 6, exit_code=0 if success else 1,
        error_summary=None if success else "failed",
    )


def record(state, evidence):
    state.apply(StateEvent.create(
        StateEventType.VERIFICATION_RECORDED, state.task_id,
        StateSource.TOOL_RUNTIME, verification_id=evidence.verification_id,
        kind=evidence.category.value, category=evidence.category.value,
        coverage=evidence.coverage.value, executed=evidence.executed,
        outcome=evidence.outcome.value, action_id=evidence.action_id,
        summary=evidence.summary, mutation_generation=evidence.mutation_generation,
        result_reference=evidence.result_reference, command=evidence.command,
        project_bench=evidence.project_bench,
    ))


class VerificationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = VerificationPolicy()
        self.state = WorkingState.start("task_verify", "change it")

    def test_no_mutation_needs_no_verification(self):
        result = self.policy.evaluate_completion(self.state)
        self.assertEqual(result.status, CompletionVerificationStatus.NOT_REQUIRED)

    def test_meaningless_inspection_does_not_verify_mutation(self):
        mutate(self.state)
        for index, command in enumerate(("pwd", "ls -la", "git status")):
            evidence = self.policy.evidence_for_tool(
                envelope(command, action_id=f"c{index}"), {"command": command}, self.state)
            self.assertEqual(evidence.coverage, VerificationCoverage.NONE)
            record(self.state, evidence)
        self.assertEqual(self.policy.evaluate_completion(self.state).status,
                         CompletionVerificationStatus.UNVERIFIED)

    def test_build_and_failure_classification(self):
        category, coverage = self.policy.classify_command("cmake --build build")
        self.assertEqual((category, coverage),
                         (VerificationCategory.BUILD, VerificationCoverage.FULL))
        mutate(self.state)
        evidence = self.policy.evidence_for_tool(
            envelope("make", False), {"command": "make"}, self.state)
        record(self.state, evidence)
        self.assertEqual(self.policy.evaluate_completion(self.state).status,
                         CompletionVerificationStatus.FAILED)

    def test_unit_test_success_and_targeted_partial(self):
        mutate(self.state)
        full = self.policy.evidence_for_tool(
            envelope("pytest"), {"command": "pytest"}, self.state)
        self.assertEqual(full.category, VerificationCategory.UNIT_TEST)
        self.assertEqual(full.coverage, VerificationCoverage.FULL)
        record(self.state, full)
        self.assertEqual(self.policy.evaluate_completion(self.state).status,
                         CompletionVerificationStatus.VERIFIED)
        mutate(self.state, "edit2", "b.py")
        targeted = self.policy.evidence_for_tool(
            envelope("pytest tests/test_a.py::test_one", action_id="target"),
            {"command": "pytest tests/test_a.py::test_one"}, self.state)
        record(self.state, targeted)
        self.assertEqual(self.policy.evaluate_completion(self.state).status,
                         CompletionVerificationStatus.PARTIALLY_VERIFIED)

    def test_lint_is_not_tests(self):
        category, coverage = self.policy.classify_command("ruff check .")
        self.assertEqual(category, VerificationCategory.LINT)
        self.assertEqual(coverage, VerificationCoverage.PARTIAL)

    def test_later_mutation_invalidates_prior_pass(self):
        mutate(self.state)
        passed = self.policy.evidence_for_tool(
            envelope("pytest"), {"command": "pytest"}, self.state)
        record(self.state, passed)
        mutate(self.state, "edit2", "b.py")
        self.assertEqual(self.policy.evaluate_completion(self.state).status,
                         CompletionVerificationStatus.UNVERIFIED)

    def test_failed_history_remains_after_later_success(self):
        mutate(self.state)
        failed = self.policy.evidence_for_tool(
            envelope("pytest", False, "fail"), {"command": "pytest"}, self.state)
        record(self.state, failed)
        passed = self.policy.evidence_for_tool(
            envelope("pytest", True, "pass"), {"command": "pytest"}, self.state)
        record(self.state, passed)
        self.assertEqual(len(self.state.verifications), 2)
        self.assertIn("verification:" + failed.verification_id, self.state.failures)
        self.assertEqual(self.policy.evaluate_completion(self.state).status,
                         CompletionVerificationStatus.VERIFIED)

    def test_project_bench_is_full_acceptance_evidence(self):
        mutate(self.state)
        evidence = self.policy.project_bench_evidence(
            self.state, executed=True, passed=True, action_id="bench")
        self.assertTrue(evidence.project_bench)
        self.assertEqual(evidence.coverage, VerificationCoverage.FULL)
        record(self.state, evidence)
        self.assertEqual(self.policy.evaluate_completion(self.state).status,
                         CompletionVerificationStatus.VERIFIED)

    def test_project_hints_are_provider_and_language_neutral(self):
        policy = VerificationPolicy(VerificationHints.from_project({
            "build_commands": ["custom-build"],
            "test_commands": "custom-check",
        }))
        self.assertEqual(policy.classify_command("custom-build --all")[0],
                         VerificationCategory.BUILD)
        self.assertEqual(policy.classify_command("custom-check")[0],
                         VerificationCategory.UNIT_TEST)

    def test_unrecognised_command_proves_nothing_even_when_it_succeeds(self):
        """A command nothing can classify exits zero; that is all it says.

        The benchmark saw the consequence: a denied `python3 -m pytest` was
        followed by a command the classifier did not recognise, and its
        success was recorded as `verification_passed` with category=unknown
        and coverage=unknown -- a verification, in the trace and in every
        count drawn from it, of a change nothing had tested.
        """

        mutate(self.state)
        evidence = self.policy.evidence_for_tool(
            envelope("frobnicate --all"), {"command": "frobnicate --all"}, self.state)
        self.assertEqual(evidence.category, VerificationCategory.UNKNOWN)
        self.assertEqual(evidence.coverage, VerificationCoverage.UNKNOWN)
        self.assertEqual(evidence.outcome, VerificationOutcome.PASSED)
        self.assertFalse(evidence.proves_change)
        record(self.state, evidence)
        self.assertEqual(self.policy.evaluate_completion(self.state).status,
                         CompletionVerificationStatus.UNVERIFIED)

    def test_real_verification_still_proves_the_change(self):
        mutate(self.state)
        evidence = self.policy.evidence_for_tool(
            envelope("pytest"), {"command": "pytest"}, self.state)
        self.assertTrue(evidence.proves_change)

    def test_a_build_that_exercises_nothing_that_changed_is_inconclusive(self):
        """Compiling an unrelated program is not evidence about the change.

        A benchmark run edited main.py, then wrote and compiled a standalone C
        program that reimplemented the same function. The command classified
        as build/full and was traced as a verification that passed, for a
        Python change no compiler had ever seen.
        """

        mutate(self.state, path="main.py")
        evidence = self.policy.evidence_for_tool(
            envelope("gcc -o add_check add_check.c"),
            {"command": "gcc -o add_check add_check.c"}, self.state)
        self.assertEqual(evidence.category, VerificationCategory.UNRELATED)
        self.assertEqual(evidence.coverage, VerificationCoverage.NONE)
        self.assertFalse(evidence.proves_change)
        record(self.state, evidence)
        self.assertEqual(self.policy.evaluate_completion(self.state).status,
                         CompletionVerificationStatus.UNVERIFIED)

    def test_full_coverage_needs_the_command_to_reach_the_change(self):
        for command, changed, expected in (
            # Names the file it changed: the case direct compilation exists for.
            ("gcc -c src/wire.c -o /tmp/a.o", ("src/wire.c",),
             (VerificationCategory.BUILD, VerificationCoverage.FULL)),
            # Same language, names none of them: it may have missed the change.
            ("gcc -c src/other.c", ("src/wire.c",),
             (VerificationCategory.BUILD, VerificationCoverage.PARTIAL)),
            # Wrong language entirely.
            ("gcc -o probe probe.c", ("main.py",),
             (VerificationCategory.UNRELATED, VerificationCoverage.NONE)),
            ("./probe", ("main.py",),
             (VerificationCategory.UNRELATED, VerificationCoverage.NONE)),
            ("python3 tool.py", ("src/wire.c",),
             (VerificationCategory.UNRELATED, VerificationCoverage.NONE)),
            # A build of the whole tree needs no argument about relevance.
            ("make", ("main.py",),
             (VerificationCategory.BUILD, VerificationCoverage.FULL)),
            ("python3 -m unittest discover", ("main.py",),
             (VerificationCategory.UNIT_TEST, VerificationCoverage.FULL)),
            # Targeted means targeted AT the change.
            ("grep -n add main.py", ("main.py",),
             (VerificationCategory.TARGETED_INSPECTION, VerificationCoverage.PARTIAL)),
            ("grep -n add notes.txt", ("main.py",),
             (VerificationCategory.TARGETED_INSPECTION, VerificationCoverage.UNKNOWN)),
        ):
            with self.subTest(command=command, changed=changed):
                self.assertEqual(
                    self.policy.classify_command(command, changed_paths=changed),
                    expected)

    def test_a_narrowed_unittest_run_stays_partial(self):
        """One module is not the suite, whichever runner names it.

        Naming a single test module proves that module works; calling that
        full coverage would let a turn certify a five-file change with one
        file's tests.
        """

        for command in ("python3 -m unittest tests.test_main",
                        "python3 -m unittest tests.test_main.AddTests.test_add",
                        "pytest tests/test_main.py"):
            with self.subTest(command=command):
                self.assertEqual(
                    self.policy.classify_command(command, changed_paths=("main.py",)),
                    (VerificationCategory.UNIT_TEST, VerificationCoverage.PARTIAL))

        for command in ("python3 -m unittest discover -s tests",
                        "python3 -m unittest discover", "pytest"):
            with self.subTest(command=command):
                self.assertEqual(
                    self.policy.classify_command(command, changed_paths=("main.py",)),
                    (VerificationCategory.UNIT_TEST, VerificationCoverage.FULL))

    def test_the_projects_own_bench_is_full_whatever_it_runs(self):
        """The project declared it; the harness does not second-guess it."""

        hinted = VerificationPolicy(VerificationHints(
            acceptance_commands=("./benches/run.sh",),
            test_commands=("gcc -o t t.c && ./t",)))
        self.assertEqual(
            hinted.classify_command("./benches/run.sh", changed_paths=("main.py",)),
            (VerificationCategory.INTEGRATION_TEST, VerificationCoverage.FULL))
        self.assertEqual(
            hinted.classify_command("gcc -o t t.c && ./t", changed_paths=("main.py",)),
            (VerificationCategory.UNIT_TEST, VerificationCoverage.FULL))
        self.assertEqual(
            self.policy.classify_command("anything", project_bench=True,
                                         changed_paths=("main.py",)),
            (VerificationCategory.INTEGRATION_TEST, VerificationCoverage.FULL))

    def test_assistant_prose_is_not_an_input(self):
        mutate(self.state)
        self.assertEqual(self.policy.evaluate_completion(self.state).status,
                         CompletionVerificationStatus.UNVERIFIED)

    def test_verification_generation_survives_serialization(self):
        mutate(self.state)
        evidence = self.policy.evidence_for_tool(
            envelope("pytest"), {"command": "pytest"}, self.state)
        record(self.state, evidence)
        restored = WorkingState.from_json(self.state.to_json())
        self.assertEqual(restored.mutation_generation, 1)
        self.assertEqual(restored.verifications[-1].mutation_generation, 1)
        self.assertEqual(self.policy.evaluate_completion(restored).status,
                         CompletionVerificationStatus.VERIFIED)


if __name__ == "__main__":
    unittest.main()
