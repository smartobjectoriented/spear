"""Grounded, generation-aware verification policy.

The policy classifies observed command/tool results.  It never consumes
assistant claims, and successful inspection commands do not verify changes.
"""

from __future__ import annotations

import re
import shlex
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping, Sequence

from tool_router import ToolResultEnvelope
from working_state import VerificationOutcome, WorkingState


class VerificationCategory(StrEnum):
    BUILD = "build"
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    LINT = "lint"
    STATIC_ANALYSIS = "static_analysis"
    EXECUTION = "execution"
    TARGETED_INSPECTION = "targeted_inspection"
    GENERAL_INSPECTION = "general_inspection"
    UNRELATED = "unrelated"
    UNKNOWN = "unknown"


class VerificationCoverage(StrEnum):
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"
    UNKNOWN = "unknown"


class CompletionVerificationStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    FAILED = "failed"


# What can stand as evidence that a change works.  An inspection, an
# unrelated command, and a command nothing recognised all observe something,
# but none of them observes the change: they are recorded, and they never
# count.

NON_VERIFYING_CATEGORIES = frozenset({
    VerificationCategory.GENERAL_INSPECTION,
    VerificationCategory.UNRELATED,
    VerificationCategory.UNKNOWN,
})
MEANINGFUL_COVERAGE = frozenset({
    VerificationCoverage.FULL, VerificationCoverage.PARTIAL,
})


def evidence_proves_change(category: str, coverage: str) -> bool:
    """Whether a (category, coverage) pair says anything about the change."""

    return (category not in {item.value for item in NON_VERIFYING_CATEGORIES}
            and coverage in {item.value for item in MEANINGFUL_COVERAGE})


@dataclass(frozen=True)
class VerificationEvidence:
    verification_id: str
    category: VerificationCategory
    coverage: VerificationCoverage
    executed: bool
    outcome: VerificationOutcome
    mutation_generation: int
    action_id: str | None = None
    command: str | None = None
    result_reference: str | None = None
    affected_paths: tuple[str, ...] = ()
    project_bench: bool = False
    summary: str | None = None

    @property
    def proves_change(self) -> bool:
        """Whether this evidence can stand as verification of the change.

        An unrecognised command that exits zero exits zero; it does not say
        the change works.  Callers that report an outcome -- a trace event, a
        counter, a completion gate -- must ask this first, or a turn whose
        only "verification" was a command nothing could classify reads, ever
        afterwards, as a turn that was verified.
        """

        return evidence_proves_change(self.category.value, self.coverage.value)


@dataclass(frozen=True)
class CompletionEvaluation:
    status: CompletionVerificationStatus
    mutation_generation: int
    evidence: tuple[object, ...] = ()
    reason: str = ""

    @property
    def meaningfully_verified(self) -> bool:
        return self.status == CompletionVerificationStatus.VERIFIED


@dataclass(frozen=True)
class VerificationHints:
    build_commands: tuple[str, ...] = ()
    test_commands: tuple[str, ...] = ()
    lint_commands: tuple[str, ...] = ()
    acceptance_commands: tuple[str, ...] = ()

    @classmethod
    def from_project(cls, raw: Mapping[str, object] | None) -> "VerificationHints":
        raw = raw or {}

        def values(name: str) -> tuple[str, ...]:
            value = raw.get(name, ())

            if isinstance(value, str):
                return (value,)

            if isinstance(value, Sequence) and not isinstance(value, (bytes, str)):
                return tuple(str(item) for item in value if str(item).strip())

            return ()

        return cls(values("build_commands"), values("test_commands"),
                   values("lint_commands"), values("acceptance_commands"))


_GENERAL = {"pwd", "ls", "dir", "find", "tree", "stat", "du", "df",
            "git status", "git diff", "git log", "git show"}
_UNIT = re.compile(r"(?:^|[/ ])(?:pytest|unittest|nosetests|jest|vitest|rspec|go test|cargo test|ctest)(?:\s|$)", re.I)
_INTEGRATION = re.compile(r"\b(?:integration|e2e|end[-_ ]to[-_ ]end)\b", re.I)

# A build of the whole tree exercises whatever is in the tree, so it needs no
# argument about relevance.  A compiler invoked directly does: see
# _direct_invocation below.

_BUILD_TREE = re.compile(r"(?:^|[/ ])(?:make|ninja|cmake --build|cargo build|go build|mvn|gradle|npm run build|pnpm build|yarn build)(?:\s|$)", re.I)

# A compiler invoked directly is the most basic build check there is, and it
# was the only one this list could not see: a turn that edited three C files
# and compiled each with `gcc -c` had, as far as every gate was concerned,
# verified nothing.
_COMPILER = re.compile(r"(?:^|[/ ])(?:gcc|g\+\+|cc|c\+\+|clang|clang\+\+)(?:\s|$)", re.I)
_LINT = re.compile(r"(?:^|[/ ])(?:ruff|flake8|pylint|eslint|stylelint|shellcheck|golangci-lint)(?:\s|$)", re.I)
_STATIC = re.compile(r"(?:^|[/ ])(?:mypy|pyright|tsc|clang-tidy|cppcheck|cargo check)(?:\s|$)", re.I)
_EXECUTION = re.compile(r"(?:^|[;&|]\s*)(?:python\d*|node|ruby|perl|java|dotnet run|cargo run|go run|\./[^ ]+)(?:\s|$)", re.I)
_TARGETED = re.compile(r"(?:^|[/ ])(?:diff|cmp|grep|rg)(?:\s|$)", re.I)

# What a directly invoked tool is ABOUT.  A command that neither names a
# changed file nor handles the kind of file that changed did not exercise the
# change, whatever else it did: a benchmark run whose only "verification" was
# a standalone C program reimplementing the Python function it had just edited
# was recorded as build/full and traced as passed.
#
# Build files sit in the C family because editing a Makefile and running a
# compiler is one workflow, not two.

_C_FAMILY = frozenset({".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx",
                       ".s", ".inc", ".mk", ".cmake", "makefile", "cmakelists.txt"})
_COMPILED = _C_FAMILY | {".rs", ".go", ".zig", ".d", ".m", ".mm", ".asm"}
_INTERPRETER_LANGUAGES = (
    (re.compile(r"^python\d*$"), frozenset({".py", ".pyi"})),
    (re.compile(r"^node$"), frozenset({".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"})),
    (re.compile(r"^ruby$"), frozenset({".rb"})),
    (re.compile(r"^perl$"), frozenset({".pl", ".pm"})),
    (re.compile(r"^java$"), frozenset({".java"})),
    (re.compile(r"^cargo$"), frozenset({".rs"})),
    (re.compile(r"^go$"), frozenset({".go"})),
    (re.compile(r"^dotnet$"), frozenset({".cs", ".fs", ".vb"})),
)


class VerificationPolicy:
    def __init__(self, hints: VerificationHints | None = None) -> None:
        self.hints = hints or VerificationHints()

    def classify_command(self, command: str, *, project_bench: bool = False,
                         changed_paths: Sequence[str] = ()) -> tuple[VerificationCategory, VerificationCoverage]:
        """What a command verifies, and how much of the change it covers.

        Project-supplied hints are consulted before the built-in patterns, so a
        project's own build or test command always wins over a guess.
        """

        normalized = self._normalize(command)

        if project_bench or self._matches_hint(normalized, self.hints.acceptance_commands):
            return VerificationCategory.INTEGRATION_TEST, VerificationCoverage.FULL

        if self._matches_hint(normalized, self.hints.test_commands):
            return VerificationCategory.UNIT_TEST, VerificationCoverage.FULL

        if self._matches_hint(normalized, self.hints.build_commands):
            return VerificationCategory.BUILD, VerificationCoverage.FULL

        if self._matches_hint(normalized, self.hints.lint_commands):
            return VerificationCategory.LINT, VerificationCoverage.PARTIAL

        # Looking around is not verifying: these carry NONE coverage, which is
        # what stops `ls` from being recorded as evidence that a change works.

        if normalized in _GENERAL or any(normalized.startswith(item + " ") for item in _GENERAL):
            return VerificationCategory.GENERAL_INSPECTION, VerificationCoverage.NONE

        if _INTEGRATION.search(normalized) and _UNIT.search(normalized):
            return VerificationCategory.INTEGRATION_TEST, VerificationCoverage.FULL

        # A test run narrowed to one test proves less than the whole suite, so
        # the selector syntaxes downgrade it to partial coverage.

        if _UNIT.search(normalized):
            targeted = bool(re.search(r"::|\b-k\s|--test\s|test_[\w./-]+|[\w./-]+_test\.", normalized))
            return (VerificationCategory.UNIT_TEST,
                    VerificationCoverage.PARTIAL if targeted else VerificationCoverage.FULL)

        if _BUILD_TREE.search(normalized):
            return VerificationCategory.BUILD, VerificationCoverage.FULL

        if _COMPILER.search(normalized):
            return self._direct_invocation(
                VerificationCategory.BUILD, VerificationCoverage.FULL,
                normalized, changed_paths, _C_FAMILY)

        if _LINT.search(normalized):
            return VerificationCategory.LINT, VerificationCoverage.PARTIAL

        if _STATIC.search(normalized):
            return VerificationCategory.STATIC_ANALYSIS, VerificationCoverage.PARTIAL

        if _EXECUTION.search(normalized):
            return self._direct_invocation(
                VerificationCategory.EXECUTION, VerificationCoverage.PARTIAL,
                normalized, changed_paths, self._execution_languages(normalized))

        # "Targeted" means targeted at the change. A diff or a grep of
        # something else is inspection with nothing to say about it.

        if _TARGETED.search(normalized):
            coverage = (VerificationCoverage.PARTIAL
                        if changed_paths and self._names_a_changed_path(normalized, changed_paths)
                        else VerificationCoverage.UNKNOWN)
            return VerificationCategory.TARGETED_INSPECTION, coverage

        return VerificationCategory.UNKNOWN, VerificationCoverage.UNKNOWN

    def evidence_for_tool(self, envelope: ToolResultEnvelope,
                          arguments: Mapping[str, object], state: WorkingState,
                          *, project_bench: bool = False) -> VerificationEvidence | None:
        """Evidence from an observed tool result, or None if it proves nothing."""

        if envelope.tool_name != "bash" and not project_bench:
            return None

        command = str(arguments.get("command", ""))
        category, coverage = self.classify_command(
            command, project_bench=project_bench,
            changed_paths=tuple(sorted(state.modified_files | state.created_files |
                                       state.deleted_files)),
        )

        # A command that was refused or never started did not observe anything,
        # so it is recorded as not run rather than as a failure of the code.

        executed = envelope.status.value not in {
            "denied", "cancelled", "invalid_arguments", "unknown_tool",
        }
        outcome = (VerificationOutcome.PASSED if executed and envelope.success
                   else VerificationOutcome.FAILED if executed
                   else VerificationOutcome.NOT_RUN)

        # An unrecognised command that failed says nothing about the change:
        # it may simply have been the wrong command.

        if category == VerificationCategory.UNKNOWN and not envelope.success:
            return None

        return VerificationEvidence(
            "verification_" + uuid.uuid4().hex, category, coverage,
            executed, outcome, state.mutation_generation,
            envelope.action_id, command or None, envelope.result_reference,
            tuple(envelope.affected_paths), project_bench,
            self._summary(category, coverage, outcome),
        )

    def project_bench_evidence(self, state: WorkingState, *, executed: bool,
                               passed: bool | None, action_id: str | None = None,
                               result_reference: str | None = None,
                               summary: str | None = None) -> VerificationEvidence:
        outcome = (VerificationOutcome.NOT_RUN if not executed else
                   VerificationOutcome.PASSED if passed else VerificationOutcome.FAILED)
        return VerificationEvidence(
            "verification_" + uuid.uuid4().hex,
            VerificationCategory.INTEGRATION_TEST, VerificationCoverage.FULL,
            executed, outcome, state.mutation_generation, action_id,
            result_reference=result_reference, project_bench=True,
            summary=summary or self._summary(
                VerificationCategory.INTEGRATION_TEST,
                VerificationCoverage.FULL, outcome,
            ),
        )

    def evaluate_completion(self, state: WorkingState) -> CompletionEvaluation:
        """Whether the current generation of changes is verified.

        Only evidence from THIS generation counts: every mutation invalidates
        what came before it, however thoroughly the old code was tested.
        """

        generation = state.mutation_generation

        if generation == 0:
            return CompletionEvaluation(CompletionVerificationStatus.NOT_REQUIRED,
                                        generation, reason="no grounded mutation")

        current = tuple(item for item in state.verifications
                        if item.mutation_generation == generation)
        meaningful = tuple(item for item in current
                           if evidence_proves_change(item.category, item.coverage))

        # A failure stands unless something passed after it: re-running a check
        # is how a fix is demonstrated, and the later result is the current one.

        if any(item.outcome == VerificationOutcome.FAILED for item in meaningful):
            latest_meaningful = meaningful[-1]
            later_pass = any(
                item.outcome == VerificationOutcome.PASSED
                for item in meaningful[meaningful.index(latest_meaningful) + 1:]
            )

            if latest_meaningful.outcome == VerificationOutcome.FAILED and not later_pass:
                return CompletionEvaluation(CompletionVerificationStatus.FAILED,
                                            generation, meaningful,
                                            "latest meaningful verification failed")

        passed = tuple(item for item in meaningful
                       if item.outcome == VerificationOutcome.PASSED)

        # VERIFIED needs one check of full coverage; any number of partial ones
        # do not add up to it.

        if any(item.coverage == VerificationCoverage.FULL.value for item in passed):
            return CompletionEvaluation(CompletionVerificationStatus.VERIFIED,
                                        generation, passed,
                                        "full current-generation evidence passed")

        if passed:
            return CompletionEvaluation(CompletionVerificationStatus.PARTIALLY_VERIFIED,
                                        generation, passed,
                                        "only partial current-generation evidence passed")

        return CompletionEvaluation(CompletionVerificationStatus.UNVERIFIED,
                                    generation, current,
                                    "no meaningful current-generation evidence")

    @staticmethod
    def _direct_invocation(category: VerificationCategory,
                           best: VerificationCoverage, normalized: str,
                           changed_paths: Sequence[str],
                           extensions: frozenset[str]
                           ) -> tuple[VerificationCategory, VerificationCoverage]:
        """How much a directly invoked tool proves about the current change.

        FULL is reserved for a command that actually exercises a changed file:
        it either names one, or it comes from the project's own bench or
        hints, which are settled before this is reached. A tool that handles
        the kind of file that changed but names none of them may have touched
        the change and may not -- partial. A tool that handles neither the
        files nor their language exercised something else entirely, and that
        is not weak evidence, it is none.
        """

        if not changed_paths:
            # Nothing has changed yet, so there is nothing to be unrelated to.

            return category, best

        if VerificationPolicy._names_a_changed_path(normalized, changed_paths):
            return category, best

        if VerificationPolicy._touches_language(changed_paths, extensions):
            return category, VerificationCoverage.PARTIAL

        return VerificationCategory.UNRELATED, VerificationCoverage.NONE

    @staticmethod
    def _names_a_changed_path(normalized: str, changed_paths: Sequence[str]) -> bool:
        """True when a command token IS one of the changed files.

        Token equality rather than substring containment: `gcc adder.c` must
        not count as naming `add.c`, and a command mentioning `src` must not
        count as naming everything under it.
        """

        tokens = set(normalized.split())

        for path in changed_paths:
            candidate = str(path).lower().lstrip("./")
            name = candidate.rsplit("/", 1)[-1]

            if any(token.lstrip("./") in {candidate, name} for token in tokens):
                return True

        return False

    @staticmethod
    def _touches_language(changed_paths: Sequence[str],
                          extensions: frozenset[str]) -> bool:
        for path in changed_paths:
            name = str(path).lower().rsplit("/", 1)[-1]
            suffix = name[name.rfind("."):] if "." in name else ""

            if name in extensions or (suffix and suffix in extensions):
                return True

        return False

    @staticmethod
    def _execution_languages(normalized: str) -> frozenset[str]:
        """The files the interpreter or program in this command is about."""

        for token in normalized.split():
            if token.startswith("./"):
                # Something built in the tree: whatever a compiler produces.

                return _COMPILED

            program = token.rsplit("/", 1)[-1]

            for pattern, extensions in _INTERPRETER_LANGUAGES:
                if pattern.match(program):
                    return extensions

        return _COMPILED

    @staticmethod
    def _normalize(command: str) -> str:
        """One spelling of a command, so patterns and hints compare alike."""

        # Unbalanced quotes make shlex refuse; whitespace splitting is a good
        # enough fallback for a classifier that only ever reads the command.

        try:
            return " ".join(shlex.split(command)).strip().lower()
        except ValueError:
            return " ".join(command.split()).strip().lower()

    @staticmethod
    def _matches_hint(command: str, hints: Sequence[str]) -> bool:
        return any(command == VerificationPolicy._normalize(hint)
                   or command.startswith(VerificationPolicy._normalize(hint) + " ")
                   for hint in hints if hint.strip())

    @staticmethod
    def _summary(category: VerificationCategory, coverage: VerificationCoverage,
                 outcome: VerificationOutcome) -> str:
        return f"{category.value} {outcome.value} ({coverage.value} coverage)"
