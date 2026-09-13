"""Small deterministic benchmark fixtures.

Fixtures are created in a caller-owned temporary directory.  Nothing in this
module writes to the source checkout or assumes a particular model/provider.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Mapping, Sequence

from memory_store import MarkdownMemoryStore, MemoryScope, MemorySource


def fragments(expectation: str | Sequence[str]) -> tuple[str, ...]:
    """The alternative spellings one content expectation accepts.

    A single string is one fragment; a sequence is a set of alternatives, any
    one of which satisfies an ``expected`` entry and all of which must be
    absent for a ``forbidden_content`` entry.  It exists because an answer
    written into a file may honestly be "JSON" or "json", while a code
    fragment may not be spelled freely at all.
    """

    if isinstance(expectation, str):
        return (expectation,)

    return tuple(str(item) for item in expectation)


@dataclass(frozen=True)
class MemoryFixture:
    """One memory supplied to the run, outside the workspace it can edit."""

    content: str
    scope: MemoryScope = MemoryScope.PROJECT
    supersedes_previous: bool = False


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    task_class: str
    objective: str
    files: Mapping[str, str]
    expected: Mapping[str, str | tuple[str, ...]] = ()
    forbidden_paths: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    required_reads: tuple[str, ...] = ()
    # The oracle tests: name -> source, held HERE and never written into the
    # fixture. They are run in a copy of the finished workspace, under the
    # same confinement as every other command the harness runs. The fixture's
    # own tests stay in `files`, where the task may tell the model to change
    # them -- and a test the model can change decides nothing.
    acceptance: Mapping[str, str] = ()
    forbidden_content: Mapping[str, str | tuple[str, ...]] = ()
    memories: tuple[MemoryFixture, ...] = ()
    # Substrings the final ANSWER must (or must not) contain, matched without
    # regard to case. Each entry may give alternatives. For a task that only
    # reports, reading the right files is not the same as answering
    # correctly, and nothing in the workspace tells them apart.
    expected_answer: tuple[str | tuple[str, ...], ...] = ()
    forbidden_answer: tuple[str | tuple[str, ...], ...] = ()
    # Set where the objective says to run or test what it changed. The run
    # must then show one verification of its own that covers the change --
    # full coverage, conclusive, and not the harness's bench running on its
    # behalf. The hidden oracle proves the file is right; this is what
    # separates that from the run having done the task.
    requires_agent_verification: bool = False

    @property
    def oracles(self) -> tuple[str, ...]:
        """Which kinds of evidence can decide this task, in fixture terms."""

        return tuple(name for name, declared in (
            ("expected", bool(dict(self.expected))),
            ("forbidden_content", bool(dict(self.forbidden_content))),
            ("required_reads", bool(self.required_reads)),
            ("acceptance", bool(dict(self.acceptance))),
            ("expected_answer", bool(self.expected_answer)),
            ("forbidden_answer", bool(self.forbidden_answer)),
            ("agent_verification", self.requires_agent_verification),
        ) if declared)


def unittest_case(imports: str, name: str, body: str) -> str:
    """One fixture test file, in the form the harness can actually run.

    The fixtures used to be bare pytest functions, and pytest is not a
    dependency of this checkout: told to run the test, a model reached for
    `python3 -c`, which is refused, and finished the turn having verified
    nothing. unittest is in the standard library and `python -m unittest` is
    allowlisted, so the obvious command now works. `tests/__init__.py` is part
    of every fixture for the same reason: without it `python -m unittest
    discover` collects nothing and says "NO TESTS RAN".
    """

    return (f"import unittest\n\n{imports}\n\n\nclass {name}(unittest.TestCase):\n"
            f"{body}\n\nif __name__ == '__main__':\n    unittest.main()\n")


# The oracle tests.  They are never written into a fixture: the model cannot
# read them, cannot run them, and cannot make them pass by editing them.  Each
# one states the REQUIREMENT, which is why several of them check more than the
# fixture's own test does.

ADD_ORACLE = """from main import add


def test_add_returns_the_sum():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
"""

GREET_ORACLE = """from greet import greet


def test_greeting():
    assert greet('A') == 'Hello A'
    assert greet('Bo') == 'Hello Bo'
"""


def _tasks() -> tuple[BenchmarkTask, ...]:
    base = {
        "main.py": "def add(a, b):\n    return a - b\n",
        "tests/__init__.py": "",
        "tests/test_main.py": unittest_case(
            "from main import add", "AddTests",
            "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n"),
    }
    return (
        BenchmarkTask(
            "control-symbol", "A-control",
            "Locate the add function in main.py and report its file and symbol.",
            base, tags=("read",), required_reads=("main.py",),
            expected_answer=("main.py", "add")),

        BenchmarkTask(
            "control-edit", "A-control",
            "Fix add in main.py so it returns the sum. Run the test.",
            base, {"main.py": "return a + b"}, ("unrelated.py",), ("mutation",),
            acceptance={"test_add.py": ADD_ORACLE},
            requires_agent_verification=True),

        BenchmarkTask(
            "control-test", "A-control",
            "Make the existing add test pass with the smallest change.",
            base, {"main.py": "return a + b"}, tags=("mutation",),
            acceptance={"test_add.py": ADD_ORACLE},
            requires_agent_verification=True),

        BenchmarkTask(
            "control-doc", "A-control",
            "Answer which file defines add without editing files.",
            base, forbidden_paths=("main.py", "tests/test_main.py"),
            tags=("read",), expected_answer=("main.py",)),

        BenchmarkTask(
            "call-chain", "B-exploration",
            "Trace the API call chain from api.py to storage.py and name the persistence "
            "function: give its file and its name.",
            {
                "api.py": "from service import handle\ndef request(x):\n    return handle(x)\n",
                "service.py": "from storage import save\ndef handle(x):\n    return save(x)\n",
                "storage.py": "def save(value):\n    return value\n",
                "tests/__init__.py": "",
                "tests/test_api.py": unittest_case(
                    "from api import request", "RequestTests",
                    "    def test_request(self):\n"
                    "        self.assertEqual(request('x'), 'x')\n"),
                "notes.txt": "unrelated notes\n",
            }, tags=("exploration",),
            required_reads=("api.py", "service.py", "storage.py"),
            expected_answer=("storage.py", "save"),
            forbidden_answer=(("notes.txt",),)),

        # service.py used to call normalize without importing it, so the
        # fixture raised NameError before anything was asked of it. And the
        # oracle named ONE correct fix -- `return str(x)` -- while `return x`
        # preserves the input better and reads the objective more faithfully.
        # Behaviour decides now; the reads prove the chain was traced.
        BenchmarkTask(
            "unknown-target", "B-exploration",
            "Find the implementation behind request and fix it to preserve the input.",
            {
                "api.py": "from service import handle\ndef request(x): return handle(x)\n",
                "service.py": "from utils import normalize\n\n\ndef handle(x): return normalize(x)\n",
                "utils.py": "def normalize(x): return str(x).upper()\n",
                "tests/__init__.py": "",
                "tests/test_api.py": unittest_case(
                    "from api import request", "RequestTests",
                    "    def test_request(self):\n"
                    "        self.assertEqual(request('x'), 'x')\n"),
            }, tags=("exploration", "mutation"),
            required_reads=("api.py", "service.py", "utils.py"),
            acceptance={"test_request.py": """from api import request


def test_request_preserves_its_input():
    assert request('x') == 'x'
    assert request('AbC') == 'AbC'
    assert request('  spaced  ') == '  spaced  '
"""}),

        BenchmarkTask(
            "relevant-tests", "B-exploration",
            "Name the test files that cover the service implementation. Name only "
            "those: do not mention any test file that does not cover it.",
            {
                "service.py": "def status(): return 'ok'\n",
                "tests/__init__.py": "",
                "tests/test_service.py": unittest_case(
                    "from service import status", "StatusTests",
                    "    def test_status(self):\n"
                    "        self.assertEqual(status(), 'ok')\n"),
                "tests/test_other.py": unittest_case(
                    "", "OtherTests",
                    "    def test_other(self):\n        self.assertTrue(True)\n"),
            }, tags=("exploration",),
            required_reads=("service.py", "tests/test_service.py"),
            expected_answer=("test_service",), forbidden_answer=("test_other",)),

        BenchmarkTask(
            "multi-api", "C-multifile",
            "Add a multiply API, implementation, and a focused test in api.py, service.py, and tests/test_service.py.",
            {
                "api.py": "from service import add\n",
                "service.py": "def add(a,b): return a+b\n",
                "tests/__init__.py": "",
                "tests/test_service.py": unittest_case(
                    "from service import add", "ServiceTests",
                    "    def test_add(self):\n        self.assertEqual(add(1, 2), 3)\n"),
            },
            {"service.py": "def multiply", "api.py": "multiply",
             "tests/test_service.py": "test_multiply"},
            tags=("planning", "mutation"),
            acceptance={"test_multiply.py": """import api
import service


def test_service_multiplies():
    assert service.multiply(3, 4) == 12


def test_api_exposes_multiply():
    assert api.multiply(3, 4) == 12


def test_add_still_works():
    assert service.add(1, 2) == 3
"""}),

        BenchmarkTask(
            "config-migration", "C-multifile",
            "Rename the configuration key host to endpoint in config.py and its test, without changing unrelated.py.",
            {
                "config.py": "SETTINGS = {'host': 'localhost'}\n",
                "tests/__init__.py": "",
                "tests/test_config.py": unittest_case(
                    "from config import SETTINGS", "ConfigTests",
                    "    def test_key(self):\n        self.assertIn('host', SETTINGS)\n"),
                "unrelated.py": "KEEP = True\n",
            },
            {"config.py": "endpoint", "tests/test_config.py": "endpoint",
             "unrelated.py": "KEEP = True"},
            ("unrelated.py",), tags=("planning", "mutation"),
            forbidden_content={"config.py": "'host'", "tests/test_config.py": "'host'"},
            acceptance={"test_config_key.py": """from config import SETTINGS


def test_key_is_renamed_not_duplicated():
    assert 'endpoint' in SETTINGS
    assert 'host' not in SETTINGS
    assert SETTINGS['endpoint'] == 'localhost'
"""}),

        BenchmarkTask(
            "refactor-three", "C-multifile",
            "Move the value formatting out of client.py into a new formatter.py, "
            "as a function named format_value. Keep client.py using it, and keep "
            "its test passing.",
            {
                "client.py": "def render(x): return '[' + str(x) + ']'\n",
                "tests/__init__.py": "",
                "tests/test_client.py": unittest_case(
                    "from client import render", "RenderTests",
                    "    def test_render(self):\n"
                    "        self.assertEqual(render('x'), '[x]')\n"),
                "README.txt": "client\n",
            },
            {"client.py": "formatter", "formatter.py": "def format_value",
             "tests/test_client.py": "test_render"},
            tags=("planning", "mutation"), requires_agent_verification=True,
            acceptance={"test_refactor.py": """import formatter
from client import render


def test_client_still_renders():
    assert render('x') == '[x]'


def test_formatting_moved_to_formatter():
    assert formatter.format_value('x') == '[x]'
"""}),

        BenchmarkTask(
            "requirement-regression", "D-review",
            "Implement safe_divide: it must raise ValueError on zero and return the quotient otherwise; add a regression test.",
            {
                "maths.py": "def safe_divide(a,b):\n    return a / b\n",
                "tests/__init__.py": "",
                "tests/test_maths.py": unittest_case(
                    "from maths import safe_divide", "SafeDivideTests",
                    "    def test_normal(self):\n"
                    "        self.assertEqual(safe_divide(4, 2), 2)\n"),
                "RULES.txt": "Zero denominator must raise ValueError.\n",
            },
            {"maths.py": "ValueError", "tests/test_maths.py": "zero"},
            tags=("review", "mutation"),
            acceptance={"test_safe_divide.py": """from maths import safe_divide


def test_quotient():
    assert safe_divide(4, 2) == 2


def test_zero_denominator_raises_value_error():
    try:
        safe_divide(1, 0)
    except ValueError:
        return

    raise AssertionError('safe_divide(1, 0) must raise ValueError')
"""}),

        BenchmarkTask(
            "unrelated-change", "D-review",
            "Fix the greeting function and do not edit the unrelated data file.",
            {
                "greet.py": "def greet(name): return 'Hi ' + name\n",
                "data.txt": "DO NOT CHANGE\n",
                "tests/__init__.py": "",
                "tests/test_greet.py": unittest_case(
                    "from greet import greet", "GreetTests",
                    "    def test_greet(self):\n"
                    "        self.assertEqual(greet('A'), 'Hello A')\n"),
            },
            {"greet.py": "Hello"}, ("data.txt",), tags=("review", "mutation"),
            acceptance={"test_greeting.py": GREET_ORACLE}),

        # The long files below are the point of these two: they force the
        # context to be compacted before the small fix can be made.

        BenchmarkTask(
            "context-history", "E-context",
            "Inspect the repository and fix the marked function while ignoring irrelevant notes.",
            {**base, "old_notes.txt": "irrelevant\n" * 300},
            {"main.py": "return a + b"}, tags=("compaction", "mutation"),
            acceptance={"test_add.py": ADD_ORACLE}),

        BenchmarkTask(
            "context-evidence", "E-context",
            "Find the failing assertion in the long evidence file and correct main.py.",
            {**base, "evidence.log": ("noise\n" * 200) + "EXPECTED add returns sum\n"},
            {"main.py": "return a + b"}, tags=("compaction", "mutation"),
            acceptance={"test_add.py": ADD_ORACLE}),

        # These two used to carry no expectation at all -- the comment here
        # said the runner supplied the memories, and it did not: nothing was
        # injected, nothing was checked, and every run of them was scored a
        # success. The memories are now real (see `memories` below, written
        # outside the workspace so the model cannot simply read the file), and
        # what the run must do with them lands in a file, because a benchmark
        # that grades prose grades the model's fluency.

        BenchmarkTask(
            "memory-preference", "F-memory",
            "Write the project's configured output format, and nothing else, into result.txt. "
            "Use the active project memory, not the one it superseded.",
            {"README.txt": "Output format is configured by project memory.\n"},
            {"result.txt": ("json", "JSON", "Json")},
            tags=("memory", "mutation"),
            forbidden_content={"result.txt": ("xml", "XML", "Xml")},
            memories=(
                MemoryFixture("Output format for this project is XML."),
                MemoryFixture("Output format for this project is JSON.",
                              supersedes_previous=True),
                MemoryFixture("Prefer tabs over spaces.", MemoryScope.USER),
            )),

        BenchmarkTask(
            "memory-scope", "F-memory",
            "Write the retention period this project's own procedure requires, and nothing "
            "else, into retention.txt. A user-scoped habit is not the project's procedure.",
            {"procedure.txt": "Retain audit evidence for seven days.\n"},
            {"retention.txt": ("seven", "Seven", "7")},
            tags=("memory", "mutation"), required_reads=("procedure.txt",),
            forbidden_content={"retention.txt": ("thirty", "Thirty", "30")},
            memories=(
                MemoryFixture("I keep my own audit evidence for thirty days.",
                              MemoryScope.USER),
                MemoryFixture("This project follows the retention procedure in "
                              "procedure.txt.", MemoryScope.PROJECT),
            )),

        BenchmarkTask(
            "recover-test", "G-recovery",
            "Fix the failing test in main.py, retry after the first failed command, and leave unrelated.py unchanged.",
            base | {"unrelated.py": "UNCHANGED\n"},
            {"main.py": "return a + b"}, ("unrelated.py",),
            tags=("recovery", "mutation"), requires_agent_verification=True,
            acceptance={"test_add.py": ADD_ORACLE}),

        BenchmarkTask(
            "recover-wrong-file", "G-recovery",
            "Locate the wrong implementation attempt and repair the function in the correct file.",
            {
                "entry.py": "from impl import value\n",
                "impl.py": "def value(): return 0\n",
                "tests/__init__.py": "",
                "tests/test_impl.py": unittest_case(
                    "from impl import value", "ValueTests",
                    "    def test_value(self):\n        self.assertEqual(value(), 1)\n"),
            },
            {"impl.py": "return 1"}, tags=("recovery", "exploration", "mutation"),
            acceptance={"test_value.py": """from entry import value as entry_value
from impl import value


def test_value_is_repaired_in_the_right_file():
    assert value() == 1
    assert entry_value() == 1
"""}),

        # Targeted causal Explorer fixtures: the objective deliberately uses
        # repository-scale language and does not name the implementation file.
        BenchmarkTask("explorer-causal-chain", "B-exploration", "Trace the repository call chain across multiple files for the public lookup behavior, identify the persistence boundary, and preserve the input.", {
            "public_api.py": "from routing import dispatch\ndef lookup(value): return dispatch(value)\n",
            "routing.py": "from persistence import store\ndef dispatch(value): return store(value)\n",
            "persistence.py": "def store(value): return value.upper()\n",
            "cache.py": "def store(value): return 'cached'\n",
            "tests/__init__.py": "",
            "tests/test_lookup.py": unittest_case(
                "from public_api import lookup", "LookupTests",
                "    def test_lookup(self):\n        self.assertEqual(lookup('x'), 'x')\n"),
            "notes.txt": "unrelated architecture notes\n",
        }, {"persistence.py": "return value"}, tags=("exploration", "mutation"),
           required_reads=("public_api.py", "routing.py", "persistence.py"),
           forbidden_content={"persistence.py": ".upper()"},
           acceptance={"test_public_lookup.py": """import cache
from public_api import lookup


def test_lookup_preserves_its_input():
    assert lookup('x') == 'x'
    assert lookup('AbC') == 'AbC'


def test_the_decoy_module_was_left_alone():
    assert cache.store('x') == 'cached'
"""}),
        BenchmarkTask("explorer-causal-tests", "B-exploration", "Investigate the repository architecture across multiple files, locate the tests covering the public status behavior, and report the relevant implementation path. Name only the tests that cover that behavior: do not mention the unrelated ones.", {
            "entrypoint.py": "from domain import status\ndef public_status(): return status()\n",
            "domain.py": "from backend import current_status\ndef status(): return current_status()\n",
            "backend.py": "def current_status(): return 'ready'\n",
            "tests/__init__.py": "",
            "tests/test_status.py": unittest_case(
                "from entrypoint import public_status", "StatusTests",
                "    def test_status(self):\n"
                "        self.assertEqual(public_status(), 'ready')\n"),
            "tests/test_unrelated.py": unittest_case(
                "", "UnrelatedTests",
                "    def test_other(self):\n        self.assertTrue(True)\n"),
            "archive.txt": "irrelevant archived output\n" * 20,
        }, tags=("exploration",),
           required_reads=("tests/test_status.py", "entrypoint.py", "domain.py"),
           expected_answer=("test_status", ("backend.py", "current_status")),
           forbidden_answer=("test_unrelated",)),
        # Targeted Reviewer fixtures: ordinary tests cover the happy path, but
        # the explicit requirement supplies a deterministic boundary contract.
        BenchmarkTask("review-seeded-boundary", "D-review", "Implement parse_port for positive ports; the explicit requirement is that zero and negative values must raise ValueError, and add the normal positive-path test.", {
            "ports.py": "def parse_port(value):\n    return int(value)\n",
            "tests/__init__.py": "",
            "tests/test_ports.py": unittest_case(
                "from ports import parse_port", "PortTests",
                "    def test_positive(self):\n"
                "        self.assertEqual(parse_port('8080'), 8080)\n"),
            "RULES.txt": "Port values must be strictly positive; zero and negatives raise ValueError.\n",
        }, {"ports.py": "ValueError", "tests/test_ports.py": "positive"},
           tags=("review", "mutation"),
           acceptance={"test_parse_port.py": """from ports import parse_port


def test_positive_ports_parse():
    assert parse_port('8080') == 8080


def test_zero_and_negative_raise_value_error():
    for value in ('0', '-1'):
        try:
            parse_port(value)
        except ValueError:
            continue

        raise AssertionError('parse_port(%r) must raise ValueError' % value)
"""}),
        BenchmarkTask("review-seeded-compat", "D-review", "Implement normalize_name while preserving the public API: None must remain None, and ordinary tests cover only strings.", {
            "names.py": "def normalize_name(value):\n    return value.strip().lower()\n",
            "tests/__init__.py": "",
            "tests/test_names.py": unittest_case(
                "from names import normalize_name", "NameTests",
                "    def test_text(self):\n"
                "        self.assertEqual(normalize_name(' Alice '), 'alice')\n"),
            "RULES.txt": "Existing callers may pass None; normalize_name(None) must return None for compatibility.\n",
        }, {"names.py": "None", "tests/test_names.py": "text"},
           tags=("review", "mutation"),
           acceptance={"test_normalize_name.py": """from names import normalize_name


def test_strings_are_normalised():
    assert normalize_name(' Alice ') == 'alice'


def test_none_is_preserved_for_existing_callers():
    assert normalize_name(None) is None
"""}),
        BenchmarkTask("review-clean-style", "D-review", "Make the harmless requested greeting change; do not block completion for subjective formatting choices.", {
            "greet.py": "def greet(name): return 'Hi ' + name\n",
            "tests/__init__.py": "",
            "tests/test_greet.py": unittest_case(
                "from greet import greet", "GreetTests",
                "    def test_greet(self):\n"
                "        self.assertEqual(greet('A'), 'Hello A')\n"),
        }, {"greet.py": "Hello"}, tags=("review", "mutation"),
           acceptance={"test_greeting.py": GREET_ORACLE}),
    )


TASKS = _tasks()


def get_tasks(task_ids: tuple[str, ...] = ()) -> tuple[BenchmarkTask, ...]:
    if not task_ids:
        return TASKS

    wanted = set(task_ids)

    return tuple(task for task in TASKS if task.task_id in wanted)


def fixture_version(task: BenchmarkTask) -> str:
    """Identity of everything that decides this task's outcome.

    The oracles are part of it, not just the files: two runs scored against
    different expectations are not comparable, and a manifest that says they
    are is worse than one that says nothing.
    """

    payload = json.dumps({
        "id": task.task_id, "files": dict(task.files),
        "expected": {key: list(fragments(value))
                     for key, value in dict(task.expected).items()},
        "forbidden_content": {key: list(fragments(value))
                              for key, value in dict(task.forbidden_content).items()},
        "forbidden_paths": list(task.forbidden_paths),
        "required_reads": list(task.required_reads),
        "acceptance": dict(task.acceptance),
        "requires_agent_verification": task.requires_agent_verification,
        "expected_answer": [list(fragments(item)) for item in task.expected_answer],
        "forbidden_answer": [list(fragments(item)) for item in task.forbidden_answer],
        "memories": [[item.content, item.scope.value, item.supersedes_previous]
                     for item in task.memories],
    }, sort_keys=True)

    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def create_memory_fixture(task: BenchmarkTask, path: str | Path) -> int:
    """Write the task's memories to a store OUTSIDE the workspace.

    Inside it, the model would simply read the file and the task would measure
    nothing but grep.  The runner points the child at this path instead, so
    the memories arrive the way real ones do -- selected, scoped, and with the
    superseded one already filtered out.
    """

    path = Path(path)
    store = MarkdownMemoryStore(path, default_scope=MemoryScope.PROJECT)
    previous = None

    for memory in task.memories:
        previous = store.add(
            memory.content, scope=memory.scope, source=MemorySource.USER,
            provenance="benchmark fixture",
            supersedes=(previous.memory_id,) if memory.supersedes_previous and previous else (),
        )

    return len(task.memories)


def create_fixture(task: BenchmarkTask, root: str | Path) -> dict[str, str]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    for relative, content in task.files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    return {relative: hashlib.sha256(content.encode()).hexdigest()
            for relative, content in task.files.items()}


def changed_files(root: str | Path, initial: Mapping[str, str]) -> tuple[str, ...]:
    root = Path(root)
    found = {}

    for path in root.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            found[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()

    return tuple(sorted(set(initial) | set(found) - {key for key in initial if key in found and initial[key] == found[key]}))
