"""Import each named test file and call every test function it defines.

Copied into a throwaway copy of the workspace by ``benchmarks.acceptance``
and run there under Bubblewrap.  It prints one JSON report per file on stdout
and exits non-zero if anything failed, so the parent never has to parse prose.

The report is kept small on purpose: the sandbox truncates a long stdout, and
a truncated report is one the parent cannot read at all.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import traceback


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")

    module = importlib.util.module_from_spec(spec)

    # Registered before execution so a test module importing itself, or a
    # dataclass defined in it, resolves the same way it would under pytest.

    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


MAX_FAILURES = 5
MAX_ERROR_CHARS = 300


def _failure(name: str, exc: BaseException) -> dict:
    return {"test": name, "error": f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS],
            "traceback": traceback.format_exc()[-MAX_ERROR_CHARS:]}


def run_file(relative: str, index: int) -> dict:
    path = Path(relative)
    report: dict = {"path": relative, "status": "passed", "tests": 0, "failures": []}

    try:
        module = _load(path, f"spear_acceptance_{index}")
    except BaseException as exc:  # a broken module is a failed acceptance test
        report["status"] = "import_error"
        report["failures"].append(_failure("<import>", exc))

        return report

    names = sorted(name for name in dir(module)
                   if name.startswith("test") and callable(getattr(module, name)))

    for name in names:
        report["tests"] += 1

        try:
            getattr(module, name)()
        except BaseException as exc:
            report["status"] = "failed"

            if len(report["failures"]) < MAX_FAILURES:
                report["failures"].append(_failure(name, exc))

    # A file that collects nothing proves nothing, and a task whose acceptance
    # test silently stopped existing must not read as a pass.

    if not report["tests"]:
        report["status"] = "empty"
        report["failures"].append({"test": "<collection>", "error": "no test functions"})

    return report


def main(argv: list[str]) -> int:
    # The workspace root, not this script's directory: the tests import the
    # modules the run produced, which sit at the top of the tree.

    sys.path.insert(0, str(Path.cwd()))
    reports = [run_file(relative, index) for index, relative in enumerate(argv)]
    print(json.dumps(reports))

    return 0 if all(item["status"] == "passed" for item in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
