"""CLI runner for real local-model trials and deterministic manifest output.

The ``--real`` path launches the normal ``rag_chat.py`` CLI in an isolated
fixture.  It therefore exercises the configured ModelBackend and security
boundary; it never imports a fake provider.  ``--smoke`` only validates
fixture/manifest/report plumbing and is not a quality measurement.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import datetime as dt
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import uuid

from . import SUITE_VERSION
from .config import CONFIGURATIONS, AblationConfig, get_configuration
from .reporting import write_report
from .scoring import score_task, summarize_trace
from .tasks import create_fixture, create_memory_fixture, fixture_version, get_tasks


def model_config() -> dict[str, object]:
    # Deliberately exclude credential-like variables and full environment.

    return {"backend": "openai-compatible", "api_base": os.environ.get("SPEAR_API_BASE", "http://127.0.0.1:8080/v1"),
            "model": os.environ.get("SPEAR_MODEL_NAME", "qwen3"),
            "temperature": float(os.environ.get("SPEAR_TEMP", "0.25")),
            "top_p": 0.8, "top_k": 20, "max_output_tokens": int(os.environ.get("SPEAR_MAX_TOKENS", "8192")),
            "context_limit": int(os.environ.get("SPEAR_CTX", "32768")), "thinking_disabled": True}


def hardware_config() -> dict[str, object]:
    result = {"host": platform.node(), "cpu_count": os.cpu_count(),
              "platform": platform.platform(), "gpu": "unknown"}

    try:
        result["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=2,
        ).strip() or "none reported"
    except (OSError, subprocess.SubprocessError):
        result["gpu"] = "unavailable"

    return result


def run_one(task, configuration: AblationConfig, output: Path, repetition: int, *,
            real: bool, timeout: int, acceptance_timeout: int = 60) -> dict:
    run_id = f"{task.task_id}-r{repetition}-{uuid.uuid4().hex[:8]}"

    # The child process runs with cwd set to its isolated fixture.  Keep the
    # trace in the run directory, never inside (or counted as part of) that
    # fixture, by resolving this path before launching the child.

    trace_path = output.resolve() / f"{run_id}.trace.jsonl"
    memories_path = (output.resolve() / f"{run_id}.memories.md"
                     if task.memories else None)
    answer_path = output.resolve() / f"{run_id}.answer.txt"

    # Everything the child ACCUMULATES -- history, trajectories, training
    # data, sessions, checkpoints, the audit trail -- lands here and nowhere
    # else. Without it a run wrote into the operator's own state: benchmark
    # trajectories joined the real fine-tuning corpus, and each repetition
    # left a history file behind in the checkout.

    state_dir = output.resolve() / "state" / run_id

    # One Chroma database for the whole campaign, never the production one.
    # Each fixture is a fresh temporary directory, so the corpus tag -- and
    # with it the collection name -- is unique per run: pointed at the real
    # database, a campaign leaves one dead collection in it per task per
    # repetition, forever. Indexing the fixtures costs what it costs, and that
    # cost belongs inside `duration_seconds` where it can be seen.

    db_path = campaign_db(output)
    start = time.monotonic()

    with tempfile.TemporaryDirectory(prefix=f"spear-bench-{task.task_id}-") as directory:
        root = Path(directory)
        initial = create_fixture(task, root)

        process = None
        failure = None

        if real:
            state_dir.mkdir(parents=True, exist_ok=True)
            db_path.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update({
                "SPEAR_TRACE": "1", "SPEAR_TRACE_FILE": str(trace_path),
                "SPEAR_TRAINING_DATA_ORIGIN": "BENCHMARK",
                "SPEAR_STATE_DIR": str(state_dir),
                "SPEAR_DB_PATH": str(db_path),
                "SPEAR_BENCH_ANSWER_FILE": str(answer_path),
                # The fixtures are a `tests/` package and have no owner to
                # ask how they are tested, so the harness is allowed to infer
                # it here. Real trees say so themselves or are not verified
                # automatically at all.
                "SPEAR_INFER_TEST_COMMAND": "1",
            })

            # A memory task's memories live beside the trace, not in the
            # fixture: inside the workspace the model would read the file and
            # the task would measure grep rather than memory selection.

            if memories_path is not None:
                create_memory_fixture(task, memories_path)
                env["SPEAR_MEMORIES_FILE"] = str(memories_path)

            # Benchmark-only switches are explicit child-process inputs.  They
            # leave ordinary CLI invocations unchanged and are consumed when
            # rag_chat constructs its TaskRequest.

            for key, enabled in configuration.to_dict().items():
                if key != "name":
                    env[f"SPEAR_BENCH_{key.upper()}"] = "1" if enabled else "0"

            # Targeted long-context fixtures opt into a benchmark-only
            # history prelude. Production invocations have no default bloat;
            # the prelude exercises the existing compaction threshold.

            if "compaction" in task.tags:
                env["SPEAR_BENCH_CONTEXT_TURNS"] = "38"

            # --single-root, always. Without it the child declares every
            # registered corpus as an additional WRITABLE root and says so in
            # its prompt: a benchmark run was offering the model more than
            # twenty real repositories -- SO3, LVGL, Infrabase, spear
            # itself -- to edit. A run may write to its fixture and nowhere
            # else, whatever the operator's corpus registry happens to hold.

            command = [sys.executable,
                       str(Path(__file__).resolve().parents[1] / "rag_chat.py"),
                       "--auto", "--here", "--single-root"]

            try:
                process = subprocess.run(command, input=task.objective + "\nexit\n", cwd=root, env=env,
                                         text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)

                if process.returncode != 0:
                    failure = f"process_exit:{process.returncode}"
            except (OSError, subprocess.TimeoutExpired) as exc:
                failure = f"{type(exc).__name__}: {exc}"

        trace = summarize_trace(trace_path)
        score = score_task(task, root, initial, trace,
                           answer=_read_answer(answer_path),
                           acceptance_timeout=acceptance_timeout)

    row = {"run_id": run_id, "task_id": task.task_id, "task_class": task.task_class,
           "repetition": repetition, "commit": _commit(), "suite_version": SUITE_VERSION,
           "fixture_version": fixture_version(task), "configuration": configuration.to_dict(),
           "model": model_config(),
           # Two verdicts, kept apart: the file can be right and the task
           # still not done. `success` is the task, which is what a campaign
           # is asked about; `functionally_correct` is what the oracle saw.
           # Named twice on purpose: `task_success` is what it is, and
           # `success` is what every reader and every old manifest already
           # calls it. They are the same value and always will be.
           "task_success": score.task_success and failure is None,
           "success": score.task_success and failure is None,
           "functionally_correct": score.functionally_correct,
           "runtime_verified": score.runtime_verified,
           "scores": score.to_dict(), "usage": trace, "duration_seconds": round(time.monotonic() - start, 3),
           "trace_reference": str(trace_path), "failure_reason": failure or score.failure_reason,
           "oracles": list(task.oracles),
           "memories_reference": str(memories_path) if memories_path else None,
           "answer_reference": str(answer_path) if answer_path.exists() else None,
           "state_reference": str(state_dir) if state_dir.exists() else None,
           "db_reference": str(db_path) if db_path.exists() else None,
           "real_model": real, "hardware": hardware_config(),
           "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()}

    return row


def campaign_db(output: Path) -> Path:
    """The campaign's own Chroma directory, under its output directory."""

    return Path(output).resolve() / "chromadb"


def _read_answer(path: Path) -> str | None:
    """The run's final answer, as the child recorded it.

    None means the turn produced no answer at all, which is not the same as an
    empty one and must not read as a wrong one.
    """

    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except OSError: return "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", "--config", default="full", choices=tuple(CONFIGURATIONS))
    parser.add_argument("--task", action="append", dest="tasks", default=[])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--real", action="store_true", help="run the normal local ModelBackend CLI")
    parser.add_argument("--smoke", action="store_true", help="fixture/manifest smoke run; not a quality result")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--acceptance-timeout", type=int, default=60,
                        help="seconds allowed for a task's acceptance tests")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", action="store_true", help="aggregate an existing manifest in --output")
    args = parser.parse_args(argv)

    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    # Default output is a timestamped directory beside the audit trail, so
    # successive runs never overwrite one another.

    output = Path(args.output or (
        Path(__file__).resolve().parents[1] / "audit" / "benchmarks"
        / dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.jsonl"

    if args.report:
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]

        write_report(rows, output)

        return 0

    real = bool(args.real)

    if not real and not args.smoke:
        parser.error("choose --real or --smoke; smoke output is not a model measurement")

    configuration = get_configuration(args.configuration)
    tasks = get_tasks(tuple(args.tasks))
    completed = set()

    if args.resume and manifest.exists():
        completed = {json.loads(line).get("run_id")
                     for line in manifest.read_text(encoding="utf-8").splitlines()
                     if line.strip()}

    rows = []

    with manifest.open("a", encoding="utf-8") as handle:
        for task in tasks:
            for repetition in range(1, args.repetitions + 1):
                prefix = f"{task.task_id}-r{repetition}-"

                if any(str(item).startswith(prefix) for item in completed):
                    continue

                row = run_one(task, configuration, output, repetition, real=real,
                              timeout=args.timeout,
                              acceptance_timeout=args.acceptance_timeout)

                # Flushed per row, so an interrupted run keeps every result it
                # already produced and can be resumed.

                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                rows.append(row)

    if manifest.exists():
        all_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        write_report(all_rows, output)

    # The campaign's own description, written beside the manifest rather than
    # only printed: which stores it used is part of reading its numbers later.

    campaign = {"output": str(output.resolve()), "configuration": configuration.name,
                "tasks": [task.task_id for task in tasks],
                "repetitions": args.repetitions, "real_model": real,
                "suite_version": SUITE_VERSION, "commit": _commit(),
                "chroma_db": str(campaign_db(output)),
                "state_root": str(output.resolve() / "state"),
                "model": model_config(),
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()}
    (output / "campaign.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(campaign, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
