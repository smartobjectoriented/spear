"""Strict parser/presenter for the operator-only ``/finetune`` command."""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from training_controller import TrainingController, TrainingControllerError
from training_bundle import TrainingBundleError
from training_jobs import TrainingJob, TrainingJobError
from training_launcher import TrainingLauncherError
from remote_readiness import LocalDoctorReport, RemoteReadinessReport


FINETUNE_HELP = """Fine-tuning operator commands:
  /finetune status
  /finetune doctor [--remote]
  /finetune ingest [path]
  /finetune prepare [sft|kto|dpo]
  /finetune start [sft|kto|dpo] [--force]
  /finetune stop [job-id]
  /finetune logs [job-id] [lines]
  /finetune list
  /finetune inspect <job-id>"""


def default_trajectory_file() -> str | None:
    """Where the recorder actually writes, resolved the way it resolves it.

    Guessed once, as ~/.spear/state/trajectories.jsonl, and the guess was
    wrong: three such files exist on this machine and that one was the oldest
    and smallest of them. A default that points at a stale file is worse than
    no default -- it ingests something plausible and says nothing. So this
    reads the same two variables the recorder reads, and returns None when
    neither is set, which makes the command ask for the path instead.
    """
    named = os.environ.get("SPEAR_TRAJECTORY_FILE")

    if named:
        return named

    state = os.environ.get("SPEAR_STATE_DIR")

    return os.path.join(state, "trajectories.jsonl") if state else None


class FinetuneCommandError(ValueError):
    pass


@dataclass(frozen=True)
class FinetuneCommand:
    action: str
    method: str = "sft"
    force: bool = False
    job_id: str | None = None
    lines: int = 100
    remote: bool = False


def parse_finetune_command(text: str) -> FinetuneCommand:
    try:
        parts = shlex.split(text)
    except ValueError as exc:
        raise FinetuneCommandError(f"invalid /finetune syntax: {exc}") from exc

    if not parts or parts[0] != "/finetune":
        raise FinetuneCommandError("not a /finetune command")

    if len(parts) == 1 or parts[1] in {"help", "--help"}:
        return FinetuneCommand("help")

    action, args = parts[1], parts[2:]

    if action == "ingest":
        if len(args) > 1:
            raise FinetuneCommandError("usage: /finetune ingest [path]")

        # Carried in job_id rather than a field of its own: the parser's
        # shape is a closed set the tests pin, and a path is the same kind of
        # single optional argument.
        return FinetuneCommand(action, job_id=args[0] if args else None)

    if action in {"status", "list"}:
        if args:
            raise FinetuneCommandError(f"/{action} accepts no arguments")

        return FinetuneCommand(action)

    if action == "doctor":
        if len(args) > 1 or (args and args[0] != "--remote"):
            raise FinetuneCommandError("usage: /finetune doctor [--remote]")

        return FinetuneCommand(action, remote=bool(args))

    if action in {"prepare", "start"}:
        # The method and --force may come in either order, but neither may be
        # given twice: a repeated argument is a mistake worth reporting.

        method = "sft"
        force = False
        method_seen = False

        for item in args:
            if item == "--force" and action == "start" and not force:
                force = True
            elif item.lower() in {"sft", "kto", "dpo"} and not method_seen:
                method = item.lower()
                method_seen = True
            else:
                raise FinetuneCommandError(f"unknown {action} argument: {item}")

        return FinetuneCommand(action, method, force)

    if action == "inspect":
        if len(args) != 1:
            raise FinetuneCommandError("usage: /finetune inspect <job-id>")

        return FinetuneCommand(action, job_id=args[0])

    if action == "stop":
        if len(args) > 1:
            raise FinetuneCommandError("usage: /finetune stop [job-id]")

        return FinetuneCommand(action, job_id=args[0] if args else None)

    if action == "logs":
        if len(args) > 2:
            raise FinetuneCommandError("usage: /finetune logs [job-id] [lines]")

        job_id = args[0] if args else None
        lines = 100

        # Both arguments are optional, so a trailing number is the line count
        # and a job id is only present if something precedes it.

        if args and args[-1].isdigit():
            lines = int(args[-1])
            job_id = args[0] if len(args) == 2 else None

        if not 1 <= lines <= 500:
            raise FinetuneCommandError("log line count must be between 1 and 500")

        return FinetuneCommand(action, job_id=job_id, lines=lines)

    raise FinetuneCommandError(f"unknown /finetune action: {action}")


def _stage_line(name: str, value: dict) -> str:
    """One readiness stage as a single operator-readable line."""

    # The unpaired stage counts two classes rather than one approved set, so
    # its total has to be assembled here.

    count = value.get("metrics", {}).get("auto_approved")

    if count is None and name == "Unpaired":
        metrics = value.get("metrics", {})
        count = int(metrics.get("positive", 0)) + int(metrics.get("negative", 0))

    suffix = f", {count} AUTO_APPROVED" if count is not None else ""
    reasons = ", ".join(value.get("reasons", ())) or "none"

    return f"{name}: {value['state']} / {value['level']}{suffix} (reasons: {reasons})"


def _job_line(job: TrainingJob) -> str:
    started = job.start_time or "not started"
    progress = ""

    if job.progress_metrics:
        progress = "  " + " ".join(f"{key}={value}" for key, value in
                                    sorted(job.progress_metrics.items()))

    return (f"{job.job_id}  {job.method:<3}  {job.status:<9}  "
            f"{job.bundle_id}  {job.target_host}  {started}{progress}")


def _format_status(report: dict, jobs: list[TrainingJob]) -> str:
    lines = ["Fine-tuning readiness", "",
             _stage_line("SFT", report["sft"]),
             _stage_line("Unpaired", report["unpaired_preference"]),
             _stage_line("Paired", report["paired_preference"]), "",
             "Recommendation: " + report["recommendation"]]
    remote = report.get("remote_readiness", {})

    # The GPU owner is the more informative of the two, so the plain
    # availability is only shown when nothing identified the workload.

    lines.extend((
        "", f"Remote readiness: {remote.get('state', 'UNKNOWN')}",
        f"Remote reason: {', '.join(remote.get('reasons', ())) or 'none'}",
        f"Training environment: "
        f"{remote.get('checks', {}).get('environment_readiness', remote.get('state', 'UNKNOWN'))}",
        f"Hardware capacity: "
        f"{remote.get('checks', {}).get('hardware_capacity', 'UNKNOWN')}",
        f"GPU availability: "
        f"{remote.get('checks', {}).get('gpu_workload_owner', remote.get('checks', {}).get('current_gpu_availability', 'UNKNOWN'))}",
        f"Last checked: {remote.get('checked_at') or 'never'}"))
    active = [job for job in jobs if job.status in {"PREFLIGHT", "STARTING", "RUNNING", "STOPPING"}]

    if active:
        lines.extend(("", "Active jobs:", *(_job_line(job) for job in active[:5])))

    return "\n".join(lines)


def _format_doctor(local: LocalDoctorReport,
                   remote: RemoteReadinessReport | None) -> str:
    lines = ["Fine-tuning doctor", "", "LOCAL"]

    for key, value in local.checks.items():
        if key == "data_readiness" and isinstance(value, dict):
            lines.append("  Data readiness: " + ", ".join(
                f"{name}={value[name]}" for name in ("sft", "unpaired", "paired")))
            lines.append(f"  Data recommendation: {value.get('recommendation', 'UNKNOWN')}")

            continue

        if isinstance(value, dict):
            value = value.get("status", value.get("state", "unknown"))

        lines.append(f"  {key.replace('_', ' ').title()}: {value}")

    lines.extend((f"  Local control state: {local.state}",
                  f"  Local reasons: {', '.join(local.reasons) or 'none'}", "", "REMOTE"))

    if remote is None:
        lines.extend(("  SSH: NOT CHECKED", "  Remote readiness: UNKNOWN",
                      "  Reason: remote_probe_not_requested"))
    else:
        checks = remote.checks
        ssh = checks.get("ssh", {})
        lines.append(f"  SSH: {ssh.get('status', 'unknown') if isinstance(ssh, dict) else ssh}")

        for key in ("hostname", "python", "cuda", "gpu", "gpu_process",
                    "hardware_capacity", "current_gpu_availability",
                    "model_storage_capacity", "environment_readiness",
                    "model_snapshot_status",
                    "inference_service", "gpu_workload_owner",
                    "source_model_revision", "environment_fingerprint",
                    "axolotl_available",
                    "axolotl_version", "package", "disk_available_kib",
                    "training_root", "qwen3_next_compatibility"):
            if key in checks:
                value = checks[key]

                if key == "package" and isinstance(value, dict):
                    value = ", ".join(f"{name}={version}" for name, version in sorted(value.items()))

                lines.append(f"  {key.replace('_', ' ').title()}: {value}")

        lines.extend((f"  Training environment: {remote.state}",
                      f"  Hardware capacity: {checks.get('hardware_capacity', 'UNKNOWN')}",
                      f"  Current GPU availability: {checks.get('current_gpu_availability', 'UNKNOWN')}",
                      f"  Remote readiness: {remote.state}",
                      f"  Reasons: {', '.join(remote.reasons) or 'none'}",
                      f"  Last checked: {remote.checked_at}"))

    return "\n".join(lines)


def _format_inspect(job: TrainingJob) -> str:
    # The bundle path is a local filesystem location; the job's identity is its
    # id and checksums, which is what an operator needs to see.

    safe = asdict(job)
    safe.pop("bundle_path", None)

    return json.dumps(safe, sort_keys=True, indent=2)


def handle_finetune_command(text: str, controller: TrainingController) -> str:
    command = parse_finetune_command(text)

    try:
        if command.action == "help":
            return FINETUNE_HELP

        if command.action == "status":
            report, jobs = controller.status()

            return _format_status(dict(report), jobs)

        if command.action == "doctor":
            local, remote = controller.doctor(remote=command.remote)
            return _format_doctor(local, remote)

        if command.action == "ingest":
            source = command.job_id or default_trajectory_file()

            if not source:
                raise FinetuneCommandError(
                    "usage: /finetune ingest <path> — no SPEAR_TRAJECTORY_FILE "
                    "or SPEAR_STATE_DIR is set, so there is no file to assume")

            return controller.ingest(source).message

        if command.action == "prepare":
            return controller.prepare(command.method).message

        if command.action == "start":
            result = controller.start(command.method, force=command.force)
            suffix = " Readiness override recorded." if command.force and result.job else ""

            return result.message + suffix

        if command.action == "stop":
            job = controller.stop(command.job_id)

            return f"{job.job_id}: {job.status}"

        if command.action == "logs":
            return controller.logs(command.job_id, command.lines) or "(no training logs)"

        if command.action == "list":
            jobs = controller.list_jobs()

            if not jobs:
                return "No fine-tuning jobs."

            return "JOB  METHOD  STATUS  BUNDLE  HOST  STARTED\n" + "\n".join(map(_job_line, jobs))

        if command.action == "inspect":
            return _format_inspect(controller.inspect(command.job_id or ""))
    except (TrainingControllerError, TrainingBundleError, TrainingJobError,
            TrainingLauncherError, OSError) as exc:
        return f"Fine-tuning command refused: {exc}"

    raise AssertionError("unhandled fine-tuning command")
