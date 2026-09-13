"""Controlled training launchers.  No launcher is model-visible."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Mapping, Protocol, Sequence

from training_jobs import TrainingJob


class TrainingLauncherError(RuntimeError):
    pass


_SAFE_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,254}\Z")
_SAFE_EXECUTABLE = re.compile(r"(?:/[A-Za-z0-9_.+-]+)+|[A-Za-z0-9][A-Za-z0-9_.+-]*\Z")
_SAFE_REMOTE_ROOT = re.compile(r"(?:/[A-Za-z0-9_.+-]+)+\Z")
_SAFE_LOCAL_PATH = re.compile(r"(?:/[A-Za-z0-9_.+-]+)+\Z")
_SAFE_PROCESS = re.compile(r"[1-9][0-9]{0,18}\Z")
_IMMUTABLE_REVISION = re.compile(r"[0-9a-fA-F]{40,64}\Z")
_GPU_SELECTOR = re.compile(r"(?:GPU-[0-9a-fA-F-]{8,}|[0-9]+(?:,[0-9]+)*)\Z")


def validate_source_model_revision(revision: str | None) -> str:
    """Refuse anything but an immutable commit hash.

    A branch or tag would let the same bundle train against different weights
    on different days, which is exactly what a training run must not do.
    """

    if not isinstance(revision, str) or not _IMMUTABLE_REVISION.fullmatch(revision):
        raise TrainingLauncherError(
            "source model revision must be an immutable 40-64 character commit hash")

    return revision


@dataclass(frozen=True)
class TrainingExecutionConfiguration:
    launcher: str = "ssh"

    # No defaults for the four fields below. They named one institute's
    # machine and one institute's directory layout, which made a public
    # dataclass carry a private deployment -- and made an operator who
    # configured nothing launch against somebody else's host instead of being
    # told they had configured nothing.

    host: str = ""
    ssh_identity_file: str | None = None
    remote_root: str = ""
    axolotl_executable: str = "axolotl"
    python_executable: str = "python3"
    tokenizer_cache_root: str | None = None
    hf_cache_root: str | None = None
    cuda_visibility_policy: str = "operator_default"
    cuda_visible_devices: str | None = None

    # Conservative Qwen3-Next no-expert QLoRA planning floor; operators may
    # explicitly set None or a different policy after inspecting the host.

    minimum_free_vram_gib: float | None = 47.0
    minimum_model_storage_gib: float | None = 160.0
    inference_service_type: str = "manual_wrapper"
    inference_service_wrapper: str | None = None
    inference_service_binary: str | None = None
    inference_service_model: str | None = None
    inference_service_port: int = 8010
    inference_service_health_path: str = "/health"
    inference_service_gpu_uuid: str | None = None
    restart_inference_after_training: bool = True
    inference_stop_timeout_seconds: int = 30
    source_model_revision: str | None = None
    preprocess_examples: int = 3
    stop_timeout_seconds: int = 10

    def validate(self) -> None:
        if self.launcher != "ssh":
            raise TrainingLauncherError("only the ssh training launcher is configured")

        # "not configured" and "configured badly" are different operator
        # errors and get different sentences.

        if not self.host:
            raise TrainingLauncherError(
                "no SSH host configured for training (set \"host\" in the "
                "training execution configuration)")

        if not _SAFE_HOST.fullmatch(self.host):
            raise TrainingLauncherError("unsafe SSH host")

        if self.ssh_identity_file is not None:
            if (not PurePosixPath(self.ssh_identity_file).is_absolute() or
                    not _SAFE_LOCAL_PATH.fullmatch(self.ssh_identity_file)):
                raise TrainingLauncherError("unsafe SSH identity-file path")

        if not self.remote_root:
            raise TrainingLauncherError(
                "no remote training root configured (set \"remote_root\" in "
                "the training execution configuration)")

        root = PurePosixPath(self.remote_root)

        if (not root.is_absolute() or ".." in root.parts or self.remote_root == "/"
                or not _SAFE_REMOTE_ROOT.fullmatch(self.remote_root)):
            raise TrainingLauncherError("unsafe remote training root")

        if not _SAFE_EXECUTABLE.fullmatch(self.axolotl_executable):
            raise TrainingLauncherError("unsafe Axolotl executable")

        if not _SAFE_EXECUTABLE.fullmatch(self.python_executable):
            raise TrainingLauncherError("unsafe Python executable")

        for name, value in (("tokenizer cache", self.tokenizer_cache_root),
                            ("HF cache", self.hf_cache_root)):
            if value is not None:
                cache = PurePosixPath(value)

                if (not cache.is_absolute() or ".." in cache.parts or
                        not _SAFE_REMOTE_ROOT.fullmatch(value)):
                    raise TrainingLauncherError(f"unsafe {name} root")

        if self.cuda_visibility_policy not in {"operator_default", "all", "configured"}:
            raise TrainingLauncherError("unknown CUDA visibility policy")

        if self.cuda_visible_devices is not None and not _GPU_SELECTOR.fullmatch(self.cuda_visible_devices):
            raise TrainingLauncherError("unsafe CUDA visible-device selector")

        if (self.minimum_free_vram_gib is not None and
                self.minimum_free_vram_gib < 0):
            raise TrainingLauncherError("minimum free VRAM cannot be negative")

        if (self.minimum_model_storage_gib is not None and
                self.minimum_model_storage_gib < 0):
            raise TrainingLauncherError("minimum model storage cannot be negative")

        if self.inference_service_type not in {"manual_wrapper", "systemd", "disabled"}:
            raise TrainingLauncherError("unknown inference service type")

        for name, value in (("inference service wrapper", self.inference_service_wrapper),
                            ("inference service binary", self.inference_service_binary),
                            ("inference service model", self.inference_service_model)):
            if value is not None and (not PurePosixPath(value).is_absolute() or
                                      not _SAFE_REMOTE_ROOT.fullmatch(value)):
                raise TrainingLauncherError(f"unsafe {name} path")

        # The controller matches the running process on binary AND model, so a
        # service it is expected to manage cannot be identified without both.
        # Refused here rather than at the far end of an ssh call.

        if self.inference_service_type != "disabled":
            for name, value in (("binary", self.inference_service_binary),
                                ("model", self.inference_service_model)):
                if not value:
                    raise TrainingLauncherError(
                        f"inference service {name} path is not configured "
                        f"(required unless inference_service_type is "
                        f"\"disabled\")")

        if not 1 <= self.inference_service_port <= 65535:
            raise TrainingLauncherError("invalid inference service port")

        if not re.fullmatch(r"/[A-Za-z0-9_.-]+", self.inference_service_health_path):
            raise TrainingLauncherError("unsafe inference health path")

        if self.inference_service_gpu_uuid is not None and not _GPU_SELECTOR.fullmatch(self.inference_service_gpu_uuid):
            raise TrainingLauncherError("unsafe inference GPU selector")

        if not 1 <= self.inference_stop_timeout_seconds <= 300:
            raise TrainingLauncherError("invalid inference stop timeout")

        if not 1 <= self.preprocess_examples <= 20:
            raise TrainingLauncherError("preprocess example count must be between 1 and 20")

        if not 1 <= self.stop_timeout_seconds <= 60:
            raise TrainingLauncherError("stop timeout must be between 1 and 60 seconds")

        if self.source_model_revision is not None:
            validate_source_model_revision(self.source_model_revision)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "TrainingExecutionConfiguration":
        value = json.loads(Path(path).read_text(encoding="utf-8"))

        if not isinstance(value, dict):
            raise TrainingLauncherError("training execution config must be an object")

        allowed = set(cls.__dataclass_fields__)

        if set(value) - allowed:
            raise TrainingLauncherError("unknown training execution configuration fields")

        # Unknown fields were already refused above, so this cannot silently
        # ignore a setting the operator believed they had set.

        config = cls(**value)
        config.validate()

        return config


@dataclass(frozen=True)
class PreflightResult:
    success: bool
    summary: str
    axolotl_version: str | None = None
    hardware: Mapping[str, object] = field(default_factory=dict)
    checksum_verified: bool = False
    output: str = ""


@dataclass(frozen=True)
class StartResult:
    success: bool
    process_identity: Mapping[str, object] = field(default_factory=dict)
    summary: str = ""


@dataclass(frozen=True)
class StatusResult:
    process_running: bool
    terminal: bool = False
    exit_code: int | None = None
    started_at: str | None = None
    completed_at: str | None = None
    output_reference: str | None = None
    metrics: Mapping[str, object] = field(default_factory=dict)
    identity_valid: bool = True


class TrainingLauncher(Protocol):
    launcher_type: str

    def preflight(self, job: TrainingJob, bundle: Path, config_name: str) -> PreflightResult: ...

    def start(self, job: TrainingJob, bundle: Path, config_name: str) -> StartResult: ...

    def status(self, job: TrainingJob) -> StatusResult: ...

    def stop(self, job: TrainingJob) -> StatusResult: ...

    def logs(self, job: TrainingJob, lines: int) -> str: ...

    def inspect(self, job: TrainingJob) -> Mapping[str, object]: ...


class CommandRunner:
    def run(self, argv: Sequence[str], *, input_text: str | None = None,
            timeout: int = 30) -> subprocess.CompletedProcess[str]:
        return subprocess.run(list(argv), input=input_text, text=True,
                              capture_output=True, timeout=timeout, check=False)


class SSHTrainingLauncher:
    """A narrow SSH transport using trusted scripts and validated positional args."""

    launcher_type = "ssh"

    def __init__(self, configuration: TrainingExecutionConfiguration,
                 runner: CommandRunner | None = None) -> None:
        configuration.validate()
        self.configuration = configuration
        self.runner = runner or CommandRunner()

    def _ssh_options(self) -> list[str]:
        return (["-i", self.configuration.ssh_identity_file]
                if self.configuration.ssh_identity_file else [])

    def remote_directory(self, job_id: str) -> str:
        # TrainingJob validation supplies the opaque-id check.

        TrainingJobStoreShim.validate_job_id(job_id)

        root = PurePosixPath(self.configuration.remote_root)
        path = root / "jobs" / job_id

        # Belt and braces against a traversal that survived the id check: every
        # remote script re-checks this same shape before touching anything.

        if path.parent != root / "jobs":
            raise TrainingLauncherError("remote job path escaped configured root")

        return str(path)

    def _ssh(self, script: str, args: Sequence[str] = (), *, timeout: int = 30):
        """Run a trusted script remotely; only these arguments ever vary.

        The scripts are constants in this file. Values reach them as positional
        arguments, never by interpolation, so nothing a job carries can become
        remote shell code.
        """

        for value in args:
            if "\x00" in value or "\n" in value:
                raise TrainingLauncherError("unsafe remote argument")

        return self.runner.run(
            ["ssh", *self._ssh_options(), "--", self.configuration.host,
             "sh", "-s", "--", *args],
            input_text=script, timeout=timeout,
        )

    @staticmethod
    def _bounded(result: subprocess.CompletedProcess[str], limit: int = 16000) -> str:
        return ((result.stdout or "") + (result.stderr or ""))[-limit:]

    def probe(self) -> PreflightResult:
        """Check the host can run training at all, before anything is transferred."""

        script = r'''set -eu
printf 'hostname=%s\n' "$(hostname)"
if command -v "$1" >/dev/null 2>&1; then
  printf 'axolotl_path=%s\n' "$(command -v "$1")"
  version=$("$1" --version 2>&1 | head -n 1 || true)
  test -n "$version" && printf 'axolotl_version=%s\n' "$version" || true
else
  printf 'axolotl_missing=1\n'; exit 20
fi
command -v setsid >/dev/null 2>&1 || { printf 'setsid_missing=1\n'; exit 21; }
if command -v nvidia-smi >/dev/null 2>&1; then
  if test -n "$3"; then
    export CUDA_VISIBLE_DEVICES="$3"
    nvidia-smi --id="$3" --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || true
  else
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || true
  fi
else printf 'cuda_unknown=1\n'; fi
df -Pk "$2" 2>/dev/null | tail -n 1 || df -Pk "$(dirname "$2")" 2>/dev/null | tail -n 1 || true
'''
        result = self._ssh(script, (self.configuration.axolotl_executable,
                                    self.configuration.remote_root,
                                    self.configuration.cuda_visible_devices or ""))
        output = self._bounded(result)

        return PreflightResult(result.returncode == 0,
                               "remote environment available" if result.returncode == 0
                               else "remote environment preflight failed",
                               self._version(output), self._hardware(output), False, output)

    @staticmethod
    def _version(output: str) -> str | None:
        for line in output.splitlines():
            if line.startswith("axolotl_version=") and len(line) <= 220:
                return line.split("=", 1)[1].strip() or None

        return None

    @staticmethod
    def _hardware(output: str) -> Mapping[str, object]:
        gpu_lines = [line.strip() for line in output.splitlines()
                     if "," in line and any(char.isdigit() for char in line)]
        total_mib = 0.0
        free_mib = 0.0
        models = []

        for line in gpu_lines[:16]:
            fields = [item.strip() for item in line.split(",")]

            if len(fields) < 3:
                continue

            try:
                total_mib += float(fields[-2])
                free_mib += float(fields[-1])
                models.append(",".join(fields[:-2]))
            except ValueError:
                continue

        total_gib = total_mib / 1024 if total_mib else None

        if total_gib is None:
            feasibility = "UNKNOWN"
        elif total_gib >= 160:
            feasibility = "LIKELY_FITS"
        elif total_gib >= 96:
            feasibility = "MAY_FIT"
        elif total_gib < 48:
            feasibility = "LIKELY_DOES_NOT_FIT"
        else:
            feasibility = "UNKNOWN"

        return {"gpu_models": models, "gpu_count": len(models),
                "total_vram_gib": total_gib,
                "available_vram_gib": free_mib / 1024 if free_mib else None,
                "vram_feasibility": feasibility, "source": "remote_preflight",
                "warning": ("Peak memory depends on sequence length, batch size, LoRA "
                            "targets, optimizer, checkpointing, packing, and versions.")}

    def _prepare_remote(self, job: TrainingJob) -> subprocess.CompletedProcess[str]:
        remote = self.remote_directory(job.job_id)
        script = r'''set -eu
root=$1; job=$2; remote=$3
case "$remote" in "$root"/jobs/"$job") ;; *) exit 41;; esac
umask 077
mkdir -p "$remote/bundle" "$remote/output" "$remote/logs" "$remote/state"
printf '%s\n' "$job" > "$remote/.edgem-training-job"
'''

        return self._ssh(script, (self.configuration.remote_root, job.job_id, remote))

    def _transfer(self, bundle: Path, remote: str) -> subprocess.CompletedProcess[str]:
        # --checksum rather than size/mtime: a re-transfer must be decided by
        # content, and the remote side verifies checksums.sha256 afterwards.

        if not bundle.is_dir() or not (bundle / "checksums.sha256").is_file():
            raise TrainingLauncherError("invalid local training bundle")

        destination = f"{self.configuration.host}:{remote}/bundle/"
        rsh = (f"ssh -i {self.configuration.ssh_identity_file}"
               if self.configuration.ssh_identity_file else "ssh")

        return self.runner.run(["rsync", "--archive", "--checksum", "--protect-args",
                                "--rsh", rsh,
                                "--", str(bundle.resolve()) + "/", destination], timeout=600)

    def preflight(self, job: TrainingJob, bundle: Path, config_name: str) -> PreflightResult:
        """Probe, transfer, verify checksums, and let Axolotl preprocess the data.

        Nothing trains here. The point is to fail on this host, with a readable
        reason, rather than partway into a run that has already taken the GPU.
        """

        # The name goes into a remote path, so it is chosen from a fixed set
        # rather than merely sanitized.

        if config_name not in {"axolotl-sft.yml", "axolotl-kto.yml", "axolotl-dpo.yml"}:
            raise TrainingLauncherError("invalid training config name")

        probe = self.probe()

        if not probe.success:
            return probe

        created = self._prepare_remote(job)

        if created.returncode:
            return PreflightResult(False, "remote job directory creation failed",
                                   probe.axolotl_version, probe.hardware,
                                   output=self._bounded(created))

        remote = self.remote_directory(job.job_id)
        transferred = self._transfer(bundle, remote)

        if transferred.returncode:
            return PreflightResult(False, "bundle transfer failed", probe.axolotl_version,
                                   probe.hardware, output=self._bounded(transferred))

        script = r'''set -eu
root=$1; job=$2; remote=$3; ax=$4; config=$5; examples=$6; bundle_sum=$7; visible=$8
case "$remote" in "$root"/jobs/"$job") ;; *) exit 41;; esac
test "$(cat "$remote/.edgem-training-job")" = "$job"
cd "$remote/bundle"
if test -n "$visible"; then export CUDA_VISIBLE_DEVICES="$visible"; fi
actual=$(sha256sum checksums.sha256 | awk '{print $1}')
test "$actual" = "$bundle_sum"
sha256sum -c checksums.sha256
"$ax" preprocess "configs/$config" --debug --debug-num-examples "$examples"
'''
        result = self._ssh(script, (self.configuration.remote_root, job.job_id, remote,
                                    self.configuration.axolotl_executable, config_name,
                                    str(self.configuration.preprocess_examples),
                                    job.bundle_checksum,
                                    self.configuration.cuda_visible_devices or ""), timeout=1800)
        output = self._bounded(result)

        return PreflightResult(result.returncode == 0,
                               "preprocessing validation passed" if result.returncode == 0
                               else "preprocessing validation failed",
                               probe.axolotl_version, probe.hardware,
                               result.returncode == 0, output)

    def start(self, job: TrainingJob, bundle: Path, config_name: str) -> StartResult:
        remote = self.remote_directory(job.job_id)
        script = r'''set -eu
        root=$1; job=$2; remote=$3; ax=$4; config=$5; visible=$6
case "$remote" in "$root"/jobs/"$job") ;; *) exit 41;; esac
test "$(cat "$remote/.edgem-training-job")" = "$job"
cat > "$remote/run.sh" <<'SPEAR_RUN'
#!/bin/sh
set -u
remote=$1; ax=$2; config=$3
cd "$remote" || exit 90
umask 077
visible=$4
if test -n "$visible"; then export CUDA_VISIBLE_DEVICES="$visible"; fi
date -u +%Y-%m-%dT%H:%M:%SZ > state/started_at
printf '%s\n' "$$" > state/pid
ps -o pgid= -p "$$" | tr -d ' ' > state/pgid
"$ax" train "bundle/configs/$config" > logs/train.log 2>&1
code=$?
printf '%s\n' "$code" > "state/.exit_code.$$"
mv "state/.exit_code.$$" state/exit_code
date -u +%Y-%m-%dT%H:%M:%SZ > state/completed_at
exit "$code"
SPEAR_RUN
chmod 700 "$remote/run.sh"
nohup setsid "$remote/run.sh" "$remote" "$ax" "$config" "$visible" </dev/null >/dev/null 2>&1 &
pid=$!
printf 'pid=%s\npgid=%s\n' "$pid" "$pid"
'''
        result = self._ssh(script, (self.configuration.remote_root, job.job_id, remote,
                                    self.configuration.axolotl_executable, config_name,
                                    self.configuration.cuda_visible_devices or ""))

        if result.returncode:
            return StartResult(False, summary=self._bounded(result))

        values = self._key_values(result.stdout)

        if not _SAFE_PROCESS.fullmatch(values.get("pid", "")):
            return StartResult(False, summary="launcher returned invalid process identity")

        return StartResult(True, {"pid": int(values["pid"]),
                                  "pgid": int(values.get("pgid", values["pid"])),
                                  "marker": job.job_id}, "detached training process started")

    @staticmethod
    def _key_values(output: str) -> dict[str, str]:
        values = {}

        for line in output.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()

        return values

    def status(self, job: TrainingJob) -> StatusResult:
        remote = self.remote_directory(job.job_id)
        expected = str(job.remote_process_identity.get("pgid", ""))

        if expected and not _SAFE_PROCESS.fullmatch(expected):
            raise TrainingLauncherError("invalid stored process identity")

        script = r'''set -eu
root=$1; job=$2; remote=$3; expected=$4
case "$remote" in "$root"/jobs/"$job") ;; *) exit 41;; esac
test "$(cat "$remote/.edgem-training-job" 2>/dev/null || true)" = "$job" || { printf 'identity=invalid\n'; exit 0; }
test -f "$remote/state/started_at" && printf 'started_at=%s\n' "$(cat "$remote/state/started_at")"
if test -f "$remote/state/exit_code"; then
 printf 'terminal=1\nexit_code=%s\n' "$(cat "$remote/state/exit_code")"
 test -f "$remote/state/completed_at" && printf 'completed_at=%s\n' "$(cat "$remote/state/completed_at")"
elif test -n "$expected" && test "$(cat "$remote/state/pgid" 2>/dev/null || true)" = "$expected" && kill -0 -- "-$expected" 2>/dev/null; then
 printf 'running=1\n'
else printf 'lost=1\n'; fi
tail -n 50 "$remote/logs/train.log" 2>/dev/null | sed 's/^/SPEAR_LOG:/' || true
'''
        result = self._ssh(script, (self.configuration.remote_root, job.job_id, remote, expected))

        if result.returncode:
            raise TrainingLauncherError("remote status failed")

        values = self._key_values(result.stdout)

        # The marker file no longer names this job, so whatever is in that
        # directory is not ours to report on -- or to stop.

        if values.get("identity") == "invalid":
            return StatusResult(False, identity_valid=False)

        exit_code = int(values["exit_code"]) if values.get("exit_code", "").lstrip("-").isdigit() else None

        return StatusResult(values.get("running") == "1", values.get("terminal") == "1",
                            exit_code, values.get("started_at"), values.get("completed_at"),
                            remote + "/output", self._parse_metrics(result.stdout))

    @staticmethod
    def _parse_metrics(output: str) -> Mapping[str, object]:
        """Best-effort progress figures scraped from the tail of the training log.

        Trainers print these in several shapes and none of it is a contract, so
        an unreadable value is dropped rather than reported wrongly.
        """

        text = "\n".join(line[len("SPEAR_LOG:"):] for line in output.splitlines()
                         if line.startswith("SPEAR_LOG:"))[-16000:]
        patterns = {
            "loss": r"(?:['\"]?loss['\"]?)\s*[:=]\s*(-?[0-9]+(?:\.[0-9]+)?)",
            "step": r"(?:['\"]?step['\"]?)\s*[:=/]\s*([0-9]+)",
            "epoch": r"(?:['\"]?epoch['\"]?)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            "learning_rate": r"(?:learning_rate|lr)['\"]?\s*[:=]\s*([0-9.eE+-]+)",
        }
        metrics = {}

        for name, pattern in patterns.items():
            matches = re.findall(pattern, text, re.I)

            # The last occurrence is the most recent progress reported.

            if matches:
                raw = matches[-1]

                try:
                    metrics[name] = int(raw) if name == "step" else float(raw)
                except ValueError:
                    pass

        return metrics

    def stop(self, job: TrainingJob) -> StatusResult:
        """Signal the recorded process group, and only after proving it is ours."""

        # Checked here AND again remotely against the marker and the recorded
        # pgid: a signal sent to the wrong group would kill somebody else's work.

        remote = self.remote_directory(job.job_id)
        expected = str(job.remote_process_identity.get("pgid", ""))

        if not _SAFE_PROCESS.fullmatch(expected):
            raise TrainingLauncherError("cannot stop job without validated process group")

        script = r'''set -eu
root=$1; job=$2; remote=$3; expected=$4; timeout=$5
case "$remote" in "$root"/jobs/"$job") ;; *) exit 41;; esac
test "$(cat "$remote/.edgem-training-job")" = "$job"
actual=$(cat "$remote/state/pgid")
test "$actual" = "$expected"
case "$actual" in ''|*[!0-9]*) exit 42;; esac
kill -TERM -- "-$actual"
i=0
while kill -0 -- "-$actual" 2>/dev/null && test "$i" -lt "$timeout"; do sleep 1; i=$((i+1)); done
if kill -0 -- "-$actual" 2>/dev/null; then printf 'running=1\n'; else printf 'stopped=1\n'; fi
'''
        result = self._ssh(script, (self.configuration.remote_root, job.job_id, remote,
                                    expected, str(self.configuration.stop_timeout_seconds)),
                           timeout=self.configuration.stop_timeout_seconds + 10)

        if result.returncode:
            raise TrainingLauncherError("validated remote stop failed")

        values = self._key_values(result.stdout)

        return StatusResult(values.get("running") == "1",
                            terminal=values.get("stopped") == "1")

    def logs(self, job: TrainingJob, lines: int) -> str:
        if not 1 <= lines <= 500:
            raise TrainingLauncherError("log line count must be between 1 and 500")

        remote = self.remote_directory(job.job_id)
        script = r'''set -eu
root=$1; job=$2; remote=$3; lines=$4
case "$remote" in "$root"/jobs/"$job") ;; *) exit 41;; esac
test "$(cat "$remote/.edgem-training-job")" = "$job"
tail -n "$lines" "$remote/logs/train.log" 2>/dev/null || true
'''
        result = self._ssh(script, (self.configuration.remote_root, job.job_id,
                                    remote, str(lines)))

        if result.returncode:
            raise TrainingLauncherError("remote logs failed")

        return result.stdout[-65536:]

    def inspect(self, job: TrainingJob) -> Mapping[str, object]:
        status = self.status(job)
        return {"status": status.__dict__, "remote_directory": self.remote_directory(job.job_id)}


class TrainingJobStoreShim:
    @staticmethod
    def validate_job_id(job_id: str) -> str:
        # Avoid an import cycle while keeping exactly one externally visible format.

        if not re.fullmatch(r"ftjob_[0-9a-f]{24}", job_id or ""):
            raise TrainingLauncherError("unsafe training job id")

        return job_id
