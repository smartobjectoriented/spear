"""Fail-closed local and remote training-host diagnostics."""

from __future__ import annotations

import json
import os
import re
import tempfile
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from training_launcher import CommandRunner, TrainingExecutionConfiguration, TrainingLauncherError
from training_store import TrainingStore

REMOTE_READINESS_SCHEMA_VERSION = 1
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_.+:/ -]{0,240}$")


class RemoteReadinessState(StrEnum):
    UNKNOWN = "UNKNOWN"
    READY = "READY"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"


class SSHDiagnosis(StrEnum):
    OK = "ok"
    ALIAS_MISSING = "host_alias_missing"
    CONNECTIVITY_FAILURE = "dns_or_connectivity_failure"
    HOST_KEY_FAILURE = "host_key_failure"
    AUTHENTICATION_FAILURE = "authentication_unavailable"
    REMOTE_SHELL_FAILURE = "remote_shell_unavailable"
    TIMEOUT = "timeout"
    CONFIGURATION_FAILURE = "ssh_configuration_failure"


@dataclass(frozen=True)
class RemoteTrainingEnvironment:
    """Operator-managed environment contract; no shell fragments."""

    host: str
    remote_root: str
    python_executable: str
    axolotl_executable: str
    tokenizer_cache_root: str | None = None
    hf_cache_root: str | None = None
    cuda_visibility_policy: str = "operator_default"
    cuda_visible_devices: str | None = None

    @classmethod
    def from_configuration(cls, value: TrainingExecutionConfiguration) -> "RemoteTrainingEnvironment":
        return cls(value.host, value.remote_root, value.python_executable,
                   value.axolotl_executable, value.tokenizer_cache_root,
                   value.hf_cache_root, value.cuda_visibility_policy,
                   value.cuda_visible_devices)


@dataclass(frozen=True)
class RemoteReadinessReport:
    state: str
    reasons: tuple[str, ...]
    checks: Mapping[str, object]
    host: str
    checked_at: str
    schema_version: int = REMOTE_READINESS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RemoteReadinessReport":
        if int(value.get("schema_version", 0)) != REMOTE_READINESS_SCHEMA_VERSION:
            raise ValueError("unsupported remote readiness schema")

        return cls(str(value["state"]), tuple(value.get("reasons", ())),
                   dict(value.get("checks", {})), str(value["host"]),
                   str(value["checked_at"]), REMOTE_READINESS_SCHEMA_VERSION)


@dataclass(frozen=True)
class LocalDoctorReport:
    checks: Mapping[str, object]
    state: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    # Durable before the rename: a cached verdict is either the whole previous
    # one or the whole new one, never a truncated file.

    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)

    try:
        os.fchmod(fd, 0o600)

        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _diagnose_error(stderr: str, returncode: int | None = None) -> SSHDiagnosis:
    """Turn ssh's stderr into a cause an operator can act on."""

    text = (stderr or "").lower()

    if "permission denied" in text or "too many authentication" in text:
        return SSHDiagnosis.AUTHENTICATION_FAILURE

    if "host key" in text or "known_hosts" in text or "authenticity of host" in text:
        return SSHDiagnosis.HOST_KEY_FAILURE

    if "could not resolve" in text or "name or service not known" in text:
        return SSHDiagnosis.CONNECTIVITY_FAILURE

    if "connection timed out" in text or "connecttimeout" in text:
        return SSHDiagnosis.TIMEOUT

    if "connection refused" in text or "no route to host" in text:
        return SSHDiagnosis.CONNECTIVITY_FAILURE

    # Nothing recognised: a nonzero exit means the remote shell itself failed,
    # while a zero exit means ssh is fine and nothing is wrong to report.

    return SSHDiagnosis.REMOTE_SHELL_FAILURE if returncode else SSHDiagnosis.OK


class SSHRemoteProbe:
    """Fixed SSH diagnostics; this is not a general-purpose remote shell."""

    def __init__(self, configuration: TrainingExecutionConfiguration,
                 runner: CommandRunner | None = None, *, timeout: int = 20) -> None:
        configuration.validate()

        self.configuration = configuration
        self.environment = RemoteTrainingEnvironment.from_configuration(configuration)
        self.runner = runner or CommandRunner()

        # A diagnostic must never hang the operator's terminal, so the bound is
        # clamped rather than taken on trust.

        self.timeout = max(2, min(timeout, 30))

    def _ssh(self, script: str, args: tuple[str, ...] = ()):
        """Run the fixed probe script remotely, passing values as arguments only."""

        # The script is a constant; only these arguments vary, and each is
        # checked so nothing can be spliced into the remote shell.

        for value in args:
            if "\x00" in value or "\n" in value or not _SAFE_VALUE.fullmatch(value):
                raise TrainingLauncherError("unsafe remote probe argument")

        return self.runner.run(
            ["ssh", *(["-i", self.configuration.ssh_identity_file]
                       if self.configuration.ssh_identity_file else []),
             "-o", "BatchMode=yes", "-o", f"ConnectTimeout={self.timeout}",
             "--", self.configuration.host, "sh", "-s", "--", *args],
            input_text=script, timeout=self.timeout + 5)

    def diagnose_ssh(self) -> tuple[SSHDiagnosis, Mapping[str, object]]:
        try:
            # `ssh -G` resolves the alias without connecting, which separates
            # "this host is not configured" from "this host is unreachable".

            config = self.runner.run(
                ["ssh", *(["-i", self.configuration.ssh_identity_file]
                           if self.configuration.ssh_identity_file else []),
                 "-G", "--", self.configuration.host], timeout=self.timeout)

            if config.returncode:
                diagnosis = _diagnose_error(config.stderr, config.returncode)

                if diagnosis == SSHDiagnosis.REMOTE_SHELL_FAILURE:
                    diagnosis = SSHDiagnosis.ALIAS_MISSING

                return diagnosis, {"detail": "ssh configuration rejected host alias"}

            result = self.runner.run(
                ["ssh", *(["-i", self.configuration.ssh_identity_file]
                           if self.configuration.ssh_identity_file else []),
                 "-o", "BatchMode=yes", "-o", f"ConnectTimeout={self.timeout}",
                 "--", self.configuration.host, "true"], timeout=self.timeout + 2)
        except TimeoutError:
            return SSHDiagnosis.TIMEOUT, {"detail": "SSH connection timed out"}
        except OSError as exc:
            return SSHDiagnosis.CONFIGURATION_FAILURE, {"detail": type(exc).__name__}

        if result.returncode:
            diagnosis = _diagnose_error(result.stderr, result.returncode)

            return diagnosis, {"detail": diagnosis.value}

        return SSHDiagnosis.OK, {"detail": "non-interactive SSH succeeded"}

    def inspect(self) -> RemoteReadinessReport:
        """One bounded probe of the training host, returning a fail-closed verdict."""

        # Without SSH nothing else can be established, so the verdict is
        # UNKNOWN rather than BLOCKED: the host was never actually examined.

        diagnosis, ssh_check = self.diagnose_ssh()
        checks: dict[str, object] = {"ssh": {"status": diagnosis.value, **ssh_check}}

        if diagnosis != SSHDiagnosis.OK:
            return RemoteReadinessReport(RemoteReadinessState.UNKNOWN.value, (diagnosis.value,),
                                         checks, self.configuration.host, _now())

        # A single fixed POSIX-sh script: one round trip, no remote state, and
        # nothing the caller can influence but the four arguments below.

        script = r'''set -u
root=$1; py=$2; ax=$3; visible=$4
if test -n "$visible"; then export CUDA_VISIBLE_DEVICES="$visible"; fi
printf 'hostname=%s\n' "$(hostname 2>/dev/null || printf unknown)"
printf 'uname=%s\n' "$(uname -sr 2>/dev/null || printf unknown)"
printf 'python=%s\n' "$($py --version 2>&1 | head -n 1 || printf missing)"
if command -v "$ax" >/dev/null 2>&1; then
  printf 'axolotl_available=yes\n'
  printf 'axolotl_version=%s\n' "$("$ax" --version 2>&1 | head -n 1 || printf unknown)"
else printf 'axolotl_available=no\naxolotl_version=missing\n'; fi
if command -v nvidia-smi >/dev/null 2>&1; then
  printf 'cuda=available\n'
  if test -n "$visible"; then
    nvidia-smi --id="$visible" --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,driver_version --format=csv,noheader,nounits 2>/dev/null \
      | head -n 16 | while IFS= read -r line; do printf 'gpu=%s\n' "$line"; done
  else
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,driver_version --format=csv,noheader,nounits 2>/dev/null \
      | head -n 16 | while IFS= read -r line; do printf 'gpu=%s\n' "$line"; done
  fi
  if test -n "$visible"; then
    nvidia-smi --id="$visible" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null \
      | head -n 16 | while IFS= read -r line; do printf 'gpu_process=%s\n' "$line"; done
  fi
else printf 'cuda=unavailable\n'; fi
if test -d "$root" && test -r "$root" && test -x "$root"; then printf 'training_root=usable\n';
else printf 'training_root=unavailable\n'; fi
disk=$(df -Pk "$root" 2>/dev/null | awk 'NR==2 {print $4}'); total_disk=$(df -Pk "$root" 2>/dev/null | awk 'NR==2 {print $2}')
printf 'disk_available_kib=%s\n' "${disk:-unknown}"; printf 'disk_total_kib=%s\n' "${total_disk:-unknown}"
if find "$root/models" -maxdepth 3 -type f \( -name '*.safetensors' -o -name '*.bin' \) 2>/dev/null | head -n 1 | grep -q .; then
  printf 'model_snapshot_status=READY\n'
else printf 'model_snapshot_status=NOT_PRESENT\n'; fi
ram=$(awk '/MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || true); printf 'ram_kib=%s\n' "${ram:-unknown}"
for module in torch transformers peft bitsandbytes qwen3_next; do
  "$py" - "$module" <<'PY' 2>/dev/null || true
import importlib, sys
name = sys.argv[1]
try:
    module = importlib.import_module(name)
    print("package=%s:%s" % (name, getattr(module, "__version__", "available")))
except Exception:
    print("package=%s:missing" % name)
PY
done
'''

        try:
            result = self._ssh(script, (self.configuration.remote_root,
                                        self.configuration.python_executable,
                                        self.configuration.axolotl_executable,
                                        self.configuration.cuda_visible_devices or ""))
        except TimeoutError:
            return RemoteReadinessReport(RemoteReadinessState.UNKNOWN.value, (SSHDiagnosis.TIMEOUT.value,),
                                         checks, self.configuration.host, _now())
        except OSError as exc:
            checks["ssh"] = {"status": SSHDiagnosis.REMOTE_SHELL_FAILURE.value,
                             "detail": type(exc).__name__}
            return RemoteReadinessReport(RemoteReadinessState.UNKNOWN.value,
                                         (SSHDiagnosis.REMOTE_SHELL_FAILURE.value,), checks,
                                         self.configuration.host, _now())

        parsed = self._parse(result.stdout)
        checks.update(parsed)

        if result.returncode:
            checks["ssh"] = {"status": SSHDiagnosis.REMOTE_SHELL_FAILURE.value,
                             "detail": "remote inspection exited nonzero"}
            return RemoteReadinessReport(RemoteReadinessState.UNKNOWN.value,
                                         (SSHDiagnosis.REMOTE_SHELL_FAILURE.value,), checks,
                                         self.configuration.host, _now())

        # Every reason is collected before any verdict is formed, so the
        # report names all that is wrong rather than only the first thing.

        reasons: list[str] = []

        python_version = self._python_version(str(parsed.get("python", "")))

        if python_version is None or python_version < (3, 11):
            reasons.append("python_too_old" if python_version else "python_unavailable")

        if parsed.get("axolotl_available") != "yes":
            reasons.append("axolotl_missing")

        if parsed.get("cuda") != "available":
            reasons.append("cuda_unavailable")

        if not parsed.get("gpu"):
            reasons.append("no_gpu")

        if parsed.get("training_root") != "usable":
            reasons.append("training_root_unavailable")

        packages = parsed.get("package", {})

        for module in ("torch", "transformers", "peft", "bitsandbytes"):
            if packages.get(module, "missing") == "missing":
                reasons.append(f"{module}_missing")

        torch_version = self._package_version(packages.get("torch"))

        if torch_version is not None and torch_version < (2, 11):
            reasons.append("pytorch_too_old")

        # Free VRAM says whether the card is usable NOW; total VRAM, below,
        # says whether it could ever hold this model. They are separate verdicts.

        vram = self._free_vram(parsed.get("gpu", ()))

        if (self.configuration.minimum_free_vram_gib is not None and vram is not None
                and vram < self.configuration.minimum_free_vram_gib):
            reasons.append("insufficient_current_free_vram")

        capacity = self._hardware_capacity(parsed.get("gpu", ()),
                                           self.configuration.minimum_free_vram_gib)
        availability = self._availability(parsed, self.configuration.minimum_free_vram_gib)

        checks["hardware_capacity"] = capacity
        checks["current_gpu_availability"] = availability

        disk_state = self._disk_capacity(parsed.get("disk_available_kib"),
                                         self.configuration.minimum_model_storage_gib)
        checks["model_storage_capacity"] = disk_state

        # Only insufficient CAPACITY blocks: a busy card frees up, a card too
        # small for the model never will.

        if capacity == "INSUFFICIENT":
            reasons.append("hardware_capacity_insufficient")

        checks["source_model_revision"] = ("PINNED" if self.configuration.source_model_revision
                                             else "MISSING")

        if not self.configuration.source_model_revision:
            reasons.append("model_revision_missing")

        checks["environment_fingerprint"] = self._fingerprint(parsed)

        # The reasons that make a launch impossible rather than merely
        # inadvisable; anything outside this set leaves the host READY.

        hard = {"python_too_old", "python_unavailable", "axolotl_missing", "cuda_unavailable",
                "no_gpu", "training_root_unavailable", "torch_missing", "pytorch_too_old",
                "transformers_missing", "peft_missing", "bitsandbytes_missing",
                "model_revision_missing", "hardware_capacity_insufficient"}

        state = (RemoteReadinessState.BLOCKED if hard.intersection(reasons)
                 else RemoteReadinessState.READY)

        checks["environment_readiness"] = state.value
        checks["qwen3_next_compatibility"] = state.value

        return RemoteReadinessReport(state.value, tuple(dict.fromkeys(reasons)), checks,
                                     self.configuration.host, _now())

    @staticmethod
    def _parse(output: str) -> dict[str, object]:
        """Read the probe's key=value output; every value is bounded on the way in."""

        result: dict[str, object] = {"gpu": [], "package": {}}

        for line in output.splitlines()[:300]:
            if "=" not in line:
                continue

            key, value = line.split("=", 1)

            # gpu and package repeat, one line per card or module; every other
            # key appears once and simply overwrites.

            if key == "gpu":
                result["gpu"].append(value[:240])
            elif key == "package" and ":" in value:
                name, version = value.split(":", 1)
                result["package"][name] = version[:120]
            else:
                result[key] = value[:240]

        return result

    @staticmethod
    def _python_version(value: str) -> tuple[int, int] | None:
        """Major and minor from `python --version`, or None if it did not answer."""

        match = re.search(r"Python\s+(\d+)\.(\d+)", value)

        return (int(match.group(1)), int(match.group(2))) if match else None

    @staticmethod
    def _package_version(value: object) -> tuple[int, int] | None:
        """Major and minor of a package version; the rest of the string is ignored."""

        match = re.match(r"(\d+)\.(\d+)", str(value or ""))

        return (int(match.group(1)), int(match.group(2))) if match else None

    @staticmethod
    def _free_vram(values: object) -> float | None:
        """Free VRAM in GiB across the visible cards, or None if unreadable."""

        if not isinstance(values, (list, tuple)):
            return None

        total = 0.0

        for value in values:
            fields = [part.strip() for part in str(value).split(",")]

            if len(fields) >= 3:
                # Current probe emits name,total,used,free,utilization,driver.
                # Keep compatibility with older name,total,free[,driver] fixtures.

                free_index = 3 if len(fields) >= 6 else (-2 if len(fields) >= 4 else -1)

                # An unparsable row is skipped rather than failing the probe:
                # one odd card must not cost the whole verdict.

                try:
                    total += float(fields[free_index]) / 1024
                except ValueError:
                    pass

        return total or None

    @staticmethod
    def _hardware_capacity(values: object, requirement: float | None) -> str:
        """Whether the biggest card could ever hold the run, ignoring current load."""

        if not isinstance(values, (list, tuple)) or not values:
            return "UNKNOWN"

        totals = []

        for value in values:
            fields = [part.strip() for part in str(value).split(",")]

            try:
                totals.append(float(fields[1]) / 1024)
            except (IndexError, ValueError):
                pass

        if not totals or requirement is None:
            return "UNKNOWN" if not totals else "READY"

        # The largest single card, not the sum: the model has to fit on one.

        return "READY" if max(totals) >= requirement else "INSUFFICIENT"

    @classmethod
    def _availability(cls, parsed: Mapping[str, object], requirement: float | None) -> str:
        """Whether the card is free right now, unlike the capacity check above."""

        values = parsed.get("gpu", ())

        if not isinstance(values, (list, tuple)) or not values:
            return "UNKNOWN"

        # A named compute process settles it without arithmetic: something else
        # is already on the card.

        if parsed.get("gpu_process"):
            return "BUSY"

        free = cls._free_vram(values)

        return "UNKNOWN" if free is None else (
            "AVAILABLE" if requirement is None or free >= requirement else "BUSY")

    @staticmethod
    def _fingerprint(parsed: Mapping[str, object]) -> str:
        """Identify the environment a run was launched against."""

        # Only what could change a training outcome: hostname and free memory
        # move on their own and would make every probe look like a new host.

        relevant = {key: parsed.get(key) for key in
                    ("python", "axolotl_version", "cuda", "package", "gpu")}

        return hashlib.sha256(json.dumps(relevant, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def _disk_capacity(value: object, requirement: float | None) -> str:
        """Whether the training root has room for the model snapshot."""

        try:
            available_gib = float(str(value)) / (1024 * 1024)
        except (TypeError, ValueError):
            return "UNKNOWN"

        if requirement is None:
            return "READY"

        return "READY" if available_gib >= requirement else "INSUFFICIENT"


def local_doctor(store: TrainingStore | str | os.PathLike[str],
                 training_root: str | os.PathLike[str],
                 configuration: TrainingExecutionConfiguration | None,
                 *, model_revision: str | None = None) -> LocalDoctorReport:
    """What can be established without leaving this machine."""

    source = store if isinstance(store, TrainingStore) else TrainingStore(store)
    root = Path(training_root).resolve()

    # A store that has never been used is reported as not-yet-created rather
    # than missing: it is created on first use, and its absence blocks nothing.

    checks: dict[str, object] = {
        "training_store": {"status": "OK", "path": str(source.root)},
        "training_bundle": {"status": "OK" if (root / "bundles").is_dir() else "NOT_YET_CREATED"},
        "job_store": {"status": "OK" if (root / "jobs").is_dir() else "NOT_YET_CREATED"},
    }

    config = configuration or TrainingExecutionConfiguration()

    checks["launcher"] = "ssh"
    checks["host"] = config.host
    checks["ssh_identity"] = "CONFIGURED" if config.ssh_identity_file else "SYSTEM_DEFAULT"
    checks["model_source"] = "Qwen/Qwen3-Coder-Next"
    checks["model_revision"] = ("PINNED" if (model_revision or config.source_model_revision)
                                else "MISSING")

    reasons: list[str] = []

    if not (model_revision or config.source_model_revision):
        reasons.append("model_revision_missing")

    if configuration is None:
        checks["configuration"] = "DEFAULTS_ONLY"

    try:
        config.validate()
        checks["configuration_validation"] = "OK"
    except TrainingLauncherError as exc:
        checks["configuration_validation"] = "BLOCKED"
        checks["configuration_detail"] = str(exc)
        reasons.append("invalid_execution_configuration")

    state = RemoteReadinessState.BLOCKED.value if reasons else RemoteReadinessState.READY.value

    return LocalDoctorReport(checks, state, tuple(reasons))


def load_cached_remote_readiness(store: TrainingStore | str | os.PathLike[str]) -> RemoteReadinessReport | None:
    root = store.root if isinstance(store, TrainingStore) else Path(store).resolve()

    # Absent, unreadable or of an older schema all mean the same thing to the
    # caller: no verdict is available, so the launch gate stays closed.

    try:
        return RemoteReadinessReport.from_dict(
            json.loads((root / "remote-readiness.json").read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def save_remote_readiness(store: TrainingStore | str | os.PathLike[str],
                          report: RemoteReadinessReport) -> Path:
    root = store.root if isinstance(store, TrainingStore) else Path(store).resolve()
    path = root / "remote-readiness.json"

    _atomic_json(path, report.to_dict())

    return path
