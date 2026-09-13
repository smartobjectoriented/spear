"""Operator-only control for the configured inference service."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Sequence

from training_launcher import CommandRunner, TrainingExecutionConfiguration


class InferenceServiceError(RuntimeError):
    pass


class InferenceServiceStatus(StrEnum):
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    STOPPING = "STOPPING"
    UNKNOWN = "UNKNOWN"


class GPUWorkloadOwner(StrEnum):
    AVAILABLE = "AVAILABLE"
    BUSY_BY_MANAGED_INFERENCE = "BUSY_BY_MANAGED_INFERENCE"
    BUSY_BY_OTHER_WORKLOAD = "BUSY_BY_OTHER_WORKLOAD"
    BUSY_UNKNOWN = "BUSY_UNKNOWN"


@dataclass(frozen=True)
class InferenceServiceResult:
    status: str
    health: str
    identity_verified: bool
    process_group: str | None = None
    detail: str = ""


class InferenceServiceController(Protocol):
    def status(self) -> InferenceServiceResult: ...

    def stop(self) -> InferenceServiceResult: ...

    def start(self) -> InferenceServiceResult: ...

    def wait_stopped(self) -> InferenceServiceResult: ...

    def wait_ready(self) -> InferenceServiceResult: ...


class SSHInferenceServiceController:
    """Fixed-command controller; never accepts a PID or shell from the CLI."""

    def __init__(self, configuration: TrainingExecutionConfiguration, runner=None) -> None:
        configuration.validate()

        # Only a wrapper this harness itself declared may be controlled: the
        # whole point is never to stop a process someone else owns.

        if (configuration.inference_service_type != "manual_wrapper"
                or not configuration.inference_service_wrapper):
            raise InferenceServiceError("manual inference wrapper is not configured")

        # The GPU UUID is how a running process is proven to be OUR service
        # rather than a lookalike on another card.

        if not configuration.inference_service_gpu_uuid:
            raise InferenceServiceError("inference GPU UUID is not configured")

        self.configuration = configuration
        self.runner = runner or CommandRunner()

    def _ssh(self, script: str, args: Sequence[str] = (), *, timeout: int = 30):
        """Run a fixed script remotely, passing values as arguments only."""

        for value in args:
            if "\x00" in value or "\n" in value:
                raise InferenceServiceError("unsafe inference argument")

        return self.runner.run(
            ["ssh",
             *(["-i", self.configuration.ssh_identity_file]
               if self.configuration.ssh_identity_file else []),
             "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             "--", self.configuration.host, "sh", "-s", "--", *args],
            input_text=script, timeout=timeout)

    def status(self) -> InferenceServiceResult:
        """Find the service and prove it is ours before reporting it as running."""

        c = self.configuration

        # The process is matched on binary, model and port, and only then is
        # its CUDA_VISIBLE_DEVICES compared with the configured card. A match
        # on the wrong GPU is reported as UNKNOWN, never as our service.

        script = r'''set -u
binary=$1; model=$2; port=$3; gpu=$4; health_path=$5
found=$(ps -eo pid=,pgid=,args= 2>/dev/null | awk -v b="$binary" -v m="$model" -v p="--port $port" '$3==b && index($0,m)&&index($0,p){print; exit}')
if test -z "$found"; then printf 'status=STOPPED\nhealth=NOT_READY\nidentity=absent\n'; exit 0; fi
pid=$(printf '%s\n' "$found" | awk '{print $1}'); pgid=$(printf '%s\n' "$found" | awk '{print $2}')
env_gpu=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -n 1)
if test "$env_gpu" != "$gpu"; then printf 'status=UNKNOWN\nhealth=UNKNOWN\nidentity=wrong_gpu\n'; exit 0; fi
code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$port$health_path" 2>/dev/null || printf 000)
if test "$code" = 200; then health=READY; else health=NOT_READY; fi
printf 'status=RUNNING\nhealth=%s\nidentity=verified\npid=%s\npgid=%s\n' "$health" "$pid" "$pgid"
'''

        result = self._ssh(script, (c.inference_service_binary,
                                    c.inference_service_model,
                                    str(c.inference_service_port),
                                    c.inference_service_gpu_uuid,
                                    c.inference_service_health_path))

        if result.returncode:
            return InferenceServiceResult("UNKNOWN", "UNKNOWN", False,
                                          detail="remote status failed")

        values = self._values(result.stdout)

        return InferenceServiceResult(values.get("status", "UNKNOWN"),
                                      values.get("health", "UNKNOWN"),
                                      values.get("identity") == "verified",
                                      values.get("pgid"),
                                      values.get("identity", ""))

    def stop(self) -> InferenceServiceResult:
        current = self.status()

        if current.status == "STOPPED":
            return current

        # Nothing is signalled unless status() positively identified the
        # process as ours; an unverified match is refused, not killed.

        if current.status != "RUNNING" or not current.identity_verified:
            raise InferenceServiceError("inference identity was not verified; refusing stop")

        # The process group comes back from the remote host, so it is checked
        # here as well before being interpolated into a kill.

        pgid = current.process_group or ""

        if not re.fullmatch(r"[1-9][0-9]{0,18}", pgid):
            raise InferenceServiceError("verified process group is invalid")

        # TERM to the whole group, then wait: the wrapper spawns children, and
        # signalling the leader alone would leave them holding the card.

        script = r'''set -eu
pgid=$1; timeout=$2
case "$pgid" in ''|*[!0-9]*) exit 42;; esac
kill -TERM -- "-$pgid"
i=0
while kill -0 -- "-$pgid" 2>/dev/null && test "$i" -lt "$timeout"; do sleep 1; i=$((i+1)); done
if kill -0 -- "-$pgid" 2>/dev/null; then printf 'status=STOPPING\n'; else printf 'status=STOPPED\n'; fi
'''

        result = self._ssh(script,
                           (pgid, str(self.configuration.inference_stop_timeout_seconds)),
                           timeout=self.configuration.inference_stop_timeout_seconds + 12)

        if result.returncode:
            raise InferenceServiceError("validated inference stop failed")

        return self.wait_stopped()

    def start(self) -> InferenceServiceResult:
        # setsid detaches the wrapper into its own session, so it survives the
        # SSH connection closing and gets the process group stop() will use.

        script = ('set -eu\nwrapper=$1; test -x "$wrapper"\n'
                  'nohup setsid "$wrapper" </dev/null >/dev/null 2>&1 &\n'
                  'printf "status=STARTING\\n"\n')

        result = self._ssh(script, (self.configuration.inference_service_wrapper,))

        if result.returncode:
            raise InferenceServiceError("inference wrapper start failed")

        return self.wait_ready()

    def wait_stopped(self):
        """Poll until the service is gone, or report whatever it still is."""

        for _ in range(max(1, self.configuration.inference_stop_timeout_seconds)):
            result = self.status()

            if result.status == "STOPPED":
                return result

            time.sleep(1)

        return self.status()

    def wait_ready(self):
        """Poll until the service answers its health check, or give up saying so."""

        # Running is not enough: the model has to be loaded before the service
        # is of any use, which is what the health check reports.

        for _ in range(30):
            result = self.status()

            if result.status == "RUNNING" and result.health == "READY":
                return result

            time.sleep(1)

        return self.status()

    @staticmethod
    def _values(output):
        """Read the script's key=value lines; anything else is ignored."""

        values = {}

        for line in output.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()

        return values


class FakeInferenceServiceController:
    """Deterministic test double; it never executes a remote process."""

    def __init__(self, *, running=True, owner=GPUWorkloadOwner.BUSY_BY_MANAGED_INFERENCE,
                 stop_ok=True, start_ok=True, release_ok=True):
        self.running = running
        self.owner = owner

        # The three flags drive the failure paths a test wants to exercise:
        # a stop that raises, a start that raises, and -- release_ok -- a stop
        # that reports success without actually freeing the card.

        self.stop_ok = stop_ok
        self.start_ok = start_ok
        self.release_ok = release_ok

        self.stop_calls = 0
        self.start_calls = 0

    def status(self):
        return (InferenceServiceResult("RUNNING", "READY", True, "1") if self.running
                else InferenceServiceResult("STOPPED", "NOT_READY", True))

    def stop(self):
        self.stop_calls += 1

        if not self.stop_ok:
            raise InferenceServiceError("fake stop failed")

        if self.release_ok:
            self.running = False

        return self.status()

    def start(self):
        self.start_calls += 1

        if not self.start_ok:
            raise InferenceServiceError("fake start failed")

        self.running = True

        return self.status()

    def wait_stopped(self):
        return self.status()

    def wait_ready(self):
        return self.status()
