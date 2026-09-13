"""Operator-only training control plane over FT3 bundles and launchers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Mapping

from training_bundle import (
    SourceModelProfile, TrainingBundleBuilder, TrainingBundleConfiguration,
    TrainingBundleError, VRAMFeasibility, validate_training_bundle,
)
from training_data import TrainingRedactionPolicy
from training_jobs import (
    TrainingJob, TrainingJobError, TrainingJobStatus, TrainingJobStore,
    TrainingMethod, utc_now,
)
from training_launcher import (
    SSHTrainingLauncher, StatusResult, TrainingExecutionConfiguration,
    TrainingLauncher, TrainingLauncherError, validate_source_model_revision,
)
from training_readiness import (
    ReadinessLevel, TrainingReadinessEvaluator, TrainingReadinessPolicy,
)
import trajectory_episodes
import training_promotion
from training_store import TrainingStore
from remote_readiness import (
    LocalDoctorReport, RemoteReadinessReport, RemoteReadinessState,
    SSHRemoteProbe, load_cached_remote_readiness, local_doctor,
    save_remote_readiness,
)
from inference_service import SSHInferenceServiceController, InferenceServiceError
from training_handoff import TrainingResourceHandoff, TrainingResourceLease


class TrainingControllerError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingControlResult:
    ok: bool
    action: str
    message: str
    readiness: Mapping[str, object] | None = None
    bundle_id: str | None = None
    bundle_path: str | None = None
    job: TrainingJob | None = None


_METHOD_CONFIG = {
    TrainingMethod.SFT: "axolotl-sft.yml",
    TrainingMethod.KTO: "axolotl-kto.yml",
    TrainingMethod.DPO: "axolotl-dpo.yml",
}


class TrainingController:
    """Deterministic orchestration called only by explicit operator commands."""

    def __init__(self, training_store: TrainingStore | str,
                 training_root: str | Path, *,
                 execution: TrainingExecutionConfiguration | None = None,
                 launcher: TrainingLauncher | None = None,
                 remote_probe: SSHRemoteProbe | None = None,
                 resource_handoff: TrainingResourceHandoff | None = None,
                 bundle_configuration: TrainingBundleConfiguration | None = None,
                 event_sink: Callable[[str, Mapping[str, object]], None] | None = None) -> None:
        self.training_store = (training_store if isinstance(training_store, TrainingStore)
                               else TrainingStore(training_store))
        self.training_root = Path(training_root).resolve()
        self.job_store = TrainingJobStore(self.training_root / "jobs")

        self.execution = execution
        self.launcher = launcher
        self.remote_probe = remote_probe
        self.resource_handoff = resource_handoff
        self.bundle_configuration = bundle_configuration or TrainingBundleConfiguration()
        self.event_sink = event_sink

    def _handoff(self) -> TrainingResourceHandoff | None:
        """The GPU handoff, built on first use; None when nothing owns the card."""

        if self.resource_handoff:
            return self.resource_handoff

        # Only a real SSH launch contends with the inference service for the
        # GPU; a fake launcher has no card to hand over.

        if not self.execution or getattr(self._launcher(), "launcher_type", "ssh") != "ssh":
            return None

        self.resource_handoff = TrainingResourceHandoff(
            SSHInferenceServiceController(self.execution),
            TrainingResourceLease(self.training_root / "assigned-gpu.lease"))

        return self.resource_handoff

    def _emit(self, event: str, **details: object) -> None:
        if self.event_sink:
            self.event_sink(event, details)

    @staticmethod
    def _method(value: str | TrainingMethod | None) -> TrainingMethod:
        if value is None:
            return TrainingMethod.SFT

        try:
            return value if isinstance(value, TrainingMethod) else TrainingMethod(value.upper())
        except ValueError as exc:
            raise TrainingControllerError("method must be sft, kto, or dpo") from exc

    @staticmethod
    def _stage(report: Mapping[str, object], method: TrainingMethod) -> Mapping[str, object]:
        """The readiness stage a training method depends on."""

        key = {TrainingMethod.SFT: "sft", TrainingMethod.KTO: "unpaired_preference",
               TrainingMethod.DPO: "paired_preference"}[method]

        return report[key]

    def readiness(self) -> Mapping[str, object]:
        return TrainingReadinessEvaluator(
            self.training_store,
            policy=self.bundle_configuration.readiness,
            governance=self.bundle_configuration.governance,
            split=self.bundle_configuration.split,
        ).evaluate().to_dict()

    def doctor(self, *, remote: bool = False) -> tuple[LocalDoctorReport, RemoteReadinessReport | None]:
        """Run local checks, and optionally one bounded fresh remote probe."""

        local = local_doctor(self.training_store, self.training_root, self.execution,
                             model_revision=(self.bundle_configuration.model.revision or
                                             (self.execution.source_model_revision
                                              if self.execution else None)))

        # Data readiness is folded into the local report so one command answers
        # both "can this host launch" and "is there anything to launch with".

        data = self.readiness()

        local = LocalDoctorReport(
            {**dict(local.checks), "data_readiness": {
                "sft": data.get("sft", {}).get("state", "UNKNOWN"),
                "unpaired": data.get("unpaired_preference", {}).get("state", "UNKNOWN"),
                "paired": data.get("paired_preference", {}).get("state", "UNKNOWN"),
                "recommendation": data.get("recommendation", "UNKNOWN"),
            }}, local.state, local.reasons)

        # Probing the remote host is never implicit: without --remote the last
        # cached verdict is returned, however old it is.

        if not remote:
            return local, load_cached_remote_readiness(self.training_store)

        configuration = self.execution or TrainingExecutionConfiguration()
        report = (self.remote_probe or SSHRemoteProbe(configuration)).inspect()

        # Who currently holds the GPU decides whether a launch may proceed, so
        # a reachable host is asked about its inference service as well.

        if (report.checks.get("ssh", {}).get("status") == "ok"
                and configuration.inference_service_type != "disabled"):
            try:
                service = SSHInferenceServiceController(configuration).status()
                checks = dict(report.checks)

                checks["inference_service"] = {
                    "status": service.status, "health": service.health,
                    "identity_verified": service.identity_verified,
                    "controller": configuration.inference_service_type,
                }

                # Only a service this harness can positively identify counts as
                # a workload it may stop; anything else is BUSY_UNKNOWN, which
                # start() refuses rather than displacing.

                checks["gpu_workload_owner"] = (
                    "BUSY_BY_MANAGED_INFERENCE"
                    if service.status == "RUNNING" and service.identity_verified
                    else ("BUSY_UNKNOWN"
                          if report.checks.get("current_gpu_availability") == "BUSY"
                          else "AVAILABLE"))

                report = RemoteReadinessReport(report.state, report.reasons, checks,
                                               report.host, report.checked_at,
                                               report.schema_version)
            except (InferenceServiceError, OSError):
                # An unanswerable service is treated as an unknown owner, which
                # is the conservative reading: it may well be using the card.

                checks = dict(report.checks)

                checks["inference_service"] = {
                    "status": "UNKNOWN", "health": "UNKNOWN",
                    "identity_verified": False,
                    "controller": configuration.inference_service_type,
                }
                checks["gpu_workload_owner"] = "BUSY_UNKNOWN"

                report = RemoteReadinessReport(report.state, report.reasons, checks,
                                               report.host, report.checked_at,
                                               report.schema_version)

        save_remote_readiness(self.training_store, report)

        return local, report

    def _configuration(self, *, force: bool = False) -> TrainingBundleConfiguration:
        config = self.bundle_configuration
        model = config.model

        if self.execution and self.execution.source_model_revision:
            model = replace(model, revision=self.execution.source_model_revision)

        if force:
            # Threshold override only. Governance, checksums, structure, and model
            # validation remain exactly the production policies.

            old = config.readiness
            readiness = replace(
                old, sft_smoke_train_samples=1, minimum_validation_samples=0,
                minimum_test_samples=0, unpaired_smoke_per_class=1,
                paired_smoke_pairs=1, allow_preference_without_sft_baseline=True,
            )
            config = replace(config, readiness=readiness)

        return replace(config, model=model)

    def ingest(self, path: str | Path) -> TrainingControlResult:
        """Turn what usage recorded into episodes the builders can read.

        The two halves of the pipeline were built apart: the harness appends
        a row per turn to trajectories.jsonl, the exporters read canonical
        episodes, and nothing joined them -- so recorded usage sat in a file
        no trainer opens. Nothing here judges anything: the verdict was
        decided when the turn ran, by the project's own build and tests.

        Converting twice is safe. An episode's identifier comes from its
        content, so a row already stored is written again to the same name
        rather than duplicated.
        """
        source = Path(path)

        if not source.is_file():
            raise TrainingControllerError(f"no trajectory file at {source}")

        self._emit("finetune_ingest_started", source=str(source))

        stored = skipped = refused = positive = 0
        problems: list[str] = []

        # Conversion says what the row IS; promotion says what may be done
        # with it. Both run here, so nothing unscrubbed reaches the store and
        # nothing unverified is ever labelled worth imitating.
        for row, error in trajectory_episodes.rows_from_file(source):
            if error is not None:
                skipped += 1

                if len(problems) < 3:
                    problems.append(error)

                continue

            try:
                episode, decision = training_promotion.promote(row)
            except trajectory_episodes.TrajectoryConversionError as exc:
                skipped += 1

                if len(problems) < 3:
                    problems.append(str(exc))

                continue

            if episode is None:
                refused += 1

                if len(problems) < 3:
                    problems.append(decision.describe())

                continue

            self.training_store.save_episode(episode)
            stored += 1
            positive += 1 if decision.positive else 0

        self._emit("finetune_ingest_finished", stored=stored, skipped=skipped,
                   refused=refused, positive=positive)

        detail = f"; {'; '.join(problems)}" if problems else ""
        counts = (f"{stored} episode(s) stored ({positive} to imitate), "
                  f"{refused} refused, {skipped} row(s) skipped")

        return TrainingControlResult(
            ok=skipped == 0 and refused == 0, action="ingest",
            message=f"Ingested {source}: {counts}{detail}")

    def prepare(self, method: str | TrainingMethod | None = None, *,
                force: bool = False) -> TrainingControlResult:
        selected = self._method(method)

        self._emit("finetune_prepare_started", method=selected.value)

        config = self._configuration(force=force)

        summary, path = TrainingBundleBuilder(
            self.training_store, configuration=config,
        ).freeze(self.training_root)

        assert path is not None

        manifest = validate_training_bundle(path, self.training_store)
        readiness = json.loads((path / "readiness.json").read_text())
        stage = self._stage(readiness, selected)
        config_path = path / "configs" / _METHOD_CONFIG[selected]

        # Both halves are required: the stage may be ready and the bundle still
        # carry no config for this method.

        ready = stage["level"] != ReadinessLevel.NOT_READY.value and config_path.is_file()

        self._emit("finetune_bundle_selected", bundle_id=manifest["bundle_id"],
                   method=selected.value, ready=ready)

        reasons = ", ".join(stage.get("reasons", ())) or "none"
        revision = manifest["source_model"].get("revision")

        message = (f"Prepared bundle {manifest['bundle_id']} for {selected.value}. "
                   f"Readiness: {stage['state']} / {stage['level']}. Reasons: {reasons}. "
                   f"Source revision: {revision or 'UNPINNED (start will refuse)' }.")

        return TrainingControlResult(ready, "prepare", message, readiness,
                                     manifest["bundle_id"], str(path))

    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _hard_bundle_checks(self, path: Path, manifest: Mapping[str, object],
                            method: TrainingMethod) -> tuple[Path, str]:
        """The gates a bundle must clear to be trained; `force` never relaxes these."""

        if manifest.get("smoke_test_only"):
            raise TrainingControllerError("smoke/synthetic bundle cannot be remotely trained")

        source = manifest["source_model"]

        SourceModelProfile(**source).validate()

        # An unpinned revision would make the run irreproducible, so the source
        # model has to name a concrete one.

        if not source.get("revision"):
            raise TrainingControllerError("a concrete source model revision is required")

        try:
            validate_source_model_revision(source.get("revision"))
        except TrainingLauncherError as exc:
            raise TrainingControllerError(str(exc)) from exc

        config_path = path / "configs" / _METHOD_CONFIG[method]

        if not config_path.is_file():
            raise TrainingControllerError("requested training configuration is not active")

        # The config is re-hashed against the manifest: what is launched must be
        # the file the bundle was frozen with, not one edited since.

        expected = manifest["config_checksums"].get(f"configs/{config_path.name}")
        actual = self._digest(config_path)

        if not expected or actual != expected:
            raise TrainingControllerError("training configuration checksum mismatch")

        return config_path, actual

    def _launcher(self) -> TrainingLauncher:
        if self.launcher:
            return self.launcher

        if not self.execution:
            raise TrainingControllerError("training execution is not configured")

        self.launcher = SSHTrainingLauncher(self.execution)

        return self.launcher

    def start(self, method: str | TrainingMethod | None = None, *,
              force: bool = False) -> TrainingControlResult:
        selected = self._method(method)
        initial = self.readiness()
        initial_stage = self._stage(initial, selected)
        metrics = initial_stage.get("metrics", {})

        # KTO trains on both labels, so its sample count is the two classes
        # together; the others count only what was auto-approved.

        if selected == TrainingMethod.KTO:
            available = int(metrics.get("positive", 0)) + int(metrics.get("negative", 0))
        else:
            available = int(metrics.get("auto_approved", 0))

        # No data at all is refused even under --force: the override relaxes
        # thresholds, and there is no threshold below zero samples.

        if available == 0:
            reasons = ", ".join(initial_stage.get("reasons", ())) or "no_training_samples"

            return TrainingControlResult(
                False, "start", f"Fine-tuning not started. {selected.value} readiness: "
                f"{initial_stage['state']} / {initial_stage['level']}. Reason: {reasons}. "
                f"Recommendation: {initial['recommendation']}.", initial)

        if initial_stage["level"] == ReadinessLevel.NOT_READY.value and not force:
            return TrainingControlResult(
                False, "start", "Fine-tuning not started: stage is not ready.", initial)

        # A real SSH launch must have a recent operator-triggered remote doctor
        # result. Fake launchers are intentionally exempt for deterministic tests.

        if (self.execution and getattr(self._launcher(), "launcher_type", "ssh") == "ssh"):
            remote = load_cached_remote_readiness(self.training_store)

            if remote is None or remote.state not in {
                RemoteReadinessState.READY.value, RemoteReadinessState.WARNING.value,
            }:
                reason = "remote_readiness_unknown" if remote is None else \
                    ",".join(remote.reasons) or "remote_not_ready"

                return TrainingControlResult(
                    False, "start",
                    f"Fine-tuning not started: remote host is not ready ({reason}).", initial)

            if remote.state == RemoteReadinessState.WARNING.value and not force:
                return TrainingControlResult(
                    False, "start",
                    "Fine-tuning not started: remote host has warnings; use --force to acknowledge.",
                    initial)

            # An uncontrolled workload is never displaced: the handoff can only
            # stop a service this harness positively identified as its own.

            owner = remote.checks.get("gpu_workload_owner")

            if owner in {"BUSY_BY_OTHER_WORKLOAD", "BUSY_UNKNOWN"}:
                return TrainingControlResult(
                    False, "start",
                    "Fine-tuning not started: assigned GPU is busy by an uncontrolled workload.",
                    initial)

            if remote.checks.get("model_storage_capacity") == "INSUFFICIENT":
                return TrainingControlResult(
                    False, "start",
                    "Fine-tuning not started: model storage capacity is insufficient.", initial)

        prepared = self.prepare(selected, force=force)
        path = Path(prepared.bundle_path or "")

        manifest = validate_training_bundle(path, self.training_store)
        config_path, config_checksum = self._hard_bundle_checks(path, manifest, selected)

        readiness = prepared.readiness or initial
        stage = self._stage(readiness, selected)

        if stage["level"] == ReadinessLevel.NOT_READY.value:
            return TrainingControlResult(
                False, "start", "Fine-tuning not started: stage remains not ready.", readiness)

        launcher = self._launcher()

        # The same bundle, method and config already running is a duplicate
        # launch, which would have two jobs writing the same remote output.

        active = {TrainingJobStatus.STARTING.value, TrainingJobStatus.RUNNING.value,
                  TrainingJobStatus.PREFLIGHT.value}

        for existing in self.job_store.list():
            if (existing.status in active and existing.bundle_id == manifest["bundle_id"]
                    and existing.method == selected.value
                    and existing.training_config_checksum == config_checksum):
                raise TrainingControllerError(f"equivalent job already active: {existing.job_id}")

        job_id = self.job_store.new_job_id()
        remote = (launcher.remote_directory(job_id)
                  if isinstance(launcher, SSHTrainingLauncher) else f"fake://jobs/{job_id}")

        # The job records everything the run is pinned to, so its provenance
        # survives independently of the bundle directory.

        job = TrainingJob(
            job_id, selected.value, manifest["bundle_id"], str(path),
            self._digest(path / "checksums.sha256"), config_checksum,
            manifest["source_model"]["base_model"], manifest["source_model"]["revision"],
            manifest["dataset_checksums"], launcher.launcher_type,
            self.execution.host if self.execution else "fake", remote,
            output_reference=remote + "/output", logs_reference=remote + "/logs/train.log",
            readiness_override=force,
        )

        self.job_store.create(job)

        # Taking the GPU means stopping the inference service, which is recorded
        # on the job so it can be restored however the run ends.

        handoff = self._handoff()
        handoff_metadata: dict[str, object] = {}

        if handoff:
            gpu_uuid = (self.execution.cuda_visible_devices if self.execution else None)

            if not gpu_uuid and getattr(launcher, "launcher_type", "ssh") != "ssh":
                gpu_uuid = "fake-gpu"

            # Without a GPU identity the lease cannot name what it holds, and a
            # handoff that cannot be named cannot be safely reversed.

            if not gpu_uuid:
                job = self.job_store.transition(
                    job, TrainingJobStatus.FAILED,
                    failure_summary="assigned GPU UUID is required for handoff",
                    completion_time=utc_now())

                return TrainingControlResult(
                    False, "start", "Training handoff refused: GPU identity is missing.",
                    readiness, job.bundle_id, job.bundle_path, job)

            acquired = handoff.acquire(job.job_id, gpu_uuid)

            handoff_metadata = {"service_identity": self.execution.inference_service_wrapper,
                                "before_status": "RUNNING" if acquired.was_running else "STOPPED",
                                "stopped_by_job": acquired.stopped_by_job,
                                "restart_required": acquired.restart_required,
                                "stop_timestamp": utc_now() if acquired.stopped_by_job else None}

            job = self.job_store.save(TrainingJob.from_dict({**job.to_dict(),
                "inference_handoff": handoff_metadata}), event="finetune_inference_handoff")

            if not acquired.ok:
                job = self.job_store.transition(job, TrainingJobStatus.FAILED,
                                                failure_summary=acquired.message,
                                                completion_time=utc_now())

                return TrainingControlResult(
                    False, "start", f"Training handoff refused: {acquired.message}.",
                    readiness, job.bundle_id, job.bundle_path, job)

        job = self.job_store.transition(job, TrainingJobStatus.PREFLIGHT,
                                        event="finetune_preflight_started")

        self._emit("finetune_preflight_started", job_id=job.job_id)

        # The preflight measures the remote host rather than trusting the
        # bundle's own hardware table, which is why it runs before starting.

        try:
            preflight = launcher.preflight(job, path, config_path.name)
        except Exception as exc:
            preflight = None
            failure = str(exc)
        else:
            failure = preflight.summary

        if not preflight or not preflight.success or not preflight.checksum_verified:
            if handoff and handoff_metadata:
                handoff_metadata = dict(handoff.restore(job.job_id, handoff_metadata))

            job = self.job_store.transition(
                job, TrainingJobStatus.FAILED, event="finetune_preflight_failed",
                failure_summary=failure[:1000], completion_time=utc_now(),
                inference_handoff=handoff_metadata)

            return TrainingControlResult(False, "start", f"Preflight failed: {failure}", readiness,
                                         job.bundle_id, job.bundle_path, job)

        if (preflight.hardware.get("vram_feasibility") ==
                VRAMFeasibility.LIKELY_DOES_NOT_FIT.value and not force):
            if handoff and handoff_metadata:
                handoff_metadata = dict(handoff.restore(job.job_id, handoff_metadata))

            job = self.job_store.transition(job, TrainingJobStatus.FAILED,
                                            failure_summary="remote hardware likely does not fit",
                                            completion_time=utc_now(),
                                            inference_handoff=handoff_metadata)

            return TrainingControlResult(
                False, "start",
                "Remote hardware likely does not fit; use --force to acknowledge.",
                readiness, job.bundle_id, job.bundle_path, job)

        job = self.job_store.transition(
            job, TrainingJobStatus.STARTING,
            training_library_metadata={"axolotl_version": preflight.axolotl_version or "unknown"},
            hardware_metadata=dict(preflight.hardware))

        # A raised launch and a launch that reports failure are the same
        # outcome here: the job failed and the GPU has to go back.

        try:
            started = launcher.start(job, path, config_path.name)
        except Exception as exc:
            if handoff and handoff_metadata:
                handoff_metadata = dict(handoff.restore(job.job_id, handoff_metadata))

            job = self.job_store.transition(job, TrainingJobStatus.FAILED,
                                            event="finetune_job_failed",
                                            failure_summary=str(exc)[:1000],
                                            completion_time=utc_now(),
                                            inference_handoff=handoff_metadata)

            return TrainingControlResult(False, "start", "Training launch failed.", readiness,
                                         job.bundle_id, job.bundle_path, job)

        if not started.success:
            if handoff and handoff_metadata:
                handoff_metadata = dict(handoff.restore(job.job_id, handoff_metadata))

            job = self.job_store.transition(job, TrainingJobStatus.FAILED,
                                            event="finetune_job_failed",
                                            failure_summary=started.summary[:1000],
                                            completion_time=utc_now(),
                                            inference_handoff=handoff_metadata)

            return TrainingControlResult(False, "start", "Training launch failed.", readiness,
                                         job.bundle_id, job.bundle_path, job)

        job = self.job_store.transition(job, TrainingJobStatus.RUNNING,
                                        event="finetune_job_started",
                                        remote_process_identity=dict(started.process_identity),
                                        start_time=utc_now())

        return TrainingControlResult(True, "start", f"Training job {job.job_id} is RUNNING.",
                                     readiness, job.bundle_id, job.bundle_path, job)

    def _resolve_job(self, job_id: str | None, *, active_only: bool = False) -> TrainingJob:
        """The job an operator meant, which is only implicit when exactly one fits."""

        if job_id:
            return self.job_store.load(job_id)

        jobs = self.job_store.list()

        if active_only:
            jobs = [job for job in jobs if job.status in {
                TrainingJobStatus.PREFLIGHT.value, TrainingJobStatus.STARTING.value,
                TrainingJobStatus.RUNNING.value, TrainingJobStatus.STOPPING.value}]

        if len(jobs) != 1:
            raise TrainingControllerError("specify a job ID" if jobs else "no training job exists")

        return jobs[0]

    def refresh(self, job: TrainingJob) -> TrainingJob:
        """Reconcile one job against the remote process that is actually running."""

        if job.status not in {TrainingJobStatus.RUNNING.value,
                              TrainingJobStatus.STOPPING.value,
                              TrainingJobStatus.STARTING.value}:
            return job

        result = self._launcher().status(job)

        if result.terminal:
            # A nonzero exit is a failure unless a stop was requested, in which
            # case it is the stop taking effect.

            status = (TrainingJobStatus.COMPLETED if result.exit_code == 0
                      else (TrainingJobStatus.STOPPED
                            if job.status == TrainingJobStatus.STOPPING.value
                            else TrainingJobStatus.FAILED))
            event = ("finetune_job_completed" if status == TrainingJobStatus.COMPLETED
                     else "finetune_job_failed")

            # However the run ended, the inference service gets its GPU back --
            # once: the recorded attempt is what keeps this idempotent.

            handoff = job.inference_handoff

            if self.resource_handoff and handoff and not handoff.get("restart_attempted"):
                handoff = self.resource_handoff.restore(job.job_id, handoff)

            return self.job_store.transition(
                job, status, event=event,
                exit_code=result.exit_code,
                completion_time=result.completed_at or utc_now(),
                output_reference=result.output_reference or job.output_reference,
                inference_handoff=handoff)

        # A running job whose process or ownership marker is gone is LOST, not
        # failed: what became of it is genuinely unknown from here.

        if not result.process_running or not result.identity_valid:
            return self.job_store.transition(
                job, TrainingJobStatus.LOST,
                failure_summary="remote process or ownership marker missing")

        if result.metrics and dict(result.metrics) != dict(job.progress_metrics):
            return self.job_store.transition(job, TrainingJobStatus(job.status),
                                             event="finetune_job_status_refreshed",
                                             progress_metrics=dict(result.metrics))

        return job

    def status(self) -> tuple[Mapping[str, object], list[TrainingJob]]:
        report = dict(self.readiness())

        # Remote readiness is reported from cache only: status must not reach
        # out over SSH, which is what `doctor --remote` is for.

        cached = load_cached_remote_readiness(self.training_store)

        report["remote_readiness"] = (cached.to_dict() if cached else {
            "state": RemoteReadinessState.UNKNOWN.value,
            "reasons": ["remote_readiness_not_checked"],
            "checked_at": None,
        })

        jobs = []

        for job in self.job_store.list():
            jobs.append(self.refresh(job))

        return report, jobs

    def list_jobs(self) -> list[TrainingJob]:
        return [self.refresh(job) for job in self.job_store.list()]

    def inspect(self, job_id: str) -> TrainingJob:
        return self.refresh(self.job_store.load(job_id))

    def stop(self, job_id: str | None = None) -> TrainingJob:
        job = self.refresh(self._resolve_job(job_id, active_only=True))

        if job.status != TrainingJobStatus.RUNNING.value:
            raise TrainingControllerError("job is not running")

        job = self.job_store.transition(job, TrainingJobStatus.STOPPING,
                                        event="finetune_job_stop_requested")

        result = self._launcher().stop(job)

        # STOPPING stands until the remote process is actually gone; a later
        # refresh() closes the job if the stop takes longer than this call.

        if result.terminal and not result.process_running:
            handoff = job.inference_handoff

            if self.resource_handoff and handoff and not handoff.get("restart_attempted"):
                handoff = self.resource_handoff.restore(job.job_id, handoff)

            return self.job_store.transition(job, TrainingJobStatus.STOPPED,
                                             event="finetune_job_stopped",
                                             completion_time=utc_now(),
                                             inference_handoff=handoff)

        return job

    def logs(self, job_id: str | None = None, lines: int = 100) -> str:
        if not 1 <= lines <= 500:
            raise TrainingControllerError("log line count must be between 1 and 500")

        job = self._resolve_job(job_id)
        raw = self._launcher().logs(job, lines)

        # Training logs echo the config and the environment, so they are
        # redacted before being cached or shown.

        redacted, _ = TrainingRedactionPolicy().redact(raw)
        content = str(redacted)[-65536:]

        self.job_store.cache_logs(job.job_id, content)

        return content
