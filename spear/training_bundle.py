"""Immutable, reproducible training bundles; this module never runs training."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Sequence

from preference_dataset import PreferenceConfiguration, PreferenceDatasetBuilder
from sft_dataset import (
    SFTDatasetBuilder, SFTExportConfiguration, _canonical_bytes, _hash,
    _validate_structure,
)
from training_governance import DataOrigin, TrainingDataGovernancePolicy
from training_readiness import (
    ReadinessLevel, TrainingReadinessEvaluator, TrainingReadinessPolicy,
)
from training_store import TrainingStore
from training_splits import SplitConfiguration


BUNDLE_SCHEMA_VERSION = 1
AXOLOTL_PROFILE_VERSION = 1
MASKING_POLICY_VERSION = 1


class TrainingBundleError(RuntimeError):
    pass


class SourceModelKind(StrEnum):
    INSTRUCT = "instruct_post_trained"
    BASE = "base_pretrained"


class VRAMFeasibility(StrEnum):
    LIKELY_FITS = "LIKELY_FITS"
    MAY_FIT = "MAY_FIT"
    LIKELY_DOES_NOT_FIT = "LIKELY_DOES_NOT_FIT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SourceModelProfile:
    profile: str = "qwen3_coder_next_sft"
    base_model: str = "Qwen/Qwen3-Coder-Next"
    source_kind: str = SourceModelKind.INSTRUCT.value
    revision: str | None = None

    @classmethod
    def named(cls, name: str) -> "SourceModelProfile":
        if name == "qwen3_coder_next_sft":
            return cls()

        if name == "qwen3_coder_next_base":
            return cls(name, "Qwen/Qwen3-Coder-Next-Base", SourceModelKind.BASE.value)

        raise ValueError(f"unknown source model profile: {name}")

    def validate(self) -> None:
        # A GGUF path here means someone pointed training at a quantized
        # inference build, which cannot be fine-tuned.

        if self.base_model.lower().rstrip("/").endswith(".gguf") or ".gguf/" in self.base_model.lower():
            raise TrainingBundleError("GGUF is an inference artifact, not a training source")

        if self.source_kind not in {item.value for item in SourceModelKind}:
            raise TrainingBundleError("unknown source model kind")


@dataclass(frozen=True)
class AxolotlProfile:
    sequence_len: int = 32768
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 0.0001
    num_epochs: int = 1
    max_steps: int | None = None
    sample_packing: bool = False
    gradient_checkpointing: str = "true"
    lora_r: int = 16
    lora_alpha: int = 32
    moe_targeting: str = "conservative_no_expert_targeting"
    expert_target_modules: tuple[str, ...] = ()

    # `load_in_4bit` alone does NOT quantize this model's routed experts under
    # transformers 5.x, where Qwen3NextExperts fuses them into 3-D
    # nn.Parameter tensors that bitsandbytes -- which replaces nn.Linear --
    # cannot reach. Axolotl added quantize_moe_experts for exactly that: it
    # intercepts the load, quantizes to NF4 and frees the bf16 copies.
    # Emitting a QLoRA config without it asks for 4-bit and gets 148 GiB of
    # bf16 experts anyway, which is how a run ended up on a 179 GiB card.
    # Both halves are measured, not read: LAYOUT_ONLY=1 in
    # qwen3-finetune/cloud/load_preflight.py builds the model on the meta
    # device and reports per_expert_modules (512,) on transformers 4.57.6
    # against fused_parameter (512, 1024, 2048) on 5.9.0, over the same
    # 77.3B expert parameters -- 145 GiB left in bf16 versus 39 GiB in NF4.

    quantize_moe_experts: bool = True

    def __post_init__(self) -> None:
        if self.sequence_len <= 0 or self.micro_batch_size <= 0:
            raise ValueError("invalid Axolotl size setting")

        if self.moe_targeting not in {"conservative_no_expert_targeting",
                                      "expert_targeting"}:
            raise ValueError("unknown MoE targeting profile")

        # The profile and the module list have to agree in both directions:
        # expert targeting materially changes the VRAM the run will need, so it
        # is never inferred from one of the two being set.

        if self.moe_targeting == "expert_targeting" and not self.expert_target_modules:
            raise ValueError("expert targeting requires explicit module names")

        if (self.moe_targeting == "conservative_no_expert_targeting" and
                self.expert_target_modules):
            raise ValueError("conservative profile cannot target expert modules")


@dataclass(frozen=True)
class TrainingHardwareReport:
    cuda_available: bool | None
    gpu_models: tuple[str, ...]
    gpu_count: int
    total_vram_gib: float | None
    compute_capabilities: tuple[str, ...]
    system_ram_gib: float | None
    disk_available_gib: float | None
    platform: str
    source: str
    vram_feasibility: str = VRAMFeasibility.UNKNOWN.value
    warnings: tuple[str, ...] = ()

    @classmethod
    def detect(cls, path: str | os.PathLike[str] = ".") -> "TrainingHardwareReport":
        models: list[str] = []
        memories: list[float] = []
        capabilities: list[str] = []

        # Every probe below is best-effort: this host is not necessarily the
        # training host, and an undetectable GPU is reported as absent rather
        # than raised, so bundling still works from a laptop.

        try:
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ], capture_output=True, text=True, timeout=2, check=False)

            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    fields = [item.strip() for item in line.split(",")]

                    if len(fields) >= 2:
                        models.append(fields[0])
                        memories.append(float(fields[1]) / 1024)

                        if len(fields) >= 3:
                            capabilities.append(fields[2])
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

        ram = None

        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    ram = int(line.split()[1]) / 1024 / 1024
        except (OSError, ValueError, IndexError):
            pass

        # The output root may not exist yet, so free space is measured on the
        # nearest ancestor that does.

        disk_path = Path(path).resolve()

        while not disk_path.exists() and disk_path != disk_path.parent:
            disk_path = disk_path.parent

        disk = shutil.disk_usage(disk_path).free / 1024 ** 3

        return cls(bool(models), tuple(models), len(models), sum(memories) if memories else None,
                   tuple(capabilities), ram, disk, platform.platform(), "local_detection")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TrainingHardwareReport":
        return cls(
            value.get("cuda_available"), tuple(value.get("gpu_models", ())),
            int(value.get("gpu_count", 0)),
            float(value["total_vram_gib"]) if value.get("total_vram_gib") is not None else None,
            tuple(value.get("compute_capabilities", ())),
            float(value["system_ram_gib"]) if value.get("system_ram_gib") is not None else None,
            float(value["disk_available_gib"]) if value.get("disk_available_gib") is not None else None,
            str(value.get("platform", "unknown")), str(value.get("source", "target_json")),
            str(value.get("vram_feasibility", VRAMFeasibility.UNKNOWN.value)),
            tuple(value.get("warnings", ())),
        )

    def classify(self, *, quantize_moe_experts: bool | None = None,
                 expert_targeting: bool = False) -> "TrainingHardwareReport":
        """Feasibility, which depends on HOW the experts are loaded.

        A bare threshold table cannot answer this. For an 80B MoE, 96 GiB is
        insufficient or ample depending on one flag: with the routed experts
        left in bf16 the weights alone measured 152 010 MiB on a B200, and
        with them quantized Axolotl documents ~47 GiB (no expert targeting)
        and ~71 GiB (with). Told nothing, this stays coarse and pessimistic.

        Nothing here returns LIKELY_FITS on the quantized path: those numbers
        are Axolotl's, not ours, and the harness has already been bitten once
        by a documented figure that did not survive contact with the actual
        stack. MAY_FIT means "provision it and run the load preflight", not
        "go".
        """
        extra: tuple[str, ...] = ()

        if self.total_vram_gib is None:
            value = VRAMFeasibility.UNKNOWN
        elif quantize_moe_experts is False:
            # Experts stay bf16: the measured 148 GiB of weights, plus
            # activations, optimizer and logits.

            value = (VRAMFeasibility.LIKELY_FITS if self.total_vram_gib >= 160
                     else VRAMFeasibility.LIKELY_DOES_NOT_FIT)
            extra = ("quantize_moe_experts is false: the routed experts load in "
                     "bf16 and the weights alone need ~148 GiB.",)
        elif quantize_moe_experts is True:
            # Never LIKELY_FITS here: the figures are Axolotl's own, and the
            # docstring above says why a documented number is not a measurement.

            floor = 80.0 if expert_targeting else 56.0
            value = (VRAMFeasibility.MAY_FIT if self.total_vram_gib >= floor
                     else VRAMFeasibility.LIKELY_DOES_NOT_FIT)
            extra = (f"Quantized-expert path: Axolotl documents ~{71 if expert_targeting else 47} "
                     f"GiB for this model. Unverified here -- confirm with a load "
                     f"preflight before committing a training run.",)
        elif self.total_vram_gib >= 160:
            value = VRAMFeasibility.LIKELY_FITS
        elif self.total_vram_gib >= 96:
            value = VRAMFeasibility.MAY_FIT
        elif self.total_vram_gib < 48:
            value = VRAMFeasibility.LIKELY_DOES_NOT_FIT
        else:
            value = VRAMFeasibility.UNKNOWN

        if quantize_moe_experts is None:
            extra = ("MoE load mode not declared: this verdict assumes nothing "
                     "and is therefore coarse.",)

        warning = ("Peak memory depends on sequence length, batch size, LoRA targets, "
                   "optimizer, checkpointing, packing, and library versions.")

        self = replace(self, warnings=tuple(self.warnings) + extra)

        return TrainingHardwareReport(
            self.cuda_available, self.gpu_models, self.gpu_count,
            self.total_vram_gib, self.compute_capabilities,
            self.system_ram_gib, self.disk_available_gib, self.platform,
            self.source, value.value, tuple(dict.fromkeys((*self.warnings, warning))),
        )


@dataclass(frozen=True)
class FrozenDatasetSnapshot:
    snapshot_id: str
    schema_version: int
    source_episode_checksums: Mapping[str, str]
    sft_samples: Mapping[str, Mapping[str, str]]
    unpaired_samples: Mapping[str, Mapping[str, str]]
    paired_samples: Mapping[str, Mapping[str, str]]
    split_policy: Mapping[str, object]
    policy_versions: Mapping[str, int]


@dataclass(frozen=True)
class TrainingBundleConfiguration:
    model: SourceModelProfile = field(default_factory=SourceModelProfile)
    axolotl: AxolotlProfile = field(default_factory=AxolotlProfile)
    readiness: TrainingReadinessPolicy = field(default_factory=TrainingReadinessPolicy)
    governance: TrainingDataGovernancePolicy = field(default_factory=TrainingDataGovernancePolicy)
    split: SplitConfiguration = field(default_factory=SplitConfiguration)
    tokenizer_path: str | None = None
    hardware_report: TrainingHardwareReport | None = None
    smoke_test_only: bool = False


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    # Durable before the rename: a bundle file is either wholly there or not
    # there at all, which is what the checksum list is later able to assume.

    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)

    try:
        os.fchmod(descriptor, 0o600)

        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float)):
        return str(value)

    return json.dumps(str(value))


class AxolotlConfigGenerator:
    """Small deterministic generator, based on Axolotl's semantic JSONL formats."""

    def __init__(self, model: SourceModelProfile, profile: AxolotlProfile) -> None:
        model.validate()
        self.model = model
        self.profile = profile

    def _common(self, dataset: str, output: str) -> list[str]:
        p = self.profile
        lines = [
            f"base_model: {_yaml_scalar(self.model.base_model)}",
            *( [f"revision: {_yaml_scalar(self.model.revision)}"] if self.model.revision else []),
            "tokenizer_type: AutoTokenizer", "trust_remote_code: false",
            "load_in_4bit: true",
            f"quantize_moe_experts: {_yaml_scalar(p.quantize_moe_experts)}",
            "adapter: qlora", "lora_r: " + str(p.lora_r),
            "lora_alpha: " + str(p.lora_alpha), "lora_dropout: 0.0",
            "lora_target_linear: false", "lora_target_modules:",
            "  - q_proj", "  - k_proj", "  - v_proj", "  - o_proj",
            # Routed experts are PARAMETERS, not modules: they are fused 3-D
            # tensors, so naming them under lora_target_modules matches
            # nothing. Axolotl takes them under lora_target_parameters.
            *( ["lora_target_parameters:"] if p.expert_target_modules else []),
            *(f"  - {item}" for item in p.expert_target_modules),
            f"sequence_len: {p.sequence_len}",
            f"micro_batch_size: {p.micro_batch_size}",
            f"gradient_accumulation_steps: {p.gradient_accumulation_steps}",
            f"learning_rate: {p.learning_rate}", f"num_epochs: {p.num_epochs}",
            *( [f"max_steps: {p.max_steps}"] if p.max_steps is not None else []),
            f"gradient_checkpointing: {p.gradient_checkpointing}",
            f"sample_packing: {_yaml_scalar(p.sample_packing)}",
            "eval_sample_packing: false", "optimizer: adamw_torch_4bit",
            "lr_scheduler: cosine", "bf16: auto", "tf32: true",
            "val_set_size: 0", f"output_dir: {_yaml_scalar(output)}",
            "datasets:", f"  - path: {_yaml_scalar(dataset)}", "    ds_type: json",
        ]

        return lines

    def sft(self, include_validation: bool = True) -> str:
        lines = self._common("datasets/sft/train.jsonl", "outputs/sft")
        lines.extend([
            "    type: chat_template", "    field_messages: messages",
            "    field_tools: tools", "    message_field_training: train",
            "    roles_to_train: []", "    train_on_eos: turn",
            "chat_template: tokenizer_default",
        ])

        if include_validation:
            lines.extend([
                "test_datasets:", "  - path: datasets/sft/validation.jsonl",
                "    type: chat_template", "    field_messages: messages",
                "    field_tools: tools", "    message_field_training: train",
                "    roles_to_train: []",
            ])

        return "\n".join(lines) + "\n"

    def kto(self) -> str:
        lines = self._common("datasets/unpaired/train.jsonl", "outputs/kto")
        lines.remove("    ds_type: json")
        lines[0:0] = ["# Requires Axolotl preprocessing validation on the training host."]
        lines.extend(["    type:", "      field_prompt: prompt",
                      "      field_completion: completion", "      field_label: label",
                      "rl: kto", "rl_beta: 0.1", "kto_desirable_weight: 1.0",
                      "kto_undesirable_weight: 1.0", "remove_unused_columns: false",
                      "chat_template: tokenizer_default"])

        return "\n".join(lines) + "\n"

    def dpo(self) -> str:
        lines = self._common("datasets/paired/train.jsonl", "outputs/dpo")
        lines.remove("    ds_type: json")
        lines.extend(["    type:", "      field_prompt: prompt", "      field_chosen: chosen",
                      "      field_rejected: rejected", "rl: dpo", "dpo_beta: 0.1",
                      "remove_unused_columns: false", "chat_template: tokenizer_default"])

        return "\n".join(lines) + "\n"


class TokenizerValidator:
    def __init__(self, tokenizer_path: str | None = None, tokenizer=None) -> None:
        self.path = tokenizer_path
        self.tokenizer = tokenizer

    def validate(self, records: Sequence[Mapping[str, object]], sequence_len: int
                 ) -> Mapping[str, object]:
        tokenizer = self.tokenizer

        # Without a real tokenizer no exact count is possible, so the result
        # says so and defers the check to Axolotl's own preprocessing step.

        if tokenizer is None and self.path:
            try:
                from transformers import AutoTokenizer  # optional dependency

                tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
            except (ImportError, OSError, ValueError) as exc:
                return {"exact": False, "kind": "approximate",
                        "requires_preprocessing_validation": True,
                        "requires_axolotl_preprocessing_validation": True,
                        "reason": f"local_tokenizer_unavailable:{type(exc).__name__}"}

        if tokenizer is None:
            return {"exact": False, "kind": "approximate",
                    "requires_preprocessing_validation": True,
                    "requires_axolotl_preprocessing_validation": True,
                    "reason": "tokenizer_not_supplied"}

        lengths, overlong = [], []

        for index, record in enumerate(records):
            # Tool schemas are part of the rendered prompt, so a record that
            # carries them must be measured with them.

            kwargs = {"tokenize": True, "add_generation_prompt": False}

            if record.get("tools"):
                kwargs["tools"] = record["tools"]

            tokens = tokenizer.apply_chat_template(record["messages"], **kwargs)
            length = len(tokens["input_ids"] if isinstance(tokens, Mapping) else tokens)

            lengths.append(length)

            if length > sequence_len:
                overlong.append(index)

        return {"exact": True, "kind": "exact", "tokenizer": self.path or "injected",
                "count": len(lengths), "minimum": min(lengths, default=0),
                "maximum": max(lengths, default=0), "overlong_indexes": overlong,
                "requires_preprocessing_validation": False,
                "requires_axolotl_preprocessing_validation": True}


def validate_loss_mask_intent(record: Mapping[str, object]) -> None:
    messages = record.get("messages")

    if not isinstance(messages, list) or not messages:
        raise TrainingBundleError("SFT record has no messages")

    # Exactly one trainable message, and it must be the final assistant turn:
    # anything else means loss would be taken on context the model was given.

    trained = [index for index, item in enumerate(messages) if item.get("train") is True]

    if trained != [len(messages) - 1] or messages[-1].get("role") != "assistant":
        raise TrainingBundleError("one-target loss-mask invariant violated")

    # Masking must be stated, not merely absent, so a missing field is refused
    # rather than read as False.

    if any(item.get("train") is not False for item in messages[:-1]):
        raise TrainingBundleError("context message is not explicitly masked")


class TrainingBundleBuilder:
    def __init__(self, store: TrainingStore | str | os.PathLike[str], *,
                 configuration: TrainingBundleConfiguration | None = None,
                 tokenizer=None) -> None:
        self.store = store if isinstance(store, TrainingStore) else TrainingStore(store)
        self.configuration = configuration or TrainingBundleConfiguration()
        self.configuration.model.validate()
        self.tokenizer = tokenizer

    def _inputs(self):
        checksums = {}
        episodes = {}
        origins = Counter()
        excluded = Counter()

        # Sorted by episode id so the bundle is byte-identical across runs.

        for metadata in sorted(self.store.iterate_metadata(),
                               key=lambda item: str(item.get("episode_id"))):
            episode = self.store.load_episode(str(metadata["episode_id"]))
            decision = self.configuration.governance.assess(episode)

            origins[decision.origin] += 1

            if decision.allowed:
                episodes[episode.episode_id] = episode
                checksums[episode.episode_id] = str(metadata["checksum"])
            else:
                excluded.update(decision.reasons)

        # The builders run over the whole store: the split assignment must not
        # depend on which episodes governance happens to allow, so filtering
        # comes after, never by restricting what they were given.

        sft_build = SFTDatasetBuilder(
            self.store, configuration=SFTExportConfiguration(
                split=self.configuration.split,
            ),
        ).build()
        pref_build = PreferenceDatasetBuilder(
            self.store, configuration=PreferenceConfiguration(
                split=self.configuration.split,
            ),
        ).build()

        allowed = set(episodes)

        sft = [item for item in sft_build.samples if item.episode_id in allowed]
        unpaired = [item for item in pref_build.unpaired if item.source_episode_id in allowed]

        # A pair needs BOTH of its episodes allowed; one excluded side would
        # leave a preference with nothing to compare against.

        paired = [item for item in pref_build.paired
                  if item.chosen_episode_id in allowed and item.rejected_episode_id in allowed]

        return episodes, checksums, sft, unpaired, paired, sft_build, pref_build, origins, excluded

    @staticmethod
    def _sft_record(sample) -> dict[str, object]:
        record = sample.trainer_record()
        record["messages"] = [dict(message, train=(index == sample.target_message_index))
                              for index, message in enumerate(record["messages"])]
        validate_loss_mask_intent(record)

        return record

    def freeze(self, output_root: str | os.PathLike[str], *, dry_run: bool = False
               ) -> tuple[Mapping[str, object], Path | None]:
        (episodes, checksums, sft, unpaired, paired, sft_build, pref_build,
         origins, excluded) = self._inputs()

        sft_records = [self._sft_record(item) for item in sft]

        token_validation = TokenizerValidator(
            self.configuration.tokenizer_path, self.tokenizer,
        ).validate(sft_records, self.configuration.axolotl.sequence_len)

        # A sample longer than the configured sequence would be silently
        # truncated by the trainer, so it is dropped from the bundle instead.

        overlong = set(token_validation.get("overlong_indexes", ()))

        if overlong:
            sft = [item for index, item in enumerate(sft) if index not in overlong]
            sft_records = [item for index, item in enumerate(sft_records) if index not in overlong]

        readiness = TrainingReadinessEvaluator(
            self.store, policy=self.configuration.readiness,
            governance=self.configuration.governance,
            split=self.configuration.split,
        ).evaluate(token_validation=token_validation)

        # The evaluator did not see the exclusion, so the verdict is amended
        # here: a warning while samples remain, NOT_READY once none do.

        if overlong:
            stage = readiness.sft
            metrics = {**stage.metrics, "exact_overlong_excluded": len(overlong),
                       "auto_approved_in_bundle": len(sft)}

            if sft:
                stage = replace(stage, metrics=metrics,
                                warnings=tuple(dict.fromkeys(
                                    (*stage.warnings, "exact_token_overlong_excluded"))))
            else:
                stage = replace(stage, state="NEEDS_REVIEW", level="NOT_READY",
                                reasons=tuple(dict.fromkeys(
                                    (*stage.reasons, "all_samples_exact_token_overlong"))),
                                metrics=metrics)
                readiness = replace(readiness, recommendation="NEEDS_MANUAL_DATA_REVIEW")

            readiness = replace(readiness, sft=stage)

        # Everything the bundle's identity depends on. The id is derived from
        # this alone, so an identical input set always yields the same bundle.

        logical = {
            "schema": BUNDLE_SCHEMA_VERSION, "sources": checksums,
            "sft": [(item.sample_id, _hash(item.to_dict()), item.split) for item in sft],
            "unpaired": [(item.sample_id, _hash(item.to_dict()), item.split) for item in unpaired],
            "paired": [(item.preference_id, _hash(item.to_dict()), item.split) for item in paired],
            "model": asdict(self.configuration.model),
            "axolotl": asdict(self.configuration.axolotl),
            "readiness": asdict(self.configuration.readiness),
            "governance": self.configuration.governance.to_dict(),
            "split": asdict(self.configuration.split),
            "smoke_test_only": self.configuration.smoke_test_only,
            "hardware": (asdict(self.configuration.hardware_report)
                         if self.configuration.hardware_report else None),
        }
        bundle_id = "training_" + _hash(logical)[:24]

        snapshot = FrozenDatasetSnapshot(
            "snapshot_" + _hash(logical)[:24], BUNDLE_SCHEMA_VERSION, checksums,
            {item.sample_id: {"checksum": _hash(item.to_dict()), "split": item.split,
                              "episode_id": item.episode_id,
                              "split_group_id": item.split_group_id} for item in sft},
            {item.sample_id: {"checksum": _hash(item.to_dict()), "split": item.split,
                              "episode_id": item.source_episode_id,
                              "split_group_id": item.split_group_id} for item in unpaired},
            {item.preference_id: {"checksum": _hash(item.to_dict()), "split": item.split,
                                  "chosen_episode_id": item.chosen_episode_id,
                                  "rejected_episode_id": item.rejected_episode_id,
                                  "split_group_id": item.split_group_id}
             for item in paired},
            {"method": "shared_sha256_split_group", "version": 1,
             **asdict(self.configuration.split)},
            {"sft_selection": 1, "preference_outcome": 1,
             "readiness": self.configuration.readiness.policy_version,
             "governance": self.configuration.governance.policy_version,
             "masking": MASKING_POLICY_VERSION},
        )

        # Dated by its newest source episode rather than by the clock, so
        # rebuilding the same inputs produces a byte-identical manifest.

        created = max((episode.timestamp for episode in episodes.values()),
                      default="1970-01-01T00:00:00+00:00")

        manifest = {
            "bundle_id": bundle_id, "schema_version": BUNDLE_SCHEMA_VERSION,
            "smoke_test_only": self.configuration.smoke_test_only,
            "creation_timestamp": created, "source_training_store": str(self.store.root),
            "source_model": asdict(self.configuration.model),
            "frozen_snapshot": asdict(snapshot), "logical_fingerprint": _hash(logical),
            "dataset_counts": {"sft": len(sft), "unpaired": len(unpaired),
                               "paired": len(paired)},
            "dataset_checksums": {}, "config_checksums": {},
            "token_validation": token_validation,
            "masking": {"policy_version": MASKING_POLICY_VERSION,
                        "message_field": "train", "trainable_messages_per_sample": 1,
                        "prior_assistant_messages_masked": True,
                        "axolotl_preprocess_debug_required": True},
            "axolotl_profile_version": AXOLOTL_PROFILE_VERSION,
            "hardware_compatibility": (asdict(self.configuration.hardware_report)
                                       if self.configuration.hardware_report else
                                       {"vram_feasibility": VRAMFeasibility.UNKNOWN.value,
                                        "reason": "training_host_not_specified"}),
            "moe_targeting": {"profile": self.configuration.axolotl.moe_targeting,
                              "experts_targeted": self.configuration.axolotl.moe_targeting ==
                              "expert_targeting",
                              "warning": "Expert targeting materially changes VRAM requirements."},
            "future_output_lineage_schema": {
                "source_base_model": "required", "base_revision_hash": "required",
                "training_bundle_id": "required", "dataset_checksums": "required",
                "training_config_checksum": "required", "harness_commits": "required",
                "training_library_versions": "required", "result_hash": "required"},
        }

        governance = self._governance_report(episodes, origins, excluded, sft_build,
                                             pref_build, token_validation)

        summary = {"manifest": manifest, "readiness": readiness.to_dict(),
                   "governance": governance}

        if dry_run:
            return summary, None

        bundles = Path(output_root).resolve() / "bundles"
        destination = bundles / bundle_id

        # The id is the fingerprint of the inputs, so an existing directory is
        # already this exact bundle: validate it and hand it back unchanged.

        if destination.exists():
            validate_training_bundle(destination, self.store)

            return json.loads((destination / "manifest.json").read_text()), destination

        bundles.mkdir(parents=True, exist_ok=True, mode=0o700)

        # Built under a temporary name and renamed at the end, so a bundle
        # directory never exists in a half-written state.

        temporary = Path(tempfile.mkdtemp(prefix=f".{bundle_id}.", dir=bundles))

        try:
            datasets = {"sft": list(zip(sft, sft_records)),
                        "unpaired": [(item, item.trainer_record()) for item in unpaired],
                        "paired": [(item, item.trainer_record()) for item in paired]}

            for product, items in datasets.items():
                for split in ("train", "validation", "test"):
                    content = b"".join(_canonical_bytes(record) + b"\n"
                                       for sample, record in items if sample.split == split)
                    relative = f"datasets/{product}/{split}.jsonl"
                    _write(temporary / relative, content)
                    manifest["dataset_checksums"][relative] = hashlib.sha256(content).hexdigest()

            generator = AxolotlConfigGenerator(self.configuration.model,
                                                self.configuration.axolotl)

            # A config is only emitted for a stage the readiness evaluator
            # cleared: an absent config is how a not-ready stage stays unrunnable.

            configs = {}

            if readiness.sft.level != ReadinessLevel.NOT_READY.value:
                configs["configs/axolotl-sft.yml"] = generator.sft(
                    any(item.split == "validation" for item in sft),
                )

            if readiness.unpaired_preference.level != ReadinessLevel.NOT_READY.value:
                configs["configs/axolotl-kto.yml"] = generator.kto()

            if readiness.paired_preference.level != ReadinessLevel.NOT_READY.value:
                configs["configs/axolotl-dpo.yml"] = generator.dpo()

            for relative, content in configs.items():
                encoded = content.encode()

                _write(temporary / relative, encoded)
                manifest["config_checksums"][relative] = hashlib.sha256(encoded).hexdigest()

            inactive = {
                "kto": readiness.unpaired_preference.level == ReadinessLevel.NOT_READY.value,
                "dpo": readiness.paired_preference.level == ReadinessLevel.NOT_READY.value,
            }
            _write(temporary / "configs/preference-readiness.json",
                   _canonical_bytes(inactive) + b"\n")

            _write(temporary / "readiness.json", _canonical_bytes(readiness.to_dict()) + b"\n")
            _write(temporary / "readiness.md", readiness.to_markdown().encode())
            _write(temporary / "governance.json", _canonical_bytes(governance) + b"\n")
            _write(temporary / "evaluation-plan.json",
                   _canonical_bytes(self._evaluation_plan(snapshot)) + b"\n")
            _write(temporary / "RUNBOOK.md", self._runbook(
                configs, token_validation, self.configuration.smoke_test_only,
            ).encode())

            if self.configuration.hardware_report:
                _write(temporary / "hardware.json",
                       _canonical_bytes(asdict(self.configuration.hardware_report)) + b"\n")

            _write(temporary / "manifest.json", _canonical_bytes(manifest) + b"\n")

            # Written last and covering every other file, which is why it has
            # to exclude itself from the walk.

            checksum_rows = []

            for path in sorted(item for item in temporary.rglob("*") if item.is_file()
                               and item.name != "checksums.sha256"):
                relative = path.relative_to(temporary).as_posix()
                checksum_rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}\n")

            _write(temporary / "checksums.sha256", "".join(checksum_rows).encode())

            os.replace(temporary, destination)
        finally:
            # Gone on the success path, where the rename consumed it.

            if temporary.exists():
                shutil.rmtree(temporary)

        validate_training_bundle(destination, self.store)

        return manifest, destination

    def _governance_report(self, episodes, origins, excluded, sft, pref, token_validation):
        allowed = set(episodes)

        # `samples` is what the bundle ships; `all_samples` includes the ones
        # held back for review, which the quarantine counts below are about.

        sft_samples = [item for item in sft.samples if item.episode_id in allowed]
        all_sft = [item for item in sft.all_samples if item.episode_id in allowed]
        all_pairs = [item for item in pref.all_paired
                     if item.chosen_episode_id in allowed
                     and item.rejected_episode_id in allowed]

        projects = Counter(str(item.provenance.get("project") or "unknown")
                           for item in episodes.values())
        provenance = Counter(value for item in sft_samples for value in item.provenance_classes)

        return {
            "policy": self.configuration.governance.to_dict(),
            "source_projects_corpora": dict(sorted(projects.items())),
            "origin_distribution": dict(sorted(origins.items())),
            "provenance_categories": dict(sorted(provenance.items())),
            "exclusions": dict(sorted(excluded.items())),
            "external_content_exclusions": sum(
                "external_web" in turn.provenance
                for episode in episodes.values() for turn in episode.turns),
            "redaction_counts": sum(int(episode.training_metadata.get("redaction", {}).get(
                "fields_redacted", 0)) for episode in episodes.values()),
            "quarantine_counts": sum(item.review_status == "NEEDS_REVIEW"
                                     for item in all_sft),
            "manual_review_counts": sum(item.review_status == "NEEDS_REVIEW"
                                        for item in all_pairs),
            "memory_dependency_count": sum(bool(item.memory_provenance_ids)
                                           for item in all_sft),
            "source_harness_commits": sorted({item.harness_commit for item in all_sft
                                               if item.harness_commit}),
            "license_origin_status": "preserved_for_human_review_no_legal_inference",
            "token_validation": token_validation,
        }

    def _evaluation_plan(self, snapshot):
        return {"schema_version": 1, "training_split_group_ids": sorted({
                    value["split_group_id"] for value in snapshot.sft_samples.values()}),
                "permanent_holdout_group_ids": list(
                    self.configuration.governance.permanent_holdout_groups),
                "held_out_only": True,
                "required_control_suite": ["simple_coding", "repository_navigation",
                                           "tool_calling", "no_change_informational",
                                           "general_error_recovery"],
                "future_comparison": ["original_model", "fine_tuned_model"],
                "metrics": ["deterministic_task_success", "tool_selection_quality",
                            "recovery_success", "verification_correctness",
                            "context_tool_cost", "general_control_regression"]}

    @staticmethod
    def _runbook(configs, token_validation, smoke_test_only=False):
        has_sft = "configs/axolotl-sft.yml" in configs
        lines = ["# Training bundle runbook", "",
                 "This bundle does not execute training. Run these only on an approved training host.", "",
                 "1. Validate `checksums.sha256` and run `python -m training validate-bundle ...`.",
                 "2. Inspect governance, readiness, and held-out evaluation plan.",
                 "3. Verify masking with Axolotl preprocessing debug output.",
                 "4. Measure, do not assume, that the card holds this config: the "
                 "readiness verdict is a table, the preflight is a measurement.", "",
                 "```sh",
                 *(["axolotl preprocess configs/axolotl-sft.yml --debug"] if has_sft else
                   ["# No active training config: continue collecting/reviewing data."]),
                 *(["CONFIG_YAML=configs/axolotl-sft.yml bash "
                    "qwen3-finetune/cloud/load_preflight.sh"] if has_sft else []),
                 *(["axolotl train configs/axolotl-sft.yml"]
                   if has_sft and not smoke_test_only else []),
                 "```", "",
                 "After evaluation, a future phase may run `axolotl merge-lora`; do not merge before approval.",
                 "", "Tokenizer validation: " + json.dumps(token_validation, sort_keys=True)]

        if smoke_test_only:
            lines[2] = ("SMOKE TEST ONLY: synthetic/evaluation-origin data is training-prohibited; "
                        "validate preprocessing only and do not train.")

        return "\n".join(lines) + "\n"


def validate_training_bundle(bundle: str | os.PathLike[str],
                             store: TrainingStore | str | os.PathLike[str] | None = None
                             ) -> Mapping[str, object]:
    root = Path(bundle).resolve()

    required = ["manifest.json", "readiness.json", "readiness.md", "governance.json",
                "evaluation-plan.json", "RUNBOOK.md", "checksums.sha256"]

    if any(not (root / item).is_file() for item in required):
        raise TrainingBundleError("bundle omits required files")

    manifest = json.loads((root / "manifest.json").read_text())

    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise TrainingBundleError("unsupported bundle schema")

    SourceModelProfile(**manifest["source_model"]).validate()

    # Every file is re-hashed against the checksum list, and the path is
    # confined to the bundle so a crafted row cannot reach outside it.

    listed = {}

    for line in (root / "checksums.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = (root / relative).resolve()

        if root not in path.parents or not path.is_file():
            raise TrainingBundleError("checksum references invalid file")

        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise TrainingBundleError(f"bundle checksum mismatch: {relative}")

        listed[relative] = digest

    # The checksum file and the manifest are independent records of the same
    # bytes, so they are also checked against each other.

    for relative, expected in {**manifest["dataset_checksums"],
                               **manifest["config_checksums"]}.items():
        if listed.get(relative) != expected:
            raise TrainingBundleError(f"manifest checksum mismatch: {relative}")

    counts = Counter()
    group_splits = defaultdict(set)
    episode_splits = defaultdict(set)
    snapshot = manifest["frozen_snapshot"]

    for product, field_name in (("sft", "sft_samples"), ("unpaired", "unpaired_samples"),
                                ("paired", "paired_samples")):
        for split in ("train", "validation", "test"):
            path = root / "datasets" / product / f"{split}.jsonl"

            if not path.is_file():
                raise TrainingBundleError("bundle omits dataset split")

            for line in path.read_text().splitlines():
                record = json.loads(line)
                counts[product] += 1

                # Each product has its own record shape, but all three are held
                # to the same structural rules the exporters wrote them under.

                if product == "sft":
                    validate_loss_mask_intent(record)

                    messages = [{key: value for key, value in item.items()
                                 if key != "train"} for item in record["messages"]]
                    _validate_structure(messages, record.get("tools", ()), len(messages) - 1)
                elif product == "unpaired":
                    messages = [*record.get("prompt", ()), *record.get("completion", ())]
                    _validate_structure(messages, record.get("tools", ()), len(messages) - 1)

                    if not isinstance(record.get("label"), bool):
                        raise TrainingBundleError("unpaired label is not boolean")
                else:
                    for completion in (record.get("chosen", ()), record.get("rejected", ())):
                        messages = [*record.get("prompt", ()), *completion]
                        _validate_structure(messages, record.get("tools", ()), len(messages) - 1)

        # Which splits each group and each source episode appear in; anything
        # landing in more than one is leakage, checked once below.

        for value in snapshot[field_name].values():
            group_splits[value["split_group_id"]].add(value["split"])

            for key in ("episode_id", "chosen_episode_id", "rejected_episode_id"):
                if value.get(key):
                    episode_splits[value[key]].add(value["split"])

    # Empty products are absent from the manifest counts, so a mismatch is
    # re-checked per product to name the one that actually disagrees.

    if dict(counts) != {key: value for key, value in manifest["dataset_counts"].items() if value}:
        for key in manifest["dataset_counts"]:
            if counts[key] != manifest["dataset_counts"][key]:
                raise TrainingBundleError("dataset counts disagree with manifest")

    if any(len(value) > 1 for value in group_splits.values()):
        raise TrainingBundleError("split-group leakage across datasets")

    if any(len(value) > 1 for value in episode_splits.values()):
        raise TrainingBundleError("source episode leakage across datasets")

    # Evaluation and synthetic data may be preprocessed but never trained on,
    # so their presence is only tolerated in a bundle declared smoke-test-only.

    governance = json.loads((root / "governance.json").read_text())
    forbidden = {DataOrigin.BENCHMARK.value, DataOrigin.TEST_FIXTURE.value,
                 DataOrigin.SYNTHETIC.value}
    allowed = set(governance["policy"]["allowed_origins"])

    if forbidden & allowed and not manifest.get("smoke_test_only"):
        raise TrainingBundleError("evaluation/synthetic origin enabled in real bundle")

    # Given the originating store, the frozen samples are rebuilt from it and
    # compared: this is what proves the bundle still describes that store.

    if store is not None:
        source = store if isinstance(store, TrainingStore) else TrainingStore(store)

        for episode_id, checksum in snapshot["source_episode_checksums"].items():
            current = (source.episodes / f"{episode_id}.sha256").read_text().strip()

            if current != checksum:
                raise TrainingBundleError("source episode checksum mismatch")

        split_raw = snapshot["split_policy"]
        split = SplitConfiguration(int(split_raw["train"]),
                                   int(split_raw["validation"]),
                                   int(split_raw["test"]))

        sft_source = {item.sample_id: _hash(item.to_dict())
                      for item in SFTDatasetBuilder(
                          source, configuration=SFTExportConfiguration(split=split),
                      ).build().samples}

        preference = PreferenceDatasetBuilder(
            source, configuration=PreferenceConfiguration(split=split),
        ).build()

        unpaired_source = {item.sample_id: _hash(item.to_dict())
                           for item in preference.unpaired}
        paired_source = {item.preference_id: _hash(item.to_dict())
                         for item in preference.paired}

        for field, available in (("sft_samples", sft_source),
                                 ("unpaired_samples", unpaired_source),
                                 ("paired_samples", paired_source)):
            for sample_id, metadata in snapshot[field].items():
                if available.get(sample_id) != metadata["checksum"]:
                    raise TrainingBundleError("frozen source sample checksum mismatch")

    # Config presence must match the readiness verdict in both directions: a
    # not-ready stage carries no config, and a ready one must carry a usable
    # config that actually points at this bundle's masked data.

    readiness_value = json.loads((root / "readiness.json").read_text())
    sft_path = root / "configs/axolotl-sft.yml"

    if readiness_value["sft"]["level"] == ReadinessLevel.NOT_READY.value:
        if sft_path.exists():
            raise TrainingBundleError("SFT config active while stage is not ready")
    else:
        if not sft_path.is_file():
            raise TrainingBundleError("ready SFT stage omits config")

        sft_config = sft_path.read_text()

        if ("datasets/sft/train.jsonl" not in sft_config or
                "message_field_training: train" not in sft_config):
            raise TrainingBundleError("SFT config does not reference masked bundle data")

    for stage, config in (("unpaired", "configs/axolotl-kto.yml"),
                          ("paired", "configs/axolotl-dpo.yml")):
        ready = json.loads((root / "readiness.json").read_text())[
            "unpaired_preference" if stage == "unpaired" else "paired_preference"]

        if ready["level"] == ReadinessLevel.NOT_READY.value and (root / config).exists():
            raise TrainingBundleError("preference config active while stage is not ready")

    return manifest
