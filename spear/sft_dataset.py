"""Deterministic turn-level SFT curation from canonical training episodes.

This module deliberately depends on the provider-neutral FT0 records, not on a
trainer or model tokenizer.  A sample contains one selected assistant target;
earlier assistant messages are context only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Sequence

from training_data import (
    TrainingEligibility, TrainingEpisode, TrainingProvenance,
    TrainingRedactionPolicy, TrainingTurn,
)
from training_store import TrainingStore, TrainingStoreError
from training_splits import SplitConfiguration, split_group_for_episode


SFT_SCHEMA_VERSION = 1
SFT_SELECTION_POLICY_VERSION = 1
SFT_EXPORTER_VERSION = 1


class SFTDatasetError(RuntimeError):
    pass


class SFTSampleType(StrEnum):
    FINAL_RESPONSE = "final_response"
    TOOL_ACTION = "tool_action"
    RECOVERY_ACTION = "recovery_action"
    MUTATION_ACTION = "mutation_action"
    VERIFICATION_ACTION = "verification_action"


class ReviewStatus(StrEnum):
    AUTO_APPROVED = "AUTO_APPROVED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    REJECTED = "REJECTED"


class LengthStatus(StrEnum):
    EXPORTABLE = "EXPORTABLE"
    OVERLONG = "OVERLONG"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class ExportProfile(StrEnum):
    QWEN3_AXOLOTL = "qwen3_axolotl"
    FINAL_RESPONSE_ONLY = "final_response_only"
    AGENT_TOOL_USE = "agent_tool_use"


@dataclass(frozen=True)
class SFTExportConfiguration:
    profile: str = ExportProfile.QWEN3_AXOLOTL.value
    split: SplitConfiguration = field(default_factory=SplitConfiguration)
    allow_external_web: bool = False
    max_approximate_tokens: int = 32768
    project_filters: tuple[str, ...] = ()
    superseded_memory_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.profile not in {item.value for item in ExportProfile}:
            raise ValueError(f"unsupported SFT export profile: {self.profile}")

        if self.max_approximate_tokens <= 0:
            raise ValueError("max approximate tokens must be positive")


@dataclass(frozen=True)
class SFTSample:
    sample_id: str
    episode_id: str
    source_turn_id: str
    task_id: str
    split_group_id: str
    schema_version: int
    messages: tuple[Mapping[str, object], ...]
    tools: tuple[Mapping[str, object], ...]
    target: Mapping[str, object]
    target_message_index: int
    sample_type: str
    source_labels: tuple[str, ...]
    task_outcome: str
    verification_state: str
    mutation_generation: int
    tool_action_ids: tuple[str, ...]
    recovery_turn_ids: tuple[str, ...]
    provenance_classes: tuple[str, ...]
    memory_provenance_ids: tuple[str, ...]
    redaction_status: Mapping[str, object]
    quality_gates_passed: tuple[str, ...]
    review_status: str
    review_flags: tuple[str, ...]
    length_status: str
    character_count: int
    token_count: int
    token_count_kind: str
    message_count: int
    tool_schema_size: int
    source_episode_checksum: str
    harness_commit: str | None
    tool_view_hash: str
    system_prompt_hash: str
    fingerprint: str
    split: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "SFTSample":
        if not isinstance(raw, Mapping) or raw.get("schema_version") != SFT_SCHEMA_VERSION:
            raise SFTDatasetError("unsupported or malformed SFT sample")

        required = set(cls.__dataclass_fields__)

        if not required <= set(raw):
            raise SFTDatasetError("SFT sample omits required fields")

        values = dict(raw)

        for name in (
            "messages", "tools", "source_labels", "tool_action_ids",
            "recovery_turn_ids", "provenance_classes", "memory_provenance_ids",
            "quality_gates_passed", "review_flags",
        ):
            values[name] = tuple(values[name])

        try:
            sample = cls(**{name: values[name] for name in cls.__dataclass_fields__})
            _validate_structure(sample.messages, sample.tools, sample.target_message_index)

            if sample.target != sample.messages[sample.target_message_index]:
                raise SFTDatasetError("SFT target metadata disagrees with messages")

            return sample
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise SFTDatasetError(f"malformed SFT sample: {exc}") from exc

    def trainer_record(self) -> dict[str, object]:
        value: dict[str, object] = {"messages": [dict(item) for item in self.messages]}

        if self.tools:
            value["tools"] = [dict(item) for item in self.tools]

        return value


@dataclass
class SFTBuildResult:
    samples: list[SFTSample]
    all_samples: list[SFTSample]
    report: dict[str, object]
    source_checksums: dict[str, str]
    tool_schemas: dict[str, Mapping[str, object]]
    duplicate_sources: dict[str, list[Mapping[str, str]]]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _openai_tools(snapshot: Mapping[str, object] | None) -> tuple[Mapping[str, object], ...]:
    if not snapshot:
        return ()

    result = []

    for item in snapshot.get("tools", ()):
        if not isinstance(item, Mapping) or not item.get("name"):
            raise SFTDatasetError("malformed tool schema snapshot")

        result.append({"type": "function", "function": {
            "name": str(item["name"]),
            "description": str(item.get("description", "")),
            "parameters": dict(item.get("input_schema", {})),
        }})

    return tuple(result)


def _openai_context(turn: TrainingTurn) -> list[dict[str, object]]:
    """Materialize the exact captured context without consulting later history.

    What is rebuilt is what the model actually saw at that moment. Nothing that
    happened afterwards may leak in, or the sample would train on hindsight.
    """

    result: list[dict[str, object]] = [{"role": "system", "content": turn.system}]

    for message in turn.messages:
        role = message.get("role")
        blocks = message.get("content", ())

        if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
            raise SFTDatasetError("unsupported canonical message content")

        text = "".join(str(block.get("text", "")) for block in blocks
                       if isinstance(block, Mapping) and block.get("type") == "text")
        calls = [block for block in blocks if isinstance(block, Mapping)
                 and block.get("type") == "tool_use"]
        tool_results = [block for block in blocks if isinstance(block, Mapping)
                        and block.get("type") == "tool_result"]

        if role == "assistant":
            item: dict[str, object] = {"role": "assistant", "content": text}

            if calls:
                item["tool_calls"] = [{
                    "id": str(call.get("id", "")), "type": "function",
                    "function": {
                        "name": str(call.get("name", "")),
                        "arguments": json.dumps(call.get("arguments", {}),
                                                ensure_ascii=False, sort_keys=True,
                                                separators=(",", ":")),
                    },
                } for call in calls]

            result.append(item)
        elif role == "user" and tool_results:
            # The canonical form carries tool results inside a user message;
            # the OpenAI form gives each its own "tool" message.

            if text:
                result.append({"role": "user", "content": text})

            for block in tool_results:
                result.append({
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_call_id", "")),
                    "content": str(block.get("content", "")),
                })
        elif role == "user":
            result.append({"role": "user", "content": text})
        else:
            raise SFTDatasetError(f"unsupported canonical message role: {role!r}")

    return result


def _target_message(turn: TrainingTurn) -> dict[str, object]:
    target: dict[str, object] = {"role": "assistant", "content": turn.assistant_output}

    if turn.tool_calls:
        target["tool_calls"] = [{
            "id": call.tool_call_id, "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(call.arguments, ensure_ascii=False,
                                        sort_keys=True, separators=(",", ":")),
            },
        } for call in turn.tool_calls]

    return target


def _validate_structure(messages: Sequence[Mapping[str, object]],
                        tools: Sequence[Mapping[str, object]],
                        target_index: int, *,
                        allow_target_schema_mismatch: bool = False) -> None:
    """Check a sample is a well-formed conversation a trainer will accept.

    ``allow_target_schema_mismatch`` exists for one case: a turn recorded as
    invalid_arguments is a real observation worth keeping, and refusing it here
    would discard the very examples that show what a bad call looks like.
    """

    definitions = {str(item.get("function", {}).get("name")):
                   item.get("function", {}).get("parameters", {}) for item in tools}
    available = set(definitions)

    # Call ids seen so far, so every tool result can be matched to the call it
    # answers -- and no call is answered twice.

    calls: dict[str, str] = {}

    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            for call in message.get("tool_calls", ()):
                call_id = str(call.get("id", ""))
                function = call.get("function", {})
                name = str(function.get("name", ""))

                if not call_id or call_id in calls or not name:
                    raise SFTDatasetError("malformed or duplicate tool call")

                # The model cannot have called a tool it was never offered, so
                # this means the captured tool view does not match the turn.

                if name not in available:
                    raise SFTDatasetError("tool call was absent from captured tool view")

                arguments = function.get("arguments")

                try:
                    parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError as exc:
                    raise SFTDatasetError("tool arguments are invalid JSON") from exc

                if not isinstance(parsed, Mapping):
                    raise SFTDatasetError("tool arguments must be a JSON object")

                if not (allow_target_schema_mismatch and index == target_index):
                    _validate_arguments(parsed, definitions[name])

                calls[call_id] = name
        elif message.get("role") == "tool":
            call_id = str(message.get("tool_call_id", ""))

            if call_id not in calls:
                raise SFTDatasetError("tool result has no preceding tool call")
        elif message.get("role") not in {"system", "user"}:
            raise SFTDatasetError("unsupported OpenAI message role")

    # One target, and it is the last message: everything before it is context,
    # which is what the loss mask in the bundle depends on.

    if target_index != len(messages) - 1 or messages[target_index].get("role") != "assistant":
        raise SFTDatasetError("sample must end in exactly one selected assistant target")


def _validate_arguments(arguments: Mapping[str, object], schema: object) -> None:
    """Validate the useful dependency-free subset of JSON Schema."""

    if not isinstance(schema, Mapping):
        return

    required = schema.get("required", ())

    if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
        missing = [str(name) for name in required if str(name) not in arguments]

        if missing:
            raise SFTDatasetError(f"tool arguments omit required field: {missing[0]}")

    properties = schema.get("properties", {})

    if not isinstance(properties, Mapping):
        return

    expected_types = {
        "string": str, "object": Mapping, "array": (list, tuple),
        "integer": int, "number": (int, float), "boolean": bool,
    }

    for name, value in arguments.items():
        definition = properties.get(name)

        if not isinstance(definition, Mapping) or definition.get("type") not in expected_types:
            continue

        expected = expected_types[str(definition["type"])]

        if not isinstance(value, expected) or (definition["type"] in {"integer", "number"}
                                               and isinstance(value, bool)):
            raise SFTDatasetError(f"tool argument has wrong type: {name}")


class SFTSelectionPolicy:
    """Conservative, grounded target selection.  Failed history is never a target."""

    VERSION = SFT_SELECTION_POLICY_VERSION
    BAD_LABELS = frozenset({
        "failed_tool_action", "repeated_action", "stalled", "invalid_model_turn",
        "failed_model_call", "policy_failure", "security_failure",
        "stale_verification", "budget_exhausted", "ungrounded_completion_claim",
        "cancelled", "cancelled_action",
    })
    USEFUL_LABELS = frozenset({
        "new_discovery", "useful_read", "useful_search", "grounded_mutation",
        "verification_action", "meaningful_verification", "recovery_after_failure",
        "plan_progress", "meaningful_progress",
    })
    READ_SEARCH_NAMES = frozenset({
        "read_file", "read_files", "search_files", "search_code", "find_files",
        "list_files", "grep", "rg", "retrieve", "memory_search", "search_corpus",
    })
    REASON_NAMES = {
        "failed_tool_action": "failed_action",
        "policy_failure": "policy_security_failure",
        "security_failure": "policy_security_failure",
        "cancelled_action": "cancelled",
    }

    def select(self, episode: TrainingEpisode, turn: TrainingTurn,
               *, allow_external_web: bool = False) -> tuple[SFTSampleType | None, tuple[str, ...]]:
        """Whether this turn is worth imitating, and if not, every reason why.

        The gates below run first and are absolute. Only a turn that clears all
        of them is then classified by what it accomplished.
        """

        reasons: list[str] = []

        if episode.training_metadata.get("eligibility") != TrainingEligibility.POSITIVE_CANDIDATE.value:
            reasons.append("episode_not_positive")

        if not turn.primary_candidate or turn.role.lower() in {"explorer", "reviewer"}:
            reasons.append("auxiliary_turn")

        labels = set(turn.labels)
        bad = sorted({self.REASON_NAMES.get(label, label)
                      for label in labels & self.BAD_LABELS})
        reasons.extend(bad)

        if turn.error or turn.stop_reason in {"error", "max_tokens", "refusal"}:
            reasons.append("invalid_model_turn")

        if (TrainingProvenance.EXTERNAL_WEB.value in turn.provenance
                and not allow_external_web):
            reasons.append("external_web_dependency")

        if reasons:
            return None, tuple(dict.fromkeys(reasons))

        if turn.tool_calls:
            if any(call.status not in {"ok", "success"} for call in turn.tool_calls):
                return None, ("failed_action",)

            # A recovery is only a good example if it actually got somewhere:
            # either it made progress, or it was purely looking, which is the
            # reasonable thing to do straight after a failure.

            if "recovery_after_failure" in labels and turn.recovers_turn_ids:
                recovery_progress = ((labels & (self.USEFUL_LABELS - {
                    "recovery_after_failure",
                })) or {call.name for call in turn.tool_calls} <= self.READ_SEARCH_NAMES)

                if recovery_progress:
                    return SFTSampleType.RECOVERY_ACTION, ()

                return None, ("insufficient_recovery_progress",)

            if "grounded_mutation" in labels:
                return SFTSampleType.MUTATION_ACTION, ()

            if "verification_action" in labels or "meaningful_verification" in labels:
                if (turn.verification_generation is not None
                        and turn.verification_generation != turn.mutation_generation_after):
                    return None, ("stale_verification",)

                return SFTSampleType.VERIFICATION_ACTION, ()

            # Reading and searching are always defensible; anything else has
            # to have been labelled useful by something that observed it.

            names = {call.name for call in turn.tool_calls}

            if labels & self.USEFUL_LABELS or names <= self.READ_SEARCH_NAMES:
                return SFTSampleType.TOOL_ACTION, ()

            return None, ("insufficient_progress_evidence",)

        is_final = ("final_success" in labels or (
            turn.assistant_output.strip()
            and turn.assistant_output == episode.outcome.get("final_response")
        ))

        # A final answer is only exemplary if the task it concluded actually
        # succeeded, which is what the four checks below establish.

        if is_final and turn.assistant_output.strip():
            outcome = episode.outcome

            if outcome.get("terminal_task_status") != "completed":
                return None, ("task_not_completed",)

            if outcome.get("unresolved_failures"):
                return None, ("unresolved_failure",)

            mutated = int(episode.execution.get("mutation_generation", 0) or 0) > 0

            if mutated and outcome.get("verification_status") != "verified":
                return None, ("unverified",)

            if mutated and outcome.get("checkpoint_status") != "finalized":
                return None, ("checkpoint_not_finalized",)

            return SFTSampleType.FINAL_RESPONSE, ()

        return None, ("not_a_selected_target_type",)


class SFTDatasetBuilder:
    def __init__(self, store: TrainingStore | str | os.PathLike[str], *,
                 configuration: SFTExportConfiguration | None = None,
                 redaction_policy: TrainingRedactionPolicy | None = None,
                 selection_policy: SFTSelectionPolicy | None = None) -> None:
        self.store = store if isinstance(store, TrainingStore) else TrainingStore(store)
        self.configuration = configuration or SFTExportConfiguration()
        self.redaction_policy = redaction_policy or TrainingRedactionPolicy.from_environment()
        self.selection_policy = selection_policy or SFTSelectionPolicy()

    def _source_checksum(self, episode_id: str) -> str:
        path = self.store.episodes / f"{episode_id}.sha256"

        try:
            value = path.read_text(encoding="ascii").strip()
        except OSError as exc:
            raise SFTDatasetError(f"missing source checksum for {episode_id}") from exc

        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise SFTDatasetError(f"malformed source checksum for {episode_id}")

        return value

    def _profile_allows(self, sample_type: SFTSampleType) -> bool:
        """Whether the configured profile wants this kind of sample at all."""

        profile = self.configuration.profile

        if profile == ExportProfile.FINAL_RESPONSE_ONLY.value:
            return sample_type == SFTSampleType.FINAL_RESPONSE

        if profile == ExportProfile.AGENT_TOOL_USE.value:
            return sample_type != SFTSampleType.FINAL_RESPONSE

        return True

    def _make_sample(self, episode: TrainingEpisode, turn: TrainingTurn,
                     sample_type: SFTSampleType, checksum: str) -> SFTSample:
        snapshot = episode.tool_views.get(turn.tool_view_hash)

        if snapshot is None:
            raise SFTDatasetError("selected turn references a missing tool view")

        has_tool_context = bool(turn.tool_calls) or any(
            isinstance(block, Mapping) and block.get("type") in {"tool_use", "tool_result"}
            for message in turn.messages for block in message.get("content", ())
        )
        tools = _openai_tools(snapshot) if turn.tools_enabled or has_tool_context else ()
        context = _openai_context(turn)
        target = _target_message(turn)

        # Kept for comparison: redacting the CONTEXT is harmless, but redacting
        # the TARGET changes what would be trained on, and is flagged below.

        unredacted_target = _canonical_bytes(target)

        payload, redaction = self.redaction_policy.redact({
            "messages": [*context, target], "tools": list(tools),
        })
        messages = tuple(payload["messages"])
        tools = tuple(payload["tools"])
        target = messages[-1]
        target_changed = _canonical_bytes(target) != unredacted_target
        _validate_structure(messages, tools, len(messages) - 1)

        trainer = {"messages": messages, "tools": tools}
        chars = len(_canonical_bytes(trainer).decode("utf-8"))

        # Dependency-free, explicitly approximate heuristic.  The estimate is
        # deliberately conservative for code/JSON-heavy tool trajectories.

        approx_tokens = max(1, math.ceil(chars / 3.5))
        length_status = (LengthStatus.OVERLONG if approx_tokens >
                         self.configuration.max_approximate_tokens else
                         LengthStatus.EXPORTABLE)
        flags: list[str] = []
        review = ReviewStatus.AUTO_APPROVED

        if target_changed:
            flags.append("target_redacted")
            review = ReviewStatus.NEEDS_REVIEW
        elif int(redaction.get("redaction_count", 0)):
            flags.append("input_redacted")

        superseded = sorted(set(turn.memory_provenance_ids) &
                            set(self.configuration.superseded_memory_ids))

        if superseded:
            flags.append("superseded_memory")
            review = ReviewStatus.NEEDS_REVIEW

        if length_status == LengthStatus.OVERLONG:
            flags.append("overlong")
            review = ReviewStatus.NEEDS_REVIEW

        # The fingerprint covers only what a trainer would see, so two turns
        # producing the same example deduplicate against each other. The sample
        # id additionally names its source, so provenance survives.

        fingerprint = _hash(trainer)
        split_group = split_group_for_episode(episode)

        identity = {
            "schema": SFT_SCHEMA_VERSION, "episode": episode.episode_id,
            "turn": turn.turn_id, "fingerprint": fingerprint,
            "policy": self.selection_policy.VERSION,
        }
        sample_id = "sft_" + _hash(identity)[:32]
        provenance = episode.provenance

        return SFTSample(
            sample_id=sample_id, episode_id=episode.episode_id,
            source_turn_id=turn.turn_id, task_id=episode.task_id,
            split_group_id=split_group, schema_version=SFT_SCHEMA_VERSION,
            messages=messages, tools=tools, target=target,
            target_message_index=len(messages) - 1, sample_type=sample_type.value,
            source_labels=tuple(turn.labels),
            task_outcome=str(episode.outcome.get("terminal_task_status", "")),
            verification_state=str(episode.outcome.get("verification_status", "")),
            mutation_generation=turn.mutation_generation_after,
            tool_action_ids=tuple(call.action_id for call in turn.tool_calls if call.action_id),
            recovery_turn_ids=tuple(turn.recovers_turn_ids),
            provenance_classes=tuple(turn.provenance),
            memory_provenance_ids=tuple(turn.memory_provenance_ids),
            redaction_status={**dict(redaction), "target_changed": target_changed},
            quality_gates_passed=tuple((
                "positive_episode", "primary_turn", "structure_valid",
                "external_policy", "export_redaction_applied",
                *(("target_unmodified",) if not target_changed else ()),
            )),
            review_status=review.value, review_flags=tuple(flags),
            length_status=length_status.value, character_count=chars,
            token_count=approx_tokens, token_count_kind="approximate",
            message_count=len(messages),
            tool_schema_size=len(_canonical_bytes(tools)) if tools else 0,
            source_episode_checksum=checksum,
            harness_commit=(str(provenance.get("harness_git_commit"))
                            if provenance.get("harness_git_commit") else None),
            tool_view_hash=turn.tool_view_hash,
            system_prompt_hash=str(provenance.get("system_prompt_hash", "")),
            fingerprint=fingerprint, split=self.configuration.split.assign(split_group),
        )

    def build(self) -> SFTBuildResult:
        """Curate every exportable sample the stored episodes support."""

        candidates: list[SFTSample] = []
        source_checksums: dict[str, str] = {}
        tool_schemas: dict[str, Mapping[str, object]] = {}
        episode_counts = Counter({key: 0 for key in (
            "finalized", "positive", "failure", "incomplete", "excluded")})
        episode_counts["incomplete"] = sum(1 for _ in self.store.drafts.glob("*.json"))
        rejection_counts: Counter[str] = Counter()
        considered = 0
        represented_projects: Counter[str] = Counter()

        metadata = sorted(self.store.iterate_metadata(), key=lambda item: str(item.get("episode_id")))

        for item in metadata:
            episode_id = str(item.get("episode_id", ""))

            try:
                episode = self.store.load_episode(episode_id)
            except TrainingStoreError as exc:
                raise SFTDatasetError(f"corrupt source episode {episode_id}: {exc}") from exc

            checksum = self._source_checksum(episode_id)

            if checksum != str(item.get("checksum", "")):
                raise SFTDatasetError(f"source manifest checksum mismatch: {episode_id}")

            source_checksums[episode_id] = checksum
            episode_counts["finalized"] += 1
            eligibility = str(episode.training_metadata.get("eligibility", ""))
            eligibility_key = {
                "positive_candidate": "positive", "failure_trajectory": "failure",
                "incomplete": "incomplete", "excluded": "excluded",
            }.get(eligibility)

            if eligibility_key:
                episode_counts[eligibility_key] += 1

            project = str(episode.provenance.get("project") or "unknown")
            considered += len(episode.turns)

            if self.configuration.project_filters and project not in self.configuration.project_filters:
                rejection_counts["project_filtered"] += len(episode.turns)
                continue

            represented_projects[project] += 1

            for view_hash, snapshot in episode.tool_views.items():
                if _hash(snapshot) != str(view_hash):
                    raise SFTDatasetError("captured tool-view hash mismatch")

                # Two different schema sets under one hash would make every
                # sample referencing it ambiguous.

                existing = tool_schemas.setdefault(str(view_hash), snapshot)

                if _hash(existing) != _hash(snapshot):
                    raise SFTDatasetError("tool-view hash collision")

            for turn in episode.turns:
                sample_type, reasons = self.selection_policy.select(
                    episode, turn, allow_external_web=self.configuration.allow_external_web,
                )

                if sample_type is None:
                    rejection_counts.update(reasons or ("policy_rejected",))
                    continue

                if not self._profile_allows(sample_type):
                    rejection_counts["profile_filtered"] += 1
                    continue

                try:
                    candidates.append(self._make_sample(episode, turn, sample_type, checksum))
                except SFTDatasetError:
                    rejection_counts["malformed"] += 1
                    continue

        # Sorted by id so the survivor of a duplicate group is always the same
        # one, whatever order the store was walked in.

        unique: list[SFTSample] = []
        by_fingerprint: dict[str, SFTSample] = {}
        duplicate_sources: dict[str, list[Mapping[str, str]]] = defaultdict(list)

        for sample in sorted(candidates, key=lambda item: item.sample_id):
            existing = by_fingerprint.get(sample.fingerprint)

            if existing is None:
                by_fingerprint[sample.fingerprint] = sample
                unique.append(sample)
            else:
                rejection_counts["duplicate"] += 1
                duplicate_sources[existing.sample_id].append({
                    "episode_id": sample.episode_id,
                    "turn_id": sample.source_turn_id,
                })

        self._check_leakage(unique)

        # `approved` is what ships; `unique` is kept whole so the report can
        # account for everything that was held back and why.

        approved = [item for item in unique
                    if item.review_status == ReviewStatus.AUTO_APPROVED.value
                    and item.length_status == LengthStatus.EXPORTABLE.value]

        for sample in unique:
            if "target_redacted" in sample.review_flags:
                rejection_counts["secret_redaction_quarantine"] += 1

            if "overlong" in sample.review_flags:
                rejection_counts["overlong"] += 1

            if "superseded_memory" in sample.review_flags:
                rejection_counts["superseded_memory"] += 1

        report = self._quality_report(
            episode_counts, considered, candidates, unique, approved,
            rejection_counts, represented_projects,
        )

        return SFTBuildResult(approved, unique, report, source_checksums,
                              tool_schemas, dict(duplicate_sources))

    @staticmethod
    def _check_leakage(samples: Sequence[SFTSample]) -> None:
        episode_splits: dict[str, set[str]] = defaultdict(set)
        group_splits: dict[str, set[str]] = defaultdict(set)
        fingerprint_splits: dict[str, set[str]] = defaultdict(set)

        for sample in samples:
            episode_splits[sample.episode_id].add(sample.split)
            group_splits[sample.split_group_id].add(sample.split)
            fingerprint_splits[sample.fingerprint].add(sample.split)

        if any(len(value) > 1 for value in episode_splits.values()):
            raise SFTDatasetError("episode leakage across splits")

        if any(len(value) > 1 for value in group_splits.values()):
            raise SFTDatasetError("split-group leakage across splits")

        if any(len(value) > 1 for value in fingerprint_splits.values()):
            raise SFTDatasetError("exact duplicate leakage across splits")

    @staticmethod
    def _quality_report(episodes: Counter, considered: int,
                        candidates: Sequence[SFTSample], unique: Sequence[SFTSample],
                        approved: Sequence[SFTSample], exclusions: Counter,
                        projects: Counter) -> dict[str, object]:
        review = Counter(item.review_status for item in unique)
        types = Counter(item.sample_type for item in approved)
        splits = Counter(item.split for item in approved)
        tools = Counter(call["function"]["name"] for item in approved
                        for message in item.messages if message.get("role") == "assistant"
                        for call in message.get("tool_calls", ()))
        lengths = sorted(item.token_count for item in unique)

        def percentile(p: float) -> int:
            if not lengths:
                return 0

            return lengths[min(len(lengths) - 1, int((len(lengths) - 1) * p))]

        return {
            "episodes": dict(episodes),
            "turn_candidates": {
                "considered": considered, "selected_before_dedup": len(candidates),
                "selected": len(approved), "rejected": considered - len(approved),
            },
            "selected_sample_types": dict(sorted(types.items())),
            "exclusion_reasons": dict(sorted(exclusions.items())),
            "review_status": dict(sorted(review.items())),
            "distribution": {
                "projects_corpora": dict(sorted(projects.items())),
                "tool_names": dict(sorted(tools.items())),
                "target_types": dict(sorted(types.items())),
                "split_sizes": dict(sorted(splits.items())),
                "context_lengths": {
                    "count": len(lengths), "minimum": min(lengths, default=0),
                    "median": percentile(.5), "p95": percentile(.95),
                    "maximum": max(lengths, default=0), "kind": "approximate_tokens",
                },
            },
            "small_sample_warning": (len(approved) < 100),
        }

    def persist_episode_candidates(self, episode: TrainingEpisode) -> list[Path]:
        """Cheap append-safe post-finalization derivation; never materializes JSONL."""
        checksum = self._source_checksum(episode.episode_id)
        root = self.store.root / "datasets" / "sft"
        sample_dir = root / "samples"
        schema_dir = root / "schemas"
        manifest_dir = root / "manifests"

        for directory in (sample_dir, schema_dir, manifest_dir):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)

        written: list[Path] = []
        manifest_rows = []

        for view_hash, snapshot in episode.tool_views.items():
            path = schema_dir / f"{view_hash}.json"
            self._immutable_write(path, _canonical_bytes(snapshot) + b"\n")

        for turn in episode.turns:
            sample_type, _ = self.selection_policy.select(
                episode, turn, allow_external_web=self.configuration.allow_external_web,
            )

            if sample_type is None:
                continue

            sample = self._make_sample(episode, turn, sample_type, checksum)
            path = sample_dir / f"{sample.sample_id}.json"
            self._immutable_write(path, _canonical_bytes(sample.to_dict()) + b"\n")
            written.append(path)
            manifest_rows.append({"sample_id": sample.sample_id,
                                  "checksum": _hash(sample.to_dict())})

        manifest = manifest_dir / f"{episode.episode_id}.json"
        self._immutable_write(manifest, _canonical_bytes({
            "schema_version": SFT_SCHEMA_VERSION, "episode_id": episode.episode_id,
            "source_checksum": checksum, "samples": manifest_rows,
        }) + b"\n")

        return written

    @staticmethod
    def _immutable_write(path: Path, content: bytes) -> None:
        if path.exists():
            if path.read_bytes() != content:
                raise SFTDatasetError(f"immutable SFT artifact conflict: {path.name}")

            return

        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)

        try:
            os.write(descriptor, content)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def materialize(self, output_directory: str | os.PathLike[str], *,
                    dry_run: bool = False, report_only: bool = False) -> tuple[SFTBuildResult, Path | None]:
        result = self.build()

        if dry_run or report_only:
            return result, None

        output_root = Path(output_directory).resolve()
        logical = {
            "sources": result.source_checksums,
            "profile": self.configuration.profile,
            "split": asdict(self.configuration.split),
            "policy": self.selection_policy.VERSION,
            "external": self.configuration.allow_external_web,
            "max_tokens": self.configuration.max_approximate_tokens,
            "projects": self.configuration.project_filters,
        }
        dataset_id = "sft_" + _hash(logical)[:20]
        self._persist_build_artifacts(result, dataset_id)
        dataset_dir = output_root / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        checksums: dict[str, str] = {}

        for split in ("train", "validation", "test"):
            rows = [sample.trainer_record() for sample in result.samples if sample.split == split]
            content = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
            path = dataset_dir / f"{split}.jsonl"
            self._write_deterministic(path, content)
            checksums[path.name] = hashlib.sha256(content).hexdigest()

        side_rows = []

        for sample in sorted(result.all_samples, key=lambda item: item.sample_id):
            rich = sample.to_dict()

            for field_name in ("messages", "tools", "target"):
                rich.pop(field_name, None)

            rich["duplicate_sources"] = result.duplicate_sources.get(sample.sample_id, [])
            side_rows.append(rich)

        side_content = b"".join(_canonical_bytes(row) + b"\n" for row in side_rows)
        side_path = dataset_dir / "sample-manifest.jsonl"
        self._write_deterministic(side_path, side_content)
        checksums[side_path.name] = hashlib.sha256(side_content).hexdigest()
        source_timestamps = []

        for item in self.store.iterate_metadata():
            if item.get("episode_id") in result.source_checksums:
                source_timestamps.append(str(item.get("timestamp", "")))

        created = max(source_timestamps, default="1970-01-01T00:00:00+00:00")
        manifest = {
            "dataset_id": dataset_id, "schema_version": SFT_SCHEMA_VERSION,
            "creation_timestamp": created,
            "source_training_store": str(self.store.root),
            "source_episode_count": len(result.source_checksums),
            "source_harness_commits": sorted({item.harness_commit for item in result.all_samples
                                               if item.harness_commit}),
            "selection_policy_version": self.selection_policy.VERSION,
            "exporter_profile": self.configuration.profile,
            "exporter_version": SFT_EXPORTER_VERSION,
            "redaction_policy_version": 1,
            "split_policy": {"method": "sha256_split_group", **asdict(self.configuration.split)},
            "sample_counts": dict(Counter(item.split for item in result.samples)),
            "sample_type_counts": dict(Counter(item.sample_type for item in result.samples)),
            "exclusion_counts": result.report["exclusion_reasons"],
            "duplicate_counts": {"exact": sum(len(value) for value in result.duplicate_sources.values())},
            "provenance_distribution": dict(Counter(value for item in result.samples
                                                      for value in item.provenance_classes)),
            "tool_view_distribution": dict(Counter(item.tool_view_hash for item in result.samples)),
            "length_statistics": result.report["distribution"]["context_lengths"],
            "output_checksums": checksums,
            "source_episode_checksums": result.source_checksums,
            "source_sample_checksums": {item.sample_id: _hash(item.to_dict())
                                        for item in result.all_samples},
            "tool_schema_registry": sorted(result.tool_schemas),
            "small_sample_warning": result.report["small_sample_warning"],
        }
        self._write_deterministic(dataset_dir / "dataset-manifest.json",
                                  _canonical_bytes(manifest) + b"\n")
        self._write_deterministic(dataset_dir / "quality-report.json",
                                  _canonical_bytes(result.report) + b"\n")
        self._write_deterministic(dataset_dir / "quality-report.md",
                                  self._human_report(result.report).encode("utf-8"))

        return result, dataset_dir

    def _persist_build_artifacts(self, result: SFTBuildResult, dataset_id: str) -> None:
        root = self.store.root / "datasets" / "sft"
        sample_dir = root / "samples"
        schema_dir = root / "schemas"
        manifest_dir = root / "manifests"

        for directory in (sample_dir, schema_dir, manifest_dir):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)

        for view_hash, snapshot in result.tool_schemas.items():
            self._immutable_write(schema_dir / f"{view_hash}.json",
                                  _canonical_bytes(snapshot) + b"\n")

        for sample in result.all_samples:
            self._immutable_write(sample_dir / f"{sample.sample_id}.json",
                                  _canonical_bytes(sample.to_dict()) + b"\n")

        manifest = {
            "schema_version": SFT_SCHEMA_VERSION, "dataset_id": dataset_id,
            "source_episode_checksums": result.source_checksums,
            "sample_ids": sorted(item.sample_id for item in result.all_samples),
            "duplicate_sources": result.duplicate_sources,
        }
        self._immutable_write(manifest_dir / f"{dataset_id}.json",
                              _canonical_bytes(manifest) + b"\n")

    @staticmethod
    def _write_deterministic(path: Path, content: bytes) -> None:
        if path.exists() and path.read_bytes() == content:
            return

        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)

        try:
            os.write(descriptor, content)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

        os.replace(temporary, path)

    @staticmethod
    def _human_report(report: Mapping[str, object]) -> str:
        episodes = report["episodes"]
        turns = report["turn_candidates"]
        lines = [
            "# SFT quality report", "",
            f"Finalized episodes: {episodes.get('finalized', 0)}",
            f"Positive episodes: {episodes.get('positive', 0)}",
            f"Failure trajectories: {episodes.get('failure', 0)}",
            f"Incomplete episodes: {episodes.get('incomplete', 0)}",
            f"Excluded episodes: {episodes.get('excluded', 0)}", "",
            f"Turns considered: {turns.get('considered', 0)}",
            f"Samples exported: {turns.get('selected', 0)}", "",
            "## Exclusions", "",
        ]
        lines.extend(f"- {key}: {value}" for key, value in
                     report.get("exclusion_reasons", {}).items())

        if report.get("small_sample_warning"):
            lines.extend(["", "The dataset is too small for statistically useful split claims."])

        return "\n".join(lines) + "\n"
