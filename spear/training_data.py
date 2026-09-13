"""Canonical, provider-neutral training trajectory capture.

The records in this module are evidence for later dataset construction.  They
are deliberately not trainer-specific examples and never contain private
provider reasoning or wire payloads.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

from context_engine import ContextLayer
from model_backend import (
    ConversationMessage, ModelTurn, TextBlock, ToolDefinition, ToolResultBlock,
    ToolUseBlock,
)
from tool_router import ToolResultEnvelope, ToolResultStatus
from verification import CompletionVerificationStatus
from working_state import ActionStatus, TerminalStatus, WorkingState


TRAINING_SCHEMA_VERSION = 1


class TrainingDataError(ValueError):
    pass


class TrainingEligibility(StrEnum):
    POSITIVE_CANDIDATE = "positive_candidate"
    FAILURE_TRAJECTORY = "failure_trajectory"
    INCOMPLETE = "incomplete"
    EXCLUDED = "excluded"


class TrainingProvenance(StrEnum):
    LOCAL_USER = "local_user"
    LOCAL_WORKSPACE = "local_workspace"
    DURABLE_MEMORY = "durable_memory"
    GENERATED = "generated"
    EXTERNAL_WEB = "external_web"
    TOOL_OUTPUT = "tool_output"
    LICENSED_STANDARD = "licensed_standard"


@dataclass(frozen=True)
class TrainingToolCall:
    tool_call_id: str
    name: str
    arguments: Mapping[str, object]
    action_id: str | None = None
    status: str | None = None
    model_content: str | None = None
    result_reference: str | None = None
    provenance: str = TrainingProvenance.TOOL_OUTPUT.value


@dataclass
class TrainingTurn:
    turn_id: str
    episode_id: str
    task_id: str
    session_id: str | None
    turn_index: int
    role: str
    purpose: str
    primary_candidate: bool
    system: str
    messages: tuple[Mapping[str, object], ...]
    tool_view_hash: str
    tool_names: tuple[str, ...]
    tools_enabled: bool
    assistant_output: str = ""
    stop_reason: str | None = None
    tool_calls: tuple[TrainingToolCall, ...] = ()
    labels: tuple[str, ...] = ()
    recovers_turn_ids: tuple[str, ...] = ()
    mutation_generation_before: int = 0
    mutation_generation_after: int = 0
    verification_generation: int | None = None
    model_action_id: str | None = None
    usage: Mapping[str, int] | None = None
    error: str | None = None
    provenance: tuple[str, ...] = ()
    memory_provenance_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingEpisode:
    schema_version: int
    episode_id: str
    task_id: str
    session_id: str | None
    timestamp: str
    provenance: Mapping[str, object]
    task: Mapping[str, object]
    turns: tuple[TrainingTurn, ...]
    tool_views: Mapping[str, object]
    execution: Mapping[str, object]
    outcome: Mapping[str, object]
    training_metadata: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "TrainingEpisode":
        if not isinstance(raw, Mapping):
            raise TrainingDataError("training episode must be an object")

        if raw.get("schema_version") != TRAINING_SCHEMA_VERSION:
            raise TrainingDataError(
                f"unsupported training schema: {raw.get('schema_version')!r}"
            )

        required = ("episode_id", "task_id", "timestamp", "provenance", "task",
                    "turns", "tool_views", "execution", "outcome",
                    "training_metadata")

        if any(key not in raw for key in required):
            raise TrainingDataError("malformed training episode")

        if (not isinstance(raw["turns"], Sequence)
                or isinstance(raw["turns"], (str, bytes))):
            raise TrainingDataError("training turns must be an array")

        turns = []

        # Unknown keys are dropped rather than refused: an episode written by a
        # newer harness stays readable as long as its schema version matches.

        try:
            for value in raw["turns"]:
                calls = tuple(TrainingToolCall(**item)
                              for item in value.get("tool_calls", ()))
                turns.append(TrainingTurn(
                    **{key: value[key] for key in TrainingTurn.__dataclass_fields__
                       if key in value and key != "tool_calls"},
                    tool_calls=calls,
                ))

            return cls(
                schema_version=int(raw["schema_version"]),
                episode_id=str(raw["episode_id"]), task_id=str(raw["task_id"]),
                session_id=(str(raw["session_id"])
                            if raw.get("session_id") is not None else None),
                timestamp=str(raw["timestamp"]), provenance=dict(raw["provenance"]),
                task=dict(raw["task"]), turns=tuple(turns),
                tool_views=dict(raw["tool_views"]), execution=dict(raw["execution"]),
                outcome=dict(raw["outcome"]),
                training_metadata=dict(raw["training_metadata"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainingDataError(f"malformed training episode: {exc}") from exc


class TrainingRedactionPolicy:
    """Best-effort export hygiene; never a runtime security authority."""

    _KEY = re.compile(
        r"(?:^|[_-])(?:authorization|api[_-]?key|password|passwd|secret|token|credential)"
        r"(?:$|[_-])", re.I
    )
    _PATTERNS = (
        ("private_key", re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?"
            r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.S)),
        ("authorization", re.compile(r"(?i)\bAuthorization\s*:\s*\S+(?:\s+\S+)?")),
        ("api_key", re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{16,}\b")),
        ("credential", re.compile(
            r"(?i)\b(password|passwd|api[_-]?key|secret|token)\s*[:=]\s*"
            r"([^\s,;]+)")),
        ("environment_credential", re.compile(
            r"(?im)\b[A-Z][A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY)"
            r"\s*=\s*[^\s,;]+")),
    )

    def __init__(self, sensitive_values: Sequence[str] = ()) -> None:
        # Longest first, so a secret that contains another is replaced whole
        # rather than being left half-redacted. Values under eight characters
        # are ignored: they match ordinary text far too often.

        self.sensitive_values = tuple(sorted(
            {value for value in sensitive_values if isinstance(value, str)
             and len(value) >= 8}, key=len, reverse=True,
        ))

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None):
        """Take the VALUES of credential-looking variables as literals to redact."""

        environment = environment if environment is not None else os.environ

        return cls(value for key, value in environment.items() if cls._KEY.search(key))

    def redact(self, value: object) -> tuple[object, Mapping[str, object]]:
        """Walk a structure, redacting text and whole fields, and count what went."""

        counts: dict[str, int] = {}

        def replace_text(text: str) -> str:
            result = text

            for secret in self.sensitive_values:
                count = result.count(secret)

                if count:
                    result = result.replace(secret, "[REDACTED:environment_secret]")
                    counts["environment_secret"] = counts.get("environment_secret", 0) + count

            for kind, pattern in self._PATTERNS:
                result, count = pattern.subn(f"[REDACTED:{kind}]", result)

                if count:
                    counts[kind] = counts.get(kind, 0) + count

            return result

        def walk(item: object, key: str | None = None) -> object:
            # A field whose NAME looks like a credential is dropped whole: its
            # value need not look like a secret to be one.

            if key is not None and self._KEY.search(key):
                counts["sensitive_field"] = counts.get("sensitive_field", 0) + 1
                return "[REDACTED:sensitive_field]"

            if isinstance(item, str):
                return replace_text(item)

            if isinstance(item, Mapping):
                return {str(k): walk(v, str(k)) for k, v in item.items()}

            if isinstance(item, (list, tuple)):
                return [walk(part) for part in item]

            return item

        redacted = walk(value)

        return redacted, {
            "policy_version": 1,
            "redaction_count": sum(counts.values()),
            "categories": tuple(sorted(counts)),
        }


class TrainingEligibilityPolicy:
    VERSION = 1

    def classify(self, *, state: WorkingState, task_status: str,
                 terminal_reason: str, verification_status: str,
                 checkpoint_status: str | None, complete: bool = True,
                 excluded: bool = False) -> tuple[TrainingEligibility, tuple[str, ...]]:
        """Classify an episode, and say every reason it is not a positive example.

        Nothing here throws data away. An episode that fails a check becomes a
        FAILURE_TRAJECTORY, which is still worth keeping -- it is what a
        recovery looks like -- just not something to imitate.
        """

        reasons = []

        if excluded:
            return TrainingEligibility.EXCLUDED, ("policy_excluded",)

        if not complete:
            return TrainingEligibility.INCOMPLETE, ("not_finalized",)

        if state.terminal_status == TerminalStatus.INTERRUPTED or task_status == "interrupted":
            reasons.append("cancelled")

        if terminal_reason == "stalled":
            reasons.append("stalled")

        if task_status == "budget_exhausted" or terminal_reason == "budget_exhausted":
            reasons.append("budget_exhausted")

        if task_status != "completed" or state.terminal_status != TerminalStatus.COMPLETED:
            reasons.append("task_not_completed")

        # A task that changed nothing has nothing to verify or roll back, so
        # these two requirements only apply once something was actually written.

        mutated = state.mutation_generation > 0

        if mutated:
            if verification_status != CompletionVerificationStatus.VERIFIED.value:
                reasons.append("current_generation_not_verified")

            if checkpoint_status != "finalized":
                reasons.append("checkpoint_not_finalized")

        # Failures of the harness itself, as opposed to failures of the work:
        # an unresolved one means the episode does not describe a clean run.

        blocking_categories = {
            "runtime_error", "model_error", "interrupted", "checkpoint_conflict",
        }

        if any(item.category in blocking_categories for item in state.unresolved_failures):
            reasons.append("unresolved_blocking_runtime_failure")

        reasons = list(dict.fromkeys(reasons))

        if reasons:
            return TrainingEligibility.FAILURE_TRAJECTORY, tuple(reasons)

        return TrainingEligibility.POSITIVE_CANDIDATE, ()


def _canonical_message(message: ConversationMessage) -> Mapping[str, object]:
    blocks = []

    for block in message.content:
        if isinstance(block, TextBlock):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            blocks.append({"type": "tool_use", "id": block.id, "name": block.name,
                           "arguments": dict(block.arguments)})
        elif isinstance(block, ToolResultBlock):
            blocks.append({"type": "tool_result", "tool_call_id": block.tool_call_id,
                           "content": block.content, "is_error": block.is_error})

    return {"role": message.role, "content": blocks}


def _tool_view(tools: Sequence[ToolDefinition]) -> tuple[str, Mapping[str, object]]:
    snapshot = {"schema_version": 1, "tools": [
        {"name": item.name, "description": item.description,
         "input_schema": dict(item.input_schema)} for item in tools
    ]}
    encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()

    return hashlib.sha256(encoded).hexdigest(), snapshot


def _opaque_id(prefix: str, task_id: str, session_id: str | None) -> str:
    value = f"{task_id}\0{session_id or ''}".encode()
    return f"{prefix}_{hashlib.sha256(value).hexdigest()[:32]}"


def discover_git_commit(workspace: str | os.PathLike[str]) -> str | None:
    """The workspace's HEAD, so an episode records the code it was produced by."""

    # Best-effort and quick: a workspace need not be a Git checkout, and
    # capture must never wait on one.

    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(workspace), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=1, check=False,
        )
        value = result.stdout.strip()

        return value if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40,64}", value) else None
    except (OSError, subprocess.SubprocessError):
        return None


class TrainingRecorder:
    """Task-scoped observer.  It projects authoritative state; it does not own it."""

    def __init__(self, store: object, *, task_id: str, session_id: str | None,
                 objective: str, workspace: str, project: str | None = None,
                 data_origin: str = "NORMAL_USAGE",
                 provider: str | None = None, model: str | None = None,
                 generation_config: Mapping[str, object] | None = None,
                 harness_commit: str | None = None,
                 redaction_policy: TrainingRedactionPolicy | None = None,
                 standard_binding: Mapping[str, object] | None = None) -> None:
        self.store = store
        self.task_id = task_id
        self.session_id = session_id
        self.episode_id = _opaque_id("episode", task_id, session_id)
        self.objective = objective
        self.workspace = os.path.realpath(workspace)
        self.project = project
        self.data_origin = data_origin
        self.provider = provider
        self.model = model
        self.generation_config = dict(generation_config or {})
        self.harness_commit = harness_commit
        self.redaction_policy = redaction_policy or TrainingRedactionPolicy.from_environment()
        self.standard_binding = dict(standard_binding) if standard_binding else None
        self.turns: list[TrainingTurn] = []
        self.tool_views: dict[str, object] = {}
        self.auxiliary_calls: list[Mapping[str, object]] = []
        self._redaction_counts: dict[str, int] = {}
        self._open_turn: TrainingTurn | None = None
        self._failed_turn_ids: list[str] = []
        self._final_episode: TrainingEpisode | None = None

        # A draft from an interrupted run is reloaded so a resumed task keeps
        # its earlier turns instead of recording a truncated episode.

        draft = self.store.load_draft(self.episode_id)

        if draft is not None:
            try:
                self.turns = []

                for value in draft.get("turns", ()):
                    calls = tuple(TrainingToolCall(**item)
                                  for item in value.get("tool_calls", ()))
                    self.turns.append(TrainingTurn(
                        **{key: value[key] for key in TrainingTurn.__dataclass_fields__
                           if key in value and key != "tool_calls"},
                        tool_calls=calls,
                    ))

                self.tool_views = dict(draft.get("tool_views", {}))
                self._failed_turn_ids = [turn.turn_id for turn in self.turns
                                         if "failed_tool_action" in turn.labels]
            except (KeyError, TypeError, ValueError) as exc:
                raise TrainingDataError(f"malformed training draft: {exc}") from exc

        self.store.save_draft(self.episode_id, self._draft_payload())

    def _clean(self, value: object) -> object:
        cleaned, metadata = self.redaction_policy.redact(value)

        for category in metadata["categories"]:
            self._redaction_counts[category] = self._redaction_counts.get(category, 0) + 1

        return cleaned

    def begin_model_turn(self, *, system: str,
                         conversation: Sequence[ConversationMessage],
                         tools: Sequence[ToolDefinition], role: str, purpose: str,
                         mutation_generation: int,
                         memory_provenance_ids: Sequence[str] = (),
                         external_context: bool = False,
                         tools_enabled: bool = True) -> None:

        # Tool schemas are stored once per distinct set and referenced by hash:
        # they are large and rarely change between turns.

        view_hash, view = _tool_view(tools)
        self.tool_views.setdefault(view_hash, self._clean(view))

        # Where this turn's content came from, which decides what the episode
        # may later be used for.

        provenance = {TrainingProvenance.GENERATED.value}

        if any(item.role == "user" for item in conversation):
            provenance.add(TrainingProvenance.LOCAL_USER.value)

        if external_context:
            provenance.add(TrainingProvenance.EXTERNAL_WEB.value)

        if memory_provenance_ids:
            provenance.add(TrainingProvenance.DURABLE_MEMORY.value)

        turn = TrainingTurn(
            turn_id=f"turn_{len(self.turns):06d}", episode_id=self.episode_id,
            task_id=self.task_id, session_id=self.session_id,
            turn_index=len(self.turns), role=role, purpose=purpose,
            primary_candidate=(role == "main" and purpose == "primary_agent"),
            system=str(self._clean(system)),
            messages=tuple(self._clean([_canonical_message(item)
                                       for item in conversation])),
            tool_view_hash=view_hash, tool_names=tuple(item.name for item in tools),
            tools_enabled=tools_enabled,
            mutation_generation_before=mutation_generation,
            mutation_generation_after=mutation_generation,
            provenance=tuple(sorted(provenance)),
            memory_provenance_ids=tuple(memory_provenance_ids),
        )
        self.turns.append(turn)
        self._open_turn = turn

    def complete_model_turn(self, turn: ModelTurn, *, model_action_id: str,
                            mutation_generation: int) -> None:
        target = self._open_turn

        if target is None:
            return

        target.assistant_output = str(self._clean(turn.text))
        target.stop_reason = turn.stop_reason.value
        target.tool_calls = tuple(TrainingToolCall(
            item.id, item.name, self._clean(dict(item.arguments)),
        ) for item in turn.tool_calls)
        target.model_action_id = model_action_id
        target.mutation_generation_after = mutation_generation
        target.usage = dict(turn.usage) if turn.usage else None
        self._open_turn = None
        self.store.save_draft(self.episode_id, self._draft_payload())

    def fail_model_turn(self, error: Exception, *, model_action_id: str,
                        mutation_generation: int) -> None:
        if self._open_turn is None:
            return

        self._open_turn.error = str(self._clean(str(error)[:240]))
        self._open_turn.model_action_id = model_action_id
        self._open_turn.mutation_generation_after = mutation_generation
        self._open_turn.labels = ("failed_model_call",)
        self._failed_turn_ids.append(self._open_turn.turn_id)
        self._open_turn = None
        self.store.save_draft(self.episode_id, self._draft_payload())

    def record_tool_result(self, envelope: ToolResultEnvelope, *,
                           mutation_generation_before: int,
                           mutation_generation_after: int,
                           verification_generation: int | None = None) -> None:
        """Attach an observed tool result to the turn that asked for it."""

        # Searched from the end: the call being answered is almost always the
        # most recent, and tool call ids are unique across the episode anyway.

        target = next((turn for turn in reversed(self.turns)
                       if any(call.tool_call_id == envelope.tool_call_id
                              for call in turn.tool_calls)), None)

        if target is None:
            return

        failed = not envelope.success
        updated = []

        for call in target.tool_calls:
            if call.tool_call_id == envelope.tool_call_id:
                updated.append(TrainingToolCall(
                    call.tool_call_id, call.name, call.arguments,
                    envelope.action_id, envelope.status.value,
                    str(self._clean(envelope.model_content)), envelope.result_reference,
                ))
            else:
                updated.append(call)

        target.tool_calls = tuple(updated)
        labels = list(target.labels)
        label = "failed_tool_action" if failed else "successful_tool_action"

        if label not in labels:
            labels.append(label)

        if envelope.status == ToolResultStatus.CACHED and "repeated_action" not in labels:
            labels.append("repeated_action")

        if envelope.mutation and "grounded_mutation" not in labels:
            labels.append("grounded_mutation")

        if (not failed and envelope.read_paths
                and "useful_read" not in labels):
            labels.append("useful_read")

        if (not failed and envelope.tool_name == "search_corpus"
                and envelope.model_content.strip()
                and "useful_search" not in labels):
            labels.append("useful_search")

        if verification_generation is not None and "verification_action" not in labels:
            labels.append("verification_action")

        if envelope.tool_name in {"search_internet", "web"}:
            target.provenance = tuple(sorted(set(target.provenance) | {
                TrainingProvenance.EXTERNAL_WEB.value,
                TrainingProvenance.TOOL_OUTPUT.value,
            }))

        # A success following recorded failures is what a recovery looks like,
        # and the turns it recovered from are named so the pair can be learned.

        if not failed and self._failed_turn_ids:
            labels.append("recovery_after_failure")
            target.recovers_turn_ids = tuple(self._failed_turn_ids)
            self._failed_turn_ids.clear()

        if failed and target.turn_id not in self._failed_turn_ids:
            self._failed_turn_ids.append(target.turn_id)

        target.labels = tuple(dict.fromkeys(labels))
        target.mutation_generation_before = min(
            target.mutation_generation_before, mutation_generation_before,
        )
        target.mutation_generation_after = mutation_generation_after
        target.verification_generation = verification_generation

        # This mutation invalidates every verification recorded before it, so
        # those turns are marked rather than left looking like passing evidence.

        if envelope.mutation:
            for prior in self.turns:
                if (prior.verification_generation is not None
                        and prior.verification_generation < mutation_generation_after):
                    prior.labels = tuple(dict.fromkeys(
                        (*prior.labels, "stale_verification")
                    ))

        self.store.save_draft(self.episode_id, self._draft_payload())

    def _draft_payload(self) -> Mapping[str, object]:
        return {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "episode_id": self.episode_id, "task_id": self.task_id,
            "session_id": self.session_id, "status": "incomplete",
            "turns": [asdict(item) for item in self.turns],
            "tool_views": self.tool_views,
        }

    def finalize(self, *, state: WorkingState, task_status: str,
                 terminal_reason: str, verification_status: str,
                 checkpoint_status: str | None, final_response: str,
                 review_status: str | None = None,
                 budget: Mapping[str, object] | None = None,
                 standard_source_ids: Sequence[str] = ()) -> TrainingEpisode:
        """Close the episode: classify it, project the state, and store it once."""

        # Finalizing twice would write a second episode for one task, so the
        # first result is remembered and returned.

        if self._final_episode is not None:
            return self._final_episode

        eligibility, reasons = TrainingEligibilityPolicy().classify(
            state=state, task_status=task_status, terminal_reason=terminal_reason,
            verification_status=verification_status,
            checkpoint_status=checkpoint_status,
        )

        if eligibility == TrainingEligibility.POSITIVE_CANDIDATE and self.turns:
            labels = list(self.turns[-1].labels)
            labels.append("final_success")
            self.turns[-1].labels = tuple(dict.fromkeys(labels))

        # A snapshot of the working state as evidence. The working state stays
        # authoritative; this is a copy for the record, not a second source.

        state_projection = {
            "files_read": sorted(state.files_read),
            "files_modified": sorted(state.modified_files),
            "files_created": sorted(state.created_files),
            "files_deleted": sorted(state.deleted_files),
            "mutation_generation": state.mutation_generation,
            "actions": [asdict(item) for item in state.actions],
            "failures": [asdict(item) for item in state.failures.values()],
            "verifications": [asdict(item) for item in state.verifications],
            "retries": state.retry_count,
            "repeated_actions": state.repeated_action_count,
            "plan_steps": [asdict(item) for item in state.plan_steps.values()],
            "result_references": sorted({
                call.result_reference for turn in self.turns for call in turn.tool_calls
                if call.result_reference
            }),
            "auxiliary_calls": self.auxiliary_calls,
        }

        # The workspace is identified by hash rather than by path: the episode
        # says which workspace it came from without recording where it lives.

        workspace_fingerprint = hashlib.sha256(self.workspace.encode()).hexdigest()

        raw = {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "episode_id": self.episode_id, "task_id": self.task_id,
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provenance": {
                "harness_git_commit": self.harness_commit,
                "project": self.project, "data_origin": self.data_origin,
                "workspace_fingerprint": workspace_fingerprint,
                "model_backend": self.provider, "model": self.model,
                "generation_configuration": self.generation_config,
                "system_prompt_hash": hashlib.sha256(
                    (self.turns[0].system if self.turns else "").encode()
                ).hexdigest(),
                "policy_versions": {"eligibility": 1, "redaction": 1},
                "standard_binding": self.standard_binding,
                "standard_source_ids_used": sorted(set(standard_source_ids)),
            },
            "task": {"objective": self.objective,
                     "explicit_constraints": list(state.user_constraints),
                     "acceptance_criteria": list(state.acceptance_criteria)},
            "turns": [asdict(item) for item in self.turns],
            "tool_views": self.tool_views,
            "execution": state_projection,
            "outcome": {
                "terminal_task_status": task_status,
                "working_state_terminal_status": state.terminal_status.value,
                "verification_status": verification_status,
                "checkpoint_status": checkpoint_status,
                "unresolved_failures": [asdict(item) for item in state.unresolved_failures],
                "review_status": review_status,
                "cancelled": state.terminal_status == TerminalStatus.INTERRUPTED,
                "budget_status": dict(budget or {}),
                "terminal_reason": terminal_reason,
                "final_response": final_response,
            },
            "training_metadata": {
                "eligibility": eligibility.value,
                "quality_labels": sorted({label for turn in self.turns
                                          for label in turn.labels}),
                "exclusion_reasons": list(reasons),
                "redaction": {"policy_version": 1,
                              "categories": sorted(self._redaction_counts),
                              "fields_redacted": sum(self._redaction_counts.values())},
                "contains_external_web": any(
                    TrainingProvenance.EXTERNAL_WEB.value in turn.provenance
                    for turn in self.turns
                ),
                "contains_licensed_standard": bool(
                    self.standard_binding and
                    self.standard_binding.get("data_origin") == "LICENSED_STANDARD"),
                "standard_export_prohibited": bool(self.standard_binding),
                "requires_governance_review": bool(self.standard_binding),
                "default_sft_turns": [turn.turn_id for turn in self.turns
                                      if turn.primary_candidate
                                      and "failed_tool_action" not in turn.labels
                                      and not turn.error],
            },
        }

        # Redacted as a whole once more before it is stored: the individual
        # fields were cleaned as they arrived, this catches what they compose into.

        cleaned = self._clean(raw)
        episode = TrainingEpisode.from_dict(cleaned)

        self.store.save_episode(episode)
        self.store.discard_draft(self.episode_id)

        self._final_episode = episode

        return episode
