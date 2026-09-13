"""Provider boundary for SPEAR model turns.

This module intentionally knows nothing about SPEAR tool execution, workspace
policy, RAG, persistence, or UI.  It translates a small canonical conversation
to/from a model provider only.
"""

from __future__ import annotations

import json
import importlib
import os
import re
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Literal, Mapping, Protocol, Sequence

from openai import APIConnectionError, APIStatusError


class ModelBackendConfigurationError(RuntimeError):
    """A provider cannot be used before a local configuration prerequisite."""


class StopReason(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    ERROR = "error"
    OTHER = "other"


@dataclass(frozen=True)
class ModelToolCall:
    id: str
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True)
class ModelTurn:
    text: str
    tool_calls: tuple[ModelToolCall, ...]
    stop_reason: StopReason
    usage: Mapping[str, int] | None = None
    error: str | None = None


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, object]


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class ToolUseBlock:
    id: str
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True)
class ToolResultBlock:
    tool_call_id: str
    content: str
    is_error: bool = False


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


@dataclass(frozen=True)
class ConversationMessage:
    """Provider-neutral message made from typed text and tool blocks."""

    role: Literal["user", "assistant"]
    content: tuple[ContentBlock, ...]

    # Who wrote it. The API has one user role and the harness speaks in it:
    # every nudge, demand and redirect is appended as a user message, because
    # that is the only way to say something to the model mid-turn. Nothing
    # downstream could tell those apart from the operator's own words, and
    # the routing that reads "the user's request" read the harness's last
    # nudge instead -- a nudge containing the word "fix", which made a plain
    # question look like a request to edit the tree. The provider never sees
    # this field; it exists so the harness can recognise its own voice.
    authored_by: Literal["operator", "harness"] = "operator"

    def __post_init__(self) -> None:
        if self.role == "assistant" and any(
            isinstance(block, ToolResultBlock) for block in self.content
        ):
            raise ValueError("assistant messages cannot contain tool results")

        if self.role == "user" and any(
            isinstance(block, ToolUseBlock) for block in self.content
        ):
            raise ValueError("user messages cannot contain tool calls")


class ModelBackend(Protocol):
    def complete(
        self,
        *,
        system: str,
        conversation: Sequence[ConversationMessage],
        tools: Sequence[ToolDefinition],
        use_tools: bool,
        on_token: Callable[[], None] | None = None,
    ) -> ModelTurn: ...


def canonical_tools_from_openai(tools: Sequence[Mapping[str, object]]) -> tuple[ToolDefinition, ...]:
    """Adapt the existing SPEAR tool declarations without duplicating schemas."""
    result = []

    for item in tools:
        function = item["function"]

        if not isinstance(function, Mapping):
            raise ValueError("OpenAI tool definition has no function object")

        result.append(ToolDefinition(
            name=str(function["name"]),
            description=str(function.get("description", "")),
            input_schema=function["parameters"],
        ))

    return tuple(result)


class OpenAICompatibleBackend:
    """Current llama.cpp/vLLM/Qwen contract behind a provider-neutral API."""

    # A dot is part of a tool name: standard.search, standard.fetch,
    # standard.get_structure, standard.cite -- and they are the ONLY tools
    # whose names carry one. \w+ therefore recovered every leaked call except
    # the normative ones, which is the worst possible place to miss.
    #
    # A bound turn spent twelve rounds reading C, planned the change, and
    # ended on a leaked <function=standard.fetch>. Unrecovered, it read as
    # prose, so the runtime took a plan for a conclusion and the turn wrote
    # nothing. The task had asked for an implementation.

    _LEAKED_CALL_RE = re.compile(
        r"(?:<tool_call>\s*)?<function=([\w.]+)>(.*?)</function>(?:\s*</tool_call>)?",
        re.DOTALL,
    )
    _LEAKED_PARAM_RE = re.compile(r"<parameter=(\w+)>\n?(.*?)\n?</parameter>", re.DOTALL)
    _JSON_CALL_RE = re.compile(r'\{\s*"name"\s*:\s*"([\w.]+)"\s*,\s*"arguments"\s*:')

    def __init__(
        self,
        client: object,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.client = client
        self.model = model or os.environ.get("SPEAR_MODEL_NAME", "qwen3")
        self.temperature = temperature if temperature is not None else float(
            os.environ.get("SPEAR_TEMP", "0.25")
        )
        self.max_tokens = max_tokens if max_tokens is not None else int(
            os.environ.get("SPEAR_MAX_TOKENS", "8192")
        )
        self._sleep = sleep_fn

    def discover_model_name(self) -> str | None:
        try:
            return self.client.models.list().data[0].id
        except Exception:
            return None

    @staticmethod
    def _stop_reason(value: str | None, has_calls: bool) -> StopReason:
        if has_calls or value == "tool_calls":
            return StopReason.TOOL_USE

        if value == "stop" or value is None:
            return StopReason.END_TURN

        if value == "length":
            return StopReason.MAX_TOKENS

        if value == "content_filter":
            return StopReason.REFUSAL

        return StopReason.OTHER

    @staticmethod
    def _openai_tools(tools: Sequence[ToolDefinition]) -> list[dict[str, object]]:
        return [
            {"type": "function", "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.input_schema),
            }}
            for tool in tools
        ]

    @staticmethod
    def _openai_messages(
        system: str, conversation: Sequence[ConversationMessage]
    ) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = [{"role": "system", "content": system}]

        for message in conversation:
            text = "".join(block.text for block in message.content if isinstance(block, TextBlock))
            calls = [block for block in message.content if isinstance(block, ToolUseBlock)]
            results = [block for block in message.content if isinstance(block, ToolResultBlock)]

            if message.role == "assistant":
                item: dict[str, object] = {"role": "assistant", "content": text}

                if calls:
                    item["tool_calls"] = [{
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    } for call in calls]

                messages.append(item)
            elif results:
                for result in results:
                    messages.append({
                        "role": "tool", "tool_call_id": result.tool_call_id,
                        "content": result.content,
                    })
            else:
                messages.append({"role": "user", "content": text})

        return messages

    @classmethod
    def _recover_leaked_calls(cls, text: str) -> tuple[str, list[ModelToolCall]]:
        calls: list[ModelToolCall] = []
        counter = 0

        for match in cls._LEAKED_CALL_RE.finditer(text):
            calls.append(ModelToolCall(
                id=f"leaked_{counter}", name=match.group(1),
                arguments={key: value for key, value in cls._LEAKED_PARAM_RE.findall(match.group(2))},
            ))
            counter += 1

        if calls:
            text = cls._LEAKED_CALL_RE.sub("", text).strip()

        out, index = [], 0

        while True:
            match = cls._JSON_CALL_RE.search(text, index)

            if match is None:
                out.append(text[index:])
                break

            out.append(text[index:match.start()])
            depth, in_string, escaped, end = 0, False, False, None

            for pos in range(match.start(), len(text)):
                char = text[pos]

                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                elif char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1

                    if depth == 0:
                        end = pos + 1
                        break

            if end is None:
                out.append(text[match.start():])
                break

            blob = text[match.start():end]

            try:
                value = json.loads(blob)
                arguments = value.get("arguments")

                if isinstance(arguments, dict):
                    calls.append(ModelToolCall(
                        id=f"leaked_{counter}", name=str(value["name"]), arguments=arguments
                    ))
                    counter += 1
                else:
                    out.append(blob)
            except (TypeError, ValueError, json.JSONDecodeError):
                out.append(blob)

            index = end

        return "".join(out).strip(), calls

    def complete(
        self,
        *,
        system: str,
        conversation: Sequence[ConversationMessage],
        tools: Sequence[ToolDefinition],
        use_tools: bool,
        on_token: Callable[[], None] | None = None,
    ) -> ModelTurn:
        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": self._openai_messages(system, conversation),
            "temperature": self.temperature,
            "top_p": 0.8,
            "max_tokens": self.max_tokens,
            "stream": True,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
                "top_k": 20,
                "repeat_penalty": 1.05,
                "repetition_penalty": 1.05,
            },
        }

        if use_tools:
            kwargs["tools"] = self._openai_tools(tools)

        stream = None
        malformed_retries = 0

        for attempt in range(120):
            try:
                stream = self.client.chat.completions.create(**kwargs)
                break
            except APIStatusError as exc:
                if exc.status_code == 503 and "loading" in str(exc).lower():
                    self._sleep(5)
                    continue

                if exc.status_code == 500 and "tool call" in str(exc).lower():
                    malformed_retries += 1

                    if malformed_retries <= 3:
                        self._sleep(1)
                        continue

                    return ModelTurn("", (), StopReason.ERROR,
                                     error="model kept emitting an unparseable tool call")

                raise
            except APIConnectionError:
                self._sleep(5)

        if stream is None:
            return ModelTurn("", (), StopReason.ERROR, error="model never became ready")

        text, calls, finish_reason = [], {}, None

        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue

            choice = chunk.choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            delta = getattr(choice, "delta", None)

            if delta is None:
                continue

            if getattr(delta, "content", None):
                text.append(delta.content)

                if on_token:
                    on_token()

            for tool_delta in getattr(delta, "tool_calls", None) or ():
                item = calls.setdefault(tool_delta.index, {"id": None, "name": "", "arguments": []})

                if getattr(tool_delta, "id", None):
                    item["id"] = tool_delta.id

                function = getattr(tool_delta, "function", None)

                if function is not None:
                    if getattr(function, "name", None):
                        item["name"] += function.name

                    if getattr(function, "arguments", None):
                        item["arguments"].append(function.arguments)

                if on_token:
                    on_token()

        canonical_calls: list[ModelToolCall] = []

        for index, call in sorted(calls.items()):
            raw_arguments = "".join(call["arguments"])

            try:
                arguments = json.loads(raw_arguments or "{}")
            except json.JSONDecodeError:
                return ModelTurn("".join(text), (), StopReason.ERROR,
                                 error=f"invalid JSON arguments for tool {call['name'] or index}")

            if not isinstance(arguments, dict):
                return ModelTurn("".join(text), (), StopReason.ERROR,
                                 error=f"non-object arguments for tool {call['name'] or index}")

            canonical_calls.append(ModelToolCall(
                id=call["id"] or f"call_{index}", name=call["name"], arguments=arguments
            ))

        clean_text, leaked = self._recover_leaked_calls("".join(text))
        canonical_calls.extend(leaked)

        return ModelTurn(
            clean_text,
            tuple(canonical_calls),
            self._stop_reason(finish_reason, bool(canonical_calls)),
        )


def anthropic_credentials_available() -> bool:
    """True when the Anthropic SDK can resolve a credential by itself.

    The SDK owns the resolution chain — ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
    an ``ant auth login`` profile under ~/.config/anthropic, then workload
    identity federation.  Asking it is the whole point: reimplementing that
    order here would drift.  A missing SDK or an unreadable profile is a
    negative answer, never an exception — the caller decides what to do about
    it, and this module never prompts.
    """

    if os.environ.get("ANTHROPIC_API_KEY"):
        return True

    try:
        credentials = importlib.import_module("anthropic.lib.credentials")
        resolved = credentials.default_credentials()

        if resolved is None:
            return False

        # Selecting a profile is NOT holding a credential: `ant auth logout`
        # leaves the config behind, the chain still resolves it, and the
        # failure only surfaces at the first request — the harness would then
        # skip the sign-in offer and report a backend error instead.  Minting
        # settles it: cached while the token is valid, a refresh when it is
        # not, an exception when there is nothing to mint from.

        resolved.provider()

        return True
    except Exception:
        return False


# Prompt caching is a prefix contract: a marked block caches everything up to
# and including itself.  The adapter therefore marks what it has *observed* to
# be stable -- tools sent unchanged since the previous call, and the longest
# common prefix of the two system strings -- rather than what a caller
# promises.  Nothing in the canonical turn had to grow a field for this.

_CACHE_CONTROL = {"type": "ephemeral"}

# A breakpoint below the model's minimum cacheable prefix (1024 tokens on the
# Sonnet and Opus tiers) is silently ignored, so a short prefix is not marked
# at all.  Four characters per token is the same rough charge the runtime uses
# for text it cannot count; the margin keeps a marked block above the floor
# rather than exactly on it.

_MIN_CACHE_CHARS = 6000

# Anthropic constrains a tool name to ^[a-zA-Z0-9_-]{1,128}$, and the
# canonical registry names the normative tools `standard.search` and
# `standard.fetch`. The dot made every Anthropic turn fail the request
# outright -- 400 invalid_request_error, before a single token -- so the
# provider that could not run at all was the one holding the master leg of
# the distillation. The name is a wire detail of one provider: it is
# translated at the boundary and nowhere else, so the registry, the tool
# loop, the policies and the transcript all keep the canonical name.
_WIRE_UNSAFE = re.compile(r"[^a-zA-Z0-9_-]")


def _stable_system_prefix(previous: str | None, current: str) -> str:
    """The opening of ``current`` that also opened the previous call.

    ``ContextEngine`` composes the system text from layered items with the
    stable layers first -- system rules, then project rules, then selected
    memory -- so what two consecutive calls share really is a prefix and not a
    coincidence.  The cut is moved back to the last line break: a breakpoint
    inside a sentence buys nothing, since the tail is re-sent either way, and
    it produces a cached block nobody can read in a trace.
    """

    if not previous:
        return ""

    limit = min(len(previous), len(current))
    index = 0

    while index < limit and previous[index] == current[index]:
        index += 1

    return current[:current.rfind("\n", 0, index) + 1]


class AnthropicBackend:
    """Native Anthropic Messages API adapter for the canonical SPEAR turn.

    One instance spans a whole session, which is what lets it compare a call
    with the one before it and cache the part that did not move.  Roles that
    share the instance simply miss: an Explorer turn between two Main turns
    shares no long prefix with either, so the comparison yields nothing and
    the call is sent uncached.
    """

    def __init__(
        self,
        client: object | None = None,
        *,
        model: str = "claude-sonnet-5",
        max_tokens: int | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens if max_tokens is not None else int(
            os.environ.get("SPEAR_MAX_TOKENS", "8192")
        )

        # What the previous call sent, kept only to recognise what repeats.
        # A first call marks nothing: there is no prior turn to compare
        # against, and writing a cache costs more than reading uncached.

        self._previous_system: str | None = None
        self._previous_tools: list[dict[str, object]] | None = None

        # wire name -> canonical name, for the tools whose names had to be
        # translated. Kept for the session: a turn offered no tools still
        # replays a history that names them.
        self._canonical_names: dict[str, str] = {}

        if client is not None:
            self.client = client
            return

        try:
            anthropic = importlib.import_module("anthropic")
        except ImportError as exc:
            raise ModelBackendConfigurationError(
                "Anthropic provider requires the 'anthropic' package in bin/"
            ) from exc

        key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")

        if key:
            self.client = anthropic.Anthropic(api_key=key)
        elif anthropic_credentials_available():
            # An OAuth profile (ant auth login) or a federated identity is
            # active: construct bare and let the SDK resolve and refresh it.
            # Passing api_key=None explicitly would short-circuit that chain.

            self.client = anthropic.Anthropic()
        else:
            raise ModelBackendConfigurationError(
                "no Anthropic credential: export ANTHROPIC_API_KEY, or open a "
                "session with `ant auth login`"
            )

    def discover_model_name(self) -> str:
        # Anthropic does not expose the OpenAI-compatible /models discovery
        # contract. The selected model is therefore the authoritative label.

        return self.model

    @staticmethod
    def _wire_name(name: str) -> str:
        """The name Anthropic will accept for this tool."""
        return _WIRE_UNSAFE.sub("_", name)

    def _tools(self, tools: Sequence[ToolDefinition]) -> list[dict[str, object]]:
        payload = []

        for tool in tools:
            wire = self._wire_name(tool.name)

            if wire != tool.name:
                self._canonical_names[wire] = tool.name

            payload.append({
                "name": wire,
                "description": tool.description,
                "input_schema": dict(tool.input_schema),
            })

        return payload

    def _cached_tools(
        self, payload: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """The tool schemas, with the last one marked when it is a repeat.

        The whole tool block is one prefix, so marking the final entry caches
        every schema before it.  The unmarked payload is what gets remembered:
        comparing a marked list against an unmarked one would never match
        again after the first hit.
        """

        repeated = payload == self._previous_tools
        self._previous_tools = payload

        if not repeated or len(json.dumps(payload)) < _MIN_CACHE_CHARS:
            return payload

        marked = list(payload)
        marked[-1] = {**marked[-1], "cache_control": dict(_CACHE_CONTROL)}

        return marked

    def _cached_system(self, system: str) -> str | list[dict[str, object]]:
        """The system text, split at the cache boundary when there is one.

        Below the threshold the plain string is returned unchanged -- it is
        what the API took before this existed, and one less shape to reason
        about in a trace.
        """

        stable = _stable_system_prefix(self._previous_system, system)
        self._previous_system = system

        if len(stable) < _MIN_CACHE_CHARS:
            return system

        blocks: list[dict[str, object]] = [{
            "type": "text", "text": stable, "cache_control": dict(_CACHE_CONTROL),
        }]
        volatile = system[len(stable):]

        # The tail is the working-state projection, the retrieved chunks and
        # the summaries: rebuilt every turn by design, and never cached.

        if volatile:
            blocks.append({"type": "text", "text": volatile})

        return blocks

    @staticmethod
    def _messages(conversation: Sequence[ConversationMessage]) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = []

        for message in conversation:
            content: list[dict[str, object]] = []

            for block in message.content:
                if isinstance(block, TextBlock):
                    # Anthropic refuses an empty text block outright -- 400,
                    # "text content blocks must be non-empty", for a block
                    # every other provider ignores. A turn that answers with
                    # tool calls and no prose produces exactly that, so the
                    # empty block is dropped rather than sent.
                    if block.text:
                        content.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUseBlock):
                    content.append({
                        "type": "tool_use", "id": block.id,
                        "name": AnthropicBackend._wire_name(block.name),
                        "input": dict(block.arguments),
                    })
                elif isinstance(block, ToolResultBlock):
                    result: dict[str, object] = {
                        "type": "tool_result", "tool_use_id": block.tool_call_id,
                        "content": block.content,
                    }

                    if block.is_error:
                        result["is_error"] = True

                    content.append(result)
                else:  # defensive in case the canonical union grows
                    raise ValueError(f"unsupported canonical content block: {type(block).__name__}")

            # A message whose every block was empty carries nothing; sending
            # it as an empty content array is the same 400 by another route.
            if content:
                messages.append({"role": message.role, "content": content})

        return messages

    @staticmethod
    def _stop_reason(value: str | None, has_calls: bool) -> StopReason:
        if has_calls and value == "tool_use":
            return StopReason.TOOL_USE

        if value == "end_turn":
            return StopReason.END_TURN

        if value == "max_tokens":
            return StopReason.MAX_TOKENS

        if value == "refusal":
            return StopReason.REFUSAL

        if value == "model_context_window_exceeded":
            return StopReason.ERROR

        return StopReason.OTHER

    @staticmethod
    def _usage(message: object) -> Mapping[str, int] | None:
        usage = getattr(message, "usage", None)

        if usage is None:
            return None

        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)

        if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
            return None

        counts = {"input_tokens": input_tokens, "output_tokens": output_tokens}

        # What makes the breakpoints above checkable rather than believed: a
        # warm turn reads most of its prefix instead of being charged for it.
        # Reported only when the provider sends them, so a turn that was not
        # cached carries the same two keys it always did.

        for name in ("cache_creation_input_tokens", "cache_read_input_tokens"):
            value = getattr(usage, name, None)

            if isinstance(value, int):
                counts[name] = value

        return counts

    def complete(
        self,
        *,
        system: str,
        conversation: Sequence[ConversationMessage],
        tools: Sequence[ToolDefinition],
        use_tools: bool,
        on_token: Callable[[], None] | None = None,
    ) -> ModelTurn:
        # Both helpers record what they saw for the next call, so each runs
        # exactly once per turn -- hence the local rather than a second call
        # inside the conditional below.

        tool_payload = self._cached_tools(self._tools(tools)) if use_tools else None

        kwargs: dict[str, object] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self._cached_system(system),
            "messages": self._messages(conversation),
            "thinking": {"type": "disabled"},
        }

        if tool_payload is not None:
            kwargs["tools"] = tool_payload

        try:
            with self.client.messages.stream(**kwargs) as stream:
                for _delta in stream.text_stream:
                    if on_token:
                        on_token()

                final = stream.get_final_message()
        except Exception as exc:
            return ModelTurn("", (), StopReason.ERROR,
                             error=f"Anthropic backend error: {type(exc).__name__}: {exc}")

        text: list[str] = []
        calls: list[ModelToolCall] = []

        try:
            for block in getattr(final, "content", ()):
                block_type = getattr(block, "type", None)

                if block_type == "text":
                    value = getattr(block, "text", None)

                    if not isinstance(value, str):
                        raise ValueError("text block has no string text")

                    text.append(value)
                elif block_type == "tool_use":
                    call_id, name, arguments = (
                        getattr(block, "id", None), getattr(block, "name", None),
                        getattr(block, "input", None),
                    )

                    if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(arguments, dict):
                        raise ValueError("tool_use block is malformed")

                    calls.append(ModelToolCall(
                        call_id, self._canonical_names.get(name, name),
                        arguments))
                else:
                    raise ValueError(f"unexpected Anthropic content block: {block_type!r}")
        except ValueError as exc:
            return ModelTurn("".join(text), (), StopReason.ERROR,
                             usage=self._usage(final), error=str(exc))

        provider_stop = getattr(final, "stop_reason", None)

        if provider_stop == "pause_turn":
            return ModelTurn("".join(text), (), StopReason.ERROR, usage=self._usage(final),
                             error="unexpected Anthropic pause_turn with client-tools-only backend")

        if provider_stop == "tool_use" and not calls:
            return ModelTurn("".join(text), (), StopReason.ERROR, usage=self._usage(final),
                             error="Anthropic ended with tool_use but returned no tool calls")

        if calls and provider_stop != "tool_use":
            return ModelTurn("".join(text), (), StopReason.ERROR, usage=self._usage(final),
                             error=f"Anthropic returned tool calls with stop_reason {provider_stop!r}")

        if provider_stop == "model_context_window_exceeded":
            return ModelTurn("".join(text), (), StopReason.ERROR, usage=self._usage(final),
                             error="Anthropic model context window exceeded")

        stop_reason = self._stop_reason(provider_stop, bool(calls))

        return ModelTurn("".join(text), tuple(calls), stop_reason, usage=self._usage(final))
