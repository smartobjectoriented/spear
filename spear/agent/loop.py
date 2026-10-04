"""The coding core's conversation loop, after Hermes Agent's.

agent/conversation_loop.py (0cbc6e37), for a chat-completions endpoint with
the CLI defaults, does per iteration:

  send the history (contents stripped, tool-call arguments re-serialised
  compact with sorted keys) with the tools and max_tokens;
  a text-only reply cut by the output limit -> keep it, ask to continue
  (at most 4 times);
  a tool-call reply cut by the output limit -> drop it, ask again (at most
  4 times), then give up with "Response truncated ...";
  finish_reason "tool_calls" with no calls -> the dropped-tool-call nudge
  (at most 3 times);
  invalid tool-call JSON -> ask again silently twice, then answer each call
  with the invalid-JSON message;
  tool calls -> drop exact duplicates in the batch, keep the assistant turn,
  run each call through the guardrails and the dispatcher, append one tool
  message per call (role, name, tool_call_id, content);
  no tool calls -> the empty-after-tools nudge (once), the trailing-intent
  stall guard (twice), else the reply, think-stripped, is the answer.

Deviations: calls run sequentially (Hermes runs independent read-only calls
concurrently -- the results and their order are the same); there is no
context compression (Hermes compresses at about half of a 512K window; the
loop stops at that point instead and asks for a summary); tool results are
not spilled to disk above 100K characters (every tool here caps its own
output well below that).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from . import dispatch, tools
from .hermes import loop_constants as text
from .hermes import tool_guardrails
from .hermes.think import strip_think_blocks
from .host import Host, ToolRecord

MAX_TOKENS = 65536                  # the custom provider profile's default
LENGTH_CONTINUATIONS = 4
TRUNCATED_CALL_RETRIES = 4
DROPPED_CALL_NUDGES = 3
INVALID_JSON_RETRIES = 2
CONTINUE_NUDGES = 2


@dataclass
class CoreResult:
    final: str
    messages: list
    records: list[ToolRecord] = field(default_factory=list)
    stop: str = "completed"         # completed | iterations | tool_budget |
                                    # context | halted | truncated | model_error |
                                    # cancelled
    iterations: int = 0
    error: str | None = None


def _canonical_arguments(raw: str) -> str:
    """_canonicalize_api_tool_calls: compact, sorted, ASCII-escaped."""
    try:
        return json.dumps(json.loads(raw or "{}"), separators=(",", ":"), sort_keys=True)
    except ValueError:
        return "{}"


def _wire(messages: list) -> list:
    """The copy that is sent: contents stripped, arguments canonical."""
    out = []

    for message in messages:
        item = dict(message)

        if isinstance(item.get("content"), str):
            item["content"] = item["content"].strip()

        if item.get("tool_calls"):
            item["tool_calls"] = [
                {**call, "function": {**call["function"], "arguments":
                                      _canonical_arguments(call["function"]["arguments"])}}
                for call in item["tool_calls"]]

        out.append(item)

    return out


def _assistant(content: str, calls=()) -> dict:
    message = {"role": "assistant", "content": content}

    if calls:
        message["tool_calls"] = [{"id": call["id"], "type": "function",
                                  "function": {"name": call["name"],
                                               "arguments": call["arguments"] or "{}"}}
                                 for call in calls]

    return message


def _guardrails(guard, name, arguments, call_id, result):
    """run_agent._append_guardrail_observation, in its order."""
    failed, _ = tool_guardrails.classify_tool_failure(name, result)
    decision = guard.after_call(name, arguments, result, failed=failed)
    observation = guard.observe_call(name, arguments, result, tool_call_id=call_id,
                                     failed=failed)
    halt = None

    if observation.stub:
        result = observation.stub

    if decision.action in {"warn", "halt"}:
        result = tool_guardrails.append_toolguard_guidance(result, decision)

    if decision.should_halt:
        halt = decision
    else:
        streak = guard.halt_decision

        if streak is not None and streak.code == "identical_call_streak_halt":
            result = tool_guardrails.append_toolguard_guidance(result, streak)
            halt = streak

    if observation.notice:
        result = (result or "") + "\n\n" + observation.notice

    return result, halt


def run(*, model: Callable, host: Host, system: str, history: list, request: str,
        tool_definitions: list, max_iterations: int, max_tool_calls: int,
        context_window: int, max_tokens: int = MAX_TOKENS,
        on_model: Callable | None = None, on_tool: Callable | None = None,
        cancelled: Callable[[], bool] = lambda: False) -> CoreResult:
    """Run one user turn to its answer.

    `model(messages, tools, max_tokens)` returns a model_backend.RawTurn.
    `history` holds earlier turns as {"role", "content"} dicts. Raises
    dispatch.ToolContractError before the first request when a tool, or one
    of its arguments, would be offered that nothing here executes.
    """
    dispatch.check_contract(tool_definitions)
    messages = [{"role": "system", "content": system}, *history,
                {"role": "user", "content": request}]
    names = tuple(item["function"]["name"] for item in tool_definitions)
    guard = tool_guardrails.ToolCallGuardrailController(
        tool_guardrails.ToolCallGuardrailConfig.from_mapping({}, platform="cli"))
    guard.reset_for_turn()
    state = tools.new_state()
    result = CoreResult("", messages)
    counters = {"length": 0, "truncated_call": 0, "dropped": 0, "invalid": 0,
                "continue": 0}
    empty_nudged = after_tools = False
    compress_at = int((context_window - max_tokens) * 0.5) if context_window else 0

    def finish(stop, final="", error=None):
        result.stop, result.final, result.error = stop, final, error
        return result

    def summarize(stop):
        """Hermes' last word when a bound is reached: one call, no tools."""
        messages.append({"role": "user", "content": text.MAX_ITERATIONS_SUMMARY_REQUEST})
        turn = model(_wire(messages), [], max_tokens)
        final = strip_think_blocks(turn.text or "").strip()
        messages.append(_assistant(final))
        return finish(stop, final)

    while True:
        if cancelled():
            return finish("cancelled")

        if result.iterations >= max_iterations:
            return summarize("iterations")

        if len(result.records) >= max_tool_calls:
            return summarize("tool_budget")

        result.iterations += 1

        if on_model:
            on_model(result.iterations)

        turn = model(_wire(messages), tool_definitions, max_tokens)

        if turn.error:
            return finish("model_error", error=turn.error)

        if (compress_at and turn.usage
                and turn.usage.get("prompt_tokens", 0) >= compress_at):
            return summarize("context")

        content = strip_think_blocks(turn.text or "").strip()
        calls = list(turn.calls)

        if turn.finish_reason == "length":
            if calls:
                counters["truncated_call"] += 1

                if counters["truncated_call"] > TRUNCATED_CALL_RETRIES:
                    return finish("truncated", text.TRUNCATED_RESPONSE)

                continue

            if counters["length"] < LENGTH_CONTINUATIONS:
                counters["length"] += 1
                messages.append(_assistant(content))
                messages.append({"role": "user",
                                 "content": text._LENGTH_CONTINUATION_OUTPUT_LIMIT})
                continue

        if not calls and turn.finish_reason == "tool_calls":
            if counters["dropped"] < DROPPED_CALL_NUDGES:
                counters["dropped"] += 1
                messages.append(_assistant(content))
                messages.append({"role": "user",
                                 "content": text._DROPPED_TOOLCALL_NUDGE_CONTENT})
                continue

        if calls:
            invalid = {}

            for call in calls:
                raw = call["arguments"]

                if raw and raw.strip():
                    try:
                        json.loads(raw)
                    except ValueError as exc:
                        invalid[call["id"]] = str(exc)

            if invalid:
                counters["invalid"] += 1

                if counters["invalid"] <= INVALID_JSON_RETRIES:
                    continue

                counters["invalid"] = 0
                messages.append(_assistant(content, calls))

                for call in calls:
                    body = (text.INVALID_JSON_RESULT.format(err=invalid[call["id"]])
                            if call["id"] in invalid else text.INVALID_JSON_SIBLING_RESULT)
                    messages.append({"role": "tool", "name": call["name"],
                                     "tool_call_id": call["id"], "content": body})

                after_tools = True
                continue

            counters["invalid"] = 0

            # _deduplicate_tool_calls: an exact repeat inside one batch runs once.
            seen, unique = set(), []

            for call in calls:
                key = (call["name"], _canonical_arguments(call["arguments"]))

                if key not in seen:
                    seen.add(key)
                    unique.append(call)

            calls = unique
            messages.append(_assistant(content, calls))
            halt = None

            for call in calls:
                if cancelled():
                    messages.append({"role": "tool", "name": call["name"],
                                     "tool_call_id": call["id"],
                                     "content": "[Command interrupted]"})
                    continue

                arguments = json.loads(call["arguments"] or "{}")

                if not isinstance(arguments, dict):
                    arguments = {}

                blocked = guard.before_call(call["name"], arguments)

                if not blocked.allows_execution:
                    output = tool_guardrails.toolguard_synthetic_result(blocked)
                    halt = halt or blocked
                    record = ToolRecord(call["id"], call["name"], arguments, output,
                                        False, refused=True)
                    host.after_tool(record)
                else:
                    if on_tool:
                        on_tool(call["name"], arguments)

                    output, record = dispatch.execute(host, state, call["id"], call["name"],
                                                      arguments, names)
                    output, decision = _guardrails(guard, call["name"], arguments,
                                                   call["id"], output)
                    halt = halt or decision

                result.records.append(record)
                messages.append({"role": "tool", "name": call["name"],
                                 "tool_call_id": call["id"], "content": output})

            after_tools = True

            if halt is not None:
                tool = halt.tool_name or "a tool"
                return finish("halted", (
                    f"I stopped retrying {tool} because it hit the tool-call guardrail "
                    f"({halt.code}) after {halt.count} repeated non-progressing "
                    "attempts. The last tool result explains the blocker; the next "
                    "step is to change strategy instead of repeating the same call."))

            continue

        if not content and after_tools and not empty_nudged:
            empty_nudged = True
            messages.append(_assistant("(empty)"))
            messages.append({"role": "user", "content": text._EMPTY_TOOL_RESPONSE_NUDGE})
            continue

        if (counters["continue"] < CONTINUE_NUDGES and names
                and text.trailing_continue_intent(content)):
            counters["continue"] += 1
            messages.append(_assistant(content))
            messages.append({"role": "user", "content": text._CODEX_ACK_CONTINUATION_NUDGE})
            continue

        messages.append(_assistant(content))

        return finish("completed", content)
