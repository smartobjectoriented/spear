"""MODEL_USE_V1 harness: drive the real model against the frozen normative tools.

Generic on purpose. It knows how to run a question through the live tool loop
and record everything that happened; it knows nothing about VITA. The cases and
their ground truth live beside it and stay private, because they quote the
document being evaluated.

The deterministic policy is NOT implemented here. Every decision -- when to
bootstrap, when to recover, what the evidence establishes, what the finished
answer may say and cite -- belongs to `standard_answer_policy`, the same object
the interactive runtime drives. This file only runs the loop and records it, so
what the battery measures is what production executes.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import rag_chat
import standard_answer_policy
from standard_tools import STANDARD_TOOL_NAMES, StandardToolService
from tool_registry import ToolRegistry
from tool_router import ToolExecutionContext
from tracing import NullTraceRecorder, TraceEmitter

SID, REV = "ANSI-VITA-49.2", "2017-R2024"
ENDPOINT = "http://127.0.0.1:8082/v1/chat/completions"
MODEL = "qwen3"
# Deterministic as far as llama-server allows.
INFERENCE = {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 20260901,
             "max_tokens": 900}
MAX_ROUNDS = 14


def build_registry():
    store = rag_chat.STANDARD_STORE
    registry = ToolRegistry()
    StandardToolService(store).register(registry)
    binding = store.binding(SID, REV).to_dict()
    return registry, binding


def tool_specs(registry):
    """The four normative tools, in the OpenAI function shape the model sees."""
    found = []
    for spec in sorted(registry.list_specs(), key=lambda s: s.name):
        if spec.name not in STANDARD_TOOL_NAMES:
            continue
        found.append({"type": "function",
                      "function": {"name": spec.name,
                                   "description": spec.description,
                                   "parameters": spec.input_schema}})
    return found


def _post(payload, timeout=600):
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run_case(question, registry, binding, *, system=None,
             max_rounds=MAX_ROUNDS):
    """One question through the real loop. Returns the whole trace."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    specs = tool_specs(registry)
    context = ToolExecutionContext(
        "task_modeluse01", TraceEmitter(NullTraceRecorder()), {},
        metadata={"standard_binding": binding})
    trace = {"question": question, "calls": [], "answer": "", "rounds": 0,
             "error": None, "sources_returned": [], "latency_s": 0.0}

    # The same policy object the interactive runtime attaches to a bound turn.
    policy = standard_answer_policy.policy_for(binding, question)
    started = time.time()
    try:
        # The standard is read before the first round, not after the last one.
        # Driven here too, because a policy the evaluation does not run is a
        # policy the battery does not measure.
        opening = policy.opening() if policy is not None else None

        if opening is not None:
            record = _execute(registry, context, opening.tool,
                              dict(opening.arguments), origin=opening.origin)
            policy.observe_tool_result(opening.tool, record["result"],
                                       origin=opening.origin)
            trace["calls"].append(record)
            messages.append({"role": "assistant", "content": None,
                             "tool_calls": [{
                                 "id": "policy-opening", "type": "function",
                                 "function": {
                                     "name": opening.tool,
                                     "arguments": json.dumps(
                                         opening.arguments)}}]})
            messages.append({"role": "tool",
                             "tool_call_id": "policy-opening",
                             "content": record["result"][:12000]})

        for _round in range(max_rounds):
            trace["rounds"] += 1
            reply = _post({"model": MODEL, "messages": messages,
                           "tools": specs, **INFERENCE})
            choice = reply["choices"][0]["message"]
            messages.append({k: v for k, v in choice.items()
                             if k in ("role", "content", "tool_calls")})
            calls = choice.get("tool_calls") or []
            if not calls:
                answer = choice.get("content") or ""

                # One deterministic retrieval, decided by the policy, when the
                # turn is about to conclude without the evidence to do it.
                injected = policy.answer_boundary(answer)

                if injected is not None:
                    record = _execute(registry, context, injected.tool,
                                      dict(injected.arguments),
                                      origin=injected.origin)
                    policy.observe_tool_result(injected.tool, record["result"],
                                               origin=injected.origin)
                    trace["calls"].append(record)
                    messages.pop()      # the un-evidenced answer never stands
                    messages.append({"role": "assistant", "content": None,
                                     "tool_calls": [{
                                         "id": "policy-0", "type": "function",
                                         "function": {
                                             "name": injected.tool,
                                             "arguments": json.dumps(
                                                 injected.arguments)}}]})
                    messages.append({"role": "tool",
                                     "tool_call_id": "policy-0",
                                     "content": record["result"][:12000]})
                    continue

                trace["answer"] = answer
                policy.stopped_by = standard_answer_policy.STOPPED_BY_MODEL
                break

            for call in calls:
                name = call["function"]["name"]
                raw = call["function"].get("arguments") or "{}"
                try:
                    arguments = json.loads(raw)
                except ValueError:
                    arguments = {"__unparsable__": raw}
                record = _execute(registry, context, name, arguments)
                policy.observe_tool_result(name, record["result"])
                policy.observe_code_read(name, arguments, record["result"])
                trace["calls"].append(record)
                for value in record["metadata"].get("standard_source_ids", ()):
                    if value not in trace["sources_returned"]:
                        trace["sources_returned"].append(value)
                try:
                    payload = json.loads(record["result"])
                except ValueError:
                    payload = None
                for value in _harvest_sources(payload):
                    if value not in trace["sources_returned"]:
                        trace["sources_returned"].append(value)
                messages.append({"role": "tool",
                                 "tool_call_id": call.get("id", ""),
                                 "content": record["result"][:12000]})

            policy.close_round(trace["rounds"])
        else:
            trace["error"] = "MAX_ROUNDS"
            policy.stopped_by = standard_answer_policy.STOPPED_BY_BUDGET
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        trace["error"] = f"{type(exc).__name__}: {exc}"
    # The full model-visible context, so a preference example can reuse the
    # exact prompt the completion was produced under.
    trace["messages"] = messages
    trace["latency_s"] = round(time.time() - started, 1)

    # The deterministic boundary, run by the policy exactly as production runs
    # it. Nothing about it is decided here.
    trace["answer"] = policy.finalize(trace["answer"], rounds=trace["rounds"])
    trace.update(policy.trace())
    trace["guard_ledger"] = _ledger_view(policy.evidence)

    return trace


def _execute(registry, context, name, arguments, *, origin="MODEL"):
    record = {"tool": name, "arguments": arguments, "ok": False, "result": "",
              "metadata": {}, "origin": origin}
    try:
        handler = registry.handler(name)
    except Exception as exc:
        record["result"] = f"NO SUCH TOOL: {exc}"
        record["unknown_tool"] = True
        return record
    try:
        outcome = handler(context, arguments)
        record["ok"] = True
        record["result"] = outcome.text
        record["metadata"] = dict(outcome.metadata)
        record["mutation"] = outcome.mutation
        record["affected_paths"] = list(outcome.affected_paths)
    except Exception as exc:
        record["result"] = f"{type(exc).__name__}: {exc}"
    return record


def _ledger_view(ledger):
    return {"established": sorted(ledger.established),
            "unresolved": sorted(ledger.unresolved),
            "structures": list(ledger.structures),
            "octet_fields": {key: {"offset": item["offset"],
                                   "length": item["length"]}
                             for key, item in ledger.octet_fields.items()},
            "word_fields": {key: item["word_index"]
                            for key, item in ledger.word_fields.items()},
            "record": dict(ledger.record),
            "cell_fields": {key: {"label": item["label"], "msb": item["msb"],
                                  "lsb": item["lsb"],
                                  "page": item["provenance"]["page"]}
                            for key, item in ledger.cell_fields.items()}}


def _harvest_sources(payload, found=None):
    """Every std-… id a tool actually returned, at any depth."""
    found = [] if found is None else found
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str) and value.startswith("std-"):
                found.append(value)
            else:
                _harvest_sources(value, found)
    elif isinstance(payload, list):
        for value in payload:
            if isinstance(value, str) and value.startswith("std-"):
                found.append(value)
            else:
                _harvest_sources(value, found)
    return found
