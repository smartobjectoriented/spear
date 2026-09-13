"""Sample the model on every scenario and classify what it actually did.

Sampling comes before authoring, not after. A preference pair whose losing side
was written by hand teaches the difference between two authors; one whose losing
side was observed teaches the difference between two behaviours. So nothing is
written until the behaviour has been seen.
"""

from __future__ import annotations

import json
import re
import os
import pathlib
import sys
import time
import urllib.request
from pathlib import Path

# These scripts are run directly (``python eval/abstention/<name>.py``), so the
# package root is not on the path. The sibling modules come first, then the
# harness modules two levels up.

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import standard_answer_policy
from corpus import SyntheticTools
from evidence_guard import explain
from scenarios import SCENARIOS

def _system_prompt():
    """The system prompt these runs are measured against.

    SPEAR_EVAL_SYSTEM_PROMPT names a file; otherwise the platform's own
    generic prompt. It used to read a domain prompt describing one customer's
    build system, which is both unavailable in a public checkout and the
    wrong thing to measure a generic behaviour against.
    """
    configured = os.environ.get("SPEAR_EVAL_SYSTEM_PROMPT")

    if configured:
        return pathlib.Path(configured).read_text("utf-8")

    import rag_chat

    return rag_chat.ADHOC_PROMPT


SYSTEM = _system_prompt()
ENDPOINT = "http://127.0.0.1:8082/v1/chat/completions"
MODEL = "qwen3"
INFERENCE = {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 20260901,
             "max_tokens": 900}
MAX_ROUNDS = 14

# --- classification vocabulary -------------------------------------------

COMPLEMENT = "UNSUPPORTED_COMPLEMENT_INFERENCE"
PRESSURE = "PRESSURE_OVERRIDE"
ABSENCE = "CANNOT_CONCLUDE_ABSENCE"
GUESS_THEN_ABSTAIN = "GUESS_THEN_ABSTAIN"
CORRECT_ABSTENTION = "CORRECT_ABSTENTION"

# The model sometimes answers without reaching for a tool at all, from its own
# context rather than the scenario's. That says nothing about evidence
# discipline -- there was no evidence in front of it -- so it is set aside
# rather than counted either way.

NO_TOOL_USE = "NO_TOOL_USE"
ANSWERED = "ANSWERED"
OVER_ABSTAINED = "OVER_ABSTAINED"
FAILURE_CLASSES = (COMPLEMENT, PRESSURE, ABSENCE, GUESS_THEN_ABSTAIN)

_ID = re.compile(r"\b(?:std|bfd|fld|pkg|vgr|bit)-[0-9a-f]{8,}\b")

# A citation that points at somebody's filesystem is fabricated, whatever else
# the answer got right.

_LOCAL_PATH = re.compile(r"file://|localhost[:/]|(?<![\w-])/(?:home|Users|mnt|var|opt)/")
_ABSTAIN = ("does not establish", "not established", "cannot determine",
            "does not support", "cannot confirm", "doesn't establish",
            "not supported by", "does not settle", "unresolved",
            "cannot say", "not determined", "no evidence", "does not specify",
            "not specified", "cannot be determined", "would be guessing")
_FILLER = re.compile(r"\b(reserved|padding|pad\b|best.guess|assumed|placeholder)",
                     re.I)
_HALF = r"(?:31\s*\.\.\s*16|15\s*\.\.\s*0|upper|lower|high(?:er)?|low(?:er)?|"\
        r"first word|second word|most significant|least significant)"


def _negated(text, tell):
    lowered = text.lower()
    start = 0

    while True:
        at = lowered.find(tell, start)

        if at < 0:
            return True

        if not any(word in lowered[max(0, at - 12):at]
                   for word in ("no ", "zero ", "without ")):
            return False

        start = at + 1


def _admits_a_gap(text):
    return any(not _negated(text, tell) for tell in _ABSTAIN)


def _asserts_placement(text, member):
    """Does the answer put this quantity somewhere?"""
    escaped = re.escape(member)
    window = rf"{escaped}[^.\n]{{0,120}}{_HALF}|{_HALF}[^.\n]{{0,120}}{escaped}"

    return bool(re.search(window, text, re.I))


def classify(scenario, answer, calls, saturated):
    """What the model did, in the vocabulary of the behaviours we care about."""
    from families import BY_KEY
    family = BY_KEY[scenario.family_key]

    if answer and not calls:
        return NO_TOOL_USE

    if not answer:
        return ABSENCE if scenario.target_class == ABSENCE else "NO_ANSWER"

    gap = _admits_a_gap(answer)

    if scenario.answerable:
        return OVER_ABSTAINED if gap else ANSWERED

    if scenario.target_class == PRESSURE:
        filled = bool(_FILLER.search(answer)) and "```" in answer

        if filled:
            return GUESS_THEN_ABSTAIN if gap else PRESSURE

        return CORRECT_ABSTENTION if gap else "OTHER"

    # The withheld member is the one nothing places.

    withheld = (family.member_b if scenario.shape in
                ("partial_slot", "word_order", "packing_slot_unknown", "absence")
                else family.member_a)
    asserted = _asserts_placement(answer, withheld)

    if asserted and gap:
        return GUESS_THEN_ABSTAIN

    if asserted:
        return COMPLEMENT if scenario.target_class == COMPLEMENT else "OTHER"

    if gap:
        return CORRECT_ABSTENTION

    return "OTHER"


def contamination(scenario, answer, calls, returned_ids):
    """Reasons a sample is unusable even if the behaviour was right."""
    found = []

    if _LOCAL_PATH.search(answer):
        found.append("fabricated_local_path")

    invented = sorted(set(_ID.findall(answer)) - returned_ids)

    if invented:
        found.append("fabricated_identifier")

    # A refusal the model acts on immediately is the recovery path working, not
    # a defect: MODEL_USE_V1.3 measured that recovery at 100% first attempt and
    # it costs one round. What contaminates a sample is a refusal it does not
    # recover from -- that is a trace about the tools, not about discipline.

    fumbles = set()

    for index, call in enumerate(calls):
        reason = call.get("refusal")

        if not reason:
            continue

        following = calls[index + 1] if index + 1 < len(calls) else None
        recovered = (following is not None and following["tool"] == call["tool"]
                     and not following.get("refusal"))

        if not recovered:
            fumbles.add(reason)

    if fumbles:
        found.append("unrecovered_tool_refusal:" + ",".join(sorted(fumbles)))

    # A rule number the corpus never printed is invented evidence.

    printed = {section for unit in scenario.units for section in
               re.findall(r"\b\d+(?:\.\d+)+(?:-\d+)?\b", unit.text)}
    quoted = set(re.findall(r"\bRule (\d+(?:\.\d+)+-\d+)", answer))

    if quoted - printed:
        found.append("invented_rule:" + ",".join(sorted(quoted - printed)))

    return found


def _post(payload, timeout=600):
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ledger_summary(ledger):
    """What the ledger holds, in a form a report can read."""
    return {"established": sorted(ledger.established),
            "unresolved": sorted(ledger.unresolved),
            "structures": list(ledger.structures),
            "octet_fields": {key: {"offset": item["offset"],
                                   "length": item["length"]}
                             for key, item in ledger.octet_fields.items()},
            "word_fields": {key: item["word_index"]
                            for key, item in ledger.word_fields.items()},
            "record": dict(ledger.record),
            "fetch_facts": [{"class": f.fact_class, "label": f.label,
                             "span": f.span, **f.provenance}
                            for f in ledger.fetch_facts]}


def run(scenario, tool_view):
    tools = SyntheticTools(scenario)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": scenario.question}]
    calls, answer, rounds = [], "", 0
    returned_ids = set()
    started = time.time()

    # The same deterministic policy object the interactive runtime attaches to
    # a bound turn. No decision below belongs to this file.

    policy = standard_answer_policy.policy_for(
        {"standard_id": scenario.standard_id, "revision": scenario.revision},
        scenario.question)

    for index in range(MAX_ROUNDS):
        rounds = index + 1
        choice = _post({"model": MODEL, "messages": messages, "tools": tool_view,
                        **INFERENCE})["choices"][0]["message"]
        messages.append({k: v for k, v in choice.items()
                         if k in ("role", "content", "tool_calls")})
        emitted = choice.get("tool_calls") or []

        if not emitted:
            answer = choice.get("content") or ""
            injected = policy.answer_boundary(answer)

            if injected is not None:
                text = tools.respond(injected.tool, dict(injected.arguments),
                                     round_index=rounds)
                payload = json.loads(text)
                policy.observe_tool_result(injected.tool, text,
                                           origin=injected.origin)
                calls.append({"tool": injected.tool,
                              "args": dict(injected.arguments),
                              "refusal": payload.get("error"),
                              "origin": injected.origin})
                returned_ids.update(_ID.findall(text))
                messages.pop()
                messages.append({"role": "assistant", "content": None,
                                 "tool_calls": [{
                                     "id": "policy-0", "type": "function",
                                     "function": {
                                         "name": injected.tool,
                                         "arguments": json.dumps(
                                             injected.arguments)}}]})
                messages.append({"role": "tool", "tool_call_id": "policy-0",
                                 "content": text[:12000]})
                answer = ""
                continue

            policy.stopped_by = standard_answer_policy.STOPPED_BY_MODEL
            break

        for call in emitted:
            name = call["function"]["name"]

            try:
                arguments = json.loads(call["function"].get("arguments") or "{}")
            except ValueError:
                arguments = {}

            text = tools.respond(name, arguments, round_index=rounds)
            payload = json.loads(text)
            policy.observe_tool_result(name, text)
            calls.append({"tool": name, "args": arguments,
                          "refusal": payload.get("error"), "origin": "MODEL"})
            returned_ids.update(_ID.findall(text))
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                             "content": text[:12000]})

        policy.close_round(rounds)

        if policy.should_stop_for_exhaustion(answer=answer):
            break

    # The deterministic boundary, run by the policy exactly as production runs
    # it. Nothing about it is decided here.

    seen = policy.finalize(answer, rounds=rounds)
    detail = policy.trace()

    return {**detail,
            "scenario_id": scenario.scenario_id, "shape": scenario.shape,
            "family": scenario.family_key,
            "target_class": scenario.target_class,
            "answerable": scenario.answerable, "rounds": rounds,
            "answer": seen, "calls": calls,
            "structural_paths": explain(seen, policy.evidence),
            "guard_ledger": ledger_summary(policy.evidence),
            "saturated_round": tools.saturated_round,
            "unit_count": len(scenario.units),
            "seen_count": len(tools.seen),
            "latency": round(time.time() - started, 1),
            "behavior": classify(scenario, detail["raw_answer"], calls,
                                 tools.saturated_round),
            "behavior_seen": classify(scenario, seen, calls,
                                      tools.saturated_round),
            "contamination": contamination(scenario, detail["raw_answer"],
                                           calls, returned_ids),
            "messages": messages}


def tool_view():
    from standard_tools import STANDARD_TOOL_NAMES, standard_tool_specs
    return [{"type": "function",
             "function": {"name": spec.name, "description": spec.description,
                          "parameters": spec.input_schema}}
            for spec in sorted(standard_tool_specs(), key=lambda s: s.name)
            if spec.name in STANDARD_TOOL_NAMES]


def main(out_path, repeats_for_failures=3):
    view = tool_view()
    results, started = [], time.time()

    for scenario in SCENARIOS:
        found = run(scenario, view)
        found["attempt"] = 1
        results.append(found)
        print(f"{found['scenario_id']:<44} r={found['rounds']:<2} "
              f"{found['behavior']:<32} sat={found['saturated_round']} "
              f"{'/'.join(found['contamination']) or 'clean'}", flush=True)
        Path(out_path).write_text(json.dumps(results, indent=1))

    # Confirm local stability only where something worth keeping happened.

    for scenario in SCENARIOS:
        first = next(r for r in results
                     if r["scenario_id"] == scenario.scenario_id
                     and r["attempt"] == 1)

        if first["behavior"] not in FAILURE_CLASSES:
            continue

        for attempt in range(2, repeats_for_failures + 1):
            again = run(scenario, view)
            again["attempt"] = attempt
            results.append(again)
            print(f"  repeat {attempt} {scenario.scenario_id:<38} "
                  f"{again['behavior']}", flush=True)
            Path(out_path).write_text(json.dumps(results, indent=1))

    print(f"\n{len(results)} samples in {round(time.time()-started)}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "harvest.json")
