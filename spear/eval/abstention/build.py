"""Turn the authored items into trainer records, and sample real rejections.

The prompt of a preference example is the whole model-visible context up to the
point of decision -- system, question, the tool calls that were made and what
they returned -- and the two sides differ only in the final assistant message.
That is deliberate: the tool path is already clean, and nothing here should
teach the model to navigate differently.
"""

from __future__ import annotations

import json
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

from pairs import PAIRS
from serve import respond

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


def tool_view():
    """The four normative tools as the model is shown them, from their specs.

    Read from the live registry so the training context cannot drift away from
    the serving one, and without touching any store: only the descriptions and
    schemas are needed.
    """
    from standard_tools import STANDARD_TOOL_NAMES, standard_tool_specs

    return [{"type": "function",
             "function": {"name": spec.name, "description": spec.description,
                          "parameters": spec.input_schema}}
            for spec in sorted(standard_tool_specs(), key=lambda s: s.name)
            if spec.name in STANDARD_TOOL_NAMES]


def prompt_messages(item):
    """System, question, and the evidence as it would have arrived."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": item["user"]}]

    for index, step in enumerate(item["evidence"]):
        call_id = f"call_{item['pair_id'].replace('-', '_')}_{index}"
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": step["tool"],
                                         "arguments": json.dumps(step["args"],
                                                                 sort_keys=True)}}]})
        messages.append({"role": "tool", "tool_call_id": call_id,
                         "content": json.dumps(step["result"], sort_keys=True)})

    return messages


MAX_ROUNDS = 5


def converse(item, messages, tools):
    """Let the scenario answer follow-up calls until the model settles.

    The extra turns become part of the example's prompt, which is what we want:
    the decision we are training is the one taken after looking has stopped
    paying, not the one taken on first sight of a single search hit.
    """
    messages = list(messages)

    for _round in range(MAX_ROUNDS):
        answer, calls, raw = _one(messages, tools)

        if not calls:
            return messages, answer, _round

        messages.append({k: v for k, v in raw.items()
                         if k in ("role", "content", "tool_calls")})

        for call in raw["tool_calls"]:
            name = call["function"]["name"]

            try:
                arguments = json.loads(call["function"].get("arguments") or "{}")
            except ValueError:
                arguments = {}

            messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                             "content": respond(item, name, arguments)})

    return messages, "", MAX_ROUNDS


def _one(messages, tools, timeout=300):
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps({"model": MODEL, "messages": messages, "tools": tools,
                         **INFERENCE}).encode("utf-8"),
        headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]

    calls = raw.get("tool_calls") or []

    return (raw.get("content") or ""), calls, raw


def sample(messages, tools, timeout=300):
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps({"model": MODEL, "messages": messages, "tools": tools,
                         **INFERENCE}).encode("utf-8"),
        headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(request, timeout=timeout) as response:
        choice = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]

    if choice.get("tool_calls"):
        return None, [c["function"]["name"] for c in choice["tool_calls"]]

    return choice.get("content") or "", []


def main():
    tools = tool_view()
    out, started = [], time.time()

    for item in PAIRS:
        settled, answer, rounds = converse(item, prompt_messages(item), tools)
        out.append({"pair_id": item["pair_id"], "class": item["class"],
                    "sampled_answer": answer, "extra_rounds": rounds,
                    "prompt": settled})
        print(f"{item['pair_id']:<12} {item['class'][:22]:<23} "
              f"+{rounds} rounds  "
              f"{len(answer)} chars{'  NO ANSWER' if not answer else ''}",
              flush=True)
        json.dump(out, open(HERE / "synthetic_samples.json", "w"), indent=1)

    print(f"\n{len(out)} sampled in {round(time.time()-started)}s")


if __name__ == "__main__":
    main()
