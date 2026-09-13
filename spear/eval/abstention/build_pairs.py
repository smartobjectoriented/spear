"""Assemble the DPO records from the specs and the private traces.

The prompt is taken from the observed trace, not rebuilt, so the winning and
losing answers are answering the same context down to the byte -- including the
tool calls the model actually made and what came back. Only the last assistant
message differs.

Every negative is audited for entailment on the way through. A pair whose
answer the evidence already settles never reaches the file.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Mapping

# These scripts are run directly (``python eval/abstention/<name>.py``), so the
# package root is not on the path. The sibling modules come first, then the
# harness modules two levels up.

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import complement
import pressure
import shapes
from entailment import NOT_ENTAILED, audit, facts_for
from families import BY_KEY
from harvest import tool_view
from pair_spec import AUTHORED, OBSERVED, SPECS, SUPPORTED
from pair_spec_supported import SUPPORTED_SPECS
from state_paths import state_dir

STATE = state_dir()
TRACES = {"FT0.1": STATE / "ft0.1/harvest.json",
          "FT0.1P": STATE / "ft0.1p/harvest.json",
          "FT0.1C": STATE / "ft0.1c/harvest_v2.json"}

# The four keys a DPO record is read through. Everything else in an audited
# pair is provenance for people, not input for a trainer.

TRAINER_FIELDS = ("prompt", "chosen", "rejected", "tools")
# What Axolotl's DPO field-mapping strategy reads: three strings, nothing else.
AXOLOTL_FIELDS = ("prompt", "chosen", "rejected")

# The strategy renders an assistant turn on its own by pairing it with a
# throwaway user turn and slicing the result from the content onward.
# Reproduced rather than approximated: the token sequence the trainer sees has
# to be the one the template would have produced.

_DUMMY_TURN = {"role": "user", "content": "[[dummy_message]]"}

BUILDERS = {"shapes.partial_slot": shapes.partial_slot,
            "shapes.partial_slot_supported": shapes.partial_slot_supported,
            "shapes.absence": shapes.absence,
            "shapes.word_assignment_unknown": shapes.word_assignment_unknown,
            "pressure.whole_word": pressure.whole_word,
            "pressure.enum_gap": pressure.enum_gap,
            "pressure.field_to_word": pressure.field_to_word,
            "pressure.reserved_region": pressure.reserved_region,
            "complement.byte_offset": complement.byte_offset,
            "complement.two_word": complement.two_word}


def trainer_arguments(arguments):
    """Tool-call arguments in the form the chat template reads.

    The evaluation harness records them as a JSON string, which is what the
    OpenAI wire format uses and what the harness reads back. The Qwen chat
    template iterates them with .items() and raises on a string. Both are right
    for their own consumer, so the conversion lives here, at the boundary
    between them, and not in the harness or the audited pair files.

    A string that is not a JSON object is refused rather than repaired --
    guessing what a broken argument meant would put invented evidence into
    training data.
    """

    if isinstance(arguments, Mapping):
        return dict(arguments)

    if not isinstance(arguments, str):
        raise ValueError(f"tool arguments must be a string or mapping, "
                         f"got {type(arguments).__name__}")

    text = arguments.strip()

    if not text:
        return {}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tool arguments are not valid JSON: {exc}") from None

    if not isinstance(parsed, Mapping):
        raise ValueError(f"tool arguments must decode to an object, got "
                         f"{type(parsed).__name__}")

    return dict(parsed)


def trainer_record(record):
    """One audited pair as the trainer reads it: four keys, mapping arguments.

    Nothing else is touched -- not the call id, the tool name, the ordering,
    the tool results, nor either answer.
    """
    found = {key: copy.deepcopy(record[key]) for key in TRAINER_FIELDS
             if key in record}

    for message in found.get("prompt", ()):
        for call in message.get("tool_calls") or ():
            function = call.get("function")

            if isinstance(function, Mapping) and "arguments" in function:
                function["arguments"] = trainer_arguments(function["arguments"])

    return found


def render_completion(turn, tokenizer):
    """One assistant turn as the trainer wants it: content plus its terminator.

    No header and no generation marker -- the prompt already ends with one, and
    emitting a second would put the model at a boundary it never sees at
    serving time.
    """
    content = turn.get("content")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("an assistant target must carry non-empty text")

    text = tokenizer.apply_chat_template([_DUMMY_TURN, turn],
                                         add_generation_prompt=False,
                                         tokenize=False)
    at = text.find(content)

    if at < 0:
        raise ValueError("the chat template did not reproduce the answer text")

    return text[at:].rstrip()


def render_dpo_strings(record, tokenizer):
    """The Axolotl DPO view: a shared prompt and two competing continuations.

    Axolotl's field-mapping strategy formats these three fields as strings
    before TRL ever sees them, so a conversational record reaches the tokenizer
    as the repr of a list and is rejected. The decomposition matters as much as
    the type: the prompt carries the whole shared prefix and stops where the
    assistant is expected to speak, and each completion carries only its own
    answer. Repeating the prefix inside a completion would train the model on
    a conversation it never has.
    """
    prompt = list(record["prompt"])

    for side in AXOLOTL_FIELDS[1:]:
        turns = record[side]

        if len(turns) != 1 or turns[0].get("role") != "assistant":
            raise ValueError(f"{side} must be exactly one assistant target")

    if not prompt or (prompt[-1].get("role") == "assistant"
                      and not prompt[-1].get("tool_calls")):
        raise ValueError("the shared prefix must not already end in an answer")

    rendered = tokenizer.apply_chat_template(
        prompt, tools=record.get("tools") or None,
        add_generation_prompt=True, tokenize=False)

    return {"prompt": rendered,
            "chosen": render_completion(record["chosen"][0], tokenizer),
            "rejected": render_completion(record["rejected"][0], tokenizer)}


def conversational_reference(record, side, tokenizer):
    """The whole conversation with one answer attached, for equivalence checks."""
    return tokenizer.apply_chat_template(
        list(record["prompt"]) + list(record[side]),
        tools=record.get("tools") or None, tokenize=False)


def _traces():
    found = {}

    for phase, path in TRACES.items():
        for row in json.loads(path.read_text()):
            if row["attempt"] == 1:
                found[(phase, row["scenario_id"])] = row

            found.setdefault((phase, row["scenario_id"]), row)

    return found


def _scenario(spec):
    maker = BUILDERS[spec.builder]
    args = dict(spec.builder_args)
    family = BY_KEY[args.pop("family")]
    strength = args.pop("strength", None)

    if strength is not None:
        return maker(family, getattr(pressure, strength), **args)

    return maker(family, **args)


def _reproductions(rows, phase, sample_id, behavior):
    return sum(1 for row in json.loads(TRACES[phase].read_text())
               if row["scenario_id"] == sample_id and row["behavior"] == behavior)


def build():
    traces = _traces()
    records, problems = [], []

    for spec in list(SPECS) + list(SUPPORTED_SPECS):
        trace = traces.get((spec.source_phase, spec.sample_id))

        if trace is None:
            problems.append(f"{spec.pair_id}: no trace for {spec.sample_id}")
            continue

        scenario = _scenario(spec)

        if spec.failure_class == SUPPORTED:
            checked, result = True, "NOT_APPLICABLE"
            why = "control: the evidence answers the question"
        else:
            result, why = audit(facts_for(scenario, **spec.audit_args),
                                " ".join(u.text for u in scenario.units))
            checked = True

            if result != NOT_ENTAILED:
                problems.append(f"{spec.pair_id}: entailment {result} -- {why}")
                continue

        messages = trace["messages"]

        if not messages or messages[-1].get("role") != "assistant":
            problems.append(f"{spec.pair_id}: trace does not end in an answer")
            continue

        prompt = messages[:-1]
        rejected = (spec.rejected_text.strip() if spec.rejected_origin == AUTHORED
                    else (trace["answer"] or "").strip())

        if not rejected:
            problems.append(f"{spec.pair_id}: no rejected text")
            continue

        records.append({
            "pair_id": spec.pair_id, "class": spec.failure_class,
            "semantic_family": spec.family, "structural_shape": spec.shape,
            "source_phase": spec.source_phase, "sample_id": spec.sample_id,
            "rejected_origin": spec.rejected_origin,
            "reproductions": _reproductions(traces, spec.source_phase,
                                            spec.sample_id, trace["behavior"]),
            "entailment_checked": checked, "entailment_result": result,
            "entailment_reason": why,
            "tool_path_clean": not trace.get("contamination"),
            "train_or_eval": spec.split, "licensed_content": False,
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": spec.chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "tools": tool_view(),
        })

    return records, problems


def load_tokenizer():
    """The pinned tokenizer, offline. The rendering has to be the real one."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-Coder-Next",
        revision="a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb")


def main(out_dir, tokenizer=None):
    tokenizer = tokenizer or load_tokenizer()
    records, problems = build()

    for problem in problems:
        print("EXCLUDED:", problem)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split, name in (("train", "preference_train.jsonl"),
                        ("eval", "preference_eval.jsonl")):
        chosen = [r for r in records if r["train_or_eval"] == split]

        with (out / name).open("w") as handle:
            for record in chosen:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

        print(f"{len(chosen):>3} pairs -> {out / name}")

        # The trainer-facing view of the same pairs, written by the builder
        # rather than by hand so the two cannot drift apart -- and rendered to
        # strings, because that is what the installed DPO strategy reads.

        view = out / name.replace(".jsonl", "_trainer.jsonl")

        with view.open("w") as handle:
            for record in chosen:
                handle.write(json.dumps(
                    render_dpo_strings(trainer_record(record), tokenizer),
                    sort_keys=True) + "\n")

        print(f"{len(chosen):>3} pairs -> {view}")

    return records


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(STATE / "ft0.2"))
