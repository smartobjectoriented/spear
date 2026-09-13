#!/usr/bin/env python3
"""Merge chat corpora into one training/validation pair.

    python scripts/merge_corpora.py --out data/ib2 \\
        data/ib_train.jsonl:data/ib_val.jsonl \\
        data/ib_usage_train.jsonl:data/ib_usage_val.jsonl

Each argument is a `train:val` pair, because the splits were drawn separately
by each builder and must not be reshuffled here: a sample that was held out of
one corpus has to stay held out of the merge, or the validation loss is
measured on something the model has been trained on.

Two things a `cat` would not do:

  - **Interleave, don't concatenate.** Round-robin over the sources, so any
    prefix of the file holds both corpora. Trainer shuffles anyway, but a run
    stopped early, a `--max-steps` smoke test, or a hand-read `head` all see the
    mixture rather than 311 samples of one corpus followed by 66 of another.
  - **Refuse duplicates across sources.** The same question answered twice, by
    two builders that each thought they owned it, teaches whichever answer came
    last. Exact question matches are dropped and reported, never merged
    silently.

The system prompts are deliberately NOT unified. ib.jsonl is French and
ib_usage.jsonl is English, each with its own system prompt, which is what
teaches the model to answer in the language it was addressed in. The report
below lists them so that stays a decision rather than an accident.
"""
import argparse
import json
import unicodedata
import re
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def load(path):
    """Read a chat jsonl, refusing anything that is not one."""
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{number}: not JSON ({exc})")
        messages = row.get("messages")
        if not messages or not isinstance(messages, list):
            raise SystemExit(f"{path}:{number}: no messages list")
        roles = [m.get("role") for m in messages]
        if roles[-1] != "assistant" or "user" not in roles:
            raise SystemExit(f"{path}:{number}: not a user/assistant sample "
                             f"({'+'.join(str(r) for r in roles)})")
        rows.append(row)
    return rows


def question(row):
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def system(row):
    return next((m["content"] for m in row["messages"] if m["role"] == "system"), "")


def normalise(text):
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def interleave(groups):
    """Round-robin, longest source spread evenly through the result."""
    out, index = [], 0
    while any(index < len(g) for g in groups):
        for group in groups:
            if index < len(group):
                out.append(group[index])
        index += 1
    return out


def merge(sources, label):
    groups, seen, dropped = [], {}, []
    for name, rows in sources:
        kept = []
        for row in rows:
            key = normalise(question(row))
            if key in seen:
                dropped.append((name, seen[key], question(row)))
                continue
            seen[key] = name
            kept.append(row)
        groups.append(kept)
        print(f"  {name:34s} {len(rows):4d} {label}")
    merged = interleave(groups)
    for name, first, text in dropped:
        print(f"  ! dropped from {name}: already in {first} — {text[:48]}")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="+", metavar="TRAIN:VAL")
    parser.add_argument("--out", required=True,
                        help="prefix, e.g. data/ib2 -> ib2_train.jsonl + ib2_val.jsonl")
    args = parser.parse_args()

    trains, vals = [], []
    for pair in args.pairs:
        if ":" not in pair:
            raise SystemExit(f"expected TRAIN:VAL, got {pair!r}")
        train_path, val_path = pair.split(":", 1)
        trains.append((Path(train_path).name, load(train_path)))
        vals.append((Path(val_path).name, load(val_path)))

    print("train:")
    merged_train = merge(trains, "samples")
    print("validation:")
    merged_val = merge(vals, "samples")

    # A question in train that is also in val is the one mistake this whole
    # file exists to prevent, and it survives any per-corpus check because
    # neither builder can see the other's splits.
    train_keys = {normalise(question(r)) for r in merged_train}
    leaked = [question(r) for r in merged_val
              if normalise(question(r)) in train_keys]
    if leaked:
        raise SystemExit("these validation questions are also in train:\n  "
                         + "\n  ".join(q[:70] for q in leaked))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix, rows in (("_train.jsonl", merged_train), ("_val.jsonl", merged_val)):
        path = out.with_name(out.name + suffix)
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        chars = sum(len(json.dumps(r, ensure_ascii=False)) for r in rows)
        print(f"\n{path}  ({len(rows)} samples, {chars//1000} k chars)")

    prompts = Counter(system(r) for r in merged_train)
    print(f"\n{len(prompts)} system prompt(s), kept distinct on purpose:")
    for text, count in prompts.most_common():
        print(f"  {count:4d}  {text[:96]}…")
    longest = max(merged_train, key=lambda r: len(json.dumps(r)))
    print(f"\nlongest sample: {len(json.dumps(longest, ensure_ascii=False))} chars"
          f" — check it against FT_MAX_LEN before training")


if __name__ == "__main__":
    main()
