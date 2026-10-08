#!/usr/bin/env python3
"""RETRIEVAL-ONLY eval harness: measures what actually lands in the prompt,
without ever calling the LLM. A few seconds per run.

    ./bin/python eval/run_eval.py --collection NAME --questions FILE [--json out.json]

The question set describes one tree, so it lives with that tree's owner, not
here: SPEAR_EVAL_QUESTIONS and SPEAR_EVAL_COLLECTION supply the defaults.

Metric: recall@k = did the question see AT LEAST ONE of its expected files in
the injected context? We read the `seen` set returned by retrieve_context, so
we measure after truncation to MAX_CONTEXT_CHARS — a chunk that was found but
cut by the budget counts as a failure, which is exactly what the model sees.
"""
import os
import sys
import json
import time
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chromadb
from cli import rag_chat

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    argv = sys.argv[1:]
    coll_name = os.environ.get("SPEAR_EVAL_COLLECTION")
    questions_file = os.environ.get("SPEAR_EVAL_QUESTIONS")
    out_json = None

    if "--collection" in argv:
        coll_name = argv[argv.index("--collection") + 1]

    if "--questions" in argv:
        questions_file = argv[argv.index("--questions") + 1]

    if not coll_name or not questions_file:
        sys.exit("run_eval: name the collection and the question set "
                 "(--collection/--questions or SPEAR_EVAL_COLLECTION/"
                 "SPEAR_EVAL_QUESTIONS)")

    if "--json" in argv:
        out_json = argv[argv.index("--json") + 1]

    questions = json.load(open(questions_file))
    col = chromadb.PersistentClient(path=rag_chat.DB_PATH).get_collection(coll_name)

    rows, timings = [], []

    for e in questions:
        t0 = time.perf_counter()
        ctx, seen = rag_chat.retrieve_context(col, e["q"])
        dt = (time.perf_counter() - t0) * 1000
        timings.append(dt)
        hit = any(f in seen for f in e["expect"])
        rows.append({**e, "hit": hit, "n_files": len(seen), "ms": round(dt, 1)})

    by = collections.defaultdict(list)

    for r in rows:
        by[r["kind"]].append(r["hit"])

    print(f"\ncollection: {coll_name}   TOP_K={rag_chat.TOP_K}   "
          f"MAX_CONTEXT_CHARS={rag_chat.MAX_CONTEXT_CHARS}")
    print(f"{'class':<12} {'recall@k':>9}   {'n':>3}")
    print("-" * 30)

    for kind in ("prose", "identifier", "path"):
        h = by.get(kind, [])

        if h:
            print(f"{kind:<12} {sum(h)/len(h):>8.0%}   {len(h):>3}")

    allh = [r["hit"] for r in rows]
    print("-" * 30)
    print(f"{'GLOBAL':<12} {sum(allh)/len(allh):>8.0%}   {len(allh):>3}")
    print(f"\nlatency: {sum(timings)/len(timings):.0f} ms/query "
          f"(max {max(timings):.0f} ms)")

    misses = [r for r in rows if not r["hit"]]

    if misses:
        print(f"\nmisses ({len(misses)}):")

        for r in misses:
            print(f"  [{r['kind']:<10}] {r['q'][:52]:<54} -> {r['expect'][0]}")

    if out_json:
        json.dump(rows, open(out_json, "w"), ensure_ascii=False, indent=1)
        print(f"\ndetails -> {out_json}")


if __name__ == "__main__":
    main()
