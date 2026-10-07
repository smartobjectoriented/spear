"""Labelled benchmark for the skill gate: which skills a turn is offered.

Eligibility and relevance are kept apart. Eligibility is the deterministic
selector's (scope, project, task class) and is never questioned here; the
policies differ only in how they choose among the eligible skills:

    current     the similarity lookup as it runs today: the library's eight
                nearest skills by MiniLM cosine distance, those within 0.55,
                then eligibility, then the two nearest
    relaxed     the same with another distance threshold and cap
    metadata    a skill's declared ``triggers:`` decide when the request uses
                one; the similarity lookup decides only for skills that
                declare none
    all         every eligible skill, no relevance gate

Each case labels the skills relevant to it and those clearly irrelevant; a
skill in neither is neutral. Labels are read by this script only.

Usage: bench.py [--skills DIR] [--cases FILE] [--projects FILE]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import context_selection as cs                      # noqa: E402
import context_sources as src                       # noqa: E402
import skill_library                                # noqa: E402
from workspace_context import from_session         # noqa: E402

LIBRARY_HITS, DISTANCE, CAP = 8, 0.55, 2


def workspace_for(name, workspaces, projects):
    if name in projects:
        spec = projects[name]
        return from_session(project=name, spec=spec, registered=True,
                            project_root=spec["path"], corpus_root=spec["path"])

    registered = bool((workspaces.get(name) or {}).get("registered"))
    root = f"/nonexistent/{name}"

    return from_session(project=name if registered else "",
                        spec=(workspaces.get(name) or {}).get("spec") or {},
                        registered=registered, project_root=root, corpus_root=root)


class MiniLM:
    """The model and metric of the production skills collection."""

    def __init__(self):
        from chromadb.utils import embedding_functions

        self.function = embedding_functions.DefaultEmbeddingFunction()
        self.cache = {}

    def vector(self, text):
        if text not in self.cache:
            self.cache[text] = list(self.function([text])[0])

        return self.cache[text]

    def distance(self, left, right):
        a, b = self.vector(left), self.vector(right)
        dot = sum(x * y for x, y in zip(a, b))
        norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)

        return 1.0 - dot / norm


def triggers(skill):
    """A skill's declared ``triggers:``, kept by skill_library as an unknown key."""
    value = dict(skill.extra).get("triggers", "")

    return tuple(item.strip().strip("'\"") for item in value.strip("[]").split(",")
                 if item.strip())


def triggered(skill, request):
    """Does the request use one of the skill's declared triggers?"""
    words = request.lower()

    return any(re.search(rf"(?<![\w-]){re.escape(term.lower())}(?![\w-])", words)
               for term in triggers(skill))


def evaluate(spec, library, projects, minilm):
    by_id = {f"skill:{skill.name}": skill for skill in library}
    candidates = src.skill_candidates(
        [(skill.name, skill.document, skill.scope if skill.scope_declared else ())
         for skill in library])
    rows = []

    for case in spec["cases"]:
        workspace = workspace_for(case["workspace"], spec.get("workspaces", {}), projects)

        for phase in case["phases"]:
            started = time.perf_counter()
            eligible = [item.item_id for item in cs.DeterministicContextSelector().select(
                workspace, phase, candidates, request=case["request"]).selected]
            distance = {item_id: minilm.distance(case["request"], by_id[item_id].document)
                        for item_id in by_id}
            nearest = sorted(by_id, key=lambda item_id: distance[item_id])
            rows.append({"case": case, "phase": phase, "eligible": eligible,
                         "distance": distance, "nearest": nearest,
                         "ms": (time.perf_counter() - started) * 1000})

    return rows


def choose(row, policy, library, threshold=DISTANCE, cap=CAP):
    by_id = {f"skill:{skill.name}": skill for skill in library}
    eligible, distance, request = row["eligible"], row["distance"], row["case"]["request"]
    ranked = sorted(eligible, key=lambda item_id: distance[item_id])

    if policy == "all":
        chosen = ranked
    elif policy == "metadata":
        chosen = [item_id for item_id in ranked
                  if (triggers(by_id[item_id]) and triggered(by_id[item_id], request))
                  or (not triggers(by_id[item_id]) and distance[item_id] <= threshold)]
    else:
        hits = [item_id for item_id in row["nearest"][:LIBRARY_HITS]
                if distance[item_id] <= threshold]
        chosen = [item_id for item_id in ranked if item_id in hits]

    return chosen if cap is None else chosen[:cap]


def tally(rows, picks, tokens):
    true = false_in = false_out = 0
    selected, mistakes, cost = [], [], 0

    for row, chosen in zip(rows, picks):
        eligible = set(row["eligible"])
        relevant = set(row["case"]["relevant"]) & eligible
        irrelevant = set(row["case"]["irrelevant"]) & eligible
        chosen = set(chosen)
        true += len(chosen & relevant)
        false_in += len(chosen & irrelevant)
        false_out += len(relevant - chosen)
        selected.append(len(chosen))
        cost += sum(tokens[item_id] for item_id in chosen)
        where = f"{row['case']['id']}/{row['phase']}"
        mistakes += [f"{where}: included {item}" for item in sorted(chosen & irrelevant)]
        mistakes += [f"{where}: missed {item}" for item in sorted(relevant - chosen)]

    return {"precision": true / (true + false_in) if true + false_in else 1.0,
            "recall": true / (true + false_out) if true + false_out else 1.0,
            "false_inclusions": false_in, "false_exclusions": false_out,
            "average": sum(selected) / len(selected), "maximum": max(selected),
            "tokens": cost / len(rows), "mistakes": mistakes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skills", default=HERE / "fixture")
    parser.add_argument("--cases", default=HERE / "cases.json")
    parser.add_argument("--projects", default=None)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args(argv)

    spec = json.loads(Path(args.cases).read_text())
    projects = json.loads(Path(args.projects).read_text()) if args.projects else {}
    library = skill_library.load_library(str(args.skills))
    tokens = {f"skill:{skill.name}": cs.estimate_tokens(skill.document) for skill in library}
    rows = evaluate(spec, library, projects, minilm=MiniLM())
    policies = [("current (0.55, cap 2)", "current", DISTANCE, CAP),
                ("current gate, no cap", "current", DISTANCE, None)]
    policies += [(f"relaxed ({threshold}, cap {cap or 'none'})", "current", threshold, cap)
                 for threshold in (0.65, 0.75, 0.85) for cap in (2, 3, None)]
    policies += [("metadata, cap 2", "metadata", DISTANCE, CAP),
                 ("metadata, no cap", "metadata", DISTANCE, None),
                 ("all eligible, cap 2", "all", None, CAP),
                 ("all eligible, no cap", "all", None, None)]

    print(f"{len(spec['cases'])} cases, {len(rows)} selections, {len(library)} skills, "
          f"{sum(1 for skill in library if triggers(skill))} with triggers")

    for label, policy, threshold, cap in policies:
        row = tally(rows, [choose(item, policy, library, threshold, cap) for item in rows], tokens)
        print(f"  {label:28} precision {row['precision']:.3f}  recall {row['recall']:.3f}  "
              f"false+ {row['false_inclusions']}  false- {row['false_exclusions']}  "
              f"skills avg {row['average']:.2f} max {row['maximum']}  "
              f"tokens/turn {row['tokens']:.0f}")

        if args.details or (policy in ("current", "metadata") and cap == CAP
                            and threshold == DISTANCE):
            for mistake in row["mistakes"]:
                print(f"       {mistake}")

    # Rank quality: where the relevant eligible skills sit by distance.
    top = {1: 0, 2: 0, 3: 0}
    relevant_total = 0

    for row in rows:
        ranked = sorted(row["eligible"], key=lambda item_id: row["distance"][item_id])
        relevant = set(row["case"]["relevant"]) & set(row["eligible"])
        relevant_total += len(relevant)

        for k in top:
            top[k] += len(relevant & set(ranked[:k]))

    print("  rank of relevant eligible skills by MiniLM distance: "
          + ", ".join(f"in top-{k} {count}/{relevant_total}" for k, count in top.items()))
    print("  per selection (eligible skill=distance, * relevant, x irrelevant):")

    for row in rows:
        marks = {**{item: "*" for item in row["case"]["relevant"]},
                 **{item: "x" for item in row["case"]["irrelevant"]}}
        shown = ", ".join(f"{item.split(':', 1)[1]}{marks.get(item, '')}={row['distance'][item]:.2f}"
                          for item in sorted(row["eligible"], key=lambda item: row["distance"][item]))
        print(f"    {row['case']['id']}/{row['phase']}: {shown or '(none eligible)'}")

    print(f"  skill tokens: {tokens}")
    print(f"  selection ms (eligibility + distances, warm model): "
          f"median {sorted(r['ms'] for r in rows)[len(rows) // 2]:.1f}")


if __name__ == "__main__":
    main()
