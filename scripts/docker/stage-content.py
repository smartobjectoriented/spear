#!/usr/bin/env python3
"""Stage the rules, skills and benches an image may carry: what its corpora admit.

A deployment's content tree serves every corpus on the host, and an image that
carries a few of them must not carry the rules written for the others: a rule
scoped to a corpus names that corpus, its build and its conventions. What
travels is read the way the harness reads it -- a rule's `scope:` header, a
skill's `scope:` front matter -- and matched against what the image carries:
the baked corpora and the families they declare.

    stage-content.py --bake a,b [--registry FILE] --rules DIR --skills DIR \
        --benches DIR OUT

Kept: rules declared global, rules and skills scoped to a baked corpus or to a
family a baked corpus declares, skills declared `any`, the orientation map of
each baked corpus, the bench each baked corpus names. A rule or skill that
declares no scope applies nowhere in a session either, and is left out for
the same reason; a rule scoped to a host path names a directory the image
does not have. Files that are not rules or skills (a README) travel as they
are.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil

REPO = pathlib.Path(__file__).resolve().parents[2]
FAMILY = "family:"
_HEADER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.S)


def header(text: str) -> dict[str, str] | None:
    match = _HEADER.match(text)

    if not match:
        return None

    return dict(re.findall(r"^(\w+):\s*(.*?)\s*$", match.group(1), re.M))


def values(text: str) -> tuple[str, ...]:
    return tuple(v for v in re.split(r"[,\s\[\]]+", text or "") if v)


def rule_verdict(text: str, corpora: set[str], families: set[str]) -> tuple[bool, str]:
    """(kept, why) for a rules.d file, on its scope header alone."""
    fields = header(text)

    if fields is None or not fields.get("scope"):
        return False, "declares no scope"

    kind, _, rest = fields["scope"].strip().partition(" ")
    names = values(rest)

    if kind == "global" and not names:
        return True, "global"

    if kind == "corpus":
        hit = sorted(set(names) & corpora)
        return bool(hit), f"corpus {', '.join(hit or names)}"

    if kind == "family":
        hit = sorted(set(n.casefold() for n in names) & families)
        return bool(hit), f"family {', '.join(hit or names)}"

    if kind == "path":
        return False, "scoped to a host path"

    return False, "its scope header cannot be read"


def skill_verdict(text: str, corpora: set[str], families: set[str]) -> tuple[bool, str]:
    """(kept, why) for a skill file, on its declared scope alone."""
    fields = header(text) or {}
    scope = values(fields.get("scope", ""))

    if not scope:
        return False, "declares no scope"

    if "any" in scope:
        return True, "any"

    fams = {s[len(FAMILY):].casefold() for s in scope if s.startswith(FAMILY)}
    names = {s for s in scope if not s.startswith(FAMILY)}
    hit = sorted(names & corpora) + sorted(FAMILY + f for f in fams & families)

    return bool(hit), ", ".join(hit or scope)


def stage_dir(src: pathlib.Path, dst: pathlib.Path, label: str, verdict,
              corpora: set[str], families: set[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)

    if not src.is_dir():
        return

    for path in sorted(src.iterdir()):
        if path.is_dir():
            continue

        if path.suffix != ".md" or path.stem.upper() == "README":
            shutil.copy2(path, dst / path.name)
            continue

        kept, why = verdict(path.read_text(encoding="utf-8", errors="replace"),
                            corpora, families)

        if kept:
            shutil.copy2(path, dst / path.name)
            print(f"   {label:<8} {path.name}  ({why})")
        else:
            print(f"   {label:<8} {path.name}  LEFT OUT — {why}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", default="", help="comma-separated corpus names")
    ap.add_argument("--registry", default=str(REPO / "docker" / "projects.docker.json"))
    ap.add_argument("--rules", default="")
    ap.add_argument("--skills", default="")
    ap.add_argument("--benches", default="")
    ap.add_argument("out")
    args = ap.parse_args()

    out = pathlib.Path(args.out)

    if out.exists():
        shutil.rmtree(out)

    registry = json.loads(pathlib.Path(args.registry).read_text())
    corpora = {n for n in args.bake.split(",") if n.strip()}
    specs = {n: registry.get(n) or {} for n in corpora}
    families = set()

    for spec in specs.values():
        raw = spec.get("families") or ()
        families |= {str(f).casefold() for f in ((raw,) if isinstance(raw, str) else raw)}

    rules, skills, benches = (pathlib.Path(p) if p else None
                              for p in (args.rules, args.skills, args.benches))

    if rules:
        stage_dir(rules, out / "rules.d", "rules", rule_verdict, corpora, families)
        maps = rules / "corpora"

        if maps.is_dir():
            (out / "rules.d" / "corpora").mkdir(parents=True)

            for path in sorted(maps.glob("*.md")):
                if path.stem in corpora:
                    shutil.copy2(path, out / "rules.d" / "corpora" / path.name)
                    print(f"   rules    corpora/{path.name}  (map of a baked corpus)")
                else:
                    print(f"   rules    corpora/{path.name}  LEFT OUT — not a baked corpus")

    if skills:
        stage_dir(skills, out / "skills", "skills", skill_verdict, corpora, families)

    if benches:
        (out / "benches").mkdir(parents=True, exist_ok=True)
        named = {spec["bench"]: name for name, spec in specs.items() if spec.get("bench")}

        for path in sorted(benches.iterdir()) if benches.is_dir() else []:
            if path.is_dir():
                continue

            if path.name in named:
                shutil.copy2(path, out / "benches" / path.name)
                print(f"   benches  {path.name}  (bench of {named[path.name]})")
            elif path.suffix in (".md", ".txt", ""):
                shutil.copy2(path, out / "benches" / path.name)
            else:
                print(f"   benches  {path.name}  LEFT OUT — no baked corpus names it")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
