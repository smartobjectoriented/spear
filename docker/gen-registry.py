#!/usr/bin/env python3
"""Derive docker/projects.docker.json from the host registry.

Run by build.sh, not by hand. Its output is BUILD OUTPUT and is gitignored:
this file must be regenerated whenever projects.json changes, and a committed
copy would be both stale -- a corpus added on the host would simply be absent
from the image, and the entrypoint would report it missing without anyone
knowing why -- and a publication of one machine's corpus graph, with its
project names, its federations and its collection hashes.

The repository therefore carries exactly one registry by hand, and it is an
example: spear/projects.example.json. There is no second one to keep in step.

Two rewrites happen here, and both are about things that move:

  paths       Absolute host paths become paths relative to the container's
              mount root; a corpus already relative to the repository gets the
              `spear/` prefix, because that is where the repository is
              mounted inside /corpora.

  collections The chroma collection is named EXPLICITLY, computed from the
              resolved HOST path. The derivation is an md5 of the absolute
              path, so under /corpora the same tree hashes differently and the
              shipped index goes unfound -- the container starts cleanly with
              no retrieval at all, which is the whole point of the tool.
"""
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
HOST_ROOTS = [(os.path.expanduser("~") + "/", ""), ("/opt/llm/", "")]


def container_path(raw):
    """Where this corpus is mounted under /corpora, or None if unreachable."""
    if not os.path.isabs(raw):
        return f"spear/{raw}"                  # inside the repository
    for prefix, replacement in HOST_ROOTS:
        if raw.startswith(prefix):
            return replacement + raw[len(prefix):]
    return None


def collection_of(spec, resolved):
    """The same rule the platform uses: a named index, or one derived from
    the tree. The kind decides nothing -- it used to derive a second naming
    scheme here, which meant this generator and rag_chat could disagree about
    which collection a corpus had."""
    if spec.get("collection"):
        return spec["collection"]
    return "adhoc_" + hashlib.md5(os.path.realpath(resolved).encode()).hexdigest()[:8]


def host_registry():
    """What this machine has registered, or nothing.

    projects.json is per-machine and gitignored, so a clean clone does not have
    one -- and the example beside it is a template full of `/path/to/...`
    placeholders, which would bake fictional corpora into the image. An empty
    registry is the honest answer: the image carries the harness and whatever
    index the host had, and the operator registers their own trees.
    """
    live = os.path.join(REPO, "spear", "projects.json")

    if not os.path.isfile(live):
        print("   registry: no spear/projects.json — image ships no corpora",
              file=sys.stderr)
        return {}

    return json.load(open(live))


def main():
    src = host_registry()
    out, skipped = {}, []
    for name, spec in src.items():
        if not isinstance(spec, dict):
            continue                           # the template's own _comment

        raw = spec["path"]
        rel = container_path(raw)
        if rel is None:
            skipped.append((name, raw))
            continue
        resolved = raw if os.path.isabs(raw) else os.path.join(REPO, raw)
        entry = dict(spec)
        entry["path"] = rel
        entry["collection"] = collection_of(spec, resolved)
        out[name] = entry

    target = os.path.join(REPO, "docker", "projects.docker.json")
    with open(target, "w") as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"   registry: {len(out)} corpora -> docker/projects.docker.json")
    for name, raw in skipped:
        print(f"   SKIPPED {name}: {raw} is under no known root", file=sys.stderr)
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
