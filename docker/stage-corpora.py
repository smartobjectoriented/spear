#!/usr/bin/env python3
"""Copy selected corpus trees into the image, where the registry expects them.

The image mounts working copies by default, and that is right for a
workstation: a tree baked into an image is stale the next morning. It is
wrong for the one case this exists for -- handing the container to someone
who has none of those trees. There, an empty /corpora is not a stale corpus,
it is no corpus, and the session answers by citing files that are not there.

So a build names what to bake. Nothing is baked by default, and a mount at
/corpora still shadows whatever was baked, so the workstation keeps behaving
exactly as it did.

    stage-corpora.py --bake so3,so3-doc,u-boot  OUT

Where each tree lands is not decided here: it is read from the registry
gen-registry.py has just written, which is the same mapping the container
resolves paths through. Deriving it a second time is how the two would come
to disagree.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent

#: Never copied into an image. Build products and history, not sources: a
#: .git alone routinely outweighs the tree it belongs to.
PRUNE = (".git", "__pycache__", "node_modules", ".venv", "venv")

#: What the Dockerfile copies out of the repository itself, whatever a build
#: chose to bake. Kept in step with the COPY lines there by hand, because the
#: alternative is a registry that promises two corpora the image deliberately
#: does not carry -- llama.cpp-next and qwen3-finetune are gigabytes of venv
#: and build artefacts, excluded on purpose, and every container then opened
#: with two "missing" lines that were never going to be anything else.
FROM_REPO = ("spear/claude", "spear/corpora")


def registries():
    host = json.loads((REPO / "spear" / "projects.json").read_text())
    image = json.loads((HERE / "projects.docker.json").read_text())
    return host, image


def human(n: int) -> str:
    return f"{n / 1024 / 1024:.0f} MB" if n else "0 MB"


def copy(src: pathlib.Path, dst: pathlib.Path) -> int:
    """Copy a tree without its build products. Returns the bytes written."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, symlinks=True,
                    ignore=shutil.ignore_patterns(*PRUNE))
    out = subprocess.run(["du", "-sb", str(dst)], capture_output=True, text=True)
    return int(out.stdout.split()[0]) if out.stdout.split() else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", default="", help="comma-separated corpus names")
    ap.add_argument("--restrict-registry", action="store_true",
                    help="drop from the image registry every corpus that was "
                         "neither baked nor carried by the repository itself")
    ap.add_argument("out")
    args = ap.parse_args()

    out = pathlib.Path(args.out)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    wanted = [n for n in args.bake.split(",") if n.strip()]

    if not wanted:
        print("   corpora  (none baked — the container expects them mounted)")
        return 0

    host, image = registries()
    total, missing = 0, []

    for name in wanted:
        spec, placed = host.get(name), image.get(name)

        if not spec or not placed:
            missing.append(name)
            continue

        src = pathlib.Path(os.path.expanduser(spec["path"]))

        if not src.is_dir():
            missing.append(name)
            continue

        written = copy(src, out / placed["path"])
        total += written
        print(f"   corpora  {name:<16} -> /corpora/{placed['path']}  ({human(written)})")

    for name in missing:
        print(f"   corpora  {name:<16} NOT baked — unknown or absent on this host",
              file=sys.stderr)

    print(f"   corpora  {human(total)} baked into the image")

    # A registry describing the BUILD HOST's corpora, in an image that carries
    # three of them, greets the recipient with twenty-two "missing" lines
    # before the first question. What a handed-over image registers should be
    # what it has.
    #
    # Corpora carried by the repository itself stay: they are in the image
    # whatever was baked. Anything else the recipient mounts, they register.
    if args.restrict_registry:
        keep = set(wanted) | {
            n for n, spec in image.items()
            if str(spec.get("path", "")).startswith(FROM_REPO)}
        dropped = sorted(set(image) - keep)
        kept = {n: spec for n, spec in image.items() if n in keep}
        (HERE / "projects.docker.json").write_text(
            json.dumps(kept, indent=2, ensure_ascii=False) + "\n")
        print(f"   corpora  registry restricted to {len(kept)} corpora "
              f"({len(dropped)} dropped: not in this image)")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
