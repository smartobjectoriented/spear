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

What of a tree is copied depends on what the tree is. A tree git tracks is
copied as git has it: the tracked files, and none of the build output, the
attached components or the snapshots beside them -- a checkout with its build
products is several times its own size. A tree git does not track is a
fetched component, and is copied whole but for its build output. Either way
the directories the registry excludes from indexing are left out of the image
too: they are the vendored trees that are corpora of their own, and the
generated ones. A corpus that lives inside another is baked at its own path,
so naming both bakes both, each by its own rule.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
#: Where the image definition lives, and the registry build.sh generates.
HERE = REPO / "docker"

#: Never copied into an image. Build products and history, not sources: a
#: .git alone routinely outweighs the tree it belongs to.
PRUNE = (".git", "__pycache__", "node_modules", ".venv", "venv")

#: What a fetched component's build leaves behind, by directory name and by
#: file suffix. A tracked tree needs none of this: git already says. A
#: `<tree>.back` is the previous attached tree, kept by a re-attach: a stale
#: snapshot that is never a source, wherever it sits.
BUILD_DIRS = ("build", "out", ".pristine")
SNAPSHOT_SUFFIX = ".back"
BUILD_SUFFIXES = (".o", ".a", ".so", ".ko", ".elf", ".bin", ".img", ".itb",
                  ".dtb", ".cpio", ".cpio.gz", ".tar", ".zst", ".xz")

#: What the Dockerfile copies out of the repository itself, whatever a build
#: chose to bake. Kept in step with the COPY lines there by hand, because the
#: alternative is a registry that promises two corpora the image deliberately
#: does not carry -- llama.cpp-next and qwen3-finetune are gigabytes of venv
#: and build artefacts, excluded on purpose, and every container then opened
#: with two "missing" lines that were never going to be anything else.
FROM_REPO = ("spear/claude", "spear/corpora")


def registries():
    host = json.loads((REPO / "client" / "projects.json").read_text())
    image = json.loads((HERE / "projects.docker.json").read_text())
    return host, image


def human(n: int) -> str:
    return f"{n / 1024 / 1024:.0f} MB" if n else "0 MB"


def tracked_files(src: pathlib.Path) -> list[str] | None:
    """The files git tracks under src, relative to it, or None for a tree git
    does not track: a fetched component, or a tree outside any repository."""
    try:
        out = subprocess.run(["git", "-C", str(src), "ls-files", "-z", "--", "."],
                             capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None

    files = [f for f in out.stdout.decode().split("\0") if f]
    return files or None


class Excluder:
    """The registry's exclusions, with the indexer's reading: a bare name skips
    every directory so named, a path skips that one directory under the root."""

    def __init__(self, excludes):
        self.names = {e for e in excludes if "/" not in e}
        self.paths = {os.path.normpath(e) for e in excludes if "/" in e}

    def skips(self, rel: str) -> bool:
        parts = pathlib.PurePath(rel).parts

        if any(p in self.names for p in parts):
            return True

        return any(os.path.normpath(os.path.join(*parts[:i + 1])) in self.paths
                   for i in range(len(parts)))


def copy_tracked(src: pathlib.Path, dst: pathlib.Path, files, excluder) -> None:
    for rel in files:
        if excluder.skips(rel) or any(part.endswith(SNAPSHOT_SUFFIX)
                                      for part in pathlib.PurePath(rel).parts[:-1]):
            continue

        source, target = src / rel, dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        if source.is_symlink():
            if target.is_symlink() or target.exists():
                target.unlink()
            os.symlink(os.readlink(source), target)
        elif source.is_file():
            shutil.copy2(source, target)


def copy_fetched(src: pathlib.Path, dst: pathlib.Path, excluder) -> None:
    def ignore(directory, names):
        rel = os.path.relpath(directory, src)
        skip = set()

        for name in names:
            path = name if rel == "." else os.path.join(rel, name)
            full = os.path.join(directory, name)

            if name in PRUNE or excluder.skips(path):
                skip.add(name)
            elif os.path.isdir(full) and (name in BUILD_DIRS
                                          or name.endswith(SNAPSHOT_SUFFIX)):
                skip.add(name)
            elif not os.path.isdir(full) and name.endswith(BUILD_SUFFIXES):
                skip.add(name)

        return skip

    shutil.copytree(src, dst, symlinks=True, ignore=ignore, dirs_exist_ok=True)


def copy(src: pathlib.Path, dst: pathlib.Path, spec: dict) -> tuple[int, str]:
    """Copy one corpus tree. Returns (bytes written, how it was copied)."""
    dst.mkdir(parents=True, exist_ok=True)
    before = size(dst)
    excluder = Excluder(spec.get("exclude") or ())
    files = tracked_files(src)

    if files:
        copy_tracked(src, dst, files, excluder)
        how = "tracked files"
    else:
        copy_fetched(src, dst, excluder)
        how = "fetched tree, build output left out"

    return size(dst) - before, how


def size(path: pathlib.Path) -> int:
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True)
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

    # Shallow paths first: a corpus nested in another is copied after its
    # container, so its own rule decides what of it travels.
    def depth(name):
        placed = image.get(name) or {}
        return len(pathlib.PurePath(placed.get("path", "")).parts)

    for name in sorted(wanted, key=depth):
        spec, placed = host.get(name), image.get(name)

        if not spec or not placed:
            missing.append(name)
            continue

        src = pathlib.Path(os.path.expanduser(spec["path"]))

        if not src.is_dir():
            missing.append(name)
            continue

        written, how = copy(src, out / placed["path"], spec)
        total += written
        print(f"   corpora  {name:<16} -> /corpora/{placed['path']}  ({human(written)}, {how})")

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

        # A federation may not name what the image does not carry.
        for spec in kept.values():
            if "corpora" in spec:
                spec["corpora"] = [c for c in spec["corpora"] if c in keep]

        (HERE / "projects.docker.json").write_text(
            json.dumps(kept, indent=2, ensure_ascii=False) + "\n")
        print(f"   corpora  registry restricted to {len(kept)} corpora "
              f"({len(dropped)} dropped: not in this image)")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
