#!/usr/bin/env python3
"""Stage the retrieval index an image may carry: its own corpora's collections.

The host index holds one collection per corpus ever indexed on this machine --
product trees, customer code, the notes corpus -- and the Dockerfile copies the
directory whole. An image built for more people than this machine takes a COPY
of it with every collection it has no corpus for deleted, so the index says
exactly what the registry says.

    stage-index.py --bake a,b [--registry FILE] [--index DIR] OUT

The host index is never opened: chromadb rewrites a store it opens, and the
host's index must stay what it was. The copy is opened, pruned and compacted.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sqlite3
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

# chromadb lives in the client's virtualenv, not on the system python this
# script is started with; run under that interpreter when it is there.

VENV_PYTHON = REPO / "client" / "bin" / "python"

if VENV_PYTHON.is_file() and pathlib.Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    import os
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])


def human(n: int) -> str:
    return f"{n / 1024 / 1024:.0f} MB"


def size(path: pathlib.Path) -> int:
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True)
    return int(out.stdout.split()[0]) if out.stdout.split() else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", default="", help="comma-separated corpus names")
    ap.add_argument("--registry", default=str(REPO / "docker" / "projects.docker.json"),
                    help="the image registry gen-registry.py wrote")
    ap.add_argument("--index", default=str(REPO / "client" / "chromadb"),
                    help="the host index to take the collections from")
    ap.add_argument("out")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    index = pathlib.Path(args.index)
    wanted = [n for n in args.bake.split(",") if n.strip()]

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    if not (index / "chroma.sqlite3").is_file():
        print(f"   index    (none at {index} — image carries no index)")
        return 0

    registry = json.loads(pathlib.Path(args.registry).read_text())
    keep = {}

    for name in wanted:
        spec = registry.get(name)

        if spec and spec.get("collection"):
            keep[spec["collection"]] = name
        else:
            print(f"   index    {name:<16} not in the image registry — no collection to keep",
                  file=sys.stderr)

    shutil.rmtree(out)
    shutil.copytree(index, out, symlinks=True)

    import chromadb

    client = chromadb.PersistentClient(path=str(out))
    present = {c.name: c for c in client.list_collections()}

    for cname in sorted(present):
        if cname not in keep:
            client.delete_collection(cname)

    for cname, name in sorted(keep.items(), key=lambda kv: kv[1]):
        if cname in present:
            print(f"   index    {name:<16} {cname}  ({present[cname].count()} chunks)")
        else:
            print(f"   index    {name:<16} NO INDEX on this host — the corpus will "
                  f"answer without retrieval", file=sys.stderr)

    dropped = sorted(set(present) - set(keep))
    del client

    # Deleting a collection leaves its HNSW directory behind, and frees nothing
    # in the sqlite file until it is compacted.

    con = sqlite3.connect(out / "chroma.sqlite3")
    segments = {row[0] for row in con.execute("SELECT id FROM segments")}
    con.execute("VACUUM")
    con.close()

    for path in out.iterdir():
        if path.is_dir() and path.name not in segments:
            shutil.rmtree(path)

    print(f"   index    {len(set(keep) & set(present))} collections kept, {len(dropped)} "
          f"dropped: {human(size(out))} (host index {human(size(index))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
