#!/usr/bin/env python3
"""Select the normative documents an image may carry, from what they declare.

The store already records, per document, what a build needs to know:

    source_origin      PUBLIC | LICENSED_STANDARD
    raw_pdf_retained   whether the original document itself is kept beside
                       the extracted corpus

So the choice is read, never hand-maintained. A list of "the public ones"
kept in this repository would have to name a customer's standard in order to
exclude it, would go stale the first time a document is ingested, and would
put the whole decision one forgotten edit away from shipping a licensed PDF.

    stage-standards.py --profile public      OUT
    stage-standards.py --profile private  OUT

`public` takes only what declares itself PUBLIC. `private` takes
everything, and says plainly what that means.

The active binding travels only if the document it names travelled: a binding
pointing at an absent store is worse than none, because the session opens
looking bound and answers from nothing. Its fingerprints are not rewritten --
a fabricated binding is exactly the kind of thing that fails quietly later.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

PUBLIC = "PUBLIC"
LICENSED = "LICENSED_STANDARD"


def documents(root: pathlib.Path):
    """(standard_id, revision, manifest) for every ingested document."""
    for manifest in sorted(root.glob("*/*/manifest.json")):
        try:
            body = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"   !! unreadable manifest {manifest}: {exc}", file=sys.stderr)
            continue
        yield manifest.parent, body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=("public", "private"), required=True)
    ap.add_argument("--store", default=None,
                    help="the normative store (default: $SPEAR_STATE_DIR/standards)")
    ap.add_argument("out", help="directory to stage into (emptied first)")
    args = ap.parse_args()

    import os
    store = pathlib.Path(args.store or os.path.join(
        os.environ.get("SPEAR_STATE_DIR",
                       os.path.expanduser("~/.local/state/spear")), "standards"))
    out = pathlib.Path(args.out)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    if not store.is_dir():
        print(f"   standards (no store at {store} — image carries none)")
        return 0

    taken, left, licensed = [], [], []

    for path, body in documents(store):
        origin = body.get("source_origin", LICENSED)
        name = f"{body.get('standard_id', path.parent.name)} " \
               f"{body.get('revision', path.name)}"
        # Unknown provenance is treated as licensed. A document that does not
        # say what it is has not earned the benefit of the doubt in an image
        # somebody is about to hand to someone else.
        if args.profile == "public" and origin != PUBLIC:
            left.append((name, origin))
            continue

        rel = path.relative_to(store)
        shutil.copytree(path, out / rel, symlinks=True)
        taken.append((name, origin))

        if origin == LICENSED:
            licensed.append(name)
            if body.get("raw_pdf_retained"):
                pdf = out / rel / "source"
                kept = [p.name for p in pdf.glob("*.pdf")] if pdf.is_dir() else []
                if kept:
                    licensed[-1] += f"  [+ the document itself: {', '.join(kept)}]"

    for name, origin in taken:
        print(f"   standards {name}  ({origin})")
    for name, origin in left:
        print(f"   standards {name}  ({origin}) — LEFT OUT of a public image")

    # The binding, only if its document came along.
    binding = store / ".active-binding.json"

    if binding.is_file():
        try:
            active = json.loads(binding.read_text()).get("active", {})
        except (OSError, json.JSONDecodeError):
            active = {}

        bound = f"{active.get('standard_id')} {active.get('revision')}"

        if any(bound == name for name, _ in taken):
            shutil.copy2(binding, out / ".active-binding.json")
            print(f"   standards bound on open: {bound}")
        elif active:
            print(f"   standards binding dropped ({bound} is not in this image) —"
                  f" the container opens unbound")

    if licensed:
        print("   ")
        print("   !! this image carries licensed normative material:")
        for name in licensed:
            print(f"   !!   {name}")
        print("   !! it is not redistributable. See docker/README.md.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
