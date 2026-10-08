#!/usr/bin/env python3
"""Select the workspace knowledge an image carries as its common state.

    stage-knowledge.py --workspaces so3,so3-doc OUT
    stage-knowledge.py --workspaces all OUT

Only ACTIVE records of the named workspaces travel, each with its current
version alone: a proposal nobody accepted, a revoked record and the wording a
record had before it was amended are one user's, not the team's. Ad hoc
workspaces never travel, since they are named after a path on this host.

The records come from the team's common store, $SPEAR_COMMON_STATE_DIR, where
spear-consolidate merges what users have learned; without one, from the
builder's own store. With no workspace named, OUT is left empty and the image
has no common knowledge.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sqlite3
import sys

APP = pathlib.Path(__file__).resolve().parents[2] / "client"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspaces", default="",
                    help="comma-separated registered workspaces, or 'all'")
    ap.add_argument("--store", default=None,
                    help="the knowledge store (default: the common one, "
                         "else the builder's own)")
    ap.add_argument("out", help="directory to stage into (emptied first)")
    args = ap.parse_args()

    sys.path.insert(0, str(os.environ.get("SPEAR_APP", APP)))
    from context import workspace_knowledge as wk

    out = pathlib.Path(args.out)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    names = [name for name in args.workspaces.split(",") if name]

    if not names:
        print("   knowledge (none named — image carries no common knowledge)")
        return 0

    source = pathlib.Path(args.store or wk.default_common_path() or wk.default_path())

    if not source.is_file():
        print(f"   knowledge (no store at {source} — image carries none)")
        return 0

    reader = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    present = sorted({row[0] for row in reader.execute(
        "SELECT workspace_id FROM records WHERE workspace_id NOT LIKE 'adhoc:%'")})
    wanted = present if names == ["all"] else names
    unknown = sorted(set(wanted) - set(present))

    if unknown:
        print(f"no knowledge recorded for {', '.join(unknown)} "
              f"(recorded: {', '.join(present) or 'none'})", file=sys.stderr)
        return 1

    staged = out / "knowledge.sqlite3"
    writer = sqlite3.connect(staged)
    writer.executescript(wk._SCHEMA)

    for name in wanted:
        rows = reader.execute(
            "SELECT record_id, workspace_id, lifecycle, body FROM records "
            "WHERE workspace_id = ? AND lifecycle = ?", (name, wk.Lifecycle.ACTIVE)).fetchall()

        for record_id, workspace_id, lifecycle, body in rows:
            writer.execute("INSERT INTO records VALUES (?, ?, ?, ?)",
                           (record_id, workspace_id, lifecycle, body))
            writer.execute("INSERT INTO versions VALUES (?, ?, ?, ?)",
                           (record_id, json.loads(body)["version"],
                            json.loads(body)["updated_at"], body))

        print(f"   knowledge {name}: {len(rows)} active records")

    writer.commit()
    writer.close()
    reader.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
