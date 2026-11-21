"""What a user's state adds to an image, written into the image itself.

A session in a container registers trees, indexes them and records what it
learned, all in the user's state directory. Baked, that becomes part of a new
image, so someone else picks the work up with nothing but the image and a
checkout of the tree:

    the corpora the user registered   -> the image's registry
    their index                       -> the image's index
    the knowledge that was accepted   -> the image's common knowledge

Run as root in a container of the image, with the state mounted; the launcher
then commits that container (spear --build).

A corpus path under the user's home is rewritten relative to it, which is how
an image names its trees: whoever has the same checkout at the same place
under their own home gets the index.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tempfile

BATCH = 500


def bake_registry(user, shipped, home, collection_name_for):
    """The shipped registry with the user's corpora added; the names added.

    Each entry is given its collection by name: derived from the path, it
    would not survive the path being rewritten.
    """
    merged, added = dict(shipped), []
    prefix = home.rstrip("/") + "/" if home else None

    for name, spec in sorted(user.items()):
        if name in shipped or not isinstance(spec, dict) or "path" not in spec:
            continue

        entry = dict(spec)
        entry["collection"] = collection_name_for(spec)

        if prefix and spec["path"].startswith(prefix):
            entry["path"] = spec["path"][len(prefix):]

        merged[name] = entry
        added.append(name)

    return merged, added


def bake_index(own, shipped, wanted):
    """Copy the named collections of `own` into `shipped`, vectors included.
    {name: chunks}; a collection the user's index does not have is left out."""
    copied = {}

    for name in sorted(wanted):
        try:
            source = own.get_collection(name)
        except Exception:
            continue

        try:
            shipped.delete_collection(name)
        except Exception:
            pass

        target = shipped.create_collection(name=name, metadata=source.metadata or None)
        total = source.count()

        for offset in range(0, total, BATCH):
            rows = source.get(limit=BATCH, offset=offset,
                              include=["documents", "metadatas", "embeddings"])

            if rows["ids"]:
                target.add(ids=rows["ids"], documents=rows["documents"],
                           metadatas=rows["metadatas"], embeddings=rows["embeddings"])

        copied[name] = total

    return copied


def bake_knowledge(source, target, schema, active):
    """Add the user's active records to the common store; how many.

    Ad hoc workspaces stay behind: they are named after a path of one machine.
    """
    if not os.path.isfile(source):
        return 0

    reader = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    writer = sqlite3.connect(target)
    writer.executescript(schema)

    try:
        rows = reader.execute(
            "SELECT record_id, workspace_id, lifecycle, body FROM records "
            "WHERE lifecycle = ? AND workspace_id NOT LIKE 'adhoc:%'", (active,)).fetchall()
    except sqlite3.Error:
        rows = []

    for record_id, workspace_id, lifecycle, body in rows:
        record = json.loads(body)
        writer.execute("INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?)",
                       (record_id, workspace_id, lifecycle, body))
        writer.execute("INSERT OR REPLACE INTO versions VALUES (?, ?, ?, ?)",
                       (record_id, record["version"], record["updated_at"], body))

    writer.commit()
    writer.close()
    reader.close()

    return len(rows)


def _read(path):
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}

    return data if isinstance(data, dict) else {}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if len(argv) != 2:
        print("usage: spear-bake-state STATE_DIR HOME_OF_THE_USER", file=sys.stderr)

        return 2

    state, home = argv

    import chromadb

    from cli import chat_settings, corpus_registry, corpus_search
    from context import workspace_knowledge as wk
    from runtime import state_paths

    shipped_registry = _read(corpus_registry.PROJECTS_FILE)
    merged, added = bake_registry(_read(f"{state}/projects.json"), shipped_registry,
                                  home, corpus_search.collection_name_for)

    with open(corpus_registry.PROJECTS_FILE, "w") as handle:
        json.dump(merged, handle, indent=2)

    print(f"   registry  {len(added)} added" + (f": {', '.join(added)}" if added else ""))

    # Every collection a registered corpus names, the shipped ones included: a
    # corpus of the image that the user indexed again is baked as they left it.

    copied = {}

    if chat_settings.SHIPPED_DB_PATH and os.path.isdir(f"{state}/chromadb"):
        wanted = {spec["collection"] for spec in merged.values()
                  if isinstance(spec, dict) and spec.get("collection")}

        # Chroma opens its files for writing even to read them, and the state
        # is mounted read-only: it reads a copy.

        with tempfile.TemporaryDirectory() as scratch:
            shutil.copytree(f"{state}/chromadb", f"{scratch}/chromadb")
            copied = bake_index(chromadb.PersistentClient(path=f"{scratch}/chromadb"),
                                chromadb.PersistentClient(path=chat_settings.SHIPPED_DB_PATH),
                                wanted)

    for name, chunks in copied.items():
        print(f"   index     {name}  ({chunks} chunks)")

    if not copied:
        print("   index     nothing of the user's to add")

    common = state_paths.common_dir()
    records = 0

    if common:
        records = bake_knowledge(f"{state}/knowledge.sqlite3", str(common / "knowledge.sqlite3"),
                                 wk._SCHEMA, wk.Lifecycle.ACTIVE)

    print(f"   knowledge {records} active records")

    # The image is run as whoever launches it.

    for root in filter(None, (chat_settings.SHIPPED_DB_PATH, str(common or ""))):
        for directory, _, files in os.walk(root):
            os.chmod(directory, 0o777)

            for name in files:
                os.chmod(os.path.join(directory, name), 0o666)

    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    sys.exit(main())
