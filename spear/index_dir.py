#!/usr/bin/env python3
"""Generic directory indexer for SPEAR ad-hoc mode.

Indexes ANY source tree (current directory by default) into a ChromaDB
collection named adhoc_<md5(realpath)[:8]>, unless ``--collection`` names a
registered collection explicitly. The default is the same tag scheme used by
rag_chat.py for ad-hoc history/memories, so the chat picks the index up
automatically on the next launch (or right after /reindex).

Usage:
    spear-index [root] [--max-files N] [--exclude DIR]... [--collection NAME]
    # root defaults to the current directory
    # --max-files: raise/lower the file cap (default 60000, env
    #   SPEAR_INDEX_MAX_FILES). It is a runaway guard, not a scope tool:
    #   truncation is silent in the index and invisible in the answers, so
    #   use --exclude to decide what belongs in a corpus. Hitting the cap is
    #   now an ERROR, not a warning: see TreeTruncated.
    # --allow-partial: build the truncated index anyway (deliberate only)
    # --exclude: extra directory names to skip (repeatable)
    # --collection: explicit destination collection for a registered corpus
"""
import os
import sys
import hashlib
import chromadb
import embedding

# reuse the proven chunker from index_corpus (pure function)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from index_corpus import chunk_text, CHUNK_SIZE  # noqa: E402

class TreeTruncated(Exception):
    """The walk hit MAX_FILES: what was collected is an arbitrary prefix."""

    def __init__(self, files):
        super().__init__("file cap reached")
        self.files = list(files)

    def report(self):
        from collections import Counter
        tops = Counter(r.split(os.sep)[0] for _, r in self.files)
        lines = [f"ERROR: file cap reached ({MAX_FILES}) — REFUSING to build a "
                 f"partial index.",
                 "  The walk stopped mid-tree, so what it collected is not a "
                 "sample of the corpus,",
                 "  it is whichever directories came first. The existing index "
                 "is left untouched.",
                 "  Fix it with --exclude DIR (vendored trees usually belong "
                 "in their own corpus),",
                 "  raise --max-files N, or accept a partial index on purpose "
                 "with --allow-partial.",
                 "  Reached so far, by top-level directory:"]
        lines += [f"    {n:6d}  {d}" for d, n in tops.most_common(8)]

        return "\n".join(lines)


def parse_args(argv):
    root, max_files, excludes, collection = None, None, [], None
    include_build = allow_partial = False
    it = iter(argv)

    for a in it:
        # `spear-index --help` used to run an INDEXATION: --help matched no
        # branch, fell through the `not a.startswith("-")` guard as a flag, and
        # the walk started on the current directory. A request for the usage
        # text must not write a collection.

        if a in ("--help", "-h"):
            print(__doc__.strip())
            raise SystemExit(0)

        if a == "--max-files":
            max_files = int(next(it, "8000"))
        elif a == "--exclude":
            excludes.append(next(it, ""))
        elif a == "--collection":
            collection = next(it, "")
        elif a == "--include-build":
            include_build = True
        elif a == "--allow-partial":
            allow_partial = True
        elif not a.startswith("-") and root is None:
            root = a

    return root, max_files, excludes, include_build, allow_partial, collection


(_root, _max_files, _excludes, _include_build,
 ALLOW_PARTIAL, _collection) = parse_args(sys.argv[1:])
ROOT = os.path.realpath(_root or os.getcwd())
TAG = hashlib.md5(ROOT.encode()).hexdigest()[:8]
COLLECTION_NAME = _collection or f"adhoc_{TAG}"

# SPEAR_DB_PATH keeps a test (or a container) from writing into the live
# store: the cap tests below index for real, and without this they left
# throwaway collections in the corpus the assistant actually queries.

DB_PATH = os.environ.get("SPEAR_DB_PATH") or os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "chromadb")

EXTENSIONS = {
    ".c", ".h", ".S", ".s", ".cpp", ".hpp", ".cc",
    ".py", ".sh", ".pl", ".mk", ".cmake",
    ".md", ".rst", ".txt",
    ".dts", ".dtsi", ".its", ".cfg", ".conf", ".ini",
    ".yaml", ".yml", ".json",
    ".bb", ".bbclass", ".bbappend", ".inc",
}
BASENAMES = {"Makefile", "Kconfig", "Kbuild", "Dockerfile", "README",
             "defconfig", "CMakeLists.txt"}
SKIP_DIRS = {".git", ".svn", "build", "out", "output", "tmp", "dist",
             "__pycache__", "node_modules", ".cache", ".vscode", ".idea"}

# Build-generated pristine/backup snapshots (avz.back, usr.back,
# foo.pristine, bar.GOOD) — Infrabase keeps these so updiff.sh can diff the
# working tree against pristine to regenerate patches. They duplicate real
# source and pollute retrieval, so skip them by suffix wherever they appear.
# NOTE: ".0" is deliberately NOT here — it collides with version dirs
# (e.g. zynq-rev1.0, enchant/1.6.0). Exclude any real ".0" snapshot via
# --exclude if you have one.

SKIP_DIR_SUFFIXES = (".back", ".pristine", ".GOOD", ".orig", ".bak")
MAX_FILE_BYTES = 200_000

# 8000 was set when embedding ran on CPU at a few chunks per second and a large
# tree meant an afternoon. It no longer protects anything useful: it silently
# truncated u-boot (11298 indexable files) and, on a parent directory, filled a
# whole corpus with the first tree walked alphabetically. A partial index is
# worse than a slow one — it answers confidently from a fraction of the code.
# Raised to a level that only stops genuine runaways; use --exclude to decide
# what belongs in a corpus, which is the honest tool for scope.

MAX_FILES = _max_files or int(os.environ.get("SPEAR_INDEX_MAX_FILES", "60000"))
SKIP_DIRS |= set(_excludes)

# --include-build: index a bitbake/Infrabase build/ tree (recipes in
# meta-*, conf, classes) WITHOUT its huge generated work dirs. We stop
# skipping the literal "build" dir and instead skip the heavy subpaths by
# relative path.

SKIP_REL_PATHS = set()

if _include_build:
    SKIP_DIRS.discard("build")
    SKIP_REL_PATHS = {
        os.path.join("build", d) for d in
        ("tmp", "sstate-cache", "downloads", "cache", "deploy")
    }


def skip_dir(name, relpath):
    if name in SKIP_DIRS or name.endswith(SKIP_DIR_SUFFIXES):
        return True

    if relpath in SKIP_REL_PATHS:
        return True

    return False


def wanted(name):
    if name in BASENAMES or name.endswith("_defconfig"):
        return True

    return os.path.splitext(name)[1] in EXTENSIONS


def collect():
    files = []

    for root, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs
                   if not skip_dir(d, os.path.relpath(os.path.join(root, d),
                                                      ROOT))]

        for n in names:
            if not wanted(n):
                continue

            p = os.path.join(root, n)

            try:
                if os.path.getsize(p) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue

            files.append((p, os.path.relpath(p, ROOT)))

            if len(files) >= MAX_FILES and not ALLOW_PARTIAL:
                raise TreeTruncated(files)

    return files


def main():
    print(f"Indexing {ROOT}\n  -> collection {COLLECTION_NAME}")

    try:
        files = collect()
    except TreeTruncated as truncated:
        # A truncated index is worse than none: it answers confidently from a
        # fraction of the tree, and the fraction is whichever directory the
        # walk happened to reach first. This used to be a WARNING that scrolled
        # past in buffered output -- an infrabase index came out 99.8% vendored
        # QEMU and u-boot, with none of the build system it existed for, and
        # nothing failed. Refuse instead, and leave the previous index intact:
        # the staged collection is only swapped in at the very end.

        print(truncated.report(), file=sys.stderr)
        sys.exit(2)

    print(f"  {len(files)} files")

    client = chromadb.PersistentClient(path=DB_PATH)

    # build alongside, swap at the end of main — see staged_collection

    coll = embedding.staged_collection(client, COLLECTION_NAME)

    docs, ids, metas = [], [], []

    for fpath, rel in files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError:
            continue

        if not content.strip():
            continue

        header = f"# File: {rel}\n\n"

        if len(content) <= CHUNK_SIZE:
            docs.append(header + content)
            ids.append(hashlib.md5(rel.encode()).hexdigest()[:16])
            metas.append({"filepath": rel, "chunk": "full"})
        else:
            for j, ch in enumerate(chunk_text(content, rel)):
                docs.append(f"# File: {rel} (lines {ch['start_line']}-"
                            f"{ch['end_line']})\n\n" + ch["text"])
                ids.append(hashlib.md5(f"{rel}:{j}".encode()).hexdigest()[:16])
                metas.append({"filepath": rel,
                              "chunk": f"{ch['start_line']}-{ch['end_line']}"})

    vecs = embedding.embed_documents(docs, progress=True)

    for i in range(0, len(docs), 100):
        kw = {} if vecs is None else {"embeddings": vecs[i:i+100]}
        coll.add(documents=docs[i:i+100], ids=ids[i:i+100],
                 metadatas=metas[i:i+100], **kw)
        print(f"  indexed {min(i+100, len(docs))}/{len(docs)}", end="\r")

    embedding.commit_staged(client, COLLECTION_NAME)
    print(f"\nDone: {len(docs)} chunks in {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
