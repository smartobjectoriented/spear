#!/usr/bin/env python3
"""
Index a build-system checkout into a ChromaDB vector store.
Run once, or again after the tree changes, to rebuild the index.
"""

import os
import sys
import hashlib
import chromadb
import embedding
from chromadb.config import Settings

# Root of the checkout to index: the CLI argument, else the current
# directory. Each checkout gets its own collection in the same DB_PATH,
# named after the directory it was built from.

APP_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)

# Same reason as index_dir.py: a request for the usage text must not index a
# tree. Here it would have indexed a directory literally named "--help".

# __main__ only: index_dir.py imports chunk_text from here, and without the
# guard `spear-index --help` printed THIS module's usage and exited during
# that import.
if __name__ == "__main__" and ("--help" in sys.argv[1:] or "-h" in sys.argv[1:]):
    print(__doc__.strip() if __doc__ else
          "usage: spear-reindex [checkout]   (default: the current directory)")
    raise SystemExit(0)

# The tree to index: the one named, or the one the caller is standing in.
# It used to default to a checkout that existed on a single workstation.

PROJECT_ROOT = os.path.abspath(
    sys.argv[1] if len(sys.argv) > 1 else os.getcwd())

DB_PATH = os.environ.get("SPEAR_DB_PATH") or os.path.join(APP_DIR, "chromadb")
COLLECTION_NAME = "edgem1_" + os.path.basename(PROJECT_ROOT)

TEXT_EXTENSIONS = {
    ".bbclass", ".bb", ".bbappend", ".inc", ".conf",
    ".its", ".sh", ".py", ".txt", ".cfg", ".md",
    ".rst", ".ini",
}

SKIP_DIRS = {"tmp", "cache", "bitbake", "__pycache__", ".git"}
SKIP_NAMES = {"COPYING", "LICENSE", "COPYING.linux"}

# Source trees whose .c/.h files are ALSO indexed ("where in the code"
# questions). Scoped to avz/ only: linux/ and u-boot/ contain full kernel
# trees that would drown the index. Embedded third-party libs are excluded.

SOURCE_DIRS = ("avz/",)
SOURCE_SKIP = ("avz/net/lwip", "avz/include/net/lwip",
               "avz/fs/fat", "avz/include/fat")

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# A chunk must NEVER exceed this cap: chunk_text splits on lines, and a binary
# (or minified) file has almost none, so it would produce a single chunk of
# several MB — enough to OOM the GPU.

MAX_CHUNK_CHARS = 4 * CHUNK_SIZE

SCRIPTS_SKIP_DIRS = {".venv", "venv", "node_modules", ".git", "__pycache__"}
MAX_SCRIPT_BYTES = 100_000


def is_indexable_text(fpath):
    """True if the file is text of reasonable size. The NUL test over the
    first few KB is the same heuristic as `grep -I`."""

    try:
        if os.path.getsize(fpath) > MAX_SCRIPT_BYTES:
            return False

        with open(fpath, "rb") as f:
            return b"\x00" not in f.read(8192)
    except OSError:
        return False


def should_index(filepath):
    name = os.path.basename(filepath)
    ext = os.path.splitext(name)[1]

    if ext == ".patch":
        # Patches are excluded EXCEPT the buildroot rootfs ones: the
        # defconfigs (BR2_PACKAGE_*) live there and are essential to answer
        # "add package X to the rootfs" questions.

        return "recipes-rootfs/buildroot/files" in filepath

    if ext in (".c", ".h"):
        rel = os.path.relpath(filepath, PROJECT_ROOT)

        if any(rel.startswith(s) for s in SOURCE_SKIP):
            return False

        return any(rel.startswith(d) for d in SOURCE_DIRS)

    if name in SKIP_NAMES:
        return False

    for skip in SKIP_DIRS:
        if f"/{skip}/" in filepath:
            return False

    if ext in TEXT_EXTENSIONS:
        return True

    return False


def chunk_text(text, filepath, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    current_size = 0

    for i, line in enumerate(lines):
        line_len = len(line) + 1

        if current_size + line_len > chunk_size and current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "start_line": i - len(current_chunk) + 1,
                "end_line": i,
            })
            overlap_lines = []
            overlap_size = 0

            for prev_line in reversed(current_chunk):
                if overlap_size + len(prev_line) + 1 > overlap:
                    break

                overlap_lines.insert(0, prev_line)
                overlap_size += len(prev_line) + 1

            current_chunk = overlap_lines
            current_size = overlap_size

        current_chunk.append(line)
        current_size += line_len

    if current_chunk:
        chunk_text = "\n".join(current_chunk)
        chunks.append({
            "text": chunk_text,
            "start_line": len(lines) - len(current_chunk) + 1,
            "end_line": len(lines),
        })

    # Safety net: the loop above appends a line even when that single line
    # exceeds chunk_size (you cannot split on a newline that is not there). So
    # we hard-cut whatever is still too large, keeping the original line range.

    capped = []

    for ch in chunks:
        if len(ch["text"]) <= MAX_CHUNK_CHARS:
            capped.append(ch)
            continue

        for i in range(0, len(ch["text"]), chunk_size):
            capped.append({**ch, "text": ch["text"][i:i + chunk_size]})

    return capped


def classify_file(relpath):
    if relpath.startswith("build/meta-e1c/"):
        return "capsule-layer"

    if relpath.startswith("build/meta-bsp/"):
        return "bsp"

    if relpath.startswith("build/meta-linux/"):
        return "kernel"

    if relpath.startswith("build/meta-uboot/"):
        return "uboot"

    if relpath.startswith("build/meta-atf/"):
        return "atf-optee"

    if relpath.startswith("build/meta-so3/"):
        return "avz-so3"

    if relpath.startswith("build/meta-rootfs/"):
        return "rootfs"

    if relpath.startswith("build/meta-filesystem/"):
        return "filesystem"

    if relpath.startswith("build/meta-torizon/"):
        return "torizon"

    if relpath.startswith("build/meta-usr/"):
        return "userspace"

    if relpath.startswith("build/meta/"):
        return "core-bitbake"

    if relpath.startswith("build/conf/"):
        return "build-config"

    if relpath.startswith("scripts/"):
        return "scripts"

    if relpath.startswith("home_assistant/"):
        return "home-assistant"

    if relpath.startswith("doc/"):
        return "documentation"

    if "/target/" in relpath and relpath.endswith(".its"):
        return "its-image"

    if relpath.startswith("memory/"):
        return "project-knowledge"

    return "other"


def file_type(filepath):
    ext = os.path.splitext(filepath)[1]
    ext_map = {
        ".bbclass": "bitbake-class",
        ".bb": "bitbake-recipe",
        ".bbappend": "bitbake-append",
        ".inc": "bitbake-include",
        ".conf": "config",
        ".its": "image-tree-source",
        ".sh": "shell-script",
        ".py": "python",
        ".cfg": "kernel-config-fragment",
        ".txt": "text",
        ".md": "markdown",
    }

    return ext_map.get(ext, "unknown")


def collect_files():
    files = []

    for root, dirs, filenames in os.walk(os.path.join(PROJECT_ROOT, "build")):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in filenames:
            fpath = os.path.join(root, fname)

            if should_index(fpath):
                relpath = os.path.relpath(fpath, PROJECT_ROOT)
                files.append((fpath, relpath))

    # source trees (.c/.h, "where in the code" questions) — see SOURCE_DIRS

    for srcdir in [d.rstrip("/") for d in SOURCE_DIRS]:
        for root, dirs, filenames in os.walk(os.path.join(PROJECT_ROOT, srcdir)):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for fname in filenames:
                fpath = os.path.join(root, fname)

                if should_index(fpath):
                    files.append((fpath, os.path.relpath(fpath, PROJECT_ROOT)))

    # home_assistant/ (virt64 checkout, home-assistant branch): small
    # curated directory — index all its text files, including those without
    # an extension (st, startchrome) and the Dockerfiles.

    ha_dir = os.path.join(PROJECT_ROOT, "home_assistant")

    if os.path.isdir(ha_dir):
        for root, dirs, filenames in os.walk(ha_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for fname in filenames:
                fpath = os.path.join(root, fname)

                if os.path.getsize(fpath) > 100_000:
                    continue

                files.append((fpath, os.path.relpath(fpath, PROJECT_ROOT)))

    # doc/: Sphinx documentation (.rst)

    doc_dir = os.path.join(PROJECT_ROOT, "doc")

    if os.path.isdir(doc_dir):
        for root, dirs, filenames in os.walk(doc_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and d != "_build"]

            for fname in filenames:
                if fname.endswith((".rst", ".md")):
                    fpath = os.path.join(root, fname)
                    files.append((fpath, os.path.relpath(fpath, PROJECT_ROOT)))

    for fname in ["env.sh", "ci_env.sh", ".gitmodules"]:
        fpath = os.path.join(PROJECT_ROOT, fname)

        if os.path.isfile(fpath):
            files.append((fpath, fname))

    for subdir in ["linux/target", "torizon/target"]:
        dirpath = os.path.join(PROJECT_ROOT, subdir)

        if os.path.isdir(dirpath):
            for fname in os.listdir(dirpath):
                if fname.endswith(".its"):
                    fpath = os.path.join(dirpath, fname)
                    files.append((fpath, os.path.join(subdir, fname)))

    # scripts/: extension-less files are indexed on purpose (the shell helpers
    # have no extension), so no extension filter here. But without a guard the
    # tree also swallows binaries — tezi-custom/ holds uuu (1.2 MB),
    # imx-boot-sd, and a whole .venv with python3.12's ELF. Read as UTF-8 with
    # errors="replace", they produced chunks of several hundred KB of garbage.
    # MiniLM never complained (ONNX truncates at 256 tokens silently); a real
    # GPU embedder does.

    scripts_dir = os.path.join(PROJECT_ROOT, "scripts")

    if os.path.isdir(scripts_dir):
        for root, dirs, filenames in os.walk(scripts_dir):
            dirs[:] = [d for d in dirs
                       if d not in SKIP_DIRS and d not in SCRIPTS_SKIP_DIRS]

            for fname in filenames:
                fpath = os.path.join(root, fname)

                if not is_indexable_text(fpath):
                    continue

                relpath = os.path.relpath(fpath, PROJECT_ROOT)
                files.append((fpath, relpath))

    return files


def main():
    print(f"Collecting files from {PROJECT_ROOT} ...")
    files = collect_files()
    print(f"Found {len(files)} files to index")

    client = chromadb.PersistentClient(path=DB_PATH)

    # We build ALONGSIDE: the live collection is replaced only once indexing
    # has finished (embedding.commit_staged at the end of main). A crash
    # therefore leaves the old index intact. DB_PATH is shared between corpora
    # — only this checkout's collection is touched.

    collection = embedding.staged_collection(client, COLLECTION_NAME)

    all_docs = []
    all_ids = []
    all_metas = []

    for fpath, relpath in files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            print(f"  SKIP {relpath}: {e}")
            continue

        if not content.strip():
            continue

        category = classify_file(relpath)
        ftype = file_type(fpath)

        header = f"# File: {relpath}\n# Type: {ftype} | Category: {category}\n\n"

        if len(content) <= CHUNK_SIZE:
            doc_id = hashlib.md5(relpath.encode()).hexdigest()[:16]
            all_docs.append(header + content)
            all_ids.append(doc_id)
            all_metas.append({
                "filepath": relpath,
                "category": category,
                "filetype": ftype,
                "chunk": "full",
            })
        else:
            chunks = chunk_text(content, relpath)

            for j, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{relpath}:{j}".encode()).hexdigest()[:16]
                chunk_header = f"# File: {relpath} (lines {chunk['start_line']}-{chunk['end_line']})\n# Type: {ftype} | Category: {category}\n\n"
                all_docs.append(chunk_header + chunk["text"])
                all_ids.append(doc_id)
                all_metas.append({
                    "filepath": relpath,
                    "category": category,
                    "filetype": ftype,
                    "chunk": f"{chunk['start_line']}-{chunk['end_line']}",
                })

    print(f"Indexing {len(all_docs)} chunks "
          f"(embedder: {embedding.active_model()}) ...")

    # Encode everything at once: on a GPU, batches of 100 leave the device
    # idle between insertions. Returns None for chroma-default, in which case
    # Chroma embeds batch by batch itself, as before.

    vecs = embedding.embed_documents(all_docs, progress=True)

    batch_size = 100

    for i in range(0, len(all_docs), batch_size):
        end = min(i + batch_size, len(all_docs))
        kw = {} if vecs is None else {"embeddings": vecs[i:end]}
        collection.add(
            documents=all_docs[i:end],
            ids=all_ids[i:end],
            metadatas=all_metas[i:end],
            **kw,
        )
        print(f"  Indexed {end}/{len(all_docs)}")

    embedding.commit_staged(client, COLLECTION_NAME)

    print(f"\nDone! {len(all_docs)} chunks indexed in {DB_PATH}")
    print(f"Collection: {COLLECTION_NAME}")

    categories = {}

    for m in all_metas:
        cat = m["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print("\nChunks per category:")

    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
