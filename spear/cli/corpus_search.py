"""Retrieval over the session's corpora, the history archive, reindexing."""

import os
import re
import sys
import hashlib
import subprocess
from retrieval import embedding
from cli import session_workspace
from cli.chat_settings import APP_DIR, DB_PATH
from cli.corpus_registry import corpora_below, load_projects
from cli.terminal_ui import C_DIM, C_RST, C_WARN


# The federation opened for this session, bound in main(). search_corpus needs
# it at tool-dispatch time, where the main loop's local is out of reach.

COLLECTION = None

TOP_K = 12
MAX_CONTEXT_CHARS = 12000


# ── ChromaDB / RAG ───────────────────────────────────────────────────

def collection_name_for(spec):
    """The chroma collection a registered corpus is indexed into.

    An explicit ``"collection"`` wins over the derived name. The derivation is
    an md5 of the ABSOLUTE path, which is fine while the index and the trees
    live on the same machine and wrong the moment they do not: mounted under
    /corpora, the same tree hashes differently and the shipped index goes
    unfound -- the container started cleanly with no retrieval at all, which is
    the whole point of the tool. So a registry that travels with an index names
    the collections instead of recomputing them.
    """
    explicit = spec.get("collection")

    if explicit:
        return explicit

    tag = hashlib.md5(os.path.realpath(spec["path"]).encode()).hexdigest()[:8]

    return f"adhoc_{tag}"


def corpus_prefix(path):
    """How a chunk of `path` must be addressed from the current tree.

    Relative while the corpus sits under the launch directory -- that is the
    federation case, and bash resolves it. Otherwise ABSOLUTE: a shared corpus
    lives in another tree entirely, and `../../../opt/llm/...` is both unusable
    and refused by the command policy. Absolute paths outside the workspace are
    readable now, so the model can actually open what it is shown.
    """
    corpus, root = os.path.realpath(path), os.path.realpath(session_workspace.PROJECT_ROOT)

    if corpus == root:
        return ""

    if corpus.startswith(root + os.sep):
        return os.path.relpath(corpus, root)

    return corpus


def _cli_values(flag):
    """Repeated `--flag value` occurrences, in order."""
    argv, out = sys.argv[1:], []

    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            out.append(argv[i + 1])
        elif a.startswith(flag + "="):
            out.append(a.split("=", 1)[1])

    return out


def attached_corpus_names(spec, projects):
    """Every corpus this session should retrieve from, besides its own.

    Three sources, in this order:
      - `"corpora": [...]` on the project -- a federation of ONE tree's parts;
      - every corpus marked `"shared": true` -- cross-cutting knowledge that
        belongs to no single tree (the build system, an API reference). This is
        what lets `ib` answer a build question in whatever tree you happen to
        be standing in, without redeclaring it in all 22 projects;
      - `--with <name>` on the command line, for a one-off.
    `--without <name>` removes any of them, so a shared corpus is never a
    sentence: a session that does not want it says so and pays nothing.
    """
    names = list(spec.get("corpora") or [])
    names += [n for n, sub in projects.items()
              if sub.get("shared") and n not in names]
    names += [n for n in _cli_values("--with") if n not in names]
    dropped = set(_cli_values("--without"))

    return [n for n in names if n not in dropped]


def federated_corpora():
    """Corpora this session retrieves from, as (collection, path prefix) pairs.

    A project may declare `"corpora": ["a", "b"]` in projects.json instead of
    owning one index. Questions then reach every one of them without switching
    session — which is the point: a kernel question, a userspace question and a
    bootloader question all belong to the same tree. A corpus marked
    `"shared": true` is attached to EVERY session for the same reason, one
    level up: the build system is not the property of one checkout.

    Merging them into a single index would not do: u-boot holds 11298 indexable
    files against so3's 1482, so kernel code would compete 7-to-1 for the same
    twelve slots. Kept apart and fused by rank, each contributes its own best
    hits.

    The prefix is what makes the result usable: chunks are indexed relative to
    THEIR corpus root, while tools run at the federation root. Without it the
    model reads `usr/src/x.c` and bash needs `so3/usr/src/x.c`.
    """
    projects = load_projects()
    spec = projects.get(session_workspace.PROJECT) or session_workspace.PROJECT_SPEC or {}
    names = attached_corpus_names(spec, projects)

    if not names:
        return []

    client = _db()
    auto = set(spec.get("auto_corpora") or ())
    AUTO_CORPUS_BY_COLLECTION.clear()
    # `own` is the corpus this session already adds itself (init_chromadb),
    # and that is CORPUS_ROOT — not the cwd. Compared against the cwd, a
    # session standing outside its own tree failed to recognise it and
    # attached its own index a second time.

    out, own = [], os.path.realpath(session_workspace.CORPUS_ROOT)

    for n in names:
        sub = projects.get(n)

        if not sub:
            print(f"{C_WARN}corpus '{n}' is not registered — skipped{C_RST}")
            continue

        if os.path.realpath(sub["path"]) == own:
            continue            # the session's own corpus is added by the caller

        try:
            col = client.get_collection(name=collection_name_for(sub))
        except Exception:
            print(f"{C_WARN}corpus '{n}' has no index yet "
                  f"({collection_name_for(sub)}) — index it with: "
                  f"spear-index {sub['path']}{C_RST}")
            continue

        out.append((col, corpus_prefix(sub["path"])))

        if n in auto:
            AUTO_CORPUS_BY_COLLECTION[col.name] = n

    return out


def init_chromadb():
    """The session's retrieval set: its own corpus PLUS whatever is attached.

    Attached corpora used to REPLACE the project's own index rather than join
    it, which was harmless while the only federation (sye_sol) owned no index
    of its own -- and would have silently emptied every other session the day a
    shared corpus was declared.
    """
    attached = federated_corpora()
    client = _db()

    try:
        own = [(client.get_collection(name=session_workspace.COLLECTION_NAME), "")]
    except Exception:
        own = []

        if corpus_autoindexes():
            # This corpus declares that a missing index is built on sight.

            print(f"{C_WARN}Corpus {session_workspace.COLLECTION_NAME} missing — indexing "
                  f"{session_workspace.CORPUS_ROOT}...{C_RST}")
            subprocess.run(reindex_command())
            own = [(client.get_collection(name=session_workspace.COLLECTION_NAME), "")]
        else:
            # generic project: indexing is opt-in (trees can be huge). Say so
            # even when other corpora are attached — the banner sums whatever
            # the session retrieves from and prints the total under THIS
            # project's name, so an agency session owning no index announced
            # "agency · 506 chunks": 506 chunks of an attached corpus, and not
            # one line of agency.

            print(f"{C_DIM}No index for this corpus yet — use /reindex "
                  f"to build one (optional).{C_RST}")

            if not attached:
                return None

    corpora = own + attached

    if attached:
        total = sum(c.count() for c, _ in corpora)
        print(f"{C_DIM}retrieving from {len(corpora)} corpora "
              f"({total} chunks){C_RST}")

    return corpora or None


# ── hybrid lexical + dense ───────────────────────────────────────────
# MiniLM's cosine distance does NOT discriminate identifiers: measured on
# a build-system index, the query "ou est defini __sys_empty" returns 12 chunks all
# within 0.635-0.669 — the same band as an off-topic question (0.72-0.80). With
# 0.03 between the 1st and the 12th, the dense ranking means nothing there.
# Chroma's full-text filter, on the other hand, cuts clean (syscalls.c in
# 0.04 s). We fuse both rankings with Reciprocal Rank Fusion, which does not
# require their scores to be comparable.
# Measured by eval/run_eval.py: recall@12 identifiers 58% -> 92%, global
# 65% -> 92%.

_IDENT_RX = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*[A-Za-z0-9_]")
LEX_MAX_TERMS = 4       # beyond this we only dilute the fusion
LEX_PER_TERM = 6        # lexical hits kept per term
LEX_SATURATION = 40     # a term in >= 40 chunks discriminates nothing


def _ident_terms(query, max_terms=LEX_MAX_TERMS):
    """Query tokens deserving an EXACT search: snake_case, CamelCase,
    UPPERCASE, file names. Ordinary French words carry neither underscore, dot
    nor capital — they are dropped here, and the dense channel handles them
    well anyway."""
    terms = []

    for t in _IDENT_RX.findall(query):
        if len(t) < 4:
            continue

        # "_" / "." / UPPERCASE / internal capital = identifier or file name.
        # Leading capital alone is NOT enough: "Comment", "Dans", "Montre"
        # carry one too, and firing a $contains at them costs a query for
        # nothing (saturation eventually drops them, but later).

        if ("_" in t or "." in t or t.isupper()
                or any(c.isupper() for c in t[1:])):
            terms.append(t)

    # longest first: those are the most specific

    return sorted(dict.fromkeys(terms), key=len, reverse=True)[:max_terms]


def _definition_score(doc, meta, term):
    """Ranking for lexical hits. `get()` returns no order, so we must build one.

    Two signals. (1) The PATH: Chroma's `$contains` is a raw SUBSTRING match,
    so a query for a file name also returns every longer name ending the same
    way. When the term IS the file name that is decisive, hence the
    overwhelming weight. (2) The BODY: a chunk that DEFINES the term (see
    _definition_lines) weighs more than one merely using it."""
    path = meta.get("filepath", "")
    score = 0

    if term == os.path.basename(path):
        score += 100
    elif term in path:
        score += 50

    body = doc.split("\n\n", 1)[-1]

    return score + 3 * _definition_lines(body, term) + body.count(term)


def _definition_lines(body, term):
    """How many lines DEFINE `term` rather than use it.

    Three shapes cover most of the corpus:
      - `term` at line start   -> `IB_X = "..."`, `CONFIG_X=y`, a label
      - `#define term`         -> a macro
      - `term(` with no trailing `;` -> a C function header. The `;` is what
        separates the DEFINITION (`static long __sys_empty(args)`) from the
        declaration (`... args);`) and from the call (`x = __sys_empty(a);`) —
        in none of the three is the identifier at line start.
    """
    n = 0

    for line in body.split("\n"):
        s = line.lstrip()

        if (s.startswith(term)
                or s.startswith(f"#define {term}")
                or (f"{term}(" in s and not s.rstrip().endswith(";"))):
            n += 1

    return n


def _rrf(ranked_lists, k=60):
    """Reciprocal Rank Fusion: merges heterogeneous rankings without having to
    normalise their scores. k=60 is the original constant (Cormack et al.) —
    it flattens the weight of the list heads."""
    score = {}

    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst):
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(score, key=score.get, reverse=True)


_FILENAME_RX = re.compile(r"[A-Za-z0-9_.-]+\.[A-Za-z0-9_]{1,8}")


def _lex_get(collection, needle):
    """Lexical candidates for one needle, or None when it discriminates nothing."""

    try:
        hit = collection.get(where_document={"$contains": needle},
                             include=["documents", "metadatas"],
                             limit=LEX_SATURATION)
    except Exception:
        return None     # the lexical channel is a bonus, never a point of
                        # failure: the chat must answer without it.
    ids = hit["ids"]

    if not ids or len(ids) >= LEX_SATURATION:
        return None     # term absent, or too common to discriminate

    return list(zip(ids, hit["documents"], hit["metadatas"]))


def _lex_candidates(collection, term):
    """Hits for a query term, anchored on a path separator when it names a file.

    `$contains` matches raw substrings, so "ls.c" is equally found inside
    "globals.c", "parserInternals.c" and "syscalls.c". On a real corpus that
    turned the most specific term of the query into its least useful one: 104
    chunks matched, 98 of them for files merely ENDING in "ls.c", the count
    tripped the saturation guard, and the term was dropped entirely -- so
    asking to edit ls.c retrieved everything except ls.c, and handed the model
    libxml2 instead. Anchoring on "/" restores the file-name meaning: the same
    query then returns six chunks, all of them the actual file.
    """

    if _FILENAME_RX.fullmatch(term):
        anchored = _lex_get(collection, "/" + term)

        if anchored:
            return anchored

    return _lex_get(collection, term)


def _retrieve_one(collection, query, top_k):
    """Dense + lexical candidates from ONE collection.

    Returns (pool, lists): the documents keyed by id, and the rankings to be
    fused. Split out of retrieve_context so several corpora can be searched
    per turn without each one's ranking swamping the others.
    """

    # The embedding model is read from the collection METADATA, not from the
    # config: a collection indexed with another model stays correctly
    # queryable even if active-embedder.conf changed since. Without this,
    # Chroma would report nothing and return random neighbours.

    qvec = embedding.embed_query(query, embedding.collection_model(collection))

    if qvec is None:
        dense = collection.query(
            query_texts=[query], n_results=top_k * 2,
            include=["documents", "metadatas", "distances"],
        )
    else:
        dense = collection.query(
            query_embeddings=[qvec], n_results=top_k * 2,
            include=["documents", "metadatas", "distances"],
        )

    pool = {i: (d, m) for i, d, m in zip(
        dense["ids"][0], dense["documents"][0], dense["metadatas"][0])}
    lists = [list(dense["ids"][0])]

    for term in _ident_terms(query):
        candidates = _lex_candidates(collection, term)

        if not candidates:
            continue

        ranked = sorted(candidates,
                        key=lambda x: -_definition_score(x[1], x[2], term))
        ranked = ranked[:LEX_PER_TERM]

        for i, d, m in ranked:
            pool.setdefault(i, (d, m))

        lists.append([i for i, _, _ in ranked])

    return pool, lists


SEARCH_CORPUS_TOP_K = 5      # a mid-turn result, not a whole turn's context
SEARCH_CORPUS_MAX_CHARS = 4000


def search_corpus(query, top_k=SEARCH_CORPUS_TOP_K):
    """Let the model query the index while it works, not only at turn start.

    The Retrieved Context is chosen once, from the USER's sentence. That
    sentence carries the task ("modify ls.c so it handles wildcards"), not the
    sub-question the model hits three steps later ("what matches a pattern?").
    Without a way to ask, the model answers that sub-question from whatever the
    first retrieval happened to include -- which is how one session ended up
    linking libxml2 into ls because triostr.c was in the context and fnmatch
    was not.
    """

    if COLLECTION is None:
        return "ERROR: no corpus is indexed for this project"

    try:
        context, _ = retrieve_context(COLLECTION, query, top_k)
    except Exception as e:
        return f"ERROR: corpus search failed: {e}"

    if not context or not context.strip():
        return "(no match in the indexed corpora)"

    if len(context) > SEARCH_CORPUS_MAX_CHARS:
        context = (context[:SEARCH_CORPUS_MAX_CHARS]
                   + "\n… (truncated; ask a narrower query)")

    return context


def _reroot(doc, meta, prefix):
    """Rewrite a chunk's `# File:` header to be relative to the FEDERATION
    root, not to its own corpus. Chunks are indexed per corpus, but tools run
    at the federation root — leaving `usr/src/x.c` when bash needs
    `so3/usr/src/x.c` is what sent the model chasing absolute paths."""

    if not prefix:
        return doc, meta.get("filepath", "")

    fp = meta.get("filepath", "")
    rooted = os.path.join(prefix, fp) if fp else fp
    head, sep, rest = doc.partition("\n\n")

    if sep and head.startswith("# File: ") and fp:
        head = head.replace(fp, rooted, 1)
        doc = head + sep + rest

    return doc, rooted


# collection name -> corpus name, for the corpora an umbrella DISCOVERED. Only
# these are selected per question; a declared federation and a shared corpus
# were chosen deliberately and are attached every turn.

AUTO_CORPUS_BY_COLLECTION = {}


def _mentioned(name, path, query):
    """Does the question name this corpus, or the directory it lives in?

    Whole words, so `so3` does not match `so3-doc`'s path fragment, and the
    basename counts too: in a workspace session "a chapter in doc" is naming
    `doc/`, which is corpus so3-doc. The hint's stopword list deliberately does
    NOT apply here -- it exists to avoid telling someone to `cd` somewhere on
    the strength of the word "build", while inside a workspace that same word
    really does name the subdirectory being asked about.
    """
    words = {name, os.path.basename(path.rstrip("/"))}

    return any(word and re.search(rf"\b{re.escape(word)}\b", query, re.I)
               for word in words)


def select_corpora(corpora, query, projects=None):
    """Narrow the DISCOVERED corpora to the ones the question names.

    A workspace launch federates everything registered under it -- eight trees
    at ~/soo/so3 -- and querying all of them every turn costs an embedding call
    each for corpora the question never touches. Named ones win; if the
    question names none, the part sharing the workspace's own name is kept, on
    the same umbrella-shape reasoning that used to pick it outright.

    Anything not discovered this way is untouched: the session's own index, a
    declared federation, a shared corpus, `--with`.
    """

    if not AUTO_CORPUS_BY_COLLECTION or not query:
        return corpora

    projects = projects if projects is not None else load_projects()
    primary = (session_workspace.PROJECT.split(":", 1)[1]
               if session_workspace.PROJECT.startswith("workspace:") else None)

    # Longest name first, blanking what it matched: `-` is a word boundary, so
    # `\bso3\b` fires inside `micropython-so3` and would attach the kernel to a
    # question that named only the library. Same rule the mention hint uses.

    remaining, named = query, set()

    for name in sorted(set(AUTO_CORPUS_BY_COLLECTION.values()), key=len, reverse=True):
        path = (projects.get(name) or {}).get("path", "")

        for word in sorted({name, os.path.basename(path.rstrip("/"))},
                           key=len, reverse=True):
            if not word:
                continue

            # Not \b: `-` is a word boundary, so `\bso3\b` fires inside
            # `micropython-so3` and would attach the kernel to a question that
            # named only the library. A following `/` is fine -- `so3/usr/src`
            # names the corpus as plainly as `so3` does.

            pattern = rf"(?<![\w-]){re.escape(word)}(?![\w-])"

            if re.search(pattern, remaining, re.I):
                named.add(name)
                remaining = re.sub(pattern, " ", remaining, flags=re.I)

                break

    keep = named or ({primary} if primary in AUTO_CORPUS_BY_COLLECTION.values()
                     else set(AUTO_CORPUS_BY_COLLECTION.values()))

    return [(col, prefix) for col, prefix in corpora
            if col.name not in AUTO_CORPUS_BY_COLLECTION
            or AUTO_CORPUS_BY_COLLECTION[col.name] in keep]


def retrieve_context(corpora, query, top_k=TOP_K):
    """Retrieve from one corpus or several, fusing the rankings.

    `corpora` is a collection, or a list of (collection, prefix) pairs. With
    several, each contributes its own top-k and Reciprocal Rank Fusion merges
    them: a chunk ranked first in a small corpus weighs as much as the first of
    a large one, which is exactly what a single merged index cannot offer.
    """

    if not isinstance(corpora, list):
        corpora = [(corpora, "")]

    corpora = select_corpora(corpora, query)

    pool, lists = {}, []

    for col, prefix in corpora:
        try:
            sub_pool, sub_lists = _retrieve_one(col, query, top_k)
        except Exception:
            continue        # one broken corpus must not sink the whole turn

        name = getattr(col, "name", id(col))

        for i, (d, m) in sub_pool.items():
            # Ids are md5(relpath) per corpus, so two corpora can collide on
            # the same relative path. Key by corpus as well.

            pool[(name, i)] = (d, m, prefix)

        lists.extend([[(name, i) for i in l] for l in sub_lists])

    order = _rrf(lists) if len(lists) > 1 else (lists[0] if lists else [])

    parts, total, seen = [], 0, set()

    for key in order[:top_k * 2]:
        doc, meta, prefix = pool[key]
        doc, rooted = _reroot(doc, meta, prefix)

        if total + len(doc) > MAX_CONTEXT_CHARS:
            continue        # `continue`, not `break`: one large chunk must no
                            # longer condemn every chunk after it.
        parts.append(doc)
        total += len(doc)
        seen.add(rooted)

    return "\n\n---\n\n".join(parts), seen


ARCHIVE_COLLECTION = "edgem_archive"


def _db():
    """The Chroma client, and the only place chromadb is imported.

    Importing it at module level cost 0.78 s on every launch -- including the
    ones that never open the index (--help, a --model switch, a session that
    ends on a shell command). Deferred here, that second is paid on the first
    retrieval, where the caller is already waiting for the model.
    """
    import chromadb

    return chromadb.PersistentClient(path=DB_PATH)


def archive_index(rec):
    """Embed one archive record for search_history (incremental)."""
    content = (rec.get("content") or "")[:1500]

    if len(content) < 40:
        return

    try:
        coll = _db().get_or_create_collection(
            name=ARCHIVE_COLLECTION, metadata={"hnsw:space": "cosine"})
        rid = hashlib.md5((rec.get("ts", "") + content[:80]).encode()).hexdigest()[:16]
        coll.upsert(ids=[rid], documents=[content],
                    metadatas=[{"ts": rec.get("ts", ""),
                                "role": rec.get("role", ""),
                                "project": session_workspace.PROJECT}])
    except Exception:
        pass   # search index is best-effort, never break the chat


def archive_forget(pattern):
    """Remove archive-search entries whose text matches `pattern` (regex,
    case-insensitive). Prunes the SEARCH INDEX only — the append-only
    history-archive.jsonl trace on disk is never touched."""

    try:
        coll = _db().get_collection(ARCHIVE_COLLECTION)
    except Exception:
        return "(history index empty)"

    got = coll.get(include=["documents"])
    rx = re.compile(pattern, re.I)
    victims = [i for i, d in zip(got["ids"], got["documents"]) if rx.search(d or "")]

    if victims:
        coll.delete(ids=victims)

    return f"removed {len(victims)} entries (index now {coll.count()})"


def archive_search(query, n=5):
    try:
        coll = _db().get_collection(ARCHIVE_COLLECTION)
    except Exception:
        return "(history index empty)"

    if coll.count() == 0:
        return "(history index empty)"

    # over-fetch, then filter: drop self-echoes (the query itself, just
    # indexed), dedupe, and rank entries holding REAL tool output first —
    # the archive also contains past hallucinated answers, and grounded
    # excerpts are the antidote, not more model prose.

    r = coll.query(query_texts=[query],
                   n_results=min(4 * n, coll.count()),
                   include=["documents", "metadatas", "distances"])
    seen, hits = set(), []

    for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0],
                               r["distances"][0]):
        if dist < 0.05:                       # the query echoing itself
            continue

        key = doc[:120]

        if key in seen:
            continue

        seen.add(key)
        hits.append((0 if "[tool]" in doc or "$ " in doc else 1, dist,
                     doc, meta))

    hits.sort(key=lambda h: (h[0], h[1]))
    out = []

    for _, _, doc, meta in hits[:n]:
        out.append(f"[{meta.get('ts','?')} · {meta.get('role','?')}"
                   f" · {meta.get('project', meta.get('checkout','?'))}]\n{doc[:400]}")

    return "\n\n".join(out) if out else "(no match)"


def corpus_property(name, default=None):
    """A declared property of the corpus this session opened.

    Read from the registry entry, falling back to the spec the session was
    launched with (an ad-hoc corpus has no registry entry at all). Behaviour
    that used to be inferred from `kind` is declared here instead: what a
    corpus DOES should be visible in the line that describes it, not in a
    branch somewhere that tests its type.
    """
    spec = load_projects().get(session_workspace.PROJECT) or session_workspace.PROJECT_SPEC or {}

    return spec.get(name, default)


def corpus_indexer():
    """Which indexer rebuilds this corpus: "buildsystem" or "generic".

    The buildsystem walk is curated for BitBake/Yocto trees -- recipe and
    ITS extensions, a scoped source subtree, a hand-maintained skip list --
    and indexes a materially different set of files from the generic one.
    """
    return corpus_property("indexer", "generic")


def corpus_autoindexes():
    """Whether a missing index is built on first sight.

    Off by default: a generic tree can be enormous, and indexing one because
    a session happened to open it is a surprise. A corpus whose curated walk
    is cheap and expected declares it.
    """
    return bool(corpus_property("autoindex", False))


def reindex_options():
    """Indexing options for this project: what projects.json declares, plus
    the corpora that live INSIDE this one.

    The declared exclusions are the deliberate ones, and they belong in the
    registry rather than on the command line: spelled out at the call site, a
    /reindex silently brings back what we took care to leave out. so3 vendors
    lvgl and micropython, which already have their own corpora — unexcluded
    they made up 85% of its index and surfaced instead of the kernel code.

    The derived ones close the
    trap a workspace split leaves behind: registering agency, buildroot, qemu
    and u-boot as four corpora does not stop a /reindex of the tree ABOVE them
    from walking all four again — 60000 files, the file cap, a refusal, and all
    of it for chunks the federation already retrieves from their own indexes. A
    tree that owns a corpus is never part of another one.
    """
    projects = load_projects()
    spec = projects.get(session_workspace.PROJECT) or session_workspace.PROJECT_SPEC or {}
    excludes = list(spec.get("exclude", ()))

    # As a path relative to the corpus root, not a bare name: index_dir skips
    # a bare name wherever it occurs, and a component usually shares its name
    # with a directory the corpus needs — a tree registering its vendored
    # linux/ would lose build/meta-bsp/recipes-bsp/linux with it. A bare name
    # already declared by hand (so3's lvgl, micropython) still counts as the
    # same exclusion, so it is not repeated in the other spelling.

    root = os.path.realpath(session_workspace.CORPUS_ROOT)

    for path in corpora_below(projects, session_workspace.CORPUS_ROOT).values():
        rel = os.path.relpath(path, root)

        if rel in excludes or os.path.basename(path) in excludes:
            continue

        excludes.append(os.path.join(".", rel))

    opts = []

    for d in excludes:
        opts += ["--exclude", d]

    if spec.get("include_build"):
        opts.append("--include-build")

    return opts


def reindex_command():
    """The command that rebuilds THIS corpus's index.

    The tree to index is CORPUS_ROOT, never PROJECT_ROOT. The two differ
    whenever the session runs outside its own corpus, which is the normal case
    right after a workspace split: the corpora sit one level below the
    directory you are standing in. Handed the cwd, /reindex walked the umbrella
    instead of the corpus — four sibling trees, the file cap, a refusal — and
    it derived its destination collection from that same cwd, so even under the
    cap it would have filled adhoc_<md5(cwd)>, which no session reads. The one
    this session queries is COLLECTION_NAME, keyed by the corpus tree; name it
    rather than letting the indexer guess it back.
    """
    curated = corpus_indexer() == "buildsystem"
    script = (f"{APP_DIR}/retrieval/index_corpus.py" if curated
              else f"{APP_DIR}/retrieval/index_dir.py")
    # Both indexers are told which collection this session queries. The
    # curated walk used to derive its own from the tree's basename, which
    # agreed with the session only when the registry pinned that same name.

    return ([sys.executable, script, session_workspace.CORPUS_ROOT] + reindex_options()
            + ["--collection", session_workspace.COLLECTION_NAME])
