#!/usr/bin/env python3
"""Embedder bake-off, measured on eval/retrieval.json.

Re-embeds the chunks ALREADY indexed in edgem1_verdin rather than re-walking
the tree: the chunking is then strictly identical from one candidate to the
next, so we measure the embedder and nothing else. Writes into `bench_<tag>`
collections — production is never touched.

    ./bin/python eval/bench_embedders.py BAAI/bge-m3
    ./bin/python eval/bench_embedders.py --list
    ./bin/python eval/bench_embedders.py --report      # summary table

Asymmetric prefixes (e5, Qwen3-Embedding) are declared per model: applying
them on the wrong side costs several points of recall.
"""
import os
import re
import sys
import json
import time
import hashlib

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # xet hangs on this host

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chromadb
import rag_chat

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = "edgem1_verdin"
RESULTS = os.path.join(HERE, "bench_results.json")

# name -> (document prefix, query prefix, trust_remote_code)

MODELS = {
    "chroma-default":                  ("", "", False),   # MiniLM-L6, reference
    "BAAI/bge-m3":                     ("", "", False),
    # Functional value, not prose: this prefix is fed to the model and shapes
    # the query vector. Kept verbatim in French because that is the form
    # bench_results.json was measured with. Mirrors embedding.py.
    "Qwen/Qwen3-Embedding-0.6B":       ("", "Instruct: Retrouve le fichier source "
                                            "ou le code pertinent\nQuery: ", False),
    "intfloat/multilingual-e5-large":  ("passage: ", "query: ", False),
    "jinaai/jina-embeddings-v2-base-code": ("", "", True),
}


def tag(model):
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")[:40]


def source_chunks():
    col = chromadb.PersistentClient(path=rag_chat.DB_PATH).get_collection(SOURCE)
    g = col.get(include=["documents", "metadatas"])

    return g["ids"], g["documents"], g["metadatas"]


def build(model):
    """Create bench_<tag> with the candidate's embeddings. Returns (name, seconds)."""
    doc_pfx, _, trust = MODELS[model]
    name = f"bench_{tag(model)}"
    client = chromadb.PersistentClient(path=rag_chat.DB_PATH)

    if model == "chroma-default":
        return SOURCE, 0.0            # already indexed, this is the reference

    ids, docs, metas = source_chunks()
    from sentence_transformers import SentenceTransformer
    import torch
    print(f"  loading {model} ...", flush=True)
    st = SentenceTransformer(model, device="cuda",
                             trust_remote_code=trust,
                             model_kwargs={"dtype": torch.float16})
    t0 = time.time()
    vecs = st.encode([doc_pfx + d for d in docs], batch_size=16,
                     normalize_embeddings=True, show_progress_bar=True,
                     convert_to_numpy=True)
    dt = time.time() - t0
    print(f"  {len(docs)} chunks encoded in {dt:.0f} s "
          f"({len(docs)/dt:.0f} chunks/s), dim={vecs.shape[1]}")

    try:
        client.delete_collection(name)
    except Exception:
        pass

    col = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(ids), 500):
        j = i + 500
        col.add(ids=ids[i:j], documents=docs[i:j], metadatas=metas[i:j],
                embeddings=vecs[i:j].tolist())

    del st, vecs
    torch.cuda.empty_cache()

    return name, dt


def retrieve(col, qvec, query, hybrid):
    """Reproduces rag_chat.retrieve_context, but on a supplied query vector
    (the candidate is not the collection's embedding function)."""
    top_k = rag_chat.TOP_K

    if qvec is None:
        dense = col.query(query_texts=[query], n_results=top_k * 2,
                          include=["documents", "metadatas"])
    else:
        dense = col.query(query_embeddings=[qvec], n_results=top_k * 2,
                          include=["documents", "metadatas"])

    pool = {i: (d, m) for i, d, m in zip(
        dense["ids"][0], dense["documents"][0], dense["metadatas"][0])}
    lists = [list(dense["ids"][0])]

    if hybrid:
        for term in rag_chat._ident_terms(query):
            try:
                hit = col.get(where_document={"$contains": term},
                              include=["documents", "metadatas"],
                              limit=rag_chat.LEX_SATURATION)
            except Exception:
                continue

            if not hit["ids"] or len(hit["ids"]) >= rag_chat.LEX_SATURATION:
                continue

            ranked = sorted(zip(hit["ids"], hit["documents"], hit["metadatas"]),
                            key=lambda x: -rag_chat._definition_score(x[1], x[2], term))
            ranked = ranked[:rag_chat.LEX_PER_TERM]

            for i, d, m in ranked:
                pool.setdefault(i, (d, m))

            lists.append([i for i, _, _ in ranked])

    order = rag_chat._rrf(lists) if len(lists) > 1 else lists[0]
    total, seen = 0, set()

    for doc_id in order[:top_k * 2]:
        doc, meta = pool[doc_id]

        if total + len(doc) > rag_chat.MAX_CONTEXT_CHARS:
            continue

        total += len(doc)
        seen.add(meta["filepath"])

    return seen


def score(model, coll_name):
    _, q_pfx, trust = MODELS[model]
    questions = json.load(open(os.path.join(HERE, "retrieval.json")))
    col = chromadb.PersistentClient(path=rag_chat.DB_PATH).get_collection(coll_name)

    qvecs = [None] * len(questions)

    if model != "chroma-default":
        from sentence_transformers import SentenceTransformer
        import torch
        st = SentenceTransformer(model, device="cuda", trust_remote_code=trust,
                                 model_kwargs={"dtype": torch.float16})
        qvecs = st.encode([q_pfx + e["q"] for e in questions],
                          normalize_embeddings=True,
                          convert_to_numpy=True).tolist()
        del st
        torch.cuda.empty_cache()

    out = {}

    for mode in ("dense", "hybrid"):
        per = {}
        t0 = time.time()

        for e, qv in zip(questions, qvecs):
            seen = retrieve(col, qv, e["q"], hybrid=(mode == "hybrid"))
            per.setdefault(e["kind"], []).append(any(f in seen for f in e["expect"]))

        allh = [h for v in per.values() for h in v]
        out[mode] = {k: sum(v) / len(v) for k, v in per.items()}
        out[mode]["global"] = sum(allh) / len(allh)
        out[mode]["ms"] = (time.time() - t0) * 1000 / len(questions)

    return out


def report():
    if not os.path.isfile(RESULTS):
        print("no results yet — run a model first")
        return

    res = json.load(open(RESULTS))

    for mode in ("dense", "hybrid"):
        print(f"\n=== {mode.upper()} ===")
        print(f"{'model':<38}{'prose':>7}{'ident':>7}{'path':>7}{'GLOBAL':>9}{'ms':>7}")
        print("-" * 75)

        for m, r in sorted(res.items(), key=lambda x: -x[1][mode]["global"]):
            d = r[mode]
            print(f"{m:<38}{d.get('prose',0):>6.0%} {d.get('identifier',0):>6.0%} "
                  f"{d.get('path',0):>6.0%} {d['global']:>8.0%} {d['ms']:>6.0f}")


def main():
    argv = sys.argv[1:]

    if not argv or argv[0] == "--list":
        print("models:", *MODELS, sep="\n  ")
        return

    if argv[0] == "--report":
        return report()

    rescore = "--rescore" in argv        # replay the questions without re-encoding
    argv = [a for a in argv if not a.startswith("--")]
    model = argv[0]

    if model not in MODELS:
        sys.exit(f"unknown: {model} (see --list)")

    print(f"[{model}]")

    if rescore:
        coll = SOURCE if model == "chroma-default" else f"bench_{tag(model)}"
        build_s = (json.load(open(RESULTS)).get(model, {}).get("build_s", 0)
                   if os.path.isfile(RESULTS) else 0)
    else:
        coll, build_s = build(model)

    r = score(model, coll)
    r["build_s"] = round(build_s)
    res = json.load(open(RESULTS)) if os.path.isfile(RESULTS) else {}
    res[model] = r
    json.dump(res, open(RESULTS, "w"), indent=1)

    for mode in ("dense", "hybrid"):
        d = r[mode]
        print(f"  {mode:<7} prose {d.get('prose',0):.0%}  ident {d.get('identifier',0):.0%}"
              f"  path {d.get('path',0):.0%}  -> GLOBAL {d['global']:.0%}")


if __name__ == "__main__":
    main()
