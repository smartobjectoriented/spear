# Retrieval eval harness

Measures what **actually** lands in the prompt, without ever calling the LLM:
a few seconds per run, so it is usable on every change.

    ./bin/python eval/run_eval.py                      # edgem1_verdin
    ./bin/python eval/run_eval.py --collection adhoc_xxxxxxxx
    ./bin/python eval/run_eval.py --json /tmp/run.json # per-question detail

## Metric

`recall@k` = did the question see at least one of its expected files in the
injected context? We read the `seen` set returned by `retrieve_context`, hence
**after** truncation to `MAX_CONTEXT_CHARS`: a chunk that was found but cut by
the budget counts as a failure — that is what the model sees.

Three question classes, because they fail for different reasons and are fixed
by different levers:

| class        | example                                | lever                  |
|--------------|----------------------------------------|------------------------|
| `prose`      | "comment ajouter un paquet au rootfs"  | embedding quality      |
| `identifier` | "ou est defini `__sys_empty`"          | lexical channel        |
| `path`       | "montre moi `rootfs.bbclass`"          | file-path scoring      |

The questions themselves stay in French: that is how they are actually asked,
and French-over-English-code is precisely what the embedder has to handle.

## Ground truth

`retrieval.json`. The `identifier` targets were picked among identifiers
**present in exactly one file** of the corpus, verified by a full scan at
generation time — no ambiguous target. The `prose` targets are a human
judgement.

Regenerate after a major `/reindex`: the expected paths must exist in the
collection, which the original script asserted. That happened during the bge-m3
migration — the tree had been reorganised (the `.its` files moved from
`linux/target/` to `meta-bsp/recipes-bsp/linux/files/its/`, some memories were
renamed) and 5 targets pointed at nothing.

## History

Measured over the 40 questions (`prose` n=24, `identifier` n=12, `path` n=4).
The column that matters is HYBRID: that is what production does.

| embedder                       | dense | hybrid  | prose (hyb.) | indexing |
|--------------------------------|-------|---------|--------------|----------|
| chroma-default (MiniLM-L6)     |  57%  |   80%   |     67%      | CPU      |
| **BAAI/bge-m3**                |  80%  | **92%** |   **88%**    | 34 chk/s |
| Qwen/Qwen3-Embedding-0.6B      |  72%  |   88%   |     79%      | 14 chk/s |
| intfloat/multilingual-e5-large |  48%  |   82%   |     71%      | 34 chk/s |

The lexical channel puts EVERY embedder at 100% on `identifier` and `path`: it
is an exact search, independent of the model. So embedders can only be told
apart on `prose` — which is why that class grew from 10 to 24 questions. On the
initial 26-question set, prose was saturated at 80% for all four and the bench
wrongly concluded that a bigger embedder brought nothing.

e5-large collapses on `identifier` in dense mode (8%): its window is 512 tokens
while chunks are 1500 characters — it only sees a third of the chunk. That is
not a flaw of the model, it is an incompatibility with our chunking.

Reproduce:

    ./bin/python eval/bench_embedders.py BAAI/bge-m3            # build + score
    ./bin/python eval/bench_embedders.py BAAI/bge-m3 --rescore  # score only
    ./bin/python eval/bench_embedders.py --report

## Production

`edgem1_verdin` moved to bge-m3 (4260 chunks). End-to-end measurement on the
real collection:

    prose 75%   identifier 100%   path 100%   ->  GLOBAL 85%
    latency 110 ms steady-state (211 ms before, on MiniLM)
    + 9.6 s on the very first call of the session (model load)

That 85% is NOT comparable to the bench's 92%: the bench works on the 3524
chunks of the old index, production on the current 4260 — more content
competing for the same 12 slots. The bench stays the valid way to COMPARE
embedders (identical chunks); production measures the system as it runs.
