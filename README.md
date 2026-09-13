# SPEAR — Specification-driven Platform for Embedded Agentic Reasoning

HEIG-VD/REDS. A **fully local, privacy-preserving** AI coding assistant for
embedded source trees — build systems, hypervisors, UI stacks, or any repo
you register. Nothing leaves the machine.

In one sentence: a custom harness around **Qwen3-Coder-Next (80B-A3B, Q8_0,
MoE, served by llama.cpp)**, with **RAG** over my repos, plus persistent
memory and skills that make it learn between sessions.

## Getting started

If you have access to the REDS server and want to *use* this, you do not need
to install anything but Docker:

```sh
git clone <this repo> ~/spear             # anywhere: paths inside are relative
cd ~/spear
docker/build.sh                              # ~20 min, mostly the embedder
docker/spear-docker.sh --reds --auto         # opens the tunnel, then chats
```

The clone location is free: the corpora that live inside the repository are
registered relatively and resolved against it. Only your own source trees are
registered by absolute path, because they genuinely are machine-specific.

That image carries the harness, the embedder and a **prebuilt retrieval
index**; you mount your own source trees and it talks to the model served on
reds-ml over an SSH tunnel. Read `doc/source/container.rst` for what is baked,
what is mounted, and the two `--security-opt` flags without which the harness
refuses to run any command at all.

Retrieval is not a detail: measured on 37 build-system questions, the same
model answers **18-19 %** of them cold and **90 %** with the corpus injected
(measured with a deployment's own recall probe). Running without an index is running a
different, much worse assistant.

To run it natively instead — which is what you want if you intend to *change*
the harness, re-index, or register a corpus — see `deploy/install.sh` and
`doc/source/operations.rst`.

> **Model history.** Started on Qwen3-Coder-30B-A3B + a homemade QLoRA
> adapter (v1→v3, re-trained on RunPod). Migrated 2026-06-08 to stock
> **Qwen3.6-35B-A3B**: the newer base beat the fine-tuned Coder outright, so
> the adapter was dropped. Since 2026-06-23 the served model is
> **Qwen3-Coder-Next (80B-A3B)**, chosen for native tool-calls and targeted
> edits where the dense Coder-32B rewrote whole files; reds-ml serves it Q8_0
> at ~157 t/s.
>
> **No fine-tune served today.** The first runs were done in bf16 on a rented
> B200, on a rented pod, because the weights
> then occupy 152 GB and no single card here holds them.
>
> **That was the expensive way, and it was avoidable** — recorded here because
> the reasoning that led to it looks convincing and is wrong. 4-bit was ruled
> out from reading the *installed* transformers 5.x, where `Qwen3NextExperts`
> fuses the routed experts into 3-D `nn.Parameter` tensors that bitsandbytes,
> which replaces `nn.Linear`, cannot reach. But the published checkpoint stores
> them SEPARATELY (24 576 `mlp.experts.N.{gate,up,down}_proj.weight`), and the
> pod was pinned to transformers **4.57.6**, which loads them as ordinary
> `nn.Linear` — where bnb reaches every one. A 4-bit run would very likely have
> fitted a 48-80 GB card. Under transformers 5.x the fix is not bf16 either:
> Axolotl added `quantize_moe_experts: true` for exactly this, and documents
> ~47 GiB for Qwen3-Next QLoRA without expert targeting, ~71 GiB with.
> Neither figure is ours yet; the next step is a load preflight on reds-ml's
> own card, not another rental.
>
> Those runs did validate the chain end to end — dataset, pod, adapter, GGUF,
> `--lora`, audit — which is what they were for.
>
> **Measured, 2026-08-24: it does not pay.** Two adapters were trained on a
> rented B200 and served on the same Q8_0 base. Without retrieval, the
> usage-shaped corpus helps a little (18.2 % → 22.1 %) and the descriptive one
> hurts (16.9 %). With retrieval — the way the harness actually runs — the
> adapter *costs* six complete answers out of 37 (89.6 % → 83.1 %). It does not
> complement retrieval, it competes with it. So no adapter is served, and the
> pipeline is kept for the thing that would justify one: behaviour, not facts.
> Numbers and reasoning in `qwen3-finetune/cloud/README.md`.

### Why this is more than "RAG + a fine-tune"

Four independent learning loops, each on a different clock:

```
                    ┌──────────────────────────────────────────────┐
  real usage ──/good──► dataset ──QLoRA on RunPod (~3$)──► adapter ─┤
       │            (+ teacher-student distillation: I grade the    │
       │             model's answers vs the real source code)       │
       │                                                            ▼
       └── /remember · save_skill ──► instant, local, no retraining ──► better
                                                                     model
```

- **RAG** brings *content* on the fly (indexed code/docs, refreshed by
  reindex — no retraining). ChromaDB, ~per-project collection.
- **QLoRA adapter** brings *behavior* (domain style, vocabulary, reflexes)
  frozen into a 26 MB adapter served via `--lora` on top of the frozen
  base — only ~0.1% of the weights are trained, hence ~3$/run.
- **Memory + skills** are immediate, local context the model writes itself
  (with confirmation) and recalls across sessions.

Comparable to Nous Research's **Hermes Agent** for the contextual learning
loop (skills + memory), but it *also* learns in the weights (fine-tuning)
and runs 100% offline.

## Architecture

```
┌─ laptop (RTX 4060 8GB + 62GB RAM) ─────────────────────────────────┐
│  spear-chat ──► llama-server :8080                                 │
│   │           Qwen3-Coder-Next 80B-A3B  Q8_0 (MoE, 512 exp)      │
│   │              hybrid offload: attention→GPU, experts→RAM        │
│   │              ~7 t/s, no network needed                         │
│   ├─ tools: bash · edit_file · append_file · write_file ·          │
│   │         web_search · remember · save_skill · search_history    │
│   │         (OpenAI tools API, parsed by --jinja)                  │
│   ├─ RAG: ChromaDB, one collection per project                     │
│   └─ knowledge: system-prompt.md · tool-guide.md · rules.d/ ·      │
│                 memories-*.md · skills/ · history                  │
└────────────────────────────────────────────────────────────────────┘
   optional fine-tuning: build datasets locally → QLoRA on a GPU
   (RunPod ~3$, or the REDS RTX PRO 6000 Blackwell 96GB) → adapter via --lora
```

Default: **Qwen3-Coder-Next 80B-A3B Q8_0, no adapter** (~79 GB, served on
reds-ml's RTX PRO 6000). Q8 was chosen for max accuracy: measured against Q4,
it costs 7 % of throughput to double the precision of the weights. Override
the model per run with
`SPEAR_SERVER_MODEL=/path/to.gguf spear-server`; load an adapter with
`SPEAR_SERVER_LORA=...`. Under `spear-chat --local` the client's own
`SPEAR_MODEL`/`SPEAR_LORA` are mapped onto them.

## Corpora (the central concept)

Everything is a **corpus**: a working tree with its own RAG index, history
and memories. Two kinds:

- A corpus declares what it does: `indexer: buildsystem` selects the curated
  BitBake/Yocto walk (`index_corpus.py`), `autoindex` builds a missing index
  on sight, `prompt_file` gives it a domain prompt, `collection` names an
  existing index instead of deriving one from the path. `kind` is a label.
- **generic** — any other repo (lvgl, so3, u-boot…): generic index
  (`index_dir.py`), collection `adhoc_<path-hash>`, generic prompt.

The active corpus is auto-detected from the cwd; otherwise a picker lists
them (banner shown first). Launching in an unregistered multi-component
workspace offers to split it into one corpus per big sub-tree (so a huge
upstream tree like u-boot never dilutes the index).

| Command | Purpose |
|---|---|
| `spear-chat` | assistant (auto-detects corpus from cwd; picker otherwise) |
| `spear-chat --corpus lvgl` | open a registered corpus |
| `spear-chat --here` | ad-hoc on the CURRENT directory |
| `spear-chat -y` | bypass permissions (auto-accept tool actions, network included) |
| `spear-chat --no-network` | offline: no network in the shell, and the web tools are not exposed |
| `spear-chat --ctx 65536 --temp 0.1 …` | session settings that used to be environment variables; `--help` lists them all |
| `spear-corpus list/add/rm/scan` | manage the corpus registry (= `/corpus` in the chat) |
| `spear-server` | llama-server alone (`SPEAR_MODEL`/`SPEAR_LORA` overrides) |
| `spear-reindex [path]` | rebuild a corpus with the curated walk |
| `spear-index [dir] [--max-files N] [--exclude D]` | index any tree |

`spear-corpus scan <workspace>` splits a multi-component tree into
per-component corpora by file count; the chat offers the same split
automatically when you open such a workspace. The CLI and the in-chat
`/corpus` are one implementation — same registry, same walk.

### Runtime tracing

Tracing is disabled by default. Enable provider-neutral JSONL traces for a
benchmark run with:

```console
SPEAR_TRACE=1 spear-chat
```

Events are appended to `spear/audit/runtime-trace.jsonl`. Set
`SPEAR_TRACE_FILE=/path/to/trace.jsonl` to choose another location. Traces
record timing, counts, provider/model identifiers, normalized outcomes, and
safe tool metadata; they do not record raw prompts, model responses, command
strings, tool content/query/note values, environment variables, or
authorization data. Workspace-relative file paths and tool argument names are
retained to support debugging and benchmark analysis.

### Agent architecture

The interactive application delegates each task to a UI-free
`TaskController`. It coordinates the common `AgentRuntime`, grounded
`WorkingState`, layered `ContextEngine`, optional isolated Explorer and
Reviewer, verification, checkpoints, resumable sessions and hierarchical
budgets. See [the architecture documentation](doc/source/architecture.rst).

Durable memories remain editable `memories-*.md` files. Structured IDs, scope,
tags and explicit supersession live in an optional adjacent
`.metadata.json` sidecar; only a bounded relevant active selection enters task
context.

## In-chat reference

- `!read !ls !grep !find !edit !run !web` — direct tools, no LLM
- `/search <q>` `/reindex` `/history` `/skills` `/undo` `/clear` `/tools`
- The model's web tools: `search_internet` finds pages, `fetch_url` reads
  one (HTML, PDF, plain text) or SAVES it to disk. They run in the chat
  process, not in the sandboxed shell, so reading works in safe mode — but
  `fetch_url` is http/https only and refuses any host resolving to a
  private, loopback or link-local address: it reaches the internet, not the
  LAN the assistant happens to sit on. A long PDF comes back in slices and
  the result names the `pages` range that continues it. `save_as=<path>`
  downloads the file itself (never a truncated one) and is a mutation, so it
  obeys the permission mode like any write — that is the route to a document
  you mean to `/standard ingest`. `--no-network` removes both tools.
- `/corpus [list|add|rm|scan]` — the registry, without leaving the session.
  Registering does not switch corpus (`spear-chat --corpus <name>` does),
  but a tree registered inside another is excluded from it at the next
  `/reindex` — which is the usual reason to register one.
- `/remember <note>` — add a long-term memory for THIS corpus (no arg: list them)
- `/recall <rule>` — add a rule for EVERY corpus, injected into every request
  (no arg: list them). Use it for what holds everywhere — "an existing
  copyright header is never rewritten" — and `/remember` for what is true of
  one tree only.
- `/forget <regex>` — prune the history-search index (the archive file is kept)
- `/good` — save the last exchange as a fine-tuning sample
  (`experience.jsonl`); `/bad` — log a bad one (never trained on)
- Multi-line paste is supported; ctrl+c interrupts generation; Enter
  confirms tool prompts; arrows/Home/End/ctrl+r via readline.

## Knowledge layers (different clocks)

| Layer | Latency | Where |
|---|---|---|
| memories (remember tool, `/remember`) | next turn | `memories-*.md` |
| learned rules (`/recall`) | next launch, every corpus | `rules-learned.md` (under `STATE_DIR`) |
| skills (save_skill, learned procedures) | next turn (similarity-injected, scope/prerequisite gated) | `skills/*.md` |
| rules (project conventions) | next launch | `rules.d/*.md` |
| tool behavior rules | next launch | `tool-guide.md` |
| RAG corpus | after `/reindex` | `chromadb/` |
| history search (cross-session recall) | continuous | `edgem_archive` collection |

Qwen3 models have a **thinking mode** — the chat disables it via
`chat_template_kwargs.enable_thinking=false` (rag_chat.py); leave it off for
direct answers, it only burns tokens here.

The model USES its memory correctly but verbally denies having one when
asked — pretraining reflex, ignore its self-description and judge by
behavior. If a bad answer lands in history, `/undo` it (the model imitates
its own past answers).

## Web access

1. Freshness questions ("dernière/latest version ...") auto-trigger a
   search before the LLM (results + sources shown Claude-style).
2. Phrases like "cherche sur internet ..." auto-trigger too.
3. `!web <query>` — manual.
4. The model's own `web_search` tool calls — the Qwen3 bases are agentic
   and calls tools readily; verify its native tool-call format works in
   the harness on a real task.

## Indexers

- `index_corpus.py` — `indexer: buildsystem` only: curated walk (build/, avz/,
  scripts/, doc/, home_assistant/), buildroot-defconfig patch exception,
  per-layer categories in chunk metadata. Don't use elsewhere.
- `index_dir.py` — any tree: wide extension list, standard exclusions,
  200KB/file, a 60000-file runaway guard that REFUSES rather than truncates
  (`--exclude`, `--max-files`, `--allow-partial`), collection
  `adhoc_<path-hash>`.
- `/reindex` inside the chat dispatches to the right one automatically. It
  rebuilds the CORPUS's index — its tree, its collection — which is not the
  cwd whenever the session was launched outside its corpus. Corpora
  registered under that tree are excluded: they have their own indexes.

## Fine-tuning (see `qwen3-finetune/`)

Generic machinery only: a QLoRA trainer, a merge-and-quantize step, resumable
weight downloads, and `cloud/load_preflight.sh` — which answers empirically
whether a card holds the model loaded the way training will load it, rather
than deducing it from a documentation figure. That answer was guessed twice
and was wrong twice, once at the cost of a rented B200.

It trains nothing by itself. A run needs a dataset, and a dataset comes from
your own trees: harvesting a project's sources, documentation and commit
history into samples is deployment-specific work, under a system prompt that
describes that project, and no corpus builder is shipped here.

The contract is small — a builder writes JSONL lines of
`{"messages": [...]}`, `scripts/merge_corpora.py` combines sample sets
without unifying their system prompts, and `scripts/train_qlora.py` trains on
the result. `/good` in the chat appends validated exchanges to
`$SPEAR_STATE_DIR/experience.jsonl` (override with `SPEAR_EXPERIENCE_FILE`),
which is one such source and is read by nothing unless you point a run at it.

## Server notes

- `server/inference/serve.sh` holds no flag values. Context size, model,
  binary, port, threads and card come from `server/config/server.conf` (or
  `SPEAR_SERVER_*` in the environment); see `server/config/server.conf.example`.
  The script itself contributes only what is a property of the build rather
  than of a deployment: KV cache `q8_0`, `--flash-attn`, `--jinja` (required
  for the chat template and tool calling), and `--n-cpu-moe` for a
  mixture-of-experts model unless told otherwise. Context size has no default
  at all: it is the value two earlier implementations disagreed about.
- Sampling lives in `rag_chat.py` (temp 0.7,
  top_p 0.8, top_k 20, repeat_penalty 1.05 — lower temperatures cause
  repetition death-loops; a stream circuit-breaker truncates them anyway).
- Run as a transient unit to survive sessions:
  `sudo systemd-run --unit=spear-llm /opt/llm/spear/server/inference/serve.sh
  --lora <adapter>`.

## Safety guards (earned the hard way)

- read-only commands auto-approved; anything else asks (Enter = yes)
- `write_file` refuses to overwrite a file much larger than the proposed
  content (LLM "full rewrites" of big files are hallucination magnets)
- per-turn command budget (15) + round cap + result cache (no re-runs)
- anti-fabrication: invented "[tool]" output is stripped and retried
- grounded edit errors: a failed exact-match EDIT returns the file's real
  tail and points the model to append_file

## License

SPEAR is licensed under the Apache License, Version 2.0.
See [LICENSE](LICENSE) for details.
