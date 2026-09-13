# SPEAR — Specification-driven Platform for Embedded Agentic Reasoning

[![Documentation](https://img.shields.io/badge/documentation-spear-blue)](https://smartobjectoriented.github.io/spear/)

**📖 Full documentation: <https://smartobjectoriented.github.io/spear/>**

HEIG-VD/REDS. A **fully local, privacy-preserving** AI coding assistant for
embedded source trees — build systems, hypervisors, UI stacks, or any repo
you register. Nothing leaves the machine.

In one sentence: a custom harness around **Qwen3-Coder-Next (80B-A3B, Q8_0,
MoE, served by llama.cpp)**, with **RAG** over your repos, plus persistent
memory and skills that make it learn between sessions.

## Quick start

If you have access to a model server and want to *use* this, you do not need
to install anything but Docker:

```sh
git clone https://github.com/smartobjectoriented/spear ~/spear
cd ~/spear
docker/build.sh                              # ~20 min, mostly the embedder
docker/spear-docker.sh --reds --auto         # opens the tunnel, then chats
```

To run it natively instead — which is what you want if you intend to *change*
the harness, re-index, or register a corpus — start from
`spear/deploy/install.sh`.

Both paths, and the two `--security-opt` flags without which the harness
refuses to run any command at all, are in
[Getting started](https://smartobjectoriented.github.io/spear/getting_started.html).

Retrieval is not a detail: measured on 37 build-system questions, the same
model answers **18–19 %** of them cold and **90 %** with the corpus injected.
Running without an index is running a different, much worse assistant.

## What is here

| Directory | What it holds |
|---|---|
| `spear/` | the client: chat, agent runtime, retrieval, and the execution harness |
| `server/` | the generic inference server component (llama.cpp launcher, model fetch, embedding) |
| `doc/` | the Sphinx documentation published at the link above |
| `docker/` | the container build and launcher |
| `qwen3-finetune/` | the fine-tuning machinery — a QLoRA trainer, a merge-and-quantize step, a load preflight |

## Where to read what

| If you want to | Read |
|---|---|
| run it | [Getting started](https://smartobjectoriented.github.io/spear/getting_started.html) |
| use it day to day — commands, `/remember`, guards | [Using the assistant](https://smartobjectoriented.github.io/spear/usage.html) |
| understand the confinement — the reason this exists | [Tool execution harness](https://smartobjectoriented.github.io/spear/tool_harness.html) and [Security model](https://smartobjectoriented.github.io/spear/security_model.html) |
| understand retrieval and corpora | [Retrieval](https://smartobjectoriented.github.io/spear/retrieval.html) |
| know why this model, and what fine-tuning measured | [Model history](https://smartobjectoriented.github.io/spear/model_history.html) |
| fine-tune something | [Training and fine-tuning](https://smartobjectoriented.github.io/spear/training.html) |
| hand it to someone else | [Container](https://smartobjectoriented.github.io/spear/container.html) |

The documentation builds locally too:

```sh
pip install -r doc/requirements.txt
make -C doc html          # doc/build/html/index.html
```

## License

SPEAR is licensed under the Apache License, Version 2.0.
See [LICENSE](LICENSE) for details.
