# SPEAR — Specification-driven Platform for Embedded Agentic Reasoning

[![Documentation](https://img.shields.io/badge/documentation-spear-blue)](https://smartobjectoriented.github.io/spear/)

**📖 Full documentation: <https://smartobjectoriented.github.io/spear/>**

HEIG-VD/REDS. SPEAR combines authoritative specifications, project source
code and controlled agentic workflows to support **evidence-grounded
engineering**.

It is built for tasks where an agent must reason from an authoritative
technical source, inspect an implementation, make controlled changes to it,
and retain the evidence for every conclusion it reports. A specified system
has two sources of truth — the specification says what is *required*, the code
says what it *does* — and SPEAR keeps the two roles distinct rather than
letting one stand in for the other.

Everything runs on your own machine. Nothing leaves it unless a tool call is
explicitly granted network access.

## What it does

- **Authoritative-source grounding** — bind a specification; normative
  questions are answered from it first, and every claim carries its citation.
- **Codebase-aware reasoning** — registered trees are indexed and retrieved
  from, so answers come from *your* project.
- **Controlled code modification** — investigate, plan, edit, test, review; a
  file is writable because a planned item named it.
- **Validation-aware workflows** — how a change will be proved is decided
  before the code is written.
- **Traceable evidence** — the closing report is generated from the record of
  what was retrieved and run, not from the agent's own summary.
- **Multiple model backends** — any OpenAI-compatible endpoint, local or
  remote, and the Anthropic API.
- **Confined execution** — fail-closed: a confinement that cannot be applied
  is an error, never a silent downgrade.

Where the evidence does not support a claim, SPEAR withholds the answer and
says why. "Incomplete" is a result, not a failure.

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
| use it day to day — commands, `/remember`, guards | [spear-chat](https://smartobjectoriented.github.io/spear/usage.html) |
| bind a specification and read normative answers | [Authoritative standards](https://smartobjectoriented.github.io/spear/standards.html) |
| understand how a change is made under control | [The engineering workflow](https://smartobjectoriented.github.io/spear/workflow.html) |
| know why an answer was withheld | [Evidence and guards](https://smartobjectoriented.github.io/spear/guards.html) |
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
