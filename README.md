# SPEAR — Specification-driven Platform for Embedded Agentic Reasoning

[![Documentation](https://img.shields.io/badge/documentation-spear-blue)](https://smartobjectoriented.github.io/spear/)

**📖 Full documentation: <https://smartobjectoriented.github.io/spear/>**

HEIG-VD/REDS. SPEAR is an engineering agent for **specified systems** — code
that has to do what a standard, a specification or a datasheet requires.

A specified system has two sources of truth: the specification says what is
*required*, the code says what it *does*. A general coding agent merges them,
and reports its own reading of a document as the document. SPEAR keeps them
apart, and keeps the agent that changes the code apart from the evidence that
judges the change:

| Concern | In SPEAR |
|---|---|
| agentic implementation | a standalone **coding core** — six tools, nothing else |
| control and policy | **SpearHost** and a fail-closed sandbox around every call |
| execution evidence | what each call actually did, recorded structurally |
| normative reasoning | provisions of a bound standard, retrieved, identified and cited |
| compliance evidence | deterministic source predicates and project-bound conformance checks — never a model's opinion |
| final verdicts | computed from that evidence on the **final** source state |

Everything runs on your own machine. Nothing leaves it unless a command is
explicitly granted network access.

## How a request runs

```text
                         user request
                              │
                        TaskController
             ┌────────────────┼─────────────────┐
     IMPLEMENTATION       NORMATIVE            MIXED
             │                 │                 │
        coding core     normative runtime   normative pre-pass
             │                 │            constraint packet
         SpearHost        provisions,       coding core + SpearHost
             │            answer guards     final normative adjudication
             ▼                 ▼                 ▼
  VERIFIED / UNVERIFIED   cited answer,     composite verdict, e.g.
       / NO_CHANGE        or withheld       COMPLIANCE NOT DEMONSTRATED
```

- **Implementation** — a change runs on the coding core inside a contained
  workspace. It is `VERIFIED` only if the checks that show what the answer
  claims ran and passed *after the last change*.
- **Normative** — a question about a bound standard is answered from the
  document first, every normative claim cited; an unsupported answer is
  withheld with the reason.
- **Mixed** — a change the standard governs is implemented against a compact,
  structurally complete constraint packet and judged constraint by constraint.
  Where authoritative evidence is missing, the verdict is *compliance not
  demonstrated* — never a guess in either direction.

## Quick start

With Docker and access to a model server:

```sh
git clone https://github.com/smartobjectoriented/spear ~/spear
cd ~/spear
scripts/docker/build.sh                              # ~20 min, mostly the embedder
scripts/docker/spear-docker.sh --reds --auto         # opens the tunnel, then chats
```

To run it natively — to change the harness, index your projects or bind your
own standards — start from `spear/deploy/install.sh`. Both paths, and the
three `--security-opt` flags without which the sandbox cannot start in a
container, are in
[Getting started](https://smartobjectoriented.github.io/spear/start/getting_started.html).

## Where to read what

| If you want to | Read |
|---|---|
| run it | [Getting started](https://smartobjectoriented.github.io/spear/start/getting_started.html) |
| understand the design | [Architecture](https://smartobjectoriented.github.io/spear/overview/architecture.html) |
| change code with it | [Implementation mode](https://smartobjectoriented.github.io/spear/reasoning/implementation.html) |
| bind a standard and ask about it | [Normative mode](https://smartobjectoriented.github.io/spear/reasoning/standards.html) |
| make a change a standard governs | [Mixed mode](https://smartobjectoriented.github.io/spear/reasoning/mixed.html) |
| read a verdict | [Evidence and verdicts](https://smartobjectoriented.github.io/spear/reasoning/evidence.html) |
| configure a project and its conformance checks | [Projects](https://smartobjectoriented.github.io/spear/using/projects.html) |
| use it day to day | [spear-chat](https://smartobjectoriented.github.io/spear/using/usage.html) |
| understand the confinement | [Security model](https://smartobjectoriented.github.io/spear/harness/security_model.html) |
| serve or fine-tune a model | [The model](https://smartobjectoriented.github.io/spear/model/model.html) |

The documentation builds locally too:

```sh
pip install -r doc/requirements.txt
make -C doc html          # doc/build/html/index.html
```

## What is here

| Directory | What it holds |
|---|---|
| `spear/` | the client: chat, the coding core and its control plane, the normative runtime, retrieval, and the execution harness |
| `server/` | the generic inference server component (llama.cpp launcher, model fetch, embedding) |
| `doc/` | the Sphinx documentation published at the link above |
| `docker/`, `scripts/` | the container image and the release, configuration and image scripts |
| `qwen3-finetune/` | the fine-tuning machinery — a QLoRA trainer, a merge-and-quantize step, a load preflight |

## Status

SPEAR is research software from the [REDS institute](https://reds.heig-vd.ch)
of HEIG-VD, in active development.
Development happens on `main`; every minor version gets a long-lived
`release/vX.Y` branch on which patch releases are tagged — see
[Release process](https://smartobjectoriented.github.io/spear/contributing/release_process.html)
and the [Releases page](https://github.com/smartobjectoriented/spear/releases).

| Line | Branch | Latest release | Status |
|------|--------|----------------|--------|
| 0.3  | [`release/v0.3`](https://github.com/smartobjectoriented/spear/tree/release/v0.3) | [v0.3.0-rc1](https://github.com/smartobjectoriented/spear/releases/tag/v0.3.0-rc1) | Release candidate |
| 0.2  | [`release/v0.2`](https://github.com/smartobjectoriented/spear/tree/release/v0.2) | [v0.2.0](https://github.com/smartobjectoriented/spear/releases/tag/v0.2.0) | Current stable |

## License and provenance

SPEAR is licensed under the Apache License, Version 2.0 — see
[LICENSE](LICENSE). The coding core is SPEAR's own module, built by porting
parts of Hermes Agent (MIT); every copied or adapted file, with its upstream
revision and license, is listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
