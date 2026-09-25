# SPEAR harness in a container

Hands the whole assistant to someone else: the harness, its dependencies, the
embedder, a prebuilt retrieval index, the normative store and — when a build
asks for it — the corpus trees themselves. They point it at a model endpoint
and have the same state you do.

## What is in the image, and why

| | where | why |
|---|---|---|
| harness + venv | image | pinned, reproducible |
| **bge-m3 embedder** (4.3 GB) | image | downloading it on first run is a 4.3 GB surprise on a machine that may have no HF access |
| **ChromaDB index** (8.5 GB) | image, *if the building host has one* | re-indexing takes hours *and* needs every corpus tree present — the one thing a newcomer does not have |
| rules, skills, benches, notes | image, *if present* | a deployment's own content; `build.sh` says which it found |
| **normative store** | image, *filtered by profile* | a harness that can bind a standard and no standard to bind answers every normative question from the source tree — the one failure this platform exists to prevent |
| corpus trees | **mounted**, or baked with `--bake` | working copies change daily; baked is for the container that is handed over with nothing to mount |
| model weights | **neither** | the harness talks to an endpoint (`--endpoint`), it does not host a model |

## Two profiles

The image can carry a normative store and the corpus trees themselves, which
is what makes it something you hand over rather than something you mount into.
What it may carry is a per-build decision, and it defaults to the safe one.

```sh
scripts/docker/build.sh --profile public     --bake so3,so3-doc,avz
scripts/docker/build.sh --profile engagement --bake so3,acme-firmware
```

**`public`** (the default) takes only normative documents that declare
themselves `PUBLIC`, and nothing of a customer's. Safe to give to anyone.

**`engagement`** takes everything this machine has — licensed documents, and
the original PDFs where the store retained them. The image is labelled
`redistributable=false` and `scripts/docker/push.sh` refuses to publish it without
`--allow-push`.

Which is which is never a list kept here. Each ingested document already
records `source_origin` and `raw_pdf_retained` in its manifest, and
`stage-standards.py` reads them: a list in this repository would have to name
a customer's standard in order to exclude it, would go stale on the next
ingestion, and would put the whole decision one forgotten edit away from
shipping a licensed PDF. A document that declares nothing is treated as
licensed.

The build says what it took and what it left:

```
   standards NIST-RS274NGC NISTIR6556  (PUBLIC)
   standards ACME-1234.5 2019-R2023  (LICENSED_STANDARD) — LEFT OUT of a public image
   standards binding dropped (… is not in this image) — the container opens unbound
```

The active binding travels only if the document it names travelled. A binding
pointing at an absent store is worse than none: the session opens looking
bound and answers from nothing. Where the image carries exactly one document,
the entrypoint binds it at startup — through the harness, so the fingerprints
are computed rather than fabricated.

## Baking the trees

`--bake` copies named registered corpora into the image, at the paths the
generated registry already resolves them to. Nothing is baked by default, and
a bind mount on `/corpora` still shadows whatever was, so a workstation keeps
behaving exactly as before.

A baked corpus must be **registered and indexed on the host first**: the image
ships the index, and a tree without its collection answers with no retrieval
at all. Registering the engagement's pilot tree is therefore a prerequisite of
putting it in an image, not an alternative to it:

```sh
spear-corpus add acme-firmware "$SPEAR_PILOT_TREE"
spear-index "$SPEAR_PILOT_TREE"
scripts/docker/build.sh --profile engagement --bake so3,acme-firmware
```

When `--bake` is used the image registry is restricted to what the image
actually carries. Otherwise the recipient opens the container to a list of
twenty-two corpora they do not have.

## Handing it over

```sh
scripts/docker/push.sh ghcr.io/<org>/spear:1.0-public          # public: goes
scripts/docker/push.sh ghcr.io/<org>/spear-private:1.0-engagement
                                                       # engagement: refused
scripts/docker/push.sh --allow-push ghcr.io/<org>/spear-private:1.0-engagement
docker save spear:1.0-engagement | zstd -T0 -19 -o spear-engagement.tar.zst
```

The guard reads the label, not the tag: a tag gets retyped and shortened, a
label travels with the bytes through `docker save`, a registry and back. An
image not built by `build.sh` carries no label and is refused rather than
guessed at.

`--allow-push` on an engagement image is a decision about a licence and a
contract, not about a registry. It should cost a deliberate word.

## Run

```sh
scripts/docker/build.sh                                      # never `docker build`
scripts/docker/spear-docker.sh                               # reference layout
scripts/docker/spear-docker.sh --corpora ~/work --endpoint http://127.0.0.1:8082
```

A fully baked image needs nothing mounted:

```sh
docker run --rm -it \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  --security-opt systempaths=unconfined \
  spear:1.0-public --endpoint http://host.docker.internal:8082
```

`build.sh` is not a convenience wrapper: the image is assembled from several
named build contexts, and a bare `docker build` fails on the first of them.

The reference layout binds your corpus roots and
`/opt/llm/spear` under `/corpora`. With `--corpora DIR`, that one directory
becomes `/corpora` and must hold the four names above (whichever you have).

## Corpus paths are relative, on purpose

`docker/projects.docker.json` is generated by `build.sh` from this machine's
`projects.json`; it is not in the repository, and a clone without one gets an
empty registry. It registers every corpus **relative** to `SPEAR_CORPUS_ROOT`
(`/corpora` in the image), where the workstation registry uses absolute paths
under `/home/operator`. That is what makes the same image
work for someone whose checkouts live elsewhere. `resolve_corpus_path()` leaves
absolute paths untouched, so the workstation keeps behaving exactly as before.

A tree you do not have is not an error: the entrypoint prints what resolved and
what did not, at startup, rather than letting a missing corpus show up three
questions later as an empty retrieval.

## The two flags that are not optional

```
--security-opt seccomp=unconfined --security-opt apparmor=unconfined
```

The harness runs **every** command inside bubblewrap and refuses to run any
without it. Docker's default seccomp profile blocks `clone(CLONE_NEWUSER)`, so
bwrap cannot start, and the failure surfaces as "sandbox unavailable" — which
reads like a broken harness rather than a missing run flag. `spear-docker.sh`
passes both, and the entrypoint checks bwrap first and says exactly this if it
cannot.

Neither flag grants the container new privileges on the host: both are about
letting an *unprivileged* namespace be created inside it. The sandbox is still
what confines the model's commands, and it is still doing its job.

## Refreshing the index

The baked index is a snapshot. To ship a newer one, re-index on the workstation
and rebuild the image — `chromadb/` is copied late in the Dockerfile, so only
that layer and the ones after it are rebuilt.

```sh
spear-index /path/to/tree        # updates chromadb/ on the workstation
docker build -f docker/Dockerfile -t spear:1.1 .
```
