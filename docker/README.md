# SPEAR harness in a container

Hands the whole assistant to someone else: the harness, its dependencies, the
embedder and a prebuilt retrieval index. They mount their own checkouts and
point it at a model endpoint.

## What is in the image, and why

| | where | why |
|---|---|---|
| harness + venv | image | pinned, reproducible |
| **bge-m3 embedder** (4.3 GB) | image | downloading it on first run is a 4.3 GB surprise on a machine that may have no HF access |
| **ChromaDB index** (8.5 GB) | image | re-indexing takes hours *and* needs every corpus tree present — the one thing a newcomer does not have |
| source trees | **mounted** | they are working copies and change daily; an image would be stale the next morning |
| model weights | **neither** | the harness talks to an endpoint (`--endpoint`), it does not host a model |

## Run

```sh
docker build -f docker/Dockerfile -t spear:1.0 .     # from the repo root
docker/spear-docker.sh                                  # reference layout
docker/spear-docker.sh --corpora ~/work --endpoint http://127.0.0.1:8082
```

The reference layout binds your corpus roots and
`/opt/llm/spear` under `/corpora`. With `--corpora DIR`, that one directory
becomes `/corpora` and must hold the four names above (whichever you have).

## Corpus paths are relative, on purpose

`docker/projects.docker.json` registers every corpus **relative** to
`SPEAR_CORPUS_ROOT` (`/corpora` in the image), where the workstation registry
uses absolute paths under `/home/operator`. That is what makes the same image
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
