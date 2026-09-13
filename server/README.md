# server/ — the generic server side of SPEAR

What runs on the machine that holds the GPU. Public, and deliberately
ignorant of any particular deployment: every path, port and size comes from
configuration that lives outside this checkout.

```
inference/serve.sh              run llama.cpp from configuration
inference/install-llamacpp.sh   build it, with CUDA, where serve.sh expects it
scripts/fetch-model.sh          fetch a GGUF, shards and all
config/server.conf.example      what a deployment must say
config/gpu.conf.example         which card it may use
embed/                          the embedding worker's boundary — see its README
tests/                          runs without a GPU
```

## One repository, one commit

`spear/` and `server/` version together, on purpose. The two halves talk over
a wire protocol, and a protocol change that is one commit on one side and
another commit on the other is a protocol that can be half-deployed. Here,
"client commit X is tested against server commit X" is a statement about a
single hash. Neither a submodule nor a second repository can say that without
a pointer somebody has to remember to move.

## The boundary

The client owns what the answer means: retrieval semantics, corpus identity,
the vector database, the agent runtime, standards handling, and orchestration
of the remote host.

The server owns what the machine does: starting the inference process,
loading weights, choosing a GPU, encoding.

`spear/` must never import `server/`. A test enforces it. Tests may look at
both sides — that is what validating a contract means — but nothing in the
client runtime may depend on the server tree being present.

## Runtime assets live outside the checkout

Weights, CUDA build trees, caches, logs and the real configuration are not in
Git and are not under it:

```
~/spear/                     this checkout          (tens of MB)
~/spear-runtime/
    models/gguf/             weights
    llama.cpp/               build tree
    embed/venv/              the worker's environment
    log/
    config/                  server.conf, gpu.conf — the real ones
```

The tracked code assumes nothing about those locations beyond what
configuration tells it. A repository that also holds 80 GB of weights is a
repository nobody can clone.

## Nothing here names a machine

No hostname, no address, no username, no key path, no GPU UUID, no model
path. Those belong to `~/spear-runtime/config/`, which no `git add` can
reach. A test checks this tree for them, so the check survives the people who
remember why it matters.
