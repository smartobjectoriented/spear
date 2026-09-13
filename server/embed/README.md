# Embedding worker — the boundary, and the debt

Nothing here yet but this file and the dependency list. That is deliberate,
and the reason is worth writing down, because the obvious move would make
things worse.

## What runs today

The client offloads corpus embedding to the GPU host over SSH. The worker it
invokes is `spear/deploy/embed_worker.py`, and that worker does
`sys.path.insert(...)` followed by `import embedding` — it pulls in the
client's own module and, with it, the client's model registry.

So the **server** currently decides the document prefix, the sequence cap and
the normalisation. Those are retrieval semantics. They belong to the side that
owns the collection, because a collection filled with two different prefixes
is inconsistent in a way no later query reports: results merely get worse.
The registry even says so in `spear/embedding.py` — the prefixes are described
there as functional values worth several points of recall.

It is also how the two sides drifted: the deployed copy of `embedding.py` and
the client's copy are different files, and nothing compares them.

## What belongs here

    server/embed/protocol.py    the wire format. Pure stdlib, no ML stack, so
                                the contract test runs anywhere.
    server/embed/worker.py      protocol + an encoder. No harness import, no
                                model registry, no opinion about retrieval.

The client sends what it has decided — model, prefix, max_seq, normalisation,
protocol version — and the server encodes accordingly. A version field so a
mismatched pair fails loudly instead of returning subtly different vectors.

## Why not now

Moving `embed_worker.py` here today would produce a file under `server/` that
imports `spear/embedding.py`. That is precisely the coupling this directory
exists to remove, and renaming it would only make the coupling harder to see.

Worse, the worker deployed on the GPU host still speaks the old contract. A
protocol change here without a deployment there needs compatibility machinery
on both sides — machinery that exists only to make a directory reorganisation
look finished, and that someone then has to remove.

## Migration debt

`spear/deploy/embed_worker.py` stays where it is, and the client keeps
invoking the deployed copy unchanged. The replacement lands in one step,
together with the client's protocol support, the version handshake, the
numerical-equivalence check against the current worker, and the deployment
that switches over. Until then this directory is a statement of intent and a
dependency list, which is all it can honestly be.
