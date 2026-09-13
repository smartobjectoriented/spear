# Embedding worker

The client offloads corpus embedding to a machine with a GPU. This directory
is everything that machine needs: a wire format and a worker that speaks it.

    protocol.py   the format, and the whole contract. Pure stdlib.
    worker.py     one request in, one response out, then exit.

## The boundary

The client decides what an embedding **means**; this worker decides how a
machine computes it.

| | decided by | travels how |
|---|---|---|
| model identifier | client | in the request |
| prefix (document or query) | client | in the request |
| max sequence length | client | in the request |
| normalisation | client | in the request |
| batch size | client | in the request |
| `trust_remote_code` | client | in the request |
| device (cuda/cpu) | **server** | `SPEAR_EMBED_DEVICE`, else autodetected |
| which card | **server** | `SPEAR_GPU_UUID` |
| dtype | **server** | fp16 on cuda, fp32 on cpu |

There is **no model registry here**, no default prefix, and no per-model
branch. That is the point. The worker this replaced imported the client's
module on the GPU host and read the client's registry from whichever copy was
installed there — so the prefix, the cap and the normalisation were decided by
a file nobody was comparing against the client's. A collection filled with two
different prefixes is inconsistent in a way no later query reports: results
merely get worse.

## Installing it

Copy `server/embed/` to the host, alongside a Python environment that can
import `sentence_transformers`. Nothing else from this repository is needed —
not the client, not chromadb.

```sh
python3 -m venv ~/spear-embed/venv
~/spear-embed/venv/bin/pip install -r requirements.txt
cp protocol.py worker.py ~/spear-embed/
```

Then tell the client where it is. Destination and worker command are separate
because they are separate facts — a destination with no command is a
misconfiguration the client reports by name:

```sh
# on the client
echo 'operator@gpu-host'                       > spear/active-embed-remote.conf
echo '~/spear-embed/venv/bin/python ~/spear-embed/worker.py' \
                                               > spear/active-embed-remote-cmd.conf
```

or `SPEAR_EMBED_REMOTE` and `SPEAR_EMBED_REMOTE_CMD` in the environment. A
leading `~/` is expanded by the remote shell, so the command can name a path
in the remote account without the client knowing its home directory. Every
other word is quoted: a path with a space stays one word and a semicolon
cannot start a second command.

Verify without indexing anything:

```sh
python3 - <<'EOF' | ssh operator@gpu-host '~/spear-embed/venv/bin/python ~/spear-embed/worker.py' | head -c 200
import json
print(json.dumps({"spear_embed_protocol": 1, "model": "BAAI/bge-m3",
                  "texts": ["hello"], "prefix": "", "max_seq_length": 1024,
                  "normalize": True, "batch_size": 16,
                  "trust_remote_code": False}))
EOF
```

A conforming worker answers with a JSON header line whose `status` is `ok`.

## Server-side configuration

| variable | meaning |
|---|---|
| `SPEAR_EMBED_DEVICE` | force `cuda` or `cpu`. Otherwise cuda if torch sees a card. |
| `SPEAR_GPU_UUID` | pin one card on a shared host. A UUID, not an index — an index changes across reboots and with the enumeration order. |
| `HF_HOME` | where the weights are cached. |

The worker is deliberately **not a daemon**: no port to expose on a shared
host, no service to supervise, and it dies with the ssh channel that started
it. Each invocation loads the model, which is why the client sends large
batches — `SPEAR_EMBED_REMOTE_BATCH`, 20000 texts by default — rather than one
request per chunk.

## Versioning

`spear_embed_protocol` is carried and checked in **both** directions. A
mismatch is refused, naming both numbers; it is never negotiated down. Deploy
the worker and the client together.

The failure this rules out is the quiet one: two sides that still parse each
other's bytes while disagreeing about who applies the prefix would produce
vectors that are subtly wrong and perfectly well-formed.

## Failure

Every failure is a JSON header line on stdout and a non-zero exit, so a client
gets something it can act on rather than a traceback that is indistinguishable
from "this command is not a worker".

| `kind` | exit | meaning |
|---|---|---|
| `protocol` | 3 | the two sides disagree. No retry will help. |
| `encoding` | 4 | understood, and this machine could not do it. |

## Equivalence

`spear/tests/test_embed_equivalence.py` compares this worker against the one
it replaces, on real weights, and requires the vectors to be **bit-identical**
— not merely close. Run it before switching a deployment over:

```sh
SPEAR_TEST_EMBED_EQUIVALENCE=1 SPEAR_TEST_EMBED_DEVICE=cuda \
    ./bin/python -m unittest tests.test_embed_equivalence
```
