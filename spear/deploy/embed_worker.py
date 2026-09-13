"""Embedding worker: reads texts on stdin, writes vectors on stdout.

Runs on the GPU machine, invoked over ssh by embedding.py when
SPEAR_EMBED_REMOTE is set. Deliberately not a daemon: no port to expose on a
shared host, no service to supervise, and it dies with the ssh channel.

Protocol, one batch per invocation:
    stdin   JSON: {"model": "...", "texts": [...]}
    stdout  header line "<count> <dim>\n", then count*dim raw float32
Raw floats rather than JSON: 45k chunks of 1024 dims is 184 MB as float32 and
roughly 900 MB as JSON numbers, which would cost more in transfer than the GPU
saves in compute.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SPEAR_EMBED_DEVICE", "cuda")

# Force local compute here: embed_documents consults the same offload setting,
# so a stray active-embed-remote.conf on this machine would make the worker
# ship the batch on again.

os.environ["SPEAR_EMBED_REMOTE"] = ""
import embedding                                            # noqa: E402


def main():
    req = json.loads(sys.stdin.read())
    texts, model = req["texts"], req.get("model")
    vecs = embedding.embed_documents(texts, model, batch_size=req.get("batch", 16))

    if vecs is None:
        sys.stderr.write("remote worker: model resolves to chroma-default\n")
        raise SystemExit(2)

    import numpy as np
    arr = np.asarray(vecs, dtype=np.float32)
    sys.stdout.write(f"{arr.shape[0]} {arr.shape[1]}\n")
    sys.stdout.flush()
    sys.stdout.buffer.write(arr.tobytes())
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
