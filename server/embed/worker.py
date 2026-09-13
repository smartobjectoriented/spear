"""Encode texts on this machine. Nothing here knows what retrieval means.

Reads one protocol request on stdin, writes one response on stdout, exits.
See protocol.py for the wire format and for why the semantic fields travel in
the request instead of living here.

WHAT THIS WORKER DECIDES

Device and dtype, and nothing else. Those are properties of the machine it
runs on: which cards it has, which one it is allowed to use, whether half
precision is available. A client on a laptop cannot know them and has no
business asserting them.

    SPEAR_EMBED_DEVICE   force "cuda" or "cpu". Otherwise: cuda if torch sees
                         a card, else cpu.
    SPEAR_GPU_UUID       pin one card by UUID on a shared host. An index
                         changes across reboots and with the enumeration
                         order; a UUID does not. Applied before torch is
                         imported, because CUDA reads the variable at
                         initialisation and setting it afterwards does
                         nothing at all.

WHAT IT DOES NOT DECIDE

The model, the prefix, the sequence cap, the normalisation, the batch size.
Each arrives in the request and is applied verbatim. There is no registry
here, no default prefix, and no "if the model looks like X" branch. If this
file ever grows one, the two sides can disagree about an encoding while both
believing they are right -- which is the failure the protocol exists to make
impossible.
"""

from __future__ import annotations

import os
import sys

# The sibling module, and nothing else. Python puts a script's own directory
# on sys.path when it runs `python .../worker.py`, which is how this worker is
# started, so no path manipulation is needed -- and none is wanted. Editing
# that list is the shape the retired worker used to reach into the client
# tree, and the server suite refuses it on sight, in a comment as much as in
# code: a scan that made an exception for prose would be no scan at all.
import protocol


def pin_gpu(env=None):
    """Honour SPEAR_GPU_UUID unless the caller already chose a card.

    Must run before torch is imported. Returns the value it set, for the
    test that proves it does not override an explicit choice.
    """
    env = os.environ if env is None else env

    if env.get("CUDA_VISIBLE_DEVICES"):
        return None

    uuid = (env.get("SPEAR_GPU_UUID") or "").strip()

    if not uuid:
        return None

    env["CUDA_VISIBLE_DEVICES"] = uuid

    return uuid


def select_device(env=None, cuda_available=None):
    """The device this machine will encode on.

    Injectable so the decision can be tested without a card. An explicit
    SPEAR_EMBED_DEVICE is never second-guessed: an operator who set it knows
    something about the host that this function does not.
    """
    env = os.environ if env is None else env
    forced = (env.get("SPEAR_EMBED_DEVICE") or "").strip()

    if forced:
        return forced

    if cuda_available is None:
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False

    return "cuda" if cuda_available else "cpu"


_loaded = {}


def load(model_name, device, trust_remote_code):
    """The encoder, cached per (model, device).

    Half precision on cuda, full on cpu -- the same choice the worker this
    replaces made, so that moving to this one does not move the vectors.
    """
    key = (model_name, device)

    if key in _loaded:
        return _loaded[key]

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except Exception as exc:                      # pragma: no cover - env
        raise protocol.EncodingError(
            f"the embedding stack is not installed on this host: {exc}")

    extra = {"model_kwargs": {"dtype": torch.float16}} if device == "cuda" else {}

    try:
        encoder = SentenceTransformer(model_name, device=device,
                                      trust_remote_code=trust_remote_code,
                                      **extra)
    except Exception as exc:
        raise protocol.EncodingError(
            f"could not load {model_name!r} on {device}: {exc}") from exc

    _loaded[key] = encoder

    return encoder


def apply_sequence_cap(encoder, max_seq_length):
    """Bound the window, never widen it.

    `min` and not assignment: the request carries the client's cap, and a
    model whose own limit is lower must keep it. Widening a window is not a
    cap, and leaving it wide makes batches size themselves on their worst
    element -- which is enough to exhaust a small card.
    """
    if max_seq_length is None:
        return getattr(encoder, "max_seq_length", None)

    own = getattr(encoder, "max_seq_length", None) or max_seq_length
    encoder.max_seq_length = min(own, max_seq_length)

    return encoder.max_seq_length


def encode(request, device=None):
    """Vectors for this request, as a float32 numpy array."""
    device = device or select_device()
    encoder = load(request["model"], device, request["trust_remote_code"])
    apply_sequence_cap(encoder, request["max_seq_length"])

    prefix = request["prefix"]
    texts = [prefix + text for text in request["texts"]]

    try:
        import numpy

        vectors = encoder.encode(texts,
                                 batch_size=request["batch_size"],
                                 normalize_embeddings=request["normalize"],
                                 show_progress_bar=False,
                                 convert_to_numpy=True)
    except protocol.EncodingError:
        raise
    except Exception as exc:
        raise protocol.EncodingError(
            f"encoding {len(texts)} texts on {device} failed: {exc}") from exc

    array = numpy.asarray(vectors, dtype=numpy.float32)

    if array.ndim != 2 or array.shape[0] != len(texts):
        raise protocol.EncodingError(
            f"the encoder returned {array.shape} for {len(texts)} texts")

    return array


def run(stdin, stdout, stderr, device=None):
    """One request in, one response out. Returns the process exit status.

    Every failure becomes a structured response before it becomes an exit
    code: a client that reads only stderr learns nothing it can act on, and a
    traceback on stderr is indistinguishable from the command not being a
    worker at all.
    """
    try:
        request = protocol.decode_request(stdin.read())
    except protocol.ProtocolError as exc:
        stdout.write(protocol.encode_error(exc, protocol.KIND_PROTOCOL))
        stdout.flush()

        return protocol.EXIT_PROTOCOL

    try:
        array = encode(request, device)
    except protocol.EncodingError as exc:
        stdout.write(protocol.encode_error(exc, protocol.KIND_ENCODING))
        stdout.flush()

        return protocol.EXIT_ENCODING
    except Exception as exc:                      # pragma: no cover - defence
        stdout.write(protocol.encode_error(
            f"{type(exc).__name__}: {exc}", protocol.KIND_ENCODING))
        stdout.flush()

        return protocol.EXIT_ENCODING

    count, dim = array.shape
    stdout.write(protocol.encode_response_header(count, dim))
    stdout.write(array.tobytes())
    stdout.flush()

    return protocol.EXIT_OK


def main():
    pin_gpu()

    return run(sys.stdin.buffer, sys.stdout.buffer, sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
