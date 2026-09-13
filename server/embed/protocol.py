"""The SPEAR embedding wire protocol — the whole contract, in one file.

The client decides what an embedding MEANS; the server decides how a machine
computes it. That line is the reason this protocol exists, and every field
below sits on one side of it:

    the client sends   model, prefix, max sequence length, normalisation,
                       batch size, trust_remote_code
    the server decides device, dtype, how the weights are loaded and cached

A server that decided any of the first group would be deciding retrieval
semantics for a collection it does not own. That is not a hypothetical: the
worker this replaces did exactly that — it imported the client's module and
read the client's model registry, so the document prefix, the sequence cap and
the normalisation were applied by whichever copy of that file happened to be
installed on the GPU host. Two copies, nothing comparing them, and a
collection filled with two different prefixes is inconsistent in a way no
later query reports. Results merely get worse.

TRANSPORT

One request per process, stdin to stdout, and the process exits. Not a daemon:
no port to expose on a shared host, no service to supervise, and it dies with
the ssh channel that started it.

REQUEST — a single JSON object on stdin, then EOF::

    {
      "spear_embed_protocol": 1,
      "model":             "BAAI/bge-m3",   the identifier to load
      "texts":             ["...", "..."],  at least one
      "prefix":            "passage: ",     prepended to every text, verbatim
      "max_seq_length":    1024,            or null for the model's own
      "normalize":         true,
      "batch_size":        16,
      "trust_remote_code": false
    }

Every field is REQUIRED, including the ones with obvious defaults. A default
here would be the server holding an opinion about semantics, and a request
that forgot a field would then be encoded successfully with the server's
guess rather than refused.

RESPONSE — one JSON header line on stdout, then the body::

    {"spear_embed_protocol":1,"status":"ok","count":2,"dim":1024,
     "dtype":"float32","byte_count":8192}\\n
    <count * dim little-endian float32>

Raw floats rather than JSON numbers: 45k chunks of 1024 dimensions is 184 MB
as float32 and roughly 900 MB as JSON, which would cost more in transfer than
the GPU saves in compute. The header is JSON so it can carry a version and a
status; the body is bytes so it can be large.

On failure, the header line is the whole response::

    {"spear_embed_protocol":1,"status":"error","kind":"protocol",
     "error":"..."}\\n

and the process exits non-zero. `kind` separates a contract failure from an
operational one: "protocol" means the pair disagree and no retry will help,
"encoding" means this machine could not do it this time.

VERSIONING

`spear_embed_protocol` is present in both directions and is checked in both
directions. A mismatch is refused, loudly, naming both numbers — it is never
negotiated down. The failure this rules out is the quiet one: a client and a
worker that still parse each other's bytes while disagreeing about who applies
the prefix would produce vectors that are subtly wrong and perfectly
well-formed.

This module is pure stdlib on purpose. The contract test must run on a machine
with no ML stack at all, and the server's dependency list must not become a
requirement for reasoning about the wire format.
"""

from __future__ import annotations

import json
import struct

#: Bumped whenever the meaning of any field changes. Adding an optional field
#: is still a bump: a server that ignores it would silently encode something
#: other than what was asked for.
PROTOCOL_VERSION = 1

VERSION_KEY = "spear_embed_protocol"

#: float32, little-endian. Stated in the header so a reader never assumes it.
DTYPE = "float32"
ITEM_SIZE = 4

EXIT_OK = 0
EXIT_PROTOCOL = 3
EXIT_ENCODING = 4

KIND_PROTOCOL = "protocol"
KIND_ENCODING = "encoding"


class ProtocolError(Exception):
    """The request and this worker do not agree. Retrying will not help."""

    kind = KIND_PROTOCOL
    exit_status = EXIT_PROTOCOL


class EncodingError(Exception):
    """The request was understood and this machine could not satisfy it."""

    kind = KIND_ENCODING
    exit_status = EXIT_ENCODING


# ── requests ─────────────────────────────────────────────────────────

def _require(request, field, types, predicate=None, description=""):
    if field not in request:
        raise ProtocolError(f"request is missing {field!r}")

    value = request[field]

    # bool is a subclass of int; a normalize=1 that meant True must not pass
    # as an int field, and a batch_size=True must not pass as a number.
    if isinstance(value, bool) != (types is bool):
        raise ProtocolError(
            f"{field!r} must be {description or types.__name__}, "
            f"got {type(value).__name__}")

    if not isinstance(value, types):
        raise ProtocolError(
            f"{field!r} must be {description or types.__name__}, "
            f"got {type(value).__name__}")

    if predicate is not None and not predicate(value):
        raise ProtocolError(f"{field!r} is {description}")

    return value


def encode_request(*, model, texts, prefix, max_seq_length, normalize,
                   batch_size, trust_remote_code):
    """Build a request. Keyword-only: a positional call that transposed
    `prefix` and `model` would be a valid JSON object and a wrong encoding."""
    return json.dumps({
        VERSION_KEY: PROTOCOL_VERSION,
        "model": model,
        "texts": list(texts),
        "prefix": prefix,
        "max_seq_length": max_seq_length,
        "normalize": normalize,
        "batch_size": batch_size,
        "trust_remote_code": trust_remote_code,
    }).encode("utf-8")


def decode_request(raw):
    """Parse and VALIDATE a request, or raise ProtocolError.

    Validation is not politeness here. Every field is a semantic decision the
    client made, so a field that arrives malformed must stop the encoding
    rather than be coerced into something plausible.
    """
    try:
        request = json.loads(raw.decode("utf-8") if isinstance(raw, bytes)
                             else raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError(f"request is not JSON: {exc}") from None

    if not isinstance(request, dict):
        raise ProtocolError(
            f"request must be a JSON object, got {type(request).__name__}")

    if VERSION_KEY not in request:
        raise ProtocolError(
            f"request carries no {VERSION_KEY}; this worker speaks version "
            f"{PROTOCOL_VERSION}")

    version = request[VERSION_KEY]

    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol version mismatch: the request is version {version!r} "
            f"and this worker speaks version {PROTOCOL_VERSION}. Deploy the "
            f"worker and the client together.")

    _require(request, "model", str, lambda v: v.strip(), "a non-empty string")
    texts = _require(request, "texts", list, None, "a list of strings")

    if not texts:
        raise ProtocolError("'texts' is empty; there is nothing to encode")

    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise ProtocolError(
                f"texts[{index}] must be a string, got {type(text).__name__}")

    _require(request, "prefix", str, None, "a string")
    _require(request, "normalize", bool, None, "a boolean")
    _require(request, "trust_remote_code", bool, None, "a boolean")
    _require(request, "batch_size", int, lambda v: v > 0, "a positive integer")

    # Present, but allowed to be null. `.get(...) is not None` would let a
    # request that OMITS the cap through, and the encoding would then run on
    # the model's own window -- 8192 tokens for bge-m3, which is enough to
    # exhaust a small card on a batch that sizes itself on its worst element.
    # A forgotten field must stop the request, not pick a default.

    if "max_seq_length" not in request:
        raise ProtocolError("request is missing 'max_seq_length'")

    if request["max_seq_length"] is not None:
        _require(request, "max_seq_length", int, lambda v: v > 0,
                 "a positive integer or null")

    return request


# ── responses ────────────────────────────────────────────────────────

def encode_response_header(count, dim):
    """The header line for a successful encoding. The body follows it."""
    return (json.dumps({
        VERSION_KEY: PROTOCOL_VERSION,
        "status": "ok",
        "count": count,
        "dim": dim,
        "dtype": DTYPE,
        "byte_count": count * dim * ITEM_SIZE,
    }) + "\n").encode("utf-8")


def encode_error(message, kind=KIND_PROTOCOL):
    """A failure, as the whole response. There is no body."""
    return (json.dumps({
        VERSION_KEY: PROTOCOL_VERSION,
        "status": "error",
        "kind": kind,
        "error": str(message),
    }) + "\n").encode("utf-8")


def decode_response(raw):
    """(header, body) from a worker's stdout, or raise ProtocolError.

    The client has its own reader -- the two programs are deployed separately
    and neither can import the other -- so this exists for the worker's own
    tests and for the contract test that holds the two readers to the same
    bytes.
    """
    head, newline, body = raw.partition(b"\n")

    if not newline:
        raise ProtocolError("response has no header line")

    try:
        header = json.loads(head.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ProtocolError("response header is not JSON") from None

    if not isinstance(header, dict) or VERSION_KEY not in header:
        raise ProtocolError(f"response header carries no {VERSION_KEY}")

    if header[VERSION_KEY] != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol version mismatch: the response is version "
            f"{header[VERSION_KEY]!r} and this reader speaks version "
            f"{PROTOCOL_VERSION}")

    if header.get("status") == "error":
        return header, b""

    if header.get("status") != "ok":
        raise ProtocolError(f"unknown response status {header.get('status')!r}")

    expected = header.get("byte_count")

    if len(body) != expected:
        raise ProtocolError(
            f"response body is {len(body)} bytes, header says {expected}")

    return header, body


def unpack_vectors(header, body):
    """The vectors, as lists of Python floats."""
    count, dim = header["count"], header["dim"]
    flat = struct.unpack(f"<{count * dim}f", body)

    return [list(flat[i * dim:(i + 1) * dim]) for i in range(count)]
