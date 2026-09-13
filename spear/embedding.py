"""Shared embedding function — ONE single point of definition.

Indexing and querying MUST go through the same model. If they diverge, Chroma
raises nothing: it silently returns random neighbours, and nothing in the
output says so. That is the costliest failure mode in this system. Hence this
single module, and above all: the model name is written into the collection
METADATA at indexing time, then read back when the collection is opened. A
query therefore never has to restate which model to use — it cannot get it
wrong, even if active-embedder.conf changed in the meantime.

A non-default embedder resolves in this order:
    SPEAR_EMBED_MODEL (env) > active-embedder.conf > "chroma-default"

"chroma-default" = all-MiniLM-L6-v2 ONNX/CPU, the historical behaviour: no
model is loaded, Chroma embeds on its own. It is the safe fallback, and the
only one needing neither a GPU nor a download.
"""
import os
import shlex
import threading

APP_DIR = os.path.dirname(os.path.realpath(__file__))
ACTIVE_CONF = os.path.join(APP_DIR, "active-embedder.conf")
DEFAULT = "chroma-default"
MAX_SEQ = 1024

# name -> (document prefix, query prefix, trust_remote_code)
# The prefixes are ASYMMETRIC for e5 and Qwen3-Embedding: applying them on the
# wrong side, or not at all, costs several points of recall.

MODELS = {
    DEFAULT:                          ("", "", False),
    "BAAI/bge-m3":                    ("", "", False),
    # NB: this instruction prefix is a functional value, not prose — it is fed
    # to the model and changes the query vector. Kept verbatim in French
    # because that is the form eval/bench_results.json was measured with.
    "Qwen/Qwen3-Embedding-0.6B":      ("", "Instruct: Retrouve le fichier source "
                                          "ou le code pertinent\nQuery: ", False),
    "intfloat/multilingual-e5-large": ("passage: ", "query: ", False),
}

# ── GPU pinning on a shared host ─────────────────────────────────────
# On a multi-GPU host, torch takes the first VISIBLE card — which may belong to
# somebody else (on reds-ml one of the two Blackwells is 91 GB busy). We pin by
# UUID rather than by index: the index changes across reboots or with the
# enumeration order, the UUID does not.
#
# Applied HERE and not in the launcher, because the indexers are invoked
# directly without going through spear-chat.sh. And applied at IMPORT, before
# any torch.cuda call: CUDA reads this variable at initialisation, setting it
# afterwards has no effect at all.

GPU_CONF = os.path.join(APP_DIR, "active-gpu.conf")


def _pin_gpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return                             # explicit choice by the caller

    uuid = os.environ.get("SPEAR_GPU_UUID")

    if not uuid and os.path.isfile(GPU_CONF):
        with open(GPU_CONF) as f:
            uuid = f.read().strip()

    if uuid:
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid


_pin_gpu()

_lock = threading.Lock()
_loaded = {}          # (model, device) -> SentenceTransformer


def active_model():
    m = os.environ.get("SPEAR_EMBED_MODEL")

    if not m and os.path.isfile(ACTIVE_CONF):
        with open(ACTIVE_CONF) as f:
            m = f.read().strip()

    m = m or DEFAULT

    return m if m in MODELS else DEFAULT


def _is_cached(model, env=None):
    """Is this model already in the local Hub cache?

    The cache path is recomputed here rather than read from
    huggingface_hub.constants: importing that module is precisely what we must
    not do yet — the native Hub client latches HF_HUB_OFFLINE at import, so
    setting it afterwards has no effect. The resolution order is the
    documented one.
    """
    env = os.environ if env is None else env
    cache = env.get("HF_HUB_CACHE")

    if not cache:
        home = env.get("HF_HOME") or os.path.join(
            env.get("XDG_CACHE_HOME")
            or os.path.expanduser("~/.cache"), "huggingface")
        cache = os.path.join(home, "hub")

    return os.path.isdir(os.path.join(
        cache, "models--" + model.replace("/", "--")))


def should_go_offline(model, env=None):
    """Should the Hub be kept out of the loop for this load?

    Kept pure and env-injectable: the decision is what needs testing, and a
    test that flipped the real HF_HUB_OFFLINE leaked into every later test
    that loads an actual model — they then failed on local_files_only.
    """
    env = os.environ if env is None else env

    if "HF_HUB_OFFLINE" in env:
        return False                    # an explicit choice is left alone

    return _is_cached(model, env)


def _quiet_loader(model):
    """Load the embedder without narrating it.

    This runs in the middle of a chat turn, so a "Loading weights 391/391" bar
    and an unauthenticated-Hub warning read as if the assistant were
    downloading a model, when it is only reading the local cache to embed one
    query.

    Two different sources, two different levers. The bar belongs to
    transformers, not to huggingface_hub — HF_HUB_DISABLE_PROGRESS_BARS has no
    effect on it. The warning comes from the native Hub client on stderr and
    exists in no Python module, so it can only be avoided by not contacting
    the Hub at all.

    Going offline is therefore conditional: only when the weights are already
    cached, since offline mode turns a first-time download into an OSError.
    An explicit HF_HUB_OFFLINE is always left alone.
    """

    # Environment first, before anything imports the Hub client.

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    if should_go_offline(model):
        os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        import transformers.utils.logging as tl
        tl.disable_progress_bar()
    except Exception:
        pass


def _st(model, device):
    """Load the model once per (model, device). Loading costs a few seconds:
    doing it again on every turn would show."""
    key = (model, device)

    with _lock:
        if key not in _loaded:
            _quiet_loader(model)      # before any Hub import — see docstring
            import torch
            from sentence_transformers import SentenceTransformer
            kw = {"model_kwargs": {"dtype": torch.float16}} if device == "cuda" else {}
            st = SentenceTransformer(
                model, device=device,
                trust_remote_code=MODELS[model][2], **kw)

            # bge-m3 accepts 8192 tokens; our chunks are ~400. Leaving the
            # window wide open makes batches size themselves on the worst
            # element, which is enough to OOM an 8 GB GPU. We bound it here
            # rather than trusting the quality of the inputs.

            st.max_seq_length = min(getattr(st, "max_seq_length", MAX_SEQ) or MAX_SEQ,
                                    MAX_SEQ)
            _loaded[key] = st

        return _loaded[key]


def loaded(model=None, for_indexing=False):
    """Is the model already in memory for this use?  Asked, never loaded.

    The caller is the chat, which pays ~10 s the first time a query is
    embedded and wants to say so before the terminal goes quiet rather than
    after. Reading the cache under the same lock the loader takes keeps the
    answer honest while another thread is loading.
    """
    model = model or active_model()

    if model == DEFAULT:
        return True

    with _lock:
        return (model, _device(for_indexing)) in _loaded


def _device(for_indexing):
    """Indexing takes the GPU (thousands of chunks). Querying stays on the CPU
    by default: a single sentence per turn, and above all llama-server wants
    that VRAM in --local mode. SPEAR_EMBED_DEVICE forces either one."""
    forced = os.environ.get("SPEAR_EMBED_DEVICE")

    if forced:
        return forced

    if not for_indexing:
        return "cpu"

    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ── offloading the indexing GPU work ─────────────────────────────────
# Measured on identical real chunks (median 1533 characters): 28 chunks/s on
# the laptop's RTX 4060, 279 on the RTX PRO 6000. Indexing is dominated by
# embedding — reading and chunking 984 files takes 0.17 s — so this is the one
# stage worth moving.
#
# Only the compute moves. Chunking and the Chroma writes stay local, so there
# is no source tree to mirror and no index to copy back. The traffic is the
# texts up and the vectors down: for 45k chunks that is ~67 MB and ~184 MB,
# about 15 s on a 16 MB/s link against ~25 min of GPU time saved.

REMOTE_CONF = os.path.join(APP_DIR, "active-embed-remote.conf")
REMOTE_CMD_CONF = os.path.join(APP_DIR, "active-embed-remote-cmd.conf")

# ── the wire to the server-side worker ───────────────────────────────
# The format is defined once, in server/embed/protocol.py, and implemented
# twice: here and there. Not shared code, deliberately -- the worker is
# installed on a different machine, with only server/embed/ copied to it and
# none of this tree, so neither side can import the other even if the boundary
# allowed it. A contract test holds the two readers to the same bytes.
#
# What travels is everything that decides what a vector MEANS. The worker
# applies it verbatim and has no registry of its own, so the two sides cannot
# disagree about who prefixes a document. The worker this replaces imported
# THIS module on the GPU host and read its registry from whatever copy was
# installed there -- which is how a collection could be filled with two
# different prefixes and say nothing about it.

EMBED_PROTOCOL_VERSION = 1
PROTOCOL_KEY = "spear_embed_protocol"


class RemoteEmbeddingError(RuntimeError):
    """A configured remote embedder produced no vectors.

    Raised, never swallowed. Once a deployment has named a remote embedder
    that IS the embedder, and computing the vectors somewhere else is a
    different answer rather than a slower one: a collection half-filled from
    one device and half from another is silently inconsistent, and nothing
    downstream can tell. A deployment that has named no remote embedder is a
    different case entirely -- see embed_documents.
    """


def remote_target():
    """ssh destination for indexing, or None. SPEAR_EMBED_REMOTE > conf file.

    Queries are never sent remotely: a single question embeds in 44 ms here
    against 58 ms there plus 10 ms of round trip, so the laptop wins.
    """

    # `in os.environ`, not a truthiness test: SPEAR_EMBED_REMOTE="" has to
    # mean "disabled here", not "fall through to the config file". The worker
    # on the GPU host relies on exactly that to avoid shipping the batch on
    # again, and a test that set it to "" was silently still going remote —
    # comparing the remote result against itself.

    if "SPEAR_EMBED_REMOTE" in os.environ:
        return os.environ["SPEAR_EMBED_REMOTE"].strip() or None

    if os.path.isfile(REMOTE_CONF):
        with open(REMOTE_CONF) as f:
            return (f.readline().strip() or None)

    return None


def remote_ssh_opts():
    """Extra ssh options for the offload, so the target is self-contained.

    The host this runs against has a low MaxAuthTries: without
    IdentitiesOnly, ssh offers every key in ~/.ssh and is disconnected before
    reaching the right one. Keeping the options beside the destination means
    indexing works from cron or a bare shell, not only from one where the
    variable happens to be exported.
    """

    if "SPEAR_EMBED_REMOTE_SSH_OPTS" in os.environ:
        return os.environ["SPEAR_EMBED_REMOTE_SSH_OPTS"].split()

    if os.path.isfile(REMOTE_CONF):
        with open(REMOTE_CONF) as f:
            f.readline()                       # first line is the destination
            return os.path.expandvars(f.read()).split()

    return []


def remote_command():
    """The worker command to run ON the remote host, as argv, or None.

    SPEAR_EMBED_REMOTE_CMD > active-embed-remote-cmd.conf > nothing.

    There is no built-in default, and that is the point: this tree used to
    name one deployment's directory layout, which made the public code carry
    somebody's private filesystem and made every other deployment wrong by
    construction. Where the worker lives is a property of the host, so it is
    configured beside the host.

    An explicitly empty variable means unset, matching remote_target(): a
    deployment disables the offload by clearing the DESTINATION, not by
    half-configuring it.
    """
    raw = os.environ.get("SPEAR_EMBED_REMOTE_CMD")

    if raw is None and os.path.isfile(REMOTE_CMD_CONF):
        with open(REMOTE_CMD_CONF) as f:
            raw = "".join(line for line in f
                          if line.strip() and not line.lstrip().startswith("#"))

    if not raw or not raw.strip():
        return None

    return shlex.split(raw) or None


def _remote_word(word):
    """Quote one argument so the remote login shell reads it as one word.

    ssh joins its command arguments with spaces and hands the result to a
    shell, so anything built by concatenation is interpreted there: a path
    with a space becomes two arguments, and a semicolon becomes a second
    command. Everything is quoted.

    The one exception is a leading `~/`, which a deployment uses to name a
    path in the remote account without knowing its home directory. Quoted
    literally the shell would not expand it, so it becomes an expansion that
    survives quoting — and the rest of the word stays quoted.
    """
    if word == "~":
        return '"$HOME"'

    if word.startswith("~/"):
        return '"$HOME"/' + shlex.quote(word[2:])

    return shlex.quote(word)


def document_semantics(model, batch_size):
    """Everything the encoding needs, decided HERE.

    Gathered in one function because it is the answer to "what does this
    collection mean": the same values the local path applies below, so the two
    cannot drift. A reader comparing local and remote embedding should be able
    to see at a glance that they are the same encoding on a different machine.
    """
    return {
        "model": model,
        "prefix": MODELS[model][0],
        "max_seq_length": MAX_SEQ,
        "normalize": True,
        "batch_size": batch_size,
        "trust_remote_code": MODELS[model][2],
    }


def _worker_diagnosis(proc):
    """What the worker said about its own failure, in its own words."""
    import json

    head, _, _ = proc.stdout.partition(b"\n")

    try:
        header = json.loads(head.decode("utf-8"))
        message = header["error"]
    except Exception:
        return proc.stderr.decode("utf-8", "replace").strip()[:200] or \
            "no diagnosis on stdout or stderr"

    return f"{header.get('kind', 'error')}: {message}"


def _protocol_request(texts, semantics):
    """One request, as bytes. See server/embed/protocol.py for the format."""
    import json

    return json.dumps(dict(semantics, **{PROTOCOL_KEY: EMBED_PROTOCOL_VERSION,
                                         "texts": list(texts)})).encode("utf-8")


def _protocol_response(raw, expected, target):
    """Vectors from a worker's stdout, or RemoteEmbeddingError.

    Every check here is about a specific way a collection gets quietly
    corrupted: a header from something that is not a worker, a version that
    does not match ours, a body that arrived short, a count that does not line
    up with the texts we sent -- accepting that last one would misalign every
    chunk from there on.
    """
    import json, struct

    head, newline, body = raw.partition(b"\n")

    if not newline:
        raise RemoteEmbeddingError(
            f"remote embedding on {target} returned no protocol header — the "
            f"command ran, but it is not a SPEAR embedding worker")

    try:
        header = json.loads(head.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise RemoteEmbeddingError(
            f"remote embedding on {target} returned no protocol header — the "
            f"command ran, but it is not a SPEAR embedding worker") from None

    if not isinstance(header, dict) or PROTOCOL_KEY not in header:
        raise RemoteEmbeddingError(
            f"remote embedding on {target} returned no protocol header — the "
            f"command ran, but it is not a SPEAR embedding worker")

    if header[PROTOCOL_KEY] != EMBED_PROTOCOL_VERSION:
        raise RemoteEmbeddingError(
            f"protocol version mismatch with the worker on {target}: it "
            f"speaks version {header[PROTOCOL_KEY]!r} and this client speaks "
            f"version {EMBED_PROTOCOL_VERSION}. Deploy them together.")

    if header.get("status") != "ok":
        raise RemoteEmbeddingError(
            f"remote embedding on {target} failed "
            f"({header.get('kind', 'error')}): {header.get('error', header)}")

    count, dim = header.get("count"), header.get("dim")

    if len(body) != header.get("byte_count") or count != expected:
        raise RemoteEmbeddingError(
            f"remote embedding on {target} returned {count} vectors of "
            f"{dim} dimensions in {len(body)} bytes, for {expected} "
            f"texts — truncated or out of protocol")

    # Unpack ONCE. Calling struct.unpack inside the loop re-decoded the whole
    # 8 MB payload per vector, turning a 16 s round trip into 128 s and making
    # the offload slower than computing locally.

    flat = struct.unpack(f"<{count * dim}f", body)

    return [list(flat[j * dim:(j + 1) * dim]) for j in range(count)]


def _embed_remote(texts, model, target, batch_size, progress):
    """Send texts to the GPU host, read float32 vectors back.

    Batched so memory stays bounded on both ends and a long index reports
    progress. Every failure raises: see RemoteEmbeddingError.
    """
    import subprocess

    argv = remote_command()

    if not argv:
        raise RemoteEmbeddingError(
            f"remote embedding is configured for {target}, but the worker "
            f"command is not. Set SPEAR_EMBED_REMOTE_CMD, or write the "
            f"command in {REMOTE_CMD_CONF}.")

    command = " ".join(_remote_word(word) for word in argv)
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    ssh += remote_ssh_opts()
    out = []

    # Each invocation is a fresh process that loads the model (~5.5 s), so the
    # batch has to be big enough to amortise it: at 512 a 45k-chunk index would
    # spend eight minutes just loading. 20k texts is ~30 MB up and ~80 MB down,
    # comfortable on both ends, and cuts the loads to a handful.

    step = int(os.environ.get("SPEAR_EMBED_REMOTE_BATCH", "20000"))
    semantics = document_semantics(model, batch_size)

    for i in range(0, len(texts), step):
        part = texts[i:i + step]
        payload = _protocol_request(part, semantics)

        try:
            proc = subprocess.run(ssh + [target, command],
                                  input=payload, capture_output=True,
                                  timeout=1800)
        except Exception as exc:
            raise RemoteEmbeddingError(
                f"remote embedding on {target} could not be run: {exc}") from exc

        if proc.returncode != 0:
            # The worker reports structurally on stdout before it exits, so
            # prefer that: stderr on a failed ssh is as likely to be the
            # remote shell's complaint as the worker's diagnosis, and a
            # traceback there is indistinguishable from "not a worker".
            raise RemoteEmbeddingError(
                f"remote embedding on {target} failed (exit "
                f"{proc.returncode}): {_worker_diagnosis(proc)}")

        out.extend(_protocol_response(proc.stdout, len(part), target))

        if progress:
            print(f"  embedded {min(i + step, len(texts))}/{len(texts)} "
                  f"on {target}", end="\r", flush=True)

    if progress:
        print()

    return out


def embed_documents(texts, model=None, batch_size=16, progress=False):
    """CORPUS-side vectors. Returns None for chroma-default (Chroma handles it)."""
    model = model or active_model()

    if model == DEFAULT:
        return None

    target = remote_target()

    # CASE B: a remote embedder is configured, so it is the embedder. Not a
    # preference to fall back from -- quietly computing these vectors on this
    # machine would put two models' geometry in one collection, and the index
    # would be wrong in a way no later query reports.
    #
    # CASE A -- no destination configured -- falls through: local embedding is
    # a valid deployment, not a degraded one.

    if target:
        return _embed_remote(texts, model, target, batch_size,
                             progress) if texts else []

    semantics = document_semantics(model, batch_size)
    st = _st(model, _device(for_indexing=True))

    return st.encode([semantics["prefix"] + t for t in texts],
                     batch_size=semantics["batch_size"],
                     normalize_embeddings=semantics["normalize"],
                     show_progress_bar=progress,
                     convert_to_numpy=True).tolist()


def embed_query(text, model=None):
    """QUERY-side vector. Returns None for chroma-default.

    Never offloaded, configured destination or not: this is one sentence per
    turn, the round trip costs more than the compute, and in --local mode
    llama-server wants that VRAM. So the CASE A / CASE B distinction in
    embed_documents does not arise here -- there is nothing to fall back from.
    """
    model = model or active_model()

    if model == DEFAULT:
        return None

    pfx = MODELS[model][1]
    st = _st(model, _device(for_indexing=False))

    return st.encode([pfx + text], normalize_embeddings=True,
                     convert_to_numpy=True)[0].tolist()


def create_collection(client, name, model=None):
    """Create a collection, STAMPING the model into its metadata."""
    model = model or active_model()

    return client.create_collection(
        name=name, metadata={"hnsw:space": "cosine", "embed_model": model})


def collection_model(collection):
    """The model this collection was indexed with. Collections predating this
    module lack the key: they are MiniLM, hence DEFAULT."""

    try:
        return (collection.metadata or {}).get("embed_model", DEFAULT)
    except Exception:
        return DEFAULT


# ── indexing without a window of vulnerability ───────────────────────
# The indexers used to delete the collection BEFORE rebuilding it. A crash
# along the way (GPU OOM, source tree gone, ctrl-c) left the index empty, and
# nothing reported it at the next launch — the chat simply answered without
# context. That happened on 2026-08-20 to edgem1_verdin. So we build alongside,
# and only swap once indexing has succeeded.

STAGING_SUFFIX = "__building"


def staged_collection(client, name, model=None):
    """A working collection beside `name`, which stays untouched."""
    staging = name + STAGING_SUFFIX

    try:
        client.delete_collection(staging)      # leftover from an aborted run
    except Exception:
        pass

    return create_collection(client, staging, model)


def commit_staged(client, name):
    """Swap the working collection onto its final name. Call only after
    indexing has run to completion."""
    staging = client.get_collection(name + STAGING_SUFFIX)

    try:
        client.delete_collection(name)
    except Exception:
        pass                                   # first indexing of this corpus

    staging.modify(name=name)

    return staging
