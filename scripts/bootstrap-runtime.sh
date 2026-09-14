#!/bin/bash
# Reconstruct a SPEAR runtime directory from a versioned manifest.
#
#     scripts/bootstrap-runtime.sh --root ~/spear-runtime
#     scripts/bootstrap-runtime.sh --root ~/spear-runtime --verify
#     scripts/bootstrap-runtime.sh --root ~/spear-runtime --dry-run
#
# What it builds is a deployable runtime and nothing else:
#
#     <root>/llama.cpp     the pinned revision, built
#     <root>/models/gguf   the served weights
#     <root>/embed         an isolated virtualenv and its launcher
#     <root>/config        server.conf and gpu.conf
#     <root>/log           empty, for whatever supervises the server
#
# It does NOT reconstruct migration rollback trees, historical logs or any
# host secret. Those are sediment, not runtime: see "excluded" in the
# manifest. Nothing here touches Chroma, a training tree, or any path the
# manifest does not name.
#
# Every version this installs comes from server/runtime/manifest.json -- an
# immutable llama.cpp commit, exact model shards with their publisher's
# hashes, a tested constraints file. No "latest", no branch name: a moving
# ref would mean the binary serving today differs from the one measured, with
# nothing recording that it changed.
#
# The machine-specific values are arguments, never file contents: --root and
# --gpu (or SPEAR_GPU_UUID). No host, account, card or path from any real
# deployment appears in this file or in the manifest.
#
# Work is skipped when what is there already satisfies the manifest, so a
# second run neither re-downloads 79 GB nor rebuilds a correct binary. What it
# would do is visible in advance with --dry-run, and checkable afterwards --
# without writing anything -- with --verify.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
MANIFEST="${SPEAR_RUNTIME_MANIFEST:-$REPO/server/runtime/manifest.json}"

ROOT="${SPEAR_RUNTIME_ROOT:-}"
GPU="${SPEAR_GPU_UUID:-}"
MODE=install
SKIP_MODEL=0 SKIP_LLAMA=0 SKIP_EMBED=0 CHECKSUM=0 PROBE=0

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'
    cat <<'USAGE'
Options:
  --root PATH        where the runtime lives     (or SPEAR_RUNTIME_ROOT)
  --gpu UUID         card to pin the server to   (or SPEAR_GPU_UUID)
  --manifest PATH    override the manifest       (or SPEAR_RUNTIME_MANIFEST)
  --verify           report state, write nothing
  --dry-run          report intent, write nothing
  --checksum         with --verify, hash the model shards (reads ~79 GB)
  --probe            with --verify, encode through the worker end to end
  --skip-model       leave <root>/models alone
  --skip-llama       leave <root>/llama.cpp alone
  --skip-embed       leave <root>/embed alone
  -h, --help         this text

Exit: 0 all good, 1 work failed, 2 --verify found a problem, 78 misconfigured.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --root)     ROOT="${2:-}"; shift 2 ;;
        --gpu)      GPU="${2:-}"; shift 2 ;;
        --manifest) MANIFEST="${2:-}"; shift 2 ;;
        --verify)   MODE=verify; shift ;;
        --dry-run)  MODE=dryrun; shift ;;
        --checksum) CHECKSUM=1; shift ;;
        --probe)    PROBE=1; shift ;;
        --skip-model) SKIP_MODEL=1; shift ;;
        --skip-llama) SKIP_LLAMA=1; shift ;;
        --skip-embed) SKIP_EMBED=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "bootstrap-runtime.sh: unknown option: $1" >&2; echo "  --help lists them." >&2; exit 78 ;;
    esac
done

[ -n "$ROOT" ] || { echo "bootstrap-runtime.sh: --root is required (or SPEAR_RUNTIME_ROOT)." >&2; exit 78; }
[ -r "$MANIFEST" ] || { echo "bootstrap-runtime.sh: no readable manifest: $MANIFEST" >&2; exit 78; }
command -v python3 >/dev/null || { echo "bootstrap-runtime.sh: python3 is required to read the manifest." >&2; exit 78; }

# Expand ~ and make absolute without requiring the directory to exist yet.
case "$ROOT" in "~") ROOT="$HOME" ;; "~/"*) ROOT="$HOME/${ROOT#\~/}" ;; esac
case "$ROOT" in /*) ;; *) ROOT="$PWD/$ROOT" ;; esac

# ── manifest access ──────────────────────────────────────────────────
# One reader. Every value below comes from the manifest through it, so a
# profile change is a manifest edit and never a script edit.
m() { python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."):
    d = d[int(k)] if isinstance(d, list) else d[k]
print(d if not isinstance(d, (list, dict)) else "\n".join(map(str, d)))
' "$MANIFEST" "$1"; }

MANIFEST_VERSION=$(m spear_runtime_manifest)
[ "$MANIFEST_VERSION" = "1" ] || {
    echo "bootstrap-runtime.sh: manifest version $MANIFEST_VERSION, this script speaks 1." >&2
    exit 78
}

PROFILE=$(m profile.name)
LLAMA_REPO=$(m llama_cpp.repository)
LLAMA_REV=$(m llama_cpp.revision)
LLAMA_BUILD=$(m llama_cpp.build_number)
LLAMA_BIN_REL=$(m llama_cpp.binary)
CUDA_ARCH="${SPEAR_SERVER_CUDA_ARCH:-$(m llama_cpp.cuda_architecture)}"
MODEL_REPO=$(m model.repository)
MODEL_DIR=$(m model.directory)
MODEL_FIRST=$(m model.first_shard)
EMBED_PROTOCOL=$(m embedding.protocol_version)
EMBED_MODEL=$(m embedding.probe.model)
EMBED_DIM=$(m embedding.probe.dimension)
TORCH_INDEX=$(m embedding.torch_index)
TORCH_PIN=$(m embedding.torch_pin)

LLAMA_SRC="$ROOT/llama.cpp"
LLAMA_BIN="$LLAMA_SRC/$LLAMA_BIN_REL"
MODELS="$ROOT/models/gguf"
VENV="$ROOT/embed/venv"
CONF="$ROOT/config/server.conf"
GPU_CONF="$ROOT/config/gpu.conf"

# ── scratch ──────────────────────────────────────────────────────────
# One trap. Bash keeps a single EXIT handler, so a second `trap ... EXIT`
# replaces the first rather than adding to it -- and whatever the first one
# was cleaning up then leaks.
SCRATCH=()
cleanup() { [ "${#SCRATCH[@]}" -gt 0 ] && rm -rf -- "${SCRATCH[@]}"; return 0; }
trap cleanup EXIT

# ── reporting ────────────────────────────────────────────────────────
problems=0
bold() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
ok()   { printf '  \033[32mok\033[0m      %s\n' "$1"; }
info() { printf '  ..      %s\n' "$1"; }
plan() { printf '  \033[36mwould\033[0m   %s\n' "$1"; }
keep() { printf '  \033[32mreuse\033[0m   %s\n' "$1"; }
bad()  { printf '  \033[31mMISSING\033[0m %s\n' "$1"; problems=$((problems + 1)); }

writing() { [ "$MODE" = install ]; }

# ── layout ───────────────────────────────────────────────────────────
bold "layout  ($ROOT)"
info "profile $PROFILE, manifest version $MANIFEST_VERSION"
while read -r d; do
    [ -n "$d" ] || continue
    if [ -d "$ROOT/$d" ]; then ok "$d/"
    elif writing; then mkdir -p "$ROOT/$d"; ok "$d/  (created)"
    elif [ "$MODE" = dryrun ]; then plan "create $d/"
    else bad "$d/"
    fi
done < <(m layout.directories)

# ── llama.cpp ────────────────────────────────────────────────────────
llama_state() {
    [ -d "$LLAMA_SRC/.git" ] || { echo absent; return; }
    local have
    have=$(git -C "$LLAMA_SRC" rev-parse HEAD 2>/dev/null || echo none)
    [ "$have" = "$LLAMA_REV" ] || { echo wrong-revision; return; }
    [ -x "$LLAMA_BIN" ] || { echo unbuilt; return; }
    # The binary reports the commit it was built from, which is the only
    # evidence that the checkout was not moved after the build.
    if LD_LIBRARY_PATH="$(dirname "$LLAMA_BIN")" "$LLAMA_BIN" --version 2>&1 |
         grep -q "build $LLAMA_BUILD"; then echo current; else echo stale-build; fi
}

bold "llama.cpp"
if [ "$SKIP_LLAMA" = 1 ]; then
    info "skipped (--skip-llama)"
else
    state=$(llama_state)
    info "pinned to ${LLAMA_REV:0:12} (build $LLAMA_BUILD), arch $CUDA_ARCH"
    case "$MODE:$state" in
        *:current)      keep "built and current: $LLAMA_BIN" ;;
        verify:absent)  bad "no checkout at $LLAMA_SRC" ;;
        verify:*)       bad "$state: $LLAMA_SRC" ;;
        dryrun:absent)  plan "clone $LLAMA_REPO at ${LLAMA_REV:0:12} and build llama-server" ;;
        dryrun:*)       plan "$state -> fetch ${LLAMA_REV:0:12} and rebuild llama-server" ;;
        install:*)
            [ -d "$LLAMA_SRC/.git" ] || git clone -q "$LLAMA_REPO" "$LLAMA_SRC"
            git -C "$LLAMA_SRC" fetch -q origin "$LLAMA_REV" 2>/dev/null ||
                git -C "$LLAMA_SRC" fetch -q --tags origin
            git -C "$LLAMA_SRC" checkout -q --detach "$LLAMA_REV"
            info "checked out ${LLAMA_REV:0:12} (detached: a branch would move)"
            # Delegate the build. install-llamacpp.sh already picks a CUDA
            # compiler new enough for the architecture and derives the build
            # tree from the binary path; duplicating that here would be a
            # second implementation to keep in step.
            SPEAR_SERVER_LLAMA_BIN="$LLAMA_BIN" \
            SPEAR_SERVER_CUDA_ARCH="$CUDA_ARCH" \
                "$REPO/server/inference/install-llamacpp.sh" --no-model
            [ -x "$LLAMA_BIN" ] || { echo "  build produced no $LLAMA_BIN" >&2; exit 1; }
            ok "built $LLAMA_BIN"
            ;;
    esac
fi

# ── model ────────────────────────────────────────────────────────────
shard_names() { m model.shards | grep -v '^$' >/dev/null 2>&1 || true
    python3 -c '
import json, sys
for s in json.load(open(sys.argv[1]))["model"]["shards"]:
    print(s["name"], s["size"], s["sha256"])
' "$MANIFEST"; }

bold "model"
if [ "$SKIP_MODEL" = 1 ]; then
    info "skipped (--skip-model)"
else
    info "$MODEL_REPO  ($(m model.quantisation), $(m model.total_bytes) bytes)"
    missing=0
    while read -r name size sha; do
        [ -n "$name" ] || continue
        f="$MODELS/$name"
        if [ ! -f "$f" ]; then
            missing=$((missing + 1))
            case "$MODE" in
                verify) bad "$name  absent" ;;
                dryrun) plan "download $name ($size bytes)" ;;
                *)      info "$name  absent -> will fetch" ;;
            esac
            continue
        fi
        actual=$(stat -c %s "$f")
        if [ "$actual" != "$size" ]; then
            missing=$((missing + 1))
            case "$MODE" in
                verify) bad "$name  $actual bytes, manifest says $size" ;;
                dryrun) plan "re-download $name (has $actual, wants $size)" ;;
                *)      info "$name  wrong size -> will refetch" ;;
            esac
            continue
        fi
        if [ "$CHECKSUM" = 1 ] && [ "$MODE" = verify ]; then
            got=$(sha256sum "$f" | cut -d' ' -f1)
            if [ "$got" = "$sha" ]; then ok "$name  size and sha256"
            else bad "$name  sha256 $got, manifest says $sha"; fi
        else
            keep "$name  $size bytes"
        fi
    done < <(shard_names)

    if [ "$missing" -gt 0 ] && [ "$MODE" = install ]; then
        # One implementation of "download a sharded GGUF". Naming any member
        # fetches the set, and -c resumes rather than restarting.
        SPEAR_SERVER_MODEL_REPO="$MODEL_REPO" \
        SPEAR_SERVER_MODEL_FILE="$MODEL_DIR/$MODEL_FIRST" \
        SPEAR_SERVER_MODELS_DIR="$MODELS" \
            "$REPO/server/scripts/fetch-model.sh"
        ok "fetched $missing shard(s)"
    elif [ "$missing" -eq 0 ]; then
        info "all shards present -- nothing to download"
    fi
fi

# ── embedding runtime ────────────────────────────────────────────────
embed_state() {
    [ -x "$VENV/bin/python" ] || { echo absent; return; }
    "$VENV/bin/python" - "$REPO" "$EMBED_PROTOCOL" <<'PY' >/dev/null 2>&1 || { echo broken; return; }
import sys
sys.path.insert(0, sys.argv[1] + "/server/embed")
import torch, sentence_transformers, transformers, numpy   # noqa: F401
import protocol
assert protocol.PROTOCOL_VERSION == int(sys.argv[2])
PY
    echo ready
}

bold "embedding runtime"
if [ "$SKIP_EMBED" = 1 ]; then
    info "skipped (--skip-embed)"
else
    info "protocol v$EMBED_PROTOCOL, probe $EMBED_MODEL -> $EMBED_DIM dimensions"
    state=$(embed_state)
    case "$MODE:$state" in
        *:ready)       keep "venv satisfies the manifest: $VENV" ;;
        verify:absent) bad "no venv at $VENV" ;;
        verify:broken) bad "venv cannot import the stack or speaks another protocol" ;;
        dryrun:absent) plan "create $VENV and install $TORCH_PIN plus requirements" ;;
        dryrun:*)      plan "$state -> rebuild $VENV" ;;
        install:*)
            [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
            "$VENV/bin/pip" install -q --upgrade pip
            # torch first, from its own index: the CUDA build is not on PyPI
            # and a plain -c resolve against PyPI alone cannot find it.
            "$VENV/bin/pip" install -q --index-url "$TORCH_INDEX" \
                --extra-index-url https://pypi.org/simple "$TORCH_PIN"
            "$VENV/bin/pip" install -q \
                -r "$REPO/server/embed/requirements.txt" \
                -c "$REPO/server/embed/constraints-reds-tested.txt"
            [ "$(embed_state)" = ready ] || { echo "  venv still does not satisfy the manifest" >&2; exit 1; }
            ok "built $VENV"
            ;;
    esac

    # The launcher pins the card on the machine that owns it, so the client's
    # command names this wrapper and nothing about the GPU reaches the client.
    LAUNCH="$ROOT/$(m embedding.launcher)"
    if [ -x "$LAUNCH" ] && grep -q 'SPEAR_GPU_UUID' "$LAUNCH" 2>/dev/null; then
        keep "launcher $LAUNCH"
    elif writing; then
        cat > "$LAUNCH" <<'LAUNCHER'
#!/bin/bash
# The embedding worker, as this machine runs it.
#
# The card is a property of THIS host, not of the client: on a shared machine
# the other cards are somebody else's, and taking the first visible one is how
# an encoding ends in "CUDA out of memory" against a card that is already
# full. So the client's command names this wrapper, and the UUID stays here,
# beside the machine it describes.
#
# A UUID and not an index: an index changes across reboots and with
# enumeration order, and nvidia-smi and torch disagree about it on some hosts.
#
# Generated by scripts/bootstrap-runtime.sh. Edit freely -- the bootstrap
# leaves an existing launcher alone.
set -e
RT="$(cd "$(dirname "$0")" && pwd)"
[ -r "$RT/gpu.conf" ] && export SPEAR_GPU_UUID="$(sed -n '1p' "$RT/gpu.conf")"
export SPEAR_EMBED_DEVICE="${SPEAR_EMBED_DEVICE:-cuda}"
exec "$RT/venv/bin/python" "${SPEAR_CHECKOUT:-$HOME/spear}/server/embed/worker.py"
LAUNCHER
        chmod +x "$LAUNCH"
        ok "wrote $LAUNCH"
        if [ -n "$GPU" ]; then printf '%s\n' "$GPU" > "$ROOT/embed/gpu.conf"; ok "pinned embed card"; fi
    elif [ "$MODE" = dryrun ]; then plan "write $LAUNCH"
    else bad "no launcher at $LAUNCH"
    fi

    # An end-to-end probe: the real launcher, the real protocol, one short
    # text, and the dimension the profile says the answer must have. Opt-in,
    # because it loads the model -- which on a cold cache is a download.
    #
    # Importing the stack proves the venv; only this proves the worker.
    if [ "$PROBE" = 1 ] && [ "$MODE" = verify ] && [ -x "$LAUNCH" ]; then
        # The probe goes to a file rather than a here-document inside a
        # command substitution: the body of such a here-document falls
        # outside the substitution, and bash warns and reads nothing.
        PROBE_PY=$(mktemp); SCRATCH+=("$PROBE_PY")
        cat > "$PROBE_PY" <<'PROBE_SOURCE'
import subprocess, sys
sys.path.insert(0, sys.argv[1] + "/server/embed")
import protocol

request = protocol.encode_request(
    model=sys.argv[2], texts=["a short probe"], prefix="",
    max_seq_length=1024, normalize=True, batch_size=1,
    trust_remote_code=False)
done = subprocess.run([sys.argv[3]], input=request, capture_output=True, timeout=900)
if done.returncode != 0:
    print(f"worker exit {done.returncode}: "
          f"{done.stderr.decode('utf-8', 'replace').strip()[-200:]}")
    raise SystemExit(1)
header, body = protocol.decode_response(done.stdout)
if header.get("status") == "error":
    print(f"worker error: {header.get('message', '')[:200]}")
    raise SystemExit(1)
vectors = protocol.unpack_vectors(header, body)
print(f"dim {header['dim']} count {len(vectors)}")
PROBE_SOURCE
        probe_out=$("$VENV/bin/python" "$PROBE_PY" "$REPO" "$EMBED_MODEL" "$LAUNCH" 2>&1 | tail -1)
        case "$probe_out" in
            "dim $EMBED_DIM count 1")
                ok "worker probe: $EMBED_MODEL -> $EMBED_DIM dimensions" ;;
            "dim "*)
                bad "worker probe returned '$probe_out', profile says dim $EMBED_DIM" ;;
            *)  bad "worker probe failed: $probe_out" ;;
        esac
    elif [ "$PROBE" = 1 ] && [ "$MODE" != verify ]; then
        info "--probe applies to --verify only"
    fi
fi

# ── configuration ────────────────────────────────────────────────────
bold "configuration"
if [ -f "$CONF" ]; then
    keep "$CONF  (left alone: it is this deployment's, not the manifest's)"
elif writing; then
    {
        echo "# Generated by scripts/bootstrap-runtime.sh from manifest profile"
        echo "# '$PROFILE'. Every value below is a default from the manifest;"
        echo "# edit freely -- a later bootstrap will not overwrite this file."
        echo
        echo "SPEAR_SERVER_LLAMA_BIN=$LLAMA_BIN"
        echo "SPEAR_SERVER_MODEL=$MODELS/$MODEL_FIRST"
        echo
        python3 -c '
import json, sys
for k, v in json.load(open(sys.argv[1]))["server_defaults"].items():
    print(f"{k}={v}")
' "$MANIFEST"
    } > "$CONF"
    ok "wrote $CONF"
elif [ "$MODE" = dryrun ]; then plan "write $CONF from the manifest defaults"
else bad "no $CONF"
fi

if [ -f "$GPU_CONF" ]; then
    keep "$GPU_CONF"
elif [ -n "$GPU" ] && writing; then
    {
        echo "# Which GPU this deployment may use. Supplied to the bootstrap as"
        echo "# --gpu / SPEAR_GPU_UUID: it names one machine's allocation and so"
        echo "# is never carried in the public manifest."
        echo "SPEAR_SERVER_GPU_UUID=$GPU"
    } > "$GPU_CONF"
    ok "wrote $GPU_CONF"
elif [ -n "$GPU" ] && [ "$MODE" = dryrun ]; then plan "write $GPU_CONF"
elif [ "$MODE" = verify ]; then
    info "$GPU_CONF absent -- unpinned. Correct on a single-GPU host, risky on a shared one."
else
    info "no --gpu given; leaving the server unpinned (see server/config/gpu.conf.example)"
fi

# ── serving semantics ────────────────────────────────────────────────
# Not a reimplementation of the launcher: run the real serve.sh against a stub
# binary that prints its own argv, so precedence, the MoE guess and the fixed
# flags are exercised rather than assumed.
if [ "$MODE" = verify ] && [ -f "$CONF" ]; then
    bold "serving semantics"
    # Not a reimplementation of the launcher: run the real serve.sh against a
    # stub binary that prints its own argv, so precedence, the MoE guess and
    # the fixed flags are exercised rather than assumed.
    #
    # serve.sh refuses to run without a READABLE model, and rightly so. Verify
    # must work on a runtime whose weights are absent or skipped, so it stands
    # in a stub file carrying the real basename -- the name is what the MoE
    # guess reads, so the substitution changes nothing the check is about.
    STUBDIR=$(mktemp -d); SCRATCH+=("$STUBDIR")
    printf '#!/bin/bash\nprintf "%%s\\n" "$@"\n' > "$STUBDIR/llama-server"
    chmod +x "$STUBDIR/llama-server"
    stub_model="$MODELS/$MODEL_FIRST"
    [ -r "$stub_model" ] || { : > "$STUBDIR/$MODEL_FIRST"; stub_model="$STUBDIR/$MODEL_FIRST"; }

    argv=$(SPEAR_SERVER_ROOT="$ROOT" \
           SPEAR_SERVER_LLAMA_BIN="$STUBDIR/llama-server" \
           SPEAR_SERVER_MODEL="$stub_model" \
           bash "$REPO/server/inference/serve.sh" 2>/dev/null) || argv=""

    if [ -z "$argv" ]; then
        # Never read an empty result as "every flag is missing": that reports
        # a launcher failure as a configuration failure, and names the wrong
        # thing to go and fix.
        bad "serve.sh produced no command line -- cannot check serving semantics"
        SPEAR_SERVER_ROOT="$ROOT" SPEAR_SERVER_LLAMA_BIN="$STUBDIR/llama-server" \
        SPEAR_SERVER_MODEL="$stub_model" \
            bash "$REPO/server/inference/serve.sh" 2>&1 >/dev/null | sed 's/^/          /' | head -5
    else
        printf '%s\n' "$argv" | python3 -c '
import json, sys
argv = [l for l in sys.stdin.read().splitlines() if l]
manifest = json.load(open(sys.argv[1]))

pairs, i = {}, 0
while i < len(argv):
    a = argv[i]
    if a.startswith("--"):
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            pairs[a] = argv[i + 1]; i += 2
        else:
            pairs[a] = True; i += 1
    else:
        i += 1

problems = 0
for flag in manifest["server_fixed_flags"]["flags"]:
    name, _, value = flag.partition(" ")
    got = pairs.get(name)
    if (got is True and not value) or (value and str(got) == value):
        print(f"  \033[32mok\033[0m      launcher emits {flag}")
    else:
        print(f"  \033[31mMISSING\033[0m launcher emits {name}={got!r}, profile says {flag!r}")
        problems += 1

FLAG = {"SPEAR_SERVER_CTX": "--ctx-size", "SPEAR_SERVER_PARALLEL": "--parallel",
        "SPEAR_SERVER_NGL": "--n-gpu-layers", "SPEAR_SERVER_PORT": "--port",
        "SPEAR_SERVER_HOST": "--host", "SPEAR_SERVER_THREADS": "--threads"}
for key, flag in FLAG.items():
    want = manifest["server_defaults"].get(key)
    if want is None:
        continue
    got = pairs.get(flag)
    if str(got) == str(want):
        print(f"  \033[32mok\033[0m      {flag} {want}")
    else:
        # A deployment may legitimately differ from the profile default.
        print(f"  ..      {flag} {got} -- profile default is {want} (this deployment\x27s choice)")
sys.exit(1 if problems else 0)
' "$MANIFEST" || problems=$((problems + 1))
    fi
fi

# ── result ───────────────────────────────────────────────────────────
echo
case "$MODE" in
    verify)
        if [ "$problems" -eq 0 ]; then echo "runtime at $ROOT satisfies the manifest."; exit 0
        else echo "runtime at $ROOT: $problems problem(s)." >&2; exit 2; fi ;;
    dryrun) echo "dry run: nothing was created, downloaded, built or removed." ;;
    *)      echo "runtime ready at $ROOT"
            echo "  serve with: server/inference/serve.sh   (SPEAR_SERVER_ROOT=$ROOT)" ;;
esac
