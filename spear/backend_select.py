#!/usr/bin/env python3
"""Startup backend picker for spear-chat.

The backends are mutually exclusive, so this is a radio choice, not a set of
checkboxes.  It is deliberately a *separate* module from rag_chat: the picker
runs before the model server is started, and importing rag_chat would drag in
chromadb and the whole application just to list a few lines.

Contract with the launcher: the chosen backend is written as a single token on
stdout — ``local``, ``remote``, ``reds`` or ``anthropic:<model>`` — and
everything the user reads goes to stderr, so the shell can capture one without
the other.
"""
from __future__ import annotations

import os
import sys

APP_DIR = os.path.dirname(os.path.realpath(__file__))

# Which backend was last picked is accumulated state, not shipped config, so
# it follows SPEAR_STATE_DIR -- in a container the application directory is not
# the user's to write.

STATE_FILE = os.path.join(os.environ.get("SPEAR_STATE_DIR", APP_DIR),
                          "active-backend.conf")
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"

C_DIM, C_ACCENT, C_RST = "\033[2m", "\033[38;5;77m", "\033[0m"

BACKENDS = ("local", "remote", "reds", "anthropic")


def _read_conf(path, key=None):
    """First non-comment line of a conf file, or a KEY=value from it."""

    try:
        with open(path, "r") as handle:
            lines = [l.strip() for l in handle if l.strip()
                     and not l.lstrip().startswith("#")]
    except OSError:
        return ""

    if key is None:
        return lines[0] if lines else ""

    for line in lines:
        name, _, value = line.partition("=")

        if name.strip() == key:
            return value.strip().strip("'\"")

    return ""


def load_last_choice():
    """The previous selection, or the local backend on a first run."""
    raw = _read_conf(STATE_FILE)
    backend, _, model = raw.partition(":")

    if backend not in BACKENDS:
        return "local", ""

    return backend, model


def save_choice(backend, model=""):
    token = f"{backend}:{model}" if model else backend

    try:
        with open(STATE_FILE, "w") as handle:
            handle.write(token + "\n")
    except OSError:
        pass          # a read-only tree must not break the launch

    return token


def describe(backend, model=""):
    """What each option is actually pointing at, read from config only.

    No probing: listing the choices must not open an SSH connection or wake a
    pod, and a server that happens to be down should still be selectable.
    """

    if backend == "local":
        served = os.path.basename(_read_conf(os.path.join(APP_DIR, "active-model.conf")))
        return f"local llama-server :8080 — {served or 'no active model'}"

    if backend == "remote":
        pod = os.path.join(APP_DIR, "pod.conf")
        host = _read_conf(pod, "POD_HOST") or "pod.conf unset"

        return f"remote vLLM on the pod via SSH — {host}"

    if backend == "reds":
        conf = os.path.join(APP_DIR, "reds.conf")
        host = _read_conf(conf, "REDS_HOST") or "reds.conf unset"
        port = _read_conf(conf, "REDS_PORT") or "?"

        return f"REDS server (RTX PRO 6000) via SSH — {host}:{port}"

    return f"Anthropic API — {model or DEFAULT_ANTHROPIC_MODEL}"


def render_menu(last, last_model, out=sys.stderr):
    """The radio list, previous choice first and preselected."""
    ordered = [last] + [b for b in BACKENDS if b != last]
    print(f"{C_DIM}Backend:{C_RST}", file=out)

    for index, backend in enumerate(ordered, 1):
        model = last_model if backend == last else ""
        marker = f"{C_ACCENT}❯{C_RST}" if index == 1 else " "
        tail = f" {C_DIM}(last used — Enter){C_RST}" if index == 1 else ""
        print(f"  {marker} {index}. {backend:9s} {C_DIM}{describe(backend, model)}"
              f"{C_RST}{tail}", file=out)

    return ordered


def choose(argv, *, isatty=None, reader=input, out=sys.stderr):
    """Resolve the backend: an explicit flag wins, else ask, else the last one.

    An explicit flag must never be second-guessed by a prompt — a scripted
    `spear-chat --remote` has already said what it wants.

    Returns (backend, model, remember). `remember` is true only for a choice
    made AT the menu: a flag says what this run wants, not what the next one
    should default to. Persisting it meant one `spear-chat --local`, for a
    comparison, silently became the default for every later launch — including
    the ones that never see a terminal and so never get asked.
    """
    flags = set(argv)
    explicit = None

    if "--provider" in flags or any(a.startswith("--provider=") for a in argv):
        provider = ""

        for index, arg in enumerate(argv):
            if arg == "--provider" and index + 1 < len(argv):
                provider = argv[index + 1]
            elif arg.startswith("--provider="):
                provider = arg.split("=", 1)[1]

        if provider == "anthropic":
            explicit = ("anthropic", _model_from_argv(argv))

    if explicit is None and "--reds" in flags:
        explicit = ("reds", "")

    if explicit is None and ({"--remote", "--pod"} & flags):
        explicit = ("remote", "")

    if explicit is None and "--local" in flags:
        explicit = ("local", "")

    if explicit is not None:
        return explicit + (False,)

    last, last_model = load_last_choice()

    if isatty is None:
        isatty = sys.stdin.isatty()

    if not isatty:
        return last, last_model, False

    ordered = render_menu(last, last_model, out=out)

    # The prompt goes to `out` (stderr), NOT through input()'s own argument:
    # input() writes its prompt to stdout, which is the one channel this
    # module promises to keep clean. Two spaces of prompt made the launcher
    # capture "  reds", which matched no case in its dispatch, so it fell
    # through to the local branch and started a llama-server the user had not
    # chosen -- silently, and only ever on a terminal, where a prompt exists.

    print("  ", end="", file=out, flush=True)

    try:
        answer = reader("").strip()
    except (EOFError, KeyboardInterrupt):
        answer = ""

    if answer.isdigit() and 1 <= int(answer) <= len(ordered):
        backend = ordered[int(answer) - 1]
    elif answer in BACKENDS:
        backend = answer
    else:
        backend = ordered[0]

    model = last_model if backend == last else ""

    # Chosen at the menu: this one is worth remembering.

    return backend, model, True


def _model_from_argv(argv):
    for index, arg in enumerate(argv):
        if arg == "--model" and index + 1 < len(argv):
            return argv[index + 1]

        if arg.startswith("--model="):
            return arg.split("=", 1)[1]

    return ""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    backend, model, remember = choose(argv)

    if remember:
        save_choice(backend, model)

    print(f"{backend}:{model}" if model else backend)

    return 0


if __name__ == "__main__":
    sys.exit(main())
