"""The model backend, the context window and one chat completion request."""

import os
import re
import sys
import json
import time
import shutil
import subprocess
from openai import OpenAI, APIStatusError, APIConnectionError
from models.model_backend import (
    AnthropicBackend, ConversationMessage, ModelBackendConfigurationError,
    anthropic_credentials_available, OpenAICompatibleBackend, TextBlock,
)
from runtime import session_replay
from cli.chat_settings import APP_DIR, LLAMA_SERVER_URL
from cli.terminal_ui import C_ACCENT, C_DIM, C_OK, C_RST, C_WARN
from cli.tool_routing import TOOLS


def model_provider_from_argv(argv):
    """Read the small provider surface without changing legacy Qwen defaults."""
    provider = "openai-compatible"
    model = None
    index = 0

    while index < len(argv):
        arg = argv[index]

        if arg == "--provider" and index + 1 < len(argv):
            provider = argv[index + 1]
            index += 1
        elif arg.startswith("--provider="):
            provider = arg.split("=", 1)[1]
        elif arg == "--model" and index + 1 < len(argv):
            model = argv[index + 1]
            index += 1
        elif arg.startswith("--model="):
            model = arg.split("=", 1)[1]

        index += 1

    if provider not in {"openai-compatible", "anthropic"}:
        raise ModelBackendConfigurationError(
            "--provider must be openai-compatible or anthropic"
        )

    return provider, model


ANT_INSTALL_HINT = (
    "install it from https://github.com/anthropics/anthropic-cli/releases "
    "(linux_amd64 tarball, extract `ant` into ~/.local/bin)")


def offer_anthropic_login():
    """Offer an interactive `ant auth login` when no credential resolves.

    Opening a browser is an outward-facing side effect, so it is proposed and
    never automatic, and only on a TTY — a non-interactive run fails closed
    with the manual instructions instead of hanging on a prompt nobody sees.

    The login runs as a plain supervisor-side subprocess, deliberately outside
    the tool sandbox: it needs the network, a browser, and write access to
    ~/.config/anthropic, none of which a tool call is ever granted.  It is
    session setup, in the same class as starting the model server.
    """

    if anthropic_credentials_available():
        return

    if not sys.stdin.isatty():
        raise ModelBackendConfigurationError(
            "no Anthropic credential and no terminal to open a session: "
            "export ANTHROPIC_API_KEY, or run `ant auth login` first")

    print(f"{C_WARN}No Anthropic credential found "
          f"(no ANTHROPIC_API_KEY, no `ant` session).{C_RST}")
    print("Open a browser to sign in now?")
    print(f"  {C_ACCENT}❯{C_RST} 1. Yes, run `ant auth login` {C_DIM}(Enter){C_RST}")
    print("    2. No, abort")

    try:
        answer = input("  ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "2"

    if answer not in ("", "1", "o", "y", "oui", "yes"):
        raise ModelBackendConfigurationError("Anthropic sign-in declined")

    ant = shutil.which("ant")

    if ant is None:
        raise ModelBackendConfigurationError(f"`ant` is not on PATH — {ANT_INSTALL_HINT}")

    # No timeout and no output capture: the CLI prints the authorize URL when
    # it cannot reach a browser, and that text is the user's only way through.

    try:
        subprocess.run([ant, "auth", "login"], check=False)
    except OSError as exc:
        raise ModelBackendConfigurationError(f"could not run `ant auth login`: {exc}")

    if not anthropic_credentials_available():
        raise ModelBackendConfigurationError(
            "sign-in did not complete — check `ant auth status`")

    print(f"{C_OK}Anthropic session open.{C_RST}")


def create_model_backend(argv):
    """Provider construction only; tools and security stay outside this boundary."""
    provider, model = model_provider_from_argv(argv)

    # A replayed session needs no provider at all: no login is offered, no
    # server is contacted, no key is read. Checked before anything else so a
    # replay costs nothing and can run offline.

    if os.environ.get(session_replay.REPLAY_ENV, "").strip():
        return session_replay.wrap(None), provider

    if provider == "anthropic":
        offer_anthropic_login()
        return session_replay.wrap(
            AnthropicBackend(model=model or "claude-sonnet-5")), provider

    return session_replay.wrap(OpenAICompatibleBackend(
        OpenAI(base_url=LLAMA_SERVER_URL, api_key="not-needed"), model=model
    )), provider


# ── LLM call ─────────────────────────────────────────────────────────

def sanitize_history(history):
    """Guarantees the strict user/assistant alternation required by the
    Qwen3 template: merges consecutive same-role messages, drops empties.
    Without this, an orphan turn (unsaved empty response) malformed the
    prompt → the model emitted EOS immediately → cascading empty replies."""
    clean = []

    for m in history:
        content = (m.get("content") or "").strip()

        if not content:
            continue

        if clean and clean[-1]["role"] == m["role"]:
            clean[-1]["content"] += "\n\n" + content
        else:
            clean.append({"role": m["role"], "content": content})

    return clean


DEFAULT_CTX = 32768
CTX_LIMIT = int(os.environ.get("SPEAR_CTX", str(DEFAULT_CTX)))

#: Where CTX_LIMIT came from, said with the number so a fallback is never
#: mistaken for what the server reported. Settled in main().
CTX_SOURCE = "SPEAR_CTX" if os.environ.get("SPEAR_CTX") else "default"


def served_context_window(api_base, timeout=5):
    """(tokens, endpoint) the server says it serves, or (None, reason).

    The default above is a guess, and too small a guess is not harmless:
    started without SPEAR_CTX against a server holding 524288 tokens, a turn
    budgeted for 32768, compacted its history twice in six minutes with a
    twentieth of the real window in use, and spent the rest of the turn
    re-reading what the summaries had dropped. The launcher asked the server
    in one of its modes; every other way of starting the client did not.

    llama.cpp states it in /props, per slot, which is the figure that bounds
    one request. vLLM states it as max_model_len on /v1/models. A training
    length (llama.cpp's meta.n_ctx_train) is not what the server serves and
    is never used.
    """
    import urllib.request

    root = re.sub(r"/v1/?$", "", (api_base or "").rstrip("/"))

    def fetch(url):
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))

    probes = (
        ("/props", lambda data: (data.get("default_generation_settings")
                                 or {}).get("n_ctx")),
        ("/v1/models", lambda data: next(
            (item.get("max_model_len") for item in data.get("data") or ()
             if isinstance(item, dict) and item.get("max_model_len")), None)),
    )
    failures = []

    for path, pick in probes:
        try:
            value = pick(fetch(root + path))
        except Exception as exc:
            failures.append(f"{path}: {type(exc).__name__}")
            continue

        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value, path

        failures.append(f"{path}: no window stated")

    return None, "; ".join(failures)


def resolve_context_window(environ, api_base, *, provider="openai-compatible",
                           discover=served_context_window):
    """(tokens, source): an explicit SPEAR_CTX, else the server, else the default.

    The source is reported as it is. A fallback says it is a fallback and
    why; it is never presented as something the server said.
    """
    explicit = (environ.get("SPEAR_CTX") or "").strip()

    if explicit:
        return int(explicit), "SPEAR_CTX"

    if provider == "anthropic":
        return DEFAULT_CTX, "default: provider not asked"

    value, where = discover(api_base)

    if value:
        return value, f"server {where}"

    return DEFAULT_CTX, f"default: server did not report a window — {where}"


def _approx_tokens(s):
    return len(s) // 4 + 1


def enforce_ctx_budget(messages, max_out):
    """Return a COPY of `messages` shrunk so prompt + max_out fits the model
    context window. A 24k-token prompt + 8k requested output overflows a 32k
    model and the server replies 400 (whole turn lost). We only SHRINK content
    — never drop a message — so an assistant tool_call always keeps its tool
    reply (dropping one would make the request invalid). Trim order: the RAG
    block, then big tool outputs, then oldest history."""
    msgs = [dict(m) for m in messages]
    budget = CTX_LIMIT - max_out - 768          # leave a margin
    total = lambda: sum(_approx_tokens(m.get("content") or "") + 4
                        for m in msgs)

    if total() <= budget:
        return msgs

    # 1) trim the "## Retrieved Context" (RAG) tail of the system message

    for m in msgs:
        if m.get("role") == "system" and \
                "## Retrieved Context" in (m.get("content") or ""):
            head, sep, rag = m["content"].partition("## Retrieved Context")
            over = (total() - budget) * 4
            keep = max(0, len(rag) - over - 200)
            m["content"] = (head + sep + rag[:keep]
                            + "\n…(retrieved context trimmed to fit)…"
                            if keep else head
                            + "## (retrieved context dropped to fit window)")

            break

    # 2) cap big tool / result payloads, oldest first

    for m in msgs[1:]:
        if total() <= budget:
            break

        c = m.get("content") or ""
        is_result = (m.get("role") == "tool" or "[tool result]" in c
                     or "executed tool" in c or "Result of " in c)

        if is_result and len(c) > 1500:
            m["content"] = c[:1200] + "\n…(trimmed)…"

    # 3) last resort: shrink oldest history bodies (keep the last 3 intact)

    for m in msgs[1:-3]:
        if total() <= budget:
            break

        c = m.get("content") or ""

        if len(c) > 600:
            m["content"] = c[:500] + "\n…(trimmed)…"

    return msgs


def canonical_history(history):
    """Keep persisted text history unchanged while using typed turn messages."""
    return [ConversationMessage(
        message["role"], (TextBlock(message.get("content") or ""),)
    ) for message in history]


def chat_once(client, messages, use_tools=True, spinner=None):
    """One STREAMING chat completion against llama-server (native tools).
    Streams so the spinner can show a live generated-token count, Claude
    Code style. Returns (text, calls) where calls is a list of dicts
    {"id", "name", "arguments"} accumulated from the tool-call deltas."""
    _max_out = int(os.environ.get("SPEAR_MAX_TOKENS", "8192"))
    messages = enforce_ctx_budget(messages, _max_out)
    kwargs = dict(
        # llama.cpp ignores the model name; vLLM matches it against
        # --served-model-name, so make it configurable (SPEAR_MODEL_NAME).
        model=os.environ.get("SPEAR_MODEL_NAME", "qwen3"),
        messages=messages,
        # Sampling temperature (SPEAR_TEMP). 0.25 gives more conservative,
        # consistent edits on Qwen3-Coder-Next (less prone to over-eager
        # refactors like dropping includes). NOTE: on the older dense/A3B model
        # a low temp starved the sampler into repetition loops; if that recurs,
        # raise SPEAR_TEMP back toward 0.7.
        temperature=float(os.environ.get("SPEAR_TEMP", "0.25")),
        top_p=0.8,
        # Big enough that a full file rewrite (write_file with the whole new
        # content, or a large edit) completes instead of being truncated
        # mid-tool-call — a cut-off XML tool call can't be parsed/applied.
        max_tokens=_max_out,
        stream=True,
        extra_body={
            # disable thinking: without this the model may open a <think>
            # block that never closes -> empty content (the whole response
            # ends up in reasoning_content).
            "chat_template_kwargs": {"enable_thinking": False},
            # extra samplers — both penalty spellings so it works on either
            # backend: llama.cpp reads "repeat_penalty", vLLM
            # "repetition_penalty"; each ignores the other.
            "top_k": 20,
            "repeat_penalty": 1.05,
            "repetition_penalty": 1.05,
        },
    )

    if use_tools:
        kwargs["tools"] = TOOLS

    content = []
    tcalls = {}
    ntok = 0
    last_line = None
    consec_repeats = 0

    # The big Q8 (35 GB) model can still be loading when llama-server's
    # /health already answers 200, so the first completion gets a 503
    # "Loading model". Wait it out instead of crashing the chat.

    stream = None
    tool_parse_retries = 0

    for attempt in range(120):  # ~10 min max at 5 s between tries
        try:
            stream = client.chat.completions.create(**kwargs)
            break
        except APIStatusError as e:
            if e.status_code == 503 and "loading" in str(e).lower():
                if spinner:
                    spinner.label = "Loading model (first request)…"
                else:
                    msg = "  model still loading" + "." * (attempt % 4)
                    print(f"\r{C_DIM}{msg}   {C_RST}", end="", flush=True)

                time.sleep(5)

                continue

            # 500: the model emitted a tool call whose JSON arguments
            # llama-server could not parse (bad quoting/backslashes in a
            # bash/grep command). Sampling varies, so a few retries usually
            # yield a well-formed call; give up gracefully after that.

            if e.status_code == 500 and "tool call" in str(e).lower():
                tool_parse_retries += 1

                if tool_parse_retries <= 3:
                    if spinner:
                        spinner.label = "Retrying (malformed tool call)…"

                    time.sleep(1)

                    continue

                raise RuntimeError(
                    "the model kept emitting an unparseable tool call "
                    "(bad JSON quoting). Rephrase your request or try again."
                ) from e

            raise
        except APIConnectionError:
            # server not up yet / restarting — same patient wait

            time.sleep(5)
            continue

    if stream is None:
        raise RuntimeError("llama-server never became ready (still loading "
                           "after ~10 min) — check llama-server.log")

    for chunk in stream:
        if not chunk.choices:
            continue

        d = chunk.choices[0].delta

        if d is None:
            continue

        if d.content:
            content.append(d.content)
            ntok += 1

            # circuit breaker: a real death-loop repeats the SAME long line
            # many times IN A ROW. (Counting total occurrences false-triggers
            # on legitimately repeated code lines — e.g. `return -1;` or
            # `tmp = atoi(argv[arg + 1]);` in a switch — and truncates valid
            # file rewrites.) So only trip on consecutive identical long lines.

            if "\n" in d.content:
                line = "".join(content).rstrip().rsplit("\n", 1)[-1].strip()

                if len(line) > 20:
                    if line == last_line:
                        consec_repeats += 1
                    else:
                        last_line = line
                        consec_repeats = 0

                    if consec_repeats >= 8:
                        stream.close()
                        content.append("\n\n(repetition loop detected — "
                                       "response truncated)")

                        break

        if d.tool_calls:
            for tcd in d.tool_calls:
                e = tcalls.setdefault(tcd.index,
                                      {"id": None, "name": "", "args": []})

                if tcd.id:
                    e["id"] = tcd.id

                if tcd.function:
                    if tcd.function.name:
                        e["name"] += tcd.function.name

                    if tcd.function.arguments:
                        e["args"].append(tcd.function.arguments)

            ntok += 1

        if spinner:
            spinner.tokens = ntok

    calls = [{"id": e["id"] or f"call_{i}", "name": e["name"],
              "arguments": "".join(e["args"])}
             for i, e in sorted(tcalls.items())]

    return "".join(content), calls


def _usage_token(usage, *names):
    if not usage:
        return None

    for name in names:
        value = usage.get(name)

        if isinstance(value, int):
            return value

    return None


# The standing tool-usage rules now live in an editable file. Edit
# tool-guide.md to tune the model's tool behavior (no code change needed);
# the inline fallback below only kicks in if the file goes missing.

TOOL_GUIDE_FILE = f"{APP_DIR}/tool-guide.md"
TOOL_GUIDE_FALLBACK = (
    "\n\n## Tool usage rules\n"
    "- Never invent command output; never claim a fix is applied unless a "
    "tool result confirms it.\n"
    "- Always answer in the language of the user's question."
)


def load_tool_guide():
    try:
        with open(TOOL_GUIDE_FILE) as f:
            content = f.read().strip()

        return "\n\n" + content if content else TOOL_GUIDE_FALLBACK
    except OSError:
        return TOOL_GUIDE_FALLBACK


TOOL_GUIDE = load_tool_guide()
