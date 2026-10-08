"""The startup banner and the help text."""

import os
from harness.tool_runtime import ExecutionMode
from standard.standard_commands import StandardCommandError, retrieval_summary
from cli import session_workspace
from cli.chat_settings import (
    APP_DIR, ENV_OPTIONS, ENV_SWITCHES, RULES_DIR, STANDARD_OPERATOR,
    STANDARD_STORE,
)
from cli.knowledge_commands import SKILLS_DIR
from cli.operator_input import OPERATOR_COMMANDS, SESSION_COMMANDS
from cli.project_checks import BENCH_DIR
from cli.session_history import fresh_session
from cli.terminal_ui import C_DIM, C_RST, C_WARN, Spinner


HELP_TEXT = """\
spear-chat — HEIG-VD/REDS AI coding assistant

USAGE
  spear-chat [options]

PERMISSIONS  (what the assistant may do to your files)
  --safe                 read-only, and the default when no mode is given.
                         Mutations are refused, not proposed, and the network
                         is unreachable.
  --ask                  confirm each edit and each command before it runs,
                         network use included. Aliases: --confirm, --no-bypass.
  --auto                 run edits and commands without asking; network is
                         available. Aliases: -y, --yolo, --bypass-permissions.
  --no-network           no network in any mode, the web tools included.
  --single-root          restrict writes to the launch directory. By default
                         the registered corpora are writable too, each mounted
                         at /workspaces/<name> — a path that works in the
                         terminal and the file tools alike. Relative paths always
                         resolve in the launch directory and never reach them.
  --allow-absolute-paths accept host absolute paths into the launch directory.
                         Off by default; /workspace/... always works.

MODEL
  --provider NAME        openai-compatible (default) or anthropic.
  --model NAME           model id. Anthropic default: claude-sonnet-5.
  --remote, --pod        use the GPU pod's vLLM over an SSH tunnel instead of
                         the local llama-server. --local forces local.
  --reds                 use the REDS server (RTX PRO 6000) over an SSH tunnel;
                         host/port/model in reds.conf. The launcher only opens
                         the tunnel — start the server there yourself.
  --pod-host H, --pod-port P
                         point at a pod without editing pod.conf.

CORPUS  (what the assistant reads about — distinct from where tools run)
  --corpus NAME          force a registered corpus. Aliases: --project,
                         --checkout. Tools still run in the CURRENT directory.
  --here                 treat the current directory as an ad-hoc corpus.
  (none)                 the corpus containing the cwd, else ad-hoc on the cwd.

FEDERATION
  --with NAME            also retrieve from this registered corpus, for one
                         session. Repeatable.
  --without NAME         drop one that would be attached (a `corpora:` entry
                         of the project, or a corpus marked shared).

SESSION SETTINGS  (each is also an environment variable; the flag wins)
{env_options}

OPERATOR COMMANDS  (typed in the session, never reachable by the model)
  /standard              ingest a specification, bind one, inspect the binding.
                         The binding decides how normative answers are grounded
                         and is shared by every session on this machine, so
                         check it before trusting one: `/standard status`.
                         `/standard list`, `/standard use <id> <revision>`.
  /finetune              the fine-tuning control plane.

OTHER
  --trace                record a runtime JSONL trace (SPEAR_TRACE).
  --record FILE          write down every model turn of this session.
  --replay FILE          answer from a recorded file instead of the model.
                         The tools, the files and the gates all run for real;
                         only the model is a recording, so a harness change
                         can be judged in minutes and at no cost. A turn that
                         needs more rounds than were recorded simply ends.
  --help, -h             this text.

Without a backend flag, an interactive launch lists local / remote / reds /
anthropic
and preselects the one used last (remembered in active-backend.conf); a flag
skips the picker, and off a terminal the last choice is reused silently.

Tools always run in the current directory, whatever the corpus. To work on
another tree, cd into it first — no flag relocates the workspace.

ENVIRONMENT  (machine-wide defaults; machine.env is the place for them)
  ANTHROPIC_API_KEY   Anthropic credential; an `ant auth login` session is
                      used when it is unset.
  SPEAR_API_BASE      OpenAI-compatible endpoint (default 127.0.0.1:8080/v1).
  SPEAR_MODEL         GGUF path served by spear-server, overriding
                      active-model.conf. SPEAR_LORA likewise for the adapter.
"""


def _env_option_lines():
    """The settings table, rendered from ENV_OPTIONS so it cannot drift.

    A help text maintained beside its parser goes stale; this one IS the
    parser's table.
    """
    rows = []

    for flag, (variable, description) in sorted(ENV_OPTIONS.items()):
        label = f"{flag} VALUE"

        if len(label) > 24:
            rows.append(f"  {label}")
            rows.append(f"  {'':<25}{description}  ({variable})")
        else:
            rows.append(f"  {label:<25}{description}")
            rows.append(f"  {'':<25}{variable}")

    for flag, (variable, _, description) in sorted(ENV_SWITCHES.items()):
        rows.append(f"  {flag:<25}{description}")
        rows.append(f"  {'':<25}{variable}=1")

    return "\n".join(rows)


def print_help():
    """Usage text. The startup banner names the permission flags too, but only
    after a full startup — this is the answer to `spear-chat --help`."""
    print(HELP_TEXT.replace("{env_options}", _env_option_lines()), end="")


def backend_label(provider, api_base):
    """Which backend the banner is reporting: never blank, never ambiguous.

    Previously only the remote pod was tagged, so local and Anthropic looked
    identical on the banner — the one line that is supposed to tell you what
    you are about to talk to.
    """

    if provider == "anthropic":
        return "anthropic API"

    # The launcher knows which host it tunnelled to; a port number does not.
    # This line used to read a hard-coded host name for anything on
    # :8082, and kept saying so after reds.conf was repointed at another
    # machine — the banner then named a host we were not talking to.

    label = os.environ.get("SPEAR_BACKEND_LABEL", "").strip()

    if label:
        return label

    if "8081" in api_base:
        return "remote/pod vLLM"

    if "8082" in api_base:
        return "reds tunnel"

    return "local llama-server"


def permissions_row(mode):
    """The startup banner's permissions line: the real mode, and how to change it.

    This used to branch on the legacy BYPASS_PERMISSIONS boolean, which predates
    the three modes: it announced SAFE — the default — as "ask before each
    edit/run", a mode SAFE does not have (it refuses mutations outright), and
    labelled AUTO "(default)", which stopped being true when SAFE became it.
    The banner is where a user learns which mode they are in, so naming the
    flags here is what keeps them discoverable.
    """

    if mode == ExecutionMode.AUTO:
        return (f"{C_WARN}⚠ permissions: auto — edits, commands and network "
                f"access run without asking{C_RST}{C_DIM}; --ask to confirm "
                f"each, --no-network to stay offline, --safe for read-only"
                f"{C_RST}")

    if mode == ExecutionMode.ASK:
        return (f"{C_DIM}permissions: ask before each edit/run (network "
                f"included); --safe for read-only, --auto to stop asking"
                f"{C_RST}")

    return (f"{C_DIM}permissions: safe (default) — mutations refused; --ask to "
            f"confirm each edit/run, --auto to stop asking{C_RST}")


def standard_row():
    """The bound standard, on the banner, beside the model and the corpus.

    It decides how every normative answer is grounded, and it lives in ONE file
    under the state directory, shared by every session on this machine. A
    binding left behind by other work is therefore inherited in silence: a VITA
    49.2 question was once answered against an RS274/NGC binding, and the only
    thing that ever said so was the per-turn line, printed after the question
    had already been asked. A session-wide property belongs with the other
    session-wide properties.

    Returns None when no standard has ever been ingested, so a project that
    does not use one carries no row it cannot act on.
    """
    try:
        if not STANDARD_STORE.list_standards():
            return None
    except OSError:
        return None

    # A binding the store can no longer honour is a startup fact too. Raising
    # here would take the banner down; saying nothing would leave it to be
    # discovered from the first normative answer that comes out empty.

    try:
        binding = STANDARD_OPERATOR.active_binding()
    except StandardCommandError as exc:
        return f"{C_WARN}standard: unavailable — {exc}{C_RST}"

    if binding is None:
        return (f"{C_DIM}standard:{C_RST} none bound  {C_DIM}·{C_RST}  "
                f"{C_DIM}/standard list, /standard use <id> <revision>{C_RST}")

    # How it is searched follows the document, so switching documents
    # switches it; the banner is where that has to be visible.

    return (f"{C_DIM}standard:{C_RST} {binding.standard_id} {binding.revision}"
            f"  {C_DIM}·{C_RST}  {binding.data_origin.lower()}"
            f"  {C_DIM}·{C_RST}  "
            + retrieval_summary(STANDARD_STORE, binding.standard_id,
                                binding.revision))


def content_row(dirs=None):
    """Where rules, skills and benches come from, on the banner.

    They are relocatable (resource_dir) and a deployment points them at a
    tree outside the checkout. When the file that does so is missing, the
    session starts anyway on the near-empty in-tree defaults, and nothing
    said so: losing a deployment's rules looked exactly like having none.
    """
    if dirs is None:
        dirs = {"rules": RULES_DIR, "skills": SKILLS_DIR, "benches": BENCH_DIR}

    app_dir = os.path.realpath(APP_DIR)

    def is_in_tree(path):
        return os.path.realpath(path).startswith(app_dir + os.sep)

    # An absent in-tree directory is a plain checkout; an absent external one
    # is a deployment pointing at something that is not there.

    missing = [name for name, path in dirs.items()
               if not is_in_tree(path) and not os.path.isdir(path)]

    if all(is_in_tree(path) for path in dirs.values()):
        row = f"{C_DIM}content:{C_RST}  in-tree  {C_DIM}(no external rules/skills/benches){C_RST}"
    else:
        # Grouped by parent, so the usual case -- one deployment tree holding
        # all three -- reads as that tree rather than as three paths.

        parents = {}

        for name, path in dirs.items():
            parent = ("in-tree" if is_in_tree(path)
                      else os.path.dirname(os.path.normpath(path)))
            parents.setdefault(parent, []).append(name)

        home = os.path.expanduser("~")
        parts = []

        for parent, names in parents.items():
            if parent.startswith(home + os.sep):
                parent = "~" + parent[len(home):]

            parts.append(f"{parent} {C_DIM}({', '.join(names)}){C_RST}")

        row = f"{C_DIM}content:{C_RST}  " + f"  {C_DIM}·{C_RST}  ".join(parts)

    if missing:
        row += f"  {C_WARN}⚠ missing: {', '.join(missing)}{C_RST}"

    return row


def banner_art():
    """ASCII-art header in the platform gradient.
    Shown first, before the project picker."""
    art = [
        "  ███████╗██████╗ ███████╗ █████╗ ██████╗ ",
        "  ██╔════╝██╔══██╗██╔════╝██╔══██╗██╔══██╗",
        "  ███████╗██████╔╝█████╗  ███████║██████╔╝",
        "  ╚════██║██╔═══╝ ██╔══╝  ██╔══██║██╔══██╗",
        "  ███████║██║     ███████╗██║  ██║██║  ██║",
        "  ╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝",
    ]
    grad = ["112;196;78", "112;196;78", "94;182;66",
            "78;168;56", "62;154;48", "56;150;46"]
    print()

    for line, rgb in zip(art, grad):
        print(f"\033[38;2;{rgb}m{line}{C_RST}")

    print(f"\033[38;2;112;196;78m  HEIG-VD/REDS — \033[1;38;2;56;150;46m"
          f"Specification-driven Platform for Embedded "
          f"Agentic Reasoning{C_RST}\n")


def _corpus_summary(corpora):
    """One line for the banner, whether the session has one corpus or several."""

    if not isinstance(corpora, list):
        corpora = [(corpora, "")]

    if len(corpora) == 1:
        return f"{corpora[0][0].count()} chunks"

    total = sum(c.count() for c, _ in corpora)
    names = ", ".join((p or ".") for _, p in corpora)

    return f"{total} chunks across {len(corpora)} corpora ({names})"


def banner(collection, history, n_rules, model_name, n_mem=0):
    cwd = os.getcwd()
    home = os.path.expanduser("~")

    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]

    rows = [
        f"{C_DIM}model:{C_RST}    {model_name} + RAG",
        f"{C_DIM}corpus:{C_RST}   {session_workspace.PROJECT}  {C_DIM}·{C_RST}  "
        + (_corpus_summary(collection) if collection else "no RAG")
        + f"  {C_DIM}·{C_RST}  {n_rules} rules",
        f"{C_DIM}history:{C_RST}  "
        + ("fresh (stored conversation not loaded)" if fresh_session()
           else f"{len(history)} messages")
        + (f"   {C_DIM}knowledge:{C_RST} {n_mem}" if n_mem else ""),
        f"{C_DIM}tools in:{C_RST} {cwd}  {C_DIM}(current directory){C_RST}",
    ]

    # Checking the binding re-verifies the bound document, which for a
    # document of hundreds of thousands of units is minutes, not a blink.
    with Spinner("Checking the bound standard…"):
        standard = standard_row()

    if standard is not None:
        rows.insert(2, standard)

    rows.insert(-1, content_row())
    rows.append(permissions_row(session_workspace.EXECUTION_MODE))

    if session_workspace.WORKSPACE is not None and session_workspace.WORKSPACE.extra_roots:
        # Widening the write boundary must never be silent.

        rows.append(f"{C_DIM}writable:{C_RST} cwd + "
                    f"{len(session_workspace.WORKSPACE.extra_roots)} "
                    f"declared corpora {C_DIM}(--single-root to restrict to the "
                    f"cwd){C_RST}")
        rows.append(f"{C_DIM}readable:{C_RST} anywhere else by absolute path, "
                    f"read-only {C_DIM}(credential stores refused){C_RST}")

    # info (not an error): the cwd is outside the corpus tree → RAG context
    # may be about another tree than the one tools act on.

    if not (session_workspace.PROJECT_ROOT == session_workspace.CORPUS_ROOT
            or session_workspace.PROJECT_ROOT.startswith(session_workspace.CORPUS_ROOT + "/")):
        rows.append(f"{C_DIM}note: cwd is outside the '{session_workspace.PROJECT}' "
                    f"corpus tree ({session_workspace.CORPUS_ROOT}){C_RST}")

    for r in rows:
        print(f"  {r}")

    print()
    # /standard and /finetune were dispatched but named nowhere — not on this
    # line, not in --help. "Operator-only" means the MODEL never reaches them;
    # it was never meant to mean the operator has to know they exist.

    print(f"  {C_DIM}!read !ls !grep !find !edit !run !web   "
          f"{' '.join(SESSION_COMMANDS)}\n"
          f"  {' '.join(OPERATOR_COMMANDS)}{C_RST}\n")
