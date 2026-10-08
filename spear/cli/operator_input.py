"""Operator input outside the agent loop: !commands, hints, completion."""

import os
import re
import atexit
import readline
from standard.standard_commands import complete_standard
from cli import session_workspace
from cli.chat_settings import STANDARD_STORE, STATE_DIR
from cli.corpus_search import archive_search, attached_corpus_names
from cli.session_workspace import edit_file, read_file, run_cmd
from cli.terminal_ui import (
    C_DIM, C_ERR, C_RST, C_TOOL, C_WARN, Spinner, show_web_sources,
    tool_result, tool_use,
)
from cli.tool_handlers import web_search


# Corpus names that are also ordinary directory words: mentioning them in a
# sentence says nothing about where the user wants to work.

_HINT_STOPWORDS = {"src", "lib", "bin", "doc", "docs", "test", "tests",
                   "build", "include", "data", "tmp"}
_HINTED_CORPORA = set()


def corpus_mention_hint(user_input, projects, current):
    """Name a registered corpus the question mentions but the tools are not in.

    Strictly informational.  The cwd IS the sandbox root, so no phrase in a
    question may relocate it — letting natural language move where writes land
    would hand the workspace boundary to the model's reading of a sentence.
    What the harness owes the user is what it already knows: that the name is
    registered, and where.  Longest name first so `micropython-so3` wins over
    `so3`; once per name per session, so it informs without nagging.
    """

    # A corpus this session already retrieves from is not somewhere to go: an
    # umbrella federates everything under the cwd, so telling the user to `cd
    # so3` while so3's chunks are arriving as so3/usr/src/... is advice to
    # leave a session that can already reach the file.

    attached = set(attached_corpus_names(
        projects.get(current) or session_workspace.PROJECT_SPEC or {}, projects))

    for name, spec in sorted(projects.items(), key=lambda kv: -len(kv[0])):
        if name == current or name in _HINTED_CORPORA or name in attached:
            continue

        if name.lower() in _HINT_STOPWORDS:
            continue

        # A trailing slash means the word is a path fragment ("src/main.c"),
        # not a corpus the user is naming.

        if not re.search(rf"\b{re.escape(name)}\b(?!/)", user_input, re.I):
            continue

        path = spec.get("path", "")

        if not os.path.isdir(path):
            continue

        _HINTED_CORPORA.add(name)

        return (f"'{name}' is a registered corpus ({path}), but your tools run "
                f"in {session_workspace.PROJECT_ROOT}. To work there: cd {path} && spear-chat")

    return None


# ── auto-detect intent → run tools before LLM ───────────────────────

def auto_detect_and_run(user_msg, collection):
    """
    Detect if the user is asking to read/list/grep something.
    If so, run the tool and return the result to inject into LLM context.
    Returns (extra_context, description) or (None, None).
    """
    msg = user_msg.lower()

    # "cherche sur internet/le web/en ligne X" → web search
    # (must precede the local grep detection, which also matches "cherche")

    m = re.search(
        r"(?:cherche|recherche|trouve|regarde)[a-z]*\s+(?:moi\s+)?(?:sur\s+)?"
        r"(?:internet|le web|le net|en ligne|web|google)\s*[:,]?\s*(.+)",
        msg,
    )

    if m:
        query = m.group(1).strip().rstrip("?").strip()

        if query:
            tool_use("Web", query, color=C_TOOL)

            with Spinner("Searching the web…"):
                result = web_search(query)

            show_web_sources(result)

            return f"[web results]\n{result}", f"Web results for '{query}'"

    # Memory-reference questions ("te souviens-tu", "la dernière fois",
    # "qu'on avait corrigé") → pre-run a semantic search over the archive.
    # Same rationale as the freshness detector: the local model answers
    # from (polluted) rolling history instead of calling search_history.

    if re.search(r"souviens|rappelle[sz]?[- ]|derni[eè]re fois|qu'?on avait|"
                 r"on avait (vu|fait|corrig|r[ée]solu)|remember when|"
                 r"last time we", msg):
        tool_use("History", user_msg[:70], color=C_TOOL)
        result = archive_search(user_msg, n=4)
        tool_result(result, max_lines=6)

        return (f"[past conversation excerpts found by semantic search — "
                f"ground your answer on these, do not invent details]\n"
                f"{result}", "Archive search")

    # Freshness questions ("dernière version de X", "latest release") →
    # pre-run a web search. The local model won't reliably call web_search
    # on its own (it answers from stale training data with invented dates);
    # the v3 adapter should learn the reflex — this is the safety net.

    if (re.search(r"(derni[eè]re|latest|last|newest|nouvelle|plus r[ée]cente)"
                  r"\s+(version|release|stable)|vient de sortir|just released",
                  msg)
            and not re.search(r"projet|notre|utilis|du repo|dans le code|"
                              r"du build|in the (project|repo|code)", msg)):
        query = re.sub(r"^(quelle?\s+est\s+la\s+|what\s+is\s+the\s+)", "",
                       user_msg.strip().rstrip(" ?"))
        tool_use("Web", query, color=C_TOOL)

        with Spinner("Searching the web…"):
            result = web_search(query)

        show_web_sources(result)

        return (f"[web results — ground your answer on these; your training "
                f"data is outdated for release questions]\n{result}",
                f"Web results for '{query}'")

    # "lis/lire/montre/affiche le fichier X"

    m = re.search(
        r"(?:lis|lire|montre|affiche|ouvre|cat|regarde|voir)\s+(?:le fichier\s+|moi\s+)?([^\s,]+\.\w+)",
        msg,
    )

    if m:
        path = m.group(1)
        tool_use("Read", path)
        content = read_file(path)

        return content, f"Contenu de {path}"

    # "liste/parcours/ls le répertoire X" or "quels fichiers dans X"

    m = re.search(
        r"(?:liste|parcour[st]?|ls|quels? fichiers?|répertoire|dossier|contenu de|contenu du)\s+(?:le |du |de |dans |)\s*([^\s,]+/?)",
        msg,
    )

    if m:
        path = m.group(1).rstrip("?")

        if path and not path.endswith((".bb", ".bbclass", ".inc", ".conf", ".py", ".sh", ".md")):
            tool_use("List", path)
            result = run_cmd(f"ls -la {path}", need_confirm=False)

            return result, f"Contenu de {path}"

    # "cherche/grep/trouve X dans Y" or just "grep X"

    m = re.search(
        r"(?:cherche|grep|trouve|recherche|où est)\s+['\"]?(\S+)['\"]?\s*(?:dans\s+(\S+))?",
        msg,
    )

    if m:
        pattern = m.group(1)
        where = m.group(2) or "build/"
        cmd = f"grep -rn --max-count=30 '{pattern}' {where} --include='*.bb' --include='*.bbclass' --include='*.bbappend' --include='*.inc' --include='*.conf'"
        tool_use("Grep", pattern + " dans " + where)
        result = run_cmd(cmd, need_confirm=False)

        return result, f"Résultats grep pour '{pattern}'"

    # "corrige/fixe/répare/remplace/change X en Y dans Z"

    m = re.search(
        r"(?:corrige|fixe|répare|remplace|change|modifie|renomme|fix|correct|replace)"
        r".*?(?:dans|in|le fichier|fichier)\s+([^\s,]+\.\w+)",
        msg,
    )

    if m:
        path = m.group(1)
        tool_use("Read", path + " (pour modification)")
        content = read_file(path)

        return content, f"Lecture de {path} pour modification"

    # "corrige/fixe" without an explicit file → look in recent history

    if re.search(r"(?:corrige|fixe|répare|fix|correct)", msg):
        return None, None

    return None, None


# ── explicit !commands ───────────────────────────────────────────────

def handle_bang_command(cmd_line):
    """Handle !read, !ls, !grep, !edit, !run commands."""
    parts = cmd_line.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("!read", "!cat"):
        if not arg:
            print(f"{C_ERR}Usage: !read <path>{C_RST}")
            return None

        tool_use("Read", arg)
        content = read_file(arg)
        print(content[:5000])

        if len(content) > 5000:
            print(f"{C_DIM}... ({len(content)} chars total){C_RST}")

        return content

    if cmd in ("!ls", "!dir"):
        path = arg or "."
        tool_use("List", path)
        result = run_cmd(f"ls -la {path}", need_confirm=False)
        print(result)

        return result

    if cmd == "!web":
        if not arg:
            print(f"{C_ERR}Usage: !web <query>{C_RST}")
            return None

        tool_use("Web", arg, color=C_TOOL)
        result = web_search(arg)
        print(result[:4000])

        return f"[résultats web pour: {arg}]\n{result}"

    if cmd == "!grep":
        if not arg:
            print(f"{C_ERR}Usage: !grep <pattern> [path]{C_RST}")
            return None

        parts2 = arg.split(None, 1)
        pattern = parts2[0]
        where = parts2[1] if len(parts2) > 1 else "build/"
        grep_cmd = f"grep -rn --max-count=40 '{pattern}' {where} --include='*.bb' --include='*.bbclass' --include='*.bbappend' --include='*.inc' --include='*.conf'"
        tool_use("Grep", grep_cmd)
        result = run_cmd(grep_cmd, need_confirm=False)
        print(result)

        return result

    if cmd == "!find":
        if not arg:
            print(f"{C_ERR}Usage: !find <pattern>{C_RST}")
            return None

        find_cmd = f"find build/ -name '{arg}' -not -path '*/tmp/*'"
        tool_use("Find", find_cmd)
        result = run_cmd(find_cmd, need_confirm=False)
        print(result)

        return result

    if cmd == "!edit":
        print(f"{C_WARN}Interactive usage:{C_RST}")

        try:
            path = input("  File: ").strip()
            old = input("  Text to replace: ").strip()
            new = input("  New text: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        result = edit_file(path, old, new)
        print(result)

        return result

    if cmd in ("!run", "!sh", "!exec"):
        if not arg:
            print(f"{C_ERR}Usage: !run <command>{C_RST}")
            return None

        result = run_cmd(arg, need_confirm=False)
        print(result)

        return result

    print(f"{C_ERR}Unknown command: {cmd}{C_RST}")
    print(f"Available: !read !ls !grep !find !edit !run !web")

    return None


#: The slash commands, as the banner lists them and as Tab completes them:
#: one list, so a command cannot be completed and go unlisted, or the reverse.
SESSION_COMMANDS = ("/search", "/reindex", "/corpus", "/history", "/skills",
                    "/remember", "/recall", "/knowledge", "/forget", "/good", "/bad",
                    "/undo", "/clear", "/model", "/tools")
OPERATOR_COMMANDS = ("/standard", "/finetune")


def _path_candidates(text):
    """Paths completing `text`. A lone directory is opened rather than
    offered: readline would append a space after it and stop the walk."""
    import glob

    expanded = os.path.expanduser(text)
    found = sorted(glob.glob(glob.escape(expanded) + "*"))

    if len(found) == 1 and os.path.isdir(found[0]):
        inside = sorted(glob.glob(os.path.join(glob.escape(found[0]), "*")))
        found = inside or [found[0] + "/"]

    home = os.path.expanduser("~")

    if text.startswith("~"):
        found = ["~" + item[len(home):] if item.startswith(home) else item
                 for item in found]

    return [item + "/" if os.path.isdir(os.path.expanduser(item))
            and not item.endswith("/") else item for item in found]


def completion_candidates(line, text):
    """What Tab offers for `text`, the word being typed at the end of `line`.

    Only commands are completed. Everything else is a question to the model,
    and a completion there would be a guess at what the user means.
    """
    words = line.split()

    if not line.startswith("/"):
        return []

    if not words or (len(words) == 1 and not line.endswith(" ")):
        return sorted(c for c in SESSION_COMMANDS + OPERATOR_COMMANDS
                      if c.startswith(text))

    if words[0] != "/standard":
        return []

    before = words[1:] if line.endswith(" ") else words[1:-1]

    if len(before) == 1 and before[0] == "ingest":
        return _path_candidates(text)

    return complete_standard(before, text, STANDARD_STORE)


_COMPLETIONS = []


def _complete(text, state):
    if state == 0:
        try:
            line = readline.get_line_buffer()[:readline.get_endidx()]
            _COMPLETIONS[:] = completion_candidates(line, text)
        except Exception:
            # A completer that raises is silently disabled by readline.
            _COMPLETIONS[:] = []

    return _COMPLETIONS[state] if state < len(_COMPLETIONS) else None


INPUT_HISTORY = f"{STATE_DIR}/.input_history"

# Prompt for input(): ANSI codes wrapped in \001/\002 so readline does not
# count them as visible characters (otherwise line editing misaligns).

PROMPT = "\001\033[1m\002> \001\033[0m\002"


def init_readline():
    """Line editing: arrows, Home/End, ctrl+a/e, persistent input history."""
    readline.set_history_length(1000)

    try:
        readline.read_history_file(INPUT_HISTORY)
    except (FileNotFoundError, PermissionError):
        pass

    atexit.register(readline.write_history_file, INPUT_HISTORY)

    # Words split on whitespace only: standard identifiers and revisions hold
    # '-' and '.', which the default delimiters would cut them at.
    readline.set_completer_delims(" \t\n")
    readline.set_completer(_complete)

    if "libedit" in (readline.__doc__ or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
