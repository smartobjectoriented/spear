"""What SPEAR adds to every request, handed to Claude Code the same way.

SPEAR does not leave retrieval to the model: each request is searched in the
corpus index and shown with the knowledge recorded for the workspace. Claude
Code runs a hook on every prompt and reads what it prints, so the hook prints
that same text, built by the same functions:

    ## Retrieved Context      corpus_search.retrieve_context, on the prompt
    ## Workspace knowledge    workspace_knowledge.select, on the prompt

The embedder takes seconds to load, which a hook started afresh on each
prompt would pay each time. The MCP server of the session is already there
with it loaded, so it also answers on a socket (corpus_mcp --context-socket)
and the hook asks it; without a server, the hook does the work itself.

    spear-claude-context [--socket PATH] [--corpus NAME]    the hook
    spear-knowledge list | show <id> | propose '<json>'     the records
"""

from __future__ import annotations

import json
import os
import shlex
import socket
import sys
import threading

RETRIEVED_HEADER = "## Retrieved Context"
RETRIEVED_NOTE = ("Passages of the indexed corpus that match this request, each headed by "
                  "its path relative to the corpus root.")
ANSWER_TIMEOUT = 120


def workspace(corpus, cwd, projects):
    """(workspace id, root) as SPEAR names them: a registered corpus by its
    name, any other tree by where it is."""
    if corpus and corpus in projects:
        return corpus, os.path.realpath(projects[corpus]["path"])

    root = os.path.realpath(cwd or ".")

    return f"adhoc:{root}", root


def retrieved(index, prompt, corpus):
    """The retrieved passages of a prompt, or "" when the corpus has no index."""
    if index is None or not corpus or corpus not in index.names():
        return ""

    context = index.context(prompt, corpus)

    if not context or not context.strip():
        return ""

    return f"{RETRIEVED_HEADER}\n\n{RETRIEVED_NOTE}\n\n{context}"


def knowledge(prompt, workspace_id, root):
    """The workspace's active records, selected for a prompt as a turn's are."""
    from context import workspace_knowledge

    store = workspace_knowledge.KnowledgeStore(
        workspace_knowledge.default_path(),
        common=workspace_knowledge.default_common_path())

    try:
        store.validate(workspace_id, root)
        records = store.list(workspace_id, (workspace_knowledge.Lifecycle.ACTIVE,))

        return workspace_knowledge.select(records, prompt).text.strip()
    finally:
        store.close()


def compose(index, prompt, corpus, cwd, projects):
    """Everything a prompt is shown, as one text; "" when there is nothing."""
    parts = []

    for part in (lambda: retrieved(index, prompt, corpus),
                 lambda: knowledge(prompt, *workspace(corpus, cwd, projects))):
        try:
            text = part()
        except Exception as exc:
            print(f"spear-claude-context: {type(exc).__name__}: {exc}", file=sys.stderr)
            text = ""

        if text:
            parts.append(text)

    return "\n\n".join(parts)


# ── the session's server answers the hook ────────────────────────────

def serve(index, path, projects):
    """Answer the hook on a socket at `path`, from a thread, until the process
    ends. One request per connection: a JSON object, then the text."""
    try:
        os.unlink(path)
    except OSError:
        pass

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(path)
    os.chmod(path, 0o600)
    listener.listen(4)

    def loop():
        while True:
            connection, _ = listener.accept()

            with connection:
                try:
                    request = json.loads(_read(connection))
                    text = compose(index, str(request.get("prompt") or ""),
                                   request.get("corpus"), request.get("cwd"), projects)
                    connection.sendall(text.encode("utf-8"))
                except Exception as exc:
                    print(f"spear-claude-context: {type(exc).__name__}: {exc}",
                          file=sys.stderr)

    threading.Thread(target=loop, daemon=True).start()

    # The first question should not be the one that waits for the embedder.

    if index.default:
        threading.Thread(target=lambda: index.context("warm up", index.default),
                         daemon=True).start()

    return listener


def _read(connection):
    chunks = []

    while True:
        chunk = connection.recv(65536)

        if not chunk:
            return b"".join(chunks).decode("utf-8")

        chunks.append(chunk)


def ask(path, request):
    """The server's answer, or None when nobody answers at `path`."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(ANSWER_TIMEOUT)
            connection.connect(path)
            connection.sendall(json.dumps(request).encode("utf-8"))
            connection.shutdown(socket.SHUT_WR)

            return _read(connection)
    except OSError:
        return None


# ── the commands ─────────────────────────────────────────────────────

def _option(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv[:-1] else None


def _alone(prompt, corpus, cwd):
    from cli import corpus_mcp, corpus_registry

    projects = corpus_registry.load_projects()
    index = None

    if corpus:
        try:
            index = corpus_mcp.CorpusIndex(corpus)
        except Exception:
            index = None

    return compose(index, prompt, corpus, cwd, projects)


def hook(argv, stdin, out):
    """Claude Code's UserPromptSubmit hook: the event on stdin, the context on
    stdout. It never fails a prompt: with nothing to add it prints nothing."""
    try:
        event = json.loads(stdin.read() or "{}")
    except ValueError:
        event = {}

    prompt = str(event.get("prompt") or "").strip()

    if not prompt:
        return 0

    path, corpus = _option(argv, "--socket"), _option(argv, "--corpus")
    request = {"prompt": prompt, "corpus": corpus, "cwd": event.get("cwd") or os.getcwd()}
    text = ask(path, request) if path else None

    if text is None:
        # The libraries under the search print; the hook's stdout is read as
        # context, so it is kept for the result alone.

        saved = os.dup(1)
        os.dup2(2, 1)

        try:
            text = _alone(prompt, corpus, request["cwd"])
        except Exception as exc:
            print(f"spear-claude-context: {type(exc).__name__}: {exc}", file=sys.stderr)
            text = ""
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)

    if text:
        out.write(text + "\n")
        out.flush()

    return 0


def knowledge_command(argv, out):
    """`spear-knowledge` as SPEAR's own turns have it, for the tree the
    current directory is in."""
    from cli import corpus_registry
    from context import workspace_knowledge

    projects = corpus_registry.load_projects()
    workspace_id, _ = workspace(corpus_registry.detect_project(projects), os.getcwd(), projects)
    store = workspace_knowledge.KnowledgeStore(
        workspace_knowledge.default_path(),
        common=workspace_knowledge.default_common_path())

    try:
        code, text = workspace_knowledge.Door(store, workspace_id).run(
            shlex.join([workspace_knowledge.COMMAND, *argv]))
    finally:
        store.close()

    out.write(text + "\n")

    return code


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv[:1] == ["knowledge"]:
        return knowledge_command(argv[1:], sys.stdout)

    return hook(argv, sys.stdin, sys.stdout)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    sys.exit(main())
