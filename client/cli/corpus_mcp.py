"""The corpus index as an MCP server: the stdio transport, tools only.

SPEAR's retrieval -- dense and lexical candidates fused by rank, over one
corpus or a federation of them -- is reachable here by any MCP client, so an
assistant that is not SPEAR searches the same index SPEAR does.

stdio is the only transport, on purpose. The server holds private source, and
a transport that listens on a port needs authentication and encryption of its
own; over stdio the client starts the server through whatever already carries
both, ssh included:

    ssh host docker run --rm -i --entrypoint spear-mcp IMAGE --corpus NAME

The one exception is local: with --context the server also answers Claude
Code's prompt hook on a socket of this machine, private to the user
(cli/claude_context.py).

Two tools: `list_corpora` and `search_corpus`. Paths in a result are relative
to the root of the corpus that was asked for, which is how the client's own
checkout names the same files.
"""

from __future__ import annotations

import json
import os
import sys
import threading

SERVER_NAME = "spear-corpus"

# Newest first: a client asking for a version not listed is answered with the
# first one, and decides for itself whether it can speak it.

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

TOP_K = 8
TOP_K_MAX = 20
MAX_CHARS = 12000

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602

TOOLS = (
    {
        "name": "list_corpora",
        "description": (
            "List the indexed source corpora this server can search: name, "
            "number of indexed chunks, and the corpora searched along with it."),
        "inputSchema": {"type": "object", "properties": {},
                        "additionalProperties": False},
    },
    {
        "name": "search_corpus",
        "description": (
            "Search an indexed source corpus (code, build system, "
            "documentation) and return the best matching chunks, each headed "
            "by its file path relative to the corpus root. Ask in natural "
            "language or name identifiers: an identifier in the query is also "
            "matched literally, and its definition ranks first. Use it to "
            "find where something is defined or how it is done before "
            "reading files."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "What to look for."},
                "corpus": {"type": "string",
                           "description": "A name from list_corpora. Omit for "
                                          "the server's default corpus."},
                "top_k": {"type": "integer", "minimum": 1, "maximum": TOP_K_MAX,
                          "description": f"Breadth of the search (default "
                                         f"{TOP_K}); up to twice as many "
                                         f"chunks come back."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
)


class ToolError(Exception):
    """A failure the client's model should read and act on."""


class CorpusIndex:
    """The registered corpora that have an index, and how to search one."""

    def __init__(self, default=None):
        from cli import corpus_registry, corpus_search

        self._search = corpus_search
        self._projects = corpus_registry.load_projects()
        self._client = corpus_search._db()
        self._collections = {}
        self._lock = threading.Lock()
        self.default = default

    def _collection(self, name):
        if name not in self._collections:
            spec = self._projects.get(name)
            found = None

            if spec:
                try:
                    found = self._client.get_collection(
                        name=self._search.collection_name_for(spec))
                except Exception:
                    found = None

            self._collections[name] = found

        return self._collections[name]

    def _attached(self, name):
        """The corpora searched along with `name`, indexed ones only."""
        spec = self._projects[name]
        names = self._search.attached_corpus_names(spec, self._projects)

        return [n for n in names if n != name and self._collection(n) is not None]

    def names(self):
        return sorted(n for n in self._projects if self._collection(n) is not None)

    def describe(self):
        lines = []

        for name in self.names():
            line = f"{name}: {self._collection(name).count()} chunks"
            attached = self._attached(name)

            if attached:
                line += f" (searched with: {', '.join(attached)})"

            if name == self.default:
                line += " [default]"

            lines.append(line)

        return "\n".join(lines) or "(no indexed corpus)"

    def search(self, query, corpus=None, top_k=TOP_K):
        name = corpus or self.default

        if not name:
            names = self.names()

            if len(names) != 1:
                raise ToolError("name a corpus: " + (", ".join(names) or "none is indexed"))

            name = names[0]

        if self._collection(name) is None:
            raise ToolError(f"no indexed corpus named '{name}'; "
                            f"list_corpora names the ones there are")

        with self._lock:
            context, _ = self._search.retrieve_context(self._corpora(name), query, top_k)

        if not context or not context.strip():
            return "(no match)"

        if len(context) > MAX_CHARS:
            context = context[:MAX_CHARS] + "\n… (truncated; ask a narrower query)"

        return context

    def context(self, query, name):
        """What a SPEAR turn retrieves for a request, unasked: the corpus and
        the ones attached to it, at the breadth a turn uses."""
        with self._lock:
            context, _ = self._search.retrieve_context(self._corpora(name), query)

        return context

    def _corpora(self, name):
        """(collection, path prefix) for a corpus and the ones searched with it."""
        root = os.path.realpath(self._projects[name]["path"])
        corpora = [(self._collection(name), "")]

        for other in self._attached(name):
            path = os.path.realpath(self._projects[other]["path"])
            inside = path.startswith(root.rstrip("/") + "/")
            corpora.append((self._collection(other),
                            os.path.relpath(path, root) if inside else path))

        return corpora


def call_tool(index, name, arguments):
    """The text a tool returns; ToolError when the call cannot be served."""

    if not isinstance(arguments, dict):
        raise ToolError("arguments must be an object")

    if name == "list_corpora":
        return index.describe()

    if name == "search_corpus":
        query = arguments.get("query")

        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")

        corpus = arguments.get("corpus")

        if corpus is not None and not isinstance(corpus, str):
            raise ToolError("corpus must be a string")

        top_k = arguments.get("top_k", TOP_K)

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ToolError("top_k must be an integer")

        return index.search(query, corpus or None, max(1, min(top_k, TOP_K_MAX)))

    raise ToolError(f"unknown tool '{name}'")


def handle(index, message):
    """The response to one JSON-RPC message, or None when it gets none."""

    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error(None, INVALID_REQUEST, "not a JSON-RPC 2.0 message")

    method, ident = message.get("method"), message.get("id")
    params = message.get("params") or {}

    # No id: a notification, which is never answered -- nor is a response to a
    # request this server never sends.

    if ident is None or not isinstance(method, str):
        return None

    if not isinstance(params, dict):
        return _error(ident, INVALID_PARAMS, "params must be an object")

    if method == "initialize":
        asked = params.get("protocolVersion")
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]

        return _result(ident, {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": _version()},
        })

    if method == "ping":
        return _result(ident, {})

    if method == "tools/list":
        return _result(ident, {"tools": list(TOOLS)})

    if method == "tools/call":
        try:
            text = call_tool(index, params.get("name"), params.get("arguments") or {})
        except ToolError as exc:
            return _result(ident, _content(str(exc), error=True))
        except Exception as exc:
            return _result(ident, _content(
                f"corpus search failed: {type(exc).__name__}: {exc}", error=True))

        return _result(ident, _content(text))

    return _error(ident, METHOD_NOT_FOUND, f"method not found: {method}")


def _content(text, error=False):
    return {"content": [{"type": "text", "text": text}], "isError": error}


def _result(ident, result):
    return {"jsonrpc": "2.0", "id": ident, "result": result}


def _error(ident, code, text):
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": text}}


def _version():
    try:
        from cli import chat_settings

        return str(getattr(chat_settings, "VERSION", "") or "0")
    except Exception:
        return "0"


def serve(index, lines, out):
    """Answer newline-delimited JSON-RPC from `lines` on `out` until EOF."""

    for line in lines:
        line = line.strip()

        if not line:
            continue

        try:
            message = json.loads(line)
        except ValueError:
            response = _error(None, PARSE_ERROR, "not JSON")
        else:
            response = handle(index, message)

        if response is not None:
            out.write(json.dumps(response, ensure_ascii=False) + "\n")
            out.flush()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    default = os.environ.get("SPEAR_MCP_CORPUS") or None

    if "--corpus" in argv:
        at = argv.index("--corpus")

        if at + 1 >= len(argv):
            print("spear-mcp: --corpus needs a name", file=sys.stderr)

            return 2

        default = argv[at + 1]

    # stdout carries the protocol and nothing else. The retrieval code and the
    # libraries under it print, so they get stderr -- at the descriptor, which
    # is where a native library writes.

    out = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    index = CorpusIndex(default)

    if default and default not in index.names():
        print(f"spear-mcp: no indexed corpus named '{default}' "
              f"(indexed: {', '.join(index.names()) or 'none'})", file=sys.stderr)

        return 2

    # Nobody named a corpus: the one the server was started in, if it is
    # indexed, so that one registration serves whatever tree a session opens.

    if not default:
        from cli import corpus_registry

        here = corpus_registry.detect_project(index._projects)
        index.default = here if here in index.names() else None

    if "--context" in argv:
        from cli import claude_context

        claude_context.serve(index, index._projects)

    serve(index, sys.stdin, out)

    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    sys.exit(main())
