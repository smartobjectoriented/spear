"""Provider-neutral declarations and deterministic registration for tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Mapping
import re

from model_backend import ToolDefinition


class ToolCategory(StrEnum):
    COMMAND = "command"
    FILE_WRITE = "file_write"
    RETRIEVAL = "retrieval"
    MEMORY = "memory"
    WEB = "web"
    OTHER = "other"


class ToolMutability(StrEnum):
    READ_ONLY = "read_only"
    MUTATING = "mutating"
    CONDITIONAL = "conditional"


@dataclass(frozen=True)
class ToolResultPolicy:
    model_context_chars: int = 6000
    store_large_results: bool = True
    preview_chars: int = 2000

    def __post_init__(self) -> None:
        if self.model_context_chars < 1 or self.preview_chars < 1:
            raise ValueError("tool result limits must be positive")

        if self.preview_chars > self.model_context_chars:
            raise ValueError("preview_chars cannot exceed model_context_chars")


#: The execution modes a session can be in, and therefore the only values a
#: spec may name. They are the values of tool_runtime.ExecutionMode, spelled
#: here rather than imported: the registry describes tools and must not
#: depend on the command runtime that runs them. `test_execution_mode_gate`
#: holds the two lists together.
VALID_EXECUTION_MODES = frozenset({"safe", "ask", "auto"})


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, object]
    category: ToolCategory = ToolCategory.OTHER
    mutability: ToolMutability = ToolMutability.READ_ONLY
    required_capabilities: tuple[str, ...] = ()
    execution_modes: tuple[str, ...] = ()
    result_policy: ToolResultPolicy = field(default_factory=ToolResultPolicy)
    roles: frozenset[str] = field(default_factory=lambda: frozenset({"main"}))
    model_visible: bool = True
    handler_key: str | None = None

    def __post_init__(self) -> None:
        # The name goes into a provider's tool schema verbatim, so it is held
        # to the shape every backend accepts.

        if not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)?", self.name):
            raise ValueError(
                "tool name must be a simple name or one-level dotted namespace")

        # The description is the only thing telling the model when to reach for
        # this tool, so an empty one is a defect, not a default.

        if not self.description.strip():
            raise ValueError("tool description cannot be empty")

        _validate_schema(self.input_schema)

        if not self.roles:
            raise ValueError("tool roles cannot be empty")

        # What the router will enforce, checked where the mistake is made.
        #
        # An unknown mode is not a narrower permission, it is a silently
        # different one: ("asks", "auto") denies ASK, which is the mode it
        # was meant to allow, and nothing says so until a tool call fails.

        unknown = tuple(mode for mode in self.execution_modes
                        if mode not in VALID_EXECUTION_MODES)

        if unknown:
            raise ValueError(
                f"unknown execution mode(s) {unknown} for tool "
                f"'{self.name}': choose from "
                f"{tuple(sorted(VALID_EXECUTION_MODES))}")

        # An empty declaration means UNRESTRICTED at the router -- which is
        # right for a tool that reads, and is how every read-only spec is
        # written. For a tool that writes it means "runs in safe mode too",
        # which is never what anyone intends: `save_skill` carried exactly
        # that declaration and was saved only by its own handler remembering
        # to ask for authorization. So a mutating tool states its modes, and
        # says nothing that would let it run where nothing may be written.
        #
        # This is configuration validation and stays that way: the router
        # goes on enforcing whatever a spec declares, for every tool, without
        # consulting mutability.

        if self.mutability == ToolMutability.MUTATING:
            if not self.execution_modes:
                raise ValueError(
                    f"mutating tool '{self.name}' must declare "
                    f"execution_modes: an empty declaration is unrestricted, "
                    f"which would let it run in safe mode")

            if "safe" in self.execution_modes:
                raise ValueError(
                    f"mutating tool '{self.name}' cannot declare 'safe': "
                    f"safe mode writes nothing")


ToolHandler = Callable[[Any, Mapping[str, object]], Any]


class ToolRegistry:
    """An explicitly owned registry; registration order never affects output."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[ToolSpec, ToolHandler | None]] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler | None) -> None:
        if spec.name in self._entries:
            raise ValueError(f"duplicate tool: {spec.name}")

        # A command is dispatched through the caller's execution boundary
        # rather than a handler, which is why it alone may register without one.

        if spec.category != ToolCategory.COMMAND and handler is None:
            raise ValueError(f"tool {spec.name} requires a handler")

        self._entries[spec.name] = (spec, handler)

    def get(self, name: str) -> ToolSpec:
        try:
            return self._entries[name][0]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def handler(self, name: str) -> ToolHandler | None:
        try:
            return self._entries[name][1]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def list_specs(
        self, *, role: str | None = None, model_visible: bool | None = None,
        names: set[str] | frozenset[str] | None = None,
    ) -> tuple[ToolSpec, ...]:
        """The registered specs matching every filter given, in registration order."""

        specs = []

        for name, (spec, _) in self._entries.items():
            if role is not None and role not in spec.roles:
                continue

            if model_visible is not None and spec.model_visible != model_visible:
                continue

            if names is not None and name not in names:
                continue

            specs.append(spec)

        return tuple(specs)

    def definitions_for_model(
        self, *, role: str = "main",
        names: set[str] | frozenset[str] | None = None,
    ) -> tuple[ToolDefinition, ...]:
        return tuple(
            ToolDefinition(spec.name, spec.description, dict(spec.input_schema))
            for spec in self.list_specs(
                role=role, model_visible=True, names=names,
            )
        )

    def openai_definitions_for_model(
        self, *, role: str = "main",
        names: set[str] | frozenset[str] | None = None,
    ) -> list[dict[str, object]]:
        return [{"type": "function", "function": {
            "name": definition.name,
            "description": definition.description,
            "parameters": dict(definition.input_schema),
        }} for definition in self.definitions_for_model(role=role, names=names)]


def _validate_schema(schema: Mapping[str, object]) -> None:
    """The JSON-Schema subset every supported provider accepts."""

    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        raise ValueError("tool input schema must be an object schema")

    properties = schema.get("properties", {})
    required = schema.get("required", ())

    if not isinstance(properties, Mapping):
        raise ValueError("tool schema properties must be a mapping")

    if not isinstance(required, (list, tuple)) or not all(
        isinstance(name, str) for name in required
    ):
        raise ValueError("tool schema required must be a string list")

    # A required argument the properties never declare would ask the model for
    # something the router could not then type-check.

    if any(name not in properties for name in required):
        raise ValueError("required tool arguments must exist in properties")

    for name, value in properties.items():
        if not isinstance(name, str) or not isinstance(value, Mapping):
            raise ValueError("tool properties must be named schemas")

        if value.get("type") not in {"string", "integer", "number", "boolean", "array", "object"}:
            raise ValueError(f"unsupported schema type for {name}")


def native_tool_specs() -> tuple[ToolSpec, ...]:
    """The current production tool set, independently of provider syntax."""

    string = lambda description=None: {
        "type": "string", **({"description": description} if description else {}),
    }

    specs = (
        ToolSpec(
            "bash",
            # How to run the tests belongs HERE, before the first attempt, not
            # in the refusal that follows it: a run asked to "run the test"
            # spent two calls on `python3 -c`, was refused both times, and
            # finished the turn having verified nothing.
            "Run a shell command from the project root. Read-only commands "
            "(cat, ls, grep, find, ...) run immediately; anything else asks "
            "the user first. One line only — no heredocs. "
            "To run tests: make/cmake/ctest, pytest, or "
            "`python3 -m unittest discover -s tests` for a Python tests/ "
            "package. Send that command ALONE — no `||` or `&&` fallback "
            "chain, no second command appended. If it fails, read the failure "
            "and fix the code; do not reach for another way to run it. "
            "`python3 -c ...` and running a script directly (`python3 x.py`, "
            "`./x.py`) are REFUSED — the interpreter runs arbitrary code; "
            "those two module forms are the only ones available, and adding "
            "one as a fallback only refuses the whole command.",
            {"type": "object", "properties": {
                "command": string("shell command")}, "required": ["command"]},
            ToolCategory.COMMAND, ToolMutability.CONDITIONAL,
            ("dynamic_command_policy",), ("safe", "ask", "auto"),
            roles=frozenset({"main", "explorer", "reviewer", "planning"}), handler_key="bash",
        ),

        ToolSpec(
            "edit_file",
            "Replace text in an existing file. old_text must match the file "
            "content EXACTLY (read the region first) and be unique. For a "
            "full rewrite use write_file instead.",
            {"type": "object", "properties": {"path": string(),
             "old_text": string(), "new_text": string()},
             "required": ["path", "old_text", "new_text"]},
            ToolCategory.FILE_WRITE, ToolMutability.MUTATING,
            ("workspace_write",), ("ask", "auto"), handler_key="edit_file",
        ),

        ToolSpec(
            "write_file",
            "Create or fully overwrite a file. For a comprehensive "
            "improvement, pass the COMPLETE new file as content.",
            {"type": "object", "properties": {"path": string(),
             "content": string()}, "required": ["path", "content"]},
            ToolCategory.FILE_WRITE, ToolMutability.MUTATING,
            ("workspace_write",), ("ask", "auto"), handler_key="write_file",
        ),

        ToolSpec(
            "append_file", "Append content at the END of an existing file.",
            {"type": "object", "properties": {"path": string(),
             "content": string()}, "required": ["path", "content"]},
            ToolCategory.FILE_WRITE, ToolMutability.MUTATING,
            ("workspace_write",), ("ask", "auto"), handler_key="append_file",
        ),

        ToolSpec(
            "delete_file",
            "Delete a file that should no longer exist — a document you have "
            "just made redundant, a stray artefact. It is checkpointed, so "
            "/undo restores it. Emptying a file is NOT deleting it: an empty "
            "document still exists and is still reported by tools that walk "
            "the tree. Say why in `reason`.",
            {"type": "object", "properties": {
                "path": string(),
                "reason": string("why this file should no longer exist"),
            }, "required": ["path", "reason"]},
            ToolCategory.FILE_WRITE, ToolMutability.MUTATING,
            ("workspace_write",), ("ask", "auto"), handler_key="delete_file",
        ),

        ToolSpec(
            "remember", "Persist a short durable fact into your long-term "
            "memory (re-injected into every future session). Use for project "
            "facts, conventions, paths and gotchas worth keeping — not for "
            "transient details.",
            {"type": "object", "properties": {"note": string(
                "one concise sentence")}, "required": ["note"]},
            ToolCategory.MEMORY, ToolMutability.MUTATING,
            ("persistent_memory_write",), ("ask", "auto"), handler_key="remember",
        ),

        # The tool that opens the write gate on a turn that must satisfy an
        # authoritative source. It writes nothing, which is why it is
        # read-only: it is the record of what the turn found and what it
        # intends, and the gate it opens is the one in front of the tools
        # that do write.
        #
        # ONE requirement per call, and seven flat strings. It began as an
        # array of objects, which is the shape the data has and the wrong
        # shape to ask a model for: measured on one run, forty-three
        # consecutive calls were rejected before the arguments were ever
        # parsed, and the turn gave up on planning and went back to trying to
        # write. Nested tool-call payloads are exactly what this tool guide
        # already warns against for edits, and a plan is no different.
        #
        # Every field is required because the gate is not a formality. An
        # item without its evidence is an assertion, one without the current
        # behaviour is a feature request, and one without a validation is a
        # change nothing will ever check. The refusal names whichever is
        # missing, so a turn learns what the gate wants from the gate itself
        # rather than from a system prompt nobody re-reads.
        ToolSpec(
            "plan_change",
            "Record ONE evidence-backed change that a standard, specification "
            "or API contract requires. REQUIRED before edit_file/write_file "
            "on such a task: until one is accepted, the tools that modify "
            "files are refused. Call it again for each further requirement — "
            "one call per requirement, never a list. Two fields are checked "
            "against what this turn actually read: `requirement_evidence` "
            "must cite a provision you retrieved, and "
            "`implementation_evidence` must name a file you opened. Set "
            "`supersedes` to the `requirement` of an earlier entry when what "
            "you find later makes it wrong.",
            {"type": "object", "properties": {
                "requirement": string(
                    "what the authoritative source demands"),
                "requirement_evidence": string(
                    "the provision, as you read it: e.g. \"Rule 5.2.1-3\" "
                    "or a retrieval handle"),
                "current_behaviour": string("what the code does today"),
                "implementation_evidence": string(
                    "the file you read it in, e.g. src/a/b.c"),
                "gap": string("how the two differ"),
                "correction": string(
                    "the change you intend, naming the function or symbol it "
                    "lands in"),
                "validation": string(
                    "the test or command that will prove it"),
                "supersedes": string(
                    "the `requirement` of an entry this call replaces"),
                "reason": string("why that entry no longer holds"),
                "new_evidence": string("what you found that invalidated it"),
            }, "required": ["requirement", "requirement_evidence",
                            "current_behaviour", "implementation_evidence",
                            "gap", "correction", "validation"]},
            ToolCategory.OTHER, ToolMutability.READ_ONLY,
            handler_key="plan_change",
        ),

        ToolSpec(
            "search_corpus", "Search THIS project's indexed corpora (its "
            "sources, the libc headers, the POSIX man pages) for code or an "
            "API you were not given. Use it before writing a helper by hand "
            "or pulling in a third-party library: the C library often already "
            "provides the function. Query in ENGLISH keywords describing what "
            "the code DOES, e.g. 'match filename against shell wildcard "
            "pattern' — not in the user's language, and not a bare symbol name.",
            {"type": "object", "properties": {"query": string()},
             "required": ["query"]},
            ToolCategory.RETRIEVAL, ToolMutability.READ_ONLY,
            roles=frozenset({"main", "explorer", "reviewer", "planning"}),
            handler_key="search_corpus",
        ),

        ToolSpec(
            "search_internet", "Search the internet (DuckDuckGo). Use ONLY "
            "when the Retrieved Context and search_corpus lack the answer. "
            "Query = English topic keywords, never project recipe or "
            "version names.",
            {"type": "object", "properties": {"query": string()},
             "required": ["query"]},
            ToolCategory.WEB, ToolMutability.READ_ONLY, ("network",),
            handler_key="search_internet",
        ),

        # search_internet finds pages; this one reads them. Without it the
        # model could name a document and never open it, and its only fallback
        # -- curl in the shell -- is refused by the command policy in safe
        # mode, which is the default.
        ToolSpec(
            "fetch_url", "Read ONE web page or document (HTML, PDF, text), or "
            "SAVE it to disk. Without save_as it returns text: a long PDF "
            "comes back in slices, and the result names the `pages` range that "
            "continues it — never re-fetch the same url hoping for more. With "
            "save_as=<path> it writes the file itself, which is what a "
            "document meant for ingestion needs. http/https only, public "
            "addresses only.",
            {"type": "object", "properties": {
                "url": string(),
                "pages": string("PDF page range to read, e.g. \"61-121\"."),
                "save_as": string("Workspace path to write the file to. "
                                  "Downloads instead of reading."),
            }, "required": ["url"]},
            ToolCategory.WEB, ToolMutability.READ_ONLY, ("network",),
            handler_key="fetch_url",
        ),
    )

    return specs
