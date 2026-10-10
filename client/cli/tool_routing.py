"""The tool registry, its routing, and the coding core's host ports."""

import os
from pathlib import Path
from models.model_backend import ToolResultBlock
from runtime.cancellation import NEVER_CANCELLED
from harness.tool_primitives import ExecutionMode
from runtime.tracing import new_action_id, new_task_id
from runtime.agent_runtime import AgentRuntime
from runtime.session_store import SessionEventType
from harness.tool_registry import (
    CODING_TOOL_NAMES, ToolCategory, ToolMutability, ToolRegistry,
    ToolSpec, coding_tool_specs, native_tool_specs,
)
from harness.tool_router import (
    ToolExecutionContext, ToolResultStatus, ToolRouter,
)
from cli import session_workspace
from cli.chat_settings import RESULT_STORE, STANDARD_TOOL_SERVICE, TRACE
from cli.session_workspace import (
    _ROUTER_MUTATION_DENIALS, _capture_checkpoint_path, _record_mutation,
    announce_router_refusal, audit_denied_mutation, audit_mutation_result,
    authorize_mutation, resolve_path, run_cmd_result, safe_mode_refusal,
)
from cli.terminal_ui import (
    C_ACCENT, C_BOLD, C_RST, C_TOOL, tool_result, tool_use,
)
from cli.tool_handlers import (
    NETWORK_ENABLED, SANDBOX_DOWN, WORK_PHASE, _classified_handler_result,
    _registered_append_file, _registered_command, _registered_delete_file,
    _registered_edit_file, _registered_fetch_url, _registered_plan_change,
    _registered_remember, _registered_save_skill,
    _registered_search_corpus, _registered_search_history,
    _registered_search_internet, _registered_write_file, _workspace_label,
    _write_gate_closed,
)


# ------------------------------------------------------------------
# The agent core's host: SPEAR's control plane, wired to the session's
# policy. The core (agent/) decides what to call and how results read; every
# path, write, deletion and command it asks for crosses the same checks as
# any other tool call here.

_CORE_LABELS = {"read_file": "Read", "search_files": "Search", "patch": "Update",
                "write_file": "Write", "delete_file": "Delete", "terminal": "Terminal"}


def _core_summary(name, arguments):
    if name == "terminal":
        return str(arguments.get("command") or "")
    if name == "search_files":
        return f"{arguments.get('pattern', '')}  {arguments.get('path') or '.'}"
    return str(arguments.get("path") or "")


def coding_host(agent_context, cache, record):
    """The control plane for one coding turn, as an agent.host.Host."""
    from harness.control_plane import SpearHost
    from agent.host import CommandOutcome

    def context():
        return _execution_context(agent_context, cache,
                                  task_id=getattr(agent_context, "task_id", None),
                                  cancellation=getattr(agent_context, "cancellation", None),
                                  phase_ledger=getattr(agent_context, "work_phase", None))

    def authorize(name, arguments, cwd=None):
        print()
        tool_use(_CORE_LABELS.get(name, name), _core_summary(name, arguments),
                 color=C_ACCENT if name in ("patch", "write_file", "delete_file") else C_TOOL)
        ctx = context()

        if cwd:
            ctx.metadata = {**ctx.metadata, "command_cwd": cwd}

        return TOOL_ROUTER.authorize(ctx, name, arguments)

    def resolve(path, purpose):
        return str(resolve_path(path))

    def write(resolved, content, action):
        label = _workspace_label(Path(resolved))

        if cache.get(SANDBOX_DOWN):
            return "the sandbox is unavailable"

        refused = safe_mode_refusal(action)

        if refused:
            audit_denied_mutation(action, "safe mode", paths=(resolved,))
            return "this session does not allow writes"

        # A build's working copy is the more useful name for it: the edit
        # belongs in the source the build copies from, and saying so is what
        # sends the model there.

        from harness import target_policy

        protected = target_policy.refusal(resolved, label)

        if protected:
            return protected

        blocked = authorize_mutation(f"Modify {C_BOLD}{label}{C_RST} ?",
                                     action=action, paths=(Path(resolved),))

        if blocked is not None:
            return blocked.to_legacy_text()

        _capture_checkpoint_path(context(), Path(resolved))

        try:
            os.makedirs(os.path.dirname(resolved), exist_ok=True)

            with open(resolved, "w", encoding="utf-8", errors="surrogateescape",
                      newline="") as handle:
                handle.write(content)
        except OSError as exc:
            return str(exc)

        _record_mutation(context(), Path(resolved))
        audit_mutation_result(action, f"OK: {label}", paths=(Path(resolved),))

        return None

    def delete(resolved, reason):
        from harness import target_policy

        # A file no write may change may not be deleted either: deleted, it
        # could be written afresh, with nothing left to say it was protected.
        protected = target_policy.refusal(resolved, _workspace_label(Path(resolved)))

        if protected:
            return protected

        result = _registered_delete_file(context(), {"path": resolved, "reason": reason})
        return None if result.status == ToolResultStatus.OK else result.text

    def run(command, script, timeout, output_chars):
        ctx = context()
        result = run_cmd_result(
            command, need_confirm=False, cancellation=ctx.cancellation,
            exec_cmd=script, timeout=timeout, output_chars=output_chars,
            execution_mode=(ExecutionMode.SAFE
                            if ctx.read_only or _write_gate_closed(ctx)
                            or ctx.role in {"explorer", "reviewer", "planning"}
                            else None))
        status = {"ok": "ok", "failed": "ok", "denied": "denied",
                  "cancelled": "cancelled", "timeout": "timeout"}.get(result.status, "error")

        return CommandOutcome(status, (result.stdout or "") + (result.stderr or ""),
                              result.exit_code, result.summary or "")

    def recorded(item):
        tool_result(("refused" if item.refused else
                     "ok" if item.ok else "failed") + (
            f" (exit {item.exit_code})" if item.exit_code not in (None, 0) else ""))
        print()
        record(item)

    from harness.workspace import SandboxSpec

    gateway = getattr(agent_context, "capability_gateway", None)

    if gateway is not None:
        def may_mutate():
            ctx = context()
            return not (ctx.read_only or _write_gate_closed(ctx)
                        or ctx.role in {"explorer", "reviewer", "planning"})

        gateway.may_mutate = may_mutate

    return SpearHost(workspace_root=str(session_workspace.WORKSPACE.root), authorize=authorize,
                     resolve=resolve, write=write, delete=delete, run=run,
                     record=recorded, collation_locale=SandboxSpec().locale,
                     capabilities=gateway,
                     knowledge=getattr(agent_context, "knowledge_door", None))


def _core_only(context, args):
    """The coding tools run inside the agent core; nothing else may call them."""
    return _classified_handler_result(
        "ERROR: this tool runs only in the agent core", status=ToolResultStatus.DENIED)


# Native declarations are owned by tool_registry.native_tool_specs().  The
# registry instance is assembled after the concrete handlers are defined.

def build_tool_registry():
    registry = ToolRegistry()
    handlers = {
        "edit_file": _registered_edit_file,
        "write_file": _registered_write_file,
        "append_file": _registered_append_file,
        "delete_file": _registered_delete_file,
        "remember": _registered_remember,
        "plan_change": _registered_plan_change,
        "search_corpus": _registered_search_corpus,
        "search_internet": _registered_search_internet,
        "fetch_url": _registered_fetch_url,
        "read_file": _core_only,
        "search_files": _core_only,
        "patch": _core_only,
    }

    for spec in native_tool_specs() + coding_tool_specs():
        registry.register(spec, handlers.get(spec.name))

    hidden = (
        (ToolSpec(
            "save_skill", "Save a reusable skill.",
            {"type": "object", "properties": {
                "name": {"type": "string"}, "content": {"type": "string"},
                "description": {"type": "string"}},
             "required": ["name", "content"]},
            ToolCategory.MEMORY, ToolMutability.MUTATING,
            # Declared, like every other mutating tool. It was the one that
            # had not been, and it is what the central gate is for: until
            # the modes reached the router, a missing declaration cost
            # nothing and so went unnoticed. Nothing changes for it today --
            # its own authorization already refuses in safe mode -- but the
            # refusal is now the mode's, stated before the handler runs.
            required_capabilities=("persistent_memory_write",),
            execution_modes=("ask", "auto"),
            model_visible=False, handler_key="save_skill",
        ), _registered_save_skill),
        (ToolSpec(
            "search_history", "Search conversation history.",
            {"type": "object", "properties": {"query": {"type": "string"}},
             "required": ["query"]}, ToolCategory.RETRIEVAL,
            ToolMutability.READ_ONLY, model_visible=False,
            handler_key="search_history",
        ), _registered_search_history),
        (ToolSpec(
            "web_search", "Compatibility alias for internet search.",
            {"type": "object", "properties": {"query": {"type": "string"}},
             "required": ["query"]}, ToolCategory.WEB,
            ToolMutability.READ_ONLY, model_visible=False,
            handler_key="web_search",
        ), _registered_search_internet),
    )

    for spec, handler in hidden:
        registry.register(spec, handler)

    STANDARD_TOOL_SERVICE.register(registry)

    return registry


TOOL_REGISTRY = build_tool_registry()
TOOL_ROUTER = ToolRouter(TOOL_REGISTRY)

_UNBOUND_TOOL_NAMES = frozenset(
    spec.name for spec in TOOL_REGISTRY.list_specs(model_visible=True)
    if not spec.name.startswith("standard.")
    and spec.name not in CODING_TOOL_NAMES
    and (NETWORK_ENABLED or spec.category != ToolCategory.WEB))
CANONICAL_TOOLS = TOOL_REGISTRY.definitions_for_model(names=_UNBOUND_TOOL_NAMES)
TOOLS = TOOL_REGISTRY.openai_definitions_for_model(names=_UNBOUND_TOOL_NAMES)


def _operator_turns(conversation):
    """The operator's own messages, oldest first, harness nudges excluded."""
    turns = []

    for message in conversation or ():
        if (getattr(message, "role", None) != "user"
                or getattr(message, "authored_by", "operator") != "operator"):
            continue

        text = "\n".join(getattr(block, "text", "") for block in message.content
                         if getattr(block, "text", ""))

        if text:
            turns.append(text)

    return turns


def _request_scope(agent_context):
    """What this turn asked about, for the router to bound writes and reads.

    The previous operator turn is passed along so a bare acceptance ("yes,
    do it") inherits the target the question it accepts had named.
    """
    from context import request_scope

    root = getattr(session_workspace, "PROJECT_ROOT", None)

    # The turn's own request, from the state that outlives compaction. Read
    # from the conversation alone, the scope vanished the first time the
    # context was compacted -- and with it every bound it enforces, in the
    # middle of the turn that needed them.
    state = getattr(agent_context, "working_state", None)
    current = str(getattr(state, "objective", "") or "")
    cached = getattr(agent_context, "_request_scope", None)

    if cached is not None and cached[0] == current:
        return cached[1]

    turns = _operator_turns(getattr(agent_context, "conversation", None))

    if not current and turns:
        current = turns[-1]

    if not current or not root:
        return None

    # The operator turn before this one, for an acceptance to inherit from.
    earlier = [turn for turn in turns
               if request_scope.operator_text(turn).strip()
               != request_scope.operator_text(current).strip()]
    # Only the registered trees that belong to THIS project: a subtree
    # declared inside its root, wherever it resolves to. The workspace's
    # extra roots are every corpus on the machine, and one of them is how a
    # sibling checkout's build.sh passed for this project's own.
    base = os.path.realpath(root)
    extra = tuple(dict.fromkeys(
        path for item in (getattr(session_workspace.WORKSPACE, "extra_roots", ())
                          if session_workspace.WORKSPACE is not None else ())
        if str(item) == base or str(item).startswith(base.rstrip("/") + "/")
        for path in (str(item), os.path.realpath(str(item)))))
    from cli.corpus_registry import load_projects

    try:
        declared = tuple(load_projects())
    except Exception:
        declared = ()

    referents = tuple(name for name in (getattr(session_workspace, "PROJECT", None), *declared)
                      if name)
    scope = request_scope.RequestScope.of(
        current, previous=earlier[-1] if earlier else "",
        root=root, extra_roots=extra, referents=referents)

    try:
        agent_context._request_scope = (current, scope)
    except AttributeError:
        pass

    return scope


def _evidence_in_context(agent_context, tool_call_id):
    """Is the result of that tool call still in the conversation to be sent?

    Compaction rewrites the conversation and the composer decides what goes
    into each request, so an observation the harness once delivered may no
    longer be anywhere the model can read. Telling it "the output is already
    above" in that state is false, and the only move it leaves is another
    spelling of the same read.
    """

    # The conversation keeps every block; compaction and the composer change
    # only what is SENT. A result left out of the last request is gone as far
    # as the model is concerned, whatever the conversation still holds.

    if tool_call_id in (getattr(agent_context, "dropped_tool_results", None) or ()):
        return False

    conversation = getattr(agent_context, "conversation", None) or ()

    for message in conversation:
        for block in getattr(message, "content", ()) or ():
            if (isinstance(block, ToolResultBlock)
                    and block.tool_call_id == tool_call_id
                    and block.content):
                return True

    return False


def _execution_context(agent_context, cache, *, task_id=None, trace=None,
                       tool_call_id=None, cancellation=None, phase_ledger=None):
    """The tool execution context a turn's calls are judged and run in."""
    execution_context = ToolExecutionContext(
        task_id=task_id or new_task_id(),
        trace=trace or TRACE,
        cache=cache,
        result_store=RESULT_STORE,
        cancellation=cancellation or NEVER_CANCELLED,
        checkpoint_manager=(agent_context.checkpoint_manager
                            if agent_context is not None else None),
        checkpoint=(agent_context.checkpoint if agent_context is not None else None),
        role=(getattr(agent_context, "role", "main")
              if agent_context is not None else "main"),
        read_only=bool(getattr(agent_context, "read_only", False)),
        advisory=bool(getattr(agent_context, "advisory", False)),
        scope=(_request_scope(agent_context)
               if agent_context is not None else None),
        # The session's mode, carried to the one place that can hold every
        # tool to its own declaration. Stated here and nowhere else: the
        # mode is the session's, and the router's job is to enforce what the
        # registry declares, not to discover what mode it is in.
        execution_mode=str(session_workspace.EXECUTION_MODE),
        evidence_available=(
            (lambda marker: _evidence_in_context(agent_context, marker))
            if agent_context is not None else None),
        # Asked at the moment of the call, never sampled: the answer changes
        # within the turn, as the standard is read, the sources are opened and
        # a plan is accepted.
        write_gate=(phase_ledger.may_write
                    if phase_ledger is not None and phase_ledger.engaged
                    else None),
        metadata={"standard_binding": getattr(agent_context, "standard_binding", None),
                  "tool_call_id": tool_call_id,
                  "execution_core": getattr(agent_context, "execution_core", "legacy")},
    )

    return execution_context


def route_tool_envelope(
    name, args, cache, *, task_id=None, trace=None, tool_call_id=None,
    cancellation=None, agent_context=None,
):
    # The turn's lifecycle, when it governs this turn. The handler that
    # records a plan reaches it through the cache, and the router reaches the
    # decision it makes through `write_gate` -- one object, two doors, and no
    # way for the two to disagree about the same turn.

    phase_ledger = getattr(agent_context, "work_phase", None)

    if phase_ledger is not None:
        cache[WORK_PHASE] = phase_ledger

    tool_call_id = tool_call_id or new_action_id("tool_call")
    execution_context = _execution_context(
        agent_context, cache, task_id=task_id, trace=trace, tool_call_id=tool_call_id,
        cancellation=cancellation, phase_ledger=phase_ledger)
    execution_context.command_executor = lambda command: _registered_command(
        execution_context, command)
    envelope = TOOL_ROUTER.execute(execution_context, tool_call_id, name, args)

    # A mutation the router refused never reaches the handler, so the handler
    # never files it. The mutation trail is the client's, not the routing
    # layer's, and a denial that leaves no entry in it is a denial nobody
    # reviewing the session can see -- which is what the trail is for.

    if envelope.error_category in _ROUTER_MUTATION_DENIALS:
        audit_denied_mutation(
            name, envelope.error_category.replace("_", " "),
            paths=tuple(str(args.get(key)) for key in ("path", "file")
                        if args.get(key)))

    # ...and say so on screen. Nothing else will: the handler never ran.

    announce_router_refusal(name, envelope)

    if agent_context is not None and envelope.success:
        source_ids = envelope.metadata.get("standard_source_ids", ())

        if (source_ids and isinstance(source_ids, (list, tuple))
                and hasattr(agent_context, "standard_source_ids_used")):
            agent_context.standard_source_ids_used.update(
                item for item in source_ids if isinstance(item, str))

        event_type = {
            "standard_search_completed": SessionEventType.STANDARD_SEARCH_COMPLETED,
            "standard_source_fetched": SessionEventType.STANDARD_SOURCE_FETCHED,
        }.get(envelope.metadata.get("standard_event"))

        if event_type is not None and agent_context.session is not None:
            payload = {
                key: value for key, value in envelope.metadata.items()
                if key in {"standard_binding", "query_sha256", "result_count",
                           "standard_source_ids", "source_id", "citation"}
            }
            payload["result_reference"] = envelope.result_reference

            # Journal the grounded metadata now; the runtime's canonical tool
            # completion boundary will atomically advance the snapshot. Avoid
            # clearing its in-flight marker from inside the handler path.

            agent_context.session.append(event_type, agent_context.task_id, payload)

    return envelope


def execute_tool(name, args, cache, *, agent_context=None):
    """Compatibility facade returning full text; production uses the router."""
    envelope = route_tool_envelope(
        name, args, cache,
        task_id=agent_context.task_id if agent_context is not None else None,
        trace=agent_context.trace if agent_context is not None else None,
        agent_context=agent_context,
    )

    if agent_context is not None:
        AgentRuntime._record_tool_result(agent_context, envelope)

    return envelope.text
