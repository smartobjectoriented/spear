"""One tool call: authorise, run, record.

The order is the boundary's, for every tool alike: the host authorises the
call (hard policy -- read-only turns, target and shell scope, modes); the
tool runs, resolving every path and running every command through the host;
the host is told what happened (audit, evidence). A refusal is returned to
the model as Hermes' tool_error JSON with the host's short reason.
"""

from __future__ import annotations

import json
from typing import Mapping

from . import tools
from .host import Host, ToolRecord


def _refused(name, arguments, call_id, reason):
    return tools.tool_error(reason), ToolRecord(call_id, name, arguments,
                                                tools.tool_error(reason), False,
                                                refused=True)


def _has_error(text: str) -> bool:
    try:
        data = json.loads(text.split("\n\n[Hint:", 1)[0])
    except ValueError:
        return False

    return isinstance(data, dict) and bool(data.get("error"))


def read_file(host: Host, state, call_id, args):
    path = str(args.get("path") or "")
    resolved, refusal = host.resolve_read(path)

    if refusal:
        return _refused("read_file", args, call_id, refusal)

    text = tools.read_file(state, path, resolved, args.get("offset", 1),
                           args.get("limit", tools.DEFAULT_READ_LIMIT))
    ok = not _has_error(text)

    return text, ToolRecord(call_id, "read_file", args, text, ok,
                            read_paths=(resolved,) if ok else ())


def search_files(host: Host, state, call_id, args):
    where = str(args.get("path") or ".") or "."
    resolved, refusal = host.resolve_read(where)

    if refusal:
        return _refused("search_files", args, call_id, refusal)

    text = tools.search(state, host.workspace_root, where, resolved,
                        str(args.get("pattern") or ""),
                        target=args.get("target") or "content",
                        file_glob=args.get("file_glob"),
                        limit=args.get("limit", tools.SEARCH_DEFAULT_LIMIT),
                        offset=args.get("offset", 0),
                        output_mode=args.get("output_mode") or "content",
                        context=args.get("context", 0))

    return text, ToolRecord(call_id, "search_files", args, text, not _has_error(text))


def patch(host: Host, state, call_id, args):
    path = str(args.get("path") or "")
    resolved, refusal = host.resolve_write(path)

    if refusal:
        return _refused("patch", args, call_id, refusal)

    text, changed = tools.patch(
        state, path, resolved, args.get("old_string"), args.get("new_string"),
        bool(args.get("replace_all")),
        write=lambda content: host.write_file(resolved, content, action="patch"))

    return text, ToolRecord(call_id, "patch", args, text, not _has_error(text),
                            changed_paths=(resolved,) if changed else ())


def write_file(host: Host, state, call_id, args):
    path = str(args.get("path") or "")
    resolved, refusal = host.resolve_write(path)

    if refusal:
        return _refused("write_file", args, call_id, refusal)

    text, written = tools.write_file(
        state, path, resolved, args.get("content"),
        write=lambda content: host.write_file(resolved, content, action="write_file"))

    return text, ToolRecord(call_id, "write_file", args, text, written,
                            changed_paths=(resolved,) if written else ())


def delete_file(host: Host, state, call_id, args):
    """SPEAR's: Hermes deletes with `rm`, which SPEAR's sandbox refuses so
    that a deletion is a declared, audited file operation."""
    path = str(args.get("path") or "")
    resolved, refusal = host.resolve_write(path)

    if refusal:
        return _refused("delete_file", args, call_id, refusal)

    failure = host.delete_file(resolved, str(args.get("reason") or ""))

    if failure:
        text = tools.tool_error(failure)
        return text, ToolRecord(call_id, "delete_file", args, text, False)

    tools.invalidate_path(state, resolved)
    text = json.dumps({"success": True, "deleted": path}, ensure_ascii=False)

    return text, ToolRecord(call_id, "delete_file", args, text, True,
                            changed_paths=(resolved,))


def terminal(host: Host, state, call_id, args):
    command = str(args.get("command") or "")
    session = state.setdefault("terminal", {"cwd": None, "env": ""})
    timeout = tools.terminal_timeout(args.get("timeout"))

    if timeout is None:
        text = tools.tool_error(
            f"Foreground timeout {args.get('timeout')}s exceeds the maximum of "
            f"{tools.TERMINAL_MAX_TIMEOUT}s.")
        return text, ToolRecord(call_id, "terminal", args, text, False,
                                command=command)

    outcome = host.run_command(
        command, tools.session_script(command, session["cwd"], session["env"], timeout),
        timeout=timeout + 15, output_chars=tools.MAX_OUTPUT_CHARS + 30_000)
    text, exit_code, timed_out = tools.terminal_from_outcome(
        command, outcome, session, host.workspace_root, timeout)

    return text, ToolRecord(call_id, "terminal", args, text,
                            outcome.status == "ok" and exit_code == 0,
                            refused=outcome.status == "denied", command=command,
                            exit_code=exit_code, timed_out=timed_out)


HANDLERS = {"read_file": read_file, "search_files": search_files, "patch": patch,
            "write_file": write_file, "delete_file": delete_file, "terminal": terminal}


def execute(host: Host, state, call_id: str, name: str,
            arguments: Mapping[str, object], available: tuple[str, ...]):
    """(what the model reads, the record) for one call."""
    if name not in HANDLERS or name not in available:
        # agent/conversation_loop.py's unknown-tool message.
        text = tools.tool_error(f"Tool '{name}' does not exist. Available tools: "
                                f"{', '.join(available)}")
        record = ToolRecord(call_id, name, arguments, text, False, refused=True)
        host.after_tool(record)
        return text, record

    refusal = host.authorize(name, arguments)

    if refusal:
        text, record = _refused(name, arguments, call_id, refusal)
    else:
        text, record = HANDLERS[name](host, state, call_id, dict(arguments))

    host.after_tool(record)

    return text, record
