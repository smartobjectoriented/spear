"""The concrete tool handlers the registry routes to, and their bookkeeping."""

import os
import re
import sys
import shlex
import hashlib
from pathlib import Path
from runtime import progress_monitor, work_phase
from harness import tool_router, web_fetch
from harness.tool_primitives import CommandClassification, ExecutionMode, PathPolicyError
from runtime.tracing import EventStatus, EventType
from harness.tool_router import (
    ToolHandlerResult, ToolResultStatus, invalidates_reads,
)
from cli import session_workspace
from cli.corpus_search import archive_search, search_corpus
from cli.knowledge_commands import propose_knowledge, skill_save
from cli.session_workspace import (
    COMMAND_POLICY, _capture_checkpoint_path, _forget_cached_reads,
    _record_mutation, apply_unified_diff, audit_denied_mutation,
    audit_mutation_result, audit_rejected_mutation, authorize_mutation,
    command_result_text, edit_file, find_file, is_excluded_path,
    preserve_third_party_header, resolve_path, run_cmd_result,
    safe_mode_refusal,
)
from cli.standard_session import standard_url_refusal
from cli.terminal_ui import (
    C_ACCENT, C_BOLD, C_DIM, C_ERR, C_OK, C_RST, C_TOOL, Spinner,
    looks_like_diff, render_diff, show_diff, show_web_sources, tool_result,
    tool_use,
)


def web_search(query, max_results=6):
    """Web search via DuckDuckGo (ddgs package, no API key).
    Returns formatted results (title, URL, snippet)."""

    try:
        from ddgs import DDGS
        results = list(DDGS().text(query, max_results=max_results))
    except Exception as e:
        return f"ERROR web search: {e}"

    if not results:
        return "(no results)"

    out = []

    for i, r in enumerate(results, 1):
        out.append(f"[{i}] {r.get('title', '')}\n"
                   f"    {r.get('href', '')}\n"
                   f"    {r.get('body', '')}")

    return "\n\n".join(out)


# Reserved key in the per-turn cache, holding the set of files whose CONTENT
# the model has actually seen this turn. An object() cannot collide with a
# command string, which is what every other key in that dict is.

READ_PATHS = object()

# Set once the sandbox has been found unavailable this turn. It does not come
# back: bwrap is missing, or the kernel refuses it, and every later command
# fails the same way. What followed was worse than the failure -- the model ran
# eight commands, got eight refusals, then edited a file nine times on the
# strength of retrieved context alone, unable to compile or even read what it
# was changing. An edit that cannot be verified is not worth making.

SANDBOX_DOWN = object()

# The commands the policy refused this turn. Distinct from the result cache,
# because a refusal never ran: replaying it under "already executed" told a
# model its denied `sed -i` had worked.

DENIED_COMMANDS = object()

#: Blocks of source this turn has already been shown, and the mutation
#: generation they were read at. Reading a file in overlapping windows is the
#: single largest cost a governed turn pays: measured across four runs of one
#: two-turn workflow, 54% to 62% of every windowed read landed entirely on
#: lines the turn had already seen. The router's cache cannot catch those --
#: `sed -n '590,720p'` and `sed -n '600,700p'` are different arguments and the
#: same evidence.
REGIONS_READ = object()


def _already_in_evidence(context, command):
    """The note to return instead of re-reading lines already shown.

    Empty when the command is not a windowed read, when any of its blocks are
    new, or when something has been written since they were read -- a file
    that changed is a file worth reading again, and that is the whole reason
    the generation is part of the key.
    """
    blocks = progress_monitor.read_evidence(command)

    if not blocks:
        return ""

    seen = context.cache.setdefault(REGIONS_READ, {})
    generation = _mutation_generation(context)

    # Each block remembers the call that showed it, so "already above" is
    # said only while that result is still in what the model is sent. A
    # compaction that folded it away makes the block new again.

    def still_shown(block):
        record = seen.get(block)

        if not record or record[0] != generation:
            return False

        shown_by = record[1]
        available = getattr(context, "evidence_available", None)

        if shown_by is None or available is None:
            return True

        try:
            return bool(available(shown_by))
        except Exception:
            return True

    fresh = [block for block in blocks if not still_shown(block)]

    if fresh:
        shown_by = (getattr(context, "metadata", None) or {}).get("tool_call_id")

        for block in blocks:
            seen[block] = (generation, shown_by)

        return ""

    where = sorted({block.split("#", 1)[0] for block in blocks})

    return ("(ALREADY IN EVIDENCE this turn — these lines of "
            + ", ".join(where)
            + " are already above in this conversation, and nothing has been "
              "written since. Read them there. Widen the range, open a "
              "different file, or move on.)\n")


def _mutation_generation(context):
    """The router's own write counter, which is what makes a read stale."""
    ledger = context.cache.get(tool_router._REPEAT_KEY)

    return ledger.get("generation", 0) if isinstance(ledger, dict) else 0
SANDBOX_DOWN_MSG = (
    "REFUSED: the sandbox is unavailable, so nothing you write can be read "
    "back, compiled or run. Editing from retrieved context alone produces "
    "changes nobody can check. Stop and report that the sandbox is down — "
    "that IS the answer to give the user.")

# Commands that put a file's content in front of the model. `ls` and `stat`
# are deliberately absent: knowing a file exists is not knowing what is in it,
# and that is the whole point of the guard below.

CONTENT_READING_BINS = {"cat", "sed", "head", "tail", "grep", "egrep", "fgrep",
                        "nl", "awk", "less", "more", "strings", "diff", "rg"}


def _note_files_read(cache, cmd):
    """Record the files a bash command showed the model."""

    try:
        argv = shlex.split(cmd)
    except ValueError:
        return

    if not argv or os.path.basename(argv[0]) not in CONTENT_READING_BINS:
        return

    seen = cache.setdefault(READ_PATHS, set())

    for tok in argv[1:]:
        if tok.startswith("-"):
            continue

        try:
            fp = resolve_path(tok)
        except Exception:
            continue

        if os.path.isfile(fp):
            seen.add(os.path.realpath(fp))


def _was_read_this_turn(cache, fpath):
    return os.path.realpath(fpath) in cache.get(READ_PATHS, set())


def _classified_handler_result(
    text, *, mutation=False, affected_paths=(), read_paths=(), metadata=None,
    status=None, exit_code=None,
):
    # A JSON result says how it went in its own fields, so the caller that
    # built it states the status rather than this function guessing it from
    # the first word of the text.
    if status is not None:
        failed = status not in {ToolResultStatus.OK, ToolResultStatus.CACHED}

        return ToolHandlerResult(
            text=text, status=status, exit_code=exit_code,
            metadata=metadata or {}, mutation=mutation and not failed,
            affected_paths=tuple(affected_paths) if not failed else (),
            read_paths=tuple(read_paths) if not failed else (),
            error_category=status.value if failed else None,
            error_summary=text.splitlines()[0][:240] if failed else None,
        )

    exit_match = re.search(r"\(exit\s+(-?\d+)\)", text)
    exit_code = int(exit_match.group(1)) if exit_match else None

    if text.startswith("CANCELLED"):
        status = ToolResultStatus.CANCELLED
    elif text.startswith("ERROR"):
        status = (ToolResultStatus.TIMEOUT if "timed out" in text.lower()
                  else ToolResultStatus.FAILED)
    elif exit_code not in (None, 0):
        status = ToolResultStatus.FAILED
    elif text.startswith("(ALREADY EXECUTED"):
        status = ToolResultStatus.CACHED
    else:
        status = ToolResultStatus.OK

    failed = status not in {ToolResultStatus.OK, ToolResultStatus.CACHED}

    return ToolHandlerResult(
        text=text,
        status=status,
        exit_code=exit_code,
        metadata=metadata or {},
        mutation=mutation and not failed,
        affected_paths=tuple(affected_paths) if not failed else (),
        read_paths=tuple(read_paths) if not failed else (),
        error_category=status.value if failed else None,
        error_summary=text.splitlines()[0][:240] if failed else None,
    )


def _workspace_label(path):
    try:
        return str(session_workspace.WORKSPACE.relative(path)
                   if session_workspace.WORKSPACE is not None else path)
    except (PathPolicyError, OSError, TypeError):
        return "[workspace-path]"


def _registered_command(context, command):
    """The only registered command boundary; run_cmd retains all enforcement."""
    cmd = command.strip()
    print()
    tool_use("Bash", cmd)

    if cmd in context.cache:
        # A refusal is not an execution. Replaying one as "already executed"
        # is how a model concluded its denied `sed -i` had worked and moved on
        # believing the file was edited.

        refused = cmd in context.cache.get(DENIED_COMMANDS, ())
        tool_result("(refused before — same refusal)" if refused
                    else "(cached — already executed this turn)")
        print()

        if refused:
            # A replayed refusal is a refusal, not a success. Classified as
            # OK it cleared the router's failure tally for that command, so
            # the same refused command could be asked for indefinitely.

            return ToolHandlerResult(
                "REFUSED ALREADY this turn — it did not run, and repeating it "
                "will not change that. Take the other route named below.\n"
                + context.cache[cmd][:1500],
                ToolResultStatus.DENIED,
                error_category="repeated_denied_command",
                error_summary="command already refused this turn",
            )

        # The output is not repeated while the model can still see it. The
        # router decides that -- it knows which result block this command
        # produced and asks the caller whether that block survived compaction
        # -- and it puts the original content back when the answer is no. So
        # this pointer is only ever delivered alongside evidence that is
        # genuinely still in the conversation.

        return _classified_handler_result(
            "(ALREADY EXECUTED this turn — its output is already above in this "
            "conversation. Read it there. DO NOT run this command again.)\n"
        )

    # Lines this turn has already been shown, in whatever window it asked for
    # them. Checked before the command runs, because the cost being saved is
    # the round, not the disk.

    settled = _already_in_evidence(context, cmd)

    if settled:
        tool_result(settled.splitlines()[0])
        print()

        return _classified_handler_result(settled)

    reads_before = set(context.cache.get(READ_PATHS, set()))
    command_result = run_cmd_result(
        cmd, need_confirm=False, cancellation=context.cancellation,
        # The read-only roles have always run this way. A turn the USER told
        # not to change anything now does too: dropping the write tools from
        # the view is not enough on its own, because bash can write with
        # `sed -i`, `cp`, or a redirection. SAFE takes workspace:write away
        # for the whole turn, and the sandbox mounts the tree read-only.
        # ...and a turn whose write gate has not opened yet runs the same
        # way. Dropping edit_file from the model's reach is not enough on its
        # own: a turn refused an edit reached for `sed -i` within two rounds,
        # which is the same write through a different door. SAFE closes the
        # door rather than the doorway.
        execution_mode=(ExecutionMode.SAFE
                        if context.read_only
                        or _write_gate_closed(context)
                        or context.role in {"explorer", "reviewer", "planning"}
                        else None),
    )

    result = command_result_text(command_result)

    # The command guessed does not exist. The next guess is how a turn ended
    # up searching the home directory for somebody else's build.sh: say where
    # this project keeps its own, and what was found there.

    if getattr(context, "scope", None) is not None:
        from context import request_scope

        result += request_scope.missing_command_hint(
            cmd, (command_result.stdout or "") + (command_result.stderr or ""),
            command_result.exit_code, context.scope)

    # A command that reports the sandbox missing is the only way this is
    # learned; every mutation after it in the same turn is refused.

    if "sandbox unavailable" in result or "bubblewrap sandbox" in result:
        context.cache[SANDBOX_DOWN] = True

    # A refusal that is about SCOPE, not about permissions. The turn was told
    # not to change anything; saying only "command denied" invites the next
    # spelling of the same intent, and a run spent thirty-five steps finding
    # them -- `sed -i`, a redirection, python. Say what the refusal is for,
    # and what to do instead.

    # Same shape, different reason. A command refused because the gate is
    # still shut is refused temporarily, and saying only "denied" invites the
    # next spelling of the same write instead of the investigation that would
    # open it.

    if (not context.read_only and command_result.status == "denied"
            and _write_gate_closed(context)
            and COMMAND_POLICY.classify(cmd).classification
            != CommandClassification.READ_ONLY):
        gate = context.write_gate("")
        result = result.rstrip() + "\n" + getattr(gate, "message", "")
        context.trace.emit(
            EventType.TOOL_CALL_FAILED, context.task_id,
            status=EventStatus.DETECTED, tool_name="bash",
            action_id=context.action_id,
            metadata={"reason": getattr(gate, "reason", "investigation_incomplete"),
                      "arguments_recorded": False},
        )

    if (context.read_only and command_result.status == "denied"
            and COMMAND_POLICY.classify(cmd).classification
            != CommandClassification.READ_ONLY):
        violations = context.cache.get(READ_ONLY_VIOLATIONS, 0) + 1
        context.cache[READ_ONLY_VIOLATIONS] = violations
        result = result.rstrip() + "\n" + (
            ADVISORY_REFUSAL if getattr(context, "advisory", False)
            else READ_ONLY_REFUSAL)
        context.trace.emit(
            EventType.READ_ONLY_VIOLATION, context.task_id,
            status=EventStatus.DETECTED, tool_name="bash",
            action_id=context.action_id,
            metadata={"attempt": violations,
                      "classification": COMMAND_POLICY.classify(cmd).classification.value,
                      "arguments_recorded": False},
        )

    tool_result("interrupted" if result.startswith("CANCELLED") else result)
    print()
    context.cache[cmd] = result

    if command_result.status == "denied":
        context.cache.setdefault(DENIED_COMMANDS, set()).add(cmd)

    if not result.startswith(("ERROR", "CANCELLED")):
        _note_files_read(context.cache, cmd)

    reads_after = set(context.cache.get(READ_PATHS, set()))
    status = {
        "ok": ToolResultStatus.OK,
        "cancelled": ToolResultStatus.CANCELLED,
        "denied": ToolResultStatus.DENIED,
        "timeout": ToolResultStatus.TIMEOUT,
    }.get(command_result.status, ToolResultStatus.FAILED)

    # A shell command that is not read-only may have written any file in the
    # workspace, and the mutations that mangled user_space.rst were exactly
    # that -- `head ... > /tmp/x && cat /tmp/x > target`, not an edit tool. So
    # the invalidation cannot live only in the mutation handlers. The command
    # itself stays cached: re-running a write is worse than re-reading.
    #
    # Nothing here measures what a command wrote -- the sandbox reports no
    # such thing -- so a non-read-only command that RAN is declared
    # potentially mutating and treated as though it had written, whatever its
    # exit code. A read-only one is measured: the classifier is the
    # measurement, and its count is a truthful zero.
    #
    # The decision itself is `invalidates_reads`, which the router also uses
    # for its generation counter. One function, one answer: there is no state
    # where this cache considers its reads stale and the router considers
    # them current.

    mutating = (COMMAND_POLICY.classify(cmd).classification
                != CommandClassification.READ_ONLY)
    metadata = {
        "execution_status": command_result.status,
        **({"potentially_mutating": True} if mutating else {"mutation_count": 0}),
        **({"read_only_violations": context.cache[READ_ONLY_VIOLATIONS]}
           if context.cache.get(READ_ONLY_VIOLATIONS) else {}),
    }

    if invalidates_reads(status, False, metadata):
        _forget_cached_reads(context, keep=cmd)

    return ToolHandlerResult(
        text=result,
        status=status,
        exit_code=command_result.exit_code,
        stdout=command_result.stdout,
        stderr=command_result.stderr,
        metadata=metadata,
        read_paths=tuple(_workspace_label(path)
                         for path in sorted(reads_after - reads_before)),
        error_category=None if status == ToolResultStatus.OK else status.value,
        error_summary=None if status == ToolResultStatus.OK else command_result.summary,
    )


def _sandbox_down_result(label, path):
    """Refuse a mutation while the sandbox is down, in the router's shape.

    Ported from the monolithic execute_tool the registry replaced: without the
    sandbox nothing written can be read back, compiled or run, so an edit made
    from retrieved context alone is a change nobody can check. DENIED and not
    FAILED -- the tool did not break, it declined.
    """
    tool_use(label, path, color=C_ERR)
    tool_result(SANDBOX_DOWN_MSG)
    print()

    return ToolHandlerResult(
        text=SANDBOX_DOWN_MSG,
        status=ToolResultStatus.DENIED,
        error_category=ToolResultStatus.DENIED.value,
        error_summary="sandbox unavailable",
    )


def _registered_edit_file(context, args):
    path = (args.get("path") or "").strip()

    if context.cache.get(SANDBOX_DOWN):
        return _sandbox_down_result("Update", path)

    old_text = args.get("old_text") or ""
    new_text = args.get("new_text") or ""
    diff = next((text for text in (new_text, args.get("content") or "")
                 if looks_like_diff(text)), "")

    refused = safe_mode_refusal("edit_file")

    if refused:
        audit_denied_mutation("edit_file", "safe mode", paths=(path,))
        tool_result(refused)
        print()

        return _classified_handler_result(refused)

    print()
    tool_use("Update", path, color=C_ACCENT)

    if diff and not old_text:
        render_diff(diff)

        try:
            fpath = resolve_path(path)
        except PathPolicyError as exc:
            result = f"ERROR: {exc}"
            audit_rejected_mutation("apply_patch", str(exc), paths=(path,))
        else:
            blocked = authorize_mutation(
                f"Apply patch to {C_BOLD}{path}{C_RST} ?",
                action="apply_patch", paths=(fpath,),
            )

            if blocked is None:
                _capture_checkpoint_path(context, fpath)
                result = apply_unified_diff(fpath, diff)

                if result.startswith("OK"):
                    _record_mutation(context, fpath)

                audit_mutation_result("apply_patch", result, paths=(fpath,))
            else:
                result = blocked.to_legacy_text()
    else:
        show_diff(old_text, new_text)
        result = edit_file(path, old_text, new_text, execution_context=context)

    tool_result(result)
    print()
    affected = (_workspace_label(resolve_path(path)),) if result.startswith("OK") else ()

    return _classified_handler_result(result, mutation=True, affected_paths=affected)


def _registered_append_file(context, args):
    path = (args.get("path") or "").strip()

    if context.cache.get(SANDBOX_DOWN):
        return _sandbox_down_result("Append", path)

    content = (args.get("content") or "").strip("\n")

    refused = safe_mode_refusal("append_file")

    if refused:
        audit_denied_mutation("append_file", "safe mode", paths=(path,))
        tool_result(refused)
        print()

        return _classified_handler_result(refused)

    print()
    tool_use("Append", path, color=C_ACCENT)

    for index, line in enumerate(content.split("\n")[:8]):
        prefix = "⎿  " if index == 0 else "   "
        print(f"  {C_DIM}{prefix}{C_RST}{C_OK}+ {line[:150]}{C_RST}")

    try:
        resolve_path(path)
    except PathPolicyError as exc:
        result = f"ERROR: {exc}"
        audit_rejected_mutation("append_file", str(exc), paths=(path,))
        tool_result(result)
        print()

        return _classified_handler_result(result)

    fpath = find_file(path)

    if not fpath:
        result = f"ERROR: file not found: {path}"
    elif (blocked := authorize_mutation(
            f"Append to {C_BOLD}{path}{C_RST} ?", action="append_file",
            paths=(fpath,))) is None:
        _capture_checkpoint_path(context, fpath)

        with open(fpath, "a", encoding="utf-8") as stream:
            stream.write("\n" + content + "\n")

        result = f"OK: appended to {path}"
        _record_mutation(context, fpath)
        audit_mutation_result("append_file", result, paths=(fpath,))
    else:
        result = blocked.to_legacy_text()

    tool_result(result)
    print()
    affected = (_workspace_label(fpath),) if fpath and result.startswith("OK") else ()

    return _classified_handler_result(result, mutation=True, affected_paths=affected)


def named_link(path):
    """The symbolic link `path` names, inside the workspace, or None.

    Resolving a path follows its links, so deleting a link removed what it
    pointed to: a link to a tracked source took the source with it. The
    directory holding the link is what the workspace policy judges; the link
    itself is what goes.
    """
    path = str(path).strip().strip("'\"")

    if not path:
        return None

    candidate = (Path(path) if os.path.isabs(path)
                 else Path(session_workspace.WORKSPACE.root) / path)
    named = Path(resolve_path(str(candidate.parent))) / candidate.name

    return named if named.is_symlink() else None


def _registered_delete_file(context, args):
    """Remove a file, having first captured it so /undo can bring it back.

    Deleting was the one file operation nothing could do. A model that had
    made a chapter redundant tried rm, `bash -c rm`, python3 -c os.remove and
    an empty write_file -- fifteen calls -- then emptied the file with
    edit_file, leaving a zero-byte document that Sphinx still complained
    about. The gap was real; this closes it, under the same checkpoint every
    other mutation gets.
    """
    path = (args.get("path") or "").strip()

    if context.cache.get(SANDBOX_DOWN):
        return _sandbox_down_result("Delete", path)

    reason = (args.get("reason") or "").strip()
    print()
    tool_use("Delete", path, color=C_ERR)

    if reason:
        print(f"  {C_DIM}⎿  {reason[:150]}{C_RST}")

    try:
        fpath = named_link(path) or resolve_path(path)
    except PathPolicyError as exc:
        result = f"ERROR: {exc}"
        audit_rejected_mutation("delete_file", str(exc), paths=(path,))
        tool_result(result)
        print()

        return _classified_handler_result(result)

    if not os.path.isfile(fpath) and not os.path.islink(fpath):
        result = (f"ERROR: {path} is not an existing regular file. Nothing "
                  f"was deleted." if not os.path.isdir(fpath) else
                  f"ERROR: {path} is a directory. delete_file removes one "
                  f"file at a time.")
        tool_result(result)
        print()

        return _classified_handler_result(result)

    if is_excluded_path(fpath):
        result = (f"ERROR: {path} is in a third-party or snapshot tree that is "
                  f"never modified from here.")
        audit_rejected_mutation("delete_file", "excluded tree", paths=(fpath,))
        tool_result(result)
        print()

        return _classified_handler_result(result)

    blocked = authorize_mutation(
        f"Delete {C_BOLD}{path}{C_RST} ?", action="delete_file", paths=(fpath,))

    if blocked is None:
        # Capture BEFORE unlinking: the checkpoint holds the only remaining
        # copy of the bytes, and it is what /undo restores from. A link has no
        # bytes of its own, and the checkpoint refuses links: what it pointed
        # at goes into the audit record instead, which is all a link is.

        link = os.readlink(fpath) if os.path.islink(fpath) else None

        if link is None:
            _capture_checkpoint_path(context, fpath)

        try:
            os.remove(fpath)
            result = (f"OK: deleted {path}" if link is None
                      else f"OK: deleted the link {path} (it pointed to {link})")

            if link is None:
                _record_mutation(context, fpath)
            else:
                _forget_cached_reads(context)
        except OSError as exc:
            result = f"ERROR: could not delete {path}: {exc.strerror}"

        audit_mutation_result("delete_file", result, paths=(fpath,))
    else:
        result = blocked.to_legacy_text()

    tool_result(result)
    print()
    affected = (_workspace_label(fpath),) if result.startswith("OK") else ()

    return _classified_handler_result(result, mutation=True, affected_paths=affected)


def _registered_write_file(context, args):
    path = (args.get("path") or "").strip()

    if context.cache.get(SANDBOX_DOWN):
        return _sandbox_down_result("Write", path)

    content = (args.get("content") or "").strip("\n")

    if not path:
        result = ("ERROR: no path given. Pass the file to write, e.g. "
                  "write_file(path='so3/usr/src/ping.c', content=...).")
        tool_result(result)
        print()

        return _classified_handler_result(result)

    try:
        fpath = resolve_path(path)
    except PathPolicyError as exc:
        result = f"ERROR: {exc}"
        audit_rejected_mutation("write_file", str(exc), paths=(path,))
        tool_result(result)
        print()

        return _classified_handler_result(result)

    if os.path.isdir(fpath):
        result = f"ERROR: '{path}' is a directory, not a file. Give the full file path."
        tool_result(result)
        print()

        return _classified_handler_result(result)

    # The mode, before any check on the CONTENT. A malformed or escaping
    # path is still reported as itself above -- that is a fact about the
    # request, not about permission -- but "this file is larger than your
    # proposed content, use edit_file" answers a question no one in a
    # read-only session can reach, and names a tool that is equally refused.

    refused = safe_mode_refusal("write_file")

    if refused:
        audit_denied_mutation("write_file", "safe mode", paths=(fpath,))
        tool_result(refused)
        print()

        return _classified_handler_result(refused)

    if looks_like_diff(content) and os.path.isfile(fpath):
        print()
        tool_use("Update", path, color=C_ACCENT)
        render_diff(content)
        blocked = authorize_mutation(
            f"Apply patch to {C_BOLD}{path}{C_RST} ?",
            action="apply_patch", paths=(fpath,),
        )

        if blocked is None:
            _capture_checkpoint_path(context, fpath)
            result = apply_unified_diff(fpath, content)

            if result.startswith("OK"):
                _record_mutation(context, fpath)

            audit_mutation_result("apply_patch", result, paths=(fpath,))
        else:
            result = blocked.to_legacy_text()

        tool_result(result)
        print()
        affected = (_workspace_label(fpath),) if result.startswith("OK") else ()

        return _classified_handler_result(result, mutation=True, affected_paths=affected)

    exists = os.path.isfile(fpath)

    if exists:
        try:
            with open(fpath, "r", encoding="utf-8") as stream:
                content = preserve_third_party_header(stream.read(), content)
        except OSError:
            pass

    print()
    tool_use("Write", path, color=C_ACCENT)
    lines = content.count("\n") + 1

    for index, line in enumerate(content.split("\n")[:8]):
        prefix = "⎿  " if index == 0 else "   "
        print(f"  {C_DIM}{prefix}{C_RST}{C_OK}+ {line[:150]}{C_RST}")

    if lines > 8:
        print(f"  {C_DIM}   … +{lines - 8} lines{C_RST}")

    verb = "Overwrite" if exists else "Create"

    if exists and not _was_read_this_turn(context.cache, fpath):
        result = (f"ERROR: {path} already exists and you have not read it this "
                  f"turn. Overwrite refused. Read it first (bash: cat {path}), "
                  "or use edit_file for a targeted change — that is preferable "
                  "for an existing file.")
        audit_rejected_mutation(
            "write_file", "file not read this turn", paths=(fpath,))
        tool_result(result)
        print()

        return _classified_handler_result(result)

    if exists and os.path.getsize(fpath) > 2 * len(content):
        # An empty write is a delete in disguise, and it does not work: a model
        # that could not remove a redundant chapter emptied it instead, leaving
        # a zero-byte file that still warned "isn't included in any toctree".

        result = (f"ERROR: {path} exists and is much larger than your proposed "
                  f"content ({os.path.getsize(fpath)} bytes vs {len(content)}). "
                  "Overwrite refused — use edit_file for targeted changes or "
                  "append_file for additions."
                  + (" To DELETE it: no tool can, and emptying it leaves the "
                     "file behind. Say which file should go and why, and leave "
                     "it to the operator." if not content.strip() else ""))
        tool_result(result)
        print()

        return _classified_handler_result(result)

    blocked = authorize_mutation(
        f"{verb} {C_BOLD}{path}{C_RST} ?", action="write_file", paths=(fpath,))

    if blocked is None:
        try:
            _capture_checkpoint_path(context, fpath)
            os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)

            with open(fpath, "w", encoding="utf-8") as stream:
                stream.write(content + "\n")

            result = f"OK: {path} written ({lines} lines)"
            _record_mutation(context, fpath)
        except OSError as exc:
            result = f"ERROR: could not write {path}: {exc}"

        audit_mutation_result("write_file", result, paths=(fpath,))
    else:
        result = blocked.to_legacy_text()

    tool_result(result)
    print()
    affected = (_workspace_label(fpath),) if result.startswith("OK") else ()
    metadata = {"created_paths": affected} if not exists and affected else {}

    return _classified_handler_result(
        result, mutation=True, affected_paths=affected, metadata=metadata,
    )


def _registered_remember(context, args):
    note = (args.get("note") or "").strip()
    print()
    tool_use("Remember", note, color=C_ACCENT)

    if not note:
        result = "ERROR: empty note"
    elif (blocked := authorize_mutation(
            "Propose this as workspace knowledge?", action="remember")) is None:
        result = propose_knowledge(note)
        audit_mutation_result("remember", result)
    else:
        result = blocked.to_legacy_text()

    tool_result(result)
    print()

    return _classified_handler_result(result, mutation=True)


def _registered_save_skill(context, args):
    name = (args.get("name") or "").strip()
    content = (args.get("content") or "")[:2500]

    # One line saying what the procedure is for, kept out of the body and
    # embedded with it: a query resembles the purpose more than the steps.

    description = (args.get("description") or "").strip().replace("\n", " ")[:200]
    print()
    tool_use("Skill", name, color=C_ACCENT)

    for index, line in enumerate(content.split("\n")[:6]):
        prefix = "⎿  " if index == 0 else "   "
        print(f"  {C_DIM}{prefix}{line[:150]}{C_RST}")

    blocked = authorize_mutation(
        f"Save skill {C_BOLD}{name}{C_RST} to the library?", action="save_skill")

    if blocked is None:
        result = skill_save(name, content, description=description)
        audit_mutation_result("save_skill", result)
    else:
        result = blocked.to_legacy_text()

    tool_result(result)
    print()

    return _classified_handler_result(result, mutation=True)


def _registered_search_history(context, args):
    query = (args.get("query") or "").strip()
    print()
    tool_use("History", query, color=C_TOOL)
    result = archive_search(query)
    tool_result(result, max_lines=6)
    print()

    return _classified_handler_result(result)


#: Where the turn's lifecycle sits inside the per-call cache. A sentinel, for
#: the same reason the repeat ledger uses one: the cache's string keys are
#: cleared when a write makes the turn's reads stale, and the phase a turn has
#: reached is not made stale by writing.
WORK_PHASE = object()


def _write_gate_closed(context):
    """Is the turn's lifecycle still holding the writes back?

    False whenever there is no gate at all, which is every turn that is not
    changing code to satisfy an authoritative source.
    """
    gate = getattr(context, "write_gate", None)

    if gate is None:
        return False

    # A shell command names no single file the gate could check, so it asks
    # the general question: is there any planned change ready to be made?
    decision = gate("")

    return decision is not None and not getattr(decision, "allowed", True)


def _registered_plan_change(context, args):
    """Record one planned change, and say plainly what was wrong with it.

    The ledger judges the entry against what the turn actually retrieved and
    read; that verdict goes straight back as the tool result. An item rejected
    in silence is an item the model will submit again in the same words.
    """
    ledger = context.cache.get(WORK_PHASE)
    item = {name: args.get(name) for name in work_phase.GapItem.FIELDS}
    item["disposition"] = args.get("disposition")

    print()
    tool_use("Plan", str(item.get("requirement") or "")[:60], color=C_TOOL)

    if ledger is None or not getattr(ledger, "engaged", False):
        # No lifecycle on this turn: nothing gates the writes, so recording a
        # plan changes nothing. Say so rather than pretending it was filed.
        tool_result("(no investigation gate on this turn)")
        print()

        return _classified_handler_result(
            "This turn has no investigation gate — the plan was not recorded "
            "and nothing was waiting for it. Proceed with the change.")

    outcome = ledger.record_plan(
        [item],
        supersedes=str(args.get("supersedes") or "").strip(),
        reason=str(args.get("reason") or "").strip(),
        new_evidence=str(args.get("new_evidence") or "").strip())

    outstanding = ledger.uncovered_requirements()

    if outcome.any_accepted and not outstanding:
        lines = [f"Recorded. {len(ledger.items)} planned change(s) now stand.",
                 "The write gate is open. Make this change, then run the "
                 "project's own build and tests."]
    elif outcome.any_accepted:
        lines = [f"Recorded. {len(ledger.items)} planned change(s) now stand.",
                 f"{len(outstanding)} carried requirement(s) still have no "
                 f"disposition, so the write gate stays shut: "
                 + ", ".join(found.key for found in outstanding[:8])]
    else:
        lines = ["Not recorded.", outcome.report(), work_phase.WRITE_BLOCKED]

    text = "\n".join(line for line in lines if line)
    tool_result(text.splitlines()[0])
    print()

    return _classified_handler_result(text)


def _registered_search_corpus(context, args):
    query = (args.get("query") or "").strip()
    print()
    tool_use("Search", query, color=C_TOOL)

    # Cached for the turn, exactly as bash is. The corpus does not change
    # mid-turn, so the same query gives the same passages; a session was
    # observed spending a round asking the same question twice and getting the
    # same file back. Saying "already searched" is what stops the third.

    key = ("search_corpus", query)

    if key in context.cache:
        tool_result("(cached — already searched this turn)")
        print()

        return _classified_handler_result(
            "(ALREADY SEARCHED this turn — same result repeated below. Use it, "
            "or search something else.)\n" + context.cache[key])

    with Spinner("Searching the corpus…"):
        result = search_corpus(query)

    context.cache[key] = result
    tool_result(result.split("\n")[0][:150] if result else "(no match)")
    print()

    return _classified_handler_result(result)


def _offline_refusal(tool):
    return (f"ERROR {tool}: the session was launched with --no-network. "
            f"Nothing here reaches the internet — say so instead of looking "
            f"for another route.")


def _registered_search_internet(context, args):
    if not NETWORK_ENABLED:
        return _classified_handler_result(_offline_refusal("search_internet"))

    query = (args.get("query") or "").strip()
    print()
    tool_use("Web", query, color=C_TOOL)

    with Spinner("Searching the web…"):
        result = web_search(query)

    show_web_sources(result)
    print()

    return _classified_handler_result(result)


def _fetch_url_to_disk(context, url, save_as):
    """Put the document on disk. A download, not a read.

    Reading a specification into the context window is not the same act as
    having the file: asked for the RS274/NGC PDF so it could be ingested as a
    standard, the assistant fetched it, was handed 60 of 121 pages of TEXT, and
    finished by telling the user to download it in a browser — because nothing
    it could call wrote a file. /standard ingest needs a path.

    Writing is a mutation, so it goes through the same authorization as
    write_file: refused in safe mode, confirmed in ask mode, and audited.
    """
    try:
        fpath = resolve_path(save_as)
    except PathPolicyError as exc:
        audit_rejected_mutation("fetch_url", str(exc), paths=(save_as,))

        return f"ERROR fetch_url: {exc}"

    if os.path.isdir(fpath):
        return f"ERROR fetch_url: '{save_as}' is a directory."

    with Spinner("Downloading…"):
        data, content_type, final_url = web_fetch.download(url)

    blocked = authorize_mutation(
        f"Save {C_BOLD}{final_url}{C_RST} ({len(data) // 1024} KB, "
        f"{content_type}) to {C_BOLD}{save_as}{C_RST} ?",
        action="fetch_url", paths=(fpath,),
    )

    if blocked is not None:
        return blocked.to_legacy_text()

    _capture_checkpoint_path(context, fpath)
    os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)

    with open(fpath, "wb") as handle:
        handle.write(data)

    _record_mutation(context, fpath)
    digest = hashlib.sha256(data).hexdigest()
    result = (f"OK: saved {len(data)} bytes to {save_as}\n"
              f"  {content_type} from {final_url}\n"
              f"  sha256 {digest}")
    audit_mutation_result("fetch_url", result, paths=(fpath,))

    return result


def _registered_fetch_url(context, args):
    if not NETWORK_ENABLED:
        return _classified_handler_result(_offline_refusal("fetch_url"))

    url = (args.get("url") or "").strip()
    save_as = (args.get("save_as") or "").strip()
    pages = (args.get("pages") or "").strip()

    # The bound standard has one canonical source, with a sha256 the binding
    # names, and four tools that serve it. A turn that fetched the web for it
    # instead read whatever a public page happened to say and edited five
    # source files against that -- so a URL naming the bound standard is
    # refused here, with the tools that do have it.

    refusal = standard_url_refusal(url)

    if refusal:
        tool_use("Web", url, color=C_TOOL)
        tool_result(refusal, max_lines=4)
        print()

        return _classified_handler_result(refusal)

    print()
    tool_use("Web", f"{url} → {save_as}" if save_as else url, color=C_TOOL)
    mutation, affected = False, ()

    try:
        if save_as:
            result = _fetch_url_to_disk(context, url, save_as)
            mutation = result.startswith("OK:")
            affected = ((_workspace_label(resolve_path(save_as)),) if mutation
                        else ())
        else:
            with Spinner("Reading the page…"):
                result = web_fetch.fetch(url, pages=pages or None).render()
    except web_fetch.FetchRefused as exc:
        # A refusal the model can act on: it says which boundary was hit, and
        # every one of them names what to do instead. Returned as a result and
        # not raised, so the turn continues.

        result = f"ERROR fetch_url: {exc}"

    tool_result(result, max_lines=4)
    print()

    return _classified_handler_result(result, mutation=mutation,
                                      affected_paths=affected)

# --no-network is a promise about the SESSION, not about the shell alone.
# search_internet and fetch_url reach the internet from this very process,
# outside the capability policy entirely, so a flag that only emptied
# `network` out of the execution modes would have left the two tools that
# actually browse working — the flag would have read as offline and not been.
NETWORK_ENABLED = "--no-network" not in sys.argv[1:]


#: Counted by CATEGORY, not by signature: `sed -i`, a redirection, a python
#: one-liner and the next idea are the same violation of the same scope, and
#: counting them separately is how a turn gets four tries at it.

READ_ONLY_VIOLATIONS = object()
READ_ONLY_REFUSAL = (
    "The user explicitly requested a read-only task. Do not try another way "
    "to modify or test the file. Answer the original question using the "
    "evidence already collected."
)

ADVISORY_REFUSAL = (
    "The user asked whether or how this could change, not for the change: "
    "this turn is read-only. Do not try another way to modify the tree. "
    "Answer with what would have to change, in which files, and why."
)
