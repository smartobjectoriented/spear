"""read_file, search_files, patch, write_file and terminal: Hermes' behaviour.

A function-by-function port of Hermes Agent's file and terminal tools
(tools/file_tools.py, tools/file_operations.py, tools/terminal_tool.py at
0cbc6e37, MIT, Copyright (c) 2025 Nous Research; see THIRD_PARTY_NOTICES.md).
The pure helpers are copied verbatim into hermes_file_helpers, exit_codes,
terminal_hints, ansi_strip and binary_extensions; this module reproduces the
tool functions themselves -- the same constants, the same result fields, the
same messages, in the same order of checks.

Deviations, each because SPEAR's architecture differs, not because the
behaviour was judged better:

- I/O is in-process Python. Hermes routes file I/O through its terminal
  backend (`sed -n | cut`, `wc -l`, `cat`); SPEAR has no such backend, and
  its own command boundary must not see harness reads as model commands.
- Paths arrive already resolved and authorised by SPEAR's workspace policy.
  Hermes' sensitive-path, cross-profile, protected-instruction and approval
  checks are replaced by that policy (refused before this module is called).
- Structured-document extraction (docx, pdf, ...), images and vision hints,
  LSP diagnostics, shell linters (py_compile, node, tsc, go vet, rustfmt),
  V4A multi-file patches (offered only to OpenAI-family models), secret
  redaction of read output and the macOS TCC exclusions are not carried.
- The terminal keeps its session state in SPEAR's turn cache instead of a
  snapshot file, because every SPEAR command runs in a fresh sandbox.

Each tool returns the JSON string the model reads. State that Hermes keeps
per task (`_read_tracker`, patch failure counts) lives in a dict the caller
owns -- one per turn.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import stat
import subprocess

from .hermes.ansi_strip import strip_ansi
from .hermes.binary_extensions import has_binary_extension
from .hermes.exit_codes import _interpret_exit_code
from .hermes.fuzzy_match import format_no_match_hint, fuzzy_find_and_replace, is_already_applied
from .hermes.file_helpers import (
    LINTERS_INPROC, _FAIL_CLOSED_INPROC_EXTS, _READ_DEDUP_STATUS_MESSAGE,
    _detect_line_ending, _is_internal_file_tool_content, _is_likely_binary_bytes,
    _normalize_line_endings, _strip_bom, _truncate_to_char_budget,
    describe_binary_file)

# tools/tool_output_limits.py and tools/file_operations.py defaults.
MAX_LINES = 2000
MAX_LINE_LENGTH = 2000
MAX_OUTPUT_CHARS = 50_000
DEFAULT_READ_LIMIT = 500          # what _handle_read_file passes when absent
MAX_READ_CHARS = 100_000          # file_tools._DEFAULT_MAX_READ_CHARS
LARGE_FILE_HINT_BYTES = 512_000   # file_tools._LARGE_FILE_HINT_BYTES
SEARCH_DEFAULT_LIMIT = 50
SEARCH_TIMEOUT = 60

UNCHANGED_MESSAGE = _READ_DEDUP_STATUS_MESSAGE

#: Where a turn's file state lives in the tool cache (Hermes' _read_tracker).
FILE_STATE = object()


# tools/registry.py: the cap on an error body that reaches the model (static
# messages are ~115 chars; this trims runaway interpolated text). Counted in
# characters (code points), as Python's len() does.
MAX_TOOL_ERROR_CHARS = 2048
TOOL_ERROR_TRUNCATION_MARKER = "… [truncated]"


def bound_error_text(text: str) -> str:
    """tools/registry.py _bound_error_text."""
    if len(text) <= MAX_TOOL_ERROR_CHARS:
        return text

    return text[:MAX_TOOL_ERROR_CHARS] + TOOL_ERROR_TRUNCATION_MARKER


def bound_json_error_result(result: str) -> str:
    """tools/registry.py _bound_json_error_result: trim an oversized ``error``
    field of a JSON tool result; anything else passes through unchanged."""
    if len(result) <= MAX_TOOL_ERROR_CHARS or '"error"' not in result:
        return result

    try:
        payload = json.loads(result)
    except ValueError:
        return result

    if not isinstance(payload, dict):
        return result

    error = payload.get("error")

    if not isinstance(error, str) or len(error) <= MAX_TOOL_ERROR_CHARS:
        return result

    payload["error"] = bound_error_text(error)

    return json.dumps(payload, ensure_ascii=False)


def tool_error(message, **extra) -> str:
    """tools/registry.py tool_error."""
    result = {"error": bound_error_text(str(message))}

    if extra:
        result.update(extra)

    return json.dumps(result, ensure_ascii=False)


def new_state() -> dict:
    """What Hermes' _read_tracker holds for one task, plus this task's
    file_state read stamps ({resolved: (mtime, partial)})."""
    return {"last_key": None, "consecutive": 0, "read_history": set(),
            "dedup": {}, "dedup_hits": {}, "read_timestamps": {},
            "patch_failures": {}, "file_state": {}}


def _record_read(state, resolved, *, partial):
    """file_state.record_read."""
    try:
        state.setdefault("file_state", {})[resolved] = (os.path.getmtime(resolved),
                                                       bool(partial))
    except OSError:
        pass


def _note_write(state, resolved):
    """file_state.note_write: a write is a full read of what was written."""
    try:
        state.setdefault("file_state", {})[resolved] = (os.path.getmtime(resolved), False)
    except OSError:
        pass


def _check_stale(state, resolved):
    """file_state.check_stale for a single agent (no sibling subagents, so
    the last writer is always this task): an external change since the last
    read, else a partial last read."""
    stamp = state.get("file_state", {}).get(resolved)

    if stamp is None:
        return None

    try:
        current_mtime = os.path.getmtime(resolved)
    except OSError:
        return None

    read_mtime, partial = stamp

    if current_mtime != read_mtime:
        return (f"{resolved} was modified since you last read it on disk "
                "(external edit or unrecorded writer). Re-read the file before "
                "writing.")

    if partial:
        return (f"{resolved} was last read with offset/limit pagination "
                "(partial view). Re-read the whole file before overwriting it.")

    return None


def reset_dedup(state: dict) -> None:
    """file_tools.reset_file_dedup: after a compaction, reads are new again."""
    state["dedup"].clear()
    state["dedup_hits"].clear()


def invalidate_path(state: dict, resolved: str) -> None:
    """file_tools._invalidate_dedup_for_path: a write makes its reads stale."""
    for key in [key for key in state["dedup"] if key[0] == resolved]:
        state["dedup"].pop(key, None)
        state["dedup_hits"].pop(key, None)


def _coerce_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------- read


def _listing(host_dir: str, collation: str | None) -> list[str]:
    """What Hermes' `ls -1 <dir> | head -50` lists: no dotfiles, in the
    locale's collation order (en_US.UTF-8 puts bsp_rpi4_64.inc before
    bsp_rpi4.inc, code-point order the other way round). `ls` itself, run
    without a shell, so the ordering is ls's own, not an imitation of it.
    `collation` is the coding environment's locale; None keeps this
    process's own environment."""
    env = dict(os.environ)

    if collation:
        env["LC_ALL"] = collation

    try:
        done = subprocess.run(["ls", "-1", "--", host_dir], capture_output=True,
                              env=env, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []

    if done.returncode != 0:
        return []

    names = done.stdout.decode("utf-8", errors="replace").strip().split("\n")

    return [name for name in names if name][:50]


def _similar_files(display: str, resolved: str, collation: str | None = None) -> list[str]:
    """ShellFileOperations._suggest_similar_files, scoring unchanged."""
    dir_path = os.path.dirname(display) or "."
    host_dir = os.path.dirname(resolved)
    filename = os.path.basename(display)
    basename_no_ext, ext = os.path.splitext(filename)
    ext = ext.lower()
    lower_name = filename.lower()
    scored = []

    entries = _listing(host_dir, collation)

    for f in entries:
        lf = f.lower()
        score = 0

        if lf == lower_name:
            score = 100
        elif os.path.splitext(f)[0].lower() == basename_no_ext.lower():
            score = 90
        elif lf.startswith(lower_name) or lower_name.startswith(lf):
            score = 70
        elif lower_name in lf:
            score = 60
        elif lf in lower_name and len(lf) > 2:
            score = 40
        elif ext and os.path.splitext(f)[1].lower() == ext:
            common = set(lower_name) & set(lf)

            if len(common) >= max(len(lower_name), len(lf)) * 0.4:
                score = 30

        if score == 0 and difflib.SequenceMatcher(None, lower_name, lf).ratio() >= 0.8:
            score = 50

        if score > 0:
            scored.append((score, os.path.join(dir_path, f)))

    scored.sort(key=lambda x: -x[0])

    return [path for _, path in scored[:5]]


def _read_window(display: str, resolved: str, offset: int, limit: int,
                 collation: str | None = None) -> dict:
    """ShellFileOperations.read_file -> ReadResult.to_dict()."""
    if not os.path.exists(resolved):
        # ShellFileOperations._suggest_similar_files through ReadResult.to_dict:
        # the empty content is kept, an empty suggestion list is dropped.
        result = {"content": "", "total_lines": 0, "file_size": 0, "truncated": False,
                  "is_binary": False, "is_image": False,
                  "error": f"File not found: {display}"}
        similar = _similar_files(display, resolved, collation)

        if similar:
            result["similar_files"] = similar

        return result

    if not os.path.isfile(resolved):
        # ShellFileOperations._not_regular_error, through ReadResult.to_dict
        # (which keeps the empty content).
        return {"content": "", "total_lines": 0, "file_size": 0, "truncated": False,
                "is_binary": False, "is_image": False,
                "error": (f"Cannot read '{display}': not a regular file (directory, "
                          "FIFO, socket, or device). Reading it could block "
                          "indefinitely.")}

    with open(resolved, "rb") as handle:
        data = handle.read()

    file_size = len(data)
    sample = data[:1000]

    if has_binary_extension(resolved) or _is_likely_binary_bytes(sample):
        return {"content": "", "total_lines": 0, "file_size": file_size, "truncated": False,
                "is_binary": True, "is_image": False,
                "error": describe_binary_file(sample, file_size)}

    if file_size == 0:
        return {"content": "", "total_lines": 0, "file_size": 0,
                "truncated": False, "hint": "File is empty (0 bytes).",
                "is_binary": False, "is_image": False}

    text = data.decode("utf-8", "replace")
    end_line = offset + limit - 1
    total_lines = text.count("\n")      # `wc -l` counts newlines
    lines = text.split("\n")

    if text.endswith("\n"):
        lines = lines[:-1]

    if offset > total_lines > 0 and offset > len(lines):
        return {"content": "", "total_lines": total_lines, "file_size": file_size,
                "truncated": False,
                "hint": (f"Note: offset {offset} is beyond the end of the file "
                         f"({total_lines} lines total). Retry with offset <= "
                         f"{total_lines}."),
                "is_binary": False, "is_image": False}

    window = lines[offset - 1:end_line]

    if offset == 1 and window:
        window[0], _ = _strip_bom(window[0])

    truncated = total_lines > end_line

    # Hermes reads the window with `sed -n 'a,bp' | cut`, whose output always
    # ends in a newline, and numbers `output.split("\n")`: the empty element
    # after that newline is numbered too ("N+1|"). It is removed only when the
    # page reaches the end of a file that has no final newline.
    if window and (truncated or text.endswith("\n")):
        window.append("")

    numbered = []

    for number, line in enumerate(window, start=offset):
        line = line[:4 * MAX_LINE_LENGTH + 1]

        if len(line) > MAX_LINE_LENGTH:
            line = line[:MAX_LINE_LENGTH] + "... [truncated]"

        numbered.append(f"{number}|{line}")

    # ReadResult.to_dict order: the hint, when there is one, sits between
    # `truncated` and `is_binary`.
    result = {"content": "\n".join(numbered), "total_lines": total_lines,
              "file_size": file_size, "truncated": truncated}

    if truncated:
        result["hint"] = (f"Use offset={end_line + 1} to continue reading "
                          f"(showing {offset}-{end_line} of {total_lines} lines)")

    result.update(is_binary=False, is_image=False)

    return result


def _special_file_kind(path: str) -> str | None:
    """file_tools._special_file_kind: the kind of a non-regular, non-directory
    file (symlinks followed), or None."""
    try:
        mode = os.stat(path).st_mode
    except OSError:
        return None

    if stat.S_ISREG(mode) or stat.S_ISDIR(mode):
        return None

    if stat.S_ISFIFO(mode):
        return "a FIFO (named pipe)"

    if stat.S_ISSOCK(mode):
        return "a socket"

    if stat.S_ISCHR(mode):
        return "a character device"

    if stat.S_ISBLK(mode):
        return "a block device"

    return "a special (non-regular) file"


def read_file(state: dict, display: str, resolved: str, offset=1,
              limit=DEFAULT_READ_LIMIT, collation: str | None = None) -> str:
    """file_tools.read_file_tool. `collation`: the locale the coding
    environment's commands run under (orders not-found suggestions)."""
    offset = max(1, _coerce_int(offset, 1))
    limit = max(1, min(_coerce_int(limit, DEFAULT_READ_LIMIT), MAX_LINES))

    # file_tools._special_file_kind: a FIFO, socket or device is never opened.
    kind = _special_file_kind(resolved)

    if kind is not None:
        return json.dumps({
            "success": False,
            "note": (f"'{display}' is {kind}, not a regular file — reading it "
                     "would block indefinitely, so no read was attempted. Use "
                     "terminal utilities if you need to interact with it.")})

    if os.path.exists(resolved) and has_binary_extension(resolved):
        ext = os.path.splitext(resolved)[1].lower()
        return tool_error(f"Cannot read binary file '{display}' ({ext}). Use "
                          "vision_analyze for images, or terminal to inspect "
                          "binary files.")

    dedup_key = (resolved, offset, limit)
    cached_mtime = state["dedup"].get(dedup_key)

    if cached_mtime is not None:
        try:
            if os.path.getmtime(resolved) == cached_mtime:
                hits = state["dedup_hits"].get(dedup_key, 0) + 1
                state["dedup_hits"][dedup_key] = hits

                if hits >= 2:
                    return tool_error(
                        f"BLOCKED: You have called read_file on this exact "
                        f"region {hits + 1} times and the file has NOT changed. "
                        "STOP calling read_file for this path — the content "
                        "from your earlier read_file result in this "
                        "conversation is still current. Proceed with your task "
                        "using the information you already have.",
                        path=display, already_read=hits + 1)

                return json.dumps({"status": "unchanged",
                                   "message": UNCHANGED_MESSAGE, "path": display,
                                   "dedup": True, "content_returned": False},
                                  ensure_ascii=False)
        except OSError:
            pass

    result = _read_window(display, resolved, offset, limit, collation)
    content = result.get("content") or ""

    if len(content) > MAX_READ_CHARS:
        total_lines = result.get("total_lines", "unknown")
        trimmed, lines_kept, _ = _truncate_to_char_budget(content, MAX_READ_CHARS)
        next_offset = offset + lines_kept
        shown_end = offset + lines_kept - 1
        result.update(content=trimmed, truncated=True, truncated_by="bytes",
                      next_offset=next_offset)
        result["hint"] = (
            f"Output truncated at the {MAX_READ_CHARS:,}-char read budget after "
            f"{lines_kept} line(s) (showing lines {offset}-{shown_end} of "
            f"{total_lines}). Use offset={next_offset} to continue.")

        if len(trimmed.split("\n", 1)[0]) >= MAX_READ_CHARS:
            result["hint"] += (" Note: the first line alone exceeded the budget "
                               "and was clamped mid-line; its remainder is not "
                               "retrievable via offset.")

    file_size = result.get("file_size", 0)

    if (file_size and file_size > LARGE_FILE_HINT_BYTES and limit > 200
            and result.get("truncated")):
        result.setdefault("_hint", (
            f"This file is large ({file_size:,} bytes). Consider reading only "
            "the section you need with offset and limit to keep context usage "
            "efficient."))

    if result.get("error"):
        return json.dumps(result, ensure_ascii=False)

    read_key = ("read", display, offset, limit)
    state["dedup_hits"].pop(dedup_key, None)
    state["read_history"].add((display, offset, limit))

    if state["last_key"] == read_key:
        state["consecutive"] += 1
    else:
        state["last_key"], state["consecutive"] = read_key, 1

    count = state["consecutive"]

    try:
        mtime = os.path.getmtime(resolved)
        state["dedup"][dedup_key] = mtime
        state["read_timestamps"][resolved] = mtime
    except OSError:
        pass

    # file_state.record_read: a read is partial when it started past line 1
    # or did not reach the end.
    _record_read(state, resolved, partial=offset > 1 or bool(result.get("truncated")))

    if count >= 4:
        return tool_error(
            f"BLOCKED: You have read this exact file region {count} times in a "
            "row. The content has NOT changed. You already have this "
            "information. STOP re-reading and proceed with your task.",
            path=display, already_read=count)

    if count >= 3:
        result["_warning"] = (
            f"You have read this exact file region {count} times consecutively. "
            "The content has not changed since your last read. Use the "
            "information you already have. If you are stuck in a loop, stop "
            "reading and proceed with writing or responding.")

    return json.dumps(result, ensure_ascii=False)


# --------------------------------------------------------------- search


def _rg(args, cwd, timeout=SEARCH_TIMEOUT):
    """(stdout, exit_code, limit_reason) of one ripgrep run."""
    rg = shutil.which("rg")

    if rg is None:
        return "", 127, None

    try:
        done = subprocess.run([rg, *args], cwd=cwd, capture_output=True,
                              text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        out = out.decode("utf-8", "replace") if isinstance(out, bytes) else out
        return out, 124, "search_timeout"

    return (done.stdout + (done.stderr if done.returncode == 2 else ""),
            done.returncode, None)


def _head(text, n):
    return "\n".join([line for line in text.split("\n") if line != ""][:n])


def _zero_match_probe(pattern, path_arg, file_glob, cwd):
    """ShellFileOperations._zero_match_probe."""
    def tally(stdout):
        total, per_file = 0, []

        for line in (stdout or "").strip().splitlines():
            p, _sep, n = line.rpartition(":")

            if n.isdigit():
                total += int(n)
                per_file.append(p)

        return total, per_file

    def paths_note(per_file, cap=5):
        extra = len(per_file) - cap
        return ", ".join(per_file[:cap]) + (f" (+{extra} more)" if extra > 0 else "")

    glob = ["--glob", file_glob] if file_glob else []
    out, _, _ = _rg(["-i", "--count-matches", *glob, "-e", pattern, "--", path_arg], cwd, 30)
    total, paths = tally(_head(out, 50))

    if total > 0:
        return (f"0 exact matches, but {total} case-insensitive match(es) in "
                f"{len(paths)} file(s): {paths_note(paths)} — the pattern's "
                "casing may be wrong.")

    out, _, _ = _rg(["--hidden", "--no-ignore", "--count-matches", *glob, "-e",
                     pattern, "--", path_arg], cwd, 30)
    total, paths = tally(_head(out, 50))

    if total > 0:
        return (f"0 matches in visible files, but {total} match(es) in "
                f"{len(paths)} hidden or gitignored file(s): {paths_note(paths)} "
                "— these are excluded by default.")

    if re.search(r"[.\[\](){}?*+^$\\|]", pattern):
        out, _, _ = _rg(["-F", "--count-matches", *glob, "-e", pattern, "--",
                         path_arg], cwd, 30)
        total, paths = tally(_head(out, 50))

        if total > 0:
            return (f"0 regex matches, but {total} literal match(es) in "
                    f"{len(paths)} file(s): {paths_note(paths)} — the pattern "
                    "contains regex metacharacters that likely need escaping "
                    "(or pass a simpler substring).")

    return None


def _search_dict(total_count, *, matches=(), files=(), counts=None, truncated=False,
                 limit_reason=None, warning=None, error=None) -> dict:
    """SearchResult.to_dict(densify=True)."""
    result: dict = {"total_count": total_count}

    if matches:
        if len(matches) >= 5:
            lines, current = [], None

            for path, line, content in matches:
                if path != current:
                    lines.append(path)
                    current = path

                lines.append(f"  {line}: {content.rstrip()}")

            result["matches_format"] = (
                "path-grouped: each file path on its own line, followed by "
                "indented '<line>: <content>' rows for matches in that file")
            result["matches_text"] = "\n".join(lines)
        else:
            result["matches"] = [{"path": p, "line": n, "content": c}
                                 for p, n, c in matches]

    if files:
        result["files"] = list(files)

    if counts:
        result["counts"] = counts

    if truncated:
        result["truncated"] = True

    if limit_reason:
        result["limit_reason"] = limit_reason

    if warning:
        result["warning"] = warning

    if error:
        result["error"] = error

    return result


_MATCH_RE = re.compile(r'^([A-Za-z]:)?(.*?):(\d+):(.*)$')
_CONTEXT_RE = re.compile(r'^([A-Za-z]:)?(.*?)-(\d+)-(.*)$')


def search(state: dict, cwd: str, path_arg: str, resolved: str, pattern: str, *,
           target="content", file_glob=None, limit=SEARCH_DEFAULT_LIMIT, offset=0,
           output_mode="content", context=0) -> str:
    """file_tools.search_tool + ShellFileOperations.search.

    `path_arg` is the path as the model gave it: ripgrep runs from `cwd`
    with it, so results name files the way Hermes' do (`./build/...`).
    """
    offset = max(0, _coerce_int(offset, 0))
    limit = max(1, _coerce_int(limit, SEARCH_DEFAULT_LIMIT))
    context = max(0, _coerce_int(context, 0))
    target = {"grep": "content", "find": "files"}.get(target or "content",
                                                       target or "content")
    search_key = ("search", pattern, target, str(path_arg), file_glob or "",
                  limit, offset)

    if state["last_key"] == search_key:
        state["consecutive"] += 1
    else:
        state["last_key"], state["consecutive"] = search_key, 1

    count = state["consecutive"]

    if count >= 4:
        return tool_error(
            f"BLOCKED: You have run this exact search {count} times in a row. "
            "The results have NOT changed. You already have this information. "
            "STOP re-searching and proceed with your task.",
            pattern=pattern, already_searched=count)

    if not os.path.exists(resolved):
        parent = os.path.dirname(path_arg) or "."
        query = os.path.basename(path_arg)
        parts = [f"Path not found: {path_arg}"]
        host_parent = os.path.dirname(resolved)

        if os.path.isdir(host_parent) and query:
            lower_q = query.lower()
            candidates = [os.path.join(parent, entry)
                          for entry in sorted(os.listdir(host_parent))[:20]
                          if lower_q in entry.lower() or entry.lower() in lower_q
                          or entry.lower().startswith(lower_q[:3])]

            if candidates:
                parts.append("Similar paths: " + ", ".join(candidates[:5]))

        result = _search_dict(0, error=". ".join(parts))
    elif target == "files":
        glob = pattern if ("/" in pattern or pattern.startswith("*")) else f"*{pattern}"
        fetch = limit + offset
        out, _, reason = _rg(["--files", "--sortr=modified", "-g", glob, "--",
                              path_arg], cwd)
        names = [f for f in _head(out, fetch).split("\n") if f]

        if not names and not reason:
            out, _, reason = _rg(["--files", "-g", glob, "--", path_arg], cwd)
            names = [f for f in _head(out, fetch).split("\n") if f]

        result = _search_dict(len(names), files=names[offset:offset + limit],
                              truncated=len(names) >= fetch or bool(reason),
                              limit_reason=reason)
    else:
        args = ["--line-number", "--no-heading", "--with-filename"]
        multiline = "\n" in pattern or bool(re.search(r"(?<!\\)(?:\\\\)*\\n", pattern))

        if multiline:
            args.append("--multiline")

        if context > 0:
            args += ["-C", str(context)]

        if file_glob:
            args += ["--glob", file_glob]

        if output_mode == "files_only":
            args.append("-l")
        elif output_mode == "count":
            args.append("-c")

        args += ["-e", pattern, "--", path_arg]
        out, code, reason = _rg(args, cwd)
        fetch = limit + offset + 200 if context > 0 else limit + offset
        out = _head(out, fetch)
        note = ("Pattern contains \\n — multiline mode (-U) was enabled "
                "automatically so the regex can match across line boundaries."
                if multiline else None)

        if code == 2 and not _MATCH_RE.match(out.split("\n", 1)[0] or ""):
            result = _search_dict(0, error=f"Search failed: "
                                           f"{out.strip() or 'Search error'}")
        elif output_mode == "files_only":
            files = [f for f in out.split("\n") if f]
            result = _search_dict(len(files), files=files[offset:offset + limit],
                                  truncated=bool(reason), limit_reason=reason,
                                  warning=note)
        elif output_mode == "count":
            counts = {}

            for line in out.split("\n"):
                p, _, n = line.rpartition(":")

                if p and n.isdigit():
                    counts[p] = int(n)

            result = _search_dict(sum(counts.values()), counts=counts,
                                  truncated=bool(reason), limit_reason=reason)
        else:
            matches = []

            for line in out.split("\n"):
                if not line or line == "--":
                    continue

                m = _MATCH_RE.match(line) or (context > 0 and _CONTEXT_RE.match(line))

                if m:
                    matches.append(((m.group(1) or "") + m.group(2),
                                    int(m.group(3)), m.group(4)[:500]))

            total = len(matches)
            result = _search_dict(total, matches=matches[offset:offset + limit],
                                  truncated=total > offset + limit or bool(reason),
                                  limit_reason=reason, warning=note)

        # ShellFileOperations.search: zero-match steering runs for every
        # output mode -- content, files_only and count alike.
        if (not result.get("error") and result.get("total_count") == 0
                and not result.get("matches") and not result.get("files")
                and not result.get("counts")):
            try:
                hint = _zero_match_probe(pattern, path_arg, file_glob, cwd)
            except Exception:
                hint = None

            if hint:
                result["warning"] = (hint if not result.get("warning")
                                     else f"{result['warning']} {hint}")

    if count >= 3:
        result["_warning"] = (
            f"You have run this exact search {count} times consecutively. The "
            "results have not changed. Use the information you already have.")

    text = json.dumps(result, ensure_ascii=False)

    if result.get("truncated"):
        text += (f"\n\n[Hint: Results truncated. Use offset={offset + limit} to "
                 "see more, or narrow with a more specific pattern or file_glob.]")

    return text


# ----------------------------------------------------------------- lint


def lint_delta(path: str, pre: str | None, post: str) -> dict:
    """ShellFileOperations._check_lint_delta, in-process linters only."""
    ext = os.path.splitext(path)[1].lower()
    linter = LINTERS_INPROC.get(ext)

    if linter is None:
        return {"status": "skipped", "message": f"No linter for {ext} files"}

    ok, err = linter(post)

    if err == "__SKIP__":
        return {"status": "skipped",
                "message": f"No linter available for {ext} (missing dependency)"}

    if ok or pre is None:
        return {"status": "ok" if ok else "error", "output": "" if ok else err}

    pre_ok, pre_err = linter(pre)

    if pre_ok or not pre_err or pre_err == "__SKIP__":
        return {"status": "error", "output": err}

    pre_lines = {line.strip() for line in pre_err.splitlines() if line.strip()}
    new = [line for line in err.splitlines()
           if line.strip() and line.strip() not in pre_lines]

    if not new:
        return {"status": "error", "output": err,
                "message": "Pre-existing lint errors — this edit didn't introduce "
                           "new ones but the file is still broken."}

    return {"status": "error",
            "output": "New lint errors introduced by this edit (pre-existing "
                      "errors filtered out):\n" + "\n".join(new)}


def _staleness(state, display, resolved):
    """file_tools._check_file_staleness."""
    read_mtime = state["read_timestamps"].get(resolved)

    if read_mtime is None:
        return None

    try:
        if os.path.getmtime(resolved) != read_mtime:
            return (f"Warning: {display} was modified since you last read it "
                    "(external edit or concurrent agent). The content you read "
                    "may be stale. Consider re-reading the file to verify before "
                    "writing.")
    except OSError:
        pass

    return None


def _after_write(state, resolved):
    invalidate_path(state, resolved)
    _note_write(state, resolved)

    try:
        state["read_timestamps"][resolved] = os.path.getmtime(resolved)
    except OSError:
        pass


# ---------------------------------------------------------------- patch


def unified_diff(before: str, after: str, label: str) -> str:
    """ShellFileOperations._unified_diff."""
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{label}", tofile=f"b/{label}"))


def patch(state: dict, display: str, resolved: str, old_string, new_string,
          replace_all=False, *, write) -> tuple[str, bool]:
    """file_tools.patch_tool (replace mode) + ShellFileOperations.patch_replace.

    patch_tool hands patch_replace the RESOLVED path: the diff labels and its
    messages name the absolute file, whatever form the model wrote; only
    patch_tool's own repeated-failure hint quotes the path as given.

    `write(new_content)` performs the authorised write and returns an error
    string or None. Returns (the JSON, whether the file changed).
    """
    if not display:
        return tool_error("path required"), False

    if old_string is None or new_string is None:
        return tool_error("old_string and new_string required"), False

    warning = _check_stale(state, resolved) or _staleness(state, display, resolved)

    if not os.path.isfile(resolved):
        result = {"success": False, "error": f"Failed to read file: {resolved}"}
    else:
        with open(resolved, "r", encoding="utf-8", errors="surrogateescape",
                  newline="") as handle:
            raw = handle.read()

        content, _ = _strip_bom(raw)
        new_content, match_count, _strategy, error = fuzzy_find_and_replace(
            content, old_string, new_string, replace_all)

        if error or match_count == 0:
            if is_already_applied(content, old_string, new_string):
                result = {"success": True, "no_change": True,
                          "note": (f"File already contains the target text — "
                                   f"the edit appears to be already applied to "
                                   f"{resolved}. No write performed; do not "
                                   "re-send this patch.")}
            else:
                message = error or f"Could not find match for old_string in {resolved}"

                try:
                    message += format_no_match_hint(message, match_count,
                                                    old_string, content)
                except Exception:
                    pass

                result = {"success": False, "error": message}
        else:
            ending = _detect_line_ending(content)

            if ending:
                new_content = _normalize_line_endings(new_content, ending)

            failure = write(new_content)

            if failure:
                result = {"success": False,
                          "error": f"Failed to write changes: {failure}"}
            else:
                result = {"success": True,
                          "diff": unified_diff(content, new_content, resolved),
                          "files_modified": [resolved],
                          "lint": lint_delta(resolved, content, new_content)}

                if warning:
                    result["_warning"] = warning

                result["resolved_path"] = resolved
                _after_write(state, resolved)
                state["patch_failures"].pop(resolved, None)

    if warning:
        result["_warning"] = warning

    if result.get("error") and "Could not find" in str(result["error"]):
        failures = state["patch_failures"].get(resolved, 0) + 1
        state["patch_failures"][resolved] = failures

        if failures >= 3:
            result["_hint"] = (
                f"This is failure #{failures} patching {display!r}. Stop "
                "retrying with variations of the same old_string. Either: (1) "
                "re-read the file fresh to verify current content, (2) use a "
                "longer / more unique old_string with surrounding context "
                "lines, or (3) use write_file to replace the entire file if "
                "the targeted region is hard to anchor.")
        elif "Did you mean one of these sections?" not in str(result["error"]):
            result["_hint"] = ("old_string not found. Use read_file to verify "
                               "the current content, or search_files to locate "
                               "the text.")

    return json.dumps(result, ensure_ascii=False), bool(result.get("diff"))


# ----------------------------------------------------------- write_file


def write_file(state: dict, display: str, resolved: str, content, *,
               write) -> tuple[str, bool]:
    """file_tools.write_file_tool + ShellFileOperations.write_file.

    `write(content)` creates parent directories and writes; it returns an
    error string or None.
    """
    if not isinstance(content, str):
        return tool_error("content must be a string"), False

    if _is_internal_file_tool_content(content):
        return tool_error(
            "Refusing to write internal read_file display text as file "
            "content. Strip read_file line-number prefixes or reconstruct the "
            "intended file contents before writing."), False

    m = re.search(r"[\ud800-\udc7f\udd00-\udfff]", content)

    if m:
        return json.dumps({"bytes_written": 0, "dirs_created": False, "error": (
            f"Refusing to write '{display}': content contains a lone surrogate "
            f"character ({m.group(0)!r}) that cannot be encoded as UTF-8. The "
            "file was NOT created or modified.")}, ensure_ascii=False), False

    ext = os.path.splitext(resolved)[1].lower()

    if ext in _FAIL_CLOSED_INPROC_EXTS:
        ok, err = LINTERS_INPROC[ext](content)

        if not ok and err != "__SKIP__":
            return json.dumps({"bytes_written": 0, "dirs_created": False, "error": (
                f"Refusing to write '{display}': candidate content fails {ext} "
                f"syntax validation ({err}). The file was NOT created or "
                "modified. Fix the content and retry.")},
                ensure_ascii=False), False

    warning = _check_stale(state, resolved) or _staleness(state, display, resolved)
    pre = None

    if os.path.isfile(resolved):
        try:
            with open(resolved, "r", encoding="utf-8", errors="replace",
                      newline="") as handle:
                pre = handle.read()
        except OSError:
            pre = None

        if pre is not None and _detect_line_ending(pre) == "\r\n":
            content = _normalize_line_endings(content, "\r\n")

    # ShellFileOperations.write_file: "parent dirs ensured", i.e. true for
    # any path with a directory part (`mkdir -p` succeeds when they exist).
    dirs_created = bool(os.path.dirname(resolved))
    failure = write(content)

    if failure:
        result = {"bytes_written": 0, "dirs_created": False, "error": failure}

        if warning:
            result["_warning"] = warning
    else:
        result = {"bytes_written": len(content.encode("utf-8")),
                  "dirs_created": dirs_created, "verified": True,
                  "lint": lint_delta(resolved, pre, content)}

        if warning:
            result["_warning"] = warning

        result.update(resolved_path=resolved, files_modified=[resolved])
        _after_write(state, resolved)

    return json.dumps(result, ensure_ascii=False), not failure


# ------------------------------------------------------------- terminal


def terminal_result(command: str, output: str, exit_code, *, error=None,
                    cwd_changed: str | None = None) -> str:
    """The JSON terminal_tool returns, from its output-shaping steps.

    The head/tail cut at MAX_OUTPUT_CHARS is made inside the session script
    (see session_script); this applies the rest in Hermes' order: ANSI
    strip, strip, exit-code interpretation, then a failure or masked-success
    hint.
    """
    from .hermes.terminal_hints import annotate_failure, annotate_masked_success

    output = strip_ansi(output).strip() if output else ""
    result = {"output": output, "exit_code": exit_code, "error": error}

    if cwd_changed:
        result["cwd"] = cwd_changed

    if exit_code is not None and error is None:
        note = _interpret_exit_code(command, exit_code)

        if note:
            result["exit_code_meaning"] = note
        else:
            try:
                hint = (annotate_failure(command, exit_code, output)
                        if exit_code != 0
                        else annotate_masked_success(command, output))
            except Exception:
                hint = None

            if hint:
                result["hint"] = hint

    return json.dumps(result, ensure_ascii=False)


# -------------------------------------------------------------- terminal
#
# Hermes runs every command as a fresh `bash -c` that restores the previous
# call's exported variables from a snapshot, enters the recorded cwd, runs
# the command with `eval`, re-dumps `export -p` and prints the new cwd
# (tools/environments/base.py _wrap_command); a timeout kills the process
# group, keeps the partial output, appends "[Command timed out after Ns]",
# returns 124, and loses that command's state (base.py:1247-1257). SPEAR
# runs each command in a fresh sandbox, which is the same model: the state
# travels through the output instead of a snapshot file.
#
# The command runs in an inner shell under `timeout`, so a timeout is
# Hermes' -- partial output kept, the inner shell's state lost -- rather
# than the sandbox's, which would discard the output. The outer script
# prints the cwd, the exports and the output size before the output, so
# the sandbox's own output cap can never cut the state off, and hands back
# the 40% head / 60% tail that Hermes' collector keeps (base.py:96, 221).

TERMINAL_DEFAULT_TIMEOUT = 180     # TERMINAL_TIMEOUT default (terminal_tool.py:1875)
TERMINAL_MAX_TIMEOUT = 600         # FOREGROUND_MAX_TIMEOUT (terminal_tool.py:120)
_MARK = "\x1eSPEAR_SESSION\x1e"

# Shell bookkeeping that must never be carried into the next command
# (base.py:582 excludes Hermes' own variables; these are bash's).
_SESSION_VARS = re.compile(
    r"^declare -[-a-zA-Z]* (PWD|OLDPWD|SHLVL|_|BASH_[A-Z_]*|BASHOPTS|SHELLOPTS|"
    r"HOSTNAME|PPID|RANDOM|SECONDS|LINENO|EUID|UID|GROUPS|__SPEAR_[A-Z_]*)(=|$)")


def terminal_timeout(value):
    """The call's timeout, or None when it exceeds the foreground maximum."""
    try:
        timeout = int(value) if value not in (None, "") else TERMINAL_DEFAULT_TIMEOUT
    except (TypeError, ValueError):
        timeout = TERMINAL_DEFAULT_TIMEOUT

    if timeout > TERMINAL_MAX_TIMEOUT:
        return None

    return max(1, timeout)


def session_script(command: str, cwd: str | None, exports: str, timeout: int) -> str:
    """The sandboxed script that runs `command` in the carried-over session."""
    import secrets
    import shlex

    escaped = command.replace("'", "'\\''")
    tag = "__SPEAR_INNER_" + secrets.token_hex(8)
    snap_tag = "__SPEAR_SNAP_" + secrets.token_hex(8)
    head = int(MAX_OUTPUT_CHARS * 0.4)
    tail = MAX_OUTPUT_CHARS - head

    # The command runs exactly as Hermes runs it (tools/environments/base.py,
    # _wrap_command; local.py, _run_bash): `bash -c <wrapper>`, non-login,
    # with `eval` on the wrapper's line 5 -- so bash's own diagnostics read
    # "/usr/bin/bash: line 5: ...", "/usr/bin/bash: eval: line 5: ..." as
    # Hermes' do, and a failure further down an eval'd script lands on the
    # same line number. Line by line:
    #   1  the session snapshot (exports carried over), sourced
    #   2  AI_AGENT / HERMES_AGENT, exported for every command
    #   3  GIT_PAGER / PAGER
    #   4  the session cwd
    #   5  the command
    inner = [
        'source "$__SPEAR_D/snap" >/dev/null 2>&1 || true',
        'export AI_AGENT="${AI_AGENT:-hermes-agent}" HERMES_AGENT="${HERMES_AGENT:-true}"',
        'export GIT_PAGER="${GIT_PAGER:-cat}" PAGER="${PAGER:-cat}"',
        f"builtin cd -- {shlex.quote(cwd) if cwd else '.'} || exit 126",
        f"eval '{escaped}'",
        "__spear_ec=$?",
        'pwd -P > "$__SPEAR_D/cwd"', 'export -p > "$__SPEAR_D/env"',
        "exit $__spear_ec",
    ]

    return "\n".join([
        "__SPEAR_D=$(mktemp -d)",
        f"cat > \"$__SPEAR_D/snap\" <<'{snap_tag}'",
        *([exports] if exports else []),
        snap_tag,
        f"cat > \"$__SPEAR_D/inner\" <<'{tag}'",
        *inner,
        tag,
        "export __SPEAR_D",
        f"timeout -k 2 {int(timeout)} \"$(command -v bash)\" -c \"$(cat \"$__SPEAR_D/inner\")\" "
        "> \"$__SPEAR_D/out\" 2>&1 < /dev/null",
        "__spear_ec=$?",
        "__spear_size=$(wc -c < \"$__SPEAR_D/out\")",
        f"printf '%s\\n' '{_MARK}'",
        "printf 'exit %s\\nsize %s\\n' \"$__spear_ec\" \"$__spear_size\"",
        "printf 'cwd %s\\n' \"$(cat \"$__SPEAR_D/cwd\" 2>/dev/null)\"",
        "cat \"$__SPEAR_D/env\" 2>/dev/null",
        f"printf '%s\\n' '{_MARK}'",
        f"if [ \"$__spear_size\" -gt {MAX_OUTPUT_CHARS} ]; then",
        f"  head -c {head} \"$__SPEAR_D/out\"; printf '%s' '{_MARK}'; "
        f"tail -c {tail} \"$__SPEAR_D/out\"",
        "else cat \"$__SPEAR_D/out\"; fi",
        "exit $__spear_ec",
    ])


def _parse_session(stdout: str):
    """(exit, size, cwd, exports, output) from a session script's stdout."""
    if not (stdout or "").startswith(_MARK + "\n"):
        return None

    header, sep, body = stdout[len(_MARK) + 1:].partition(_MARK + "\n")

    if not sep:
        return None

    lines = header.split("\n")
    fields = dict(line.split(" ", 1) for line in lines[:3] if " " in line)
    exports = "\n".join(line for line in lines[3:] if line.startswith("declare -x ")
                        and not _SESSION_VARS.match(line))

    try:
        code, size = int(fields.get("exit", "")), int(fields.get("size", "0"))
    except ValueError:
        return None

    if _MARK in body:
        head, _, tail = body.partition(_MARK)
        omitted = size - len(head) - len(tail)
        body = (head + f"\n\n... [OUTPUT TRUNCATED - {omitted:,} chars omitted "
                f"out of {size:,} total] ...\n\n" + tail)

    return code, size, fields.get("cwd", "").strip(), exports, body


def terminal_from_outcome(command: str, outcome, session: dict, root: str,
                          timeout: int):
    """(the JSON the model reads, exit code, timed out) for one command."""
    if outcome.status == "denied":
        return (json.dumps({"output": "", "exit_code": -1,
                            "error": outcome.summary or "refused",
                            "status": "blocked"}, ensure_ascii=False), -1, False)

    if outcome.status == "cancelled":
        return terminal_result(command, "[Command interrupted]", 130), 130, False

    parsed = _parse_session(outcome.output)

    if parsed is None:
        # The sandbox itself ended the run (its own limit, or a failure to
        # start): Hermes' outer backstop answers the same way.
        if outcome.status == "timeout":
            return terminal_result(command, f"[Command timed out after {timeout}s]",
                                   124), 124, True

        return (json.dumps({"output": (outcome.output or "").strip(), "exit_code": -1,
                            "error": f"Command execution failed: "
                                     f"{outcome.summary or 'no result'}"},
                           ensure_ascii=False), -1, False)

    code, _, cwd, exports, output = parsed
    timed_out = code in (124, 137)

    if timed_out:
        code = 124
        output = (output + f"\n[Command timed out after {timeout}s]").lstrip("\n")

    before = session.get("cwd") or root
    moved = None

    if cwd and not timed_out:
        session["cwd"], session["env"] = cwd, exports

        if os.path.realpath(cwd) != os.path.realpath(before):
            moved = cwd

    return terminal_result(command, output, code, cwd_changed=moved), code, timed_out


# ------------------------------------------------------------- workspace
#
# agent/coding_context.py: _parse_status is copied unmodified; the workspace
# block below keeps build_coding_workspace_block's git lines and wording.
# Its project-fact lines (_project_facts) are not carried.

def _parse_status(porcelain: str) -> tuple[dict[str, str], dict[str, int]]:
    """Parse ``git status --porcelain=2 --branch`` into branch + counts."""
    branch: dict[str, str] = {}
    counts = {"staged": 0, "modified": 0, "untracked": 0, "conflicts": 0}
    for line in porcelain.splitlines():
        if line.startswith("# branch.head"):
            branch["head"] = line.split(maxsplit=2)[-1]
        elif line.startswith("# branch.upstream"):
            branch["upstream"] = line.split(maxsplit=2)[-1]
        elif line.startswith("# branch.ab"):
            parts = line.split()
            branch["ahead"], branch["behind"] = parts[2].lstrip("+"), parts[3].lstrip("-")
        elif line.startswith(("1 ", "2 ")):
            xy = line.split(maxsplit=2)[1]
            if xy[0] != ".":
                counts["staged"] += 1
            if xy[1] != ".":
                counts["modified"] += 1
        elif line.startswith("u "):
            counts["conflicts"] += 1
        elif line.startswith("? "):
            counts["untracked"] += 1
    return branch, counts



def _git(root: str, *args: str) -> str:
    try:
        done = subprocess.run(["git", "-C", root, *args], capture_output=True,
                              text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""

    return done.stdout.strip() if done.returncode == 0 else ""


def workspace_block(cwd: str) -> str:
    """build_coding_workspace_block, git part."""
    root = _git(cwd, "rev-parse", "--show-toplevel")

    if not root:
        return ""

    lines = ["Workspace (snapshot at session start — re-check with `git` before "
             "acting on it):", f"- Root: {root}"]
    branch, counts = _parse_status(_git(root, "status", "--porcelain=2", "--branch"))
    head = branch.get("head", "")

    if head and head != "(detached)":
        line = f"- Branch: {head}"

        if branch.get("upstream"):
            line += f" → {branch['upstream']}"
            ahead, behind = branch.get("ahead", "0"), branch.get("behind", "0")

            if ahead != "0" or behind != "0":
                line += f" (ahead {ahead}, behind {behind})"

        lines.append(line)
    elif head == "(detached)":
        lines.append("- Branch: (detached HEAD)")

    git_dir = _git(root, "rev-parse", "--git-dir")
    common_dir = _git(root, "rev-parse", "--git-common-dir")

    if git_dir and common_dir and (os.path.realpath(os.path.join(root, git_dir))
                                   != os.path.realpath(os.path.join(root, common_dir))):
        lines.append("- Worktree: linked (git state shared with primary tree)")

    dirty = [f"{n} {label}" for label, n in (
        ("staged", counts["staged"]), ("modified", counts["modified"]),
        ("untracked", counts["untracked"]), ("conflicts", counts["conflicts"])) if n]
    lines.append(f"- Status: {', '.join(dirty) if dirty else 'clean'}")
    recent = _git(root, "log", "-3", "--pretty=%h %s")

    if recent:
        lines.append("- Recent commits:")
        lines.extend(f"    {c}" for c in recent.splitlines())

    return "\n".join(lines)
