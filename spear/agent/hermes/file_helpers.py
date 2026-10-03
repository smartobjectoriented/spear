"""Pure helpers of Hermes Agent's file tools, copied verbatim.

Copied from Hermes Agent (https://github.com/NousResearch/hermes-agent) at
commit 0cbc6e37 (v0.21.0); each block below is unmodified apart from
dedenting one @staticmethod into a module function. Copyright (c) 2025 Nous
Research. MIT License -- see THIRD_PARTY_NOTICES.md at the repository root.

Sources:
  tools/file_operations.py  _detect_line_ending, _normalize_line_endings,
                            _strip_bom, _MAGIC_SIGNATURES,
                            identify_binary_bytes, describe_binary_file,
                            _lint_*_inproc, LINTERS_INPROC,
                            _FAIL_CLOSED_INPROC_EXTS,
                            ShellFileOperations._is_likely_binary_bytes
  tools/file_tools.py       _truncate_to_char_budget,
                            _READ_DEDUP_STATUS_MESSAGE,
                            _is_internal_file_status_text,
                            _looks_like_read_file_line_numbered_content,
                            _is_internal_file_tool_content
"""

from __future__ import annotations

from typing import Optional

# --- tools/file_operations.py:129-193
def _detect_line_ending(sample: str) -> Optional[str]:
    """Return the dominant line ending in ``sample`` or None if undetermined.

    Looks at the first few line breaks and picks ``\\r\\n`` if any are
    present (Windows / DOS), otherwise ``\\n`` (Unix).  Returns ``None``
    for empty / single-line content where we can't tell.  Used to
    preserve the file's original line endings across write_file and
    patch operations — without this the agent's bare-LF tool args
    silently normalize Windows-line-ending files, and patch produces
    mixed endings when only a substituted region changes.
    """
    if not sample:
        return None
    # Look at the first chunk — enough to tell, cheap to scan.
    head = sample[:4096]
    if "\r\n" in head:
        return "\r\n"
    if "\n" in head:
        return "\n"
    return None


def _normalize_line_endings(text: str, target: str) -> str:
    """Convert all line endings in ``text`` to ``target`` (``\\n`` or ``\\r\\n``).

    Idempotent: ``_normalize_line_endings(_normalize_line_endings(x, "\\r\\n"), "\\r\\n") == _normalize_line_endings(x, "\\r\\n")``.
    Strips lone ``\\r`` characters as well, so mixed-ending content is
    homogenized in a single pass.
    """
    # First collapse to LF (handle CRLF and lone CR), then expand if target
    # is CRLF.  Order matters: doing the replacements separately would
    # double-convert a CRLF -> LFLF.
    lf_normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if target == "\n":
        return lf_normalized
    if target == "\r\n":
        return lf_normalized.replace("\n", "\r\n")
    return text


# UTF-8 byte order mark. Some Windows editors (Notepad, older Visual Studio,
# some PowerShell redirects) prepend this invisible 3-byte marker
# (EF BB BF == U+FEFF) to UTF-8 text files. It renders as nothing but is a
# real character at the start of the decoded string, so without handling it:
#   - read_file would surface a stray U+FEFF as the first character (the
#     model sees a phantom char before `import ...`), and
#   - patch matches against the true first line would miss, and write_file
#     would silently drop or double the marker on rewrite.
# We strip it on read so the model sees clean content, and restore it on
# write when the original file had one — exactly mirroring the line-ending
# preservation above (detect on disk, preserve across the edit).
_UTF8_BOM = "\ufeff"


def _strip_bom(text: str) -> tuple[str, bool]:
    """Return (text-without-leading-BOM, had_bom).

    Only a single leading BOM is stripped; a BOM appearing mid-content is
    left alone (it's legitimate data there, not a file marker).
    """
    if text and text.startswith(_UTF8_BOM):
        return text[len(_UTF8_BOM):], True
    return text, False



# --- tools/file_operations.py:500-562
_MAGIC_SIGNATURES: tuple = (
    # (prefix bytes, human name) — ordered, first match wins. Longest
    # prefixes for a shared first byte come first.
    (b"\x89PNG\r\n\x1a\n", "PNG image data"),
    (b"\xff\xd8\xff", "JPEG image data"),
    (b"GIF87a", "GIF image data"),
    (b"GIF89a", "GIF image data"),
    (b"RIFF", "RIFF container (WAV/AVI/WebP family)"),
    (b"%PDF-", "PDF document"),
    (b"PK\x03\x04", "ZIP archive (also docx/xlsx/jar/apk)"),
    (b"PK\x05\x06", "ZIP archive (empty)"),
    (b"\x1f\x8b", "gzip compressed data"),
    (b"BZh", "bzip2 compressed data"),
    (b"\xfd7zXZ\x00", "xz compressed data"),
    (b"7z\xbc\xaf\x27\x1c", "7-Zip archive"),
    (b"\x7fELF", "ELF executable"),
    (b"MZ", "Windows PE executable"),
    (b"\xcf\xfa\xed\xfe", "Mach-O executable (64-bit)"),
    (b"\xca\xfe\xba\xbe", "Mach-O universal binary / Java class"),
    (b"SQLite format 3\x00", "SQLite database"),
    (b"OggS", "Ogg container"),
    (b"fLaC", "FLAC audio"),
    (b"ID3", "MP3 audio (ID3 tag)"),
    (b"\x00\x00\x00", "ISO media container (MP4/MOV family)"),  # ftyp at +4
    (b"BM", "BMP image data"),
    (b"II*\x00", "TIFF image data (little-endian)"),
    (b"MM\x00*", "TIFF image data (big-endian)"),
)


def identify_binary_bytes(sample: bytes) -> str:
    """Best-effort human name for binary content from its magic bytes.

    Returns e.g. ``"PNG image data"`` or ``"unknown binary"``. Never raises.
    The ISO-media entry additionally checks for ``ftyp`` at offset 4, since
    the leading size field alone (three NULs) is too weak a signature.
    """
    if not sample:
        return "unknown binary"
    for prefix, name in _MAGIC_SIGNATURES:
        if sample.startswith(prefix):
            if name.startswith("ISO media") and sample[4:8] != b"ftyp":
                continue
            return name
    return "unknown binary"


def describe_binary_file(sample: Optional[bytes], file_size: int) -> str:
    """One-line answer for the binary-file refusal.

    Naming the dead end: "Binary file" alone sends the model hunting for
    'appropriate tools' that may not exist in its toolset. Naming the TYPE
    ("PNG image data, 4.1 KB") answers what-is-this in a single read.
    """
    kind = identify_binary_bytes(sample or b"")
    if file_size >= 1024 * 1024:
        size = f"{file_size / (1024 * 1024):.1f} MB"
    elif file_size >= 1024:
        size = f"{file_size / 1024:.1f} KB"
    else:
        size = f"{file_size} bytes"
    return f"Binary file ({kind}, {size}) — cannot display as text."


# --- tools/file_operations.py:737-836
def _lint_json_inproc(content: str) -> tuple[bool, str]:
    """In-process JSON syntax check.  Returns (ok, error_message)."""
    import json as _json
    try:
        _json.loads(content)
        return True, ""
    except _json.JSONDecodeError as e:
        return False, f"JSONDecodeError: {e.msg} (line {e.lineno}, column {e.colno})"
    except Exception as e:  # noqa: BLE001 — any parse failure is a lint failure
        return False, f"{type(e).__name__}: {e}"


def _lint_yaml_inproc(content: str) -> tuple[bool, str]:
    """In-process YAML syntax check.  Returns (ok, error_message).

    Skipped gracefully if PyYAML isn't installed — YAML parsing is optional.

    Deliberately a *syntax-only* scan (``yaml.parse``), not ``safe_load``:
    loading rejects perfectly valid YAML that merely isn't a single plain
    document — multi-document streams (``---``-separated Kubernetes
    manifests raise ``ComposerError``) and application-defined tags
    (CloudFormation ``!Sub``/``!Ref``, Ansible ``!vault`` raise
    ``ConstructorError``).  Those are content conventions for whatever
    consumes the file, not syntax errors, and this linter's verdict is
    used as a fail-closed WRITE gate in ``write_file`` — a false positive
    here refuses a legitimate write outright.  ``yaml.parse`` still
    catches real scanner/parser failures (unclosed quotes, bad
    indentation, tab-mangled block maps).
    """
    try:
        import yaml as _yaml
    except ImportError:
        # PyYAML not available — skip silently, caller treats as no linter.
        return True, "__SKIP__"
    try:
        for _event in _yaml.parse(content):
            pass
        return True, ""
    except _yaml.YAMLError as e:
        return False, f"YAMLError: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _lint_toml_inproc(content: str) -> tuple[bool, str]:
    """In-process TOML syntax check (stdlib tomllib, Python 3.11+)."""
    import tomllib as _toml

    try:
        _toml.loads(content)
        return True, ""
    except Exception as e:  # tomllib raises TOMLDecodeError, a ValueError subclass
        return False, f"{type(e).__name__}: {e}"


def _lint_python_inproc(content: str) -> tuple[bool, str]:
    """In-process Python syntax check via ast.parse.

    Catches SyntaxError, IndentationError, and everything else the
    ast module rejects — matching py_compile's scope but with no
    subprocess overhead and no dependency on a ``python`` in PATH.
    """
    import ast as _ast
    try:
        _ast.parse(content)
        return True, ""
    except SyntaxError as e:
        loc = f" (line {e.lineno}, column {e.offset})" if e.lineno else ""
        return False, f"{type(e).__name__}: {e.msg}{loc}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


# In-process linters by file extension.  Preferred over shell linters when
# present — no subprocess overhead, microseconds per call.  Each callable
# takes file content (str) and returns (ok: bool, error: str).  An error
# string of ``"__SKIP__"`` signals the linter isn't available (missing
# dependency) and should be treated as "no linter".
LINTERS_INPROC = {
    '.py': _lint_python_inproc,
    '.json': _lint_json_inproc,
    '.yaml': _lint_yaml_inproc,
    '.yml': _lint_yaml_inproc,
    '.toml': _lint_toml_inproc,
}

# Subset of LINTERS_INPROC that the pre-write fail-closed gate in
# ``write_file`` (see below) refuses on, rather than merely reporting.
# Deliberately excludes ``.py``: unlike JSON/YAML/TOML (atomic structured
# data blobs where "doesn't parse" always means "corrupt"), ``.py`` is
# used throughout this codebase's own test fixtures as a generic
# stand-in extension for arbitrary non-Python text content (e.g.
# ``tests/tools/test_file_operations.py``'s
# ``TestPatchReplacePostWriteVerification`` writes "hello world" /
# "hi world" through a ``*.py`` path purely to exercise write-mechanics,
# not Python validity). Hard-refusing on invalid Python would treat that
# established, exercised pattern as an error and break it. Python source
# keeps the existing (unchanged) post-write lint-delta *report* — still
# visible to the caller, just not a write-blocking refusal.
_FAIL_CLOSED_INPROC_EXTS = frozenset({'.json', '.yaml', '.yml', '.toml'})

# --- tools/file_operations.py:1047-1079 (staticmethod, dedented)
def _is_likely_binary_bytes(sample: bytes) -> bool:
    """Byte-layer binary detection (the boundary for the #80308 class).

    Contract: a file is text when its sample is valid UTF-8, allowing one
    incomplete multibyte sequence at the very end (an artifact of cutting
    the sample at a byte boundary, not a property of the file). Anything
    else — NUL bytes, mid-stream invalid UTF-8 such as latin-1 or true
    binaries — stays read-only, preserving the anti-mojibake guarantee
    the old U+FFFD check existed for: a read→edit→write round-trip must
    never rewrite undecodable bytes with replacement characters.

    A file that legitimately *contains* U+FFFD (EF BF BD — e.g. logs of
    lossy output) is valid UTF-8 and reads as text; the old text-layer
    check misclassified it because it could not tell a stored replacement
    character from a transport-manufactured one.
    """
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    try:
        sample.decode("utf-8")
        return False
    except UnicodeDecodeError as exc:
        # UTF-8 sequences are at most 4 bytes: an error starting in the
        # last 3 bytes with a clean prefix is a boundary cut, not binary.
        if exc.start >= len(sample) - 3:
            try:
                sample[: exc.start].decode("utf-8")
                return False
            except UnicodeDecodeError:
                pass
        return True

# --- tools/file_tools.py:92-136
def _truncate_to_char_budget(content: str, max_chars: int) -> tuple[str, int, bool]:
    """Trim line-numbered ``read_file`` content to fit a char budget.

    Ported in spirit from nearai/ironclaw#5029 (dual line/byte cap on
    ``read_file``). Where hermes previously hard-rejected an oversized read
    (forcing the model to guess a smaller ``limit`` and burn a round-trip
    returning nothing), this trims the content to the last *complete line*
    that fits within ``max_chars`` and reports how many lines were kept so
    the caller can offer a ``next_offset`` continuation.

    ``content`` is the gutter-rendered text (``LINE_NUM|CONTENT`` joined by
    ``\\n``). Individual lines are already clamped to ``get_max_line_length()``
    upstream, so a single line never blows the whole budget on its own; the
    overflow this handles is the *accumulation* of many lines under the
    line-count limit (logs, wide CSV rows, minified data).

    Returns ``(kept_text, lines_kept, truncated)``. When ``content`` already
    fits, returns it unchanged with ``truncated=False``. If not even the
    first line fits, that single line is clamped on a code-point boundary
    (Python ``str`` slicing never splits a code point) so the read never
    returns empty and the cursor can still advance.
    """
    if len(content) <= max_chars:
        return content, (content.count("\n") + 1 if content else 0), False

    lines = content.split("\n")
    kept: list[str] = []
    running = 0
    for line in lines:
        # +1 for the "\n" that rejoins this line to the previous one.
        addition = len(line) + (1 if kept else 0)
        if running + addition > max_chars:
            break
        kept.append(line)
        running += addition

    if not kept:
        # First line alone exceeds the budget. Clamp on a code-point
        # boundary rather than emitting nothing.
        kept.append(lines[0][:max_chars])

    return "\n".join(kept), len(kept), True


# If the total file size exceeds this AND the caller didn't specify a narrow

# --- tools/file_tools.py:1195-1199
_READ_DEDUP_STATUS_MESSAGE = (
    "File unchanged since last read. The content from "
    "the earlier read_file result in this conversation is "
    "still current — refer to that instead of re-reading."
)

# --- tools/file_tools.py:1326-1397
def _is_internal_file_status_text(content: str) -> bool:
    """Return True when content looks like an internal file-tool status, not real file bytes.

    The read_file dedup status message must never be persisted as file
    content.  The obvious shape is the model echoing the message verbatim,
    but in practice it also wraps it with small framing text (a leading
    "Note:", a trailing newline + short comment, etc.) before calling
    write_file.  We treat any short-ish write whose body is dominated by
    the status message as the same class of corruption.

    Heuristic:
      * Strict equality (after strip) — the verbatim shape.
      * OR the stripped content contains the full status message AND is
        short enough that the status dominates it (<=2x the message length).
        Short, status-dominated writes can't plausibly be real files —
        legitimate docs/notes that happen to quote this internal message
        are always dramatically longer.
    """
    if not isinstance(content, str):
        return False
    stripped = content.strip()
    if not stripped:
        return False
    if stripped == _READ_DEDUP_STATUS_MESSAGE:
        return True
    if _READ_DEDUP_STATUS_MESSAGE in stripped and \
            len(stripped) <= 2 * len(_READ_DEDUP_STATUS_MESSAGE):
        return True
    return False


def _looks_like_read_file_line_numbered_content(content: str) -> bool:
    """Return True for content dominated by read_file's ``LINE_NUM|CONTENT`` display.

    ``read_file`` intentionally returns line-numbered text to the model. If
    that display format is echoed into ``write_file``, config/source files are
    silently corrupted with prefixes like `` 1|``.  We reject writes where the
    non-empty lines are mostly consecutive read_file-style numbered lines, while
    allowing sparse literal pipe content such as a single ``1|value`` line.
    """
    if not isinstance(content, str):
        return False

    lines = [line for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    numbered: list[int] = []
    for line in lines:
        stripped = line.lstrip()
        prefix, sep, _rest = stripped.partition("|")
        if sep and prefix.isdigit():
            numbered.append(int(prefix))

    if len(numbered) < 2:
        return False
    if len(numbered) / len(lines) < 0.6:
        return False

    consecutive_pairs = sum(
        1 for prev, current in zip(numbered, numbered[1:])
        if current == prev + 1
    )
    return consecutive_pairs >= len(numbered) - 1


def _is_internal_file_tool_content(content: str) -> bool:
    """Return True when content is file-tool display text, not intended file bytes."""
    return (
        _is_internal_file_status_text(content)
        or _looks_like_read_file_line_numbered_content(content)
    )
