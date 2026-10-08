"""Edits recovered from a reply's prose: code blocks and leaked tool calls."""

import os
import re
import json
from models.model_backend import ConversationMessage, TextBlock
from harness.tool_primitives import PathPolicyError
from runtime.agent_runtime import AgentRuntime
from cli.session_workspace import find_file, is_excluded_path, resolve_path
from cli.terminal_ui import Spinner
from cli.tool_routing import execute_tool


# Code-block fallback: code-tuned models (Qwen2.5-Coder) often print the whole
# improved file in a fenced ``` block instead of calling write_file. Detect
# that and apply it to the file the user named.

CODE_BLOCK_RE = re.compile(r"```([\w.+-]*)\n(.*?)```", re.DOTALL)
SRC_NAME_RE = re.compile(
    r"\b([\w./-]+\.(?:c|h|cpp|hpp|cc|cxx|S|s|py|sh|mk|cmake|md|rst|txt|dts|"
    r"dtsi|its|conf|cfg|ini|bb|bbappend|bbclass|inc|yaml|yml|json))\b")


def extract_code_block_with_language(text):
    """The largest fenced block, with the language the fence declared.

    The language is kept because it is the cheapest evidence of what the block
    IS, and the caller has to decide whether it belongs in the file it is about
    to overwrite. Returns ('', '') when there is no block.
    """
    blocks = [(m.group(2), m.group(1)) for m in CODE_BLOCK_RE.finditer(text or "")]

    if not blocks:
        return "", ""

    block, language = max(blocks, key=lambda item: len(item[0]))

    return block.rstrip("\n"), language.strip().lower()


def extract_code_block(text):
    """Return the largest fenced code block's content, or '' if none."""
    return extract_code_block_with_language(text)[0]


# What a fence language says the block is. A language absent from this table is
# not evidence of anything and never refuses on its own.

BLOCK_LANGUAGE_EXTENSIONS = {
    "c": {".c", ".h"}, "h": {".c", ".h"},
    "cpp": {".cpp", ".hpp", ".cc", ".cxx", ".h"},
    "c++": {".cpp", ".hpp", ".cc", ".cxx", ".h"},
    "python": {".py"}, "py": {".py"},
    "sh": {".sh"}, "bash": {".sh"}, "shell": {".sh"},
    "make": {".mk"}, "makefile": {".mk"}, "cmake": {".cmake"},
    "yaml": {".yaml", ".yml"}, "yml": {".yaml", ".yml"}, "json": {".json"},
    "rst": {".rst"}, "restructuredtext": {".rst"},
    "markdown": {".md"}, "md": {".md"},
    "dts": {".dts", ".dtsi"},
}

# A whole line of ==== / #### / ---- is a heading underline, and a line opening
# with `.. ` is a directive. Neither occurs in compilable C.

PROSE_MARKER_RE = re.compile(r"^\s*\.\.\s+\w|^[=#~^*+-]{4,}\s*$", re.M)
COMPILED_SOURCE_EXTENSIONS = {".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".s", ".S"}


def block_fits_target(block, language, target):
    """Could this block plausibly BE the target file?

    The fallback writes a printed block into a file named in the user's own
    sentence, and those two are not the same thing.  "add a chapter describing
    the ls.c application" names ``ls.c`` as the *subject*; the block the model
    printed was reStructuredText.  Applying it would have destroyed a source
    file in order to answer a documentation request — it was stopped only
    because ``write_file`` refuses to overwrite an existing file nobody read
    this turn.  A target that did not exist yet would simply have been created.

    So the fallback now has to believe the block belongs in that file.  The
    check refuses only on positive evidence of a mismatch, because this path
    exists for models that cannot call tools reliably and a false refusal costs
    them their edit.
    """
    extension = os.path.splitext(str(target))[1]
    declared = BLOCK_LANGUAGE_EXTENSIONS.get(language or "")

    if declared is not None and extension and extension not in declared:
        return False

    if extension in COMPILED_SOURCE_EXTENSIONS:
        if "#include" not in block and ";" not in block:
            return False

        if PROSE_MARKER_RE.search(block):
            return False

    return True


READ_CMD_RE = re.compile(
    r"\b(?:cat|head|tail|sed|bat|less|more|nl)\b[^|;&]*?"
    r"([\w./-]+\.[A-Za-z]\w*)")



def guess_target_file(user_input, tool_log=()):
    """The file the user means by e.g. 'improve ping.c'. A bare basename is
    ambiguous (several ping.c exist), so PREFER the file the model actually
    read this turn (cat/head/… in tool_log) whose basename matches; only fall
    back to a tree-wide find_file if nothing was read. NEVER return a path in
    an excluded third-party/snapshot tree (it would corrupt u-boot etc.)."""
    names = [m.group(1) for m in SRC_NAME_RE.finditer(user_input or "")]

    if not names:
        return None

    bnames = {os.path.basename(n) for n in names}

    # 1) a file the model read this turn

    for entry in tool_log:
        for m in READ_CMD_RE.finditer(entry):
            cand = m.group(1)

            if os.path.basename(cand) in bnames:
                try:
                    p = resolve_path(cand)
                except PathPolicyError:
                    continue

                if p.is_file() and not is_excluded_path(p):
                    return p

    # 2) fallback: tree-wide search, but skip third-party/snapshot trees

    for n in names:
        p = find_file(n)

        if p and not is_excluded_path(p):
            return p

    return None


# The <tool_call> wrapper is OPTIONAL: Qwen-Coder frequently emits a bare
# <function=NAME>…</function> (no wrapper), which would otherwise be dropped
# silently — the model then thinks it acted while the file is untouched.

LEAKED_CALL_RE = re.compile(
    r"(?:<tool_call>\s*)?<function=(\w+)>(.*?)</function>(?:\s*</tool_call>)?",
    re.DOTALL)
LEAKED_PARAM_RE = re.compile(
    r"<parameter=(\w+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def parse_leaked_tool_calls(text):
    """llama-server's XML tool-call parser occasionally leaks calls as raw
    text (multi-line parameters). Recover them client-side.
    Returns (cleaned_text, [(name, args), ...])."""
    calls = []

    for m in LEAKED_CALL_RE.finditer(text):
        name = m.group(1)
        args = {k: v for k, v in LEAKED_PARAM_RE.findall(m.group(2))}
        calls.append((name, args))

    if calls:
        text = LEAKED_CALL_RE.sub("", text).strip()

    # also recover bare-JSON tool calls (Qwen-style, no <tool_call> wrapper):
    #   {"name": "bash", "arguments": {"command": "..."}}

    text, json_calls = parse_json_tool_calls(text)

    return text, calls + json_calls


JSON_CALL_RE = re.compile(r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:')


def parse_json_tool_calls(text):
    """Recover tool calls the model prints as bare JSON text (Qwen/Coder
    format, no XML wrapper): {"name": "X", "arguments": {...}}. Brace-matches
    the full object so nested JSON (edit_file's escaped old/new text) parses.
    Returns (cleaned_text, [(name, args_dict), ...])."""
    calls, out, i = [], [], 0

    while True:
        m = JSON_CALL_RE.search(text, i)

        if not m:
            out.append(text[i:])
            break

        out.append(text[i:m.start()])

        # brace-match from the opening { to the matching }

        depth, j, instr, esc = 0, m.start(), False, False

        while j < len(text):
            c = text[j]

            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = not instr
            elif not instr:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1

                    if depth == 0:
                        j += 1
                        break

            j += 1

        blob = text[m.start():j]

        try:
            obj = json.loads(blob)
            name, args = obj.get("name"), obj.get("arguments")

            if name and isinstance(args, dict):
                calls.append((name, args))
            else:
                out.append(blob)
        except Exception:
            out.append(blob)

        i = j

    return "".join(out).strip(), calls


def try_surgical_edits(
    backend, system_prompt, conversation, rel, full_block, executed, tool_log,
    context_items=None, agent_context=None, runtime=None,
):
    """The model produced a WHOLE-FILE rewrite of an existing file. Give it ONE
    chance to express the same change as targeted edit_file calls instead — a
    full rewrite silently drops declarations / CLI options / third-party
    headers and stops compiling. Returns True iff >=1 surgical edit applied.
    Only edit_file/append_file count here; a write_file (whole file again)
    defeats the purpose and is ignored so the caller falls back to the block."""
    nudge = (
        "STOP. Do NOT rewrite the whole file. Your change must be expressed as "
        "one or more edit_file calls that touch ONLY the lines that actually "
        "change. Keep every existing declaration, global, helper function and "
        "CLI option, and do NOT alter the existing copyright header. For each "
        "change emit:\n"
        "<tool_call><function=edit_file><parameter=path>" + rel + "</parameter>"
        "<parameter=old_text>EXACT existing lines</parameter>"
        "<parameter=new_text>replacement</parameter></function></tool_call>\n"
        "Several small edits are better than one big one. Emit the edit_file "
        "call(s) now — no prose, no full-file code block.")
    msgs = conversation + [
        ConversationMessage("assistant", (TextBlock(full_block[:4000]),)),
        ConversationMessage("user", (TextBlock(nudge),)),
    ]

    try:
        with Spinner("Refining to a minimal edit…") as sp:
            turn = (runtime or AgentRuntime()).complete_model_turn(
                agent_context, conversation=msgs, use_tools=True,
                on_token=lambda: setattr(sp, "tokens", sp.tokens + 1),
            )
    except Exception:
        return False

    pending = [(call.name, dict(call.arguments)) for call in turn.tool_calls]
    applied = False

    for name, targs in pending:
        if name not in ("edit_file", "append_file"):
            continue            # ignore whole-file write_file here

        try:
            res = execute_tool(
                name, targs, executed, agent_context=agent_context,
            )
        except Exception as e:
            res = f"ERROR: tool '{name}' failed: {e}"

        tool_log.append(f"{name} {json.dumps(targs, ensure_ascii=False)[:200]}\n"
                        f"{res[:400]}")

        if isinstance(res, str) and res.startswith("OK"):
            applied = True

    return applied
