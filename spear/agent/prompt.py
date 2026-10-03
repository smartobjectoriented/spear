"""The coding core's system prompt, assembled as Hermes Agent assembles it.

agent/system_prompt.py (0cbc6e37) builds one system message for a Qwen model
on a coding workspace in this order: identity; task completion; parallel
tool calls; memory and skills guidance; tool-use enforcement and execution
guidance (both for "qwen" model names); environment hints; the coding block
with its edit-format line and workspace snapshot; the Python toolchain
probe; context files; the conversation/model lines. The text is
hermes.prompt_text, copied.

Deviations, each because the part does not exist in SPEAR: the identity's
first sentence names SPEAR instead of Hermes Agent and its documentation
pointer is omitted; the memory, skills, steering, profile and platform
paragraphs are omitted (no memory or skills tools, no mid-turn steering, no
profiles; SPEAR renders Markdown). SPEAR's project rules take the place of
Hermes' context files, under Hermes' heading.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime

from .hermes import prompt_text as text
from .tools import workspace_block

IDENTITY = text.DEFAULT_AGENT_IDENTITY.replace(
    "You are Hermes Agent, built by Nous Research.", "You are SPEAR, a coding agent.", 1)


def _python_probe() -> str:
    """agent/system_prompt.py's local Python toolchain line."""
    python = shutil.which("python3")

    if not python:
        return ""

    try:
        version = subprocess.run([python, "-c", "import platform;print(platform.python_version())"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""

    import sysconfig

    managed = os.path.exists(os.path.join(sysconfig.get_paths()["stdlib"], "EXTERNALLY-MANAGED"))

    return (f"Python toolchain: python3={version}, PEP 668="
            f"{'yes (use venv or uv)' if managed else 'no'}.")


def coding_block(tool_names) -> str:
    """CODING_AGENT_GUIDANCE as system_prompt_parts renders it for Qwen."""
    brief = text.CODING_AGENT_GUIDANCE

    if "todo_list" not in tool_names:
        brief = brief.replace(
            "- Track multi-step work with `todo_list`. Reference code as "
            "`path:line` instead of pasting whole files.",
            "- Reference code as `path:line` instead of pasting whole files.")

    return f"{brief}\n{text.EDIT_FORMAT_LINE_QWEN}"


def build(*, cwd: str, tool_names, model: str, project_rules: str = "",
          now: datetime | None = None) -> str:
    """The one system message of a coding turn."""
    now = now or datetime.now().astimezone()
    parts = [IDENTITY, text.TASK_COMPLETION_GUIDANCE, text.PARALLEL_TOOL_CALL_GUIDANCE,
             text.TOOL_USE_ENFORCEMENT_GUIDANCE,
             text.execution_guidance_text(set(tool_names)),
             f"Host: {platform.system()} ({platform.release()})\n"
             f"User home directory: {os.path.expanduser('~')}\n"
             f"Current working directory: {cwd}",
             coding_block(tool_names)]

    workspace = workspace_block(cwd)

    if workspace:
        parts.append(workspace)

    probe = _python_probe()

    if probe:
        parts.append(probe)

    if project_rules.strip():
        parts.append("# Project Context\n\nThe following project context files "
                     "have been loaded and should be followed:\n\n" + project_rules.strip())

    parts.append(f"Conversation started: {now.strftime('%A, %B %d, %Y')}\nModel: {model}")

    return "\n\n".join(part.strip() for part in parts if part.strip())
