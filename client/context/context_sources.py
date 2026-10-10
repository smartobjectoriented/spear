"""The context a turn could be shown, as selectable candidates.

Each source is read into ``context_selection.Candidate`` objects that say what
they are, whose they are and what they concern. Nothing here decides; the
selector does.

Rules (``rules.d/*.md``) declare themselves in a header::

    ---
    scope: global                      every workspace -- the only way a rule is generic
    scope: corpus <name>[, <name>...]  a registered project, or one of its own parts
    scope: path <dir>                  workspaces whose tree lies under <dir>
    tasks: implementation, mixed, normative, general     (default: implementation, mixed)
    paths: doc/**, src/**              only when the request names such a path
    priority: 80
    id: doc-build
    ---

A rule without a header, or with a scope SPEAR cannot read, applies nowhere: a
rule is generic because it says so, never because nobody scoped it. A rule
written for one project's build system once reached every session on every
tree for exactly that reason.
"""

from __future__ import annotations

import os
import re

from context.context_selection import (GENERIC, HOST_PATHS, NOWHERE, PROJECTS, WORKSPACE,
                               Candidate, SourceType)
from context.workspace_context import FAMILY_PREFIX

_HEADER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.S)
_TASKS = frozenset({"implementation", "mixed", "normative", "general"})
NO_SCOPE = ("not declared generic or scoped to a project -- add `scope: global` "
            "or `scope: corpus <name>` to its header")


def _values(text: str) -> tuple[str, ...]:
    return tuple(item for item in re.split(r"[,\s]+", (text or "").strip()) if item)


def parse_rule(content: str):
    """(scope, tasks, paths, priority, rule id or "", body) of a rule file."""
    match = _HEADER.match(content)

    if not match:
        return (NOWHERE, NO_SCOPE), None, (), None, "", content.strip()

    fields = dict(re.findall(r"^(\w+):\s*(.*?)\s*$", match.group(1), re.M))
    body = content[match.end():].strip()
    kind, _, value = (fields.get("scope") or "").partition(" ")
    values = _values(value)

    if kind == "global" and not values:
        scope = (GENERIC,)
    elif kind == "corpus" and values:
        scope = (PROJECTS,) + values
    elif kind == "family" and values:
        scope = (PROJECTS,) + tuple(FAMILY_PREFIX + value.casefold() for value in values)
    elif kind == "path" and values:
        scope = (HOST_PATHS,) + values
    else:
        scope = (NOWHERE, "its scope header cannot be read")

    tasks = None

    if fields.get("tasks"):
        tasks = frozenset(item.lower() for item in _values(fields["tasks"]))

        if not tasks <= _TASKS:
            scope = (NOWHERE, f"unknown task classes: {', '.join(sorted(tasks - _TASKS))}")

    try:
        priority = int(fields["priority"]) if fields.get("priority") else None
    except ValueError:
        priority = None

    return scope, tasks, _values(fields.get("paths", "")), priority, fields.get("id", ""), body


def rule_candidates(rules_dir: str) -> list[Candidate]:
    """Every ``*.md`` of a rules directory, in file-name order."""
    found = []

    if not rules_dir or not os.path.isdir(rules_dir):
        return found

    for name in sorted(os.listdir(rules_dir)):
        path = os.path.join(rules_dir, name)

        if not name.endswith(".md") or not os.path.isfile(path):
            continue

        with open(path, "r", encoding="utf-8") as handle:
            scope, tasks, paths, priority, rule_id, body = parse_rule(handle.read().strip())

        if not body:
            continue

        title = re.sub(r"^\d+-", "", os.path.splitext(name)[0])
        found.append(Candidate(
            f"rule:{rule_id or title}", SourceType.PROJECT_RULE, f"rules.d/{name}",
            f"\n\n## Rule: {title}\n\n{body}", scope=scope, tasks=tasks, paths=paths,
            priority=priority, bucket="global_rules"))

    return found


def learned_candidate(path: str) -> list[Candidate]:
    """``/recall`` rules: generic by what the command means -- true everywhere."""
    if not path or not os.path.isfile(path):
        return []

    with open(path, "r", encoding="utf-8") as handle:
        body = handle.read().strip()

    if not body:
        return []

    return [Candidate("rule:learned", SourceType.PROJECT_RULE, os.path.basename(path),
                      f"\n\n## Rule: learned\n\n{body}", scope=(GENERIC,),
                      tasks=frozenset(_TASKS - {"normative"}), bucket="global_rules")]


def corpus_rule_candidates(parts) -> list[Candidate]:
    """A project's own maps: ``(label, text, scope)`` for the in-tree file of
    the workspace and of its corpus root, or the shipped map named after the
    project. Each says whose it is."""
    found = []

    for label, text, scope in parts:
        if text:
            found.append(Candidate(
                f"rule:corpus:{label}", SourceType.PROJECT_RULE, label,
                f"\n\n## Rule: corpus ({label})\n\n{text}", scope=scope,
                bucket="project_rules"))

    return found


def skill_candidates(skills) -> list[Candidate]:
    """``(name, text, scope names)`` for the skills a lookup retrieved. A
    skill applies where its scope names this project; ``any`` must be written
    to make it generic, and a skill that declares nothing applies nowhere."""
    found = []

    for name, text, names in skills:
        names = tuple(names or ())

        if not names:
            scope = (NOWHERE, NO_SCOPE)
        elif "any" in names:
            scope = (GENERIC,)
        else:
            scope = (PROJECTS,) + names

        found.append(Candidate(f"skill:{name}", SourceType.PROJECT_SKILL, f"skills/{name}",
                               text, scope=scope, bucket="skills"))

    return found


def metadata_candidates(workspace) -> list[Candidate]:
    """What the workspace declares about itself, compactly."""
    found = []

    if workspace.registered:
        found.append(Candidate(
            "project:identity", SourceType.PROJECT_METADATA, "projects.json",
            f"\n\n## Project\n\nWorkspace: {workspace.workspace_id}.",
            bucket="metadata"))

    commands = [f"build: `{workspace.build_command}`" if workspace.build_command else "",
                f"test: `{workspace.test_command}`" if workspace.test_command else ""]
    commands = [item for item in commands if item]

    if commands:
        found.append(Candidate(
            "project:build", SourceType.BUILD_METADATA, workspace.build_source,
            f"\n\nThe project's own verification ({workspace.build_source}): "
            + "; ".join(commands) + ".", bucket="metadata"))

    return found


def runtime_candidates(items) -> list[Candidate]:
    """The runtime's own items, ``(item id, source type, source, text,
    mandatory, bucket)``: all of them belong to the workspace by construction."""
    return [Candidate(item_id, source_type, source, text, scope=(WORKSPACE,),
                      mandatory=mandatory, bucket=bucket)
            for item_id, source_type, source, text, mandatory, bucket in items if text]
