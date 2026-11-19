"""The rules and skills of this install, laid out for Claude Code.

SPEAR injects a corpus's rules on every request and offers it the skills whose
scope matches. Claude Code has the same two notions under other names, so a
project tree can carry them without SPEAR running at all:

    .claude/rules/spear-<rule>.md        loaded with the project's memory
    .claude/skills/<skill>/SKILL.md      offered by description

The export is a tar on stdout unless a directory is named, so it crosses the
same channel as the MCP server does:

    ssh host docker run --rm --entrypoint spear-claude-export IMAGE \\
        --corpus NAME | tar -x -C <project tree>

Only what applies to the named corpus is written; without --corpus, only what
applies to every tree. A rule scoped to a path is
left out: the tree it names is one of this machine, not of the reader's.
"""

from __future__ import annotations

import io
import os
import re
import sys
import tarfile
import time

RULE_PREFIX = "spear-"
DESCRIPTION_MAX = 300


def corpus_names(projects, corpus, families=()):
    """What a scope is matched against: the corpus, the ones it federates and
    the families its registry entry declares."""
    spec = projects.get(corpus) or {}

    return ({corpus} | set(spec.get("corpora") or ())
            | {f"family:{name}" for name in families})


def export_rules(rules, names):
    """{path: text} for the rules.d files (`rules`: {file name: text}).

    A rule's `paths` header becomes Claude Code's own: the rule is then read
    only when a file under one of them is.
    """
    from context.context_selection import GENERIC, PROJECTS
    from context.context_sources import parse_rule

    files = {}

    for fname in sorted(rules):
        if not fname.endswith(".md"):
            continue

        scope, _, paths, _, _, body = parse_rule(rules[fname].strip())

        if not body:
            continue

        if scope[0] != GENERIC and not (scope[0] == PROJECTS and names & set(scope[1:])):
            continue

        title = re.sub(r"^\d+-", "", os.path.splitext(fname)[0])
        header = ""

        if paths:
            listed = "".join(f'  - "{path}"\n' for path in paths)
            header = f"---\npaths:\n{listed}---\n\n"

        files[f".claude/rules/{RULE_PREFIX}{title}.md"] = f"{header}# {title}\n\n{body}\n"

    return files


def export_skills(skills, names):
    """{path: text} for the skills whose stated scope covers the corpus."""
    files = {}

    for skill in skills:
        if "any" not in skill.scope and not names & set(skill.scope):
            continue

        description = " ".join(skill.summary.split())

        if len(description) > DESCRIPTION_MAX:
            description = description[:DESCRIPTION_MAX - 1].rstrip() + "…"

        # A description is YAML: quoted, so a colon in it stays text.

        quoted = description.replace("\\", "\\\\").replace('"', '\\"')
        files[f".claude/skills/{skill.name}/SKILL.md"] = (
            f"---\nname: {skill.name}\ndescription: \"{quoted}\"\n---\n\n"
            f"{skill.body.strip()}\n")

    return files


def write_tar(files, out):
    now = time.time()

    with tarfile.open(fileobj=out, mode="w|") as archive:
        for path in sorted(files):
            data = files[path].encode("utf-8")
            info = tarfile.TarInfo(path)
            info.size, info.mtime, info.mode = len(data), now, 0o644
            archive.addfile(info, io.BytesIO(data))


def write_tree(files, directory):
    for path in sorted(files):
        target = os.path.join(directory, path)
        os.makedirs(os.path.dirname(target), exist_ok=True)

        with open(target, "w", encoding="utf-8") as handle:
            handle.write(files[path])


def _option(argv, flag):
    if flag not in argv:
        return None

    at = argv.index(flag)

    if at + 1 >= len(argv):
        raise SystemExit(f"spear-claude-export: {flag} needs a value")

    return argv[at + 1]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    corpus, directory = _option(argv, "--corpus"), _option(argv, "--out")

    if "-h" in argv or "--help" in argv:
        print("usage: spear-claude-export [--corpus NAME] [--out DIR]", file=sys.stderr)

        return 2

    from cli import corpus_registry
    from cli.chat_settings import RULES_DIR, resource_dir
    from context import skill_library
    from context.workspace_context import project_families

    projects = corpus_registry.load_projects()

    if corpus and corpus not in projects:
        print(f"spear-claude-export: no corpus named '{corpus}' "
              f"(known: {', '.join(sorted(projects)) or 'none'})", file=sys.stderr)

        return 2

    # No corpus named: a tree this install does not know, which gets what
    # applies everywhere and nothing else.

    names = set()

    if corpus:
        names = corpus_names(projects, corpus, project_families(projects[corpus])[0])

    rules = {}

    if os.path.isdir(RULES_DIR):
        for fname in os.listdir(RULES_DIR):
            path = os.path.join(RULES_DIR, fname)

            if fname.endswith(".md") and os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as handle:
                    rules[fname] = handle.read()

    skills = skill_library.load_library(resource_dir("SPEAR_SKILLS_DIR", "skills"))
    files = {**export_rules(rules, names), **export_skills(skills, names)}

    if directory:
        write_tree(files, directory)
    else:
        write_tar(files, sys.stdout.buffer)

    print(f"spear-claude-export: {len(files)} files for "
          + (f"'{corpus}'" if corpus else "any tree"), file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    sys.exit(main())
