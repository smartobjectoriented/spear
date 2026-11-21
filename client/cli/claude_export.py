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

What a SPEAR turn is shown beside rules.d goes along, each as a rule of its
own, so that both assistants start from the same text:

    spear.learned.md          the rules learned on this machine
    spear.corpus-map.md       the corpus's map, the tree's own or the shipped one
    spear.project.md          what the project declares: identity, build, test
    spear.knowledge-base.md   what reaches a request by itself, and the tools
"""

from __future__ import annotations

import io
import os
import re
import sys
import tarfile
import time

RULE_PREFIX = "spear-"
CORPUS_RULES_FILE = ".edgem-rules.md"
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


def export_session(candidates):
    """{path: text} for the candidates a turn gets beside rules.d, as
    context_sources builds them: (file name, candidates) pairs."""
    files = {}

    for name, found in candidates:
        text = "".join(item.text for item in found).strip()

        if text:
            files[f".claude/rules/{name}"] = text + "\n"

    return files


def corpus_map(tree, root, corpus, shipped):
    """(label, text) of the corpus's map as a turn finds it: the file the tree
    carries, at the working directory then at the corpus root, else the one
    shipped under the corpus's name."""
    seen = set()

    for where in (tree, root):
        path = os.path.realpath(os.path.join(where, CORPUS_RULES_FILE)) if where else ""

        if not path or path in seen or not os.path.isfile(path):
            continue

        seen.add(path)

        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read().strip()

        if text:
            return os.path.basename(os.path.realpath(where)), text

    path = os.path.join(shipped, f"{corpus}.md") if corpus else ""

    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read().strip()

        if text:
            return f"{corpus}, shipped", text

    return "", ""


def knowledge_base(corpus):
    """The rule that says what this session shares with SPEAR and how it
    arrives."""
    tree = f"the corpus `{corpus}`" if corpus else "this tree"

    return (
        "# Shared knowledge base\n\n"
        f"This session works on {tree} with the knowledge base SPEAR uses for it: "
        "the rules beside this one, the skills, the corpus index and the knowledge "
        "recorded for the workspace.\n\n"
        "- A request arrives with a `Retrieved Context` section when the corpus is "
        "indexed: the passages that match it, each headed by its path relative to "
        "the corpus root. Start from them; open the files they name rather than "
        "searching the tree again.\n"
        "- `search_corpus` (MCP server `spear-corpus`) searches the same index. Use "
        "it for what the retrieved context does not cover, before grep or a walk "
        "through the tree; `list_corpora` names what is indexed.\n"
        "- A `Workspace knowledge` section carries what has been recorded about "
        "this workspace. `spear-knowledge list` and `spear-knowledge show <id>` "
        "print the records; `spear-knowledge propose '<json>'` suggests one, with "
        "`kind`, `subject` and `statement`, which is used only once the operator "
        "accepts it.\n")


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
    tree = _option(argv, "--tree") or os.getcwd()

    if "-h" in argv or "--help" in argv:
        print("usage: spear-claude-export [--corpus NAME] [--tree DIR] [--out DIR]",
              file=sys.stderr)

        return 2

    from cli import corpus_registry
    from cli.chat_settings import LEARNED_RULES_FILE, RULES_DIR, STATE_DIR, resource_dir
    from context import context_sources, skill_library, workspace_context
    from context.workspace_context import project_families
    from evidence import project_build

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

    # The rest of what a turn starts from, read as a turn reads it.

    spec = projects.get(corpus) or {}
    root = os.path.realpath(spec["path"]) if corpus else ""
    commands = project_build.commands(
        tree, spec=spec, cache_dir=f"{STATE_DIR}/projects/{corpus.replace('/', '_')}"
    ) if corpus else None
    here = workspace_context.from_session(
        project=corpus or "", spec=spec, registered=bool(corpus), project_root=tree,
        corpus_root=root or tree, project_commands=commands)
    label, text = corpus_map(tree, root, corpus, os.path.join(RULES_DIR, "corpora"))
    files.update(export_session((
        ("spear.learned.md", context_sources.learned_candidate(LEARNED_RULES_FILE)),
        ("spear.corpus-map.md", context_sources.corpus_rule_candidates(
            [(label, text, ())] if text else [])),
        ("spear.project.md", context_sources.metadata_candidates(here)))))
    files[".claude/rules/spear.knowledge-base.md"] = knowledge_base(corpus)

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
