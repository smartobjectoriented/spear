"""The skill library: frontmatter, applicability, and the file/index contract.

A skill is a markdown procedure the model writes after a completed task.  The
file on disk is the source of truth and the Chroma collection is a derived
index of it.  This module keeps the two honest about each other: a skill
written or edited by hand used to be listed by ``/skills`` and never injected,
because only ``save_skill`` ever wrote to the collection.

The frontmatter follows the ``SKILL.md`` convention -- the same small YAML
block Hermes Agent and agentskills.io use -- and the harness reads it so the
model does not have to: it is stripped before injection, so what enters the
prompt is the procedure and not its metadata.

Nothing here imports chromadb.  The collection is passed in, which is what
keeps this module importable by a test, by ``--help``, and by the CLI listing
without paying for the vector store.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime


_FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n?", re.S)

# The keys this harness acts on.  Anything else in a frontmatter block is
# preserved on disk when a human wrote it and ignored here: an unknown key is
# somebody else's convention, not an error.

_LIST_KEYS = ("requires", "tags")
ANY_SCOPE = "any"


class SkillError(ValueError):
    """A skill cannot be written as asked."""


@dataclass(frozen=True)
class Skill:
    name: str
    body: str
    description: str = ""
    scope: tuple[str, ...] = (ANY_SCOPE,)
    requires: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    version: int = 1
    created: str = ""
    updated: str = ""
    extra: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    declared: bool = False

    @property
    def summary(self) -> str:
        """One line for a listing: the description, or the body's first line."""

        if self.description:
            return self.description

        for line in self.body.splitlines():
            stripped = line.strip().lstrip("#").strip()

            if stripped:
                return stripped

        return ""

    @property
    def document(self) -> str:
        """The text that is embedded and injected -- never the frontmatter.

        Without a declared description this is byte for byte what the library
        indexed before frontmatter existed, so the skills already in the
        collection do not need re-embedding to keep matching.
        """

        if not self.description:
            return f"# Skill: {self.name}\n\n{self.body}"

        return f"# Skill: {self.name}\n\n{self.description}\n\n{self.body}"

    @property
    def digest(self) -> str:
        """Identity of what would be indexed, for reconciling file and index."""

        return hashlib.sha256(self.document.encode("utf-8")).hexdigest()[:32]


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", (name or "").lower()).strip("-")[:60]


def _split_values(raw: str) -> tuple[str, ...]:
    """A scalar, or a ``[a, b]`` / ``a, b`` list, as a tuple of strings."""

    inner = raw.strip()

    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]

    parts = [part.strip().strip("'\"") for part in inner.split(",")]

    return tuple(part for part in parts if part)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """The frontmatter mapping and the body below it.

    Deliberately not a YAML parser: it reads the flat ``key: value`` block
    this harness writes, and a block it cannot read is treated as ordinary
    markdown rather than as a failure.  A skill that will not parse must still
    be a skill you can read and inject; losing one to a stray colon would be a
    worse outcome than ignoring metadata nobody set.
    """

    match = _FRONTMATTER.match(text)

    if match is None:
        return {}, text

    fields: dict[str, str] = {}

    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        if line[:1] in (" ", "\t") or ":" not in line:
            # Nested YAML, a continuation, a line with no key: beyond what
            # this reader claims to understand.

            return {}, text

        key, _, value = line.partition(":")
        fields[key.strip().lower()] = value.strip()

    return fields, text[match.end():]


def parse_skill(name: str, text: str) -> Skill:
    """Read one skill file.  A file with no frontmatter is a valid skill."""

    fields, body = _parse_frontmatter(text)
    known = {"name", "description", "scope", "version", "created", "updated"}
    known.update(_LIST_KEYS)

    try:
        version = max(1, int(fields.get("version", "1")))
    except ValueError:
        version = 1

    scope = _split_values(fields.get("scope", "")) or (ANY_SCOPE,)

    return Skill(
        name=normalize_name(fields.get("name") or name),
        body=body.strip(),
        description=fields.get("description", "").strip().strip("'\""),
        scope=scope,
        requires=_split_values(fields.get("requires", "")),
        tags=_split_values(fields.get("tags", "")),
        version=version,
        created=fields.get("created", ""),
        updated=fields.get("updated", ""),
        extra=tuple((key, value) for key, value in fields.items()
                    if key not in known),
        declared=bool(fields),
    )


def render_skill(skill: Skill) -> str:
    """The file as it is written back: frontmatter, then the procedure."""

    lines = [f"name: {skill.name}"]

    if skill.description:
        lines.append(f"description: {skill.description}")

    lines.append(f"scope: [{', '.join(skill.scope)}]")

    for key, values in (("requires", skill.requires), ("tags", skill.tags)):
        if values:
            lines.append(f"{key}: [{', '.join(values)}]")

    lines.append(f"version: {skill.version}")

    for key, value in (("created", skill.created), ("updated", skill.updated)):
        if value:
            lines.append(f"{key}: {value}")

    lines.extend(f"{key}: {value}" for key, value in skill.extra)

    return "---\n" + "\n".join(lines) + "\n---\n\n" + skill.body.strip() + "\n"


def load_library(directory: str) -> list[Skill]:
    """Every skill on disk, by name.  A missing directory is an empty one."""

    skills = []

    for filename in sorted(os.listdir(directory)) if os.path.isdir(directory) else []:
        if not filename.endswith(".md"):
            continue

        path = os.path.join(directory, filename)

        try:
            with open(path, encoding="utf-8") as handle:
                skills.append(parse_skill(filename[:-3], handle.read()))
        except OSError:
            continue

    return skills


def missing_requirements(skill: Skill, *, which=shutil.which) -> tuple[str, ...]:
    """Declared commands this machine does not have.

    Checked on the host rather than inside the sandbox: the sandbox binds the
    same root, so a command absent here is absent there too, and asking the
    host costs nothing on a turn that is about to build a prompt.
    """

    return tuple(command for command in skill.requires if not which(command))


def applies_to(skill: Skill, *, project: str = "", kind: str = "",
               which=shutil.which) -> bool:
    """Whether this skill belongs in *this* session's prompt.

    Scope defaults to ``any``, so nothing that exists today narrows: a skill
    is withheld only where its own frontmatter says it does not apply, or
    where a command it declares is not installed.
    """

    if missing_requirements(skill, which=which):
        return False

    if ANY_SCOPE in skill.scope:
        return True

    return bool(set(skill.scope) & ({project, kind} - {""}))


def reconcile(skills, collection, *, project: str = "") -> dict[str, int]:
    """Make the index agree with the directory, and report what moved.

    Only what changed is embedded: the digest of the indexed document is kept
    beside it, so an unchanged library costs one metadata read.  Records with
    no file behind them are dropped -- an index that answers with a procedure
    the operator deleted is worse than one that answers with nothing.
    """

    by_name = {skill.name: skill for skill in skills}
    indexed: dict[str, str] = {}

    existing = collection.get(include=["metadatas"])

    for identifier, metadata in zip(existing.get("ids") or (),
                                    existing.get("metadatas") or ()):
        indexed[identifier] = (metadata or {}).get("digest", "")

    stale = [skill for skill in skills if indexed.get(skill.name) != skill.digest]
    orphaned = [identifier for identifier in indexed if identifier not in by_name]

    if stale:
        collection.upsert(
            ids=[skill.name for skill in stale],
            documents=[skill.document for skill in stale],
            metadatas=[{
                "name": skill.name, "project": project, "digest": skill.digest,
                "scope": ",".join(skill.scope), "requires": ",".join(skill.requires),
                "ts": datetime.now().isoformat(timespec="seconds"),
            } for skill in stale],
        )

    if orphaned:
        collection.delete(ids=orphaned)

    return {
        "indexed": len(stale),
        "removed": len(orphaned),
        "added": sum(1 for skill in stale if skill.name not in indexed),
    }


def save(directory: str, name: str, content: str, *, description: str = "",
         scope=(), requires=(), now=None) -> Skill:
    """Write a skill file, preserving what a previous version established.

    An overwrite keeps the original ``created`` stamp and bumps ``version``,
    so a procedure the model rewrote three times says so.  Scope and
    requirements survive an overwrite that does not restate them: the model
    rewriting a body should not silently widen a skill an operator narrowed.
    """

    name = normalize_name(name)

    if not name or not content.strip():
        raise SkillError("a skill needs a name and a content")

    path = os.path.join(directory, f"{name}.md")
    previous = None

    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            previous = parse_skill(name, handle.read())

    stamp = (now or datetime.now()).isoformat(timespec="seconds")
    skill = Skill(
        name=name,
        body=content.strip(),
        description=(description.strip() or (previous.description if previous else "")),
        scope=tuple(scope) or (previous.scope if previous else (ANY_SCOPE,)),
        requires=tuple(requires) or (previous.requires if previous else ()),
        tags=previous.tags if previous else (),
        version=(previous.version + 1) if previous else 1,
        created=(previous.created if previous and previous.created else stamp),
        updated=stamp,
        extra=previous.extra if previous else (),
        declared=True,
    )

    os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render_skill(skill))

    return skill
