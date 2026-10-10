"""The workspace a turn runs in, as one compact, explicit record.

Everything that decides which context and which tools a turn may see starts
from here: which project this is, whether it was registered, what it declares
about its build, which standards belong to it, and what it binds to them. It
references what the project declares; it embeds no document.

Identity is what keeps one project's material out of another. A registered
project is its registry name. An unregistered tree is ``adhoc:<realpath>`` --
never its basename, which two unrelated checkouts share, and never the name of
the project it happens to sit next to. An unregistered tree matches no
project-specific rule: it inherits nothing from the nearest, the previous or
any other project.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

#: Capability families (tool_selection) a workspace may be offered.
CODING, NORMATIVE, GENERAL = "CODING", "NORMATIVE", "GENERAL"

#: How a scope names a project family, beside the projects it names directly.
FAMILY_PREFIX = "family:"

_FAMILY_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]*")


def reserved_project_id(name: str) -> bool:
    """A registry name that would read as a family in a scope."""
    return str(name).casefold().startswith(FAMILY_PREFIX)


def project_families(spec) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(the families a registry entry declares, the values it got wrong).

    Membership is what the entry says in "families", and nothing else: no
    directory, build file or rule makes a project a member of anything.
    """
    raw = (spec or {}).get("families") or ()

    if isinstance(raw, str):
        raw = (raw,)

    good, bad = [], []

    for value in raw if isinstance(raw, (list, tuple)) else (raw,):
        name = str(value).strip().casefold()

        if isinstance(value, str) and _FAMILY_NAME.fullmatch(name):
            good.append(name)
        else:
            bad.append(str(value))

    return tuple(dict.fromkeys(good)), tuple(bad)


@dataclass(frozen=True)
class WorkspaceContext:
    workspace_id: str
    registered: bool
    root: str                       # where the tools run
    corpus_root: str                # the tree the project's corpus indexes
    kind: str = ""
    corpora: tuple[str, ...] = ()   # this project's own federated parts
    build_command: str = ""
    test_command: str = ""
    build_source: str = "none"      # projects.json, or how the tree was probed
    standards: tuple[tuple[str, str], ...] = ()      # (id, revision) the project declares
    bound_standard: tuple[str, str] | None = None    # the machine's binding, if engaged
    check_bindings: tuple[str, ...] = ()             # normative_checks ids
    applicability_declarations: int = 0
    families: tuple[str, ...] = field(default=(CODING, GENERAL))
    project_families: tuple[str, ...] = ()           # declared in projects.json

    @property
    def names(self) -> frozenset[str]:
        """What a project-scoped rule may name to apply here: the registered
        project, its own parts and the families it declares (as family:<name>).
        An unregistered tree answers to nothing."""
        if not self.registered:
            return frozenset()

        names = {self.workspace_id} | set(self.corpora)
        names |= {FAMILY_PREFIX + name for name in self.project_families}

        if self.workspace_id.startswith("workspace:"):
            names.add(self.workspace_id.split(":", 1)[1])

        return frozenset(names)

    def under(self, directory: str) -> bool:
        """Whether this workspace's trees lie under a host directory."""
        base = os.path.realpath(os.path.expanduser(directory)).rstrip("/")

        return any(root == base or root.startswith(base + "/")
                   for root in (self.root, self.corpus_root) if root)

    def to_dict(self) -> dict:
        return {"workspace_id": self.workspace_id, "registered": self.registered,
                "root": self.root, "corpus_root": self.corpus_root, "kind": self.kind,
                "corpora": list(self.corpora), "build_command": self.build_command,
                "test_command": self.test_command, "build_source": self.build_source,
                "standards": [list(item) for item in self.standards],
                "bound_standard": list(self.bound_standard) if self.bound_standard else None,
                "check_bindings": list(self.check_bindings),
                "applicability_declarations": self.applicability_declarations,
                "families": list(self.families),
                "project_families": list(self.project_families)}


def _declared_standards(spec) -> tuple[tuple[str, str], ...]:
    """The standards a project declares: its ``standards`` key, and every
    standard its conformance checks and applicability declarations name."""
    found = []

    for entry in (spec.get("standards") or ()):
        if isinstance(entry, dict) and entry.get("id"):
            found.append((str(entry["id"]), str(entry.get("revision") or "")))

    for key in ("normative_checks", "normative_applicability"):
        for entry in (spec.get(key) or ()):
            if isinstance(entry, dict) and entry.get("standard"):
                found.append((str(entry["standard"]), str(entry.get("revision") or "")))

    return tuple(dict.fromkeys(found))


def from_session(*, project: str, spec: dict | None, registered: bool, project_root: str,
                 corpus_root: str, project_commands=None, binding=None) -> WorkspaceContext:
    """The workspace of the current session, from what the client already knows."""
    spec = spec or {}
    root = os.path.realpath(project_root or ".")
    workspace_id = project if registered and project else f"adhoc:{root}"
    standards = _declared_standards(spec) if registered else ()
    bound = ((str(getattr(binding, "standard_id", "")), str(getattr(binding, "revision", "")))
             if binding is not None else None)
    families = (CODING, GENERAL) + ((NORMATIVE,) if bound or standards else ())

    return WorkspaceContext(
        workspace_id=workspace_id, registered=bool(registered), root=root,
        corpus_root=os.path.realpath(corpus_root or root), kind=str(spec.get("kind") or ""),
        corpora=tuple(str(item) for item in (spec.get("corpora") or ())) if registered else (),
        build_command=str(getattr(project_commands, "build", "") or ""),
        test_command=str(getattr(project_commands, "test", "") or ""),
        build_source=str(getattr(project_commands, "source", "none") or "none"),
        standards=standards, bound_standard=bound,
        check_bindings=tuple(str(entry.get("id")) for entry in (spec.get("normative_checks") or ())
                             if isinstance(entry, dict) and entry.get("id")) if registered else (),
        applicability_declarations=len(spec.get("normative_applicability") or ())
        if registered else 0,
        families=families,
        project_families=project_families(spec)[0] if registered else ())
