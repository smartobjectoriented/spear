"""Which context a turn is given, decided deterministically and recorded.

A registered project used to be context everywhere: every rule file without a
scope reached every session, a project's memories and procedures reached its
normative questions, and the code corpus was retrieved for a question about a
standard. This module decides, per request, what each runtime is shown:

  candidates   every piece of context that could be shown, each with what it
               is (SourceType), whose it is (its scope), which task classes it
               serves, which paths it concerns, its priority, and whether it
               is mandatory.
  select       keeps a candidate only when its type serves this phase, it
               belongs to this workspace or is explicitly generic, its task
               classes include this one, and its path scope matches a path the
               request names; then drops optional items, lowest priority
               first, until the selection fits the budget. Mandatory items are
               never dropped.

Every decision carries its reason, and the selection is an audit record, not
something the model sees. Nothing here calls a model or an embedder: the same
request in the same workspace selects the same context.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class SourceType(StrEnum):
    USER_REQUEST = "USER_REQUEST"
    SYSTEM_RUNTIME = "SYSTEM_RUNTIME"
    PROJECT_RULE = "PROJECT_RULE"
    PROJECT_METADATA = "PROJECT_METADATA"
    BUILD_METADATA = "BUILD_METADATA"
    PROJECT_MEMORY = "PROJECT_MEMORY"
    PROJECT_SKILL = "PROJECT_SKILL"
    STANDARD_BINDING = "STANDARD_BINDING"
    NORMATIVE_CONSTRAINT_SET = "NORMATIVE_CONSTRAINT_SET"
    RETRIEVED_STANDARD = "RETRIEVED_STANDARD"
    RETRIEVED_CORPUS = "RETRIEVED_CORPUS"
    SESSION_CONTEXT = "SESSION_CONTEXT"


#: The phases context is selected for. A MIXED request is selected three
#: times, once per pass; a MIXED question that asks for no change is read
#: against both sides.
IMPLEMENTATION = "IMPLEMENTATION"
GENERAL = "GENERAL"
NORMATIVE = "NORMATIVE"
MIXED_QUESTION = "MIXED_QUESTION"
MIXED_PREPASS = "MIXED_PREPASS"
MIXED_IMPLEMENTATION = "MIXED_IMPLEMENTATION"
MIXED_POSTCHECK = "MIXED_POSTCHECK"

_CODING_TYPES = frozenset({
    SourceType.USER_REQUEST, SourceType.SYSTEM_RUNTIME, SourceType.PROJECT_RULE,
    SourceType.PROJECT_METADATA, SourceType.BUILD_METADATA, SourceType.PROJECT_MEMORY,
    SourceType.PROJECT_SKILL, SourceType.SESSION_CONTEXT, SourceType.RETRIEVED_CORPUS,
    SourceType.STANDARD_BINDING})
_NORMATIVE_TYPES = frozenset({
    SourceType.USER_REQUEST, SourceType.SYSTEM_RUNTIME, SourceType.STANDARD_BINDING,
    SourceType.RETRIEVED_STANDARD, SourceType.SESSION_CONTEXT, SourceType.PROJECT_RULE})

#: What each phase may be shown, by kind of source.
POLICY = {
    IMPLEMENTATION: _CODING_TYPES,
    GENERAL: frozenset({SourceType.USER_REQUEST, SourceType.SYSTEM_RUNTIME,
                        SourceType.SESSION_CONTEXT, SourceType.PROJECT_RULE}),
    NORMATIVE: _NORMATIVE_TYPES,
    MIXED_QUESTION: _CODING_TYPES | _NORMATIVE_TYPES,
    MIXED_PREPASS: _NORMATIVE_TYPES,
    MIXED_IMPLEMENTATION: (_CODING_TYPES - {SourceType.STANDARD_BINDING})
    | {SourceType.NORMATIVE_CONSTRAINT_SET},
    MIXED_POSTCHECK: frozenset({SourceType.NORMATIVE_CONSTRAINT_SET}),
}

#: The task classes a rule names (its ``tasks:`` header) that each phase serves.
RULE_TASKS = {
    IMPLEMENTATION: {"implementation"},
    GENERAL: {"general"},
    NORMATIVE: {"normative"},
    MIXED_QUESTION: {"mixed"},
    MIXED_PREPASS: {"normative"},
    MIXED_IMPLEMENTATION: {"implementation", "mixed"},
    MIXED_POSTCHECK: set(),
}

#: A rule that names no task classes serves the change-making ones.
DEFAULT_RULE_TASKS = frozenset({"implementation", "mixed"})

#: Priority order (higher is kept longer): the request, hard runtime
#: constraints, the workspace, its rules, the constraint packet, supporting
#: material, then session context.
PRIORITY = {
    SourceType.USER_REQUEST: 100,
    SourceType.SYSTEM_RUNTIME: 95,
    SourceType.STANDARD_BINDING: 92,
    SourceType.PROJECT_METADATA: 90,
    SourceType.BUILD_METADATA: 88,
    SourceType.PROJECT_RULE: 80,
    SourceType.NORMATIVE_CONSTRAINT_SET: 75,
    SourceType.PROJECT_MEMORY: 60,
    SourceType.PROJECT_SKILL: 55,
    SourceType.RETRIEVED_STANDARD: 50,
    SourceType.RETRIEVED_CORPUS: 45,
    SourceType.SESSION_CONTEXT: 30,
}

# Scopes. A candidate belongs to the workspace it was read from, to a named
# project, to trees under a host directory, or to every workspace -- the last
# only when declared so. Anything else applies nowhere.
WORKSPACE, PROJECTS, HOST_PATHS, GENERIC, NOWHERE = (
    "workspace", "projects", "host_paths", "generic", "nowhere")


@dataclass(frozen=True)
class Candidate:
    item_id: str
    source_type: SourceType
    source_id: str                  # the file, store or record it comes from
    text: str
    scope: tuple = (WORKSPACE,)     # (kind, values...)
    tasks: frozenset[str] | None = None   # rule task classes; None for non-rules
    paths: tuple[str, ...] = ()     # path scope inside the workspace
    priority: int | None = None
    mandatory: bool = False
    bucket: str = ""                # where a renderer puts it
    rendered_by_runtime: bool = False     # recorded, injected by its own runtime

    @property
    def rank(self) -> int:
        return self.priority if self.priority is not None else PRIORITY[self.source_type]


@dataclass(frozen=True)
class Decision:
    item_id: str
    source_type: str
    source_id: str
    workspace: str
    reason: str
    priority: int
    chars: int
    tokens: int
    selected: bool

    def to_dict(self) -> dict:
        return {"id": self.item_id, "type": self.source_type, "source": self.source_id,
                "workspace": self.workspace, "reason": self.reason,
                "priority": self.priority, "chars": self.chars, "tokens": self.tokens}


@dataclass(frozen=True)
class Budget:
    window: int
    output_reservation: int
    mandatory_tokens: int

    @property
    def available(self) -> int:
        return max(0, self.window - self.output_reservation - self.mandatory_tokens)


@dataclass
class ContextSelection:
    phase: str
    workspace: str
    selected: list = field(default_factory=list)     # Candidate, in candidate order
    decisions: list = field(default_factory=list)    # Decision, selected and rejected
    budget: Budget | None = None
    dropped_for_budget: list = field(default_factory=list)

    def text(self, bucket: str) -> str:
        return "".join(item.text for item in self.selected if item.bucket == bucket)

    def ids(self) -> list[str]:
        return [item.item_id for item in self.selected]

    @property
    def rejected(self):
        return [item for item in self.decisions if not item.selected]


class ContextSelector(Protocol):
    def select(self, workspace, phase: str, candidates, *, request: str = "",
               budget: Budget | None = None) -> ContextSelection: ...


CODING_PHASES = frozenset({IMPLEMENTATION, GENERAL, MIXED_IMPLEMENTATION})


def primary_phase(scope: str, *, bound: bool) -> str:
    """The phase a request is selected for, from its class (answer_scope) and
    whether a standard is engaged. Without one, every request runs on the
    coding core, and a request that names the document but no binding is a
    change like any other."""
    if not bound:
        return {"GENERAL": GENERAL, "NORMATIVE": NORMATIVE}.get(scope, IMPLEMENTATION)

    return {"IMPLEMENTATION": IMPLEMENTATION, "MIXED": MIXED_QUESTION}.get(scope, NORMATIVE)


def phases(scope: str, *, bound: bool, write: bool) -> tuple[str, ...]:
    """Every phase a request may run: its own, and for a MIXED change the
    pre-pass and the implementation the orchestration runs instead."""
    first = primary_phase(scope, bound=bound)

    if first == MIXED_QUESTION and write:
        return (first, MIXED_PREPASS, MIXED_IMPLEMENTATION)

    return (first,)


def runs_on_coding_core(phase: str, *, bound: bool) -> bool:
    return phase == MIXED_IMPLEMENTATION or (not bound and phase != MIXED_PREPASS)


@dataclass(frozen=True)
class StandardChoice:
    standard: tuple[str, str] | None
    reason: str
    ambiguous: bool = False


def _naming_terms(standard_id: str) -> set[str]:
    """How a request may name a standard: its identifier, and the parts of it
    that are words rather than numbers -- never a revision, which a clause
    number can spell by accident."""
    value = (standard_id or "").strip().lower()
    parts = [part for part in re.split(r"[-_/\s]+", value) if part]
    terms = {value} | {part for part in parts
                       if re.search(r"[a-z]", part) and len(part) > 3}

    for first, second in zip(parts, parts[1:]):
        terms |= {f"{first}-{second}", f"{first} {second}"}

    return terms


def select_standard(workspace, request: str, *, phase: str) -> StandardChoice:
    """Which standard a request concerns, by what is written down only: a
    standard the request names, then the established binding, then the one
    standard the project declares. Two named standards are ambiguous, and so
    is a request that names none in a project declaring several with nothing
    bound. Selection never rebinds: the binding is the operator's."""
    if phase not in (NORMATIVE, MIXED_QUESTION, MIXED_PREPASS, MIXED_IMPLEMENTATION,
                     IMPLEMENTATION):
        return StandardChoice(None, "this phase uses no standard")

    known = list(dict.fromkeys(workspace.standards
                               + ((workspace.bound_standard,) if workspace.bound_standard
                                  else ())))
    terms = {item: _naming_terms(item[0]) for item in known}
    shared = {term for item in known for term in terms[item]
              if sum(term in terms[other] for other in known) > 1}
    text = (request or "").lower()
    named = list(dict.fromkeys(
        item[0] for item in known
        if any(re.search(rf"(?<![\w.]){re.escape(term)}(?![\w])", text)
               for term in terms[item] - shared)))

    if len(named) > 1:
        return StandardChoice(None, f"the request names {', '.join(named)}", ambiguous=True)

    if named:
        choice = next(item for item in known if item[0] == named[0])
        return StandardChoice(choice, f"the request names {choice[0]}")

    if workspace.bound_standard:
        return StandardChoice(workspace.bound_standard, "the established binding")

    declared = list(dict.fromkeys(item[0] for item in workspace.standards))

    if len(declared) == 1:
        return StandardChoice(workspace.standards[0], "the only standard the project declares")

    if declared:
        return StandardChoice(None, f"the project declares {', '.join(declared)} and none "
                                    f"is named or bound", ambiguous=True)

    return StandardChoice(None, "no standard is engaged or declared")


def estimate_tokens(text: str) -> int:
    """The context engine's own conservative estimate."""
    from context_engine import ApproximateTokenEstimator

    return ApproximateTokenEstimator().estimate(text or "")


_PATH = re.compile(r"(?<![\w.])(?:\./)?((?:[\w.-]+/)+[\w.*-]*|[\w-]{2,}\.[A-Za-z][\w]{0,8})"
                   r"(?![\w/])")


def request_paths(request: str) -> tuple[str, ...]:
    """The workspace paths a request names: anything with a slash, or a file
    name with an extension. Read off the request's words, nothing guessed."""
    return tuple(dict.fromkeys(match.group(1) for match in _PATH.finditer(request or "")))


def _glob(pattern: str, path: str) -> bool:
    pattern = pattern.strip().lstrip("./")

    if pattern.endswith("/**"):
        base = pattern[:-3]
        return path == base or path.startswith(base + "/")

    return fnmatch.fnmatch(path, pattern.replace("**", "*"))


class DeterministicContextSelector:
    """The Phase-1 strategy: rules on declared metadata, nothing learned."""

    def select(self, workspace, phase: str, candidates, *, request: str = "",
               budget: Budget | None = None) -> ContextSelection:
        allowed = POLICY[phase]
        named = request_paths(request)
        selection = ContextSelection(phase, workspace.workspace_id, budget=budget)
        kept = []

        for item in candidates:
            reason = self._refusal(item, workspace, phase, allowed, named)

            if reason:
                selection.decisions.append(self._decision(item, workspace, reason, False))
            else:
                kept.append(item)

        if budget is not None:
            kept = self._fit(kept, budget, selection, workspace)

        # Rendered in the order the candidates were given -- rule files in
        # file-name order, as they always were; priority decides only what a
        # tight budget drops.

        for item in kept:
            selection.selected.append(item)
            selection.decisions.append(self._decision(item, workspace,
                                                      self._grounds(item, workspace), True))

        return selection

    # ----------------------------------------------------------------- rules

    @staticmethod
    def _refusal(item, workspace, phase, allowed, named) -> str:
        if item.source_type not in allowed:
            return f"wrong task class: {phase} is not given {item.source_type}"

        kind = item.scope[0] if item.scope else NOWHERE

        if kind == NOWHERE:
            return item.scope[1] if len(item.scope) > 1 else "applies nowhere"

        if kind == PROJECTS and not set(item.scope[1:]) & workspace.names:
            return (f"wrong workspace: scoped to {', '.join(item.scope[1:])}, "
                    f"this is {workspace.workspace_id}")

        if kind == HOST_PATHS and not any(workspace.under(path) for path in item.scope[1:]):
            return f"wrong workspace: scoped to trees under {', '.join(item.scope[1:])}"

        if item.source_type == SourceType.PROJECT_RULE:
            tasks = item.tasks if item.tasks is not None else DEFAULT_RULE_TASKS

            if not tasks & RULE_TASKS[phase]:
                return f"wrong task class: serves {', '.join(sorted(tasks))}, not {phase}"

        if item.paths and not any(_glob(pattern, path) for pattern in item.paths
                                  for path in named):
            return (f"path scope mismatch: concerns {', '.join(item.paths)}; the request "
                    f"names {', '.join(named) or 'no path'}")

        if item.source_type == SourceType.STANDARD_BINDING and workspace.bound_standard is None:
            return "wrong standard: no standard is engaged"

        return ""

    @staticmethod
    def _grounds(item, workspace) -> str:
        kind = item.scope[0] if item.scope else NOWHERE

        if kind == GENERIC:
            why = "declared generic"
        elif kind == PROJECTS:
            why = f"scoped to {', '.join(sorted(set(item.scope[1:]) & workspace.names))}"
        elif kind == HOST_PATHS:
            why = "scoped to this tree's location"
        else:
            why = "belongs to this workspace"

        if item.paths:
            why += f"; the request names a path under {', '.join(item.paths)}"

        return why + ("; mandatory" if item.mandatory else "")

    @staticmethod
    def _decision(item, workspace, reason, selected):
        return Decision(item.item_id, str(item.source_type), item.source_id,
                        workspace.workspace_id, reason, item.rank, len(item.text),
                        estimate_tokens(item.text), selected)

    def _fit(self, kept, budget, selection, workspace):
        """Drop optional items, lowest priority first (the larger first on a
        tie), until the optional remainder fits what the budget leaves."""
        optional = [item for item in kept if not item.mandatory]
        total = sum(estimate_tokens(item.text) for item in optional)
        drop = set()

        for item in sorted(optional, key=lambda item: (item.rank, -len(item.text))):
            if total <= budget.available:
                break

            drop.add(item.item_id)
            total -= estimate_tokens(item.text)
            selection.dropped_for_budget.append(item.item_id)
            selection.decisions.append(self._decision(
                item, workspace, f"budget: {budget.available} tokens available after "
                                 f"the mandatory context and the output reservation", False))

        return [item for item in kept if item.item_id not in drop]
