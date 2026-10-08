"""Which family of tools a turn is offered, decided at the family level.

  CODING      the coding core's surface -- read_file, search_files, patch,
              write_file, delete_file, terminal -- exactly, never a subset
              chosen per request.
  NORMATIVE   the normative runtime's tools for a bound standard.
  GENERAL     a request that names no subject: served by the same surface
              as the runtime it runs on, with no family added for it.

The members of a family are what the established exposure policy selects;
this module names the family, checks that every tool in the view can be
executed with every argument it declares, and records the decision. The
MIXED post-check is offered no tool at all: its evidence providers belong to
the orchestrator. Web, MCP and other external families are not part of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from context.context_selection import (GENERAL, IMPLEMENTATION, MIXED_IMPLEMENTATION,
                               MIXED_POSTCHECK, MIXED_PREPASS, MIXED_QUESTION, NORMATIVE)

CODING_FAMILY, NORMATIVE_FAMILY, GENERAL_FAMILY = "CODING", "NORMATIVE", "GENERAL"
EXTERNAL_FAMILY = "EXTERNAL"

#: The phases an external family may join: those that run on the coding core.
#: Nothing normative, and never the orchestrator's check.
EXTERNAL_PHASES = frozenset({IMPLEMENTATION, GENERAL, MIXED_IMPLEMENTATION})
EVIDENCE_PROVIDERS = "ORCHESTRATOR_EVIDENCE_PROVIDERS"

CODING_TOOLS = ("read_file", "search_files", "patch", "write_file", "delete_file", "terminal")


class ToolContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolSelection:
    phase: str
    family: str
    tools: tuple[str, ...]
    reason: str
    #: Every family the turn is offered: its own, and EXTERNAL when the
    #: workspace admits external capabilities for this phase. Those are
    #: reached through the control plane's gateway, never as extra tools.
    families: tuple[str, ...] = ()
    external: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"phase": self.phase, "family": self.family, "tools": list(self.tools),
                "reason": self.reason, "families": list(self.families or (self.family,)),
                "external_providers": list(self.external)}


class ToolSelector(Protocol):
    def select(self, phase: str, view_names, *, coding: bool,
               external=()) -> ToolSelection: ...


class DeterministicToolSelector:
    """The family follows the phase; the members follow the exposure policy."""

    def select(self, phase: str, view_names, *, coding: bool, external=()) -> ToolSelection:
        selection = self._own(phase, tuple(view_names), coding=coding)
        external = tuple(external) if phase in EXTERNAL_PHASES and coding else ()

        return ToolSelection(selection.phase, selection.family, selection.tools,
                             selection.reason,
                             (selection.family,) + ((EXTERNAL_FAMILY,) if external else ()),
                             external)

    def _own(self, phase: str, names, *, coding: bool) -> ToolSelection:

        if phase == MIXED_POSTCHECK:
            return ToolSelection(phase, EVIDENCE_PROVIDERS, (),
                                 "tool-less adjudication; the evidence providers belong to "
                                 "the orchestrator")

        if coding:
            stray = set(names) - set(CODING_TOOLS)

            if stray:
                raise ToolContractError(f"{phase}: tools outside the coding family: "
                                        + ", ".join(sorted(stray)))

            family = GENERAL_FAMILY if phase == GENERAL else CODING_FAMILY
            why = ("a request that names no subject, served by the coding core's surface"
                   if phase == GENERAL else "a change to the working tree: the coding core")

            return ToolSelection(phase, family, names, why)

        if phase in (NORMATIVE, MIXED_PREPASS):
            return ToolSelection(phase, NORMATIVE_FAMILY, names,
                                 "a question about the bound standard: the normative runtime")

        family = NORMATIVE_FAMILY if phase == MIXED_QUESTION else GENERAL_FAMILY
        why = ("both sides of a comparison: the normative runtime with its reading tools"
               if phase == MIXED_QUESTION else
               "a turn in a standard-bound session: the normative runtime")

        return ToolSelection(phase, family, names, why)


def check_contract(definitions, registry=None, *, coding: bool) -> None:
    """Every visible tool has a handler; every argument it declares is read.

    The coding view is held to the core's own contract. A normative view is
    held to the registry: a tool it does not hold is a tool that would be
    advertised and fail.
    """
    if coding:
        from agent import dispatch

        try:
            dispatch.check_contract([
                {"type": "function", "function": {"name": item.name,
                                                  "parameters": dict(item.input_schema)}}
                for item in definitions])
        except Exception as exc:                    # noqa: BLE001
            raise ToolContractError(str(exc)) from exc

        return

    if registry is None:
        return

    # Registration is the handler check: the registry refuses a tool that is
    # not a command and has no handler, so a registered tool can run. Only
    # the router ever reaches a handler.

    missing = []

    for item in definitions:
        try:
            registry.get(item.name)
        except KeyError:
            missing.append(f"{item.name}: not registered")

    if missing:
        raise ToolContractError("; ".join(missing))


def phase_of(task_class: str, *, mixed_phase: str = "") -> str:
    """The selection phase for a request class (answer_scope's names)."""
    if mixed_phase:
        return mixed_phase

    return {"IMPLEMENTATION": IMPLEMENTATION, "GENERAL": GENERAL,
            "NORMATIVE": NORMATIVE, "MIXED": MIXED_QUESTION}.get(task_class, GENERAL)


__all__ = ["CODING_FAMILY", "NORMATIVE_FAMILY", "GENERAL_FAMILY", "EXTERNAL_FAMILY",
           "EXTERNAL_PHASES", "EVIDENCE_PROVIDERS",
           "CODING_TOOLS", "ToolSelection", "ToolSelector", "DeterministicToolSelector",
           "ToolContractError", "check_contract", "phase_of", "MIXED_IMPLEMENTATION"]
