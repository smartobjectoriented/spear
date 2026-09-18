"""Deterministic role-aware tool exposure without semantic tool search."""

from __future__ import annotations

import json
import re

import answer_scope
from dataclasses import dataclass

from agent_roles import AgentRole, AgentRoleSpec
from model_backend import ToolDefinition
from tool_registry import ToolCategory, ToolMutability, ToolRegistry


#: Said once, in the words the turn needs: what is forbidden, and what to do
#: instead. It lives beside the decision that produces it -- ToolView.read_only
#: -- so the rule the model reads and the boundary the harness enforces are
#: the same judgement, never two that can drift apart.

READ_ONLY_RULE_ID = "task:read_only"
READ_ONLY_RULE = (
    "READ-ONLY TASK: The user explicitly prohibited changes. Do not edit, "
    "repair, compile, or run tests. Gather only the evidence needed to answer "
    "the stated question, then answer immediately."
)


@dataclass(frozen=True)
class ToolView:
    role: AgentRole
    names: tuple[str, ...]
    definitions: tuple[ToolDefinition, ...]
    schema_token_estimate: int
    approximate: bool = True
    reason: str = "static role view"
    #: The request forbade changing anything. Not a guess about what the task
    #: needs -- an instruction it gave -- so it binds the command boundary too.
    read_only: bool = False


class ToolExposurePolicy:
    _MUTATION = re.compile(
        r"\b(add|append|change|correct|create|delete|edit|fix|implement|improve|"
        r"migrate|modify|patch|refactor|remove|rename|repair|replace|rework|"
        r"update|write)\b", re.IGNORECASE,
    )
    # There is no _WEB regex any more. Guessing web intent from the wording
    # cost the capability outright whenever the phrasing missed: "please get
    # the complete Code-G pdf" matched nothing, so search_internet and
    # fetch_url were withheld and the model spent eleven turns narrating "let
    # me fetch the specification from the web" with no tool to call. Even
    # "fetch https://nvlpubs.nist.gov/..." missed. The two schemas it saved
    # measure 155 tokens of a 65536 window; a tool the model cannot see is one
    # it describes instead of using, which costs the whole turn.
    # `web_enabled` stays: withholding these is a decision (--no-network), not
    # an inference from vocabulary.
    _MEMORY = re.compile(r"\bremember|durable memory|save .*memory\b", re.IGNORECASE)

    # `fetch_url` retrieves ONE named address, so an objective that names no
    # address and no network has nothing for it to do. Withholding it is safe
    # in a way that withholding `search_internet` is not: search is how a
    # network intent is discovered, and the note above records what removing
    # it cost. This gate exists because a task reading "correct main.py" was
    # offered fetch_url and not edit_file, and the model tried
    # fetch_url(save_as="main.py") to write a local file.

    _NETWORK_INTENT = re.compile(
        r"https?://|www\.|\b(url|urls|link|links|website|web|webpage|online|"
        r"internet|download|downloaded|fetch|fetching|curl|wget|http|https|"
        r"browse|page|pdf|datasheet|upstream|changelog|release notes)\b",
        re.IGNORECASE,
    )

    # The tools a local project turn always has, whatever the objective looks
    # like. The selection may add or drop specialised tools; it may not leave
    # the main role without a way to read, search, edit and run. The heuristic
    # that decides "does this task mutate?" read "Find the failing assertion
    # in the long evidence file and correct main.py" as read-only -- "correct"
    # was not in its verb list -- and handed the turn four schemas, none of
    # which could change a file.

    _READ_FLOOR = ("bash", "search_corpus")
    _WRITE_FLOOR = ("edit_file", "write_file")

    # An explicit prohibition, and only an explicit one. The object has to be
    # general -- "without editing files", "do not change anything",
    # "read-only" -- because a prohibition scoped to one named file is the
    # opposite kind of instruction: "Fix the greeting function and do not edit
    # the unrelated data file" and "rename the key ... without changing
    # unrelated.py" are writing tasks that name something to leave alone, and
    # reading them as read-only would take away the tools they need.
    #
    # Evaluated BEFORE _MUTATION, which would otherwise find the word
    # "editing" inside "without editing" and conclude the opposite.

    _GENERIC_OBJECT = (r"(?:\s+(?:any\s+)?(?:files?|anything|the\s+"
                       r"(?:code|files?|repository|repo|tree|workspace|project)))?")
    _READ_ONLY_INTENT = re.compile(
        r"\bread[-\s]only\b"
        r"|\bwithout\s+(?:editing|modifying|changing|writing|touching)"
        + _GENERIC_OBJECT + r"\s*(?:[.,;!]|$)"
        r"|\b(?:do\s+not|don'?t|never)\s+"
        r"(?:edit|modify|change|write(?:\s+to)?|touch)"
        + _GENERIC_OBJECT + r"\s*(?:[.,;!]|$)",
        re.IGNORECASE,
    )

    @classmethod
    def read_only_intent(cls, objective: str) -> bool:
        """True when the request forbids changing anything at all."""

        return bool(cls._READ_ONLY_INTENT.search(objective or ""))

    @classmethod
    def floor(cls, role: AgentRole, *, read_only: bool = False) -> tuple[str, ...]:
        """The tools this role keeps no matter what the objective says.

        The floor guarantees a way to work; it does not grant one the request
        refused. A turn told not to edit keeps reading, searching and bash --
        and bash itself runs without workspace write, so the prohibition is
        not something the model can route around with `sed -i` or `>`.
        """

        if role == AgentRole.MAIN and not read_only:
            return cls._READ_FLOOR + cls._WRITE_FLOOR

        return cls._READ_FLOOR

    def select(
        self, registry: ToolRegistry, role: AgentRole | AgentRoleSpec, *,
        objective: str = "", mutation_expected: bool | None = None,
        web_enabled: bool = False, memory_write_enabled: bool = False,
        standard_bound: bool = False, prior_scope: str | None = None,
    ) -> ToolView:
        """The tools a role may see for this objective.

        Narrowing the set is not only a policy matter: every schema is spent
        from the model's context window, and a tool it cannot use is a tool
        whose description is paid for and wasted.
        """

        spec = role if isinstance(role, AgentRoleSpec) else None
        role_name = spec.role if spec else role
        allowed = spec.allowed_tool_names if spec else None

        candidates = registry.list_specs(
            role=role_name.value, model_visible=True, names=allowed,
        )

        # An explicit prohibition first: it is an instruction, not a guess,
        # and it wins over every verb _MUTATION can find.

        read_only = self.read_only_intent(objective)

        # What THIS turn asked about. History may say what its words refer to;
        # it does not add a second piece of work to a question that asked for
        # one thing.
        withhold_local = answer_scope.withholds_local_tools(
            answer_scope.of(objective, prior=prior_scope,
                            standard_bound=standard_bound),
            standard_bound=standard_bound)

        # Whether the task looks like it will change something, read from the
        # objective unless the caller already knows.

        mutation = False if read_only else (
            bool(self._MUTATION.search(objective))
            if mutation_expected is None else mutation_expected)

        selected = []

        for tool in candidates:
            # The standard tools answer against a bound normative corpus; with
            # none bound they have nothing to answer from.

            if tool.name.startswith("standard.") and not standard_bound:
                continue

            # Specialized roles are read-only, so nothing mutating, web-facing
            # or memory-writing is ever exposed to them.

            if role_name in {AgentRole.EXPLORER, AgentRole.REVIEWER, AgentRole.PLANNING}:
                if tool.mutability == ToolMutability.MUTATING:
                    continue

                if tool.category in {ToolCategory.WEB, ToolCategory.MEMORY}:
                    continue
            elif tool.mutability == ToolMutability.MUTATING and (
                    read_only or (tool.category != ToolCategory.MEMORY
                                  and not mutation)):
                continue

            if tool.category == ToolCategory.WEB and not web_enabled:
                continue

            if tool.name == "fetch_url" and not self._NETWORK_INTENT.search(objective):
                continue

            # Memory still needs a sign the task calls for it: `remember`
            # MUTATES durable state, so exposing it unasked invites writes
            # nobody wanted. Reading a page does not have that failure mode.

            if tool.category == ToolCategory.MEMORY and not (
                memory_write_enabled and self._MEMORY.search(objective)
            ):
                continue

            selected.append(tool)

        # The floor goes back in, in registry order, and only from tools this
        # role was already allowed: a guaranteed set is not a way around the
        # role boundary, and the read-only roles keep no write tools.

        floor = self.floor(role_name, read_only=read_only)
        keep = {tool.name for tool in selected}
        keep.update(tool.name for tool in candidates if tool.name in floor)
        selected = [tool for tool in candidates if tool.name in keep]

        # A turn that asked about the bound document, and named nothing local,
        # is not offered the tools that reach a working tree -- the floor
        # included, because the floor guarantees a way to WORK and on this turn
        # the way to work is the document. Not because reading code is wrong,
        # but because it was not what was asked: a conversation that has spent
        # the morning in a codebase otherwise answers a question about the
        # document with the codebase, and every identifier in that answer is
        # real, grounded, and unrequested. They return the moment a turn asks
        # about an implementation -- see answer_scope.

        if withhold_local:
            selected = [tool for tool in selected
                        if tool.category not in answer_scope.LOCAL_CATEGORIES]

        definitions = tuple(ToolDefinition(
            tool.name, tool.description, dict(tool.input_schema)
        ) for tool in selected)
        chars = sum(len(item.name) + len(item.description)
                    + len(json.dumps(item.input_schema, sort_keys=True))
                    for item in definitions)

        return ToolView(
            role_name, tuple(item.name for item in definitions), definitions,
            chars // 4 + (1 if chars else 0), True,
            ("explicit read-only request" if read_only
             else "deterministic role/task category exposure"),
            read_only,
        )
