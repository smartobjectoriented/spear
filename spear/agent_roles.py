"""Small provider-neutral role policies for specialized agent contexts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tool_registry import ToolCategory, ToolMutability, ToolRegistry


class AgentRole(StrEnum):
    MAIN = "main"
    EXPLORER = "explorer"
    REVIEWER = "reviewer"
    PLANNING = "planning"


@dataclass(frozen=True)
class AgentRoleSpec:
    role: AgentRole
    prompt_addition: str
    allowed_tool_names: frozenset[str]
    execution_mode: str
    max_model_turns: int
    max_tool_calls: int
    context_limit: int
    can_write_memory: bool = False
    web_access: bool = False
    checkpoint_access: bool = False
    recursive_delegation: bool = False
    model_override: str | None = None

    def __post_init__(self) -> None:
        if min(self.max_model_turns, self.max_tool_calls, self.context_limit) < 1:
            raise ValueError("role budgets must be positive")

        # Every specialized role is read-only by construction. MAIN is the sole
        # exception, so these invariants are asserted here rather than trusted
        # to whoever builds the spec.

        if self.role in {AgentRole.EXPLORER, AgentRole.REVIEWER, AgentRole.PLANNING}:
            if self.execution_mode != "safe":
                raise ValueError(f"{self.role.value} must use safe execution mode")

            if self.can_write_memory or self.web_access or self.checkpoint_access:
                raise ValueError(
                    f"{self.role.value} cannot receive write, web, or checkpoint access"
                )

            if self.recursive_delegation:
                raise ValueError(f"{self.role.value} cannot recursively delegate")

    def validate_registry(self, registry: ToolRegistry) -> None:
        """Prove the registry actually exposes this role exactly what it declares.

        The check runs in both directions: a tool the role names must be
        registered and visible to it, and every tool it will be handed must be
        one the role is allowed to have.
        """

        specs = registry.list_specs(role=self.role.value, names=self.allowed_tool_names)
        exposed = {spec.name for spec in specs}

        if exposed != set(self.allowed_tool_names):
            missing = ", ".join(sorted(set(self.allowed_tool_names) - exposed))
            raise ValueError(f"role tools are not registered/visible: {missing}")

        for spec in specs:
            if spec.mutability == ToolMutability.MUTATING:
                raise ValueError(f"role exposes mutating tool: {spec.name}")

            if spec.category == ToolCategory.WEB and not self.web_access:
                raise ValueError(f"role exposes web tool: {spec.name}")

            if spec.category == ToolCategory.MEMORY and not self.can_write_memory:
                raise ValueError(f"role exposes memory-write tool: {spec.name}")


EXPLORER_INSTRUCTIONS = """\
You are a repository Explorer. Investigate the requested codebase question using
only grounded local evidence. You are read-only: never modify files, durable
memory, checkpoints, or repository state. Return concise JSON with keys summary,
key_findings, relevant_files, entry_points, symbols, call_chains, tests,
architecture, uncertainties, unanswered_questions, and evidence_references.
Relevant files should be ordered most important first. Do not include hidden
reasoning or a transcript; cite paths and symbols instead.
"""


def explorer_role(
    *, max_model_turns: int = 8, max_tool_calls: int = 16,
    context_limit: int = 32_768, model_override: str | None = None,
) -> AgentRoleSpec:
    return AgentRoleSpec(
        AgentRole.EXPLORER, EXPLORER_INSTRUCTIONS,
        frozenset({"bash", "search_corpus"}), "safe",
        max_model_turns, max_tool_calls, context_limit,
        model_override=model_override,
    )


REVIEWER_INSTRUCTIONS = """\
You are an independent code Reviewer. Evaluate the supplied requirements,
grounded diff, project rules, and verification evidence from a fresh context.
You are strictly read-only. Reject only grounded requirement, correctness,
compatibility, security, scope, or verification problems—not subjective style.
Return concise JSON with verdict (accept, repair_required, or blocked), confidence,
findings, requirement_coverage, verification_assessment, suspected_regressions,
unrelated_changes, test_gaps, security_concerns, recommended_repairs, and
evidence_references. Each finding has severity, category, description,
affected_path, requirement_reference, evidence_reference, confidence, blocking,
and suggested_remediation. Never include hidden reasoning or a transcript.
"""


def reviewer_role(
    *, max_model_turns: int = 8, max_tool_calls: int = 16,
    context_limit: int = 32_768, model_override: str | None = None,
) -> AgentRoleSpec:
    return AgentRoleSpec(
        AgentRole.REVIEWER, REVIEWER_INSTRUCTIONS,
        frozenset({"bash", "search_corpus"}), "safe",
        max_model_turns, max_tool_calls, context_limit,
        model_override=model_override,
    )


PLANNING_INSTRUCTIONS = """\
Create a small ordered plan from grounded requirements and repository evidence.
You are read-only and must not claim implementation, verification, or review has
completed. Return structured plan steps without hidden reasoning.
"""


def planning_role(
    *, max_model_turns: int = 2, max_tool_calls: int = 6,
    context_limit: int = 16_384,
) -> AgentRoleSpec:
    return AgentRoleSpec(
        AgentRole.PLANNING, PLANNING_INSTRUCTIONS,
        frozenset({"bash", "search_corpus"}), "safe",
        max_model_turns, max_tool_calls, context_limit,
    )
