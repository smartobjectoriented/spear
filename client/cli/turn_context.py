"""What a turn is told besides the conversation: rules, selected context."""

import os
import re
import time
from context import context_selection, context_sources, workspace_context
from runtime.tracing import EventStatus, EventType
from context.context_engine import (
    ContextItem, ContextLayer, Freshness, working_state_context_item,
)
from cli import model_io, session_workspace
from cli.chat_settings import (
    APP_DIR, LEARNED_RULES_FILE, RULES_BUDGET, RULES_DIR,
)
from cli.corpus_registry import load_projects
from cli.knowledge_commands import knowledge_store
from cli.model_io import DEFAULT_CTX
from cli.session_workspace import confirm, sandbox_mount
from cli.terminal_ui import C_DIM, C_RST, C_WARN


# The external capability providers this deployment registers. Like the
# project registry it is the deployment's own file, never the repository's:
# commands, endpoints and server names are not public material.
CAPABILITIES_FILE = os.environ.get("SPEAR_CAPABILITIES_FILE") or f"{APP_DIR}/capabilities.json"


# One item in the system context of EVERY corpus: a corpus-specific domain
# prompt and the generic one are mutually exclusive, so a writing rule that
# lives in either of them is a rule half the sessions never see.
#
# It exists because asking for only the relevant items and getting them, plus
# a sentence naming the ones left out, is the commonest way a restricted
# answer stops honouring its restriction — the exclusion is explained, and in
# explaining it the excluded thing is named after all. The second sentence is
# the one that matters: "and X is not relevant" is exactly the compliance
# claim that breaks compliance.

ANSWER_SCOPE_RULE = (
    "\n\n### Answer scope\n\n"
    "When the user explicitly restricts the answer to relevant items, omit "
    "excluded items entirely. Do not name them to explain their exclusion or "
    "to claim compliance with the restriction. When the user does ask for a "
    "comparison, or asks why something was excluded, answer that fully: the "
    "rule is about mentions nobody asked for, not about questions that were.\n"
)


_SYSTEM_CONTEXT_LAYERS = {
    ContextLayer.SYSTEM_RULES,
    ContextLayer.PROJECT_RULES,
    ContextLayer.DURABLE_MEMORY,
    ContextLayer.TASK_WORKING_STATE,
    ContextLayer.CONVERSATION_SUMMARY,
    ContextLayer.RETRIEVED_CONTEXT,
}


def build_task_context_items(
    *, system_instructions, system_source, global_rules, project_rules,
    tool_guide, tool_guide_source,
    memories, skills, working_directory, working_state, retrieval,
    retrieval_source,
):
    """Collect current prompt fragments with provenance in legacy render order."""
    specs = (
        ("system:instructions", ContextLayer.SYSTEM_RULES, system_source,
         system_instructions, 100, True, Freshness.CURRENT,
         "base harness instructions"),
        ("system:global-rules", ContextLayer.SYSTEM_RULES, "rules.d",
         global_rules, 100, True, Freshness.CURRENT,
         "standing harness rules"),
        ("project:rules", ContextLayer.PROJECT_RULES, CORPUS_RULES_FILE,
         project_rules, 100, True, Freshness.CURRENT,
         "project-scoped rules"),
        ("system:tool-guide", ContextLayer.SYSTEM_RULES, tool_guide_source,
         tool_guide, 100, True, Freshness.CURRENT,
         "current tool-use instructions"),
        # Its own item rather than a tail on the instructions above: under
        # real pressure a protected item is truncated from the end as a last
        # resort, and a rule appended to the longest block in the prompt is
        # the first sentence to go. Three lines, marked not truncatable.
        ("system:answer-scope", ContextLayer.SYSTEM_RULES, "built_in_answer_scope",
         ANSWER_SCOPE_RULE, 100, True, Freshness.CURRENT,
         "how an explicitly restricted answer is written"),
        ("memory:project", ContextLayer.DURABLE_MEMORY, session_workspace.MEMORIES_FILE,
         memories, 75, False, Freshness.RECENT,
         "durable project knowledge"),
        ("retrieval:skills", ContextLayer.RETRIEVED_CONTEXT, "skill_library",
         skills, 65, False, Freshness.CURRENT,
         "task-relevant learned procedure"),
        ("project:working-directory", ContextLayer.PROJECT_RULES, "workspace_runtime",
         working_directory, 100, True, Freshness.CURRENT,
         "actual workspace and sandbox paths"),
    )
    items = [ContextItem(
        item_id=item_id, layer=layer, source=source, content=content,
        priority=priority, protected=protected, freshness=freshness,
        inclusion_reason=reason,
        truncatable=item_id != "system:answer-scope",
    ) for (item_id, layer, source, content, priority, protected, freshness, reason)
        in specs if content]
    items.append(working_state_context_item(working_state))

    if retrieval:
        items.append(ContextItem(
            item_id="retrieval:task",
            layer=ContextLayer.RETRIEVED_CONTEXT,
            source=retrieval_source,
            content=retrieval,
            priority=50,
            freshness=Freshness.CURRENT,
            protected=False,
            inclusion_reason="retrieved for current objective",
        ))

    return tuple(items)


def save_learned_rule(note):
    """Append one user-taught rule, dated, to the always-injected set."""
    line = note.strip().lstrip("-").strip()
    os.makedirs(os.path.dirname(LEARNED_RULES_FILE) or ".", exist_ok=True)

    with open(LEARNED_RULES_FILE, "a") as handle:
        handle.write(f"- {line}  ({time.strftime('%Y-%m-%d')})\n")


_RULE_HEADER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.S)


def rule_scope(content):
    """(scope, body) of a rules.d file.

    A rule file may begin with a header declaring where it applies:

        ---
        scope: corpus <name>[, <name>...]   the session's corpus, or one it
                                            federates
        scope: path <dir>                   the session's tree lies under <dir>
        scope: global                       everywhere (the same as no header)
        ---

    Without a header a rule is global, which is what every rules.d file was
    until one tree's build commands, platforms and scripts were written into
    them and reached every session on every tree. Returns scope None when a
    header is present but says nothing SPEAR can match: such a rule claims a
    scope it cannot prove, and is not injected anywhere.
    """
    match = _RULE_HEADER.match(content)

    if not match:
        return ("global",), content

    body = content[match.end():].strip()
    found = re.search(r"^scope:\s*(.+)$", match.group(1), re.M)

    if not found:
        return None, body

    kind, _, value = found.group(1).strip().partition(" ")
    values = tuple(item for item in re.split(r"[,\s]+", value.strip()) if item)

    if kind == "global" and not values:
        return ("global",), body

    if kind in ("corpus", "path") and values:
        return (kind,) + values, body

    return None, body


def session_scope():
    """What a scoped rule is matched against: this session's own corpus and
    the corpora it federates (its own parts -- not the shared ones attached
    to every session), and the trees it runs in."""
    names = {session_workspace.PROJECT} if session_workspace.PROJECT else set()

    if session_workspace.PROJECT and session_workspace.PROJECT.startswith("workspace:"):
        names.add(session_workspace.PROJECT.split(":", 1)[1])

    names |= set(session_workspace.PROJECT_SPEC.get("corpora") or ())
    roots = {os.path.realpath(root)
             for root in (session_workspace.PROJECT_ROOT, session_workspace.CORPUS_ROOT)
             if root}

    return names, roots


def rule_applies(scope):
    if scope is None:
        return False

    if scope[0] == "global":
        return True

    names, roots = session_scope()

    if scope[0] == "corpus":
        return any(name in names for name in scope[1:])

    for directory in scope[1:]:
        base = os.path.realpath(os.path.expanduser(directory))

        if any(root == base or root.startswith(base.rstrip("/") + "/")
               for root in roots):
            return True

    return False


def load_rules():
    """The rules from rules.d/*.md that apply to this session, in file-name
    order (numeric prefixes NN-name.md), followed by the learned rules.

    A rule with no scope header is global and always injected. A rule that
    declares a scope is injected only in a session that matches it -- its
    own corpus, a corpus it federates, or a tree under its path -- and a rule
    whose scope cannot be read is injected nowhere (rule_scope). Keep them
    compact: whatever applies is on every request (RULES_BUDGET)."""

    if not os.path.isdir(RULES_DIR):
        return ""

    parts = []

    for fname in sorted(os.listdir(RULES_DIR)):
        if not fname.endswith(".md"):
            continue

        with open(os.path.join(RULES_DIR, fname), "r") as f:
            scope, content = rule_scope(f.read().strip())

        if not content or not rule_applies(scope):
            continue

        title = os.path.splitext(fname)[0]

        # strip the NN- ordering prefix from the title shown to the model

        title = re.sub(r"^\d+-", "", title)
        parts.append(f"## Rule: {title}\n\n{content}")

    if os.path.isfile(LEARNED_RULES_FILE):
        with open(LEARNED_RULES_FILE, "r") as f:
            learned = f.read().strip()

        if learned:
            parts.append(f"## Rule: learned\n\n{learned}")

    if not parts:
        return ""

    rules = "\n\n" + "\n\n".join(parts)

    if len(rules) > RULES_BUDGET:
        print(f"{C_WARN}rules.d/: {len(rules)} chars injected on every request "
              f"— consider trimming (>{RULES_BUDGET}){C_RST}")

    return rules


# Per-corpus rules: an optional `.edgem-rules.md` living in the tree itself,
# so corpus-specific orientation (e.g. "SO3 user apps are in so3/usr/src")
# stays scoped to that corpus and does NOT pollute the global rules.d. We
# look at the cwd (where tools run) and at the corpus root, dedup if both
# resolve to the same file.

CORPUS_RULES_FILE = ".edgem-rules.md"


def _to_sandbox_paths(text, root):
    """Rewrite host paths in corpus rules to what bash actually sees.

    A .edgem-rules.md is written by hand and naturally spells paths the way a
    human types them — `/home/<user>/soo/so3/so3/usr/src`. But commands run
    inside the sandbox, where the corpus root is bind-mounted at
    the sandbox mount and, under the legacy /workspace mount, absolute host
    paths are refused outright. Injecting the
    file verbatim therefore contradicts the working-directory note in the same
    prompt, and the model burns rounds on "path argument may escape the
    workspace" before stumbling onto /workspace by itself.

    Only the corpus root prefix is rewritten — in both its absolute and `~`
    forms. Everything else (tool paths the *operator* runs, URLs, examples
    outside the tree) is left untouched.

    Nothing is rewritten when the tree is mounted at its own path: the paths
    the human typed are then exactly the paths bash resolves.
    """
    mount = sandbox_mount()
    home = os.path.expanduser("~")

    for prefix in (os.path.realpath(root), root):
        if not prefix or prefix == mount:
            continue

        text = text.replace(prefix, mount)

        if prefix.startswith(home):
            text = text.replace("~" + prefix[len(home):], mount)

    return text


# Orientation maps live HERE, one file per corpus name -- not in the trees they
# describe. They say how the ASSISTANT should work, which is this repository's
# concern and not the product's; putting them in a product repo pollutes it and
# puts a prompt tweak through that project's review. It also did not survive
# the trip: a blanket `.*` line in so3's .gitignore swallowed the in-tree copy,
# so a colleague cloning that tree got NO rules at all and lost the most
# behaviour-shaping context the assistant has.
#
# An in-tree .edgem-rules.md still WINS when present. That is deliberate: a
# tree someone else owns may carry its own map, and theirs should beat ours.

# Derived from RULES_DIR, not from APP_DIR: a deployment that relocates its
# rules relocates the per-corpus maps with them. They are one body of content.

SHIPPED_CORPUS_RULES = f"{RULES_DIR}/corpora"


def load_corpus_rules():
    seen = set()
    parts = []

    for root in (session_workspace.PROJECT_ROOT, session_workspace.CORPUS_ROOT):
        path = os.path.join(root, CORPUS_RULES_FILE)
        rp = os.path.realpath(path)

        if rp in seen or not os.path.isfile(rp):
            continue

        seen.add(rp)

        with open(rp, "r") as f:
            content = f.read().strip()

        if content:
            parts.append(f"## Rule: corpus ({os.path.basename(root)})\n\n"
                         + _to_sandbox_paths(content, root))

    if not parts:
        # Fallback: reached only when the tree carries nothing. A tree that has
        # its own map is never overridden by ours, which would otherwise sit
        # here going stale behind a file nobody thought to compare it against.
        # An umbrella session is named "workspace:<dir>" and has no map of its
        # own; the map that describes that tree is the one named after it. Try
        # both, or the session that federates eight corpora gets no orientation
        # at all -- which is when it needs one most.

        candidates = [session_workspace.PROJECT]

        if session_workspace.PROJECT.startswith("workspace:"):
            candidates.append(session_workspace.PROJECT.split(":", 1)[1])

        shipped = next(
            (path for path in
             (os.path.join(SHIPPED_CORPUS_RULES, f"{name}.md")
              for name in candidates)
             if os.path.isfile(path)),
            os.path.join(SHIPPED_CORPUS_RULES, f"{session_workspace.PROJECT}.md"))

        if os.path.isfile(shipped):
            with open(shipped, "r") as f:
                content = f.read().strip()

            if content:
                parts.append(f"## Rule: corpus ({session_workspace.PROJECT}, shipped)\n\n"
                             + _to_sandbox_paths(content, session_workspace.CORPUS_ROOT))

    if not parts:
        return ""

    return "\n\n" + "\n\n".join(parts)


# ── which context each runtime is shown ──────────────────────────────

CONTEXT_SELECTOR = context_selection.DeterministicContextSelector()

#: A bound on the skills one turn is shown, not a relevance cut: every skill
#: the workspace's scope admits is offered, in library order, and this only
#: stops a library grown past what a prompt should carry. Past it, skills need
#: discovering on demand rather than listing.
MAX_SKILLS = 8


def _corpus_rule_parts(workspace):
    """The project's own maps, as load_corpus_rules finds them, each with
    whose it is: the workspace's own in-tree file, its corpus root's, or the
    shipped map named after the project."""
    seen, parts = set(), []

    for root in (session_workspace.PROJECT_ROOT, session_workspace.CORPUS_ROOT):
        path = os.path.realpath(os.path.join(root, CORPUS_RULES_FILE))

        if path in seen or not os.path.isfile(path):
            continue

        seen.add(path)

        with open(path, "r") as handle:
            content = handle.read().strip()

        if not content:
            continue

        own = os.path.realpath(root) == workspace.root or not workspace.registered
        scope = ((context_selection.WORKSPACE,) if own
                 else (context_selection.PROJECTS, workspace.workspace_id))
        parts.append((os.path.basename(root), _to_sandbox_paths(content, root), scope))

    if parts:
        return parts

    names = [session_workspace.PROJECT] + (
        [session_workspace.PROJECT.split(":", 1)[1]]
        if session_workspace.PROJECT.startswith("workspace:") else [])

    for name in names:
        path = os.path.join(SHIPPED_CORPUS_RULES, f"{name}.md")

        if os.path.isfile(path):
            with open(path, "r") as handle:
                content = handle.read().strip()

            if content:
                return [(f"{session_workspace.PROJECT}, shipped",
                         _to_sandbox_paths(content, session_workspace.CORPUS_ROOT),
                         (context_selection.PROJECTS, name))]

    return []


_CAPABILITY_REGISTRY = []

# Providers whose failure the operator has been told about this session.
_PROVIDERS_REPORTED = set()


def capability_registry(trace=None, task_id="", session_id=None):
    """The session's registered providers, read once; None when there are none."""
    if not _CAPABILITY_REGISTRY:
        import atexit

        from harness import capabilities
        from harness import capability_gateway
        from harness import mcp_provider

        configs, problems = capabilities.load_configs(CAPABILITIES_FILE)
        registry = (capability_gateway.Registry(configs, factory=mcp_provider.provider_for,
                                                cwd=str(session_workspace.PROJECT_ROOT))
                    if configs else None)
        _CAPABILITY_REGISTRY.append(registry)

        if registry is not None:
            atexit.register(registry.close)

        if trace is not None:
            for config in configs:
                trace.emit(EventType.CAPABILITY_PROVIDER_REGISTERED, task_id,
                           session_id=session_id, status=EventStatus.OK,
                           metadata={"provider": config.id, "protocol": config.protocol,
                                     "transport": config.transport,
                                     "scope": list(config.scope),
                                     "tasks": sorted(config.tasks or ()),
                                     "write": config.write, "read": sorted(config.read)})

            for problem in problems:
                trace.emit(EventType.EXTERNAL_CAPABILITY_FAILED, task_id,
                           session_id=session_id, status=EventStatus.OK,
                           metadata={"stage": "registration", "reason": problem})

    return _CAPABILITY_REGISTRY[0]


def select_turn_context(*, user_input, turn_scope, binding, write, project_spec,
                        project_commands, history_text, memories, skills, retrieval,
                        system_instructions, system_source, tool_guide, working_directory,
                        task_id="", session_id=None, trace=None):
    """(workspace, {phase: selection}, {phase: rendered strings}) for one turn.

    Every phase the turn may run is selected: its own, and for a MIXED change
    the pre-pass and the implementation. Deterministic -- no model, no
    embedder -- and every decision is traced, never shown to the model.
    """
    import time
    from types import SimpleNamespace

    from agent import loop as core_loop
    from agent import prompt as core_prompt

    sel = context_selection
    started = time.perf_counter()
    bound = bool(binding)
    workspace = workspace_context.from_session(
        project=session_workspace.PROJECT, spec=project_spec,
        registered=session_workspace.PROJECT in load_projects(),
        project_root=session_workspace.PROJECT_ROOT,
        corpus_root=session_workspace.CORPUS_ROOT,
        project_commands=project_commands,
        binding=(SimpleNamespace(standard_id=binding.get("standard_id"),
                                 revision=binding.get("revision")) if bound else None))
    shared = (context_sources.rule_candidates(RULES_DIR)
              + context_sources.learned_candidate(LEARNED_RULES_FILE)
              + context_sources.corpus_rule_candidates(_corpus_rule_parts(workspace))
              + context_sources.metadata_candidates(workspace)
              + context_sources.skill_candidates(skills)
              + context_sources.runtime_candidates((
                  ("memory:project", sel.SourceType.PROJECT_MEMORY,
                   session_workspace.MEMORIES_FILE, memories, False, "memories"),
                  ("retrieval:corpus", sel.SourceType.RETRIEVED_CORPUS, "project_corpus",
                   retrieval, False, "retrieval"))))
    shared.append(sel.Candidate("request", sel.SourceType.USER_REQUEST, "user", user_input,
                                mandatory=True, rendered_by_runtime=True))

    # The workspace's own knowledge, checked against its sources first. It
    # is one candidate, already this workspace's by construction: the store
    # is only ever asked for this exact identity.
    from context import workspace_knowledge

    store = knowledge_store(trace, task_id, session_id)
    store.validate(workspace.workspace_id, workspace.root)
    known = workspace_knowledge.select(
        store.list(workspace.workspace_id, (workspace_knowledge.Lifecycle.ACTIVE,)), user_input)

    if known.text:
        shared.append(sel.Candidate("knowledge:workspace", sel.SourceType.WORKSPACE_KNOWLEDGE,
                                    "workspace knowledge", known.text, bucket="knowledge"))

        for group in known.conflicts:
            store.audit("knowledge_conflict", {
                "workspace": workspace.workspace_id, "subject": group[0].subject,
                "records": [record.record_id for record in group],
                "provenance": [record.provenance for record in group],
                "reason": "active records disagree; neither is preferred"})

    # Each registered provider is a candidate like a rule: its scope and task
    # classes decide where it applies. Its capabilities are listed only for
    # the phases that admit it.
    registry = capability_registry(trace, task_id, session_id)

    if registry is not None:
        shared += [sel.Candidate(f"capability:{config.id}", sel.SourceType.EXTERNAL_CAPABILITY,
                                 f"capabilities.json:{config.id}", "", scope=config.scope,
                                 tasks=config.tasks, bucket="capabilities")
                   for config in registry.configs.values()]

    if history_text:
        shared.append(sel.Candidate("session:history", sel.SourceType.SESSION_CONTEXT,
                                    "conversation", history_text, mandatory=True,
                                    rendered_by_runtime=True))

    if bound:
        identity = f"{binding.get('standard_id')} {binding.get('revision')}"
        choice = sel.select_standard(workspace, user_input,
                                     phase=sel.primary_phase(turn_scope, bound=True))
        scope = ((context_selection.WORKSPACE,)
                 if choice.standard and choice.standard[0] == binding.get("standard_id")
                 else (context_selection.NOWHERE,
                       f"wrong standard: {choice.reason}; the binding stays the operator's"))
        shared += [sel.Candidate("standard:binding", sel.SourceType.STANDARD_BINDING, identity,
                                 "", scope=scope, mandatory=True, rendered_by_runtime=True),
                   sel.Candidate("standard:retrieval", sel.SourceType.RETRIEVED_STANDARD,
                                 identity, "", rendered_by_runtime=True)]

    core_base = core_prompt.build(cwd=str(session_workspace.PROJECT_ROOT), tool_names=(), model="",
                                  project_rules="")
    legacy = context_sources.runtime_candidates((
        ("system:instructions", sel.SourceType.SYSTEM_RUNTIME, system_source,
         system_instructions, True, "system"),
        ("system:tool-guide", sel.SourceType.SYSTEM_RUNTIME, "tool_guide", tool_guide,
         True, "system"),
        ("project:working-directory", sel.SourceType.SYSTEM_RUNTIME, "workspace_runtime",
         working_directory, True, "system")))
    coding = [sel.Candidate("system:coding-core", sel.SourceType.SYSTEM_RUNTIME,
                            "agent/prompt.py", core_base, mandatory=True,
                            rendered_by_runtime=True)]
    selections, rendered = {}, {}

    for phase in sel.phases(turn_scope, bound=bound, write=write):
        on_core = sel.runs_on_coding_core(phase, bound=bound)
        candidates = shared + (coding if on_core else legacy)
        mandatory = sum(sel.estimate_tokens(item.text) for item in candidates if item.mandatory)

        if on_core:
            reserve = (core_loop.MAX_TOKENS
                       if not model_io.CTX_LIMIT or model_io.CTX_LIMIT > 2 * core_loop.MAX_TOKENS
                       else model_io.CTX_LIMIT // 4)
        else:
            reserve = int(os.environ.get("SPEAR_MAX_TOKENS", "8192"))

        budget = sel.Budget(int(model_io.CTX_LIMIT or DEFAULT_CTX), reserve, mandatory)

        if trace is not None:
            trace.emit(EventType.CONTEXT_SELECTION_STARTED, task_id, session_id=session_id,
                       status=EventStatus.OK, metadata={
                           "phase": phase, "workspace": workspace.to_dict(),
                           "candidates": len(candidates)})

        clock = time.perf_counter()
        selection = CONTEXT_SELECTOR.select(workspace, phase, candidates, request=user_input,
                                            budget=budget)
        selection.elapsed_ms = (time.perf_counter() - clock) * 1000
        procedures = [item for item in selection.selected if item.bucket == "skills"]

        for item in procedures[MAX_SKILLS:]:
            selection.selected.remove(item)
            selection.decisions.append(context_selection.Decision(
                item.item_id, str(item.source_type), item.source_id, workspace.workspace_id,
                f"skill limit: {MAX_SKILLS} per turn", item.rank, len(item.text),
                sel.estimate_tokens(item.text), False))

        texts = [item.text for item in selection.selected if item.bucket == "skills"]
        skills_text = ("\n\n## Skills available in this workspace\n\nProcedures learned "
                       "from past tasks. Follow one only when it fits the task at hand.\n\n"
                       + "\n\n---\n\n".join(texts)) if texts else ""
        out = {"global_rules": selection.text("global_rules"),
               "project_rules": selection.text("project_rules") + selection.text("metadata"),
               "memories": selection.text("memories"), "skills": skills_text,
               "retrieval": selection.text("retrieval")}
        out["knowledge"] = selection.text("knowledge")
        out["knowledge_door"] = None
        out["coding_context"] = "".join(out[key] for key in ("global_rules", "project_rules",
                                                             "memories", "skills", "knowledge"))

        if out["knowledge"] and on_core:
            out["knowledge_door"] = workspace_knowledge.Door(store, workspace.workspace_id)
            store.audit("knowledge_selected", {
                "workspace": workspace.workspace_id, "phase": phase, "mode": known.mode,
                "shown": [record.record_id for record in known.shown],
                "indexed": len(known.indexed), "tokens": sel.estimate_tokens(out["knowledge"]),
                "reason": "the workspace's active knowledge"})

        admitted = tuple(item.item_id.split(":", 1)[1] for item in selection.selected
                         if item.bucket == "capabilities")
        out["gateway"], out["capabilities"] = None, ""

        if admitted and on_core:
            from harness import capability_gateway

            out["gateway"] = capability_gateway.Gateway(
                registry, admitted, workspace=workspace.workspace_id, phase=phase,
                trace=trace, task_id=task_id, session_id=session_id, confirm=confirm,
                read_only=phase == sel.GENERAL)
            out["capabilities"] = out["gateway"].prepare()
            out["coding_context"] += out["capabilities"]

            for provider_id, reason in out["gateway"].failed.items():
                if provider_id not in _PROVIDERS_REPORTED:
                    _PROVIDERS_REPORTED.add(provider_id)
                    print(f"{C_DIM}  ⎿  capability provider unavailable — {reason}; "
                          f"continuing without it{C_RST}")
        elif admitted:
            # A change in a standard-bound session that is not about the
            # standard runs on the legacy runtime, which has no gateway. Its
            # behaviour is kept as it is; the scope's admission is recorded
            # rather than silently dropped.
            out["capabilities_unsupported"] = admitted

            if trace is not None:
                trace.emit(EventType.CAPABILITY_FAMILY_SELECTED, task_id,
                           session_id=session_id, status=EventStatus.OK,
                           metadata={"workspace": workspace.workspace_id, "phase": phase,
                                     "family": "EXTERNAL", "mode": "UNSUPPORTED",
                                     "providers": list(admitted),
                                     "reason": "the legacy runtime has no capability gateway"})

        selections[phase], rendered[phase] = selection, out

        if trace is not None:
            for decision in selection.decisions:
                trace.emit(EventType.CONTEXT_SELECTED if decision.selected
                           else EventType.CONTEXT_REJECTED, task_id, session_id=session_id,
                           status=EventStatus.OK, metadata={"phase": phase,
                                                            **decision.to_dict()})

            trace.emit(EventType.CONTEXT_BUDGET_APPLIED, task_id, session_id=session_id,
                       status=EventStatus.OK, metadata={
                           "phase": phase, "window": budget.window,
                           "output_reservation": budget.output_reservation,
                           "mandatory_tokens": budget.mandatory_tokens,
                           "available": budget.available,
                           "dropped": list(selection.dropped_for_budget),
                           "selector_ms": round(selection.elapsed_ms, 3),
                           "selected_tokens": sum(item.tokens for item in selection.decisions
                                                  if item.selected)})

    if trace is not None:
        trace.emit(EventType.CONTEXT_BUDGET_APPLIED, task_id, session_id=session_id,
                   status=EventStatus.OK, metadata={
                       "phase": "all", "phases": list(selections),
                       "total_ms": round((time.perf_counter() - started) * 1000, 3)})

    return workspace, selections, rendered
