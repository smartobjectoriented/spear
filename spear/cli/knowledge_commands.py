"""Durable knowledge: the skill library, remembered facts and /knowledge."""

import os
import re
import json
from context import skill_library, workspace_context
from context.memory_store import MemoryStoreError
from runtime.tracing import EventStatus, EventType
from cli import session_workspace
from cli.chat_settings import TRACE, resource_dir
from cli.corpus_registry import load_projects
from cli.session_workspace import confirm


# ── skill library (Hermes-style, confirmation-gated) ────────────────
# Skills are reusable markdown procedures the model writes after completing
# a task. Stored as files + embedded in a dedicated ChromaDB collection;
# the most similar skills are injected into the prompt on each turn.

SKILLS_DIR = resource_dir("SPEAR_SKILLS_DIR", "skills")


def skill_save(name, content, description="", scope=(), requires=()):
    """Write a skill file. Returns a result message."""

    try:
        skill = skill_library.save(SKILLS_DIR, name, content,
                                   description=description, scope=scope,
                                   requires=requires)
    except skill_library.SkillError as exc:
        return f"ERROR: {exc}"

    revised = f", revision {skill.version}" if skill.version > 1 else ""

    return (f"OK: skill '{skill.name}' saved{revised} "
            f"({len(skill_library.load_library(SKILLS_DIR))} skills in library)")


def library_skills():
    """Every runnable skill, ``(name, document, stated scope)``.

    Which of them a turn is offered is the context selector's decision, on
    scope and task class alone. A similarity gate stood in front of it and
    missed the procedure a request needed more often than it found one: a
    skill is short, and a request names its problem in words the skill's
    body rarely uses. A skill whose required commands are missing is left
    out, as it always was.
    """
    return [(skill.name, skill.document, skill.scope if skill.scope_declared else ())
            for skill in skill_library.load_library(SKILLS_DIR)
            if not skill_library.missing_requirements(skill)]


def remember_fact(note):
    """/remember: the operator's statement, as workspace knowledge."""
    from context import workspace_knowledge as wk

    store, here = knowledge_store(TRACE), current_workspace().workspace_id
    statement = " ".join((note or "").split())
    same = store.duplicate(here, wk.Kind.PROJECT_FACT, statement[:60], statement)

    if same is not None:
        return f"already recorded as {same.record_id} [{same.lifecycle}] for {here}"

    try:
        record = store.add(here, kind=wk.Kind.PROJECT_FACT, subject="", statement=statement,
                           provenance=wk.Provenance.USER_CONFIRMED)
    except wk.KnowledgeError as exc:
        return f"not recorded: {exc}"

    configured = re.search(r"\b(?:build|test|validation)\s+command\b", statement, re.I)

    return (f"{record.record_id} stored as workspace knowledge for {here} "
            f"[{record.lifecycle}, {record.verification}]" + (
                "\nA description, not configuration: projects.json's build and test "
                "entries are what SPEAR runs." if configured else ""))


def propose_knowledge(note):
    """The model's remember tool: a proposal, inert until the operator accepts it."""
    from context import workspace_knowledge as wk

    store, here = knowledge_store(TRACE), current_workspace().workspace_id

    try:
        record = store.add(here, kind=wk.Kind.PROJECT_FACT, subject="",
                           statement=" ".join((note or "").split()),
                           provenance=wk.Provenance.MODEL_DERIVED)
    except wk.KnowledgeError as exc:
        return f"ERROR: not proposed: {exc}"

    if record.lifecycle != wk.Lifecycle.PROPOSED:
        return f"OK: already recorded as {record.record_id}"

    return (f"OK: proposed {record.record_id} as workspace knowledge. It is not used until "
            f"the operator accepts it (/knowledge accept {record.record_id}).")


def legacy_notes_notice():
    """One line when this corpus has legacy remembered notes not yet migrated."""
    from context import knowledge_migration

    if not os.path.isfile(session_workspace.MEMORIES_FILE) or os.path.isfile(
            knowledge_migration.marker(session_workspace.MEMORIES_FILE)):
        return ""

    try:
        count = sum(1 for note in knowledge_migration.legacy_notes(session_workspace.MEMORIES_FILE)
                    if note.active)
    except (OSError, MemoryStoreError):
        return ""

    return (f"{count} legacy remembered note{'s are' if count != 1 else ' is'} no longer "
            f"shown to turns: /knowledge migrate-remember shows what migrating "
            f"{'them' if count != 1 else 'it'} would do." if count else "")


_KNOWLEDGE_STORE = []


def knowledge_store(trace=None, task_id="", session_id=None):
    """The workspace knowledge store, opened once per session; its audit goes
    to the trace this call names."""
    from context import workspace_knowledge

    path = workspace_knowledge.default_path()

    if not _KNOWLEDGE_STORE or _KNOWLEDGE_STORE[0].path != path:
        for opened in _KNOWLEDGE_STORE:
            opened.close()

        _KNOWLEDGE_STORE[:] = [workspace_knowledge.KnowledgeStore(
            path, common=workspace_knowledge.default_common_path())]

    store = _KNOWLEDGE_STORE[0]

    def audit(kind, metadata):
        if trace is not None:
            trace.emit(EventType(kind), task_id, session_id=session_id,
                       status=EventStatus.OK, metadata=dict(metadata))

    store.audit = audit

    return store


def current_workspace():
    """This session's workspace, as the context selector identifies it."""
    return workspace_context.from_session(
        project=session_workspace.PROJECT, spec={},
        registered=session_workspace.PROJECT in load_projects(),
        project_root=session_workspace.PROJECT_ROOT,
        corpus_root=session_workspace.CORPUS_ROOT)


_KNOWLEDGE_KINDS = {"fact": "PROJECT_FACT", "architecture": "ARCHITECTURE_FACT",
                    "build": "BUILD_FACT", "relation": "COMPONENT_RELATION",
                    "decision": "DECISION", "command": "COMMAND_KNOWLEDGE"}

KNOWLEDGE_USAGE = (
    "usage: /knowledge [list [--all|--proposed|--stale|--revoked]]\n"
    "       /knowledge add [--kind fact|architecture|build|relation|decision|command]\n"
    "                      [--subject S] [--tags a,b] [--path P]\n"
    "                      [--source FILE[:LINE] [--quote TEXT]] <statement>\n"
    "       /knowledge show|accept|revoke <id> [reason]   /knowledge amend <id> <statement>\n"
    "       /knowledge check | export [FILE] | purge | migrate-remember [--apply]")


def knowledge_command(arguments: str, *, approve=None) -> str:
    """One /knowledge command, for this session's exact workspace."""
    from context import workspace_knowledge as wk

    store, workspace = knowledge_store(TRACE), current_workspace()
    here = workspace.workspace_id

    # A verb, its options (each value one word, or quoted), then free text
    # taken exactly as written: a statement is prose, not shell, and an
    # apostrophe in it is an apostrophe.
    verb, _, remainder = arguments.strip().partition(" ")
    verb, options, words = verb or "list", {}, []

    while True:
        match = re.match(r"\s*--(\w+)\s+(\"[^\"]*\"|'[^']*'|\S+)", remainder)

        if not match:
            break

        value = match.group(2)
        options[match.group(1)] = value[1:-1] if value[:1] in "\"'" else value
        remainder = remainder[match.end():]

    text = remainder.strip()
    rest = text.split() if verb != "add" else []

    try:
        if verb == "list":
            wanted = {"all": (), "proposed": (wk.Lifecycle.PROPOSED,),
                      "stale": (wk.Lifecycle.STALE,), "revoked": (wk.Lifecycle.REVOKED,)}
            flag = next((name for name in wanted
                         if f"--{name}" in arguments.split()), "")
            store.validate(here, workspace.root)
            records = store.list(here, wanted.get(flag, (wk.Lifecycle.ACTIVE,)))
            lines = [f"{wk.summary(record)}  [{record.lifecycle}, {record.provenance}"
                     f"{', common' if store.layer(here, record.record_id) == 'common' else ''}]"
                     for record in records]

            return (f"{here}\n" + "\n".join(lines)) if lines else f"{here}: no records"

        if verb == "add":
            unknown = set(options) - {"kind", "subject", "tags", "path", "source", "quote"}

            if unknown:
                return f"unknown option --{sorted(unknown)[0]}\n{KNOWLEDGE_USAGE}"

            statement = [text]
            kind = _KNOWLEDGE_KINDS.get(options.get("kind", "fact"), options.get("kind", ""))
            sources, provenance = (), wk.Provenance.USER_CONFIRMED

            if options.get("source"):
                path, _, line = options["source"].partition(":")
                source, why = wk.source_evidence(workspace.root, path, options.get("quote", ""),
                                                 int(line) if line.isdigit() else None)

                if source is None:
                    return f"not recorded: {why}"

                sources, provenance = (source,), wk.Provenance.PROJECT_SOURCE

            record = store.add(here, kind=kind, subject=options.get("subject", ""),
                               statement=" ".join(statement), provenance=provenance,
                               tags=tuple(tag for tag in options.get("tags", "").split(",")
                                          if tag),
                               paths=(options["path"],) if options.get("path") else (),
                               sources=sources)
            clash = [group for group in wk.conflicts(store.list(here, (wk.Lifecycle.ACTIVE,)))
                     if any(item.record_id == record.record_id for item in group)]

            return (f"{record.record_id} recorded for {here} [{record.lifecycle}, "
                    f"{record.verification}]" + (
                        "\nKNOWLEDGE CONFLICT with " + ", ".join(
                            item.record_id for item in clash[0]
                            if item.record_id != record.record_id)
                        + ": both stay active and neither is preferred; amend or revoke one"
                        if clash else ""))

        if verb == "show" and len(rest) == 1:
            record = store.get(here, rest[0])

            if record is None:
                return f"{rest[0]}: no such record in this workspace"

            return wk.detail(record, store.history(here, rest[0]))

        if verb == "accept" and len(rest) == 1:
            return f"{store.accept(here, rest[0]).record_id} is now active"

        if verb == "revoke" and rest:
            record = store.revoke(here, rest[0], " ".join(rest[1:]))
            return f"{record.record_id} revoked (kept in the history; /knowledge purge deletes)"

        if verb == "amend" and len(rest) >= 2:
            record = store.amend(here, rest[0], text.split(None, 1)[1])
            return f"{record.record_id} is now version {record.version}"

        if verb == "check" and not rest:
            stale = store.validate(here, workspace.root)
            restored = store.revalidate(here, workspace.root)
            clashes = wk.conflicts(store.list(here, (wk.Lifecycle.ACTIVE,)))

            return "\n".join([f"stale: {', '.join(r.record_id for r in stale) or 'none'}",
                              f"revalidated: {', '.join(r.record_id for r in restored) or 'none'}",
                              f"conflicts: " + ("; ".join(
                                  " vs ".join(r.record_id for r in group) for group in clashes)
                                  or "none")])

        if verb == "export" and len(rest) <= 1:
            text = json.dumps(store.export(here), indent=1, ensure_ascii=False)

            if rest:
                with open(rest[0], "w", encoding="utf-8") as handle:
                    handle.write(text)

                return f"exported {here} to {rest[0]}"

            return text

        if verb == "migrate-remember":
            from context import knowledge_migration

            if set(rest) - {"--apply"}:
                return "usage: /knowledge migrate-remember [--apply]  (without --apply: a dry run)"

            report = knowledge_migration.migrate(store, here, session_workspace.MEMORIES_FILE,
                                                 apply="--apply" in rest)

            if report["applied"]:
                store.audit("knowledge_migrated", {
                    "workspace": here, "counts": report["counts"],
                    "records": report["created"], "reason": "legacy remembered notes"})

            counts = ", ".join(f"{key.replace('_', ' ')} {value}"
                               for key, value in report["counts"].items())
            return (f"{'migrated' if report['applied'] else 'dry run, nothing changed'}: "
                    f"{report['found']} legacy notes in "
                    f"{os.path.basename(session_workspace.MEMORIES_FILE)}\n"
                    f"{counts}" + ("" if report["applied"] or not report["found"] else
                                   "\n/knowledge migrate-remember --apply to migrate"))

        if verb == "purge" and not rest:
            if not (approve or confirm)(f"Delete every knowledge record of {here}, "
                                        f"history included?"):
                return "nothing deleted"

            deleted = store.purge(here)
            kept = len(store.list(here))

            return (f"deleted {deleted} records of {here}"
                    + (f"; {kept} common records stay, they come with the image" if kept else ""))
    except wk.KnowledgeError as exc:
        return f"not recorded: {exc}"

    return KNOWLEDGE_USAGE
