"""The session's workspace: bound project, mode, policy, file primitives."""

import os
import re
import sys
import hashlib
import readline
import subprocess
from pathlib import Path
from runtime import work_phase
from harness.command_policy import CommandPolicy, ToolPolicy
from harness.resource_control import ExecutionProfile
from harness.sandbox import BubblewrapSandbox
from harness.tool_primitives import (
    Capability, CommandClassification, DEFAULT_CAPABILITY_POLICY, ExecutionMode, PathPolicyError, ToolResult, shell_argv,
)
from harness.tool_runtime import CommandRunner
from harness.workspace import SandboxSpec, Workspace, effective_mount_root
from runtime.tracing import EventStatus, EventType, new_action_id
from cli.chat_settings import AUDIT_LOGGER, STATE_DIR
from cli.corpus_registry import load_projects
from cli.terminal_ui import C_ACCENT, C_BOLD, C_DIM, C_RST, tool_result


def sandbox_mount():
    """Where the sandbox shows the current workspace to a bash command.

    Normally the tree's own host path, so what the prompt tells the model and
    what bash sees are literally the same string. Computed by the sandbox's own
    function rather than hardcoded so the two can never disagree.
    """

    if WORKSPACE is None:
        return SandboxSpec().workspace_mount

    return effective_mount_root(WORKSPACE.root)

# Bound by set_project(); named here so helpers can be called before it.

WORKSPACE = None


def extra_roots_note():
    """Name the other writable trees and the ONE way to address them.

    When each tree keeps its own host path there is nothing to translate and
    the note simply lists them. Under the legacy /workspace mount the rule is
    uniform — every extra root is at /workspaces/<name> — so the prompt states
    the rule plus the names rather than a table of paths: a dozen host paths
    repeated every turn would cost more than they inform.
    """

    if WORKSPACE is None or not WORKSPACE.extra_roots:
        return ""

    mount = sandbox_mount()
    mounts = WORKSPACE.mount_map(mount, identity=mount != SandboxSpec().workspace_mount)

    if mount != SandboxSpec().workspace_mount:
        paths = ", ".join(f"`{mounts[str(root)]}`"
                          for root in WORKSPACE.extra_roots)
        return (f" Other declared trees are writable too, at their own paths: "
                f"{paths}. Those paths work in bash AND in "
                f"edit_file/write_file. Relative paths never reach them.")

    names = ", ".join(sorted(mounts[str(root)].rsplit("/", 1)[-1]
                             for root in WORKSPACE.extra_roots))

    return (f" Other declared trees are writable too, each mounted at "
            f"`{mount}s/<name>`: {names}. That path works in bash AND "
            f"in edit_file/write_file — use it, never the host path. Relative "
            f"paths never reach them.")


def enclosing_build_tree(start):
    """The build tree the cwd sits in, if any.

    A component is not buildable on its own: `build.sh usr-so3` lives in the
    umbrella above -- with env.sh, the meta-* layers and the toolchains that
    build/tmp holds -- while the corpus, and therefore the workspace root, is
    the component. Without this the build ran read-only and bitbake died on
    "[Errno 30] Read-only file system: 'pyshtables.py'", which names neither
    the tree nor the reason.

    Recognised by its own entry points rather than by name, so a checkout
    called anything works and a directory that merely happens to sit above the
    cwd does not.
    """
    current = os.path.realpath(start)

    while True:
        if (os.path.isfile(os.path.join(current, "env.sh"))
                and os.path.isfile(os.path.join(current, "scripts", "build.sh"))):
            return current

        parent = os.path.dirname(current)

        if parent == current or current == os.path.expanduser("~"):
            return None

        current = parent


def corpus_roots():
    """The registered corpora, as additional writable roots.

    Deliberately the corpus registry and not an arbitrary list: these are the
    trees the user has already declared as theirs. A stale entry whose tree has
    gone is dropped by the workspace, not fatal here.
    """
    return tuple(spec["path"] for spec in load_projects().values())


def set_project(spec):
    """Bind all corpus-dependent globals. `spec` is a dict
    {"name","path","kind"} -- see the Projects section for the behavioural keys.

    Two distinct roots:
    - CORPUS_ROOT: the registered tree the RAG index/history/memories belong
      to (identity).
    - PROJECT_ROOT: where the TOOLS run = the real current directory. Tools
      always operate on the current tree the user launched in, so an
      edit/bash can never touch the wrong checkout."""
    global PROJECT, PROJECT_ROOT, CORPUS_ROOT, PROJECT_KIND, WORKSPACE
    global COLLECTION_NAME, HISTORY_FILE, MEMORIES_FILE, PROJECT_SPEC

    # The resolved spec, kept because it is not always IN projects.json: an
    # umbrella session is synthesised from what sits under the cwd, and looking
    # its federation up by name found nothing, so the eight corpora it had just
    # announced were never attached.

    PROJECT_SPEC = dict(spec)
    PROJECT = spec["name"]
    CORPUS_ROOT = os.path.realpath(spec["path"])
    PROJECT_ROOT = os.path.realpath(os.getcwd())   # tools run in the cwd

    # Workspace boundary. The launch directory is the PRIMARY root: relative
    # paths resolve there and nowhere else. The registered corpora are declared
    # as additional roots so a file in another tree can be edited without
    # relaunching — by absolute path, which keeps the intent explicit in the
    # audit trail. --single-root restores the launch-directory-only boundary.

    WORKSPACE = Workspace.from_path(
        PROJECT_ROOT,
        allow_absolute_paths="--allow-absolute-paths" in sys.argv[1:],
        extra_roots=() if "--single-root" in sys.argv[1:]
        else corpus_roots() + tuple(
            t for t in (enclosing_build_tree(PROJECT_ROOT),) if t),
    )

    # The command classifier accepts absolute paths only inside the workspace,
    # and the sandbox now exposes each tree at its own host path: tell it where
    # this project's trees live, or every such path reads as an escape.

    if "COMMAND_POLICY" in globals():
        COMMAND_POLICY.bind_workspace(WORKSPACE)

    PROJECT_KIND = spec["kind"]

    # Corpus identity (index/history/memories) is keyed by the CORPUS root,
    # not the cwd -- so subdirs of the same corpus share them. The tag keys
    # history and memories off the corpus IDENTITY, which is the tree; only
    # the collection may be named explicitly, because it is the one thing
    # that can travel separately from the tree.
    #
    # One rule for every corpus. A kind used to select a second one, which
    # gave two corpora a hand-written collection name and hand-written
    # history filenames -- and left `kind` deciding, invisibly, four other
    # things as well. What a corpus does is now what its registry entry says.

    tag = hashlib.md5(CORPUS_ROOT.encode()).hexdigest()[:8]
    COLLECTION_NAME = spec.get("collection") or f"adhoc_{tag}"
    HISTORY_FILE = f"{STATE_DIR}/history-adhoc-{tag}.json"
    MEMORIES_FILE = f"{STATE_DIR}/memories-adhoc-{tag}.md"

    # One explicit memory store, named. The benchmark needs to hand a run its
    # own memories without writing them into the workspace it is measuring --
    # a memory the model can simply `cat` tests nothing about memory.

    if os.environ.get("SPEAR_MEMORIES_FILE"):
        MEMORIES_FILE = os.environ["SPEAR_MEMORIES_FILE"]


PROJECT_SPEC = {}

# Bound at import for piped/module use; re-resolved in main() for the CLI.
# The current directory, because that is the one thing true of every
# deployment -- it used to be a named checkout that existed on one machine.

set_project({"name": "adhoc:" + (os.path.basename(os.getcwd().rstrip("/"))
                                 or "cwd"),
             "path": os.getcwd(), "kind": "generic"})

def execution_mode_from_argv(argv):
    """Map legacy permission flags onto the explicit Phase 1 modes."""
    mode = ExecutionMode.SAFE

    for arg in argv:
        if arg in ("--safe",):
            mode = ExecutionMode.SAFE
        elif arg in ("--ask", "--confirm", "--no-bypass"):
            mode = ExecutionMode.ASK
        elif arg in ("--auto", "--bypass-permissions", "--yolo", "-y"):
            mode = ExecutionMode.AUTO

    return mode


def capability_policy_from_argv(argv):
    """Apply the sole Phase 1c-6 capability restriction flag."""
    policy = DEFAULT_CAPABILITY_POLICY

    if "--no-network" in argv:
        policy = policy.without_network()

    return policy


# Safe is now the default.  Keep BYPASS_PERMISSIONS as a legacy display flag;
# policy decisions below use EXECUTION_MODE directly.

EXECUTION_MODE = execution_mode_from_argv(sys.argv[1:])
CAPABILITY_POLICY = capability_policy_from_argv(sys.argv[1:])
BYPASS_PERMISSIONS = EXECUTION_MODE == ExecutionMode.AUTO


# ── file / shell helpers ──────────────────────────────────────────────

def resolve_path(path):
    """Resolve a filesystem tool path inside the canonical workspace."""
    return WORKSPACE.resolve(str(path).strip().strip("'\""))


def confirm(prompt):
    """Interactive approval callback used by the ask-mode policy."""

    if EXECUTION_MODE == ExecutionMode.AUTO:
        print(f"  {prompt} {C_DIM}— auto-accepted (bypass permissions){C_RST}")
        return True

    if EXECUTION_MODE == ExecutionMode.SAFE:
        return False

    print(f"  {prompt}")
    print(f"  {C_ACCENT}❯{C_RST} 1. Yes {C_DIM}(Enter){C_RST}")
    print(f"    2. No")

    try:
        answer = input(f"  ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    # keep confirmation answers out of the arrow-key input history
    # (no-op when stdin is not a tty: readline records nothing there)

    n = readline.get_current_history_length()

    if answer and n > 0 and (readline.get_history_item(n) or "").strip().lower() == answer:
        readline.remove_history_item(n - 1)

    return answer in ("", "1", "o", "oui", "y", "yes")


def safe_mode_refusal(action):
    """The categorical refusal, before any check on the content.

    The content heuristics run first for a reason -- they are about the edit
    itself, and in --ask they run before the operator is asked to approve
    one. But in --safe no approval is coming, so a refusal that reports the
    CONTENT is answering a question nobody can reach: told "probe.c exists
    and is much larger than your proposed content -- use edit_file for
    targeted changes", the model was being pointed at a tool that safe mode
    also refuses.

    Returns the refusal text, or None when the mode permits mutation at all.
    It does not name a way around the mode: there is none inside a session.
    """
    if EXECUTION_MODE != ExecutionMode.SAFE:
        return None

    return (f"ERROR: {action} is refused -- this session runs in safe mode, "
            f"where nothing is written. No other tool or command can write "
            f"either. Say what the change would be, in full, and stop there.")


def mutation_permitted(agent_context=None):
    """May this turn change the tree at all -- before anything is proposed?

    Two independent permissions, and the harness needs BOTH before it turns
    model output into an action of its own:

      task    the user asked for an inspection and said not to modify
      policy  --safe forbids mutation whatever was asked

    Asked to inspect a file and told "Do not modify anything", under --safe,
    the harness took the model's printed listing, spent a model call asking
    it to re-express the rewrite as edit_file calls, and then executed a
    write_file the tool view had deliberately withheld. Nothing was written
    -- a size heuristic happened to refuse it, and safe mode would have
    refused after that -- but the turn had manufactured a mutation attempt
    out of a read-only request and advised the model to try another one.

    So this is asked FIRST, before any mutation-specific heuristic runs. A
    refusal that arrives after the proposal is built has already cost the
    round it was meant to save.
    """

    if getattr(agent_context, "read_only", False):
        return False

    return EXECUTION_MODE != ExecutionMode.SAFE


def authorize_mutation(prompt, *, action, paths=(), command=None):
    """Authorize a mutation and audit every rejected/cancelled attempt."""
    result = ToolPolicy(EXECUTION_MODE, approve=confirm).authorize_mutation(prompt)

    if result is not None:
        AUDIT_LOGGER.record_mutation(
            action=action, mode=EXECUTION_MODE, workspace=WORKSPACE,
            paths=paths, command=command,
            approved=False if result.status in ("denied", "cancelled") else None,
            result=result,
        )

    return result


def audit_mutation_result(action, result_text, *, paths=(), command=None):
    """Write metadata-only audit event for an authorized mutation attempt."""

    if result_text.startswith("OK:"):
        result = ToolResult("ok", result_text[3:].strip())
    elif result_text.startswith("CANCELLED"):
        result = ToolResult("cancelled", result_text)
    else:
        result = ToolResult("failed", result_text[:500])

    AUDIT_LOGGER.record_mutation(
        action=action, mode=EXECUTION_MODE, workspace=WORKSPACE,
        paths=paths, command=command, approved=True, result=result,
    )


def audit_rejected_mutation(action, summary, *, paths=()):
    """Record a mutation rejected before authorization (for example a bad path)."""
    AUDIT_LOGGER.record_mutation(
        action=action, mode=EXECUTION_MODE, workspace=WORKSPACE, paths=paths,
        approved=False, result=ToolResult("invalid_path", summary[:500]),
    )


#: Router refusals that are mutation denials and belong in the mutation trail.
#: Named here rather than inferred from the status: a DENIED envelope may be a
#: role refusal or a repeat suppression, which are not mutations at all.
_ROUTER_MUTATION_DENIALS = frozenset({
    "read_only_task", "execution_mode_denied", "outside_requested_scope",
    work_phase.NO_AUTHORITY, work_phase.NO_IMPLEMENTATION,
    work_phase.NO_PLAN, work_phase.REVIEW_IS_READ_ONLY,
})

#: What the operator is told when the router refuses a call, keyed by the
#: category it stamped on the envelope.
#:
#: These refusals happen BEFORE dispatch, so the handler that would normally
#: render the call never runs and the terminal shows nothing at all: watched
#: live, a session refused two edit_file calls in safe mode and the screen
#: went straight from the model's sentence to the next one. The model was
#: told; the person watching was not.
#:
#: Keyed on the category rather than on failure in general, because a handler
#: that fails has already printed its own result -- ``error_category`` there
#: is the generic "failed", and these names are the router's own.
#:
#: Each line says what happened and stops. None of them names another mode, a
#: flag, or a different tool: the refusal is not a menu, and a turn that reads
#: one as an invitation spends its remaining rounds proving it.
_ROUTER_REFUSAL_NOTICES = {
    "read_only_task": "the task is read-only",
    "outside_requested_scope": "outside the target the request named",
    "outside_project": "outside this project",
    # Not permanent, unlike the one above, and the line says so: the turn has
    # not finished working out what it is changing.
    work_phase.NO_AUTHORITY: "the authoritative source has not been read yet",
    work_phase.NO_IMPLEMENTATION: "the implementation has not been read yet",
    work_phase.NO_PLAN: "no evidence-backed plan has been recorded yet",
    work_phase.REVIEW_IS_READ_ONLY: "the normative review is read-only",
    "execution_mode_denied": "unavailable in {mode} mode",
    "permission_denied": "not available to this role",
    "failed_action_repeated": "this call already failed this turn",
    "cached_action_repeated": "this call was already answered this turn",
    "interrupted": "interrupted before it ran",
}


def announce_router_refusal(name, envelope):
    """One line for a call the router refused before any handler saw it.

    Returns True when something was printed, so the caller can be held to
    exactly one notice per refused call.
    """
    phrase = _ROUTER_REFUSAL_NOTICES.get(envelope.error_category)

    if phrase is None:
        return False

    tool_result(f"{name} refused: {phrase.format(mode=EXECUTION_MODE)}")

    return True


def audit_denied_mutation(action, summary, *, paths=()):
    """Record a mutation the POLICY refused, before it was ever proposed.

    Not `audit_rejected_mutation`: that one records a malformed request and
    files it as `invalid_path`. A mutation stopped because the session may
    not write is denied, and the trail has to say which of the two happened
    -- they call for different responses from whoever reads it.
    """
    AUDIT_LOGGER.record_mutation(
        action=action, mode=EXECUTION_MODE, workspace=WORKSPACE, paths=paths,
        approved=False, result=ToolResult("denied", summary[:500]),
    )


def find_file(name):
    """Resolve a filename or partial path to a full path, relative to the
    current workspace. Falls back to a recursive workspace walk (was build/-only;
    now generic so it works in any corpus)."""

    try:
        fpath = resolve_path(name)
    except PathPolicyError:
        return None

    if fpath.is_file():
        return fpath

    basename = Path(str(name)).name
    candidates = []

    for root, dirs, files in os.walk(WORKSPACE.root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {".git", "tmp"}]

        if basename in files:
            # Feed the resolver a workspace-relative path: it rejects absolute
            # ones unless --allow-absolute-paths, so handing it `root/basename`
            # dropped every candidate and made this whole walk dead code.  The
            # resolver stays the containment authority; only the form changes.

            candidate = Path(root) / basename

            try:
                relative = candidate.relative_to(WORKSPACE.root)
            except ValueError:
                continue

            try:
                candidates.append(WORKSPACE.resolve(relative))
            except PathPolicyError:
                continue

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        for candidate in candidates:
            if str(name).replace("/", "") in candidate.relative_to(WORKSPACE.root).as_posix().replace("/", ""):
                return candidate

        return candidates[0]

    return None


def read_file(path):
    try:
        resolve_path(path)
    except PathPolicyError as e:
        audit_rejected_mutation("edit_file", str(e), paths=(path,))
        return f"ERROR: {e}"

    fpath = find_file(path)

    if not fpath:
        return f"ERROR: file not found: {path}"

    relpath = os.path.relpath(fpath, PROJECT_ROOT)

    with fpath.open("r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    header = f"[file: {relpath}]\n"

    if len(content) > 15000:
        content = content[:15000] + f"\n... (truncated, {len(content)} chars total)"

    return header + content


# 128 + SIGPIPE. With pipefail on, `find . | head -5` reports 141 because head
# closes the pipe and find dies on it -- which is exactly what `| head` is FOR.
# Reporting it as a failure would have the model chase a phantom error on one
# of its most common idioms, so this one code stays silent. `make | tail` is
# unaffected: tail drains its input, so make's real exit status propagates.

SIGPIPE_EXIT = 141


def run_cmd_result(cmd, need_confirm=True, cancellation=None, execution_mode=None,
                   exec_cmd=None, timeout=None, output_chars=None):
    """Return the security substrate's structured command result.

    `cmd` is what is classified and authorised. `exec_cmd`, when given, is
    what runs in its place: the same command inside a harness-built script
    (the terminal's carried-over session), never anything the policy did not
    judge -- and judged as the script it runs in. `timeout` replaces the
    sandbox's default for this call.
    """
    # The coding core's control plane answers gateway commands before they
    # get here. One that arrives anyway came from a runtime that has no
    # gateway, and no shell ever runs it.
    from harness import host_commands

    if host_commands.recognised(cmd):
        return ToolResult("denied", f"{host_commands.recognised(cmd)} is not available on "
                                    f"this runtime; nothing was run")

    mode = execution_mode or EXECUTION_MODE
    assessment = (COMMAND_POLICY.classify_script(cmd) if exec_cmd
                  else COMMAND_POLICY.classify(cmd))
    authorization = COMMAND_POLICY.authorize(assessment, mode,
                                               approve=confirm)

    if authorization.result is not None:
        if (assessment.classification != CommandClassification.READ_ONLY
                or assessment.required_capabilities):
            AUDIT_LOGGER.record_mutation(
                action="command", mode=mode, workspace=WORKSPACE,
                command=cmd, approved=False, result=authorization.result,
                required_capabilities=assessment.required_capabilities,
                granted_capabilities=authorization.granted_capabilities)

        return authorization.result

    granted_capabilities = authorization.granted_capabilities
    profile = ExecutionProfile.from_capabilities(
        granted_capabilities,
        assessment.host_read_paths if Capability.HOST_READ in granted_capabilities else (),
    )
    sandboxed = False

    # Every authorized external command crosses the same execution boundary.
    # Classification controls only argv shape; ExecutionProfile controls the
    # sandbox mounts and network backend.

    availability = COMMAND_RUNNER.ensure_sandbox(WORKSPACE, profile)

    if availability.ok:
        argv = (shell_argv(exec_cmd) if exec_cmd
                else list(assessment.argv) if assessment.classification
                != CommandClassification.SHELL_COMPLEX
                else shell_argv(cmd))
        sandbox = COMMAND_RUNNER.sandbox
        default_timeout = getattr(sandbox, "timeout_seconds", None)
        default_chars = getattr(sandbox, "max_output_chars", None)

        try:
            if timeout and default_timeout is not None:
                sandbox.timeout_seconds = timeout

            if output_chars and default_chars is not None:
                sandbox.max_output_chars = max(default_chars, output_chars)

            result = COMMAND_RUNNER.run_sandboxed(
                WORKSPACE, argv, profile, availability=availability,
                cancellation=cancellation,
            )
        finally:
            if default_timeout is not None:
                sandbox.timeout_seconds = default_timeout

            if default_chars is not None:
                sandbox.max_output_chars = default_chars

        sandboxed = True
    else:
        result = availability

    # A read that left the declared trees is recorded even though it is
    # read-only: what the assistant looked at outside the workspace is exactly
    # what the audit trail exists to answer.

    if (assessment.classification != CommandClassification.READ_ONLY
            or assessment.required_capabilities & {Capability.NETWORK,
                                                   Capability.HOST_READ}):
            AUDIT_LOGGER.record_mutation(
                action="command", mode=mode, workspace=WORKSPACE,
            command=cmd, approved=True, result=result, sandboxed=sandboxed,
            required_capabilities=assessment.required_capabilities,
            granted_capabilities=granted_capabilities,
            execution_profile=profile.audit_metadata())

    if result.status == "timeout":
        # Actionable message so the model self-corrects instead of re-running
        # the same blind command. A bare `find` over this workspace walks the
        # huge generated/third-party trees and never returns in time.

        hint = ""

        if cmd.lstrip().startswith("find "):
            hint = (" — a find over the whole tree is too slow here. Restrict "
                    "the start path (e.g. so3/usr/src, so3/so3, avz, build/"
                    "meta-*), and/or prune heavy dirs: "
                    "find <dir> -name '<pat>' -not -path '*/build/tmp/*' "
                    "-not -path '*/.git/*'. Or grep -rn '<sym>' <subdir>.")

        return ToolResult("timeout", f"timed out ({timeout or 45}s){hint}")

    return result


def command_result_text(result):
    """Preserve the historical command text at the UI/model boundary."""

    if result.status == "timeout":
        return f"ERROR: {result.summary}"

    out = result.stdout

    if result.stderr:
        out += ("\n" if out else "") + result.stderr

    if result.exit_code not in (None, 0, SIGPIPE_EXIT):
        out += f"\n(exit {result.exit_code})"

    if result.status == "ok":
        return out or "(empty)"

    return out or result.to_legacy_text()


def run_cmd(cmd, need_confirm=True):
    """Compatibility renderer around the structured command boundary."""
    return command_result_text(run_cmd_result(cmd, need_confirm=need_confirm))


def _detect_space_unit(lines):
    """Smallest positive leading-space count among lines = the indent unit
    (so 4-space model output maps to one tab per 4 spaces)."""
    units = [len(l) - len(l.lstrip(" ")) for l in lines
             if l[:1] == " " and l.strip()]
    units = [u for u in units if u]

    return min(units) if units else 4


def _reindent_to_tabs(text, space_unit):
    """Convert each line's leading spaces to tabs at `space_unit` per tab —
    used when the model emits space indentation for a tab-indented file."""
    out = []

    for line in text.split("\n"):
        stripped = line.lstrip(" ")
        nlead = len(line) - len(stripped)

        if nlead and "\t" not in line[:nlead]:
            out.append("\t" * (nlead // space_unit)
                       + " " * (nlead % space_unit) + stripped)
        else:
            out.append(line)

    return "\n".join(out)


def _fuzzy_replace(content, old_text, new_text):
    """Exact replace; on miss, a whitespace-tolerant line-window match so a
    model that emits spaces can still edit a tab-indented file (the #1 reason
    small edits fail on SO3). new_text is re-indented to the file's style.
    Returns (new_content, mode) — mode 'exact'|'fuzzy', or (None, reason)."""
    c = content.count(old_text)

    if c == 1:
        return content.replace(old_text, new_text, 1), "exact"

    if c > 1:
        return None, "ambiguous"

    flines = content.split("\n")
    olines = old_text.rstrip("\n").split("\n")
    n = len(olines)

    if not n:
        return None, "empty"

    canon = lambda s: re.sub(r"[ \t]+", " ", s.strip())
    ocanon = [canon(l) for l in olines]
    hits = [i for i in range(len(flines) - n + 1)
            if [canon(flines[i + k]) for k in range(n)] == ocanon]

    if len(hits) != 1:
        return None, "notfound" if not hits else "ambiguous"

    i = hits[0]
    file_is_tab = any(
        "\t" in flines[i + k][:len(flines[i + k]) - len(flines[i + k].lstrip())]
        for k in range(n))
    nt = _reindent_to_tabs(new_text, _detect_space_unit(olines)) \
        if file_is_tab else new_text
    flines[i:i + n] = nt.split("\n")

    return "\n".join(flines), "fuzzy"


def edit_file(path, old_text, new_text, *, execution_context=None):
    if not path.strip():
        return ("ERROR: no path given. Pass the file path, e.g. "
                "edit_file(path='so3/usr/src/ping.c', old_text=..., "
                "new_text=...).")

    if not old_text:
        return ("ERROR: old_text is empty. edit_file needs the EXACT existing "
                "text to replace (a few unique lines). For a brand-new file "
                "use write_file; to add at the end use append_file.")

    try:
        resolve_path(path)
    except PathPolicyError as e:
        return f"ERROR: {e}"

    fpath = find_file(path)

    if not fpath:
        return f"ERROR: file not found: {path}"

    # guard: a bad/incomplete path can fuzzy-resolve to a SNAPSHOT or
    # third-party tree (e.g. avz.back/…, u-boot/…). Never write there — it
    # corrupts the pristine copy and is never the file the user meant.

    if is_excluded_path(fpath):
        return (f"ERROR: {os.path.relpath(fpath, PROJECT_ROOT)} is a snapshot / "
                f"third-party copy (.back/.0/.pristine/u-boot/atf/qemu) — never "
                f"edit it. Use the LIVE source path (re-check it with ls/find).")

    with fpath.open("r", encoding="utf-8") as f:
        content = f.read()

    # guard: never edit a GENERATED / DO-NOT-MODIFY file — it is overwritten on
    # the next build/regen and editing it hides the real fix (must change the
    # source that generates it, or the Kconfig/config instead).
    # str(): resolve_path returns a Path, and re.search refuses one. This
    # raised TypeError on EVERY edit of an existing file, so edit_file has been
    # dead since the workspace started resolving to Path objects — which is why
    # the model always fell back to rewriting whole files with write_file.

    if re.search(r"/generated/|/build/tmp/", str(fpath)) or re.search(
            r"DO NOT (?:MODIFY|EDIT)|auto(?:matically)?[ -]?generated|@generated",
            content[:600], re.IGNORECASE):
        return (f"ERROR: {path} is a GENERATED / DO-NOT-MODIFY file (it gets "
                f"overwritten on the next build). Do NOT edit it — fix the "
                f"source that generates it (a script, Kconfig, .config, or a "
                f"template), not the generated output.")

    # guard: a no-op edit changes nothing but reports success — the model then
    # believes it acted. The comparison is EXACT. It used to normalise
    # whitespace, which made every whitespace-only edit impossible: inserting a
    # blank line, fixing an indent, converting tabs. A model asked to add the
    # blank line reStructuredText needs before a paragraph sent the same edit
    # nine times, each rejected as "identical", with `sed -i` refused on the
    # other side — a closed loop with no exit. A genuine no-op is still caught
    # after the replacement, where it can be seen rather than guessed.

    if old_text == new_text:
        return ("ERROR: old_text and new_text are identical — this edit would "
                "change nothing. To INSERT text, old_text must be the existing "
                "line(s) at the insertion point and new_text those SAME lines "
                "with the new content added — not the new content alone.")

    new_content, mode = _fuzzy_replace(content, old_text, new_text)

    if new_content is not None and new_content == content:
        return ("ERROR: this edit produces no change to the file (no-op). "
                "If you meant to INSERT, include the surrounding line(s) in "
                "both old_text and new_text so the difference is the addition. "
                "Otherwise skip it, or edit the lines that actually need "
                "changing.")

    if new_content is None:
        if mode == "ambiguous":
            return ("ERROR: old_text is not unique (matches several places). "
                    "Add a few more surrounding lines so it identifies ONE "
                    "spot.")

        # ground the model: show the real end of the file so a retry can
        # anchor on actual text — and remind it APPEND exists for additions

        tail = content[-1200:]

        return (f"ERROR: text not found in {path}. The OLD block does not "
                f"exist in the file — do NOT guess content. The file "
                f"actually ends with:\n...\n{tail}\n"
                f"If you are ADDING a new section, use APPEND:{path} "
                f"instead of EDIT.")

    relpath = os.path.relpath(fpath, PROJECT_ROOT)
    blocked = authorize_mutation(f"Modify {C_BOLD}{relpath}{C_RST} ?",
                                 action="edit_file", paths=(fpath,))

    if blocked is not None:
        return blocked.to_legacy_text()

    _capture_checkpoint_path(execution_context, fpath)

    with fpath.open("w", encoding="utf-8") as f:
        f.write(new_content)

    note = " (matched ignoring indentation)" if mode == "fuzzy" else ""
    result = f"OK: {relpath} updated{note}"
    _record_mutation(execution_context, fpath)
    audit_mutation_result("edit_file", result, paths=(fpath,))

    return result


_EMAIL_DOMAIN_RE = re.compile(r"[\w.+-]+@([\w.-]+)")


def _leading_comment(text):
    """The file's leading /* ... */ block comment (verbatim), or '' if the
    file does not start with one."""
    m = re.match(r"\s*/\*.*?\*/\s*", text or "", re.DOTALL)

    return m.group(0) if m else ""


#: Whose copyright headers this deployment may rewrite. A header naming
#: anyone else is third-party and is spliced back after a full rewrite.
#: Comma-separated names or email domains; unset means every existing
#: copyright header is treated as third-party, which is the safe default for
#: a platform that does not know whose code it is editing.
COPYRIGHT_OWNERS = tuple(
    part.strip().lower()
    for part in os.environ.get("SPEAR_COPYRIGHT_OWNERS", "").split(",")
    if part.strip())


def preserve_third_party_header(old, new):
    """A pre-existing THIRD-PARTY copyright header must NOT be rewritten; only
    a header belonging to this deployment gets a year bump. If a full rewrite
    replaced a third-party header, splice the original back. Returns adjusted
    `new`.

    Who "we" are used to be one organisation's name, compiled in. A platform
    that edits somebody else's tree cannot know that from a constant, and the
    conservative answer when nobody has said -- leave every existing header
    alone -- is also the correct one."""
    old_hdr = _leading_comment(old)

    if not old_hdr or "copyright" not in old_hdr.lower():
        return new

    domains = _EMAIL_DOMAIN_RE.findall(old_hdr)
    lowered = old_hdr.lower()
    is_ours = any(owner in lowered or any(d.endswith(owner) for d in domains)
                  for owner in COPYRIGHT_OWNERS)

    if is_ours:
        return new                       # ours: a rewrite may legitimately touch it

    new_hdr = _leading_comment(new)

    if new_hdr.strip() == old_hdr.strip():
        return new                       # already preserved

    return old_hdr + (new[len(new_hdr):] if new_hdr else new)


# ── native tool calling ──────────────────────────────────────────────
# Tools are declared via the OpenAI tools API; llama-server (--jinja)
# renders them into the Qwen3 chat template and parses the model's native
# tool-call output — no homemade RUN:/EDIT: text blocks anymore.

# Read-only commands are auto-approved (no confirmation), Claude Code style.

READONLY_BINS = {
    "cat", "ls", "grep", "egrep", "fgrep", "find", "head", "tail", "wc",
    "stat", "file", "pwd", "tree", "du", "df", "which", "type", "readlink",
    "basename", "dirname", "md5sum", "sha256sum", "diff", "sort", "uniq",
    "cut", "awk", "sed",
}
GIT_READONLY = {"status", "log", "diff", "show", "branch", "ls-files",
                "blame", "describe", "rev-parse", "reflog", "stash"}
COMMAND_POLICY = CommandPolicy(CAPABILITY_POLICY)
COMMAND_RUNNER = CommandRunner(sandbox=BubblewrapSandbox())


def is_readonly_cmd(cmd):
    """True if every pipeline stage is a known read-only command with no
    redirection. sed/awk are only allowed inside pipes (no -i / no file
    rewrite via redirection, which the > check already blocks)."""

    if re.search(r"[><]|\$\(|`", cmd):
        return False

    if re.search(r"\bsed\b[^|;&]*-i", cmd):
        return False

    for seg in re.split(r"\||&&|;", cmd):
        tok = seg.strip().split()

        if not tok:
            continue

        if tok[0] == "cd":
            continue

        if tok[0] == "git":
            if len(tok) > 1 and tok[1] in GIT_READONLY and tok[1] != "stash":
                continue

            return False

        if tok[0] not in READONLY_BINS:
            return False

    return True


def _dedent_diff(diff_text):
    """Models sometimes emit the diff indented (it sat inside an XML/markdown
    block). Unified-diff lines must begin with ' ', '+', '-', '@' or '\\'. If
    EVERY non-empty line shares a leading-whitespace prefix, strip it so the
    diff markers land in column 0."""
    lines = diff_text.split("\n")
    nonempty = [l for l in lines if l.strip()]

    if not nonempty:
        return diff_text

    # well-formed: a hunk/file header sits in column 0

    if any(l.startswith(("@@ ", "--- ", "+++ ")) for l in lines):
        return diff_text

    # otherwise the whole diff is probably indented — strip the common
    # leading whitespace so the markers land in column 0

    common = os.path.commonprefix([l[:len(l) - len(l.lstrip())]
                                   for l in nonempty])

    if common:
        return "\n".join(l[len(common):] if l.strip() else l for l in lines)

    return diff_text


def apply_unified_diff(fpath, diff_text):
    """Apply a unified diff to fpath. Models naturally emit diffs for
    'improve X' tasks, so we accept them. Tolerant: de-indents a wrapped
    diff, tries `patch` with fuzz, then `git apply --recount` (which fixes
    wrong hunk line numbers) as a fallback."""

    try:
        fpath = WORKSPACE.resolve(fpath, must_exist=True)
    except PathPolicyError as e:
        return f"ERROR: {e}"

    if not fpath.is_file():
        return f"ERROR: file not found: {fpath}"

    diff_text = _dedent_diff(diff_text)

    if not diff_text.endswith("\n"):
        diff_text += "\n"

    relpath = os.path.relpath(fpath, PROJECT_ROOT)
    errs = []

    # 1) patch: ignores the diff's path headers (explicit target file), fuzz
    #    tolerates small line-number drift, -l ignores whitespace.

    try:
        r = subprocess.run(
            ["patch", "--fuzz=3", "-l", "-N", "-r", "/dev/null", str(fpath)],
            input=diff_text, capture_output=True, text=True,
            cwd=PROJECT_ROOT, timeout=20)

        if r.returncode == 0:
            return f"OK: {relpath} patched"

        errs.append("patch: " + (r.stderr or r.stdout).strip()[:200])
    except (OSError, subprocess.TimeoutExpired) as e:
        errs.append(f"patch: {e}")

    # 2) git apply --recount: recomputes hunk line counts (handles a model
    #    that got the @@ numbers wrong) — works even outside a git repo.

    try:
        r = subprocess.run(
            ["git", "apply", "--recount", "--unidiff-zero",
             "--ignore-whitespace", "-p0", "--directory", str(fpath.parent)],
            input=diff_text, capture_output=True, text=True,
            cwd=PROJECT_ROOT, timeout=20)

        if r.returncode == 0:
            return f"OK: {relpath} patched (git apply)"

        errs.append("git: " + (r.stderr or r.stdout).strip()[:200])
    except (OSError, subprocess.TimeoutExpired) as e:
        errs.append(f"git: {e}")

    return (f"ERROR: could not apply the diff to {relpath} ({' | '.join(errs)}). "
            f"Make a SMALL edit_file (exact old_text/new_text) instead, or "
            f"write_file with the full new content.")


def is_excluded_path(p):
    """Third-party / snapshot trees we must NEVER auto-write into."""
    from harness import target_policy

    return target_policy.is_snapshot(str(p))


def _capture_checkpoint_path(context, path):
    """Capture only after mutation authorization and immediately before I/O."""

    if context is None or context.checkpoint_manager is None or context.checkpoint is None:
        return

    item = context.checkpoint_manager.capture(context.checkpoint, path)
    context.trace.emit(
        EventType.CHECKPOINT_PATH_CAPTURED, context.task_id,
        status=EventStatus.OK, action_id=context.action_id,
        metadata={"checkpoint_id": context.checkpoint.checkpoint_id,
                  "path": item.relative_path,
                  "original_exists": item.original_exists},
    )


def _forget_cached_reads(context, keep=None):
    """Drop this turn's cached command output after something wrote a file.

    The cache exists so a model that re-runs `cat x` gets told it already has
    the answer. After a mutation that answer is a LIE, and the model has no way
    to know: it re-ran `sed -n '1,20p' user_space.rst` after each of three
    splices, was handed the pre-edit text every time, concluded each write had
    failed, and spliced again — ending with three toctrees, a duplicated
    paragraph and a lost line. A stale read is worse than no cache.

    Only the command results go. The sentinels stay: the sandbox does not come
    back, a refusal is still a refusal, and the read-path set is cumulative.
    """

    if context is None:
        return

    for key in [key for key in context.cache
                if isinstance(key, str) and key != keep]:
        del context.cache[key]


def _record_mutation(context, path):
    """One hook for "a file just changed": checkpoint it, and forget stale reads."""
    _forget_cached_reads(context)

    if context is None or context.checkpoint_manager is None or context.checkpoint is None:
        return

    context.checkpoint_manager.record_mutation(
        context.checkpoint, path, context.action_id or new_action_id("mutation"),
    )
