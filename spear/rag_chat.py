#!/usr/bin/env python3
"""
RAG-augmented chat with a local Qwen3 MoE (llama-server, native tool calling).
Claude Code-style UI (⏺ bullets, ⎿ tool results, spinner, box).
Hybrid tool system: explicit !commands + auto-detection + LLM tool calls.
"""

import os
import re
import sys
import json
import fcntl
import signal
import select
import shlex
import time
import hashlib
import difflib
import atexit
import readline
import threading
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
import answer_scope
import embedding
import evidence_handles
import skill_library
import progress_monitor
import requirement_set
import tool_router
import standard_scope
import web_fetch
import work_phase
from datetime import datetime, timezone
from openai import (OpenAI, APIStatusError, APIConnectionError, APIError)
from model_backend import (AnthropicBackend, ConversationMessage,
                           ModelBackendConfigurationError,
                           anthropic_credentials_available,
                           OpenAICompatibleBackend, TextBlock, ToolResultBlock)
from memory_store import MarkdownMemoryStore, MemoryScope, MemorySource, MemoryStoreError
from cancellation import CancellationSource, NEVER_CANCELLED
from checkpoint import CheckpointManager
from tool_runtime import (AuditLogger, CommandClassification, CommandPolicy,
                          BubblewrapSandbox, CommandRunner, ExecutionMode, ExecutionProfile,
                          Capability, CapabilityPolicy, DEFAULT_CAPABILITY_POLICY,
                          PathNotFoundError,
                          PathPolicyError, SandboxSpec, ToolPolicy, ToolResult,
                          Workspace, effective_mount_root, shell_argv)
from tracing import (EventStatus, EventType, create_trace_emitter,
                     new_action_id, new_task_id)
from working_state import StateEventType, WorkingState
from context_engine import (ContextEngine, ContextItem, ContextLayer, Freshness,
                            working_state_context_item)
from compaction import CompactionMode, CompactionPolicy
from agent_runtime import (
    AgentContext, AgentRuntime, changed_files,
    strip_fabrications, turn_evidence, unverified_change, validate_agent_turn,
    verify_demand,
)
import session_replay
from budgets import BudgetKind, BudgetLimit, BudgetManager
from diff_evidence import DiffEvidenceService
from task_controller import TaskController, TaskRequest, TaskStatus
from result_store import ResultStore
from session_store import (
    FileSessionStore, SessionConfiguration, SessionEventType, SessionHandle,
    new_session_id,
)
from tool_registry import (
    ToolCategory, ToolMutability, ToolRegistry, ToolSpec, native_tool_specs,
)
from tool_router import (
    ToolExecutionContext, ToolHandlerResult, ToolResultStatus, ToolRouter,
    invalidates_reads,
)
from verification import VerificationHints, VerificationPolicy
from training_store import TrainingStore
import project_build
from standard_commands import (
    StandardCommandError, StandardOperator, handle_standard_command,
    retrieval_summary, standard_help_lines,
)
from standard_progress import TerminalProgress
from standard_store import StandardStore
from standard_tools import StandardToolService


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

# The federation opened for this session, bound in main(). search_corpus needs
# it at tool-dispatch time, where the main loop's local is out of reach.

COLLECTION = None

# ── session settings, as flags ────────────────────────────────────────
# These were reachable only by exporting an environment variable, which is
# fine for a machine-wide default in machine.env and wrong for a per-run
# choice: nothing in --help mentioned them, and configuring one run meant
# `VAR=x VAR2=y spear-chat`, an incantation that is neither discoverable nor
# reviewable in shell history alongside the other flags.
#
# They are applied into os.environ HERE, at the top of the module, because
# several are read while it is still being imported (STATE_DIR, CORPORA_ROOT,
# DB_PATH). An explicit flag wins over an inherited variable, the same way
# --model wins over the remembered backend choice.

ENV_OPTIONS = {
    "--api-base": ("SPEAR_API_BASE", "OpenAI-compatible endpoint URL"),
    "--ctx": ("SPEAR_CTX", "context window in tokens (default 32768)"),
    "--max-tokens": ("SPEAR_MAX_TOKENS", "cap on one reply"),
    "--temp": ("SPEAR_TEMP", "sampling temperature (default 0.25)"),
    "--max-tool-rounds": ("SPEAR_MAX_TOOL_ROUNDS", "tool rounds per task"),
    "--max-commands": ("SPEAR_MAX_COMMANDS", "bash commands per task"),
    "--operator": ("SPEAR_OPERATOR", "operator name recorded in the audit trail"),
    "--state-dir": ("SPEAR_STATE_DIR", "where the session ACCUMULATES"),
    "--corpus-root": ("SPEAR_CORPUS_ROOT", "prefix for relative corpus paths"),
    "--db-path": ("SPEAR_DB_PATH", "chromadb directory"),
    "--embed-remote": ("SPEAR_EMBED_REMOTE", "ssh host that embeds the corpora"),
    "--trace-file": ("SPEAR_TRACE_FILE", "runtime trace destination"),
    # Recording and replaying a session were reachable only through
    # environment variables, which is to say they were reachable only by
    # someone who already knew they existed. They are ordinary operator
    # tools: one records what the model said, the other hands it back for
    # free so a harness change can be judged without paying for the model
    # twice.
    "--record": ("SPEAR_RECORD_TURNS",
                 "record this session's model turns to a file"),
    "--replay": ("SPEAR_REPLAY_TURNS",
                 "replay a recorded file instead of calling the model"),
    "--standard-embed-model": ("SPEAR_STANDARD_EMBED_MODEL",
                               "embedding model for /standard indexes"),
    "--standard-embed-revision": ("SPEAR_STANDARD_EMBED_REVISION",
                                  "its pinned revision (required with the model)"),
    "--standard-embed-remote": ("SPEAR_STANDARD_EMBED_REMOTE",
                                "ssh host allowed to embed a PUBLIC standard"),
    "--standard-embed-device": ("SPEAR_STANDARD_EMBED_DEVICE",
                                "cpu (default) or cuda, for a local standard build"),
}

# Flags that carry no value: presence is the setting.
ENV_SWITCHES = {
    "--trace": ("SPEAR_TRACE", "1", "record a runtime JSONL trace"),
}


def apply_env_options(argv):
    """Move the settings flags into the environment, and out of argv.

    Out of argv deliberately: their VALUES would otherwise sit there as bare
    words, and several places in this file test membership against sys.argv.
    """
    remaining = []
    iterator = iter(argv)

    for item in iterator:
        if item in ENV_SWITCHES:
            variable, value, _ = ENV_SWITCHES[item]
            os.environ[variable] = value

            continue

        if item in ENV_OPTIONS:
            value = next(iterator, None)

            if value is None:
                raise SystemExit(f"{item} needs a value")

            os.environ[ENV_OPTIONS[item][0]] = value

            continue

        remaining.append(item)

    return remaining


sys.argv[1:] = apply_env_options(sys.argv[1:])


# Endpoint is configurable so spear-chat can target a LOCAL llama-server
# (:8080) or a REMOTE pod's vLLM via an SSH tunnel (:8081). See spear-chat.sh.
# Every path below is anchored on this file's own location. Absolute literals
# went stale the day the tree was relocated and the app stopped starting;
# deriving them means a future move costs nothing.

APP_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)

LLAMA_SERVER_URL = os.environ.get("SPEAR_API_BASE", "http://127.0.0.1:8080/v1")
DB_PATH = os.environ.get("SPEAR_DB_PATH") or f"{APP_DIR}/chromadb"


def resource_dir(env_var, name):
    """A SHIPPED resource directory, relocatable by environment.

    rules.d/, skills/ and benches/ are content, not code: a deployment's own
    rules, its learned skills and the bench that rates it are exactly the
    material that must not sit in a public tree, and pinning them to APP_DIR
    left the only copy outside the checkout unreachable with no way to say so.

    The default is the in-tree directory, so a plain checkout behaves exactly
    as before and no default ever points outside it. An empty value counts as
    unset -- an exported-but-empty variable is a mistake, not a request to
    read the filesystem root.
    """
    return os.environ.get(env_var) or f"{APP_DIR}/{name}"


RULES_DIR = resource_dir("SPEAR_RULES_DIR", "rules.d")

# Everything the session ACCUMULATES -- history, memories, trajectories, the
# audit trail, the input history -- against everything the session SHIPS with:
# code, rules, skills, the index. They were the same directory, which is fine
# on a workstation and lossy in a container, where the app lives in a
# read-only-ish image and `docker run --rm` throws the accumulation away.
# Defaults to APP_DIR, so nothing moves unless asked.

STATE_DIR = os.environ.get("SPEAR_STATE_DIR", APP_DIR)
os.makedirs(STATE_DIR, exist_ok=True)

# All of these accumulate, so all of them live under STATE_DIR: the audit
# trail, the runtime trace, the tool results a later turn may re-read, the
# session records and the file checkpoints an undo restores from. Rooting them
# at APP_DIR instead would put a session's evidence inside the image and lose
# it with the container that produced it -- which is the one place this
# evidence is worth having.
# Rules the USER taught, as opposed to the ones the harness ships. Under
# STATE_DIR and not rules.d/, for the reason every other accumulating path is:
# rules.d/ lives in the image, and a container would throw these away on --rm
# while refusing the write in the first place. Loaded after the shipped rules,
# so a later line can qualify an earlier one.

LEARNED_RULES_FILE = f"{STATE_DIR}/rules-learned.md"

# Soft limit on everything rules.d injects, shipped plus learned. Not a cap:
# the rules are the one context the model is guaranteed to see, so the harness
# warns and lets the operator decide rather than silently dropping a rule.
# Raised from 4000 once the shared documentation conventions moved in here: at
# 3557 of 4000 the next /recall would have tripped the warning, and the answer
# to "the always-injected set is nearly full" is not to keep shaving the rules
# that earned their place. 8000 characters is roughly 2k tokens against a
# 65 536-token window -- 3 %, paid on every request, for the only context the
# model cannot fail to see.

RULES_BUDGET = 8000

AUDIT_LOGGER = AuditLogger(f"{STATE_DIR}/audit/tool-actions.jsonl")
TRACE = create_trace_emitter(f"{STATE_DIR}/audit/runtime-trace.jsonl")
CONTEXT_ENGINE = ContextEngine()
RESULT_STORE = ResultStore(f"{STATE_DIR}/audit/tool-results")
SESSION_STORE = FileSessionStore(f"{STATE_DIR}/audit/sessions")
CHECKPOINT_ROOT = f"{STATE_DIR}/audit/checkpoints"
TRAINING_CAPTURE_ENABLED = os.environ.get(
    "SPEAR_TRAINING_CAPTURE", "1"
).strip().lower() not in {"0", "false", "no", "off"}
TRAINING_STORE = (TrainingStore(f"{STATE_DIR}/audit/training-data")
                  if TRAINING_CAPTURE_ENABLED else None)
# SPEAR_STANDARDS_ROOT selects a different standards store without moving
# anything else: see state_paths.standards_root, which honours the same
# variable for the readers that do not go through this module.
STANDARD_STORE = StandardStore(
    os.environ.get("SPEAR_STANDARDS_ROOT") or f"{STATE_DIR}/standards")
STANDARD_OPERATOR = StandardOperator(STANDARD_STORE, progress=TerminalProgress())
STANDARD_TOOL_SERVICE = StandardToolService(STANDARD_STORE)

# Has any turn of this session engaged the bound standard? It widens the scope
# test for follow-ups -- a question that refers back to a bound turn rather
# than restating its subject. Session state, so /clear puts it back.

STANDARD_ENGAGED_BEFORE = False

# What the operator !read in front of the next question. "carry out the task
# in doc/ack-task.md" names no standard; the file it points at does, in its
# third paragraph. Consumed by the next turn's scope decision, then cleared.

STANDARD_READ_CONTEXT = ""

# Clauses the previous turn of this session retrieved. Two prompts is the
# normal shape here -- "how should X work?", then "change the code" -- and the
# reasoning was done in the first. Carrying the sections it actually read into
# the second is what stops the change from being justified against clauses
# nobody opened: a turn that read only glossary sections still cited the one
# that governs acknowledgements.

STANDARD_PRIOR_CLAUSES = ()

# And what the previous turn CONCLUDED, not only which clauses it opened. The
# two-prompt shape is "how should X work?" then "change the code": the first
# answer is the specification, written by the model itself from the bound
# standard, and the second turn was starting from nothing. Carried whole, so
# the requirements the change must meet are the ones already established.

STANDARD_PRIOR_ANSWER = ""

# ...and what it ESTABLISHED, as provisions rather than as prose. The prose
# carry was the whole contract until it was measured: on the two runs whose
# first answer the identifier guard withheld, the follow-up inherited an
# assessment record, started from nothing, and rebuilt an arbitrary subset of
# the document. This is built from the evidence ledger instead, so it survives
# a withheld answer -- see requirement_set.publish.

STANDARD_PRIOR_REQUIREMENTS = requirement_set.RequirementSet()
TRACE_PROVIDER = None
TRACE_MODEL = None


def operator_training_controller():
    """Build the operator control plane; never exposed through ToolRegistry."""
    from training_controller import TrainingController
    from training_launcher import TrainingExecutionConfiguration

    config_path = Path(STATE_DIR) / "training-execution.json"
    execution = (TrainingExecutionConfiguration.load(config_path)
                 if config_path.is_file() else None)
    store = TRAINING_STORE or TrainingStore(f"{STATE_DIR}/audit/training-data")

    return TrainingController(store, f"{STATE_DIR}/audit/training-data",
                              execution=execution)

# ── Projects ─────────────────────────────────────────────────────────
# One unified concept: a PROJECT is a working tree with its own corpus
# (ChromaDB collection), history and memories.
#
# What a corpus DOES is declared, not inferred from its type. `kind` is a
# label -- it scopes skills and prints in the listing -- and each behaviour
# has its own key:
#   collection   name an existing index instead of deriving one from the path
#   indexer      "buildsystem" (curated BitBake/Yocto walk) or "generic"
#   autoindex    build a missing index on first sight (default: no)
#   prompt_file  a domain prompt for this corpus (default: the generic one)
#
# One key used to imply all four. It meant a registry entry did not say what
# its corpus did, and that the four could not be used apart.
# The registry lives in projects.json: {name: {"path": ..., "kind": ...}}.
# Manage it with `spear-corpus add/list/rm`.

PROJECTS_FILE = f"{APP_DIR}/projects.json"

# No corpus is built in. Two used to be -- a pair of product checkouts under
# one organisation's home directory, registered automatically wherever the
# directory happened to exist -- and a platform that ships somebody's tree as
# a first-class corpus is a platform shipping their tree. Corpora are declared
# by whoever owns them, with `/corpus add` or projects.json, and the registry
# below does not care where they came from.


# Where a RELATIVE corpus path is rooted. Absolute paths are untouched, so the
# workstation registry keeps working exactly as before; a container ships a
# registry of relative paths instead and mounts the trees under one root. That
# is what makes the same projects.json usable by someone whose checkouts are
# not under /home/operator.
# NB: not CORPUS_ROOT -- that name is already the *current* corpus's root,
# rebound by set_project() on every switch. This one is a fixed prefix for the
# registry and must never move.
# Defaults to the REPOSITORY root, so a corpus that lives inside the repo is
# registered as `llama.cpp-next`, not `/opt/llm/spear/llama.cpp-next`, and
# a clone anywhere finds it. A container overrides it with the mount root.
# Absolute paths are untouched either way: the trees outside the repo are
# genuinely machine-specific and stay spelled out.

CORPORA_ROOT = os.environ.get("SPEAR_CORPUS_ROOT") or ROOT_DIR


def resolve_corpus_path(path):
    """Absolute path of a registered corpus, honouring SPEAR_CORPUS_ROOT."""

    if os.path.isabs(path) or not CORPORA_ROOT:
        return path

    return os.path.normpath(os.path.join(CORPORA_ROOT, path))


def load_projects():
    """Return {name: {"path","kind", ...}}. Tolerates the legacy {name: path} format (treated as
    generic). Extra keys (exclude, include_build — see reindex_options) are
    PRESERVED: rebuilding specs from path/kind alone erased them at the first
    save_projects()."""

    try:
        with open(PROJECTS_FILE) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        raw = {}

    projects = {}

    for name, val in raw.items():
        if isinstance(val, str):
            projects[name] = {"path": val, "kind": "generic"}
        else:
            spec = dict(val)
            spec["kind"] = spec.get("kind", "generic")
            projects[name] = spec

    for spec in projects.values():
        spec["path"] = resolve_corpus_path(spec["path"])

    return projects


def save_projects(projects):
    try:
        with open(PROJECTS_FILE, "w") as f:
            json.dump(projects, f, indent=2)
    except OSError as exc:
        # The registry is shipped configuration, not session state: in a
        # container it lives in the image and is not the user's to change.
        # Say where it has to be done instead of raising a stack trace.

        raise SystemExit(
            f"cannot write the corpus registry ({PROJECTS_FILE}): {exc}\n"
            f"Registering a corpus is a host operation — do it there, then "
            f"rebuild the image (scripts/docker/build.sh). A one-off tree can be "
            f"reached with --corpora instead.") from exc


def detect_project(projects):
    """Project whose path contains the cwd, or None."""
    cwd = os.path.realpath(os.getcwd())
    best = None

    for name, spec in projects.items():
        p = os.path.realpath(spec["path"])

        if cwd == p or cwd.startswith(p + "/"):
            if best is None or len(p) > len(projects[best]["path"]):
                best = name

    return best


# Snapshot trees a build system leaves beside the live ones. Registered,
# indexed, and never what a question is about.

_SNAPSHOT_SUFFIXES = (".back", ".pristine", ".0")


def corpora_below(projects, cwd=None):
    """Every registered corpus sitting under the cwd.

    Launching in an umbrella checkout used to pick ONE child and chdir into it:
    from ~/soo/so3 the session moved into ~/soo/so3/so3 and lost avz/, build/,
    doc/ and the scripts -- half of what the questions are about -- while the
    orientation map, written for the umbrella, pointed at paths that no longer
    existed from there. The tools belong at the umbrella root, where every part
    is reachable, and the parts belong in the federation, where each keeps its
    own index and contributes its own best hits.
    """
    cwd = os.path.realpath(cwd or os.getcwd())
    found = {}

    for name, spec in projects.items():
        path = os.path.realpath(spec.get("path", ""))

        if path == cwd or not path.startswith(cwd + os.sep):
            continue

        if name.endswith(_SNAPSHOT_SUFFIXES) or path.endswith(_SNAPSHOT_SUFFIXES):
            continue

        found[name] = path

    return found


def enclosing_corpus(projects, cwd=None):
    """Registered corpus sitting under the cwd, when there is an obvious one.

    detect_project answers "is the cwd INSIDE a corpus". The opposite happens
    just as often and used to leave the session with no RAG at all: launched
    from ~/soo/so3, whose corpus is ~/soo/so3/so3, the chat fell back to an
    ad-hoc index of the parent — which does not exist — and the model answered
    from memory. A passive warning was printed and routinely missed.

    Ambiguity is the reason this cannot be a plain "first child wins":
    ~/soo/so3 holds six registered corpora (so3, u-boot, avz, qemu, atf,
    avz.back). Two signals are unambiguous enough to act on, and nothing else
    is:
      - exactly ONE registered corpus directly below the cwd;
      - or one whose basename matches the cwd's, the umbrella-checkout shape
        (~/soo/so3 -> ~/soo/so3/so3).
    Anything else returns None with the candidates, so the caller can name
    them instead of guessing.
    """
    cwd = os.path.realpath(cwd or os.getcwd())
    children = {}

    for name, spec in projects.items():
        p = os.path.realpath(spec["path"])

        if p != cwd and p.startswith(cwd + os.sep):
            children[name] = p

    direct = {n: p for n, p in children.items()
              if os.path.dirname(p) == cwd}

    if len(direct) == 1:
        return next(iter(direct)), sorted(direct)

    same_name = [n for n, p in direct.items()
                 if os.path.basename(p) == os.path.basename(cwd)]

    if len(same_name) == 1:
        return same_name[0], sorted(direct)

    return None, sorted(direct or children)


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


_SCAN_EXTS = {".c", ".h", ".S", ".cpp", ".hpp", ".cc", ".py", ".sh", ".md",
              ".rst", ".dts", ".dtsi", ".cfg", ".conf", ".bb", ".inc",
              ".yaml", ".yml", ".mk"}
_SCAN_SKIP = {".git", "build", "out", "tmp", "__pycache__", "node_modules"}


def _scan_count(path, cap=20000):
    """(files, capped) — capped says the walk stopped early, so `files` is a
    floor and not a count. It matters in what the split prints: agency was
    announced as "20017 files" when it holds 66246, a whole kernel tree, and
    that number is what a reader uses to decide whether it fits in one index.
    """
    c = 0

    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in _SCAN_SKIP]
        c += sum(1 for f in files
                 if f in ("Makefile", "Kconfig", "CMakeLists.txt")
                 or os.path.splitext(f)[1] in _SCAN_EXTS)

        if c > cap:
            return c, True

    return c, False


def workspace_components(path, projects, min_files=300):
    """The sub-trees of `path` big enough to deserve a corpus of their own.

    One (name, abspath, files, registered_as) per component, `registered_as`
    naming the corpus that already owns the tree, or None. Shared by the
    launch-time split offer and by `/corpus scan`, which used to carry a second
    copy of this walk inside a shell script — a copy whose skip list and
    extension set had already drifted from this one.

    An already-registered tree is reported without being counted: it is listed
    for information, and walking a kernel tree to say "already: agency" would
    cost seconds at every launch.
    """
    known = {os.path.realpath(s["path"]): n for n, s in projects.items()}
    comps = []

    for sub in sorted(os.listdir(path)):
        p = os.path.join(path, sub)

        if not os.path.isdir(p) or sub in _SCAN_SKIP:
            continue

        rp = os.path.realpath(p)
        owner = known.get(rp)

        if owner:
            comps.append((sub, rp, None, owner))
            continue

        n, capped = _scan_count(p)

        if n >= min_files:
            comps.append((sub, rp, f"{n}+" if capped else str(n), None))

    return comps


def offer_workspace_split(path, projects, min_files=300):
    """When an unregistered dir is really a multi-component workspace (several
    big sub-trees), offer to register each component as its own corpus — so
    a huge upstream tree (u-boot, qemu...) never dilutes the index. Returns a
    chosen project spec dict, or None to fall through to plain ad-hoc."""

    if not sys.stdin.isatty():
        return None

    # A virtualenv root is not a multi-component workspace: its lib/ and lib64/
    # are site-packages, and offering to index them as corpora is noise.

    if os.path.exists(os.path.join(path, "pyvenv.cfg")):
        return None

    comps = [c for c in workspace_components(path, projects, min_files)
             if c[3] is None]

    if len(comps) < 2:
        return None

    print(f"{C_WARN}This looks like a multi-component workspace "
          f"({len(comps)} big sub-trees):{C_RST}")

    for sub, _, n, _owner in comps:
        print(f"  {C_DIM}·{C_RST} {sub} ({n} files)")

    print("Register each as its own corpus (recommended — keeps indexes "
          "focused)?")
    print(f"  {C_ACCENT}❯{C_RST} 1. Yes, register them {C_DIM}(Enter){C_RST}")
    print(f"    2. No, index this whole directory as one")

    try:
        ans = input("  ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    if ans not in ("", "1", "o", "y", "oui", "yes"):
        return None

    base = os.path.basename(path.rstrip("/"))
    reg = load_projects()
    names = []

    for sub, p, _n, _owner in comps:
        name = sub if sub not in reg else f"{base}-{sub}"
        reg[name] = {"path": p, "kind": "generic"}
        names.append(name)

    save_projects(reg)
    print(f"{C_OK}Registered: {', '.join(names)}{C_RST}")
    print("Which one now? (number, or Enter to pick later)")

    for i, n in enumerate(names, 1):
        print(f"  {i}. {n}")

    try:
        a = input("  ").strip()
    except (EOFError, KeyboardInterrupt):
        a = ""

    if a.isdigit() and 1 <= int(a) <= len(names):
        n = names[int(a) - 1]
        return {"name": n, **reg[n]}

    return None


def resolve_project_at_startup():
    """The CURRENT directory decides everything — no prompt:
      1. explicit --corpus <name> (or --here) wins;
      2. else if the cwd is inside a registered corpus, use it;
      3. else ad-hoc on the cwd (offering a workspace split if it is a
         multi-component tree).
    Tools always run in the cwd regardless; the corpus only picks the RAG
    index/history/memories."""
    projects = load_projects()

    if "--here" in sys.argv[1:]:
        split = offer_workspace_split(os.path.realpath(os.getcwd()), projects)

        if split:
            return split

        return {"name": "adhoc:" + os.path.basename(os.getcwd().rstrip("/")),
                "path": os.getcwd(), "kind": "generic"}

    # --corpus (and the --project/--checkout aliases) forces a registered corpus

    for flag in ("--corpus", "--project", "--checkout"):
        if flag in sys.argv[1:]:
            i = sys.argv.index(flag)

            if i + 1 < len(sys.argv) and sys.argv[i + 1] in projects:
                n = sys.argv[i + 1]
                return {"name": n, **projects[n]}

            print(f"{C_ERR}Unknown corpus (registered: "
                  f"{', '.join(projects) or 'none'}){C_RST}")
            sys.exit(1)

    # cwd inside a registered corpus → use it

    found = detect_project(projects)

    if found:
        return {"name": found, **projects[found]}

    # cwd ABOVE registered corpora → an umbrella session: tools stay HERE, and
    # every part below is federated. The prefix machinery in federated_corpora
    # rewrites each chunk's path so what the model reads is what bash can open,
    # which is what used to force the chdir.

    below = corpora_below(projects)

    if below:
        cwd = os.path.realpath(os.getcwd())
        print(f"{C_DIM}  ⎿  {len(below)} registered corpora sit under this "
              f"directory ({', '.join(sorted(below))}) — tools run here, "
              f"retrieval spans them. --corpus <name> narrows to one.{C_RST}")

        return {"name": "workspace:" + (os.path.basename(cwd) or "root"),
                "path": cwd, "kind": "generic", "corpora": sorted(below),
                # Discovered, not declared. Declared federations and shared
                # corpora are a deliberate choice and stay attached every turn;
                # these eight are whatever happened to be registered under the
                # cwd, so they are selected per question instead.
                "auto_corpora": sorted(below)}

    # cwd outside any corpus → ad-hoc on the cwd (no picker). Offer a split
    # if it is a big multi-component workspace.

    if sys.stdin.isatty():
        split = offer_workspace_split(os.path.realpath(os.getcwd()), projects)

        if split:
            return split

    return {"name": "adhoc:" + os.path.basename(os.getcwd().rstrip("/")) or "cwd",
            "path": os.getcwd(), "kind": "generic"}


PROJECT_SPEC = {}

# Bound at import for piped/module use; re-resolved in main() for the CLI.
# The current directory, because that is the one thing true of every
# deployment -- it used to be a named checkout that existed on one machine.

set_project({"name": "adhoc:" + (os.path.basename(os.getcwd().rstrip("/"))
                                 or "cwd"),
             "path": os.getcwd(), "kind": "generic"})
TOP_K = 12
MAX_CONTEXT_CHARS = 12000
MAX_HISTORY = 80          # total kept on disk (rotation)
HISTORY_INJECT = 40       # messages re-injected into the prompt each turn

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


def model_provider_from_argv(argv):
    """Read the small provider surface without changing legacy Qwen defaults."""
    provider = "openai-compatible"
    model = None
    index = 0

    while index < len(argv):
        arg = argv[index]

        if arg == "--provider" and index + 1 < len(argv):
            provider = argv[index + 1]
            index += 1
        elif arg.startswith("--provider="):
            provider = arg.split("=", 1)[1]
        elif arg == "--model" and index + 1 < len(argv):
            model = argv[index + 1]
            index += 1
        elif arg.startswith("--model="):
            model = arg.split("=", 1)[1]

        index += 1

    if provider not in {"openai-compatible", "anthropic"}:
        raise ModelBackendConfigurationError(
            "--provider must be openai-compatible or anthropic"
        )

    return provider, model


ANT_INSTALL_HINT = (
    "install it from https://github.com/anthropics/anthropic-cli/releases "
    "(linux_amd64 tarball, extract `ant` into ~/.local/bin)")


def offer_anthropic_login():
    """Offer an interactive `ant auth login` when no credential resolves.

    Opening a browser is an outward-facing side effect, so it is proposed and
    never automatic, and only on a TTY — a non-interactive run fails closed
    with the manual instructions instead of hanging on a prompt nobody sees.

    The login runs as a plain supervisor-side subprocess, deliberately outside
    the tool sandbox: it needs the network, a browser, and write access to
    ~/.config/anthropic, none of which a tool call is ever granted.  It is
    session setup, in the same class as starting the model server.
    """

    if anthropic_credentials_available():
        return

    if not sys.stdin.isatty():
        raise ModelBackendConfigurationError(
            "no Anthropic credential and no terminal to open a session: "
            "export ANTHROPIC_API_KEY, or run `ant auth login` first")

    print(f"{C_WARN}No Anthropic credential found "
          f"(no ANTHROPIC_API_KEY, no `ant` session).{C_RST}")
    print("Open a browser to sign in now?")
    print(f"  {C_ACCENT}❯{C_RST} 1. Yes, run `ant auth login` {C_DIM}(Enter){C_RST}")
    print("    2. No, abort")

    try:
        answer = input("  ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "2"

    if answer not in ("", "1", "o", "y", "oui", "yes"):
        raise ModelBackendConfigurationError("Anthropic sign-in declined")

    ant = shutil.which("ant")

    if ant is None:
        raise ModelBackendConfigurationError(f"`ant` is not on PATH — {ANT_INSTALL_HINT}")

    # No timeout and no output capture: the CLI prints the authorize URL when
    # it cannot reach a browser, and that text is the user's only way through.

    try:
        subprocess.run([ant, "auth", "login"], check=False)
    except OSError as exc:
        raise ModelBackendConfigurationError(f"could not run `ant auth login`: {exc}")

    if not anthropic_credentials_available():
        raise ModelBackendConfigurationError(
            "sign-in did not complete — check `ant auth status`")

    print(f"{C_OK}Anthropic session open.{C_RST}")


@contextmanager
def interruptible(source):
    """Make Ctrl+C cancel the running turn instead of doing nothing.

    The runtime already polls a cancellation token at six points -- between
    rounds, before a tool call, after one -- and ends the turn cleanly when
    it is set. Nothing ever set it: AgentContext defaults to NEVER_CANCELLED
    and the CLI never passed anything else, so the cooperative path existed
    and was never armed. Measured, not assumed: SIGINT to the interpreter
    mid-turn left the turn running and the prompt gone, thirty seconds on.

    The FIRST interrupt asks the turn to stop, which lets the harness write
    its session, keep what the turn already did, and answer. A SECOND one
    restores Python's own handler, so a turn wedged somewhere that never
    polls can still be killed the ordinary way rather than trapping the
    operator in their own shell.
    """
    def on_interrupt(signum, frame):
        if source.cancel("interrupted by the operator"):
            print(f"\n{C_DIM}⏺ stopping this turn — ctrl+c again to force"
                  f"{C_RST}", flush=True)

            return

        signal.signal(signal.SIGINT, previous)
        raise KeyboardInterrupt

    stop = threading.Event()

    def watchdog():
        """Cancel a turn that has gone quiet, since it cannot cancel itself."""
        while not stop.wait(5):
            if LIVENESS.silent_for() < TURN_LIVENESS_SECONDS:
                continue

            if source.cancel(f"no sign of life for "
                             f"{int(LIVENESS.silent_for())}s"):
                print(f"\n{C_DIM}⏺ nothing has happened for "
                      f"{TURN_LIVENESS_SECONDS}s — stopping this turn{C_RST}",
                      flush=True)

            return

    LIVENESS.touch()
    watcher = threading.Thread(target=watchdog, daemon=True)
    watcher.start()

    try:
        previous = signal.signal(signal.SIGINT, on_interrupt)
    except ValueError:
        # Not the main thread; leave the default handler alone.
        try:
            yield
        finally:
            stop.set()

        return

    try:
        yield
    finally:
        stop.set()
        signal.signal(signal.SIGINT, previous)


def create_model_backend(argv):
    """Provider construction only; tools and security stay outside this boundary."""
    provider, model = model_provider_from_argv(argv)

    # A replayed session needs no provider at all: no login is offered, no
    # server is contacted, no key is read. Checked before anything else so a
    # replay costs nothing and can run offline.

    if os.environ.get(session_replay.REPLAY_ENV, "").strip():
        return session_replay.wrap(None), provider

    if provider == "anthropic":
        offer_anthropic_login()
        return session_replay.wrap(
            AnthropicBackend(model=model or "claude-sonnet-5")), provider

    return session_replay.wrap(OpenAICompatibleBackend(
        OpenAI(base_url=LLAMA_SERVER_URL, api_key="not-needed"), model=model
    )), provider


# Safe is now the default.  Keep BYPASS_PERMISSIONS as a legacy display flag;
# policy decisions below use EXECUTION_MODE directly.

EXECUTION_MODE = execution_mode_from_argv(sys.argv[1:])
CAPABILITY_POLICY = capability_policy_from_argv(sys.argv[1:])
BYPASS_PERMISSIONS = EXECUTION_MODE == ExecutionMode.AUTO

C_TOOL = "\033[36m"
C_WARN = "\033[33m"
C_OK = "\033[32m"
C_ERR = "\033[31m"
C_RST = "\033[0m"
C_DIM = "\033[2m"
C_BOLD = "\033[1m"
C_ACCENT = "\033[38;5;77m"    # the platform green
C_CODE = "\033[38;5;114m"     # soft green for inline code


# ── Claude Code-style UI ─────────────────────────────────────────────

ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def vlen(s):
    """Visible length (ANSI codes stripped)."""
    return len(ANSI_RE.sub("", s))


# The spinner runs in its own thread and repaints with "\r\033[K" -- carriage
# return, then erase to end of line. Anything another thread is midway through
# printing on that line is erased with it, and the line simply never appears.
# That is how an `edit_file` that HAD applied showed a diff and then no result
# at all: the model, told nothing, abandoned the tool that had just worked and
# went off rewriting the file with shell splices. Every terminal write that
# must survive takes this lock; the spinner holds it while it repaints.

TERMINAL_LOCK = threading.RLock()


class StatusLine:
    """The one line the harness uses to say what it is doing right now.

    Two rules, and both come from watching it get them wrong.

    IT IS NEVER BLANKED. The old spinner erased its line when an activity
    ended, so a one-second call followed by three seconds of silence showed
    the label, wiped it, and left the operator staring at nothing during the
    part they most wanted explained. The text stays until something replaces
    it.

    IT IS NEVER STACKED. The first attempt at a fix left each finished
    activity behind on its own line, and a turn then trailed a column of
    "Thinking… 4s" above the work. There is one status line. A new activity
    overwrites it; real output erases it and takes the line for itself.

    Erasing on real output is why this exists as an object rather than a
    convention: every print in this file would otherwise have to remember,
    and one that forgot would append to a spinner mid-frame.
    """

    def __init__(self):
        self.pending = False
        self.writing = False

        # One clock for the whole turn. Each round used to start a fresh
        # spinner, and since the label does not change between rounds
        # ("Analyzing results…" over and over) the only visible signal was a
        # counter jumping back to zero -- which reads as a restart when the
        # turn is simply still going. It runs from the prompt to the answer.
        self.turn_t0 = None

    def begin_turn(self):
        self.turn_t0 = time.time()

    def end_turn(self):
        self.turn_t0 = None

    def elapsed(self, fallback):
        """Seconds to show: the turn's, when a turn is running."""
        return time.time() - (self.turn_t0 if self.turn_t0 is not None
                              else fallback)

    def show(self, text):
        with TERMINAL_LOCK:
            self.writing = True

            try:
                sys.stdout.write("\r\033[K" + text)
                sys.stdout.flush()
                self.pending = True
            finally:
                self.writing = False

    def clear(self):
        """Take the line back, for something that needs to keep it."""
        with TERMINAL_LOCK:
            if not self.pending:
                return

            self.writing = True

            try:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
                self.pending = False
            finally:
                self.writing = False


STATUS = StatusLine()


class _StatusAwareStdout:
    """Anything printed takes the status line back before it prints.

    A single point of control, so no caller has to remember. The status
    line's own writes are exempt, or clearing would recurse forever.
    """

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        if text and not STATUS.writing and STATUS.pending:
            STATUS.clear()

        return self._stream.write(text)

    def flush(self):
        return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


sys.stdout = _StatusAwareStdout(sys.stdout)


@contextmanager
def terminal_output():
    with TERMINAL_LOCK:
        yield


class Spinner:
    """Animated spinner during LLM calls, Claude Code style:
    ✻ Thinking… (3s · ctrl+c to interrupt)"""
    FRAMES = "·✢✳✶✻✶✳✢"

    def __init__(self, label="Thinking…"):
        self.label = label
        self._stop = threading.Event()
        self._t = None
        self._t0 = 0.0
        self.tokens = 0   # live counter updated by the streaming consumer

    def __enter__(self):
        self._t0 = time.time()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

        return self

    @staticmethod
    def _fmt_tokens(n):
        # Claude-style: 999 -> "999", 1200 -> "1.2k", 12345 -> "12k"

        if n < 1000:
            return str(n)

        if n < 10000:
            return f"{n / 1000:.1f}".rstrip("0").rstrip(".") + "k"

        return f"{round(n / 1000)}k"

    @staticmethod
    def _fmt_elapsed(seconds):
        """Seconds while that stays readable, then minutes and seconds."""
        seconds = int(seconds)

        return (f"{seconds}s" if seconds < 60
                else f"{seconds // 60}m{seconds % 60:02d}s")

    def _run(self):
        i = 0

        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            el = self._fmt_elapsed(STATUS.elapsed(self._t0))
            tok = (f" · {self._fmt_tokens(self.tokens)} tokens"
                   if self.tokens else "")

            STATUS.show(
                f"{C_ACCENT}{frame}{C_RST} {self.label} "
                f"{C_DIM}({el}{tok} · ctrl+c to interrupt){C_RST}")

            time.sleep(0.12)
            i += 1

    def __exit__(self, *exc):
        self._stop.set()

        if self._t:
            self._t.join(timeout=1)

        elapsed = self._fmt_elapsed(STATUS.elapsed(self._t0))
        tokens = (f" · {self._fmt_tokens(self.tokens)} tokens"
                  if self.tokens else "")

        # Left standing, static: no glyph, no interrupt hint, no newline.
        # It is what the harness last did, and it holds the line until the
        # next activity overwrites it or real output takes it away.
        STATUS.show(f"{C_DIM}  {self.label} {elapsed}{tokens}{C_RST}")


class Liveness:
    """When the turn last did something observable.

    The wall-clock budget is COOPERATIVE: check_wall_time runs when
    something is charged -- a tool call, a model call. A turn suspended in a
    read that never returns charges nothing, so it never consults the clock.
    One ran for eleven hours inside an eight-minute budget, overnight, and
    the only reason it stopped was that I killed it.
    So the clock needs a heartbeat that does not depend on the turn's
    cooperation: tokens arriving, a result printed, a notice. The spinner is
    deliberately NOT a heartbeat -- it animates whether or not the provider
    is answering, which is precisely the case being watched for.
    """

    def __init__(self):
        self.at = time.monotonic()

    def touch(self):
        self.at = time.monotonic()

    def silent_for(self):
        return time.monotonic() - self.at


LIVENESS = Liveness()

# How long a turn may show no sign of life before it is cancelled. Generous:
# a long build or a slow first token is not a hang, and the cancellation is
# cooperative anyway -- it asks the runtime to stop at its next poll point.
TURN_LIVENESS_SECONDS = int(os.environ.get("SPEAR_TURN_LIVENESS", "300"))


class CliRuntimeObserver:
    """Keep terminal rendering outside the provider-neutral runtime."""

    @contextmanager
    def model_activity(self, label):
        LIVENESS.touch()

        with Spinner(label) as spinner:
            def tick():
                spinner.tokens += 1
                LIVENESS.touch()

            yield tick

    def intermediate_text(self, text):
        LIVENESS.touch()
        print_assistant(text)
        print()

    def notice(self, kind, metadata):
        LIVENESS.touch()

        if kind == "verification_nudge":
            print(f"  {C_DIM}↪ nothing ran after the change — "
                  f"asking for verification{C_RST}")
        elif kind == "investigation_nudge":
            print(f"  {C_DIM}↪ enough investigation — asking the model "
                  f"to conclude{C_RST}")
        elif kind == "intent_judged":
            if metadata.get("write"):
                print(f"  {C_DIM}↪ read as a request to change the code "
                      f"(the verb list did not recognise it){C_RST}")
        elif kind == "identical_result":
            print(f"  {C_DIM}↪ {metadata.get('tool')} returned a result "
                  f"already seen this turn — saying so{C_RST}")
        elif kind == "stalled_without_redirect":
            spent = metadata.get("redirects_spent", {})
            print(f"  {C_DIM}↪ looping and nothing sent it back "
                  f"(wrote={metadata.get('wrote')}, "
                  f"write_request={metadata.get('write_request')}, "
                  f"redirects={spent}){C_RST}")
        elif kind == "wall_time_wrap_up":
            print(f"  {C_DIM}↪ four fifths of the time budget spent — asking "
                  f"for a landing{C_RST}")
        elif kind == "compaction_failed":
            print(f"  {C_DIM}↪ the context could not be compacted "
                  f"({metadata.get('reason', '')[:90]}) — later rounds will "
                  f"carry the full history{C_RST}")
        elif kind == "empty_final_synthesis":
            print(f"  {C_DIM}↪ no conclusion from the model — reporting "
                  f"what was found{C_RST}")
        elif kind == "clauses_unaddressed":
            missing = metadata.get("missing") or []
            print(f"  {C_DIM}↪ {len(missing)} of "
                  f"{metadata.get('carried', 0)} established clauses not "
                  f"addressed — asking for them{C_RST}")
        elif kind == "next_work_item":
            print(f"  {C_DIM}↪ that change is finished — pointing at the next "
                  f"open requirement{C_RST}")
        elif kind == "validation_owed":
            print(f"  {C_DIM}↪ source changed and its test not written yet — "
                  f"asking for the validation{C_RST}")
        elif kind == "build_broken_work_item":
            print(f"  {C_DIM}↪ build down, nothing proved — sending it back "
                  f"to the work item that broke it{C_RST}")
        elif kind == "contract_closed":
            print(f"  {C_DIM}↪ every carried requirement closed and covered — "
                  f"asking it to land{C_RST}")
        elif kind == "plan_owed":
            print(f"  {C_DIM}↪ both sides read and nothing planned — asking "
                  f"for the plan{C_RST}")
        elif kind == "exploration_without_evidence":
            print(f"  {C_DIM}↪ the last few calls established nothing new — "
                  f"asking for a synthesis ({metadata.get('phase', '')})"
                  f"{C_RST}")
        elif kind == "write_request_unanswered":
            print(f"  {C_DIM}↪ nothing was changed — asking for the edit"
                  f"{C_RST}")
        elif kind == "project_build_failed":
            print(f"  {C_DIM}↪ the change does not survive "
                  f"`{metadata.get('command', '')[:52]}` — asking for a fix"
                  f"{C_RST}")
        elif kind == "work_order_sections_skipped":
            print(f"  {C_DIM}↪ work order sections not done "
                  f"({', '.join(metadata.get('sections') or [])}) — "
                  f"asking for them{C_RST}")
        elif kind == "standard_evidence_injected":
            # The opening retrieval and the boundary bootstrap are the same
            # call at two moments, and they must not be announced the same
            # way: the opening runs BEFORE the model has said anything, so
            # "the answer cited the standard without reading it" describes a
            # failure that has not happened -- and on a good turn it was the
            # only thing the line ever said.

            if metadata.get("origin") == "OPENING_RETRIEVAL":
                print(f"  {C_DIM}↪ reading the bound standard before "
                      f"answering — {metadata.get('tool_name')}{C_RST}")
            else:
                print(f"  {C_DIM}↪ the answer cited the standard without "
                      f"reading it — {metadata.get('tool_name')} run for it"
                      f"{C_RST}")
        elif kind == "tool_exception":
            tool_result(str(metadata.get("error_summary") or "tool failed"))
            print()


def render_md(text):
    """Light terminal markdown rendering: bold, colored inline code,
    bold headers, code blocks with a side bar."""
    out = []
    in_code = False

    for line in text.split("\n"):
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code = not in_code
            continue

        if in_code:
            out.append(f"  {C_DIM}│{C_RST} {C_CODE}{line}{C_RST}")
            continue

        l = re.sub(r"^#{1,4}\s+(.*)", f"{C_BOLD}\\1{C_RST}", line)
        l = re.sub(r"\*\*(.+?)\*\*", f"{C_BOLD}\\1{C_RST}", l)
        l = re.sub(r"`([^`]+)`", f"{C_CODE}\\1{C_RST}", l)
        out.append(l)

    return "\n".join(out)



def runtime_tool_log(task_result):
    """How much the turn managed to do before it failed — one number, safely."""
    result = getattr(task_result, "agent_result", None)

    return getattr(result, "tool_log", ()) or ()


def print_assistant(text):
    """Assistant response: ⏺ bullet on the first line, indented continuation."""
    text = render_md(text.strip())

    if not text:
        print(f"{C_DIM}⏺ (no response){C_RST}")
        return

    lines = text.split("\n")
    print(f"{C_BOLD}⏺{C_RST} {lines[0]}")

    for l in lines[1:]:
        print(f"  {l}")


def tool_use(name, arg, color=C_OK):
    """Tool-use header: ⏺ Bash(command)"""
    arg = arg if vlen(arg) <= 90 else arg[:87] + "…"

    with terminal_output():
        print(f"{color}⏺{C_RST} {C_BOLD}{name}{C_RST}({arg})", flush=True)


def tool_result(text, max_lines=8):
    """Tool result indented under ⎿ , truncated Claude Code style."""

    # An empty result is a result: "" rstripped and split is [""], which is
    # truthy, so the (empty) fallback never fired and a bare ⎿ was printed.

    lines = [line for line in (text or "").rstrip().split("\n")] or ["(empty)"]

    if lines == [""]:
        lines = ["(empty)"]

    with terminal_output():
        for i, l in enumerate(lines[:max_lines]):
            pfx = "⎿  " if i == 0 else "   "
            l = l if len(l) <= 160 else l[:157] + "…"
            print(f"  {C_DIM}{pfx}{l}{C_RST}")

        if len(lines) > max_lines:
            print(f"  {C_DIM}   … +{len(lines) - max_lines} lines{C_RST}")

        sys.stdout.flush()


def show_web_sources(result, max_sources=5):
    """Claude Code-style web source listing: title + URL per result."""
    shown = 0
    first = True

    for line in result.split("\n"):
        s = line.strip()

        if s.startswith("[") and shown < max_sources:
            pfx = "⎿  " if first else "   "
            first = False
            print(f"  {C_DIM}{pfx}{s[s.index(']')+1:].strip()}{C_RST}")
        elif s.startswith("http") and shown < max_sources:
            print(f"     {C_TOOL}{s}{C_RST}")
            shown += 1

    if shown == 0:
        tool_result(result, max_lines=4)
    else:
        total = result.count("\nhttp") + result.count("    http")

        if total > max_sources:
            print(f"  {C_DIM}   … +{total - max_sources} sources{C_RST}")


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
    "read_only_task", "execution_mode_denied",
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


# Corpus names that are also ordinary directory words: mentioning them in a
# sentence says nothing about where the user wants to work.

_HINT_STOPWORDS = {"src", "lib", "bin", "doc", "docs", "test", "tests",
                   "build", "include", "data", "tmp"}
_HINTED_CORPORA = set()


HELP_TEXT = """\
spear-chat — HEIG-VD/REDS AI coding assistant

USAGE
  spear-chat [options]

PERMISSIONS  (what the assistant may do to your files)
  --safe                 read-only, and the default when no mode is given.
                         Mutations are refused, not proposed, and the network
                         is unreachable.
  --ask                  confirm each edit and each command before it runs.
                         Aliases: --confirm, --no-bypass. The ONLY mode with
                         network access.
  --auto                 run edits and commands without asking. Aliases: -y,
                         --yolo, --bypass-permissions. Deliberately NO network.
  --no-network           drop network even in --ask.
  --single-root          restrict writes to the launch directory. By default
                         the registered corpora are writable too, each mounted
                         at /workspaces/<name> — a path that works in bash and
                         in edit_file/write_file alike. Relative paths always
                         resolve in the launch directory and never reach them.
  --allow-absolute-paths accept host absolute paths into the launch directory.
                         Off by default; /workspace/... always works.

MODEL
  --provider NAME        openai-compatible (default) or anthropic.
  --model NAME           model id. Anthropic default: claude-sonnet-5.
  --remote, --pod        use the GPU pod's vLLM over an SSH tunnel instead of
                         the local llama-server. --local forces local.
  --reds                 use the REDS server (RTX PRO 6000) over an SSH tunnel;
                         host/port/model in reds.conf. The launcher only opens
                         the tunnel — start the server there yourself.
  --pod-host H, --pod-port P
                         point at a pod without editing pod.conf.

CORPUS  (what the assistant reads about — distinct from where tools run)
  --corpus NAME          force a registered corpus. Aliases: --project,
                         --checkout. Tools still run in the CURRENT directory.
  --here                 treat the current directory as an ad-hoc corpus.
  (none)                 the corpus containing the cwd, else ad-hoc on the cwd.

FEDERATION
  --with NAME            also retrieve from this registered corpus, for one
                         session. Repeatable.
  --without NAME         drop one that would be attached (a `corpora:` entry
                         of the project, or a corpus marked shared).

SESSION SETTINGS  (each is also an environment variable; the flag wins)
{env_options}

OPERATOR COMMANDS  (typed in the session, never reachable by the model)
  /standard              ingest a specification, bind one, inspect the binding.
                         The binding decides how normative answers are grounded
                         and is shared by every session on this machine, so
                         check it before trusting one: `/standard status`.
                         `/standard list`, `/standard use <id> <revision>`.
  /finetune              the fine-tuning control plane.

OTHER
  --trace                record a runtime JSONL trace (SPEAR_TRACE).
  --record FILE          write down every model turn of this session.
  --replay FILE          answer from a recorded file instead of the model.
                         The tools, the files and the gates all run for real;
                         only the model is a recording, so a harness change
                         can be judged in minutes and at no cost. A turn that
                         needs more rounds than were recorded simply ends.
  --help, -h             this text.

Without a backend flag, an interactive launch lists local / remote / reds /
anthropic
and preselects the one used last (remembered in active-backend.conf); a flag
skips the picker, and off a terminal the last choice is reused silently.

Tools always run in the current directory, whatever the corpus. To work on
another tree, cd into it first — no flag relocates the workspace.

ENVIRONMENT  (machine-wide defaults; machine.env is the place for them)
  ANTHROPIC_API_KEY   Anthropic credential; an `ant auth login` session is
                      used when it is unset.
  SPEAR_API_BASE      OpenAI-compatible endpoint (default 127.0.0.1:8080/v1).
  SPEAR_MODEL         GGUF path served by spear-server, overriding
                      active-model.conf. SPEAR_LORA likewise for the adapter.
"""


def _env_option_lines():
    """The settings table, rendered from ENV_OPTIONS so it cannot drift.

    A help text maintained beside its parser goes stale; this one IS the
    parser's table.
    """
    rows = []

    for flag, (variable, description) in sorted(ENV_OPTIONS.items()):
        label = f"{flag} VALUE"

        if len(label) > 24:
            rows.append(f"  {label}")
            rows.append(f"  {'':<25}{description}  ({variable})")
        else:
            rows.append(f"  {label:<25}{description}")
            rows.append(f"  {'':<25}{variable}")

    return "\n".join(rows)


def print_help():
    """Usage text. The startup banner names the permission flags too, but only
    after a full startup — this is the answer to `spear-chat --help`."""
    print(HELP_TEXT.replace("{env_options}", _env_option_lines()), end="")


def backend_label(provider, api_base):
    """Which backend the banner is reporting: never blank, never ambiguous.

    Previously only the remote pod was tagged, so local and Anthropic looked
    identical on the banner — the one line that is supposed to tell you what
    you are about to talk to.
    """

    if provider == "anthropic":
        return "anthropic API"

    # The launcher knows which host it tunnelled to; a port number does not.
    # This line used to read "reds-server (RTX PRO 6000)" for anything on
    # :8082, and kept saying so after reds.conf was repointed at another
    # machine — the banner then named a host we were not talking to.

    label = os.environ.get("SPEAR_BACKEND_LABEL", "").strip()

    if label:
        return label

    if "8081" in api_base:
        return "remote/pod vLLM"

    if "8082" in api_base:
        return "reds tunnel"

    return "local llama-server"


def permissions_row(mode):
    """The startup banner's permissions line: the real mode, and how to change it.

    This used to branch on the legacy BYPASS_PERMISSIONS boolean, which predates
    the three modes: it announced SAFE — the default — as "ask before each
    edit/run", a mode SAFE does not have (it refuses mutations outright), and
    labelled AUTO "(default)", which stopped being true when SAFE became it.
    The banner is where a user learns which mode they are in, so naming the
    flags here is what keeps them discoverable.
    """

    if mode == ExecutionMode.AUTO:
        return (f"{C_WARN}⚠ permissions: auto — edits, commands and network "
                f"access run without asking{C_RST}{C_DIM}; --ask to confirm "
                f"each, --no-network to stay offline, --safe for read-only"
                f"{C_RST}")

    if mode == ExecutionMode.ASK:
        return (f"{C_DIM}permissions: ask before each edit/run (network "
                f"included); --safe for read-only, --auto to stop asking"
                f"{C_RST}")

    return (f"{C_DIM}permissions: safe (default) — mutations refused; --ask to "
            f"confirm each edit/run, --auto to stop asking{C_RST}")


def standard_row():
    """The bound standard, on the banner, beside the model and the corpus.

    It decides how every normative answer is grounded, and it lives in ONE file
    under the state directory, shared by every session on this machine. A
    binding left behind by other work is therefore inherited in silence: a VITA
    49.2 question was once answered against an RS274/NGC binding, and the only
    thing that ever said so was the per-turn line, printed after the question
    had already been asked. A session-wide property belongs with the other
    session-wide properties.

    Returns None when no standard has ever been ingested, so a project that
    does not use one carries no row it cannot act on.
    """
    try:
        if not STANDARD_STORE.list_standards():
            return None
    except OSError:
        return None

    # A binding the store can no longer honour is a startup fact too. Raising
    # here would take the banner down; saying nothing would leave it to be
    # discovered from the first normative answer that comes out empty.

    try:
        binding = STANDARD_OPERATOR.active_binding()
    except StandardCommandError as exc:
        return f"{C_WARN}standard: unavailable — {exc}{C_RST}"

    if binding is None:
        return (f"{C_DIM}standard:{C_RST} none bound  {C_DIM}·{C_RST}  "
                f"{C_DIM}/standard list, /standard use <id> <revision>{C_RST}")

    # How it is searched follows the document, so switching documents
    # switches it; the banner is where that has to be visible.

    return (f"{C_DIM}standard:{C_RST} {binding.standard_id} {binding.revision}"
            f"  {C_DIM}·{C_RST}  {binding.data_origin.lower()}"
            f"  {C_DIM}·{C_RST}  "
            + retrieval_summary(STANDARD_STORE, binding.standard_id,
                                binding.revision))


def content_row(dirs=None):
    """Where rules, skills and benches come from, on the banner.

    They are relocatable (resource_dir) and a deployment points them at a
    tree outside the checkout. When the file that does so is missing, the
    session starts anyway on the near-empty in-tree defaults, and nothing
    said so: losing a deployment's rules looked exactly like having none.
    """
    if dirs is None:
        dirs = {"rules": RULES_DIR, "skills": SKILLS_DIR, "benches": BENCH_DIR}

    app_dir = os.path.realpath(APP_DIR)

    def is_in_tree(path):
        return os.path.realpath(path).startswith(app_dir + os.sep)

    # An absent in-tree directory is a plain checkout; an absent external one
    # is a deployment pointing at something that is not there.

    missing = [name for name, path in dirs.items()
               if not is_in_tree(path) and not os.path.isdir(path)]

    if all(is_in_tree(path) for path in dirs.values()):
        row = f"{C_DIM}content:{C_RST}  in-tree  {C_DIM}(no external rules/skills/benches){C_RST}"
    else:
        # Grouped by parent, so the usual case -- one deployment tree holding
        # all three -- reads as that tree rather than as three paths.

        parents = {}

        for name, path in dirs.items():
            parent = ("in-tree" if is_in_tree(path)
                      else os.path.dirname(os.path.normpath(path)))
            parents.setdefault(parent, []).append(name)

        home = os.path.expanduser("~")
        parts = []

        for parent, names in parents.items():
            if parent.startswith(home + os.sep):
                parent = "~" + parent[len(home):]

            parts.append(f"{parent} {C_DIM}({', '.join(names)}){C_RST}")

        row = f"{C_DIM}content:{C_RST}  " + f"  {C_DIM}·{C_RST}  ".join(parts)

    if missing:
        row += f"  {C_WARN}⚠ missing: {', '.join(missing)}{C_RST}"

    return row


def corpus_mention_hint(user_input, projects, current):
    """Name a registered corpus the question mentions but the tools are not in.

    Strictly informational.  The cwd IS the sandbox root, so no phrase in a
    question may relocate it — letting natural language move where writes land
    would hand the workspace boundary to the model's reading of a sentence.
    What the harness owes the user is what it already knows: that the name is
    registered, and where.  Longest name first so `micropython-so3` wins over
    `so3`; once per name per session, so it informs without nagging.
    """

    # A corpus this session already retrieves from is not somewhere to go: an
    # umbrella federates everything under the cwd, so telling the user to `cd
    # so3` while so3's chunks are arriving as so3/usr/src/... is advice to
    # leave a session that can already reach the file.

    attached = set(attached_corpus_names(
        projects.get(current) or PROJECT_SPEC or {}, projects))

    for name, spec in sorted(projects.items(), key=lambda kv: -len(kv[0])):
        if name == current or name in _HINTED_CORPORA or name in attached:
            continue

        if name.lower() in _HINT_STOPWORDS:
            continue

        # A trailing slash means the word is a path fragment ("src/main.c"),
        # not a corpus the user is naming.

        if not re.search(rf"\b{re.escape(name)}\b(?!/)", user_input, re.I):
            continue

        path = spec.get("path", "")

        if not os.path.isdir(path):
            continue

        _HINTED_CORPORA.add(name)

        return (f"'{name}' is a registered corpus ({path}), but your tools run "
                f"in {PROJECT_ROOT}. To work there: cd {path} && spear-chat")

    return None


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


def run_cmd_result(cmd, need_confirm=True, cancellation=None, execution_mode=None):
    """Return the security substrate's structured command result."""
    mode = execution_mode or EXECUTION_MODE
    assessment = COMMAND_POLICY.classify(cmd)
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
        argv = (list(assessment.argv) if assessment.classification
                != CommandClassification.SHELL_COMPLEX
                else shell_argv(cmd))
        result = COMMAND_RUNNER.run_sandboxed(
            WORKSPACE, argv, profile, availability=availability,
            cancellation=cancellation,
        )
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

        return ToolResult("timeout", f"timed out (45s){hint}")

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


def web_search(query, max_results=6):
    """Web search via DuckDuckGo (ddgs package, no API key).
    Returns formatted results (title, URL, snippet)."""

    try:
        from ddgs import DDGS
        results = list(DDGS().text(query, max_results=max_results))
    except Exception as e:
        return f"ERROR web search: {e}"

    if not results:
        return "(no results)"

    out = []

    for i, r in enumerate(results, 1):
        out.append(f"[{i}] {r.get('title', '')}\n"
                   f"    {r.get('href', '')}\n"
                   f"    {r.get('body', '')}")

    return "\n\n".join(out)


# ── ChromaDB / RAG ───────────────────────────────────────────────────

def collection_name_for(spec):
    """The chroma collection a registered corpus is indexed into.

    An explicit ``"collection"`` wins over the derived name. The derivation is
    an md5 of the ABSOLUTE path, which is fine while the index and the trees
    live on the same machine and wrong the moment they do not: mounted under
    /corpora, the same tree hashes differently and the shipped index goes
    unfound -- the container started cleanly with no retrieval at all, which is
    the whole point of the tool. So a registry that travels with an index names
    the collections instead of recomputing them.
    """
    explicit = spec.get("collection")

    if explicit:
        return explicit

    tag = hashlib.md5(os.path.realpath(spec["path"]).encode()).hexdigest()[:8]

    return f"adhoc_{tag}"


def corpus_prefix(path):
    """How a chunk of `path` must be addressed from the current tree.

    Relative while the corpus sits under the launch directory -- that is the
    federation case, and bash resolves it. Otherwise ABSOLUTE: a shared corpus
    lives in another tree entirely, and `../../../opt/llm/...` is both unusable
    and refused by the command policy. Absolute paths outside the workspace are
    readable now, so the model can actually open what it is shown.
    """
    corpus, root = os.path.realpath(path), os.path.realpath(PROJECT_ROOT)

    if corpus == root:
        return ""

    if corpus.startswith(root + os.sep):
        return os.path.relpath(corpus, root)

    return corpus


def _cli_values(flag):
    """Repeated `--flag value` occurrences, in order."""
    argv, out = sys.argv[1:], []

    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            out.append(argv[i + 1])
        elif a.startswith(flag + "="):
            out.append(a.split("=", 1)[1])

    return out


def attached_corpus_names(spec, projects):
    """Every corpus this session should retrieve from, besides its own.

    Three sources, in this order:
      - `"corpora": [...]` on the project -- a federation of ONE tree's parts;
      - every corpus marked `"shared": true` -- cross-cutting knowledge that
        belongs to no single tree (the build system, an API reference). This is
        what lets `ib` answer a build question in whatever tree you happen to
        be standing in, without redeclaring it in all 22 projects;
      - `--with <name>` on the command line, for a one-off.
    `--without <name>` removes any of them, so a shared corpus is never a
    sentence: a session that does not want it says so and pays nothing.
    """
    names = list(spec.get("corpora") or [])
    names += [n for n, sub in projects.items()
              if sub.get("shared") and n not in names]
    names += [n for n in _cli_values("--with") if n not in names]
    dropped = set(_cli_values("--without"))

    return [n for n in names if n not in dropped]


def federated_corpora():
    """Corpora this session retrieves from, as (collection, path prefix) pairs.

    A project may declare `"corpora": ["a", "b"]` in projects.json instead of
    owning one index. Questions then reach every one of them without switching
    session — which is the point: a kernel question, a userspace question and a
    bootloader question all belong to the same tree. A corpus marked
    `"shared": true` is attached to EVERY session for the same reason, one
    level up: the build system is not the property of one checkout.

    Merging them into a single index would not do: u-boot holds 11298 indexable
    files against so3's 1482, so kernel code would compete 7-to-1 for the same
    twelve slots. Kept apart and fused by rank, each contributes its own best
    hits.

    The prefix is what makes the result usable: chunks are indexed relative to
    THEIR corpus root, while tools run at the federation root. Without it the
    model reads `usr/src/x.c` and bash needs `so3/usr/src/x.c`.
    """
    projects = load_projects()
    spec = projects.get(PROJECT) or PROJECT_SPEC or {}
    names = attached_corpus_names(spec, projects)

    if not names:
        return []

    client = _db()
    auto = set(spec.get("auto_corpora") or ())
    AUTO_CORPUS_BY_COLLECTION.clear()
    # `own` is the corpus this session already adds itself (init_chromadb),
    # and that is CORPUS_ROOT — not the cwd. Compared against the cwd, a
    # session standing outside its own tree failed to recognise it and
    # attached its own index a second time.

    out, own = [], os.path.realpath(CORPUS_ROOT)

    for n in names:
        sub = projects.get(n)

        if not sub:
            print(f"{C_WARN}corpus '{n}' is not registered — skipped{C_RST}")
            continue

        if os.path.realpath(sub["path"]) == own:
            continue            # the session's own corpus is added by the caller

        try:
            col = client.get_collection(name=collection_name_for(sub))
        except Exception:
            print(f"{C_WARN}corpus '{n}' has no index yet "
                  f"({collection_name_for(sub)}) — index it with: "
                  f"spear-index {sub['path']}{C_RST}")
            continue

        out.append((col, corpus_prefix(sub["path"])))

        if n in auto:
            AUTO_CORPUS_BY_COLLECTION[col.name] = n

    return out


def init_chromadb():
    """The session's retrieval set: its own corpus PLUS whatever is attached.

    Attached corpora used to REPLACE the project's own index rather than join
    it, which was harmless while the only federation (sye_sol) owned no index
    of its own -- and would have silently emptied every other session the day a
    shared corpus was declared.
    """
    attached = federated_corpora()
    client = _db()

    try:
        own = [(client.get_collection(name=COLLECTION_NAME), "")]
    except Exception:
        own = []

        if corpus_autoindexes():
            # This corpus declares that a missing index is built on sight.

            print(f"{C_WARN}Corpus {COLLECTION_NAME} missing — indexing "
                  f"{CORPUS_ROOT}...{C_RST}")
            subprocess.run(reindex_command())
            own = [(client.get_collection(name=COLLECTION_NAME), "")]
        else:
            # generic project: indexing is opt-in (trees can be huge). Say so
            # even when other corpora are attached — the banner sums whatever
            # the session retrieves from and prints the total under THIS
            # project's name, so an agency session owning no index announced
            # "agency · 506 chunks": 506 chunks of an attached corpus, and not
            # one line of agency.

            print(f"{C_DIM}No index for this corpus yet — use /reindex "
                  f"to build one (optional).{C_RST}")

            if not attached:
                return None

    corpora = own + attached

    if attached:
        total = sum(c.count() for c, _ in corpora)
        print(f"{C_DIM}retrieving from {len(corpora)} corpora "
              f"({total} chunks){C_RST}")

    return corpora or None


# ── hybrid lexical + dense ───────────────────────────────────────────
# MiniLM's cosine distance does NOT discriminate identifiers: measured on
# edgem1_verdin, the query "ou est defini __sys_empty" returns 12 chunks all
# within 0.635-0.669 — the same band as an off-topic question (0.72-0.80). With
# 0.03 between the 1st and the 12th, the dense ranking means nothing there.
# Chroma's full-text filter, on the other hand, cuts clean (syscalls.c in
# 0.04 s). We fuse both rankings with Reciprocal Rank Fusion, which does not
# require their scores to be comparable.
# Measured by eval/run_eval.py: recall@12 identifiers 58% -> 92%, global
# 65% -> 92%.

_IDENT_RX = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*[A-Za-z0-9_]")
LEX_MAX_TERMS = 4       # beyond this we only dilute the fusion
LEX_PER_TERM = 6        # lexical hits kept per term
LEX_SATURATION = 40     # a term in >= 40 chunks discriminates nothing


def _ident_terms(query, max_terms=LEX_MAX_TERMS):
    """Query tokens deserving an EXACT search: snake_case, CamelCase,
    UPPERCASE, file names. Ordinary French words carry neither underscore, dot
    nor capital — they are dropped here, and the dense channel handles them
    well anyway."""
    terms = []

    for t in _IDENT_RX.findall(query):
        if len(t) < 4:
            continue

        # "_" / "." / UPPERCASE / internal capital = identifier or file name.
        # Leading capital alone is NOT enough: "Comment", "Dans", "Montre"
        # carry one too, and firing a $contains at them costs a query for
        # nothing (saturation eventually drops them, but later).

        if ("_" in t or "." in t or t.isupper()
                or any(c.isupper() for c in t[1:])):
            terms.append(t)

    # longest first: those are the most specific

    return sorted(dict.fromkeys(terms), key=len, reverse=True)[:max_terms]


def _definition_score(doc, meta, term):
    """Ranking for lexical hits. `get()` returns no order, so we must build one.

    Two signals. (1) The PATH: Chroma's `$contains` is a raw SUBSTRING match,
    so a query for a file name also returns every longer name ending the same
    way. When the term IS the file name that is decisive, hence the
    overwhelming weight. (2) The BODY: a chunk that DEFINES the term (see
    _definition_lines) weighs more than one merely using it."""
    path = meta.get("filepath", "")
    score = 0

    if term == os.path.basename(path):
        score += 100
    elif term in path:
        score += 50

    body = doc.split("\n\n", 1)[-1]

    return score + 3 * _definition_lines(body, term) + body.count(term)


def _definition_lines(body, term):
    """How many lines DEFINE `term` rather than use it.

    Three shapes cover most of the corpus:
      - `term` at line start   -> `IB_X = "..."`, `CONFIG_X=y`, a label
      - `#define term`         -> a macro
      - `term(` with no trailing `;` -> a C function header. The `;` is what
        separates the DEFINITION (`static long __sys_empty(args)`) from the
        declaration (`... args);`) and from the call (`x = __sys_empty(a);`) —
        in none of the three is the identifier at line start.
    """
    n = 0

    for line in body.split("\n"):
        s = line.lstrip()

        if (s.startswith(term)
                or s.startswith(f"#define {term}")
                or (f"{term}(" in s and not s.rstrip().endswith(";"))):
            n += 1

    return n


def _rrf(ranked_lists, k=60):
    """Reciprocal Rank Fusion: merges heterogeneous rankings without having to
    normalise their scores. k=60 is the original constant (Cormack et al.) —
    it flattens the weight of the list heads."""
    score = {}

    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst):
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(score, key=score.get, reverse=True)


_FILENAME_RX = re.compile(r"[A-Za-z0-9_.-]+\.[A-Za-z0-9_]{1,8}")


def _lex_get(collection, needle):
    """Lexical candidates for one needle, or None when it discriminates nothing."""

    try:
        hit = collection.get(where_document={"$contains": needle},
                             include=["documents", "metadatas"],
                             limit=LEX_SATURATION)
    except Exception:
        return None     # the lexical channel is a bonus, never a point of
                        # failure: the chat must answer without it.
    ids = hit["ids"]

    if not ids or len(ids) >= LEX_SATURATION:
        return None     # term absent, or too common to discriminate

    return list(zip(ids, hit["documents"], hit["metadatas"]))


def _lex_candidates(collection, term):
    """Hits for a query term, anchored on a path separator when it names a file.

    `$contains` matches raw substrings, so "ls.c" is equally found inside
    "globals.c", "parserInternals.c" and "syscalls.c". On a real corpus that
    turned the most specific term of the query into its least useful one: 104
    chunks matched, 98 of them for files merely ENDING in "ls.c", the count
    tripped the saturation guard, and the term was dropped entirely -- so
    asking to edit ls.c retrieved everything except ls.c, and handed the model
    libxml2 instead. Anchoring on "/" restores the file-name meaning: the same
    query then returns six chunks, all of them the actual file.
    """

    if _FILENAME_RX.fullmatch(term):
        anchored = _lex_get(collection, "/" + term)

        if anchored:
            return anchored

    return _lex_get(collection, term)


def _retrieve_one(collection, query, top_k):
    """Dense + lexical candidates from ONE collection.

    Returns (pool, lists): the documents keyed by id, and the rankings to be
    fused. Split out of retrieve_context so several corpora can be searched
    per turn without each one's ranking swamping the others.
    """

    # The embedding model is read from the collection METADATA, not from the
    # config: a collection indexed with another model stays correctly
    # queryable even if active-embedder.conf changed since. Without this,
    # Chroma would report nothing and return random neighbours.

    qvec = embedding.embed_query(query, embedding.collection_model(collection))

    if qvec is None:
        dense = collection.query(
            query_texts=[query], n_results=top_k * 2,
            include=["documents", "metadatas", "distances"],
        )
    else:
        dense = collection.query(
            query_embeddings=[qvec], n_results=top_k * 2,
            include=["documents", "metadatas", "distances"],
        )

    pool = {i: (d, m) for i, d, m in zip(
        dense["ids"][0], dense["documents"][0], dense["metadatas"][0])}
    lists = [list(dense["ids"][0])]

    for term in _ident_terms(query):
        candidates = _lex_candidates(collection, term)

        if not candidates:
            continue

        ranked = sorted(candidates,
                        key=lambda x: -_definition_score(x[1], x[2], term))
        ranked = ranked[:LEX_PER_TERM]

        for i, d, m in ranked:
            pool.setdefault(i, (d, m))

        lists.append([i for i, _, _ in ranked])

    return pool, lists


SEARCH_CORPUS_TOP_K = 5      # a mid-turn result, not a whole turn's context
SEARCH_CORPUS_MAX_CHARS = 4000


def search_corpus(query, top_k=SEARCH_CORPUS_TOP_K):
    """Let the model query the index while it works, not only at turn start.

    The Retrieved Context is chosen once, from the USER's sentence. That
    sentence carries the task ("modify ls.c so it handles wildcards"), not the
    sub-question the model hits three steps later ("what matches a pattern?").
    Without a way to ask, the model answers that sub-question from whatever the
    first retrieval happened to include -- which is how one session ended up
    linking libxml2 into ls because triostr.c was in the context and fnmatch
    was not.
    """

    if COLLECTION is None:
        return "ERROR: no corpus is indexed for this project"

    try:
        context, _ = retrieve_context(COLLECTION, query, top_k)
    except Exception as e:
        return f"ERROR: corpus search failed: {e}"

    if not context or not context.strip():
        return "(no match in the indexed corpora)"

    if len(context) > SEARCH_CORPUS_MAX_CHARS:
        context = (context[:SEARCH_CORPUS_MAX_CHARS]
                   + "\n… (truncated; ask a narrower query)")

    return context


def _reroot(doc, meta, prefix):
    """Rewrite a chunk's `# File:` header to be relative to the FEDERATION
    root, not to its own corpus. Chunks are indexed per corpus, but tools run
    at the federation root — leaving `usr/src/x.c` when bash needs
    `so3/usr/src/x.c` is what sent the model chasing absolute paths."""

    if not prefix:
        return doc, meta.get("filepath", "")

    fp = meta.get("filepath", "")
    rooted = os.path.join(prefix, fp) if fp else fp
    head, sep, rest = doc.partition("\n\n")

    if sep and head.startswith("# File: ") and fp:
        head = head.replace(fp, rooted, 1)
        doc = head + sep + rest

    return doc, rooted


# collection name -> corpus name, for the corpora an umbrella DISCOVERED. Only
# these are selected per question; a declared federation and a shared corpus
# were chosen deliberately and are attached every turn.

AUTO_CORPUS_BY_COLLECTION = {}


def _mentioned(name, path, query):
    """Does the question name this corpus, or the directory it lives in?

    Whole words, so `so3` does not match `so3-doc`'s path fragment, and the
    basename counts too: in a workspace session "a chapter in doc" is naming
    `doc/`, which is corpus so3-doc. The hint's stopword list deliberately does
    NOT apply here -- it exists to avoid telling someone to `cd` somewhere on
    the strength of the word "build", while inside a workspace that same word
    really does name the subdirectory being asked about.
    """
    words = {name, os.path.basename(path.rstrip("/"))}

    return any(word and re.search(rf"\b{re.escape(word)}\b", query, re.I)
               for word in words)


def select_corpora(corpora, query, projects=None):
    """Narrow the DISCOVERED corpora to the ones the question names.

    A workspace launch federates everything registered under it -- eight trees
    at ~/soo/so3 -- and querying all of them every turn costs an embedding call
    each for corpora the question never touches. Named ones win; if the
    question names none, the part sharing the workspace's own name is kept, on
    the same umbrella-shape reasoning that used to pick it outright.

    Anything not discovered this way is untouched: the session's own index, a
    declared federation, a shared corpus, `--with`.
    """

    if not AUTO_CORPUS_BY_COLLECTION or not query:
        return corpora

    projects = projects if projects is not None else load_projects()
    primary = PROJECT.split(":", 1)[1] if PROJECT.startswith("workspace:") else None

    # Longest name first, blanking what it matched: `-` is a word boundary, so
    # `\bso3\b` fires inside `micropython-so3` and would attach the kernel to a
    # question that named only the library. Same rule the mention hint uses.

    remaining, named = query, set()

    for name in sorted(set(AUTO_CORPUS_BY_COLLECTION.values()), key=len, reverse=True):
        path = (projects.get(name) or {}).get("path", "")

        for word in sorted({name, os.path.basename(path.rstrip("/"))},
                           key=len, reverse=True):
            if not word:
                continue

            # Not \b: `-` is a word boundary, so `\bso3\b` fires inside
            # `micropython-so3` and would attach the kernel to a question that
            # named only the library. A following `/` is fine -- `so3/usr/src`
            # names the corpus as plainly as `so3` does.

            pattern = rf"(?<![\w-]){re.escape(word)}(?![\w-])"

            if re.search(pattern, remaining, re.I):
                named.add(name)
                remaining = re.sub(pattern, " ", remaining, flags=re.I)

                break

    keep = named or ({primary} if primary in AUTO_CORPUS_BY_COLLECTION.values()
                     else set(AUTO_CORPUS_BY_COLLECTION.values()))

    return [(col, prefix) for col, prefix in corpora
            if col.name not in AUTO_CORPUS_BY_COLLECTION
            or AUTO_CORPUS_BY_COLLECTION[col.name] in keep]


def retrieve_context(corpora, query, top_k=TOP_K):
    """Retrieve from one corpus or several, fusing the rankings.

    `corpora` is a collection, or a list of (collection, prefix) pairs. With
    several, each contributes its own top-k and Reciprocal Rank Fusion merges
    them: a chunk ranked first in a small corpus weighs as much as the first of
    a large one, which is exactly what a single merged index cannot offer.
    """

    if not isinstance(corpora, list):
        corpora = [(corpora, "")]

    corpora = select_corpora(corpora, query)

    pool, lists = {}, []

    for col, prefix in corpora:
        try:
            sub_pool, sub_lists = _retrieve_one(col, query, top_k)
        except Exception:
            continue        # one broken corpus must not sink the whole turn

        name = getattr(col, "name", id(col))

        for i, (d, m) in sub_pool.items():
            # Ids are md5(relpath) per corpus, so two corpora can collide on
            # the same relative path. Key by corpus as well.

            pool[(name, i)] = (d, m, prefix)

        lists.extend([[(name, i) for i in l] for l in sub_lists])

    order = _rrf(lists) if len(lists) > 1 else (lists[0] if lists else [])

    parts, total, seen = [], 0, set()

    for key in order[:top_k * 2]:
        doc, meta, prefix = pool[key]
        doc, rooted = _reroot(doc, meta, prefix)

        if total + len(doc) > MAX_CONTEXT_CHARS:
            continue        # `continue`, not `break`: one large chunk must no
                            # longer condemn every chunk after it.
        parts.append(doc)
        total += len(doc)
        seen.add(rooted)

    return "\n\n---\n\n".join(parts), seen


# ── history persistence ──────────────────────────────────────────────

def load_history():
    if os.path.isfile(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)[-MAX_HISTORY:]
        except (json.JSONDecodeError, KeyError):
            pass

    return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-MAX_HISTORY:], f, ensure_ascii=False, indent=2)


ARCHIVE_FILE = f"{STATE_DIR}/history-archive.jsonl"


# Validated exchanges saved as fine-tuning samples (/good command).
#
# The system string a sample carries describes the assistant the sample is
# training, so it is a property of the deployment and not of the platform.
# It used to name one organisation's build system, its BitBake layers and its
# board names -- shipped to everyone, and written into every sample anybody
# collected. SPEAR_FT_SYSTEM says it; the default below says only what is
# true of any SPEAR deployment.
#
# A corpus builder that merges its own samples with these must be given the
# same string, or the merged set trains against two different systems.

# Where /good writes, and it accumulates, so it lives with everything else a
# session accumulates. It used to default into a fine-tuning corpus directory
# -- a place that is a training dataset on one machine and absent on every
# other -- which made the public runtime depend on a path it had no business
# knowing. Same shape as SPEAR_TRAJECTORY_FILE below, for the same reason.
EXPERIENCE_FILE = os.environ.get(
    "SPEAR_EXPERIENCE_FILE", f"{STATE_DIR}/experience.jsonl")
FT_SYSTEM = os.environ.get("SPEAR_FT_SYSTEM") or (
    "You are a coding assistant for this project's source tree and build "
    "system. Answer precisely and concisely, using the project's real "
    "procedures and paths."
)


# Agentic trajectories kept for a future fine-tune: the tool calls and their
# results, not just the final prose. What needs training is the BEHAVIOUR --
# run the change, judge the output, name what was not tested -- and none of
# that is visible in a question/answer pair.

TRAJECTORY_FILE = os.environ.get(
    "SPEAR_TRAJECTORY_FILE", f"{STATE_DIR}/trajectories.jsonl")


# Acceptance benches belong to the harness, not to the trees they judge: a
# project should not have to carry the test rig that rates the assistant, and
# a bench a project owns is a bench the assistant can edit.

BENCH_DIR = resource_dir("SPEAR_BENCH_DIR", "benches")


def project_bench():
    """The acceptance command this project declares, if any (projects.json)."""
    return (load_projects().get(PROJECT) or {}).get("bench")


def _as_list(value):
    """One declared command, several, or none — always a list."""

    if isinstance(value, str):
        return [value]

    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]

    return []


def bench_command():
    """Resolve the declared bench to something runnable.

    A bare name is looked up in BENCH_DIR and becomes its absolute path, so
    projects.json says `"bench": "sye_sol-ls.sh"` and nothing has to live in
    the project. Anything else is passed through as a shell command, which
    keeps a project-owned script possible for whoever wants one.
    """
    declared = project_bench()

    if not declared:
        return None

    candidate = os.path.join(BENCH_DIR, declared)

    return candidate if os.path.isfile(candidate) else declared


# The project's own build and tests exercise code the model has just written,
# so they run where the model's own commands run. Their own runner because
# their own ceiling: a build is minutes, not the 45 seconds a tool call gets.

PROJECT_VERIFY_TIMEOUT = int(os.environ.get("SPEAR_PROJECT_VERIFY_TIMEOUT", "900"))

# Whether a `tests/` package may be taken as "this is how the tree is tested".
# Off for real trees -- a project says how it is verified, it is not guessed
# at -- and on for the benchmark fixtures, which have no owner to ask.

INFER_TEST_COMMAND = os.environ.get(
    "SPEAR_INFER_TEST_COMMAND", "").strip().lower() in {"1", "true", "yes", "on"}
PROJECT_VERIFY_RUNNER = CommandRunner(
    sandbox=BubblewrapSandbox(timeout_seconds=PROJECT_VERIFY_TIMEOUT),
    timeout_seconds=PROJECT_VERIFY_TIMEOUT)


# Running the project's verification on the host is the operator's call to
# make, in advance and in writing. It is never a fallback: a sandbox that
# cannot be established is a verification that did not happen, and degrading
# it into an unconfined run of code the model just wrote is the one outcome
# worse than not verifying at all.

PROJECT_VERIFY_ON_HOST = os.environ.get(
    "SPEAR_PROJECT_VERIFY_ON_HOST", "").strip().lower() in {"1", "true", "yes", "on"}


def verify_project_command(command):
    """Run one of the project's own verification commands, confined.

    Returns (status, detail) with status "passed", "failed" or "not_run".
    "not_run" is the closed door: nothing ran, nothing is claimed, and the
    turn is left unjudged rather than told its build is broken.
    """

    if not command:
        return "passed", ""

    if PROJECT_VERIFY_ON_HOST:
        ok, output = project_build.run(command, PROJECT_ROOT, PROJECT_VERIFY_TIMEOUT)

        return ("passed" if ok else "failed"), output

    if WORKSPACE is None:
        return "not_run", ("no workspace boundary for project verification "
                           "(set SPEAR_PROJECT_VERIFY_ON_HOST=1 to accept "
                           "running it unconfined)")

    profile = ExecutionProfile.from_capabilities(
        {Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE,
         Capability.SHELL_COMPLEX})
    availability = PROJECT_VERIFY_RUNNER.ensure_sandbox(WORKSPACE, profile)

    if not availability.ok:
        return "not_run", (f"{availability.summary} — project verification did "
                           f"not run (set SPEAR_PROJECT_VERIFY_ON_HOST=1 to "
                           f"accept running it unconfined)")

    result = PROJECT_VERIFY_RUNNER.run_sandboxed(
        WORKSPACE, shell_argv(command), profile, availability=availability)

    if result.status == "cancelled":
        return "not_run", result.summary

    if result.status == "ok" and result.exit_code in (None, 0):
        return "passed", ""

    return "failed", project_build.summarize(
        (result.stdout or "") + (result.stderr or "") or result.summary)


def run_project_bench(agent_context=None):
    """Run the project's acceptance command. True only on a clean exit.

    Run BY THE HARNESS, never offered to the model: it is the judge, so it
    must not be something the model can talk its way past, mistake for its own
    work, or edit. Returns None when the project declares no bench — no
    verdict, rather than a fabricated pass.
    """
    cmd = bench_command()

    if not cmd or WORKSPACE is None or COMMAND_RUNNER is None:
        return None

    task_id = agent_context.task_id if agent_context is not None else new_task_id()
    trace = agent_context.trace if agent_context is not None else TRACE
    action_id = new_action_id("project_bench")
    span = trace.start_span(
        EventType.VERIFICATION_STARTED,
        EventType.VERIFICATION_FINISHED,
        EventType.VERIFICATION_FINISHED,
        task_id,
        action_id=action_id,
        metadata={"kind": "project_bench", "command_configured": True},
    )

    # The bench lives outside the workspace, so it must be mounted for the run
    # or the sandbox cannot see it. Read-only would be cleaner still, but the
    # workspace itself must stay writable for a bench that builds.

    workspace = WORKSPACE

    if os.path.isdir(BENCH_DIR):
        try:
            workspace = Workspace.from_path(
                WORKSPACE.root,
                allow_absolute_paths=WORKSPACE.allow_absolute_paths,
                extra_roots=tuple(WORKSPACE.extra_roots) + (BENCH_DIR,))
        except (PathPolicyError, OSError):
            span.finish(
                status=EventStatus.FAILED,
                error_category="bench_workspace",
                error_summary="could not prepare bench workspace",
                metadata={"kind": "project_bench", "verdict": None},
            )

            if agent_context is not None:
                AgentRuntime._record_verification(
                    agent_context,
                    agent_context.verification_policy.project_bench_evidence(
                        agent_context.working_state, executed=False, passed=None,
                        action_id=action_id,
                        summary="bench workspace could not be prepared",
                    ),
                )

            return None

    profile = ExecutionProfile.from_capabilities(
        {Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE,
         Capability.SHELL_COMPLEX})
    availability = COMMAND_RUNNER.ensure_sandbox(workspace, profile)

    if not availability.ok:
        span.finish(
            status=EventStatus.FAILED,
            error_category="sandbox_unavailable",
            error_summary=availability.summary,
            metadata={"kind": "project_bench", "verdict": None},
        )

        if agent_context is not None:
            AgentRuntime._record_verification(
                agent_context,
                agent_context.verification_policy.project_bench_evidence(
                    agent_context.working_state, executed=False, passed=None,
                    action_id=action_id, summary="sandbox unavailable",
                ),
            )

        return None

    result = COMMAND_RUNNER.run_sandboxed(
        workspace, shell_argv(cmd), profile, availability=availability)
    verdict = result.status == "ok" and result.exit_code in (None, 0)
    span.finish(
        status=EventStatus.OK if verdict else EventStatus.FAILED,
        error_category=None if verdict else result.status,
        error_summary=None if verdict else result.summary,
        metadata={
            "kind": "project_bench",
            "verdict": verdict,
            "exit_code": result.exit_code,
            "stdout_chars": len(result.stdout),
            "stderr_chars": len(result.stderr),
        },
    )

    if agent_context is not None:
        AgentRuntime._record_verification(
            agent_context,
            agent_context.verification_policy.project_bench_evidence(
                agent_context.working_state, executed=True, passed=verdict,
                action_id=action_id,
                summary="project bench passed" if verdict else "project bench failed",
            ),
        )

    return verdict


def save_trajectory(question, trajectory, answer, verdict, source, task_id=None):
    """Append one rated trajectory, in the shape a trainer consumes.

    Failures are kept too, labelled: a dataset of successes alone cannot teach
    what to stop doing, and filtering later is free while re-running a session
    is not. So is a turn no bench judged: "unrated" says the verdict is
    missing, which a trainer can filter on -- dropping the turn says nothing
    happened, which is false.
    """
    sample = {
        # When the turn happened, which is what a converted episode carries.
        # Without it the converter had to invent one, and an invented time
        # made the same row convert to different content on every pass.
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,          # "pass" | "fail" | "unrated"
        # "bench" | "project_build" | "change" | "answer" | "user"
        "source": source,
        "project": PROJECT,
        "task_id": task_id,
        "bench": project_bench(),
        "question": question,
        "steps": trajectory,
        "answer": answer,
    }
    os.makedirs(os.path.dirname(TRAJECTORY_FILE), exist_ok=True)

    with open(TRAJECTORY_FILE, "a") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(TRAJECTORY_FILE) as f:
        return sum(1 for _ in f)


def save_experience(question, answer):
    """Append a validated exchange to the fine-tuning experience dataset.
    Format = {"messages": [...]} lines, the same shape a corpus builder
    emits, so the two sample sources merge without translation."""
    sample = {"messages": [
        {"role": "system", "content": FT_SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]}

    os.makedirs(os.path.dirname(EXPERIENCE_FILE) or ".", exist_ok=True)

    with open(EXPERIENCE_FILE, "a") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(EXPERIENCE_FILE) as f:
        return sum(1 for _ in f)


def archive_entry(entry):
    """Permanent trace: append-only, timestamped, never truncated.
    Survives MAX_HISTORY rotation and /clear (unlike history.json)."""
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), **entry}

    with open(ARCHIVE_FILE, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    archive_index(rec)


# ── auto-detect intent → run tools before LLM ───────────────────────

# ── skill library (Hermes-style, confirmation-gated) ────────────────
# Skills are reusable markdown procedures the model writes after completing
# a task. Stored as files + embedded in a dedicated ChromaDB collection;
# the most similar skills are injected into the prompt on each turn.

SKILLS_DIR = resource_dir("SPEAR_SKILLS_DIR", "skills")
SKILLS_COLLECTION = "edgem_skills"
ARCHIVE_COLLECTION = "edgem_archive"

# Reconciling the directory with the index is a per-session job, not a
# per-turn one: the files only change when this process writes one.

_SKILLS_SYNCED = False


def _db():
    """The Chroma client, and the only place chromadb is imported.

    Importing it at module level cost 0.78 s on every launch -- including the
    ones that never open the index (--help, a --model switch, a session that
    ends on a shell command). Deferred here, that second is paid on the first
    retrieval, where the caller is already waiting for the model.
    """
    import chromadb

    return chromadb.PersistentClient(path=DB_PATH)


def _skills_collection():
    return _db().get_or_create_collection(
        name=SKILLS_COLLECTION, metadata={"hnsw:space": "cosine"})


def skill_save(name, content, description="", scope=(), requires=()):
    """Write a skill file and embed it. Returns a result message."""

    try:
        skill = skill_library.save(SKILLS_DIR, name, content,
                                   description=description, scope=scope,
                                   requires=requires)
    except skill_library.SkillError as exc:
        return f"ERROR: {exc}"

    coll = _skills_collection()
    skill_library.reconcile([skill], coll, project=PROJECT)
    revised = f", revision {skill.version}" if skill.version > 1 else ""

    return (f"OK: skill '{skill.name}' saved{revised} "
            f"({coll.count()} skills in library)")


def skills_sync():
    """Reconcile the skill files with their index, once per session.

    The directory is the source of truth and the collection is derived from
    it, but only ``save_skill`` ever wrote to the collection: a skill added or
    edited by hand was listed by ``/skills`` and never injected, and a deleted
    one kept being injected. Reconciling on the first lookup costs a metadata
    read when nothing changed.
    """

    global _SKILLS_SYNCED

    if _SKILLS_SYNCED:
        return {}

    _SKILLS_SYNCED = True

    try:
        return skill_library.reconcile(
            skill_library.load_library(SKILLS_DIR), _skills_collection(),
            project=PROJECT)
    except Exception:
        # A library that will not reconcile must not take the turn with it:
        # the collection still holds whatever it held before.

        return {}


def skills_lookup(query, max_skills=2, max_dist=0.55):
    """Return the most relevant skills for this turn (empty if none close)."""

    skills_sync()

    try:
        coll = _db().get_collection(SKILLS_COLLECTION)
    except Exception:
        return ""

    count = coll.count()

    if count == 0:
        return ""

    # Asked wider than needed because what comes back is then filtered: a
    # skill whose scope or prerequisites rule it out must not consume one of
    # the two slots a turn has for a procedure.

    r = coll.query(query_texts=[query], n_results=min(max_skills * 4, count),
                   include=["documents", "distances"])

    # The record id is the skill name, and the file behind it is the source of
    # truth: a hit with no file left is a record reconciliation has not caught
    # up with, not a procedure to hand the model.

    library = {skill.name: skill for skill in skill_library.load_library(SKILLS_DIR)}
    docs = []

    for name, document, distance in zip(r["ids"][0], r["documents"][0],
                                        r["distances"][0]):
        skill = library.get(name)

        if distance > max_dist or skill is None:
            continue

        if not skill_library.applies_to(skill, project=PROJECT, kind=PROJECT_KIND):
            continue

        docs.append(document)

        if len(docs) == max_skills:
            break

    if not docs:
        return ""

    return ("\n\n## Relevant skills (procedures learned from past tasks)\n\n"
            + "\n\n---\n\n".join(docs))


def archive_index(rec):
    """Embed one archive record for search_history (incremental)."""
    content = (rec.get("content") or "")[:1500]

    if len(content) < 40:
        return

    try:
        coll = _db().get_or_create_collection(
            name=ARCHIVE_COLLECTION, metadata={"hnsw:space": "cosine"})
        rid = hashlib.md5((rec.get("ts", "") + content[:80]).encode()).hexdigest()[:16]
        coll.upsert(ids=[rid], documents=[content],
                    metadatas=[{"ts": rec.get("ts", ""),
                                "role": rec.get("role", ""),
                                "project": PROJECT}])
    except Exception:
        pass   # search index is best-effort, never break the chat


def archive_forget(pattern):
    """Remove archive-search entries whose text matches `pattern` (regex,
    case-insensitive). Prunes the SEARCH INDEX only — the append-only
    history-archive.jsonl trace on disk is never touched."""

    try:
        coll = _db().get_collection(ARCHIVE_COLLECTION)
    except Exception:
        return "(history index empty)"

    got = coll.get(include=["documents"])
    rx = re.compile(pattern, re.I)
    victims = [i for i, d in zip(got["ids"], got["documents"]) if rx.search(d or "")]

    if victims:
        coll.delete(ids=victims)

    return f"removed {len(victims)} entries (index now {coll.count()})"


def archive_search(query, n=5):
    try:
        coll = _db().get_collection(ARCHIVE_COLLECTION)
    except Exception:
        return "(history index empty)"

    if coll.count() == 0:
        return "(history index empty)"

    # over-fetch, then filter: drop self-echoes (the query itself, just
    # indexed), dedupe, and rank entries holding REAL tool output first —
    # the archive also contains past hallucinated answers, and grounded
    # excerpts are the antidote, not more model prose.

    r = coll.query(query_texts=[query],
                   n_results=min(4 * n, coll.count()),
                   include=["documents", "metadatas", "distances"])
    seen, hits = set(), []

    for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0],
                               r["distances"][0]):
        if dist < 0.05:                       # the query echoing itself
            continue

        key = doc[:120]

        if key in seen:
            continue

        seen.add(key)
        hits.append((0 if "[tool]" in doc or "$ " in doc else 1, dist,
                     doc, meta))

    hits.sort(key=lambda h: (h[0], h[1]))
    out = []

    for _, _, doc, meta in hits[:n]:
        out.append(f"[{meta.get('ts','?')} · {meta.get('role','?')}"
                   f" · {meta.get('project', meta.get('checkout','?'))}]\n{doc[:400]}")

    return "\n\n".join(out) if out else "(no match)"


def auto_detect_and_run(user_msg, collection):
    """
    Detect if the user is asking to read/list/grep something.
    If so, run the tool and return the result to inject into LLM context.
    Returns (extra_context, description) or (None, None).
    """
    msg = user_msg.lower()

    # "cherche sur internet/le web/en ligne X" → web search
    # (must precede the local grep detection, which also matches "cherche")

    m = re.search(
        r"(?:cherche|recherche|trouve|regarde)[a-z]*\s+(?:moi\s+)?(?:sur\s+)?"
        r"(?:internet|le web|le net|en ligne|web|google)\s*[:,]?\s*(.+)",
        msg,
    )

    if m:
        query = m.group(1).strip().rstrip("?").strip()

        if query:
            tool_use("Web", query, color=C_TOOL)

            with Spinner("Searching the web…"):
                result = web_search(query)

            show_web_sources(result)

            return f"[web results]\n{result}", f"Web results for '{query}'"

    # Memory-reference questions ("te souviens-tu", "la dernière fois",
    # "qu'on avait corrigé") → pre-run a semantic search over the archive.
    # Same rationale as the freshness detector: the local model answers
    # from (polluted) rolling history instead of calling search_history.

    if re.search(r"souviens|rappelle[sz]?[- ]|derni[eè]re fois|qu'?on avait|"
                 r"on avait (vu|fait|corrig|r[ée]solu)|remember when|"
                 r"last time we", msg):
        tool_use("History", user_msg[:70], color=C_TOOL)
        result = archive_search(user_msg, n=4)
        tool_result(result, max_lines=6)

        return (f"[past conversation excerpts found by semantic search — "
                f"ground your answer on these, do not invent details]\n"
                f"{result}", "Archive search")

    # Freshness questions ("dernière version de X", "latest release") →
    # pre-run a web search. The local model won't reliably call web_search
    # on its own (it answers from stale training data with invented dates);
    # the v3 adapter should learn the reflex — this is the safety net.

    if (re.search(r"(derni[eè]re|latest|last|newest|nouvelle|plus r[ée]cente)"
                  r"\s+(version|release|stable)|vient de sortir|just released",
                  msg)
            and not re.search(r"projet|notre|utilis|du repo|dans le code|"
                              r"du build|in the (project|repo|code)", msg)):
        query = re.sub(r"^(quelle?\s+est\s+la\s+|what\s+is\s+the\s+)", "",
                       user_msg.strip().rstrip(" ?"))
        tool_use("Web", query, color=C_TOOL)

        with Spinner("Searching the web…"):
            result = web_search(query)

        show_web_sources(result)

        return (f"[web results — ground your answer on these; your training "
                f"data is outdated for release questions]\n{result}",
                f"Web results for '{query}'")

    # "lis/lire/montre/affiche le fichier X"

    m = re.search(
        r"(?:lis|lire|montre|affiche|ouvre|cat|regarde|voir)\s+(?:le fichier\s+|moi\s+)?([^\s,]+\.\w+)",
        msg,
    )

    if m:
        path = m.group(1)
        tool_use("Read", path)
        content = read_file(path)

        return content, f"Contenu de {path}"

    # "liste/parcours/ls le répertoire X" or "quels fichiers dans X"

    m = re.search(
        r"(?:liste|parcour[st]?|ls|quels? fichiers?|répertoire|dossier|contenu de|contenu du)\s+(?:le |du |de |dans |)\s*([^\s,]+/?)",
        msg,
    )

    if m:
        path = m.group(1).rstrip("?")

        if path and not path.endswith((".bb", ".bbclass", ".inc", ".conf", ".py", ".sh", ".md")):
            tool_use("List", path)
            result = run_cmd(f"ls -la {path}", need_confirm=False)

            return result, f"Contenu de {path}"

    # "cherche/grep/trouve X dans Y" or just "grep X"

    m = re.search(
        r"(?:cherche|grep|trouve|recherche|où est)\s+['\"]?(\S+)['\"]?\s*(?:dans\s+(\S+))?",
        msg,
    )

    if m:
        pattern = m.group(1)
        where = m.group(2) or "build/"
        cmd = f"grep -rn --max-count=30 '{pattern}' {where} --include='*.bb' --include='*.bbclass' --include='*.bbappend' --include='*.inc' --include='*.conf'"
        tool_use("Grep", pattern + " dans " + where)
        result = run_cmd(cmd, need_confirm=False)

        return result, f"Résultats grep pour '{pattern}'"

    # "corrige/fixe/répare/remplace/change X en Y dans Z"

    m = re.search(
        r"(?:corrige|fixe|répare|remplace|change|modifie|renomme|fix|correct|replace)"
        r".*?(?:dans|in|le fichier|fichier)\s+([^\s,]+\.\w+)",
        msg,
    )

    if m:
        path = m.group(1)
        tool_use("Read", path + " (pour modification)")
        content = read_file(path)

        return content, f"Lecture de {path} pour modification"

    # "corrige/fixe" without an explicit file → look in recent history

    if re.search(r"(?:corrige|fixe|répare|fix|correct)", msg):
        return None, None

    return None, None


# ── LLM call ─────────────────────────────────────────────────────────

def sanitize_history(history):
    """Guarantees the strict user/assistant alternation required by the
    Qwen3 template: merges consecutive same-role messages, drops empties.
    Without this, an orphan turn (unsaved empty response) malformed the
    prompt → the model emitted EOS immediately → cascading empty replies."""
    clean = []

    for m in history:
        content = (m.get("content") or "").strip()

        if not content:
            continue

        if clean and clean[-1]["role"] == m["role"]:
            clean[-1]["content"] += "\n\n" + content
        else:
            clean.append({"role": m["role"], "content": content})

    return clean


CTX_LIMIT = int(os.environ.get("SPEAR_CTX", "32768"))


def _approx_tokens(s):
    return len(s) // 4 + 1


def enforce_ctx_budget(messages, max_out):
    """Return a COPY of `messages` shrunk so prompt + max_out fits the model
    context window. A 24k-token prompt + 8k requested output overflows a 32k
    model and the server replies 400 (whole turn lost). We only SHRINK content
    — never drop a message — so an assistant tool_call always keeps its tool
    reply (dropping one would make the request invalid). Trim order: the RAG
    block, then big tool outputs, then oldest history."""
    msgs = [dict(m) for m in messages]
    budget = CTX_LIMIT - max_out - 768          # leave a margin
    total = lambda: sum(_approx_tokens(m.get("content") or "") + 4
                        for m in msgs)

    if total() <= budget:
        return msgs

    # 1) trim the "## Retrieved Context" (RAG) tail of the system message

    for m in msgs:
        if m.get("role") == "system" and \
                "## Retrieved Context" in (m.get("content") or ""):
            head, sep, rag = m["content"].partition("## Retrieved Context")
            over = (total() - budget) * 4
            keep = max(0, len(rag) - over - 200)
            m["content"] = (head + sep + rag[:keep]
                            + "\n…(retrieved context trimmed to fit)…"
                            if keep else head
                            + "## (retrieved context dropped to fit window)")

            break

    # 2) cap big tool / result payloads, oldest first

    for m in msgs[1:]:
        if total() <= budget:
            break

        c = m.get("content") or ""
        is_result = (m.get("role") == "tool" or "[tool result]" in c
                     or "executed tool" in c or "Result of " in c)

        if is_result and len(c) > 1500:
            m["content"] = c[:1200] + "\n…(trimmed)…"

    # 3) last resort: shrink oldest history bodies (keep the last 3 intact)

    for m in msgs[1:-3]:
        if total() <= budget:
            break

        c = m.get("content") or ""

        if len(c) > 600:
            m["content"] = c[:500] + "\n…(trimmed)…"

    return msgs


def canonical_history(history):
    """Keep persisted text history unchanged while using typed turn messages."""
    return [ConversationMessage(
        message["role"], (TextBlock(message.get("content") or ""),)
    ) for message in history]


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
        ("memory:project", ContextLayer.DURABLE_MEMORY, MEMORIES_FILE,
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


def chat_once(client, messages, use_tools=True, spinner=None):
    """One STREAMING chat completion against llama-server (native tools).
    Streams so the spinner can show a live generated-token count, Claude
    Code style. Returns (text, calls) where calls is a list of dicts
    {"id", "name", "arguments"} accumulated from the tool-call deltas."""
    _max_out = int(os.environ.get("SPEAR_MAX_TOKENS", "8192"))
    messages = enforce_ctx_budget(messages, _max_out)
    kwargs = dict(
        # llama.cpp ignores the model name; vLLM matches it against
        # --served-model-name, so make it configurable (SPEAR_MODEL_NAME).
        model=os.environ.get("SPEAR_MODEL_NAME", "qwen3"),
        messages=messages,
        # Sampling temperature (SPEAR_TEMP). 0.25 gives more conservative,
        # consistent edits on Qwen3-Coder-Next (less prone to over-eager
        # refactors like dropping includes). NOTE: on the older dense/A3B model
        # a low temp starved the sampler into repetition loops; if that recurs,
        # raise SPEAR_TEMP back toward 0.7.
        temperature=float(os.environ.get("SPEAR_TEMP", "0.25")),
        top_p=0.8,
        # Big enough that a full file rewrite (write_file with the whole new
        # content, or a large edit) completes instead of being truncated
        # mid-tool-call — a cut-off XML tool call can't be parsed/applied.
        max_tokens=_max_out,
        stream=True,
        extra_body={
            # disable thinking: without this the model may open a <think>
            # block that never closes -> empty content (the whole response
            # ends up in reasoning_content).
            "chat_template_kwargs": {"enable_thinking": False},
            # extra samplers — both penalty spellings so it works on either
            # backend: llama.cpp reads "repeat_penalty", vLLM
            # "repetition_penalty"; each ignores the other.
            "top_k": 20,
            "repeat_penalty": 1.05,
            "repetition_penalty": 1.05,
        },
    )

    if use_tools:
        kwargs["tools"] = TOOLS

    content = []
    tcalls = {}
    ntok = 0
    last_line = None
    consec_repeats = 0

    # The big Q8 (35 GB) model can still be loading when llama-server's
    # /health already answers 200, so the first completion gets a 503
    # "Loading model". Wait it out instead of crashing the chat.

    stream = None
    tool_parse_retries = 0

    for attempt in range(120):  # ~10 min max at 5 s between tries
        try:
            stream = client.chat.completions.create(**kwargs)
            break
        except APIStatusError as e:
            if e.status_code == 503 and "loading" in str(e).lower():
                if spinner:
                    spinner.label = "Loading model (first request)…"
                else:
                    msg = "  model still loading" + "." * (attempt % 4)
                    print(f"\r{C_DIM}{msg}   {C_RST}", end="", flush=True)

                time.sleep(5)

                continue

            # 500: the model emitted a tool call whose JSON arguments
            # llama-server could not parse (bad quoting/backslashes in a
            # bash/grep command). Sampling varies, so a few retries usually
            # yield a well-formed call; give up gracefully after that.

            if e.status_code == 500 and "tool call" in str(e).lower():
                tool_parse_retries += 1

                if tool_parse_retries <= 3:
                    if spinner:
                        spinner.label = "Retrying (malformed tool call)…"

                    time.sleep(1)

                    continue

                raise RuntimeError(
                    "the model kept emitting an unparseable tool call "
                    "(bad JSON quoting). Rephrase your request or try again."
                ) from e

            raise
        except APIConnectionError:
            # server not up yet / restarting — same patient wait

            time.sleep(5)
            continue

    if stream is None:
        raise RuntimeError("llama-server never became ready (still loading "
                           "after ~10 min) — check llama-server.log")

    for chunk in stream:
        if not chunk.choices:
            continue

        d = chunk.choices[0].delta

        if d is None:
            continue

        if d.content:
            content.append(d.content)
            ntok += 1

            # circuit breaker: a real death-loop repeats the SAME long line
            # many times IN A ROW. (Counting total occurrences false-triggers
            # on legitimately repeated code lines — e.g. `return -1;` or
            # `tmp = atoi(argv[arg + 1]);` in a switch — and truncates valid
            # file rewrites.) So only trip on consecutive identical long lines.

            if "\n" in d.content:
                line = "".join(content).rstrip().rsplit("\n", 1)[-1].strip()

                if len(line) > 20:
                    if line == last_line:
                        consec_repeats += 1
                    else:
                        last_line = line
                        consec_repeats = 0

                    if consec_repeats >= 8:
                        stream.close()
                        content.append("\n\n(repetition loop detected — "
                                       "response truncated)")

                        break

        if d.tool_calls:
            for tcd in d.tool_calls:
                e = tcalls.setdefault(tcd.index,
                                      {"id": None, "name": "", "args": []})

                if tcd.id:
                    e["id"] = tcd.id

                if tcd.function:
                    if tcd.function.name:
                        e["name"] += tcd.function.name

                    if tcd.function.arguments:
                        e["args"].append(tcd.function.arguments)

            ntok += 1

        if spinner:
            spinner.tokens = ntok

    calls = [{"id": e["id"] or f"call_{i}", "name": e["name"],
              "arguments": "".join(e["args"])}
             for i, e in sorted(tcalls.items())]

    return "".join(content), calls


def _usage_token(usage, *names):
    if not usage:
        return None

    for name in names:
        value = usage.get(name)

        if isinstance(value, int):
            return value

    return None


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


# Native declarations are owned by tool_registry.native_tool_specs().  The
# registry instance is assembled after the concrete handlers are defined.

# The standing tool-usage rules now live in an editable file. Edit
# tool-guide.md to tune the model's tool behavior (no code change needed);
# the inline fallback below only kicks in if the file goes missing.

TOOL_GUIDE_FILE = f"{APP_DIR}/tool-guide.md"
TOOL_GUIDE_FALLBACK = (
    "\n\n## Tool usage rules\n"
    "- Never invent command output; never claim a fix is applied unless a "
    "tool result confirms it.\n"
    "- Always answer in the language of the user's question."
)


def load_tool_guide():
    try:
        with open(TOOL_GUIDE_FILE) as f:
            content = f.read().strip()

        return "\n\n" + content if content else TOOL_GUIDE_FALLBACK
    except OSError:
        return TOOL_GUIDE_FALLBACK


TOOL_GUIDE = load_tool_guide()


def show_diff(old_text, new_text, max_lines=14):
    """Show the lines that actually CHANGED.

    Printing the head of old_text and then the head of new_text is not a diff.
    When the edit lands past the cut -- an #include inserted below six
    unchanged ones -- both sides render identically and the user reads
    "OK: updated" under a diff showing no change at all. That is unreviewable
    in auto mode, where the diff is the only thing standing between the model
    and the file.

    An edit whose sides are identical is reported as such rather than drawn as
    a change: a no-op edit is worth seeing, not hiding.
    """
    old, new = old_text.split("\n"), new_text.split("\n")
    rows = []

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, old, new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue

        rows += [("-", C_ERR, l) for l in old[i1:i2]]
        rows += [("+", C_OK, l) for l in new[j1:j2]]

    if not rows:
        print(f"  {C_DIM}⎿  (no change: old and new text are identical){C_RST}")
        return

    for i, (sign, color, l) in enumerate(rows[:max_lines]):
        pfx = "⎿  " if i == 0 else "   "
        print(f"  {C_DIM}{pfx}{C_RST}{color}{sign} {l[:150]}{C_RST}")

    if len(rows) > max_lines:
        print(f"  {C_DIM}   … +{len(rows) - max_lines} more changed lines{C_RST}")


DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.M)


def looks_like_diff(text):
    """True if text is a unified diff (hunk header, or ---/+++ file headers)."""

    if not text:
        return False

    if DIFF_HUNK_RE.search(text):
        return True

    return text.lstrip().startswith("--- ") and "\n+++ " in text


def render_diff(diff_text, max_lines=60):
    """Print a unified diff Claude Code style: + green, - red, @@ cyan,
    file headers and context dim."""
    lines = diff_text.split("\n")

    for n, ln in enumerate(lines):
        if n >= max_lines:
            print(f"  {C_DIM}   … (+{len(lines) - n} more diff lines){C_RST}")
            break

        pfx = "⎿  " if n == 0 else "   "
        body = ln[:200]

        if ln.startswith(("+++", "---")):
            print(f"  {C_DIM}{pfx}{body}{C_RST}")
        elif ln.startswith("@@"):
            print(f"  {C_DIM}{pfx}{C_RST}{C_TOOL}{body}{C_RST}")
        elif ln.startswith("+"):
            print(f"  {C_DIM}{pfx}{C_RST}{C_OK}{body}{C_RST}")
        elif ln.startswith("-"):
            print(f"  {C_DIM}{pfx}{C_RST}{C_ERR}{body}{C_RST}")
        else:
            print(f"  {C_DIM}{pfx}{body}{C_RST}")


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


# Code-block fallback: code-tuned models (Qwen2.5-Coder) often print the whole
# improved file in a fenced ``` block instead of calling write_file. Detect
# that and apply it to the file the user named.

CODE_BLOCK_RE = re.compile(r"```([\w.+-]*)\n(.*?)```", re.DOTALL)
SRC_NAME_RE = re.compile(
    r"\b([\w./-]+\.(?:c|h|cpp|hpp|cc|cxx|S|s|py|sh|mk|cmake|md|rst|txt|dts|"
    r"dtsi|its|conf|cfg|ini|bb|bbappend|bbclass|inc|yaml|yml|json))\b")


def extract_code_block_with_language(text):
    """The largest fenced block, with the language the fence declared.

    The language is kept because it is the cheapest evidence of what the block
    IS, and the caller has to decide whether it belongs in the file it is about
    to overwrite. Returns ('', '') when there is no block.
    """
    blocks = [(m.group(2), m.group(1)) for m in CODE_BLOCK_RE.finditer(text or "")]

    if not blocks:
        return "", ""

    block, language = max(blocks, key=lambda item: len(item[0]))

    return block.rstrip("\n"), language.strip().lower()


def extract_code_block(text):
    """Return the largest fenced code block's content, or '' if none."""
    return extract_code_block_with_language(text)[0]


# What a fence language says the block is. A language absent from this table is
# not evidence of anything and never refuses on its own.

BLOCK_LANGUAGE_EXTENSIONS = {
    "c": {".c", ".h"}, "h": {".c", ".h"},
    "cpp": {".cpp", ".hpp", ".cc", ".cxx", ".h"},
    "c++": {".cpp", ".hpp", ".cc", ".cxx", ".h"},
    "python": {".py"}, "py": {".py"},
    "sh": {".sh"}, "bash": {".sh"}, "shell": {".sh"},
    "make": {".mk"}, "makefile": {".mk"}, "cmake": {".cmake"},
    "yaml": {".yaml", ".yml"}, "yml": {".yaml", ".yml"}, "json": {".json"},
    "rst": {".rst"}, "restructuredtext": {".rst"},
    "markdown": {".md"}, "md": {".md"},
    "dts": {".dts", ".dtsi"},
}

# A whole line of ==== / #### / ---- is a heading underline, and a line opening
# with `.. ` is a directive. Neither occurs in compilable C.

PROSE_MARKER_RE = re.compile(r"^\s*\.\.\s+\w|^[=#~^*+-]{4,}\s*$", re.M)
COMPILED_SOURCE_EXTENSIONS = {".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".s", ".S"}


def block_fits_target(block, language, target):
    """Could this block plausibly BE the target file?

    The fallback writes a printed block into a file named in the user's own
    sentence, and those two are not the same thing.  "add a chapter describing
    the ls.c application" names ``ls.c`` as the *subject*; the block the model
    printed was reStructuredText.  Applying it would have destroyed a source
    file in order to answer a documentation request — it was stopped only
    because ``write_file`` refuses to overwrite an existing file nobody read
    this turn.  A target that did not exist yet would simply have been created.

    So the fallback now has to believe the block belongs in that file.  The
    check refuses only on positive evidence of a mismatch, because this path
    exists for models that cannot call tools reliably and a false refusal costs
    them their edit.
    """
    extension = os.path.splitext(str(target))[1]
    declared = BLOCK_LANGUAGE_EXTENSIONS.get(language or "")

    if declared is not None and extension and extension not in declared:
        return False

    if extension in COMPILED_SOURCE_EXTENSIONS:
        if "#include" not in block and ";" not in block:
            return False

        if PROSE_MARKER_RE.search(block):
            return False

    return True


READ_CMD_RE = re.compile(
    r"\b(?:cat|head|tail|sed|bat|less|more|nl)\b[^|;&]*?"
    r"([\w./-]+\.[A-Za-z]\w*)")


def is_excluded_path(p):
    """Third-party / snapshot trees we must NEVER auto-write into."""
    return bool(re.search(r"/(?:u-boot|atf|qemu)(?:\.back)?/|\.back/|"
                          r"\.pristine/|\.0/|/build/tmp/", str(p)))



def guess_target_file(user_input, tool_log=()):
    """The file the user means by e.g. 'improve ping.c'. A bare basename is
    ambiguous (several ping.c exist), so PREFER the file the model actually
    read this turn (cat/head/… in tool_log) whose basename matches; only fall
    back to a tree-wide find_file if nothing was read. NEVER return a path in
    an excluded third-party/snapshot tree (it would corrupt u-boot etc.)."""
    names = [m.group(1) for m in SRC_NAME_RE.finditer(user_input or "")]

    if not names:
        return None

    bnames = {os.path.basename(n) for n in names}

    # 1) a file the model read this turn

    for entry in tool_log:
        for m in READ_CMD_RE.finditer(entry):
            cand = m.group(1)

            if os.path.basename(cand) in bnames:
                try:
                    p = resolve_path(cand)
                except PathPolicyError:
                    continue

                if p.is_file() and not is_excluded_path(p):
                    return p

    # 2) fallback: tree-wide search, but skip third-party/snapshot trees

    for n in names:
        p = find_file(n)

        if p and not is_excluded_path(p):
            return p

    return None


# The <tool_call> wrapper is OPTIONAL: Qwen-Coder frequently emits a bare
# <function=NAME>…</function> (no wrapper), which would otherwise be dropped
# silently — the model then thinks it acted while the file is untouched.

LEAKED_CALL_RE = re.compile(
    r"(?:<tool_call>\s*)?<function=(\w+)>(.*?)</function>(?:\s*</tool_call>)?",
    re.DOTALL)
LEAKED_PARAM_RE = re.compile(
    r"<parameter=(\w+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def parse_leaked_tool_calls(text):
    """llama-server's XML tool-call parser occasionally leaks calls as raw
    text (multi-line parameters). Recover them client-side.
    Returns (cleaned_text, [(name, args), ...])."""
    calls = []

    for m in LEAKED_CALL_RE.finditer(text):
        name = m.group(1)
        args = {k: v for k, v in LEAKED_PARAM_RE.findall(m.group(2))}
        calls.append((name, args))

    if calls:
        text = LEAKED_CALL_RE.sub("", text).strip()

    # also recover bare-JSON tool calls (Qwen-style, no <tool_call> wrapper):
    #   {"name": "bash", "arguments": {"command": "..."}}

    text, json_calls = parse_json_tool_calls(text)

    return text, calls + json_calls


JSON_CALL_RE = re.compile(r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:')


def parse_json_tool_calls(text):
    """Recover tool calls the model prints as bare JSON text (Qwen/Coder
    format, no XML wrapper): {"name": "X", "arguments": {...}}. Brace-matches
    the full object so nested JSON (edit_file's escaped old/new text) parses.
    Returns (cleaned_text, [(name, args_dict), ...])."""
    calls, out, i = [], [], 0

    while True:
        m = JSON_CALL_RE.search(text, i)

        if not m:
            out.append(text[i:])
            break

        out.append(text[i:m.start()])

        # brace-match from the opening { to the matching }

        depth, j, instr, esc = 0, m.start(), False, False

        while j < len(text):
            c = text[j]

            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = not instr
            elif not instr:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1

                    if depth == 0:
                        j += 1
                        break

            j += 1

        blob = text[m.start():j]

        try:
            obj = json.loads(blob)
            name, args = obj.get("name"), obj.get("arguments")

            if name and isinstance(args, dict):
                calls.append((name, args))
            else:
                out.append(blob)
        except Exception:
            out.append(blob)

        i = j

    return "".join(out).strip(), calls


def try_surgical_edits(
    backend, system_prompt, conversation, rel, full_block, executed, tool_log,
    context_items=None, agent_context=None, runtime=None,
):
    """The model produced a WHOLE-FILE rewrite of an existing file. Give it ONE
    chance to express the same change as targeted edit_file calls instead — a
    full rewrite silently drops declarations / CLI options / third-party
    headers and stops compiling. Returns True iff >=1 surgical edit applied.
    Only edit_file/append_file count here; a write_file (whole file again)
    defeats the purpose and is ignored so the caller falls back to the block."""
    nudge = (
        "STOP. Do NOT rewrite the whole file. Your change must be expressed as "
        "one or more edit_file calls that touch ONLY the lines that actually "
        "change. Keep every existing declaration, global, helper function and "
        "CLI option, and do NOT alter the existing copyright header. For each "
        "change emit:\n"
        "<tool_call><function=edit_file><parameter=path>" + rel + "</parameter>"
        "<parameter=old_text>EXACT existing lines</parameter>"
        "<parameter=new_text>replacement</parameter></function></tool_call>\n"
        "Several small edits are better than one big one. Emit the edit_file "
        "call(s) now — no prose, no full-file code block.")
    msgs = conversation + [
        ConversationMessage("assistant", (TextBlock(full_block[:4000]),)),
        ConversationMessage("user", (TextBlock(nudge),)),
    ]

    try:
        with Spinner("Refining to a minimal edit…") as sp:
            turn = (runtime or AgentRuntime()).complete_model_turn(
                agent_context, conversation=msgs, use_tools=True,
                on_token=lambda: setattr(sp, "tokens", sp.tokens + 1),
            )
    except Exception:
        return False

    pending = [(call.name, dict(call.arguments)) for call in turn.tool_calls]
    applied = False

    for name, targs in pending:
        if name not in ("edit_file", "append_file"):
            continue            # ignore whole-file write_file here

        try:
            res = execute_tool(
                name, targs, executed, agent_context=agent_context,
            )
        except Exception as e:
            res = f"ERROR: tool '{name}' failed: {e}"

        tool_log.append(f"{name} {json.dumps(targs, ensure_ascii=False)[:200]}\n"
                        f"{res[:400]}")

        if isinstance(res, str) and res.startswith("OK"):
            applied = True

    return applied


# Reserved key in the per-turn cache, holding the set of files whose CONTENT
# the model has actually seen this turn. An object() cannot collide with a
# command string, which is what every other key in that dict is.

READ_PATHS = object()

# Set once the sandbox has been found unavailable this turn. It does not come
# back: bwrap is missing, or the kernel refuses it, and every later command
# fails the same way. What followed was worse than the failure -- the model ran
# eight commands, got eight refusals, then edited a file nine times on the
# strength of retrieved context alone, unable to compile or even read what it
# was changing. An edit that cannot be verified is not worth making.

SANDBOX_DOWN = object()

# The commands the policy refused this turn. Distinct from the result cache,
# because a refusal never ran: replaying it under "already executed" told a
# model its denied `sed -i` had worked.

DENIED_COMMANDS = object()

#: Blocks of source this turn has already been shown, and the mutation
#: generation they were read at. Reading a file in overlapping windows is the
#: single largest cost a governed turn pays: measured across four runs of one
#: two-turn workflow, 54% to 62% of every windowed read landed entirely on
#: lines the turn had already seen. The router's cache cannot catch those --
#: `sed -n '590,720p'` and `sed -n '600,700p'` are different arguments and the
#: same evidence.
REGIONS_READ = object()


def _already_in_evidence(context, command):
    """The note to return instead of re-reading lines already shown.

    Empty when the command is not a windowed read, when any of its blocks are
    new, or when something has been written since they were read -- a file
    that changed is a file worth reading again, and that is the whole reason
    the generation is part of the key.
    """
    blocks = progress_monitor.read_evidence(command)

    if not blocks:
        return ""

    seen = context.cache.setdefault(REGIONS_READ, {})
    generation = _mutation_generation(context)
    fresh = [block for block in blocks
             if seen.get(block) != generation]

    for block in blocks:
        seen[block] = generation

    if fresh:
        return ""

    where = sorted({block.split("#", 1)[0] for block in blocks})

    return ("(ALREADY IN EVIDENCE this turn — these lines of "
            + ", ".join(where)
            + " are already above in this conversation, and nothing has been "
              "written since. Read them there. Widen the range, open a "
              "different file, or move on.)\n")


def _mutation_generation(context):
    """The router's own write counter, which is what makes a read stale."""
    ledger = context.cache.get(tool_router._REPEAT_KEY)

    return ledger.get("generation", 0) if isinstance(ledger, dict) else 0
SANDBOX_DOWN_MSG = (
    "REFUSED: the sandbox is unavailable, so nothing you write can be read "
    "back, compiled or run. Editing from retrieved context alone produces "
    "changes nobody can check. Stop and report that the sandbox is down — "
    "that IS the answer to give the user.")

# Commands that put a file's content in front of the model. `ls` and `stat`
# are deliberately absent: knowing a file exists is not knowing what is in it,
# and that is the whole point of the guard below.

CONTENT_READING_BINS = {"cat", "sed", "head", "tail", "grep", "egrep", "fgrep",
                        "nl", "awk", "less", "more", "strings", "diff", "rg"}


def _note_files_read(cache, cmd):
    """Record the files a bash command showed the model."""

    try:
        argv = shlex.split(cmd)
    except ValueError:
        return

    if not argv or os.path.basename(argv[0]) not in CONTENT_READING_BINS:
        return

    seen = cache.setdefault(READ_PATHS, set())

    for tok in argv[1:]:
        if tok.startswith("-"):
            continue

        try:
            fp = resolve_path(tok)
        except Exception:
            continue

        if os.path.isfile(fp):
            seen.add(os.path.realpath(fp))


def _was_read_this_turn(cache, fpath):
    return os.path.realpath(fpath) in cache.get(READ_PATHS, set())


def _classified_handler_result(
    text, *, mutation=False, affected_paths=(), read_paths=(), metadata=None,
):
    exit_match = re.search(r"\(exit\s+(-?\d+)\)", text)
    exit_code = int(exit_match.group(1)) if exit_match else None

    if text.startswith("CANCELLED"):
        status = ToolResultStatus.CANCELLED
    elif text.startswith("ERROR"):
        status = (ToolResultStatus.TIMEOUT if "timed out" in text.lower()
                  else ToolResultStatus.FAILED)
    elif exit_code not in (None, 0):
        status = ToolResultStatus.FAILED
    elif text.startswith("(ALREADY EXECUTED"):
        status = ToolResultStatus.CACHED
    else:
        status = ToolResultStatus.OK

    failed = status not in {ToolResultStatus.OK, ToolResultStatus.CACHED}

    return ToolHandlerResult(
        text=text,
        status=status,
        exit_code=exit_code,
        metadata=metadata or {},
        mutation=mutation and not failed,
        affected_paths=tuple(affected_paths) if not failed else (),
        read_paths=tuple(read_paths) if not failed else (),
        error_category=status.value if failed else None,
        error_summary=text.splitlines()[0][:240] if failed else None,
    )


def _record_bench_answer(text):
    """Append this turn's final answer to the benchmark's answer sink.

    Off unless SPEAR_BENCH_ANSWER_FILE names a file, which only the benchmark
    runner does. It exists so an informative task can be scored on what the
    run actually ANSWERED: reading the right files and then naming the wrong
    function is a failure, and nothing in the workspace shows it.
    """

    path = os.environ.get("SPEAR_BENCH_ANSWER_FILE")

    if not path or not text:
        return

    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n")
    except OSError:
        # Losing the sink costs the benchmark an oracle, never the turn.

        pass


def _workspace_label(path):
    try:
        return str(WORKSPACE.relative(path) if WORKSPACE is not None else path)
    except (PathPolicyError, OSError, TypeError):
        return "[workspace-path]"


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


def _registered_command(context, command):
    """The only registered command boundary; run_cmd retains all enforcement."""
    cmd = command.strip()
    print()
    tool_use("Bash", cmd)

    if cmd in context.cache:
        # A refusal is not an execution. Replaying one as "already executed"
        # is how a model concluded its denied `sed -i` had worked and moved on
        # believing the file was edited.

        refused = cmd in context.cache.get(DENIED_COMMANDS, ())
        tool_result("(refused before — same refusal)" if refused
                    else "(cached — already executed this turn)")
        print()

        if refused:
            # A replayed refusal is a refusal, not a success. Classified as
            # OK it cleared the router's failure tally for that command, so
            # the same refused command could be asked for indefinitely.

            return ToolHandlerResult(
                "REFUSED ALREADY this turn — it did not run, and repeating it "
                "will not change that. Take the other route named below.\n"
                + context.cache[cmd][:1500],
                ToolResultStatus.DENIED,
                error_category="repeated_denied_command",
                error_summary="command already refused this turn",
            )

        # The output is not repeated while the model can still see it. The
        # router decides that -- it knows which result block this command
        # produced and asks the caller whether that block survived compaction
        # -- and it puts the original content back when the answer is no. So
        # this pointer is only ever delivered alongside evidence that is
        # genuinely still in the conversation.

        return _classified_handler_result(
            "(ALREADY EXECUTED this turn — its output is already above in this "
            "conversation. Read it there. DO NOT run this command again.)\n"
        )

    # Lines this turn has already been shown, in whatever window it asked for
    # them. Checked before the command runs, because the cost being saved is
    # the round, not the disk.

    settled = _already_in_evidence(context, cmd)

    if settled:
        tool_result(settled.splitlines()[0])
        print()

        return _classified_handler_result(settled)

    reads_before = set(context.cache.get(READ_PATHS, set()))
    command_result = run_cmd_result(
        cmd, need_confirm=False, cancellation=context.cancellation,
        # The read-only roles have always run this way. A turn the USER told
        # not to change anything now does too: dropping the write tools from
        # the view is not enough on its own, because bash can write with
        # `sed -i`, `cp`, or a redirection. SAFE takes workspace:write away
        # for the whole turn, and the sandbox mounts the tree read-only.
        # ...and a turn whose write gate has not opened yet runs the same
        # way. Dropping edit_file from the model's reach is not enough on its
        # own: a turn refused an edit reached for `sed -i` within two rounds,
        # which is the same write through a different door. SAFE closes the
        # door rather than the doorway.
        execution_mode=(ExecutionMode.SAFE
                        if context.read_only
                        or _write_gate_closed(context)
                        or context.role in {"explorer", "reviewer", "planning"}
                        else None),
    )
    result = command_result_text(command_result)

    # A command that reports the sandbox missing is the only way this is
    # learned; every mutation after it in the same turn is refused.

    if "sandbox unavailable" in result or "bubblewrap sandbox" in result:
        context.cache[SANDBOX_DOWN] = True

    # A refusal that is about SCOPE, not about permissions. The turn was told
    # not to change anything; saying only "command denied" invites the next
    # spelling of the same intent, and a run spent thirty-five steps finding
    # them -- `sed -i`, a redirection, python. Say what the refusal is for,
    # and what to do instead.

    # Same shape, different reason. A command refused because the gate is
    # still shut is refused temporarily, and saying only "denied" invites the
    # next spelling of the same write instead of the investigation that would
    # open it.

    if (not context.read_only and command_result.status == "denied"
            and _write_gate_closed(context)
            and COMMAND_POLICY.classify(cmd).classification
            != CommandClassification.READ_ONLY):
        gate = context.write_gate("")
        result = result.rstrip() + "\n" + getattr(gate, "message", "")
        context.trace.emit(
            EventType.TOOL_CALL_FAILED, context.task_id,
            status=EventStatus.DETECTED, tool_name="bash",
            action_id=context.action_id,
            metadata={"reason": getattr(gate, "reason", "investigation_incomplete"),
                      "arguments_recorded": False},
        )

    if (context.read_only and command_result.status == "denied"
            and COMMAND_POLICY.classify(cmd).classification
            != CommandClassification.READ_ONLY):
        violations = context.cache.get(READ_ONLY_VIOLATIONS, 0) + 1
        context.cache[READ_ONLY_VIOLATIONS] = violations
        result = result.rstrip() + "\n" + READ_ONLY_REFUSAL
        context.trace.emit(
            EventType.READ_ONLY_VIOLATION, context.task_id,
            status=EventStatus.DETECTED, tool_name="bash",
            action_id=context.action_id,
            metadata={"attempt": violations,
                      "classification": COMMAND_POLICY.classify(cmd).classification.value,
                      "arguments_recorded": False},
        )

    tool_result("interrupted" if result.startswith("CANCELLED") else result)
    print()
    context.cache[cmd] = result

    if command_result.status == "denied":
        context.cache.setdefault(DENIED_COMMANDS, set()).add(cmd)

    if not result.startswith(("ERROR", "CANCELLED")):
        _note_files_read(context.cache, cmd)

    reads_after = set(context.cache.get(READ_PATHS, set()))
    status = {
        "ok": ToolResultStatus.OK,
        "cancelled": ToolResultStatus.CANCELLED,
        "denied": ToolResultStatus.DENIED,
        "timeout": ToolResultStatus.TIMEOUT,
    }.get(command_result.status, ToolResultStatus.FAILED)

    # A shell command that is not read-only may have written any file in the
    # workspace, and the mutations that mangled user_space.rst were exactly
    # that -- `head ... > /tmp/x && cat /tmp/x > target`, not an edit tool. So
    # the invalidation cannot live only in the mutation handlers. The command
    # itself stays cached: re-running a write is worse than re-reading.
    #
    # Nothing here measures what a command wrote -- the sandbox reports no
    # such thing -- so a non-read-only command that RAN is declared
    # potentially mutating and treated as though it had written, whatever its
    # exit code. A read-only one is measured: the classifier is the
    # measurement, and its count is a truthful zero.
    #
    # The decision itself is `invalidates_reads`, which the router also uses
    # for its generation counter. One function, one answer: there is no state
    # where this cache considers its reads stale and the router considers
    # them current.

    mutating = (COMMAND_POLICY.classify(cmd).classification
                != CommandClassification.READ_ONLY)
    metadata = {
        "execution_status": command_result.status,
        **({"potentially_mutating": True} if mutating else {"mutation_count": 0}),
        **({"read_only_violations": context.cache[READ_ONLY_VIOLATIONS]}
           if context.cache.get(READ_ONLY_VIOLATIONS) else {}),
    }

    if invalidates_reads(status, False, metadata):
        _forget_cached_reads(context, keep=cmd)

    return ToolHandlerResult(
        text=result,
        status=status,
        exit_code=command_result.exit_code,
        stdout=command_result.stdout,
        stderr=command_result.stderr,
        metadata=metadata,
        read_paths=tuple(_workspace_label(path)
                         for path in sorted(reads_after - reads_before)),
        error_category=None if status == ToolResultStatus.OK else status.value,
        error_summary=None if status == ToolResultStatus.OK else command_result.summary,
    )


def _sandbox_down_result(label, path):
    """Refuse a mutation while the sandbox is down, in the router's shape.

    Ported from the monolithic execute_tool this file replaced: without the
    sandbox nothing written can be read back, compiled or run, so an edit made
    from retrieved context alone is a change nobody can check. DENIED and not
    FAILED -- the tool did not break, it declined.
    """
    tool_use(label, path, color=C_ERR)
    tool_result(SANDBOX_DOWN_MSG)
    print()

    return ToolHandlerResult(
        text=SANDBOX_DOWN_MSG,
        status=ToolResultStatus.DENIED,
        error_category=ToolResultStatus.DENIED.value,
        error_summary="sandbox unavailable",
    )


def _registered_edit_file(context, args):
    path = (args.get("path") or "").strip()

    if context.cache.get(SANDBOX_DOWN):
        return _sandbox_down_result("Update", path)

    old_text = args.get("old_text") or ""
    new_text = args.get("new_text") or ""
    diff = next((text for text in (new_text, args.get("content") or "")
                 if looks_like_diff(text)), "")

    refused = safe_mode_refusal("edit_file")

    if refused:
        audit_denied_mutation("edit_file", "safe mode", paths=(path,))
        tool_result(refused)
        print()

        return _classified_handler_result(refused)

    print()
    tool_use("Update", path, color=C_ACCENT)

    if diff and not old_text:
        render_diff(diff)

        try:
            fpath = resolve_path(path)
        except PathPolicyError as exc:
            result = f"ERROR: {exc}"
            audit_rejected_mutation("apply_patch", str(exc), paths=(path,))
        else:
            blocked = authorize_mutation(
                f"Apply patch to {C_BOLD}{path}{C_RST} ?",
                action="apply_patch", paths=(fpath,),
            )

            if blocked is None:
                _capture_checkpoint_path(context, fpath)
                result = apply_unified_diff(fpath, diff)

                if result.startswith("OK"):
                    _record_mutation(context, fpath)

                audit_mutation_result("apply_patch", result, paths=(fpath,))
            else:
                result = blocked.to_legacy_text()
    else:
        show_diff(old_text, new_text)
        result = edit_file(path, old_text, new_text, execution_context=context)

    tool_result(result)
    print()
    affected = (_workspace_label(resolve_path(path)),) if result.startswith("OK") else ()

    return _classified_handler_result(result, mutation=True, affected_paths=affected)


def _registered_append_file(context, args):
    path = (args.get("path") or "").strip()

    if context.cache.get(SANDBOX_DOWN):
        return _sandbox_down_result("Append", path)

    content = (args.get("content") or "").strip("\n")

    refused = safe_mode_refusal("append_file")

    if refused:
        audit_denied_mutation("append_file", "safe mode", paths=(path,))
        tool_result(refused)
        print()

        return _classified_handler_result(refused)

    print()
    tool_use("Append", path, color=C_ACCENT)

    for index, line in enumerate(content.split("\n")[:8]):
        prefix = "⎿  " if index == 0 else "   "
        print(f"  {C_DIM}{prefix}{C_RST}{C_OK}+ {line[:150]}{C_RST}")

    try:
        resolve_path(path)
    except PathPolicyError as exc:
        result = f"ERROR: {exc}"
        audit_rejected_mutation("append_file", str(exc), paths=(path,))
        tool_result(result)
        print()

        return _classified_handler_result(result)

    fpath = find_file(path)

    if not fpath:
        result = f"ERROR: file not found: {path}"
    elif (blocked := authorize_mutation(
            f"Append to {C_BOLD}{path}{C_RST} ?", action="append_file",
            paths=(fpath,))) is None:
        _capture_checkpoint_path(context, fpath)

        with open(fpath, "a", encoding="utf-8") as stream:
            stream.write("\n" + content + "\n")

        result = f"OK: appended to {path}"
        _record_mutation(context, fpath)
        audit_mutation_result("append_file", result, paths=(fpath,))
    else:
        result = blocked.to_legacy_text()

    tool_result(result)
    print()
    affected = (_workspace_label(fpath),) if fpath and result.startswith("OK") else ()

    return _classified_handler_result(result, mutation=True, affected_paths=affected)


def _registered_delete_file(context, args):
    """Remove a file, having first captured it so /undo can bring it back.

    Deleting was the one file operation nothing could do. A model that had
    made a chapter redundant tried rm, `bash -c rm`, python3 -c os.remove and
    an empty write_file -- fifteen calls -- then emptied the file with
    edit_file, leaving a zero-byte document that Sphinx still complained
    about. The gap was real; this closes it, under the same checkpoint every
    other mutation gets.
    """
    path = (args.get("path") or "").strip()

    if context.cache.get(SANDBOX_DOWN):
        return _sandbox_down_result("Delete", path)

    reason = (args.get("reason") or "").strip()
    print()
    tool_use("Delete", path, color=C_ERR)

    if reason:
        print(f"  {C_DIM}⎿  {reason[:150]}{C_RST}")

    try:
        fpath = resolve_path(path)
    except PathPolicyError as exc:
        result = f"ERROR: {exc}"
        audit_rejected_mutation("delete_file", str(exc), paths=(path,))
        tool_result(result)
        print()

        return _classified_handler_result(result)

    if not os.path.isfile(fpath):
        result = (f"ERROR: {path} is not an existing regular file. Nothing "
                  f"was deleted." if not os.path.isdir(fpath) else
                  f"ERROR: {path} is a directory. delete_file removes one "
                  f"file at a time.")
        tool_result(result)
        print()

        return _classified_handler_result(result)

    if is_excluded_path(fpath):
        result = (f"ERROR: {path} is in a third-party or snapshot tree that is "
                  f"never modified from here.")
        audit_rejected_mutation("delete_file", "excluded tree", paths=(fpath,))
        tool_result(result)
        print()

        return _classified_handler_result(result)

    blocked = authorize_mutation(
        f"Delete {C_BOLD}{path}{C_RST} ?", action="delete_file", paths=(fpath,))

    if blocked is None:
        # Capture BEFORE unlinking: the checkpoint holds the only remaining
        # copy of the bytes, and it is what /undo restores from.

        _capture_checkpoint_path(context, fpath)

        try:
            os.remove(fpath)
            result = f"OK: deleted {path}"
            _record_mutation(context, fpath)
        except OSError as exc:
            result = f"ERROR: could not delete {path}: {exc.strerror}"

        audit_mutation_result("delete_file", result, paths=(fpath,))
    else:
        result = blocked.to_legacy_text()

    tool_result(result)
    print()
    affected = (_workspace_label(fpath),) if result.startswith("OK") else ()

    return _classified_handler_result(result, mutation=True, affected_paths=affected)


def _registered_write_file(context, args):
    path = (args.get("path") or "").strip()

    if context.cache.get(SANDBOX_DOWN):
        return _sandbox_down_result("Write", path)

    content = (args.get("content") or "").strip("\n")

    if not path:
        result = ("ERROR: no path given. Pass the file to write, e.g. "
                  "write_file(path='so3/usr/src/ping.c', content=...).")
        tool_result(result)
        print()

        return _classified_handler_result(result)

    try:
        fpath = resolve_path(path)
    except PathPolicyError as exc:
        result = f"ERROR: {exc}"
        audit_rejected_mutation("write_file", str(exc), paths=(path,))
        tool_result(result)
        print()

        return _classified_handler_result(result)

    if os.path.isdir(fpath):
        result = f"ERROR: '{path}' is a directory, not a file. Give the full file path."
        tool_result(result)
        print()

        return _classified_handler_result(result)

    # The mode, before any check on the CONTENT. A malformed or escaping
    # path is still reported as itself above -- that is a fact about the
    # request, not about permission -- but "this file is larger than your
    # proposed content, use edit_file" answers a question no one in a
    # read-only session can reach, and names a tool that is equally refused.

    refused = safe_mode_refusal("write_file")

    if refused:
        audit_denied_mutation("write_file", "safe mode", paths=(fpath,))
        tool_result(refused)
        print()

        return _classified_handler_result(refused)

    if looks_like_diff(content) and os.path.isfile(fpath):
        print()
        tool_use("Update", path, color=C_ACCENT)
        render_diff(content)
        blocked = authorize_mutation(
            f"Apply patch to {C_BOLD}{path}{C_RST} ?",
            action="apply_patch", paths=(fpath,),
        )

        if blocked is None:
            _capture_checkpoint_path(context, fpath)
            result = apply_unified_diff(fpath, content)

            if result.startswith("OK"):
                _record_mutation(context, fpath)

            audit_mutation_result("apply_patch", result, paths=(fpath,))
        else:
            result = blocked.to_legacy_text()

        tool_result(result)
        print()
        affected = (_workspace_label(fpath),) if result.startswith("OK") else ()

        return _classified_handler_result(result, mutation=True, affected_paths=affected)

    exists = os.path.isfile(fpath)

    if exists:
        try:
            with open(fpath, "r", encoding="utf-8") as stream:
                content = preserve_third_party_header(stream.read(), content)
        except OSError:
            pass

    print()
    tool_use("Write", path, color=C_ACCENT)
    lines = content.count("\n") + 1

    for index, line in enumerate(content.split("\n")[:8]):
        prefix = "⎿  " if index == 0 else "   "
        print(f"  {C_DIM}{prefix}{C_RST}{C_OK}+ {line[:150]}{C_RST}")

    if lines > 8:
        print(f"  {C_DIM}   … +{lines - 8} lines{C_RST}")

    verb = "Overwrite" if exists else "Create"

    if exists and not _was_read_this_turn(context.cache, fpath):
        result = (f"ERROR: {path} already exists and you have not read it this "
                  f"turn. Overwrite refused. Read it first (bash: cat {path}), "
                  "or use edit_file for a targeted change — that is preferable "
                  "for an existing file.")
        audit_rejected_mutation(
            "write_file", "file not read this turn", paths=(fpath,))
        tool_result(result)
        print()

        return _classified_handler_result(result)

    if exists and os.path.getsize(fpath) > 2 * len(content):
        # An empty write is a delete in disguise, and it does not work: a model
        # that could not remove a redundant chapter emptied it instead, leaving
        # a zero-byte file that still warned "isn't included in any toctree".

        result = (f"ERROR: {path} exists and is much larger than your proposed "
                  f"content ({os.path.getsize(fpath)} bytes vs {len(content)}). "
                  "Overwrite refused — use edit_file for targeted changes or "
                  "append_file for additions."
                  + (" To DELETE it: no tool can, and emptying it leaves the "
                     "file behind. Say which file should go and why, and leave "
                     "it to the operator." if not content.strip() else ""))
        tool_result(result)
        print()

        return _classified_handler_result(result)

    blocked = authorize_mutation(
        f"{verb} {C_BOLD}{path}{C_RST} ?", action="write_file", paths=(fpath,))

    if blocked is None:
        try:
            _capture_checkpoint_path(context, fpath)
            os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)

            with open(fpath, "w", encoding="utf-8") as stream:
                stream.write(content + "\n")

            result = f"OK: {path} written ({lines} lines)"
            _record_mutation(context, fpath)
        except OSError as exc:
            result = f"ERROR: could not write {path}: {exc}"

        audit_mutation_result("write_file", result, paths=(fpath,))
    else:
        result = blocked.to_legacy_text()

    tool_result(result)
    print()
    affected = (_workspace_label(fpath),) if result.startswith("OK") else ()
    metadata = {"created_paths": affected} if not exists and affected else {}

    return _classified_handler_result(
        result, mutation=True, affected_paths=affected, metadata=metadata,
    )


def _registered_remember(context, args):
    note = (args.get("note") or "").strip()
    print()
    tool_use("Remember", note, color=C_ACCENT)

    if not note:
        result = "ERROR: empty note"
    elif (blocked := authorize_mutation(
            "Save this to long-term memory?", action="remember")) is None:
        count = save_memory(note, source=MemorySource.AUTHORIZED_TOOL)
        result = f"OK: saved to memory ({count} memories total)"
        audit_mutation_result("remember", result)
    else:
        result = blocked.to_legacy_text()

    tool_result(result)
    print()

    return _classified_handler_result(result, mutation=True)


def _registered_save_skill(context, args):
    name = (args.get("name") or "").strip()
    content = (args.get("content") or "")[:2500]

    # One line saying what the procedure is for, kept out of the body and
    # embedded with it: a query resembles the purpose more than the steps.

    description = (args.get("description") or "").strip().replace("\n", " ")[:200]
    print()
    tool_use("Skill", name, color=C_ACCENT)

    for index, line in enumerate(content.split("\n")[:6]):
        prefix = "⎿  " if index == 0 else "   "
        print(f"  {C_DIM}{prefix}{line[:150]}{C_RST}")

    blocked = authorize_mutation(
        f"Save skill {C_BOLD}{name}{C_RST} to the library?", action="save_skill")

    if blocked is None:
        result = skill_save(name, content, description=description)
        audit_mutation_result("save_skill", result)
    else:
        result = blocked.to_legacy_text()

    tool_result(result)
    print()

    return _classified_handler_result(result, mutation=True)


def _registered_search_history(context, args):
    query = (args.get("query") or "").strip()
    print()
    tool_use("History", query, color=C_TOOL)
    result = archive_search(query)
    tool_result(result, max_lines=6)
    print()

    return _classified_handler_result(result)


#: Where the turn's lifecycle sits inside the per-call cache. A sentinel, for
#: the same reason the repeat ledger uses one: the cache's string keys are
#: cleared when a write makes the turn's reads stale, and the phase a turn has
#: reached is not made stale by writing.
WORK_PHASE = object()


def _write_gate_closed(context):
    """Is the turn's lifecycle still holding the writes back?

    False whenever there is no gate at all, which is every turn that is not
    changing code to satisfy an authoritative source.
    """
    gate = getattr(context, "write_gate", None)

    if gate is None:
        return False

    # A shell command names no single file the gate could check, so it asks
    # the general question: is there any planned change ready to be made?
    decision = gate("")

    return decision is not None and not getattr(decision, "allowed", True)


def _registered_plan_change(context, args):
    """Record one planned change, and say plainly what was wrong with it.

    The ledger judges the entry against what the turn actually retrieved and
    read; that verdict goes straight back as the tool result. An item rejected
    in silence is an item the model will submit again in the same words.
    """
    ledger = context.cache.get(WORK_PHASE)
    item = {name: args.get(name) for name in work_phase.GapItem.FIELDS}
    item["disposition"] = args.get("disposition")

    print()
    tool_use("Plan", str(item.get("requirement") or "")[:60], color=C_TOOL)

    if ledger is None or not getattr(ledger, "engaged", False):
        # No lifecycle on this turn: nothing gates the writes, so recording a
        # plan changes nothing. Say so rather than pretending it was filed.
        tool_result("(no investigation gate on this turn)")
        print()

        return _classified_handler_result(
            "This turn has no investigation gate — the plan was not recorded "
            "and nothing was waiting for it. Proceed with the change.")

    outcome = ledger.record_plan(
        [item],
        supersedes=str(args.get("supersedes") or "").strip(),
        reason=str(args.get("reason") or "").strip(),
        new_evidence=str(args.get("new_evidence") or "").strip())

    outstanding = ledger.uncovered_requirements()

    if outcome.any_accepted and not outstanding:
        lines = [f"Recorded. {len(ledger.items)} planned change(s) now stand.",
                 "The write gate is open. Make this change, then run the "
                 "project's own build and tests."]
    elif outcome.any_accepted:
        lines = [f"Recorded. {len(ledger.items)} planned change(s) now stand.",
                 f"{len(outstanding)} carried requirement(s) still have no "
                 f"disposition, so the write gate stays shut: "
                 + ", ".join(found.key for found in outstanding[:8])]
    else:
        lines = ["Not recorded.", outcome.report(), work_phase.WRITE_BLOCKED]

    text = "\n".join(line for line in lines if line)
    tool_result(text.splitlines()[0])
    print()

    return _classified_handler_result(text)


def _registered_search_corpus(context, args):
    query = (args.get("query") or "").strip()
    print()
    tool_use("Search", query, color=C_TOOL)

    # Cached for the turn, exactly as bash is. The corpus does not change
    # mid-turn, so the same query gives the same passages; a session was
    # observed spending a round asking the same question twice and getting the
    # same file back. Saying "already searched" is what stops the third.

    key = ("search_corpus", query)

    if key in context.cache:
        tool_result("(cached — already searched this turn)")
        print()

        return _classified_handler_result(
            "(ALREADY SEARCHED this turn — same result repeated below. Use it, "
            "or search something else.)\n" + context.cache[key])

    with Spinner("Searching the corpus…"):
        result = search_corpus(query)

    context.cache[key] = result
    tool_result(result.split("\n")[0][:150] if result else "(no match)")
    print()

    return _classified_handler_result(result)


def _offline_refusal(tool):
    return (f"ERROR {tool}: the session was launched with --no-network. "
            f"Nothing here reaches the internet — say so instead of looking "
            f"for another route.")


def _registered_search_internet(context, args):
    if not NETWORK_ENABLED:
        return _classified_handler_result(_offline_refusal("search_internet"))

    query = (args.get("query") or "").strip()
    print()
    tool_use("Web", query, color=C_TOOL)

    with Spinner("Searching the web…"):
        result = web_search(query)

    show_web_sources(result)
    print()

    return _classified_handler_result(result)


def _fetch_url_to_disk(context, url, save_as):
    """Put the document on disk. A download, not a read.

    Reading a specification into the context window is not the same act as
    having the file: asked for the RS274/NGC PDF so it could be ingested as a
    standard, the assistant fetched it, was handed 60 of 121 pages of TEXT, and
    finished by telling the user to download it in a browser — because nothing
    it could call wrote a file. /standard ingest needs a path.

    Writing is a mutation, so it goes through the same authorization as
    write_file: refused in safe mode, confirmed in ask mode, and audited.
    """
    try:
        fpath = resolve_path(save_as)
    except PathPolicyError as exc:
        audit_rejected_mutation("fetch_url", str(exc), paths=(save_as,))

        return f"ERROR fetch_url: {exc}"

    if os.path.isdir(fpath):
        return f"ERROR fetch_url: '{save_as}' is a directory."

    with Spinner("Downloading…"):
        data, content_type, final_url = web_fetch.download(url)

    blocked = authorize_mutation(
        f"Save {C_BOLD}{final_url}{C_RST} ({len(data) // 1024} KB, "
        f"{content_type}) to {C_BOLD}{save_as}{C_RST} ?",
        action="fetch_url", paths=(fpath,),
    )

    if blocked is not None:
        return blocked.to_legacy_text()

    _capture_checkpoint_path(context, fpath)
    os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)

    with open(fpath, "wb") as handle:
        handle.write(data)

    _record_mutation(context, fpath)
    digest = hashlib.sha256(data).hexdigest()
    result = (f"OK: saved {len(data)} bytes to {save_as}\n"
              f"  {content_type} from {final_url}\n"
              f"  sha256 {digest}")
    audit_mutation_result("fetch_url", result, paths=(fpath,))

    return result


def standard_url_refusal(url):
    """Refuse a fetch that stands in for the bound standard, or "".

    Matched on the standard's own identity terms, the same ones that decide
    the binding -- never a list of hosts. A page is a substitute for the
    source when it NAMES it: a vendor's doc portal serving the standard, or
    a local ref-<standard>.md, both do -- and neither is the corpus the
    binding pins by sha256.

    A URL is not prose, so the word boundaries the scope test relies on are
    not there: "vita492" carries no separator and no dot. Both sides are
    stripped to letters and digits and matched as substrings, which is only
    safe for a term distinctive on its own -- so the short ones are dropped.
    "ansi", "vita", "2017" and "492" would each match half the web; what
    survives is "vita492", "ansivita492", "2017r2024".
    """
    if not url or not STANDARD_ENGAGED_BEFORE:
        return ""

    try:
        binding = STANDARD_OPERATOR.active_binding()
    except StandardCommandError:
        return ""

    if binding is None:
        return ""

    flat = re.sub(r"[^a-z0-9]", "", url.lower())
    terms = {value for value in
             (re.sub(r"[^a-z0-9]", "", term)
              for term in standard_scope.identity_terms(binding))
             if len(value) >= 6}

    if not any(term in flat for term in terms):
        return ""

    return (f"ERROR fetch_url: this session is bound to "
            f"{binding.standard_id} {binding.revision}, whose canonical "
            f"source is pinned by sha256 in the binding. A web page naming "
            f"that standard is not it. Use standard.search / standard.fetch, "
            f"which serve the bound corpus and cite it.")


def _registered_fetch_url(context, args):
    if not NETWORK_ENABLED:
        return _classified_handler_result(_offline_refusal("fetch_url"))

    url = (args.get("url") or "").strip()
    save_as = (args.get("save_as") or "").strip()
    pages = (args.get("pages") or "").strip()

    # The bound standard has one canonical source, with a sha256 the binding
    # names, and four tools that serve it. A turn that fetched the web for it
    # instead read whatever a public page happened to say and edited five
    # source files against that -- so a URL naming the bound standard is
    # refused here, with the tools that do have it.

    refusal = standard_url_refusal(url)

    if refusal:
        tool_use("Web", url, color=C_TOOL)
        tool_result(refusal, max_lines=4)
        print()

        return _classified_handler_result(refusal)

    print()
    tool_use("Web", f"{url} → {save_as}" if save_as else url, color=C_TOOL)
    mutation, affected = False, ()

    try:
        if save_as:
            result = _fetch_url_to_disk(context, url, save_as)
            mutation = result.startswith("OK:")
            affected = ((_workspace_label(resolve_path(save_as)),) if mutation
                        else ())
        else:
            with Spinner("Reading the page…"):
                result = web_fetch.fetch(url, pages=pages or None).render()
    except web_fetch.FetchRefused as exc:
        # A refusal the model can act on: it says which boundary was hit, and
        # every one of them names what to do instead. Returned as a result and
        # not raised, so the turn continues.

        result = f"ERROR fetch_url: {exc}"

    tool_result(result, max_lines=4)
    print()

    return _classified_handler_result(result, mutation=mutation,
                                      affected_paths=affected)


def build_tool_registry():
    registry = ToolRegistry()
    handlers = {
        "edit_file": _registered_edit_file,
        "write_file": _registered_write_file,
        "append_file": _registered_append_file,
        "delete_file": _registered_delete_file,
        "remember": _registered_remember,
        "plan_change": _registered_plan_change,
        "search_corpus": _registered_search_corpus,
        "search_internet": _registered_search_internet,
        "fetch_url": _registered_fetch_url,
    }

    for spec in native_tool_specs():
        registry.register(spec, handlers.get(spec.name))

    hidden = (
        (ToolSpec(
            "save_skill", "Save a reusable skill.",
            {"type": "object", "properties": {
                "name": {"type": "string"}, "content": {"type": "string"},
                "description": {"type": "string"}},
             "required": ["name", "content"]},
            ToolCategory.MEMORY, ToolMutability.MUTATING,
            # Declared, like every other mutating tool. It was the one that
            # had not been, and it is what the central gate is for: until
            # the modes reached the router, a missing declaration cost
            # nothing and so went unnoticed. Nothing changes for it today --
            # its own authorization already refuses in safe mode -- but the
            # refusal is now the mode's, stated before the handler runs.
            required_capabilities=("persistent_memory_write",),
            execution_modes=("ask", "auto"),
            model_visible=False, handler_key="save_skill",
        ), _registered_save_skill),
        (ToolSpec(
            "search_history", "Search conversation history.",
            {"type": "object", "properties": {"query": {"type": "string"}},
             "required": ["query"]}, ToolCategory.RETRIEVAL,
            ToolMutability.READ_ONLY, model_visible=False,
            handler_key="search_history",
        ), _registered_search_history),
        (ToolSpec(
            "web_search", "Compatibility alias for internet search.",
            {"type": "object", "properties": {"query": {"type": "string"}},
             "required": ["query"]}, ToolCategory.WEB,
            ToolMutability.READ_ONLY, model_visible=False,
            handler_key="web_search",
        ), _registered_search_internet),
    )

    for spec, handler in hidden:
        registry.register(spec, handler)

    STANDARD_TOOL_SERVICE.register(registry)

    return registry


TOOL_REGISTRY = build_tool_registry()
TOOL_ROUTER = ToolRouter(TOOL_REGISTRY)

# --no-network is a promise about the SESSION, not about the shell alone.
# search_internet and fetch_url reach the internet from this very process,
# outside the capability policy entirely, so a flag that only emptied
# `network` out of the execution modes would have left the two tools that
# actually browse working — the flag would have read as offline and not been.
NETWORK_ENABLED = "--no-network" not in sys.argv[1:]

_UNBOUND_TOOL_NAMES = frozenset(
    spec.name for spec in TOOL_REGISTRY.list_specs(model_visible=True)
    if not spec.name.startswith("standard.")
    and (NETWORK_ENABLED or spec.category != ToolCategory.WEB))
CANONICAL_TOOLS = TOOL_REGISTRY.definitions_for_model(names=_UNBOUND_TOOL_NAMES)
TOOLS = TOOL_REGISTRY.openai_definitions_for_model(names=_UNBOUND_TOOL_NAMES)


#: Counted by CATEGORY, not by signature: `sed -i`, a redirection, a python
#: one-liner and the next idea are the same violation of the same scope, and
#: counting them separately is how a turn gets four tries at it.

READ_ONLY_VIOLATIONS = object()
READ_ONLY_REFUSAL = (
    "The user explicitly requested a read-only task. Do not try another way "
    "to modify or test the file. Answer the original question using the "
    "evidence already collected."
)


def _evidence_in_context(agent_context, tool_call_id):
    """Is the result of that tool call still in the conversation to be sent?

    Compaction rewrites the conversation and the composer decides what goes
    into each request, so an observation the harness once delivered may no
    longer be anywhere the model can read. Telling it "the output is already
    above" in that state is false, and the only move it leaves is another
    spelling of the same read.
    """

    conversation = getattr(agent_context, "conversation", None) or ()

    for message in conversation:
        for block in getattr(message, "content", ()) or ():
            if (isinstance(block, ToolResultBlock)
                    and block.tool_call_id == tool_call_id
                    and block.content):
                return True

    return False


def route_tool_envelope(
    name, args, cache, *, task_id=None, trace=None, tool_call_id=None,
    cancellation=None, agent_context=None,
):
    # The turn's lifecycle, when it governs this turn. The handler that
    # records a plan reaches it through the cache, and the router reaches the
    # decision it makes through `write_gate` -- one object, two doors, and no
    # way for the two to disagree about the same turn.

    phase_ledger = getattr(agent_context, "work_phase", None)

    if phase_ledger is not None:
        cache[WORK_PHASE] = phase_ledger

    execution_context = ToolExecutionContext(
        task_id=task_id or new_task_id(),
        trace=trace or TRACE,
        cache=cache,
        result_store=RESULT_STORE,
        cancellation=cancellation or NEVER_CANCELLED,
        checkpoint_manager=(agent_context.checkpoint_manager
                            if agent_context is not None else None),
        checkpoint=(agent_context.checkpoint if agent_context is not None else None),
        role=(getattr(agent_context, "role", "main")
              if agent_context is not None else "main"),
        read_only=bool(getattr(agent_context, "read_only", False)),
        # The session's mode, carried to the one place that can hold every
        # tool to its own declaration. Stated here and nowhere else: the
        # mode is this module's, and the router's job is to enforce what the
        # registry declares, not to discover what mode it is in.
        execution_mode=str(EXECUTION_MODE),
        evidence_available=(
            (lambda marker: _evidence_in_context(agent_context, marker))
            if agent_context is not None else None),
        # Asked at the moment of the call, never sampled: the answer changes
        # within the turn, as the standard is read, the sources are opened and
        # a plan is accepted.
        write_gate=(phase_ledger.may_write
                    if phase_ledger is not None and phase_ledger.engaged
                    else None),
        metadata={"standard_binding": getattr(agent_context, "standard_binding", None)},
    )

    execution_context.command_executor = lambda command: _registered_command(
        execution_context, command,
    )
    envelope = TOOL_ROUTER.execute(
        execution_context, tool_call_id or new_action_id("tool_call"),
        name, args,
    )

    # A mutation the router refused never reaches the handler, so the handler
    # never files it. The mutation trail is this module's, not the routing
    # layer's, and a denial that leaves no entry in it is a denial nobody
    # reviewing the session can see -- which is what the trail is for.

    if envelope.error_category in _ROUTER_MUTATION_DENIALS:
        audit_denied_mutation(
            name, envelope.error_category.replace("_", " "),
            paths=tuple(str(args.get(key)) for key in ("path", "file")
                        if args.get(key)))

    # ...and say so on screen. Nothing else will: the handler never ran.

    announce_router_refusal(name, envelope)

    if agent_context is not None and envelope.success:
        source_ids = envelope.metadata.get("standard_source_ids", ())

        if (source_ids and isinstance(source_ids, (list, tuple))
                and hasattr(agent_context, "standard_source_ids_used")):
            agent_context.standard_source_ids_used.update(
                item for item in source_ids if isinstance(item, str))

        event_type = {
            "standard_search_completed": SessionEventType.STANDARD_SEARCH_COMPLETED,
            "standard_source_fetched": SessionEventType.STANDARD_SOURCE_FETCHED,
        }.get(envelope.metadata.get("standard_event"))

        if event_type is not None and agent_context.session is not None:
            payload = {
                key: value for key, value in envelope.metadata.items()
                if key in {"standard_binding", "query_sha256", "result_count",
                           "standard_source_ids", "source_id", "citation"}
            }
            payload["result_reference"] = envelope.result_reference

            # Journal the grounded metadata now; the runtime's canonical tool
            # completion boundary will atomically advance the snapshot. Avoid
            # clearing its in-flight marker from inside the handler path.

            agent_context.session.append(event_type, agent_context.task_id, payload)

    return envelope


def execute_tool(name, args, cache, *, agent_context=None):
    """Compatibility facade returning full text; production uses the router."""
    envelope = route_tool_envelope(
        name, args, cache,
        task_id=agent_context.task_id if agent_context is not None else None,
        trace=agent_context.trace if agent_context is not None else None,
        agent_context=agent_context,
    )

    if agent_context is not None:
        AgentRuntime._record_tool_result(agent_context, envelope)

    return envelope.text


# ── explicit !commands ───────────────────────────────────────────────

def handle_bang_command(cmd_line):
    """Handle !read, !ls, !grep, !edit, !run commands."""
    parts = cmd_line.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("!read", "!cat"):
        if not arg:
            print(f"{C_ERR}Usage: !read <path>{C_RST}")
            return None

        tool_use("Read", arg)
        content = read_file(arg)
        print(content[:5000])

        if len(content) > 5000:
            print(f"{C_DIM}... ({len(content)} chars total){C_RST}")

        return content

    if cmd in ("!ls", "!dir"):
        path = arg or "."
        tool_use("List", path)
        result = run_cmd(f"ls -la {path}", need_confirm=False)
        print(result)

        return result

    if cmd == "!web":
        if not arg:
            print(f"{C_ERR}Usage: !web <query>{C_RST}")
            return None

        tool_use("Web", arg, color=C_TOOL)
        result = web_search(arg)
        print(result[:4000])

        return f"[résultats web pour: {arg}]\n{result}"

    if cmd == "!grep":
        if not arg:
            print(f"{C_ERR}Usage: !grep <pattern> [path]{C_RST}")
            return None

        parts2 = arg.split(None, 1)
        pattern = parts2[0]
        where = parts2[1] if len(parts2) > 1 else "build/"
        grep_cmd = f"grep -rn --max-count=40 '{pattern}' {where} --include='*.bb' --include='*.bbclass' --include='*.bbappend' --include='*.inc' --include='*.conf'"
        tool_use("Grep", grep_cmd)
        result = run_cmd(grep_cmd, need_confirm=False)
        print(result)

        return result

    if cmd == "!find":
        if not arg:
            print(f"{C_ERR}Usage: !find <pattern>{C_RST}")
            return None

        find_cmd = f"find build/ -name '{arg}' -not -path '*/tmp/*'"
        tool_use("Find", find_cmd)
        result = run_cmd(find_cmd, need_confirm=False)
        print(result)

        return result

    if cmd == "!edit":
        print(f"{C_WARN}Interactive usage:{C_RST}")

        try:
            path = input("  File: ").strip()
            old = input("  Text to replace: ").strip()
            new = input("  New text: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        result = edit_file(path, old, new)
        print(result)

        return result

    if cmd in ("!run", "!sh", "!exec"):
        if not arg:
            print(f"{C_ERR}Usage: !run <command>{C_RST}")
            return None

        result = run_cmd(arg, need_confirm=False)
        print(result)

        return result

    print(f"{C_ERR}Unknown command: {cmd}{C_RST}")
    print(f"Available: !read !ls !grep !find !edit !run !web")

    return None


def memory_store():
    """Current corpus' Markdown-authoritative structured memory view."""
    return MarkdownMemoryStore(MEMORIES_FILE, default_scope=MemoryScope.PROJECT)


def load_memories(query=""):
    """Memories noted by the model (remember tool) or the user (/remember).
    Re-read every turn so a note taken mid-session is active immediately.
    Injected into the system prompt — keep an eye on the size budget."""

    try:
        selection = memory_store().select(query, limit=12)
    except MemoryStoreError as exc:
        print(f"{C_WARN}memory metadata ignored: {exc}{C_RST}")

        # A malformed optional sidecar must not make the human-readable source
        # unusable.  Read through a sidecar-free view for this request.

        selection = MarkdownMemoryStore(
            MEMORIES_FILE, default_scope=MemoryScope.PROJECT,
            read_metadata=False,
        ).select(query, limit=12)

    if not selection.records:
        return ""

    return selection.render()


def save_memory(note, *, source=MemorySource.USER):
    """Persist one explicitly authorized memory and return the record count."""
    store = memory_store()
    store.add(note, source=source, provenance="confirmed durable-memory write")

    return store.count()


def save_learned_rule(note):
    """Append one user-taught rule, dated, to the always-injected set."""
    line = note.strip().lstrip("-").strip()
    os.makedirs(os.path.dirname(LEARNED_RULES_FILE) or ".", exist_ok=True)

    with open(LEARNED_RULES_FILE, "a") as handle:
        handle.write(f"- {line}  ({time.strftime('%Y-%m-%d')})\n")


def load_rules():
    """Concatenates the rules from rules.d/*.md (alphabetical order →
    numeric prefixes NN-name.md). Always injected into the system prompt,
    hence guaranteed seen by the model — keep them compact (RULES_BUDGET)."""

    if not os.path.isdir(RULES_DIR):
        return ""

    parts = []

    for fname in sorted(os.listdir(RULES_DIR)):
        if not fname.endswith(".md"):
            continue

        with open(os.path.join(RULES_DIR, fname), "r") as f:
            content = f.read().strip()

        if not content:
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


def announce_carried_spec(question):
    """Say that the previous answer is being used as this turn's specification.

    It is injected into the system rules, which the terminal never shows. A
    frame that decides what a change must satisfy should not be invisible:
    without this line the only way to know it happened was to read the code.
    """
    if not (STANDARD_PRIOR_ANSWER and is_write_request_text(question)):
        return

    carried = (f", {len(STANDARD_PRIOR_REQUIREMENTS)} requirement(s) to close"
               if len(STANDARD_PRIOR_REQUIREMENTS)
               and requirement_set.refers_back(question) else "")
    print(f"  {C_DIM}⎿  carrying forward this session's answer as the "
          f"specification ({len(STANDARD_PRIOR_ANSWER)} chars, "
          f"{len(STANDARD_PRIOR_CLAUSES)} clauses{carried}){C_RST}")


def is_write_request_text(question):
    from agent_runtime import is_write_request

    return is_write_request(question or "")


def standard_binding_for(question):
    """The active binding, but only for a turn that engages it.

    It used to attach to EVERY prompt: the system rule, the standard.* tools
    and the evidence policy came along whatever was asked. Asked to download
    the RS274/NGC specification, the assistant answered "the system is bound to
    ANSI-VITA-49.2, not RS274/NGC" eleven times over — the turn read as
    normative because the question contained the word "standard".

    Binding is announced when it happens. A frame that changes how an answer is
    produced is not something to discover from the answer.
    """
    global STANDARD_ENGAGED_BEFORE, STANDARD_READ_CONTEXT

    try:
        binding = STANDARD_OPERATOR.active_binding()
    except StandardCommandError as exc:
        # A binding the store can no longer honour is an ordinary state after
        # a re-extraction, not a reason to lose the session. Refusing to
        # rebind silently is right -- the corpus underneath moved and that
        # must be visible -- but the refusal belongs in a message with a way
        # out, and it was killing spear-chat with a traceback instead.
        print(f"  {C_DIM}⎿  {STANDARD_OPERATOR.stale_binding_status(exc)}{C_RST}")
        print(f"  {C_DIM}   answering unbound: no normative grounding this "
              f"turn{C_RST}")

        return None

    context, STANDARD_READ_CONTEXT = STANDARD_READ_CONTEXT, ""

    if not standard_scope.engages(binding, question,
                                  engaged_before=STANDARD_ENGAGED_BEFORE,
                                  context=context):
        return None

    # Follow-ups to a bound turn are part of it. "can you validate the
    # implementation ?" named nothing and ran free, then declared the code
    # compliant with a specification the turn was not allowed to open.

    STANDARD_ENGAGED_BEFORE = True

    print(f"  {C_DIM}⎿  bound standard: {binding.standard_id} "
          f"{binding.revision} — normative claims are grounded in it and "
          f"cited{C_RST}")

    return binding.to_dict()


def corpus_property(name, default=None):
    """A declared property of the corpus this session opened.

    Read from the registry entry, falling back to the spec the session was
    launched with (an ad-hoc corpus has no registry entry at all). Behaviour
    that used to be inferred from `kind` is declared here instead: what a
    corpus DOES should be visible in the line that describes it, not in a
    branch somewhere that tests its type.
    """
    spec = load_projects().get(PROJECT) or PROJECT_SPEC or {}

    return spec.get(name, default)


def corpus_indexer():
    """Which indexer rebuilds this corpus: "buildsystem" or "generic".

    The buildsystem walk is curated for BitBake/Yocto trees -- recipe and
    ITS extensions, a scoped source subtree, a hand-maintained skip list --
    and indexes a materially different set of files from the generic one.
    """
    return corpus_property("indexer", "generic")


def corpus_autoindexes():
    """Whether a missing index is built on first sight.

    Off by default: a generic tree can be enormous, and indexing one because
    a session happened to open it is a surprise. A corpus whose curated walk
    is cheap and expected declares it.
    """
    return bool(corpus_property("autoindex", False))


def reindex_options():
    """Indexing options for this project: what projects.json declares, plus
    the corpora that live INSIDE this one.

    The declared exclusions are the deliberate ones, and they belong in the
    registry rather than on the command line: spelled out at the call site, a
    /reindex silently brings back what we took care to leave out. so3 vendors
    lvgl and micropython, which already have their own corpora — unexcluded
    they made up 85% of its index and surfaced instead of the kernel code.

    The derived ones close the
    trap a workspace split leaves behind: registering agency, buildroot, qemu
    and u-boot as four corpora does not stop a /reindex of the tree ABOVE them
    from walking all four again — 60000 files, the file cap, a refusal, and all
    of it for chunks the federation already retrieves from their own indexes. A
    tree that owns a corpus is never part of another one.
    """
    projects = load_projects()
    spec = projects.get(PROJECT) or PROJECT_SPEC or {}
    excludes = list(spec.get("exclude", ()))

    # As a path relative to the corpus root, not a bare name: index_dir skips
    # a bare name wherever it occurs, and a component usually shares its name
    # with a directory the corpus needs — a tree registering its vendored
    # linux/ would lose build/meta-bsp/recipes-bsp/linux with it. A bare name
    # already declared by hand (so3's lvgl, micropython) still counts as the
    # same exclusion, so it is not repeated in the other spelling.

    root = os.path.realpath(CORPUS_ROOT)

    for path in corpora_below(projects, CORPUS_ROOT).values():
        rel = os.path.relpath(path, root)

        if rel in excludes or os.path.basename(path) in excludes:
            continue

        excludes.append(os.path.join(".", rel))

    opts = []

    for d in excludes:
        opts += ["--exclude", d]

    if spec.get("include_build"):
        opts.append("--include-build")

    return opts


def reindex_command():
    """The command that rebuilds THIS corpus's index.

    The tree to index is CORPUS_ROOT, never PROJECT_ROOT. The two differ
    whenever the session runs outside its own corpus, which is the normal case
    right after a workspace split: the corpora sit one level below the
    directory you are standing in. Handed the cwd, /reindex walked the umbrella
    instead of the corpus — four sibling trees, the file cap, a refusal — and
    it derived its destination collection from that same cwd, so even under the
    cap it would have filled adhoc_<md5(cwd)>, which no session reads. The one
    this session queries is COLLECTION_NAME, keyed by the corpus tree; name it
    rather than letting the indexer guess it back.
    """
    curated = corpus_indexer() == "buildsystem"
    script = (f"{APP_DIR}/index_corpus.py" if curated
              else f"{APP_DIR}/index_dir.py")
    cmd = [sys.executable, script, CORPUS_ROOT] + reindex_options()

    # The curated walk derives its own collection name and takes no --collection;
    # the generic one is told which collection this session queries.

    if not curated:
        cmd += ["--collection", COLLECTION_NAME]

    return cmd


# No "/corpus" prefix in it: the same string answers the slash command, where
# the user has already typed it, and the spear-corpus CLI, where it is wrong.
CORPUS_USAGE = ("usage: list | add <name> [path] [--kind K] [--indexer I] "
                "[--autoindex] [--prompt-file F] | rm <name> | "
                "scan [path] [--min N]")

# What a corpus name may hold. Every existing one passes (sye_sol,
# llama.cpp-next, avz.back, opencn-u-boot); a slash or a space does not.
CORPUS_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")


def _save_registry(projects):
    """save_projects(), without taking the session down. Returns an error
    string, or None.

    save_projects raises SystemExit when the registry is read-only — a
    container ships it in the image. That is the right answer for a one-shot
    CLI and the wrong one inside a chat loop, where a mistyped /corpus add
    would take the whole conversation with it.
    """
    try:
        save_projects(projects)
    except SystemExit as exc:
        return str(exc)

    return None


def handle_corpus_command(args, current=None):
    """The corpus registry from inside the session: list / add / rm / scan.

    Same registry as the `spear-corpus` CLI and, now, the same code. That
    script carried its own copy of the load/save, of the seeds and of the
    component walk, against a hardcoded projects.json — so it ignored
    SPEAR_CORPUS_ROOT, and its skip list and extension set had already drifted
    from the indexer's. It is a wrapper around this function instead.

    Registering does NOT move the session: the corpus decides the index, the
    history and the memories, and all three are bound at launch.
    `spear-chat --corpus <name>` is still how you switch. What does take effect
    at once is the exclusion — reindex_options() re-reads the registry — so
    registering a vendored sub-tree and running /reindex right after leaves it
    out of the corpus above it, which is the usual reason to register one.
    """
    args = list(args)
    cmd = args[0] if args else "list"
    flags, rest, opts = [], [], {}
    it = iter(args[1:])

    # These take a VALUE, so they cannot be filtered out by a
    # `startswith("-")` comprehension: the value would fall through to the
    # positionals and `scan --min 500` would read 500 as the directory to
    # scan, or `add c /tree --kind k` would read k as a second path.

    valued = {"--min", "--kind", "--indexer", "--prompt-file"}

    for a in it:
        head, sep, tail = a.partition("=")

        if head in valued:
            opts[head.lstrip("-").replace("-", "_")] = tail if sep else next(it, "")
        elif a.startswith("-"):
            flags.append(a)
        else:
            rest.append(a)

    projects = load_projects()

    def abspath(p):
        return os.path.realpath(os.path.expanduser(p))

    if cmd == "list":
        if not projects:
            return "(no corpus registered)"

        rows = [f"  {'*' if n == current else ' '} {n:18s} {s['kind']:8s} "
                f"{s['path']}" for n, s in sorted(projects.items())]

        return "\n".join(rows) + f"\n  ({len(projects)} corpora"  \
               + (", * = this session)" if current in projects else ")")

    if cmd == "add" and rest:
        name = rest[0]

        # A corpus NAME is not a path. `/corpus add agency/linux`, typed from
        # the tree above it, registered the name "agency/linux" pointing at the
        # CWD — the path defaults to it, and nothing said the first argument
        # had been read as a name. The session then displayed
        # "corpus: agency/linux" while indexing the parent, which is exactly
        # the kind of wrong that looks right.

        if not CORPUS_NAME_RE.fullmatch(name):
            hint = ""

            if os.path.isdir(abspath(name)):
                hint = (f" You gave a path: `add "
                        f"{os.path.basename(name.rstrip('/')) or 'name'} "
                        f"{name}` registers that tree instead.")

            return (f"'{name}' is not a corpus name — a name goes into "
                    f"`--corpus <name>` and into the banner, so it holds "
                    f"letters, digits and . _ + - only.{hint}")

        path = abspath(rest[1]) if len(rest) > 1 else abspath(os.getcwd())

        if not os.path.isdir(path):
            return f"not a directory: {path}"

        known = projects.get(name)

        if known and os.path.realpath(known["path"]) != path:
            return (f"'{name}' already points at {known['path']} — remove it "
                    f"first (/corpus rm {name})")

        # Re-registering the same tree must not drop what was declared about
        # it: exclude, collection and include_build are hand-written and would
        # be silently lost by rebuilding the spec from name and path.

        spec = dict(known or {})
        spec["path"] = path
        spec["kind"] = opts.get("kind") or spec.get("kind", "generic")

        # Each behaviour named separately. One flag that switched four of them
        # at once is how they became invisible in the first place.

        for option, key in (("indexer", "indexer"), ("prompt_file", "prompt_file")):
            if opts.get(option):
                spec[key] = opts[option]

        if "--autoindex" in flags:
            spec["autoindex"] = True
        projects[name] = spec
        err = _save_registry(projects)

        if err:
            return err

        return (f"registered '{name}' ({spec['kind']}) -> {path}\n"
                f"  index it:  spear-chat --corpus {name}   then /reindex\n"
                f"  a tree inside another corpus is excluded from it on the "
                f"next /reindex")

    if cmd == "rm" and rest:
        name = rest[0]

        if name == current:
            return (f"'{name}' is this session's own corpus — its index, "
                    f"history and memories are bound to it; relaunch "
                    f"elsewhere to remove it")

        if projects.pop(name, None) is None:
            return f"no such corpus: {name}"

        err = _save_registry(projects)

        return err or (f"removed '{name}' — its index is left in the store "
                       f"(nothing reads it now)")

    if cmd == "scan":
        root = abspath(rest[0]) if rest else abspath(os.getcwd())

        try:
            mn = int(opts.get("min", 300))
        except ValueError:
            return CORPUS_USAGE

        if not os.path.isdir(root):
            return f"not a directory: {root}"

        comps = workspace_components(root, projects, mn)

        if not comps:
            return f"no component of {root} reaches {mn} indexable files"

        base = os.path.basename(root.rstrip("/"))
        lines = [f"Scanning {root} (components with >= {mn} indexable files):"]
        added = []

        for name, path, n, owner in comps:
            if owner:
                lines.append(f"  {'':>8}  {name:16s} already: {owner}")
                continue

            new_name = name if name not in projects else f"{base}-{name}"
            projects[new_name] = {"path": path, "kind": "generic"}
            added.append(new_name)
            lines.append(f"  {n:>8}  {name:16s} -> corpus '{new_name}'")

        if not added:
            return "\n".join(lines + ["Nothing new to register."])

        err = _save_registry(projects)

        if err:
            return err

        return "\n".join(lines + [f"Registered {len(added)}: "
                                  f"{', '.join(added)}"])

    return CORPUS_USAGE


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

    for root in (PROJECT_ROOT, CORPUS_ROOT):
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

        candidates = [PROJECT]

        if PROJECT.startswith("workspace:"):
            candidates.append(PROJECT.split(":", 1)[1])

        shipped = next(
            (path for path in
             (os.path.join(SHIPPED_CORPUS_RULES, f"{name}.md")
              for name in candidates)
             if os.path.isfile(path)),
            os.path.join(SHIPPED_CORPUS_RULES, f"{PROJECT}.md"))

        if os.path.isfile(shipped):
            with open(shipped, "r") as f:
                content = f.read().strip()

            if content:
                parts.append(f"## Rule: corpus ({PROJECT}, shipped)\n\n"
                             + _to_sandbox_paths(content, CORPUS_ROOT))

    if not parts:
        return ""

    return "\n\n" + "\n\n".join(parts)


# ── main ─────────────────────────────────────────────────────────────

def banner_art():
    """ASCII-art header in the platform gradient.
    Shown first, before the project picker."""
    art = [
        "  ███████╗██████╗ ███████╗ █████╗ ██████╗ ",
        "  ██╔════╝██╔══██╗██╔════╝██╔══██╗██╔══██╗",
        "  ███████╗██████╔╝█████╗  ███████║██████╔╝",
        "  ╚════██║██╔═══╝ ██╔══╝  ██╔══██║██╔══██╗",
        "  ███████║██║     ███████╗██║  ██║██║  ██║",
        "  ╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝",
    ]
    grad = ["112;196;78", "112;196;78", "94;182;66",
            "78;168;56", "62;154;48", "56;150;46"]
    print()

    for line, rgb in zip(art, grad):
        print(f"\033[38;2;{rgb}m{line}{C_RST}")

    print(f"\033[38;2;112;196;78m  HEIG-VD/REDS — \033[1;38;2;56;150;46m"
          f"Specification-driven Platform for Embedded "
          f"Agentic Reasoning{C_RST}\n")


def _corpus_summary(corpora):
    """One line for the banner, whether the session has one corpus or several."""

    if not isinstance(corpora, list):
        corpora = [(corpora, "")]

    if len(corpora) == 1:
        return f"{corpora[0][0].count()} chunks"

    total = sum(c.count() for c, _ in corpora)
    names = ", ".join((p or ".") for _, p in corpora)

    return f"{total} chunks across {len(corpora)} corpora ({names})"


def banner(collection, history, n_rules, model_name, n_mem=0):
    cwd = os.getcwd()
    home = os.path.expanduser("~")

    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]

    rows = [
        f"{C_DIM}model:{C_RST}    {model_name} + RAG",
        f"{C_DIM}corpus:{C_RST}   {PROJECT}  {C_DIM}·{C_RST}  "
        + (_corpus_summary(collection) if collection else "no RAG")
        + f"  {C_DIM}·{C_RST}  {n_rules} rules",
        f"{C_DIM}history:{C_RST}  {len(history)} messages"
        + (f"   {C_DIM}memories:{C_RST} {n_mem}" if n_mem else ""),
        f"{C_DIM}tools in:{C_RST} {cwd}  {C_DIM}(current directory){C_RST}",
    ]

    standard = standard_row()

    if standard is not None:
        rows.insert(2, standard)

    rows.insert(-1, content_row())
    rows.append(permissions_row(EXECUTION_MODE))

    if WORKSPACE is not None and WORKSPACE.extra_roots:
        # Widening the write boundary must never be silent.

        rows.append(f"{C_DIM}writable:{C_RST} cwd + {len(WORKSPACE.extra_roots)} "
                    f"declared corpora {C_DIM}(--single-root to restrict to the "
                    f"cwd){C_RST}")
        rows.append(f"{C_DIM}readable:{C_RST} anywhere else by absolute path, "
                    f"read-only {C_DIM}(credential stores refused){C_RST}")

    # info (not an error): the cwd is outside the corpus tree → RAG context
    # may be about another tree than the one tools act on.

    if not (PROJECT_ROOT == CORPUS_ROOT or PROJECT_ROOT.startswith(CORPUS_ROOT + "/")):
        rows.append(f"{C_DIM}note: cwd is outside the '{PROJECT}' corpus tree "
                    f"({CORPUS_ROOT}){C_RST}")

    for r in rows:
        print(f"  {r}")

    print()
    # /standard and /finetune were dispatched but named nowhere — not on this
    # line, not in --help. "Operator-only" means the MODEL never reaches them;
    # it was never meant to mean the operator has to know they exist.

    print(f"  {C_DIM}!read !ls !grep !find !edit !run !web   "
          f"/search /reindex /corpus /history /skills /remember /recall /forget /good /bad /undo /clear /model /tools\n"
          f"  /standard /finetune{C_RST}\n")


INPUT_HISTORY = f"{STATE_DIR}/.input_history"

# Prompt for input(): ANSI codes wrapped in \001/\002 so readline does not
# count them as visible characters (otherwise line editing misaligns).

PROMPT = "\001\033[1m\002> \001\033[0m\002"


def init_readline():
    """Line editing: arrows, Home/End, ctrl+a/e, persistent input history."""
    readline.set_history_length(1000)

    try:
        readline.read_history_file(INPUT_HISTORY)
    except (FileNotFoundError, PermissionError):
        pass

    atexit.register(readline.write_history_file, INPUT_HISTORY)


ADHOC_PROMPT = (
    "You are HEIG-VD/REDS AI, an expert embedded-software assistant running "
    "in ad-hoc mode: you operate on the user's CURRENT directory (shown "
    "below). Assume nothing about its layout from any other project you "
    "may know -- inspect the actual directory with your tools "
    "(ls, grep -rn, cat) before answering. Your long-term memory (the "
    "'## Memories' section, saved via the remember tool) and conversation "
    "history persist across sessions."
)


def compaction_policy_from_environment():
    """Build the task policy while preserving the legacy fail-safe defaults."""

    try:
        return CompactionPolicy(
            mode=CompactionMode.AUTOMATIC,
            # Measured against what is ALREADY the conservative figure. Three
            # margins were stacked: the harness caps the window at 85%, then
            # the output reserve and the safety margin come off that, and
            # only then was 0.80 applied -- so compaction began at 60% of the
            # real window and, having no hysteresis, ran every round from
            # there. Two of the three margins are enough; the reserve is what
            # guarantees room for the answer, and this only decides when to
            # start making space.
            pressure_threshold=float(os.environ.get(
                "SPEAR_COMPACTION_THRESHOLD", "0.90")),
            recent_tail_groups=int(os.environ.get(
                "SPEAR_COMPACTION_RECENT_GROUPS", "3")),
            minimum_compactable_tokens=int(os.environ.get(
                "SPEAR_COMPACTION_MIN_TOKENS", "128")),
            max_summary_chars=int(os.environ.get(
                "SPEAR_COMPACTION_SUMMARY_CHARS", "2048")),
            max_retries=int(os.environ.get(
                "SPEAR_COMPACTION_RETRIES", "1")),
        )
    except (TypeError, ValueError):
        return CompactionPolicy()


def main():
    global TRACE_PROVIDER, TRACE_MODEL, STANDARD_ENGAGED_BEFORE, STANDARD_READ_CONTEXT
    global STANDARD_PRIOR_CLAUSES, STANDARD_PRIOR_ANSWER
    global STANDARD_PRIOR_REQUIREMENTS

    # Before anything with a side effect: --help must not resolve a corpus,
    # offer a workspace split, or touch the model server.

    if {"--help", "-h"} & set(sys.argv[1:]):
        print_help()
        return

    banner_art()
    set_project(resolve_project_at_startup())
    init_readline()

    # A corpus may declare a domain prompt of its own; otherwise the generic
    # one. This used to be chosen by corpus KIND, which meant a prompt naming
    # one project's directories reached every corpus of that type and the
    # domain knowledge of any other corpus reached nothing. It is a property
    # of the corpus, so the corpus says it.
    #
    # The rules.d files are path-agnostic (copyright headers, coding style,
    # build facts), so they apply either way and are injected below.
    #
    # Normative guidance is NOT here: it is attached to the standard binding,
    # so a bound turn gets it whatever prompt the corpus carries.

    prompt_file = corpus_property("prompt_file")

    if prompt_file:
        path = (prompt_file if os.path.isabs(prompt_file)
                else os.path.join(APP_DIR, prompt_file))
        try:
            with open(path, "r") as f:
                system_instructions = f.read()

            system_source = path
        except OSError as exc:
            print(f"{C_WARN}corpus prompt_file unreadable ({exc}) — "
                  f"using the generic prompt{C_RST}")
            system_instructions = ADHOC_PROMPT
            system_source = "built_in_adhoc_prompt"
    else:
        system_instructions = ADHOC_PROMPT
        system_source = "built_in_adhoc_prompt"

    global_rules = load_rules()
    project_rules = load_corpus_rules()
    base_prompt = system_instructions + global_rules + project_rules + TOOL_GUIDE

    collection = init_chromadb()
    global COLLECTION
    COLLECTION = collection

    try:
        backend, provider = create_model_backend(sys.argv[1:])
    except ModelBackendConfigurationError as exc:
        print(f"{C_ERR}Model backend configuration error: {exc}{C_RST}")
        return

    TRACE_PROVIDER = provider

    history = load_history()
    n_rules = base_prompt.count("\n## Rule: ")

    # Ask the server for the live model id. A freshly-opened SSH tunnel
    # (remote mode) can drop the very first request, so retry briefly before
    # falling back to the configured name — never show a bare "?".

    raw = ""

    for _attempt in range(3):
        try:
            discovered = backend.discover_model_name()
            raw = discovered.split("/")[-1] if discovered else ""

            break
        except Exception:
            time.sleep(0.5)

    if not raw:
        raw = ("claude-sonnet-5" if provider == "anthropic"
               else os.environ.get("SPEAR_MODEL_NAME", "") or "?")

    TRACE_MODEL = getattr(backend, "model", None) or raw
    raw = raw[:-5] if raw.endswith(".gguf") else raw       # strip .gguf
    quant = ""

    for q in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M"):
        if q in raw:
            quant = "Q" + q[1]                              # Q8_0 -> Q8
            raw = re.split(rf"-(?:UD-)?{q}", raw)[0]

            break

    model_name = f"{raw}  [{quant}]" if quant else raw
    model_name += f"  {C_DIM}({backend_label(provider, LLAMA_SERVER_URL)}){C_RST}"

    try:
        n_mem = memory_store().count()
    except MemoryStoreError:
        n_mem = MarkdownMemoryStore(
            MEMORIES_FILE, default_scope=MemoryScope.PROJECT,
            read_metadata=False,
        ).count()

    banner(collection, history, n_rules, model_name, n_mem)

    def read_user_input():
        """input() + RELIABLE multi-line paste capture. A big paste arrives as
        many lines; the old flat 80 ms slurp dropped most of it (you'd see
        "2 lines pasted" for a 50-line paste). Two robust paths:
          A) bracketed-paste markers (ESC[200~ … ESC[201~) → read until the end
             marker, independent of timing (bullet-proof when present);
          B) no markers → a TWO-PHASE timing slurp: 0.05 s after a normal typed
             line (returns fast), but once more lines are pending (a paste) we
             widen the gap to 0.4 s and keep reading until it drains."""
        first = input(PROMPT)

        if not sys.stdin.isatty():
            return first.strip()

        PSTART, PEND = "\x1b[200~", "\x1b[201~"

        if PSTART in first:                          # ── A) bracketed paste
            buf = first.split(PSTART, 1)[1]
            parts = []

            while PEND not in buf:
                parts.append(buf)
                nxt = sys.stdin.readline()

                if not nxt:
                    break

                buf = nxt.rstrip("\n")

            parts.append(buf.split(PEND, 1)[0])
            text = "\n".join(parts).strip()
        else:                                        # ── B) two-phase timing
            # Read the rest with os.read on the raw fd, never with
            # sys.stdin.readline(). The buffered reader pulls EVERYTHING
            # pending into Python's own buffer and returns one line; select
            # then looks at the kernel fd, finds it empty, and the loop exits
            # while the remaining lines sit unreachable in that buffer. A
            # three-line paste arrived as two.

            fl = fcntl.fcntl(0, fcntl.F_GETFL)
            fcntl.fcntl(0, fcntl.F_SETFL, fl | os.O_NONBLOCK)

            try:
                rest, gap = "", 0.05

                while select.select([sys.stdin], [], [], gap)[0]:
                    try:
                        block = os.read(0, 65536)
                    except BlockingIOError:
                        break

                    if not block:
                        break

                    rest += block.decode("utf-8", "replace")
                    gap = 0.4                        # paste detected → tolerate gaps
            finally:
                fcntl.fcntl(0, fcntl.F_SETFL, fl)

            lines = [first] + rest.splitlines()
            text = "\n".join(lines).strip()

        n = text.count("\n") + 1

        if n > 1:
            print(f"{C_DIM}  ({n} lines pasted){C_RST}")

        return text

    while True:
        STATUS.end_turn()

        try:
            user_input = read_user_input()
        except KeyboardInterrupt:
            print(f"\n{C_DIM}(ctrl+d or « exit » to quit){C_RST}")
            continue
        except EOFError:
            print()
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            break

        # The turn's clock starts here and runs until the answer. Every
        # activity that follows reads it rather than starting its own, so
        # the number on screen only ever goes up.
        STATUS.begin_turn()

        # ── !bang commands (direct tool use, no LLM) ──

        if user_input.startswith("!"):
            result = handle_bang_command(user_input)

            # A file read here is in front of the model for the next turn,
            # and may be the only place that turn names the bound standard.

            if result and user_input.split(None, 1)[0] in ("!read", "!cat"):
                STANDARD_READ_CONTEXT = result

            if result:
                # A !read is the operator handing the model a document, and
                # 2000 characters is a paragraph of one. A work order read in
                # for the next turn arrived with its own task sections cut
                # off, so neither the model nor the harness could see past
                # section A -- the file itself is already capped at 15000 by
                # the reader, which is the real bound.

                kept = (16000 if user_input.split(None, 1)[0] in ("!read", "!cat")
                        else 2000)
                entry = f"[tool result]\n{result[:kept]}"
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": entry})
                archive_entry({"role": "user", "content": user_input})
                archive_entry({"role": "assistant", "content": entry})
                save_history(history)

            continue

        # ── /slash commands ──

        if user_input == "/standard" or user_input.startswith("/standard "):
            # Explicit operator branch: ingestion and revision changes never
            # enter ToolRegistry, AgentRuntime, or the model provider.

            try:
                result = handle_standard_command(user_input, STANDARD_OPERATOR)
            except (StandardCommandError, OSError, ValueError) as exc:
                result = f"Standard command error: {exc}"
            finally:
                STANDARD_OPERATOR.progress.finish()

            print("\n" + result + "\n")

            continue

        if user_input == "/finetune" or user_input.startswith("/finetune "):
            # Explicit operator branch: no WorkingState, AgentRuntime, tool
            # dispatch, or ModelBackend call exists on this path.

            from finetune_commands import (FinetuneCommandError,
                                           handle_finetune_command)

            try:
                print("\n" + handle_finetune_command(
                    user_input, operator_training_controller()) + "\n")
            except (FinetuneCommandError, OSError, ValueError) as exc:
                print(f"\nFine-tuning command error: {exc}\n")

            continue

        if user_input.startswith("/search "):
            ctx, files = retrieve_context(collection, user_input[8:], top_k=5)
            print(f"\nFichiers: {', '.join(sorted(files))}\n{ctx[:3000]}\n")

            continue

        if user_input == "/corpus" or user_input.startswith("/corpus "):
            try:
                # shlex, not split(): a corpus path may hold spaces, and
                # `add x "/a b/c"` has to reach the registry as ONE argument.
                # The name is bound here because Python unbinds `exc` at the
                # end of the except clause.
                argv, err = shlex.split(user_input)[1:], None
            except ValueError as exc:          # unbalanced quote
                argv, err = None, f"/corpus: {exc}"

            print("\n" + (err or handle_corpus_command(argv, current=PROJECT))
                  + "\n")

            continue

        if user_input == "/reindex":
            print("Reindexing...")

            # Root, destination collection and exclusions all come from
            # reindex_command(): each of the three was got wrong here once.

            subprocess.run(reindex_command())
            collection = init_chromadb()
            COLLECTION = collection

            if collection:
                print(f"Index: {_corpus_summary(collection)}\n")

            continue

        if user_input == "/history":
            if not history:
                print("(empty)\n")
            else:
                for i, msg in enumerate(history):
                    role = msg["role"]
                    txt = (msg.get("content") or "")[:100]
                    pfx = ">>>" if role == "user" else "   "
                    print(f"  {i:3d} [{role:9s}] {pfx} {txt}")

                print(f"\n  ({len(history)} messages)\n")

            continue

        if user_input in ("/clear", "/new"):
            history = []
            save_history(history)

            # The turn a follow-up would refer back to is gone with it.

            STANDARD_ENGAGED_BEFORE = False
            STANDARD_PRIOR_CLAUSES = ()
            STANDARD_PRIOR_ANSWER = ""
            STANDARD_PRIOR_REQUIREMENTS = requirement_set.RequirementSet()
            print("History cleared.\n")

            continue

        if user_input.startswith("/forget"):
            pat = user_input[len("/forget"):].strip()

            if pat:
                print(archive_forget(pat) + "\n")
            else:
                print("usage: /forget <regex>  (prunes the history search "
                      "index; the archive file is kept)\n")

            continue

        if user_input.split()[0] in ("/model", "/switch"):
            arg = user_input.split(None, 1)
            arg = arg[1].strip() if len(arg) > 1 else ""

            if not arg:
                subprocess.run(["spear-model"])
            else:
                print(f"{C_DIM}Switching model — the server reloads "
                      f"(~30-60s), the next message will wait.{C_RST}")
                subprocess.run(["spear-model", arg])

            continue

        if user_input == "/skills":
            skills = skill_library.load_library(SKILLS_DIR)

            if not skills:
                print("(no skills yet — the model saves them after "
                      "completed tasks, with your confirmation)\n")
            else:
                for skill in skills:
                    missing = skill_library.missing_requirements(skill)
                    scoped = ("" if skill_library.ANY_SCOPE in skill.scope
                              else f" [{','.join(skill.scope)}]")

                    # Why a skill will not be injected here is worth more than
                    # the fact that it exists: an unmet requirement is the
                    # answer to "why did it not use that procedure".

                    if missing:
                        state = f" {C_DIM}needs {', '.join(missing)}{C_RST}"
                    elif not skill_library.applies_to(skill, project=PROJECT,
                                                      kind=PROJECT_KIND):
                        state = f" {C_DIM}other corpus{C_RST}"
                    else:
                        state = ""

                    print(f"  - {skill.name}{scoped}{state}")

                    if skill.summary:
                        print(f"    {C_DIM}{skill.summary[:100]}{C_RST}")

                print(f"  ({len(skills)} skills in {SKILLS_DIR})\n")

            continue

        if user_input.startswith("/remember"):
            note = user_input[len("/remember"):].strip()

            if note:
                n = save_memory(note)
                print(f"Saved ({n} memories in {os.path.basename(MEMORIES_FILE)})\n")
            else:
                if os.path.isfile(MEMORIES_FILE):
                    print(open(MEMORIES_FILE).read())
                else:
                    print("(no memories yet — usage: /remember <note>)\n")

            continue

        if user_input.startswith("/recall"):
            # The global counterpart of /remember. /remember writes to
            # memories-<corpus>.md and is invisible in every other corpus,
            # which is the wrong home for something like "never rewrite an
            # existing copyright header" -- true in SO3, in edgem1 and in
            # pos_sol alike. This lands in the always-injected rules instead.

            note = user_input[len("/recall"):].strip()

            if note:
                save_learned_rule(note)
                size = len(load_rules())
                print(f"Recalled into {os.path.basename(LEARNED_RULES_FILE)} "
                      f"— every corpus, every request.")

                if size > RULES_BUDGET:
                    print(f"{C_WARN}  rules are now {size} chars, injected on "
                          f"every request (soft limit {RULES_BUDGET}). Trim "
                          f"{LEARNED_RULES_FILE} or move a long one into a "
                          f"corpus map.{C_RST}")

                print()
            elif os.path.isfile(LEARNED_RULES_FILE):
                print(open(LEARNED_RULES_FILE).read())
            else:
                print("(nothing recalled yet — usage: /recall <rule>)\n")

            continue

        if user_input in ("/good", "/bad"):
            # quality feedback on the LAST exchange. /good saves it as a
            # fine-tuning sample (experience.jsonl); /bad only archives the
            # verdict (future DPO material) — never used as a positive.

            if (len(history) >= 2 and history[-1]["role"] == "assistant"
                    and history[-2]["role"] == "user"):
                q = history[-2]["content"].split(
                    "\n\n[Tools executed during this turn")[0]
                a = history[-1]["content"]
                archive_entry({"role": "feedback",
                               "verdict": user_input[1:],
                               "question": q[:500]})

                if user_input == "/good":
                    n = save_experience(q, a)
                    print(f"Saved as fine-tuning sample "
                          f"({n} in experience.jsonl)\n")
                else:
                    print("Noted as bad (archived, not used for training)\n")
            else:
                print("No complete exchange to rate.\n")

            continue

        if user_input == "/undo":
            # Removes the last exchange (assistant reply + user question).
            # A bad answer left in history acts as a few-shot example the
            # model then imitates.

            removed = 0

            if history and history[-1]["role"] == "assistant":
                history.pop()
                removed += 1

            if history and history[-1]["role"] == "user":
                history.pop()
                removed += 1

            if removed:
                save_history(history)
                print(f"Last exchange removed ({removed} message(s)). "
                      f"{len(history)} messages left.\n")
            else:
                print("Nothing to undo.\n")

            continue

        if user_input == "/tools":
            print(f"""
{C_BOLD}Direct commands (no LLM):{C_RST}
  !read <path>        Read a file
  !ls [path]          List a directory
  !grep <pat> [path]  Search BitBake files
  !find <pattern>     Find files by name
  !edit               Edit a file (interactive)
  !run <cmd>          Run a shell command
  !web <query>        Search the internet (DuckDuckGo)

{C_BOLD}LLM questions:{C_RST}
  Type in natural language. The system auto-detects when a tool
  is needed (read, grep, ...) and runs it before handing the
  question to the LLM together with the result.

{C_BOLD}Operator fine-tuning control (never sent to the LLM):{C_RST}
  /finetune status
  /finetune doctor [--remote]
  /finetune prepare [sft|kto|dpo]
  /finetune start [sft|kto|dpo] [--force]
  /finetune stop [job-id]
  /finetune logs [job-id] [lines]
  /finetune list
  /finetune inspect <job-id>

{C_BOLD}Operator normative-standard control (never sent to the LLM):{C_RST}
{chr(10).join(standard_help_lines())}
""")
            continue

        # An unrecognised slash command is a typo, not a question. It used to
        # fall through to the model: a stray "/quit" -- spear-chat has no such
        # command -- became a prompt, and the turn it started went on to edit
        # five source files with no binding, no task and no instruction. Bang
        # commands have refused their unknowns since they existed; this is the
        # same refusal, and it names the two places the real list lives.

        if user_input.startswith("//"):
            user_input = user_input[1:]
        elif user_input.startswith("/"):
            print(f"{C_ERR}Unknown command: {user_input.split()[0]}{C_RST}"
                  f"{C_DIM} — /tools lists them, or start the line with // to "
                  f"send it to the model as text.{C_RST}\n")

            continue

        # 30 rounds was the figure that cut a turn one step short of its own
        # last step: the model had written the chapter and still owed the
        # toctree line when the budget ended. A round is one model call with
        # the whole context, so this is paid in latency, not in correctness --
        # and a turn that stops mid-edit costs the user a whole second turn
        # anyway. Doubled, and still an env var for whoever wants it back.

        # Generous on purpose, and here is why the old 60 was the wrong
        # instrument. A round ceiling does not protect against a turn that
        # loops: ProgressMonitor already ends one after four actions without
        # progress, which is a direct measure of non-convergence rather than
        # a proxy for it. What the ceiling actually limited was PRODUCTIVE
        # work -- and it kept ending turns that were converging. A master run
        # needed 143 tool calls to finish a five-file change correctly; the
        # next one died at 125 with its work abandoned. Cost is bounded by
        # the token budgets below, which measure money; rounds do not.
        max_tool_rounds = int(os.environ.get("SPEAR_MAX_TOOL_ROUNDS", "250"))

        # A question is answered from what was read; a change has to be read
        # for, written, built and fixed. Sixty rounds covers the first and
        # not the second: asked to change the code, six runs in a row ended
        # on round_budget_exhausted, several of them mid-edit. The budget
        # follows the kind of request rather than being one number for both,
        # and an explicit SPEAR_MAX_TOOL_ROUNDS still wins.

        if ("SPEAR_MAX_TOOL_ROUNDS" not in os.environ
                and is_write_request_text(user_input)):
            max_tool_rounds *= 2
        max_commands = int(os.environ.get("SPEAR_MAX_COMMANDS", "500"))

        # The same reasoning as the round budget above, and it was half done
        # without this: a writing turn was given twice the rounds and the same
        # number of tool calls, then died on tool_budget_exhausted at 124 of
        # 120 with its rounds unspent. Repairing a build costs CALLS -- read
        # the error, edit, rebuild -- not deliberation.

        if ("SPEAR_MAX_COMMANDS" not in os.environ
                and is_write_request_text(user_input)):
            max_commands *= 2
        working_state = WorkingState.start(
            new_task_id(), user_input,
            max_model_rounds=max_tool_rounds,
            max_tool_actions=max_commands,
        )

        # ── the question names another registered corpus? say so, do nothing ──

        hint = corpus_mention_hint(user_input, load_projects(), PROJECT)

        if hint:
            print(f"{C_DIM}  ⎿  {hint}{C_RST}")

        # ── auto-detect → run tool → inject into LLM ──

        extra, desc = auto_detect_and_run(user_input, collection)

        # ── RAG retrieval (skipped in ad-hoc mode) ──

        skills_ctx = skills_lookup(user_input)

        if skills_ctx:
            print(f"{C_DIM}  ⎿  skill match{C_RST}")

        # tools always run in the cwd (PROJECT_ROOT); relative paths in your
        # bash/edit/write calls resolve from there — tell the model.
        # Say what bash actually SEES, not only where the workspace lives on
        # the host. The sandbox normally binds the tree at its own path, so the
        # two agree and the note can say so; under the legacy /workspace mount
        # it must warn that the host path is absent there, since announcing
        # only the host path once made the model spend a whole session trying
        # `ls /opt/llm/spear` and conclude the tree had vanished.

        mount = sandbox_mount()

        if mount == str(PROJECT_ROOT) or mount == os.path.realpath(PROJECT_ROOT):
            where = (f"Inside bash the sandbox shows that directory at that "
                     f"same path — `pwd` prints `{mount}` — so absolute paths "
                     f"into the tree work as they do on the host. ")
        else:
            where = (f"Inside bash the sandbox shows that same directory as "
                     f"`{mount}` and `pwd` prints `{mount}` — the host path "
                     f"does not exist there, and absolute paths are refused. ")

        cwd_note = (f"\n\n## Working directory\n\n"
                    f"Your tools (bash, edit_file, write_file, ...) act on the "
                    f"CURRENT directory, which is {PROJECT_ROOT} on the host. "
                    f"Always use RELATIVE paths: they resolve from there. "
                    f"{where}"
                    f"Nothing outside this directory is writable, "
                    f"and write_file will not create missing parent "
                    f"directories: create them with bash (mkdir -p) first."
                    f"{extra_roots_note()}"
                    f" Anywhere ELSE on the host is READ-ONLY, not invisible: "
                    f"name it by ABSOLUTE path in bash (cat, sed -n, ls, grep, "
                    f"find) and it is mounted read-only for that command — so "
                    f"`ls /opt/llm/claude` or `cat /etc/os-release` works. "
                    f"Credential stores (.ssh, .aws, .netrc, /etc/shadow, ...) "
                    f"are refused, and the file tools never leave the trees "
                    f"above."
                    f" The Retrieved Context may use paths relative to the "
                    f"corpus root — adapt them, or inspect with bash if "
                    f"unsure.")
        memories_ctx = load_memories(user_input)

        if collection is not None:
            # The embedder is loaded lazily, on the first query of the session:
            # measured here, 10.98 s for that first call against 0.05 s for
            # every one after it, and 0.2-0.3 s for the whole pre-model phase
            # once warm. Ten seconds of silence right after the first Enter
            # reads as a hung terminal, and a static line printed before the
            # wait does not answer "is it still alive?" -- the spinner's live
            # counter does, and it is already what every other long step uses.

            if embedding.loaded():
                context, files = retrieve_context(collection, user_input)
            else:
                with Spinner("Loading the embedder (bge-m3) — first query of "
                             "this session, once"):
                    context, files = retrieve_context(collection, user_input)

            retrieval_ctx = "\n\n## Retrieved Context\n\n" + context
            retrieval_source = "project_vector_index"
            print(f"{C_DIM}  ⎿  RAG: {len(files)} files{C_RST}\n")
        else:
            context, files = "", set()
            retrieval_ctx = "\n\nNo RAG corpus — inspect files with bash."
            retrieval_source = "no_rag_runtime_notice"

        task_context_items = build_task_context_items(
            system_instructions=system_instructions,
            system_source=system_source,
            global_rules=global_rules,
            project_rules=project_rules,
            tool_guide=TOOL_GUIDE,
            tool_guide_source=(TOOL_GUIDE_FILE if os.path.isfile(TOOL_GUIDE_FILE)
                               else "built_in_tool_guide"),
            memories=memories_ctx,
            skills=skills_ctx,
            working_directory=cwd_note,
            working_state=working_state,
            retrieval=retrieval_ctx,
            retrieval_source=retrieval_source,
        )

        # Compatibility string for helpers still accepting the historical
        # signature. Every provider request is re-composed from the items.

        system_prompt = "".join(item.content for item in task_context_items)

        # The reusable runtime owns the provider-neutral model/tool lifecycle.

        if extra:
            user_msg = (f"{user_input}\n\n---\nResult of the automatically "
                        f"executed tool:\n```\n{extra[:8000]}\n```")
        else:
            user_msg = user_input

        hist = sanitize_history(history[-HISTORY_INJECT:])

        if hist and hist[-1]["role"] == "user":
            hist.pop()

        # Bound here rather than at the AgentContext below, because what this
        # turn asked for decides what of the conversation may reach the model
        # and not only which tools may. Called once: it announces the binding
        # and consumes the read context, so a second call would do both twice.

        turn_binding = standard_binding_for(user_input)

        # Withholding the local tools stopped a normative turn from going and
        # reading a working tree; it left the working tree an earlier turn had
        # already been shown sitting in the prompt. A question that stands on
        # its own words is answered from those words and the document.
        bound_turn = bool(turn_binding)
        said_before = next((message["content"] for message in reversed(hist)
                            if message["role"] == "user"), "")
        prior_scope = answer_scope.of(answer_scope.spoken_part(said_before),
                                      standard_bound=bound_turn)
        turn_scope = answer_scope.of(user_input, prior=prior_scope,
                                     standard_bound=bound_turn)
        hist = answer_scope.carried(
            hist, turn_scope, standard_bound=bound_turn,
            self_contained=answer_scope.self_contained(user_input))

        announce_carried_spec(user_input)
        conversation = canonical_history(hist)
        conversation.append(ConversationMessage("user", (TextBlock(user_msg),)))

        # Benchmark-only long-context prelude. This is intentionally opt-in
        # and absent from normal CLI execution; it supplies realistic stale
        # conversation groups so the existing semantic-compaction policy can
        # be measured without changing production thresholds.

        try:
            benchmark_turns = int(os.environ.get("SPEAR_BENCH_CONTEXT_TURNS", "0"))
        except ValueError:
            benchmark_turns = 0

        if benchmark_turns > 0:
            prelude = []
            filler = ("Older repository discussion was exploratory and is not authoritative. "
                            "Record only grounded evidence from tools; this historical note is "
                            "intentionally irrelevant. " * 40)

            for index in range(benchmark_turns):
                if index == 0:
                    text = ("IMPORTANT EARLY FACT: the acceptance requirement is that the "
                            "public behavior preserves the original input.\n" + filler)
                else:
                    text = f"Stale historical investigation turn {index}. {filler}"

                prelude.append(ConversationMessage(
                    "user" if index % 2 == 0 else "assistant",
                    (TextBlock(text),),
                ))

            conversation = prelude + conversation

        compaction_policy = compaction_policy_from_environment()

        session_configuration = SessionConfiguration(
            workspace=os.path.realpath(PROJECT_ROOT),
            project=PROJECT,
            execution_mode=EXECUTION_MODE.value,
            provider=TRACE_PROVIDER,
            model=TRACE_MODEL,
        )
        session_handle = SessionHandle(
            SESSION_STORE, new_session_id(), session_configuration,
        )
        # What the tree is actually built and tested with, discovered once.
        # It feeds the verification policy as well as the build gate: a run of
        # the project's own test command is the project's own verification,
        # and recognising it that way costs no guess about which files it
        # reached.

        project_spec = load_projects().get(PROJECT) or {}
        project_commands = project_build.commands(
            PROJECT_ROOT, spec=project_spec,
            cache_dir=f"{STATE_DIR}/projects/{PROJECT.replace('/', '_')}",
            infer_unittest=INFER_TEST_COMMAND)
        verification_policy = VerificationPolicy(VerificationHints.from_project({
            **project_spec,
            "build_commands": (list(_as_list(project_spec.get("build_commands")))
                               + ([project_commands.build] if project_commands.build else [])),
            "test_commands": (list(_as_list(project_spec.get("test_commands")))
                              + ([project_commands.test] if project_commands.test else [])),
        }))
        checkpoint_manager = CheckpointManager(
            CHECKPOINT_ROOT, PROJECT_ROOT, extra_roots=WORKSPACE.extra_roots)
        checkpoint = checkpoint_manager.begin_checkpoint(
            working_state.task_id, session_handle.session_id,
        )

        task_budget = BudgetManager(
            "task",
            {
                # These two are SAFETY NETS against a runaway process, not
                # the operating limit -- that is max_model_rounds, enforced by
                # the round loop itself. They were set just above it (+25 and
                # +5), and the redirects legitimately extend the round loop
                # past them: three build repairs, four work-order redirects, a
                # clause redirect and a write redirect each add three rounds.
                # So a turn doing exactly what the harness asked of it died at
                # 125 model turns with "budget_exhausted" -- reported as a
                # failure, with its work abandoned, five rounds after the
                # ceiling it was told it had. The net now sits above the
                # worst case (~155 rounds) instead of underneath it.
                BudgetKind.PRIMARY_MODEL_CALLS: BudgetLimit(max_tool_rounds + 55),
                # Compaction runs on this budget, and it is asked for once
                # per round while the context is under pressure. A flat four
                # was set when a turn was a handful of rounds; against a
                # 120-round writing turn it is spent in the first minute, and
                # the trace then shows a failed compaction EVERY round for the
                # rest of the turn -- pressure never relieved, context only
                # growing. It scales with the turn like every other limit here.
                BudgetKind.AUXILIARY_MODEL_CALLS: BudgetLimit(
                    max(4, max_tool_rounds // 3)),
                BudgetKind.MODEL_TURNS: BudgetLimit(max_tool_rounds + 45),
                BudgetKind.TOOL_CALLS: BudgetLimit(max_commands + 32),
                BudgetKind.COMMAND_EXECUTIONS: BudgetLimit(max_commands + 32),
                BudgetKind.INPUT_TOKENS: BudgetLimit(
                    CTX_LIMIT * (max_tool_rounds + 25), True,
                ),
                BudgetKind.OUTPUT_TOKENS: BudgetLimit(
                    int(os.environ.get("SPEAR_MAX_TOKENS", "8192"))
                    * (max_tool_rounds + 25), True,
                ),
                # A backstop in the one unit the operator actually feels.
                # Every other limit here counts rounds, calls or tokens, and
                # a turn can crawl for half an hour inside all of them --
                # one did today, with a failing compaction adding seconds to
                # every round. BudgetKind.WALL_TIME_MS existed in budgets.py
                # and was armed nowhere, the same way the cancellation token
                # was: a mechanism written, wired and never switched on.
                BudgetKind.WALL_TIME_MS: BudgetLimit(
                    int(os.environ.get("SPEAR_MAX_TURN_SECONDS", "1800")) * 1000),
                BudgetKind.CHILD_AGENT_CALLS: BudgetLimit(3),
                BudgetKind.CHILD_MODEL_TURNS: BudgetLimit(24),
            },
        )
        # One source per turn: cancelling is a one-way latch, so a token
        # reused across turns would leave every later turn pre-cancelled.
        turn_cancellation = CancellationSource()

        agent_context = AgentContext(
            cancellation=turn_cancellation.token,
            # The operator writes in ordinary language; a verb list only
            # knows the verbs somebody typed into it. One short call per
            # turn, and only when that list is silent.
            judge_intent=True,
            working_state=working_state,
            backend=backend,
            context_engine=CONTEXT_ENGINE,
            trace=TRACE,
            system_prompt=system_prompt,
            context_items=task_context_items,
            conversation=conversation,
            tools=(),
            tool_executor=lambda ctx, call_id, name, args, cache: route_tool_envelope(
                name, dict(args), cache, task_id=ctx.task_id,
                trace=ctx.trace, tool_call_id=call_id,
                cancellation=ctx.cancellation,
                agent_context=ctx,
            ),
            max_model_rounds=max_tool_rounds,
            max_tool_actions=max_commands,
            # The window the runtime PLANS against, deliberately smaller
            # than the one the server has. Compaction decides when to fire
            # from an estimate of len(text)//4, and that undercounts code and
            # JSON: a turn that had read five C files and fetched the
            # standard fifty-one times built a 65950-token request against a
            # 65536-token window and died on a 400, with the task untouched.
            # A margin is cheaper than an exact tokeniser and it fails the
            # right way -- compacting slightly early costs a summary, running
            # slightly late costs the whole turn.
            # The window the harness will use, as a fraction of the real one.
            #
            # It was 0.85, and that was the same caution counted twice. What
            # actually has to hold is context + answer <= window: the output
            # reserve and the safety margin below are that guarantee, stated
            # explicitly. On top of them the estimator counts three
            # characters to a token where prose runs about four, so every
            # figure it reports is already 15-30% higher than the truth.
            # Between the three, compaction began at 60% of the real window
            # and then ran every round.
            context_limit=int(CTX_LIMIT * float(
                os.environ.get("SPEAR_CONTEXT_FRACTION", "0.95"))),
            output_reserve=int(os.environ.get("SPEAR_MAX_TOKENS", "8192")),
            compaction_policy=compaction_policy,
            provider=TRACE_PROVIDER,
            model=TRACE_MODEL,
            observer=CliRuntimeObserver(),
            task_trace_metadata={
                "project": PROJECT,
                "project_kind": PROJECT_KIND,
                "data_origin": os.environ.get(
                    "SPEAR_TRAINING_DATA_ORIGIN", "NORMAL_USAGE",
                ).upper(),
                "user_input_chars": len(user_input),
                "raw_content_recorded": False,
            },
            session=session_handle,
            verification_policy=verification_policy,
            checkpoint_manager=checkpoint_manager,
            checkpoint=checkpoint,
            budget_manager=task_budget,
            standard_binding=turn_binding,
            project_commands=project_commands,
            project_root=PROJECT_ROOT,
            project_verifier=verify_project_command,
            prior_clauses=STANDARD_PRIOR_CLAUSES,
            prior_answer=STANDARD_PRIOR_ANSWER,
            # A fresh lifecycle per turn. It engages itself inside the
            # runtime, which is where both of the facts it needs are known;
            # attached unconditionally here because a disengaged one decides
            # nothing and costs nothing.
            work_phase=work_phase.WorkPhaseLedger(),
            # The contract only travels to a turn that pointed back at it. A
            # new, self-contained task names its own subject and inherits
            # nobody else's obligations.
            carried_requirements=(
                STANDARD_PRIOR_REQUIREMENTS
                if requirement_set.refers_back(user_input) else None),
        )
        session_handle.append(
            SessionEventType.SESSION_STARTED, working_state.task_id,
            {"provider": TRACE_PROVIDER, "model": TRACE_MODEL},
        )
        agent_context.apply_state_event(
            StateEventType.CHECKPOINT_RECORDED,
            checkpoint_id=checkpoint.checkpoint_id,
            status=checkpoint.status.value,
        )
        TRACE.emit(
            EventType.CHECKPOINT_STARTED, working_state.task_id,
            session_id=session_handle.session_id, status=EventStatus.STARTED,
            metadata={"checkpoint_id": checkpoint.checkpoint_id},
        )
        runtime = AgentRuntime()
        controller_executor = (
            lambda ctx, call_id, name, args, cache: route_tool_envelope(
                name, dict(args), cache, task_id=ctx.task_id,
                trace=ctx.trace, tool_call_id=call_id,
                cancellation=ctx.cancellation, agent_context=ctx,
            )
        )

        def child_session_factory(role, parent_context, child_state):
            return SessionHandle(
                SESSION_STORE, new_session_id(), SessionConfiguration(
                    workspace=os.path.realpath(PROJECT_ROOT), project=PROJECT,
                    execution_mode=ExecutionMode.SAFE.value,
                    provider=parent_context.provider, model=parent_context.model,
                ),
            )

        diff_service = DiffEvidenceService(
            result_store=RESULT_STORE, model_context_chars=16_000,
        )

        def current_diff_evidence(context):
            if context.checkpoint is not None:
                try:
                    return diff_service.from_checkpoint(
                        checkpoint_manager, context.checkpoint,
                    )
                except Exception as exc:
                    return diff_service.unavailable(
                        tuple(sorted(context.working_state.relevant_files)),
                        f"checkpoint diff unavailable: {type(exc).__name__}",
                    )

            changed_paths = tuple(sorted(
                context.working_state.modified_files
                | context.working_state.created_files
                | context.working_state.deleted_files
            ))
            git_result = run_cmd_result(
                "git diff --no-ext-diff --binary -- .", need_confirm=False,
                cancellation=context.cancellation, execution_mode=ExecutionMode.SAFE,
            )

            if git_result.status != "ok" or git_result.exit_code not in (None, 0):
                return diff_service.unavailable(
                    changed_paths,
                    "read-only Git diff unavailable: "
                    f"{git_result.status} (exit {git_result.exit_code})",
                )

            return diff_service.from_git(
                context.task_id, lambda: command_result_text(git_result),
                changed_paths=changed_paths,
            )

        def compatibility_mutation(context, result):
            # Nothing is synthesized for a turn that may not write. Checked
            # before the block is even extracted, so no mutation-specific
            # heuristic runs, no refining model call is spent, and no
            # message goes back recommending a different way to edit.

            if not mutation_permitted(context):
                return result

            block, language = extract_code_block_with_language(
                result.final_response or "\n".join(result.transcript),
            )
            tool_log = list(result.tool_log)
            target = guess_target_file(user_input, tool_log)
            looks_garbage = bool(block) and bool(re.search(
                r"<function=|<parameter=|\{\s*\"name\"\s*:\s*\"", block))

            if not (block and target and len(block) > 200
                    and not looks_garbage and not is_excluded_path(target)):
                return result

            if not block_fits_target(block, language, target):
                # Say it. A silent decline here is what left the model
                # rediscovering the situation for six turns.

                print(f"\n{C_DIM}  ⎿ the printed block is not "
                      f"{os.path.basename(target)}'s kind of content — not "
                      f"applying it. Name the file to write, or ask for the "
                      f"tool call.{C_RST}")
                return result

            rel = os.path.relpath(target, PROJECT_ROOT)

            if os.path.isfile(target) and try_surgical_edits(
                backend, system_prompt, list(result.conversation), rel, block,
                result.execution_cache, tool_log, context_items=task_context_items,
                agent_context=context, runtime=runtime,
            ):
                result.did_modify = True

            if not result.did_modify:
                print(f"\n{C_DIM}  ⎿ model printed a full file — applying "
                      f"to {rel}{C_RST}")

                try:
                    applied = execute_tool(
                        "write_file", {"path": rel, "content": block},
                        result.execution_cache, agent_context=context,
                    )

                    if applied.startswith("OK"):
                        result.did_modify = True
                        tool_log.append(f"write_file {rel} (from code block)")
                except Exception as exc:
                    print(f"  {C_ERR}⎿ apply failed: {exc}{C_RST}")

            result.tool_log = tuple(tool_log)

            return result

        # Set by the observer below, read after the turn: what a bench judged
        # is recorded here, and what nothing judged is recorded there. Without
        # this flag the two paths would either duplicate a trajectory or, as
        # before, drop every turn a project has no bench for.

        bench_recorded = []

        def project_bench_observer(context, result, verdict, source):
            if source not in {"bench", "bench_retry"}:
                return

            count = save_trajectory(
                user_input, result.trajectory, result.final_response,
                "pass" if verdict else "fail", source, context.task_id,
            )
            bench_recorded.append(True)
            mark = (f"{C_OK}passed{C_RST}" if verdict
                    else f"{C_ERR}FAILED{C_RST}")
            label = "repair bench" if source == "bench_retry" else "bench"
            print(f"  {C_DIM}⎿ {label} {mark}{C_DIM} — trajectory recorded "
                  f"({count} in {os.path.basename(TRAJECTORY_FILE)}){C_RST}\n")

        controller = TaskController(
            runtime, TOOL_REGISTRY, tool_executor=controller_executor,
            child_session_factory=child_session_factory,
            verification_runner=run_project_bench,
            bench_observer=project_bench_observer,
            diff_provider=current_diff_evidence,
            compatibility_mutation=compatibility_mutation,
            training_store=TRAINING_STORE,
            standard_store=STANDARD_STORE,
        )

        def benchmark_toggle(name, default=True):
            raw = os.environ.get("SPEAR_BENCH_" + name.upper())

            if raw is None:
                return default

            return raw.strip().lower() in {"1", "true", "yes", "on"}

        # Ctrl+C stops the work, not the session. The runtime and the
        # controller each end their own turn cleanly, but everything
        # around them -- planning, the review pass, the acceptance bench
        # and the subprocesses it starts -- was outside any handler, so an
        # interrupt there escaped main() and took the shell with it.
        # Whatever the turn had already done is still on disk; the prompt
        # comes back and the operator decides what to do about it.
        try:
            with interruptible(turn_cancellation):
                task_result = controller.run(TaskRequest(
                    user_input, agent_context, (os.path.realpath(PROJECT_ROOT),),
                    project_rules=tuple(
                        item.content for item in task_context_items
                        if item.layer == ContextLayer.PROJECT_RULES
                    ),
                    # --no-network reaches THIS path too. Dropping the web tools from
                    # the TOOLS list was not enough: the role-aware exposure selects
                    # from the registry itself, which still holds them, so the flag
                    # would have been honoured on the legacy path only.
                    web_enabled=NETWORK_ENABLED,
                    enable_explorer=benchmark_toggle("explorer", default=False),
                    enable_reviewer=benchmark_toggle("reviewer", default=False),
                    enable_planning=benchmark_toggle("planning"),
                    enable_compaction=benchmark_toggle("compaction"),
                    selected_memory=benchmark_toggle("selected_memory"),
                    role_aware_tools=benchmark_toggle("role_aware_tools"),
                    enable_review_repair=(
                        benchmark_toggle("reviewer", default=False)
                        and benchmark_toggle("review_repair", default=False)
                    ),
                ))
        except KeyboardInterrupt:
            print(f"\n{C_DIM}⏺ Interrupted — back to the prompt. Anything "
                  f"the turn already changed is still on disk.{C_RST}\n")

            continue

        if task_result.status == TaskStatus.INTERRUPTED:
            # Not an error, and not nothing. Stopping a turn on purpose is an
            # ordinary thing to do, so it does not get the red marker a
            # failure gets -- and whatever the turn had already said is shown
            # rather than dropped, because the operator interrupted the work,
            # not the report of it.

            partial = (task_result.final_response or "").strip()

            if partial:
                print_assistant(partial)

            calls = len(runtime_tool_log(task_result))
            print(f"\n{C_DIM}⏺ Interrupted after {calls} tool call(s) — back "
                  f"to the prompt. Anything the turn changed is still on "
                  f"disk.{C_RST}\n")

            continue

        if task_result.status in {TaskStatus.FAILED, TaskStatus.BUDGET_EXHAUSTED}:
            # Say WHAT failed, not merely that something did. The category is
            # the exception class and the reason is where the runtime stopped;
            # both were already recorded and neither was ever shown, so a
            # failed turn printed the bare words "runtime failure" and left
            # nothing to act on -- tracing is off by default and a failure
            # during persistence loses the session record too.

            failed = task_result.agent_result
            parts = [p for p in (failed.error_summary, failed.error_category)
                     if p]
            message = " · ".join(dict.fromkeys(parts)) or "no detail recorded"

            # Running out of room is not the same as failing, and a turn that
            # answered before it ran out should not be presented as wreckage.
            # The red banner sat above answers that were finished, and above
            # edits that had been built and tested: the operator read
            # "Could not complete this turn" and threw away work that was
            # sound. The failure marker is kept for turns that produced
            # nothing.

            ceiling = (task_result.status == TaskStatus.BUDGET_EXHAUSTED
                       and (task_result.final_response or "").strip())

            if ceiling:
                print(f"\n{C_DIM}⏺ This turn reached its ceiling ({message[:160]}) "
                      f"— what it did is kept, and the answer follows.{C_RST}")
            else:
                print(f"\n{C_ERR}⏺{C_RST} {C_DIM}Could not complete this turn "
                      f"({message[:240]}){C_RST}")

            # Name the ceilings. The reason says WHICH budget ended, but the
            # count printed beside it is tool calls either way, so a round
            # budget that ended after 30 calls read as "30 calls was the
            # limit" -- and the limit was 60. An operator cannot raise what
            # the message does not name.

            print(f"  {C_DIM}reason: {task_result.terminal_reason} · "
                  f"{len(runtime_tool_log(task_result))} tool call(s) this turn"
                  f" · limits: {max_tool_rounds} rounds / {max_commands} tool "
                  f"calls / {max_tool_rounds + 45} model turns "
                  f"(SPEAR_MAX_TOOL_ROUNDS, SPEAR_MAX_COMMANDS)"
                  f"{C_RST}\n")

            # A cut turn is not an empty one. Falling straight to `continue`
            # threw away whatever the model had already concluded AND skipped
            # the history append below, so a turn that edited a file and built
            # it successfully left no trace in the session: the next prompt
            # started from nothing and did the same work again. Keep the
            # partial answer if there is one, and otherwise leave a line
            # saying what happened, so the next turn knows.

            if not task_result.final_response:
                if runtime_tool_log(task_result):
                    note = (f"[turn cut: {task_result.terminal_reason} after "
                            f"{len(runtime_tool_log(task_result))} tool calls. "
                            f"Work already done this turn:\n"
                            + "\n".join(runtime_tool_log(task_result)[-3:])[:1200]
                            + "\n]")
                    history.append({"role": "user", "content": user_input})
                    history.append({"role": "assistant", "content": note})
                    save_history(history)

                continue

        # What the turn read, kept for the next one. This is the whole of
        # the two-prompt shape: the question established the clauses, the
        # follow-up changes the code against them.

        read = tuple(getattr(agent_context, "clauses_read", ()) or ())

        if read:
            STANDARD_PRIOR_CLAUSES = read
            STANDARD_PRIOR_ANSWER = (task_result.final_response or "")[:4000]
            STANDARD_PRIOR_REQUIREMENTS = requirement_set.publish(
                getattr(agent_context, "standard_policy", None),
                task_result.final_response or "", origin=user_input[:120])

        runtime_result = task_result.agent_result
        response_text = task_result.final_response
        _record_bench_answer(response_text)
        transcript = list(runtime_result.transcript)
        tool_log = list(runtime_result.tool_log)
        trajectory = list(runtime_result.trajectory)

        # The RECORDING is deliberately wider than the judging. A turn is
        # judged only by the project's own acceptance command, and only when it
        # CHANGED something — an answered question has nothing to accept. But
        # gating the recording on a verdict meant a project without a bench
        # recorded nothing at all: one project declares one, so after months
        # the dataset held a single trajectory, and that one a failure. A turn
        # nothing judged is still worth keeping — it is labelled "unrated"
        # rather than dropped, and `/good` promotes what was right.
        # The bench observer above already saved the judged ones.

        # The project's own build and tests are a verdict, and they already
        # ran: every writing turn is built and tested by the gate. Filing
        # those turns as "unrated" beside turns nothing ever checked threw
        # away the one label the harness produces for free -- which is the
        # whole of "a verdict without /good". A tree with no build, or a turn
        # that wrote nothing, still records "unrated": not judged is not the
        # same as judged and wrong, and a corpus that confuses them teaches
        # the confusion.

        if response_text and trajectory and not bench_recorded:
            built = getattr(runtime_result, "project_build_ok", None)
            verdict = ("unrated" if built is None
                       else "pass" if built else "fail")
            save_trajectory(
                user_input, trajectory, response_text, verdict,
                "project_build" if built is not None
                else "change" if changed_files(tool_log) else "answer")

        # don't print a noisy "(no response)" when an analysis was already
        # shown in the transcript this turn — just close the turn quietly.

        if response_text:
            print_assistant(response_text)
            print()
        elif not transcript:
            print_assistant("")

        # History keeps follow-up context ("ok redo it", "what next?") but
        # tool transcripts go on the USER side — the same place they appear
        # during live execution — so the model never sees itself "writing"
        # tool output (which it would then imitate by fabricating).

        user_record = user_input

        if tool_log:
            # Words yes, handles no. The transcript is kept so a follow-up can
            # refer back to what happened; the source ids in it are stripped,
            # because a later turn that reads one can fetch that evidence
            # again and answer a new question from an old question's clauses.
            joined = evidence_handles.redact("\n\n".join(tool_log)[:2000])
            user_record += ("\n\n[Tools executed during this turn — results:\n"
                            f"{joined}\n]")

        full_response = "\n\n".join(
            transcript + ([response_text] if response_text else []))
        history.append({"role": "user", "content": user_record})
        archive_entry({"role": "user", "content": user_record})

        if full_response:
            history.append({"role": "assistant", "content": full_response})
            archive_entry({"role": "assistant", "content": full_response})

        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]

        save_history(history)


if __name__ == "__main__":
    main()
