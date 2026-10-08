"""Session settings: flags made environment, and the paths derived from it."""

import os
import sys
from harness.tool_runtime import AuditLogger
from runtime.tracing import create_trace_emitter
from context.context_engine import ContextEngine
from runtime.result_store import ResultStore
from runtime.session_store import FileSessionStore
from training.training_store import TrainingStore
from standard.standard_commands import StandardOperator
from standard.standard_store import StandardStore
from standard.standard_tools import StandardToolService


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
    "--ctx": ("SPEAR_CTX", "context window in tokens (default: asked of the server, else 32768)"),
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
    "--fresh": ("SPEAR_FRESH", "1", "start without this corpus's stored conversation, and "
                                    "leave it as it is; knowledge, rules and configuration "
                                    "still apply"),
}


def apply_env_options(argv):
    """Move the settings flags into the environment, and out of argv.

    Out of argv deliberately: their VALUES would otherwise sit there as bare
    words, and several places in the client test membership against sys.argv.
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

APP_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ROOT_DIR = os.path.dirname(APP_DIR)

LLAMA_SERVER_URL = os.environ.get("SPEAR_API_BASE", "http://127.0.0.1:8080/v1")
# A test run that names no index or state gets a temporary one: the defaults
# are this checkout's own index and state, which no test may write.
from runtime import state_paths

_TEST_STATE = state_paths.test_state_root()
DB_PATH = (os.environ.get("SPEAR_DB_PATH")
           or (f"{_TEST_STATE}/chromadb" if _TEST_STATE else f"{APP_DIR}/chromadb"))


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

STATE_DIR = os.environ.get("SPEAR_STATE_DIR") or _TEST_STATE or APP_DIR
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
STANDARD_OPERATOR = StandardOperator(STANDARD_STORE)
STANDARD_TOOL_SERVICE = StandardToolService(STANDARD_STORE)
