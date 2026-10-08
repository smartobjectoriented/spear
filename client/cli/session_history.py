"""Conversation history on disk, its archive, trajectories and experience."""

import os
import json
from datetime import datetime, timezone
from cli import session_workspace
from cli.chat_settings import STATE_DIR
from cli.corpus_search import archive_index
from cli.project_checks import project_bench


MAX_HISTORY = 80          # total kept on disk (rotation)
HISTORY_INJECT = 40       # messages re-injected into the prompt each turn


# ── history persistence ──────────────────────────────────────────────

def fresh_session() -> bool:
    """A session that neither reads nor replaces the corpus's stored conversation."""
    return os.environ.get("SPEAR_FRESH", "") not in ("", "0")


def load_history():
    if fresh_session():
        return []

    if os.path.isfile(session_workspace.HISTORY_FILE):
        try:
            with open(session_workspace.HISTORY_FILE, "r") as f:
                return json.load(f)[-MAX_HISTORY:]
        except (json.JSONDecodeError, KeyError):
            pass

    return []


def save_history(history):
    if fresh_session():
        return

    with open(session_workspace.HISTORY_FILE, "w") as f:
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
        "project": session_workspace.PROJECT,
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
