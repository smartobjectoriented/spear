#!/usr/bin/env python3
"""
RAG-augmented chat with a local Qwen3 MoE (llama-server, native tool calling).
Claude Code-style UI (⏺ bullets, ⎿ tool results, spinner, box).
Hybrid tool system: explicit !commands + auto-detection + LLM tool calls.
"""

import os
import re
import sys
import fcntl
import select
import shlex
import time
import subprocess
from pathlib import Path

# Run as a file, the script's own directory is first on the path; the
# packages it imports live one level up.

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from context import answer_scope
from context import context_selection
from retrieval import embedding
from evidence import evidence_handles
from context import skill_library
from normative import requirement_set
from runtime import work_phase
from models.model_backend import (
    ConversationMessage, ModelBackendConfigurationError, TextBlock,
)
from runtime.cancellation import CancellationSource
from harness.checkpoint import CheckpointManager
from harness.tool_primitives import ExecutionMode
from runtime.tracing import EventStatus, EventType, new_task_id
from runtime.working_state import StateEventType, WorkingState
from context.context_engine import ContextLayer
from context.workspace_context import project_families
from runtime.compaction import CompactionMode, CompactionPolicy
from runtime.agent_context import AgentContext
from runtime.agent_runtime import AgentRuntime
from runtime import session_replay
from runtime.budgets import BudgetKind, BudgetLimit, BudgetManager
from evidence.diff_evidence import DiffEvidenceService
from runtime.task_controller import TaskController, TaskRequest, TaskStatus
from runtime.session_store import (
    SessionConfiguration, SessionEventType, SessionHandle, new_session_id,
)
from evidence.verification import VerificationHints, VerificationPolicy
from training.training_store import TrainingStore
from evidence import project_build
from standard.standard_commands import (
    StandardCommandError, handle_standard_command, standard_help_lines,
)
from cli import corpus_search, model_io, session_workspace, standard_session
from cli.chat_settings import (
    APP_DIR, CHECKPOINT_ROOT, CONTEXT_ENGINE, LEARNED_RULES_FILE,
    LLAMA_SERVER_URL, RESULT_STORE, RULES_BUDGET, SESSION_STORE,
    STANDARD_OPERATOR, STANDARD_STORE, STATE_DIR, TRACE, TRAINING_STORE,
)
from cli.corpus_registry import (
    handle_corpus_command, load_projects, resolve_project_at_startup,
)
from cli.corpus_search import (
    archive_forget, corpus_property, init_chromadb, reindex_command,
    retrieve_context,
)
from cli.knowledge_commands import (
    SKILLS_DIR, current_workspace, knowledge_command, knowledge_store,
    legacy_notes_notice, library_skills, remember_fact,
)
from cli.model_io import (
    TOOL_GUIDE, TOOL_GUIDE_FILE, canonical_history, create_model_backend,
    resolve_context_window, sanitize_history,
)
from cli.operator_input import (
    PROMPT, auto_detect_and_run, corpus_mention_hint, handle_bang_command,
    init_readline,
)
from cli.project_checks import (
    INFER_TEST_COMMAND, _as_list, _record_bench_answer,
    run_normative_check, run_project_bench, verify_project_command,
)
from cli.reply_edits import (
    block_fits_target, extract_code_block_with_language, guess_target_file,
    try_surgical_edits,
)
from cli.session_history import (
    HISTORY_INJECT, MAX_HISTORY, TRAJECTORY_FILE, archive_entry,
    load_history, save_experience, save_history, save_trajectory,
)
from cli.session_workspace import (
    command_result_text, extra_roots_note, is_excluded_path,
    mutation_permitted, run_cmd_result, sandbox_mount, set_project,
)
from cli.standard_session import (
    announce_carried_spec, is_write_request_text, standard_binding_for,
)
from cli.startup_banner import (
    _corpus_summary, backend_label, banner, banner_art, print_help,
)
from cli.terminal_ui import (
    C_BOLD, C_DIM, C_ERR, C_OK, C_RST, C_WARN, CliRuntimeObserver, STATUS,
    Spinner, SpinnerProgress, interruptible, print_assistant,
    runtime_tool_log,
)
from cli.tool_handlers import NETWORK_ENABLED
from cli.tool_routing import (
    TOOL_REGISTRY, coding_host, execute_tool, route_tool_envelope,
)
from cli.turn_context import (
    build_task_context_items, load_corpus_rules, load_rules,
    save_learned_rule, select_turn_context,
)


TRACE_PROVIDER = None
TRACE_MODEL = None


def operator_training_controller():
    """Build the operator control plane; never exposed through ToolRegistry."""
    from training.training_controller import TrainingController
    from training.training_launcher import TrainingExecutionConfiguration

    config_path = Path(STATE_DIR) / "training-execution.json"
    execution = (TrainingExecutionConfiguration.load(config_path)
                 if config_path.is_file() else None)
    store = TRAINING_STORE or TrainingStore(f"{STATE_DIR}/audit/training-data")

    return TrainingController(store, f"{STATE_DIR}/audit/training-data",
                              execution=execution)


# ── main ─────────────────────────────────────────────────────────────

ADHOC_PROMPT = (
    "You are HEIG-VD/REDS AI, an expert embedded-software assistant running "
    "in ad-hoc mode: you operate on the user's CURRENT directory (shown "
    "below). Assume nothing about its layout from any other project you "
    "may know -- inspect the actual directory with your tools before "
    "answering. Your long-term memory (the "
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
    global TRACE_PROVIDER, TRACE_MODEL

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
    corpus_search.COLLECTION = collection

    try:
        backend, provider = create_model_backend(sys.argv[1:])
    except ModelBackendConfigurationError as exc:
        print(f"{C_ERR}Model backend configuration error: {exc}{C_RST}")
        return

    TRACE_PROVIDER = provider

    # An explicit SPEAR_CTX (or --ctx) wins; otherwise the server is asked.
    # A replay contacts no server, so it keeps what it was given.

    if not os.environ.get(session_replay.REPLAY_ENV, "").strip():
        model_io.CTX_LIMIT, model_io.CTX_SOURCE = resolve_context_window(
            os.environ, LLAMA_SERVER_URL, provider=provider)
        print(f"  context window: {model_io.CTX_LIMIT} ({model_io.CTX_SOURCE})")

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

    from context import workspace_knowledge

    try:
        n_mem = len(knowledge_store(TRACE).list(current_workspace().workspace_id,
                                                (workspace_knowledge.Lifecycle.ACTIVE,)))
    except Exception:                                # noqa: BLE001
        n_mem = 0

    banner(collection, history, n_rules, model_name, n_mem)
    legacy = legacy_notes_notice()

    if legacy:
        print(f"{C_DIM}  {legacy}{C_RST}")

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
                standard_session.STANDARD_READ_CONTEXT = result

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

            label = " ".join(user_input.split()[:2]) + "…"

            try:
                with Spinner(label) as spinner:
                    STANDARD_OPERATOR.progress = SpinnerProgress(spinner)

                    try:
                        result = handle_standard_command(user_input,
                                                         STANDARD_OPERATOR)
                    finally:
                        STANDARD_OPERATOR.progress.finish()
            except (StandardCommandError, OSError, ValueError) as exc:
                result = f"Standard command error: {exc}"
            finally:
                STANDARD_OPERATOR.progress = None

            print("\n" + result + "\n")

            continue

        if user_input == "/finetune" or user_input.startswith("/finetune "):
            # Explicit operator branch: no WorkingState, AgentRuntime, tool
            # dispatch, or ModelBackend call exists on this path.

            from cli.finetune_commands import (FinetuneCommandError,
                                           handle_finetune_command)

            try:
                with Spinner(" ".join(user_input.split()[:2]) + "…"):
                    result = handle_finetune_command(
                        user_input, operator_training_controller())

                print("\n" + result + "\n")
            except (FinetuneCommandError, OSError, ValueError) as exc:
                print(f"\nFine-tuning command error: {exc}\n")

            continue

        if user_input.startswith("/search "):
            with Spinner("Searching…"):
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

            print("\n" + (err or handle_corpus_command(argv, current=session_workspace.PROJECT))
                  + "\n")

            continue

        if user_input == "/reindex":
            print("Reindexing...")

            # Root, destination collection and exclusions all come from
            # reindex_command(): each of the three was got wrong here once.

            subprocess.run(reindex_command())
            collection = init_chromadb()
            corpus_search.COLLECTION = collection

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

            standard_session.STANDARD_ENGAGED_BEFORE = False
            standard_session.STANDARD_PRIOR_CLAUSES = ()
            standard_session.STANDARD_PRIOR_ANSWER = ""
            standard_session.STANDARD_PRIOR_REQUIREMENTS = requirement_set.RequirementSet()
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
                    elif not skill_library.applies_to(
                            skill, project=session_workspace.PROJECT,
                            kind=session_workspace.PROJECT_KIND,
                            families=project_families(session_workspace.PROJECT_SPEC)[0]):
                        state = f" {C_DIM}other corpus{C_RST}"
                    else:
                        state = ""

                    print(f"  - {skill.name}{scoped}{state}")

                    if skill.summary:
                        print(f"    {C_DIM}{skill.summary[:100]}{C_RST}")

                print(f"  ({len(skills)} skills in {SKILLS_DIR})\n")

            continue

        if user_input == "/knowledge" or user_input.startswith("/knowledge "):
            print(knowledge_command(user_input[len("/knowledge"):].strip()) + "\n")

            continue

        if user_input.startswith("/remember"):
            note = user_input[len("/remember"):].strip()

            if note:
                print(remember_fact(note) + "\n")
            else:
                print(knowledge_command("list") + "\n(usage: /remember <a fact about "
                      "this workspace>; /knowledge for the rest)\n")

            continue

        if user_input.startswith("/recall"):
            # The global counterpart of /remember. /remember writes to
            # memories-<corpus>.md and is invisible in every other corpus,
            # which is the wrong home for something like "never rewrite an
            # existing copyright header" -- true in SO3, in the build-system trees and in
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

        hint = corpus_mention_hint(user_input, load_projects(), session_workspace.PROJECT)

        if hint:
            print(f"{C_DIM}  ⎿  {hint}{C_RST}")

        # ── auto-detect → run tool → inject into LLM ──

        extra, desc = auto_detect_and_run(user_input, collection)

        # ── RAG retrieval (skipped in ad-hoc mode) ──

        skill_matches = library_skills()
        skills_ctx = ""

        # tools always run in the cwd (PROJECT_ROOT); relative paths in your
        # bash/edit/write calls resolve from there — tell the model.
        # Say what bash actually SEES, not only where the workspace lives on
        # the host. The sandbox normally binds the tree at its own path, so the
        # two agree and the note can say so; under the legacy /workspace mount
        # it must warn that the host path is absent there, since announcing
        # only the host path once made the model spend a whole session trying
        # `ls /opt/llm/spear` and conclude the tree had vanished.

        mount = sandbox_mount()

        if (mount == str(session_workspace.PROJECT_ROOT)
                or mount == os.path.realpath(session_workspace.PROJECT_ROOT)):
            where = (f"Inside bash the sandbox shows that directory at that "
                     f"same path — `pwd` prints `{mount}` — so absolute paths "
                     f"into the tree work as they do on the host. ")
        else:
            where = (f"Inside bash the sandbox shows that same directory as "
                     f"`{mount}` and `pwd` prints `{mount}` — the host path "
                     f"does not exist there, and absolute paths are refused. ")

        cwd_note = (f"\n\n## Working directory\n\n"
                    f"Your tools (bash, edit_file, write_file, ...) act on the "
                    f"CURRENT directory, which is {session_workspace.PROJECT_ROOT} on the host. "
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
        # Legacy remembered notes are no longer shown to a turn: what a
        # workspace keeps is its knowledge (workspace_knowledge), which
        # select_turn_context adds itself.
        memories_ctx = ""

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
            workspace=os.path.realpath(session_workspace.PROJECT_ROOT),
            project=session_workspace.PROJECT,
            execution_mode=session_workspace.EXECUTION_MODE.value,
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

        project_spec = load_projects().get(session_workspace.PROJECT) or {}
        project_commands = project_build.commands(
            session_workspace.PROJECT_ROOT, spec=project_spec,
            cache_dir=f"{STATE_DIR}/projects/{session_workspace.PROJECT.replace('/', '_')}",
            infer_unittest=INFER_TEST_COMMAND)
        verification_policy = VerificationPolicy(VerificationHints.from_project({
            **project_spec,
            "build_commands": (list(_as_list(project_spec.get("build_commands")))
                               + ([project_commands.build] if project_commands.build else [])),
            "test_commands": (list(_as_list(project_spec.get("test_commands")))
                              + ([project_commands.test] if project_commands.test else [])),
        }))
        checkpoint_manager = CheckpointManager(
            CHECKPOINT_ROOT, session_workspace.PROJECT_ROOT,
            extra_roots=session_workspace.WORKSPACE.extra_roots)
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
                    model_io.CTX_LIMIT * (max_tool_rounds + 25), True,
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

        # Which context each runtime is shown: only this workspace's, only what
        # the request's class and the pass it is in call for, and whatever is
        # explicitly generic. Decided without a model, and traced.

        workspace, context_selections, phase_contexts = select_turn_context(
            user_input=user_input, turn_scope=turn_scope, binding=turn_binding,
            write=is_write_request_text(user_input), project_spec=project_spec,
            project_commands=project_commands,
            history_text="".join(str(message.get("content") or "") for message in hist),
            memories=memories_ctx, skills=skill_matches, retrieval=retrieval_ctx,
            system_instructions=system_instructions, system_source=system_source,
            tool_guide=TOOL_GUIDE, working_directory=cwd_note,
            task_id=working_state.task_id, session_id=session_handle.session_id,
            trace=TRACE)
        turn_phase = context_selection.primary_phase(turn_scope, bound=bool(turn_binding))
        chosen = phase_contexts[turn_phase]

        if chosen["skills"]:
            offered = chosen["skills"].count("# Skill: ")
            print(f"{C_DIM}  ⎿  {offered} skill{'s' if offered != 1 else ''} available{C_RST}")

        if chosen.get("capabilities_unsupported"):
            print(f"{C_DIM}  ⎿  external capabilities are not available to this turn: a "
                  f"standard-bound change runs on the legacy runtime{C_RST}")

        def context_items_for(rendered):
            return build_task_context_items(
                system_instructions=system_instructions, system_source=system_source,
                global_rules=rendered["global_rules"], project_rules=rendered["project_rules"],
                tool_guide=TOOL_GUIDE,
                tool_guide_source=(TOOL_GUIDE_FILE if os.path.isfile(TOOL_GUIDE_FILE)
                                   else "built_in_tool_guide"),
                memories=rendered["memories"], skills=rendered["skills"],
                working_directory=cwd_note, working_state=working_state,
                retrieval=rendered["retrieval"], retrieval_source=retrieval_source)

        task_context_items = context_items_for(chosen)
        system_prompt = "".join(item.content for item in task_context_items)

        if context_selection.MIXED_PREPASS in phase_contexts:
            phase_contexts[context_selection.MIXED_PREPASS]["context_items"] = \
                context_items_for(phase_contexts[context_selection.MIXED_PREPASS])

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
            # What the agent core needs from this client: its host (the
            # control plane), and the project context its prompt carries.
            coding_host=lambda ctx, cache, record: coding_host(ctx, cache, record),
            coding_context=chosen["coding_context"],
            capability_gateway=chosen.get("gateway"),
            knowledge_door=chosen.get("knowledge_door"),
            workspace_context=workspace,
            context_selections=context_selections,
            phase_contexts=phase_contexts,
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
            context_limit=int(model_io.CTX_LIMIT * float(
                os.environ.get("SPEAR_CONTEXT_FRACTION", "0.95"))),
            output_reserve=int(os.environ.get("SPEAR_MAX_TOKENS", "8192")),
            compaction_policy=compaction_policy,
            provider=TRACE_PROVIDER,
            model=TRACE_MODEL,
            observer=CliRuntimeObserver(),
            task_trace_metadata={
                "project": session_workspace.PROJECT,
                "project_kind": session_workspace.PROJECT_KIND,
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
            project_root=session_workspace.PROJECT_ROOT,
            project_verifier=verify_project_command,
            normative_project={key: project_spec.get(key) for key in
                               ("normative_checks", "normative_applicability")
                               if project_spec.get(key)},
            normative_authority=f"projects.json:{session_workspace.PROJECT}",
            normative_check_runner=run_normative_check,
            prior_clauses=standard_session.STANDARD_PRIOR_CLAUSES,
            prior_answer=standard_session.STANDARD_PRIOR_ANSWER,
            # A fresh lifecycle per turn. It engages itself inside the
            # runtime, which is where both of the facts it needs are known;
            # attached unconditionally here because a disengaged one decides
            # nothing and costs nothing.
            work_phase=work_phase.WorkPhaseLedger(),
            # The contract only travels to a turn that pointed back at it. A
            # new, self-contained task names its own subject and inherits
            # nobody else's obligations.
            carried_requirements=(
                standard_session.STANDARD_PRIOR_REQUIREMENTS
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
                    workspace=os.path.realpath(session_workspace.PROJECT_ROOT),
                    project=session_workspace.PROJECT,
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

            # The coding loop changes files through its tools only; a code
            # block in its prose is prose.
            if getattr(context, "execution_core", "legacy") == "coding":
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

            rel = os.path.relpath(target, session_workspace.PROJECT_ROOT)

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
                    user_input, agent_context, (os.path.realpath(session_workspace.PROJECT_ROOT),),
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
                    # Unbound turns run on the coding loop unless the operator
                    # asks for the legacy one (an ablation switch).
                    coding_core=(os.environ.get("SPEAR_EXECUTION_CORE", "coding")
                                 .strip().lower() != "legacy"),
                    retrieval_available=corpus_search.COLLECTION is not None,
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
            standard_session.STANDARD_PRIOR_CLAUSES = read
            standard_session.STANDARD_PRIOR_ANSWER = (task_result.final_response or "")[:4000]
            standard_session.STANDARD_PRIOR_REQUIREMENTS = requirement_set.publish(
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
                else "change" if runtime_result.changed_paths else "answer")

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
