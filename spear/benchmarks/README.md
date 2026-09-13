# Real-model benchmark and ablations

This package measures the configured local model through the normal
`rag_chat.py` / `ModelBackend` path.  It is separate from the deterministic
regression tests: `--smoke` checks fixtures and reporting only and is **not** a
quality result.

Examples (from `spear`):

```bash
# Infrastructure smoke run (writes only under audit/benchmarks/)
/opt/llm/spear/spear/bin/python -m benchmarks.runner --smoke \
  --task control-edit --output audit/benchmarks/smoke

# Real local-model trial, one repetition of the control task
/opt/llm/spear/spear/bin/python -m benchmarks.runner --real \
  --configuration full --task control-edit --repetitions 1 \
  --output audit/benchmarks/real-control

# Resume incomplete runs and aggregate an existing manifest
/opt/llm/spear/spear/bin/python -m benchmarks.runner --real --resume \
  --output audit/benchmarks/real-control
/opt/llm/spear/spear/bin/python -m benchmarks.runner --report \
  --output audit/benchmarks/real-control
```

Configurations are explicit and do not alter normal production defaults:
`full`, `no-explorer`, `no-reviewer`, `no-planning`, `no-compaction`,
`full-tools`, `no-memory`, and `no-repair`.  Generated manifests/traces are
ignored under `audit/`; they contain safe metadata and no prompts or secrets.

The suite contains 23 small repository-local tasks across control,
exploration, multi-file, review, context-pressure, memory, and recovery
classes.

## What decides a task

Every task declares at least one oracle, and a task that declares none cannot
be scored a success — four of them used to declare none, and every run of them
was recorded as a success.  `tests/test_benchmarks.py` asserts both that
property and the one that makes the suite a measurement: **no task passes on
its own untouched fixture.**

| oracle | what it checks |
|---|---|
| `expected` | a fragment (or any of several) is present in a file |
| `forbidden_content` | a fragment the task rules out is absent |
| `forbidden_paths` | a file it was told not to touch is unchanged |
| `required_reads` | the reads happened, as grounded in the trace |
| `acceptance` | the task's own tests **run** and pass against the finished workspace |
| `expected_answer` / `forbidden_answer` | what the run answered, matched without regard to case |
| `requires_agent_verification` | the run verified its own change |

### Two verdicts

`functionally_correct` is what the workspace and the hidden oracle say: the
file the task was about ends up right.  `task_success` is whether the run did
the task.  They came apart on the first real control-edit run: the fix was
correct, the oracle green, and the run had tried `python3 -c` twice, never run
the tests, and ended on *"UNVERIFIED — main.py changed without sufficient
current verification"*.  It was right about itself; the score said 100%.

Tasks whose objective says to run or test what they changed — `control-edit`,
`control-test`, `refactor-three`, `recover-test` — carry
`requires_agent_verification`, and task success then needs one
`verification_passed` in the trace with `coverage=full` and
`proves_change=true`.

`runtime_verified` is the field, and *runtime* is the point: it covers
everything inside the run — the commands the model chose **and** the
project's own suite run deterministically by the harness, which is part of
what the complete agent does.  A run that has been shown a green suite is not
made to run it again to prove what was already observed.  What does not count
is the benchmark's hidden oracle, and it cannot: that runs out here, after the
fact, and never reaches the trace.

Both rates appear in `summary.md`, and the gap between them is runs that left
the right file behind without doing what was asked.  Every manifest row names
the verdict twice — `task_success` because that is what it is, and `success`
because that is what every reader and every earlier manifest already calls it.

### The fixture's tests must be runnable, and the command must be stated

They are `unittest.TestCase` files and every fixture carries `tests/__init__.py`,
so `python3 -m unittest discover` collects and runs them.  The harness *knows*
the command: it states it in the bash tool description, runs it itself after a
writing turn, and treats a run of it as the project's own test command rather
than guessing what it covered.  No turn is spent working the command out.

Three properties of that deterministic run, all of them load-bearing:

* **Only a configured command counts.**  A probed `make` is a good guess about
  how to build a tree and a bad basis for telling a turn it has been verified.
  Verification evidence is recorded only for a command the project declared in
  `projects.json` — or for the `tests/` shape, which the harness may *infer*
  only when asked: `SPEAR_INFER_TEST_COMMAND=1`, which the benchmark runner
  sets and a real tree does not.
* **Confined, or not at all.**  It runs through `CommandRunner` +
  `BubblewrapSandbox` with a 900 s ceiling of its own
  (`SPEAR_PROJECT_VERIFY_TIMEOUT`).  If the sandbox cannot be established the
  result is `not_run`: nothing is claimed, the turn is left *unjudged* rather
  than told its build is broken, and nothing falls back to the host.  Running
  it on the host is the operator's choice in advance —
  `SPEAR_PROJECT_VERIFY_ON_HOST=1` — never a fallback.
* **Once per `mutation_generation`.**  The gate is consulted up to six times in
  a turn and nothing can change between two consultations at the same
  generation, so the first answer stands until something is written — and it
  is recorded once, whichever consultation got there first.
* **Before any retry is decided.**  When a completion is considered with an
  unverified mutation, the configured suite runs, its evidence is recorded,
  and the completion is re-evaluated *first*.  `verification_nudge` and
  `retry(reason=unverified_change)` are emitted only if the answer is
  `failed`, `not_run` or still not enough.  The v4 run did it the other way
  round: fixed the file, ran a targeted test, was told to verify anyway, and
  the deterministic `build/full/passed` evidence appeared only after the extra
  actions — nine model calls to learn what the harness could observe for free.  They were bare
pytest functions, and pytest is not a dependency of this checkout: told to run
the test, a run reached for `python3 -c` — correctly refused — and finished
having verified nothing.  `python -m pytest` and `python -m unittest` are the
only interpreter invocations the command policy allows; everything else about
the interpreter stays refused.

### The acceptance tests belong to the task, not to the fixture

The fixture's own tests stay in the workspace — several tasks tell the model
to run or update them — but they decide nothing: a test the model can edit is
a test the model can make pass.  The oracle tests live in the task definition,
are written into a throwaway copy of the finished workspace under
`.spear-oracle/`, and the run never sees them.  They state the *requirement*,
so most of them check more than the fixture's test does.

pytest is not a dependency here, so the child runner imports each file and
calls its `test_*` functions.  A file that collects nothing is a failure, not
a pass.

The code they exercise was written by the model, so they run under the same
**Bubblewrap confinement and the same resource contract** as every other
command the harness runs (`CommandRunner` + `BubblewrapSandbox`, default
`ResourceLimits`/`CgroupLimits`, network closed).  A sandbox that cannot be
established is reported as `not_run` and is never a pass: an oracle that could
not be evaluated has decided nothing.

### Tasks that only report

`control-symbol`, `control-doc`, `call-chain`, `relevant-tests` and
`explorer-causal-tests` leave nothing behind in the workspace, so
`required_reads` alone cannot separate a run that understood the tree from one
that read it and answered wrongly.  Those carry an answer oracle: the child
appends its final answer to `SPEAR_BENCH_ANSWER_FILE` (`<run_id>.answer.txt`,
beside the trace) and the scorer matches the task's `expected_answer` /
`forbidden_answer` against it.  This is the one place a run's prose is scored,
and only for tasks whose entire product is prose.

### Isolation

`--single-root` is always passed to the child.  By default it declares every
registered corpus as an additional **writable** root and says so in its
prompt; a benchmark run was offering the model twenty-odd real repositories to
edit.  A run writes to its fixture and nowhere else, whatever the operator's
corpus registry holds.

Each run gets `SPEAR_STATE_DIR=<output>/state/<run_id>`, so everything the
session accumulates — history, trajectories, sessions, checkpoints, the audit
trail and the training corpus — stays in the run directory instead of the
operator's checkout.

The whole campaign also gets its own Chroma database at
`SPEAR_DB_PATH=<output>/chromadb`, never the production one.  Every fixture is
a fresh temporary directory, so its corpus tag — and with it its collection
name — is unique: a campaign pointed at the real database would leave one dead
collection in it per task per repetition.  Indexing the fixtures therefore
happens for real on every run, and that cost sits inside `duration_seconds`,
which is measured around the whole child process.  The first run of a campaign
additionally pays for creating the database, so read a slower first repetition
as that and not as the model.

Both stores, the suite version and the commit are written to
`<output>/campaign.json` beside the manifest, and every manifest row carries
`state_reference`, `db_reference` and `answer_reference`.

The machine configuration files (`active-embedder.conf`,
`active-embed-remote.conf`, `active-model.conf`) are read from the checkout,
not from the state directory, so a relocated state directory does not send the
embedder back to the local CPU.

Memory tasks are given real memories, written by the runner to
`<run_id>.memories.md` beside the trace and handed to the child through
`SPEAR_MEMORIES_FILE` — outside the workspace, because a memory the model can
`cat` measures nothing about memory selection.

