"""What the harness records at usage time, in the shape the trainers consume.

Two halves of the pipeline were built at different times and never met. On one
side `rag_chat` appends a row per turn to trajectories.jsonl -- the question,
the tool calls with their arguments and results, the answer, and now a verdict
the project's own build and tests produced. On the other side sft_dataset and
preference_dataset consume canonical FT0 episodes. Nothing turned the first
into the second, so months of recorded usage sat in a file no trainer reads.

This is that turn. It is a pure function of a recorded row: no model is asked
anything, no file in the tree is consulted, and a row that cannot be converted
faithfully is refused rather than guessed at.

WHAT A ROW BECOMES

An episode with one turn per assistant emission. Each turn carries the context
as it stood BEFORE that emission -- never the whole conversation, or a sample
would train on hindsight the model did not have. A row with three tool calls
and an answer becomes four turns, and only the last is a primary candidate:
it is the one that answered.

THE VERDICT BECOMES ELIGIBILITY, AND NOT AUTOMATICALLY

    pass     -> positive_candidate   the tree built and its tests passed
    fail     -> failure_trajectory   it did not; kept, because a corpus of
                                     successes cannot teach what to stop doing
    unrated  -> incomplete           nothing judged it, which is not the same
                                     as judging it wrong

That third mapping is the one that matters. Until the harness produced a
verdict of its own, every turn was "unrated", and treating those as failures
would have taught the model that unverified work is bad work rather than
simply unmeasured.
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

from training_data import (
    TRAINING_SCHEMA_VERSION, TrainingEligibility, TrainingEpisode,
    TrainingProvenance, TrainingToolCall, TrainingTurn,
)


class TrajectoryConversionError(RuntimeError):
    pass


# Recorded verdicts, and what each one licenses.
ELIGIBILITY_BY_VERDICT = {
    "pass": TrainingEligibility.POSITIVE_CANDIDATE,
    "fail": TrainingEligibility.FAILURE_TRAJECTORY,
    "unrated": TrainingEligibility.INCOMPLETE,
}

# How much of a tool result is carried. The recorder already truncates at
# 4000 characters; this is the second bound, applied where the trainer reads,
# so a long result cannot dominate a sample's context.
RESULT_LIMIT = 4000


def _text_block(text):
    return {"type": "text", "text": text}


def _canonical_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _hash(value) -> str:
    """The tool-view digest, byte for byte what the SFT builder recomputes.

    It re-hashes every snapshot and refuses the episode when a key does not
    match its value, which is the check that caught this converter's first
    attempt. Duplicated here rather than imported so the recorder does not
    depend on the exporter -- but it must not drift, and the test asserts it
    against sft_dataset's own function.
    """
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _short(value) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()[:16]


def _tool_view(names, schemas):
    """The snapshot of the tools this turn could see, and its digest.

    A recorded row names the tools it called and nothing more: descriptions
    and argument schemas belong to the harness version, not to the turn. A
    caller holding the registry passes them and the snapshot is faithful;
    without them the names still carry, with empty schemas that validate
    vacuously rather than falsely.
    """
    schemas = schemas or {}
    snapshot = {"tools": [{
        "name": name,
        "description": str((schemas.get(name) or {}).get("description", "")),
        "input_schema": dict((schemas.get(name) or {}).get("input_schema", {})),
    } for name in sorted(set(names))]}

    return _hash(snapshot), snapshot


def _steps(row) -> list[Mapping[str, object]]:
    steps = row.get("steps")

    if steps is None:
        return []

    if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence):
        raise TrajectoryConversionError("trajectory steps must be an array")

    for step in steps:
        if not isinstance(step, Mapping) or not step.get("tool"):
            raise TrajectoryConversionError("a trajectory step must name a tool")

    return list(steps)


def episode_id_for(row, index=0) -> str:
    """A stable identifier for a recorded row.

    From the row's content, so converting the same file twice writes the same
    episodes rather than duplicating them. The task id alone will not do: a
    session answers many turns under one task.
    """
    task = str(row.get("task_id") or "turn")
    stamp = _short([row.get("question"), row.get("answer"),
                     [step.get("action_id") for step in _steps(row)], index])
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in task)[:40]

    return f"{safe}-{stamp}"


def episode_from_row(row, *, system="", session_id=None, timestamp=None,
                     index=0, tool_schemas=None) -> TrainingEpisode:
    """One recorded turn, as a canonical training episode."""
    if not isinstance(row, Mapping):
        raise TrajectoryConversionError("a trajectory row must be an object")

    question = row.get("question")
    answer = row.get("answer")

    if not isinstance(question, str) or not question.strip():
        raise TrajectoryConversionError("a trajectory row must carry a question")

    if not isinstance(answer, str):
        raise TrajectoryConversionError("a trajectory row must carry an answer")

    verdict = str(row.get("verdict") or "unrated")
    eligibility = ELIGIBILITY_BY_VERDICT.get(verdict)

    if eligibility is None:
        raise TrajectoryConversionError(f"unknown verdict: {verdict!r}")

    steps = _steps(row)
    episode_id = episode_id_for(row, index)
    task_id = str(row.get("task_id") or episode_id)
    # The turn's own time, never the ingestion's. Stamping `now()` here made
    # the same row convert to different content each time, and the store --
    # correctly -- refused the second as a conflicting episode under the same
    # identifier. A row recorded before the recorder stamped anything carries
    # no time, and says so rather than borrowing today's.
    when = timestamp or str(row.get("timestamp") or "")

    # The conversation as it grows. `messages` on a turn is this list as it
    # stood before that turn spoke, which is what the model actually saw.
    seen: list[Mapping[str, object]] = [
        {"role": "user", "content": (_text_block(question),)},
    ]
    turns: list[TrainingTurn] = []
    names = [str(step.get("tool")) for step in steps]

    # One view for the whole row: the tools it could reach did not change
    # between its calls, and the builder indexes turns into this table.
    view_hash, snapshot = _tool_view(names, tool_schemas)

    for position, step in enumerate(steps):
        call_id = str(step.get("action_id") or f"call_{position}")
        name = str(step.get("tool"))
        arguments = step.get("arguments")
        arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
        result = str(step.get("result") or "")[:RESULT_LIMIT]

        turns.append(TrainingTurn(
            turn_id=f"{episode_id}-{position}",
            episode_id=episode_id,
            task_id=task_id,
            session_id=session_id,
            turn_index=position,
            role="assistant",
            purpose="tool_action",
            primary_candidate=False,
            system=system,
            messages=tuple(seen),
            tool_view_hash=view_hash,
            tool_names=tuple(sorted(set(names))),
            tools_enabled=True,
            assistant_output="",
            stop_reason="tool_use",
            tool_calls=(TrainingToolCall(
                tool_call_id=call_id, name=name, arguments=arguments,
                action_id=step.get("action_id"),
                status=step.get("status"),
                model_content=result,
                result_reference=step.get("result_reference"),
                provenance=TrainingProvenance.TOOL_OUTPUT.value,
            ),),
        ))

        seen.append({"role": "assistant", "content": (
            {"type": "tool_use", "id": call_id, "name": name,
             "arguments": arguments},)})
        seen.append({"role": "user", "content": (
            {"type": "tool_result", "tool_call_id": call_id,
             "content": result},)})

    # The turn that answered. It is the only primary candidate: the tool
    # turns are how it got there, and the SFT builder selects one target.
    turns.append(TrainingTurn(
        turn_id=f"{episode_id}-{len(steps)}",
        episode_id=episode_id,
        task_id=task_id,
        session_id=session_id,
        turn_index=len(steps),
        role="assistant",
        purpose="final_response",
        primary_candidate=True,
        system=system,
        messages=tuple(seen),
        tool_view_hash=view_hash,
        tool_names=tuple(sorted(set(names))),
        tools_enabled=bool(names),
        assistant_output=answer,
        stop_reason="end_turn",
    ))

    return TrainingEpisode(
        schema_version=TRAINING_SCHEMA_VERSION,
        episode_id=episode_id,
        task_id=task_id,
        session_id=session_id,
        timestamp=when,
        provenance={"origin": "NORMAL_USAGE",
                    "recorder": "trajectories.jsonl",
                    "source": row.get("source") or "",
                    "project": row.get("project") or ""},
        task={"objective": question, "project": row.get("project") or "",
              "bench": row.get("bench") or ""},
        turns=tuple(turns),
        tool_views={view_hash: snapshot},
        execution={"tool_calls": len(steps)},
        outcome={"final_response": answer,
                 "verdict": verdict,
                 "terminal_task_status": "completed",
                 "verification_status": {"pass": "passed", "fail": "failed"}
                 .get(verdict, "unverified")},
        training_metadata={"eligibility": eligibility.value,
                           "verdict": verdict,
                           "reasons": () if verdict == "pass"
                           else ("project build or tests failed",)
                           if verdict == "fail"
                           else ("nothing judged this turn",)},
    )


def rows_from_file(path):
    """Every parseable row of a trajectories.jsonl, in order.

    Rows rather than episodes, because what happens between the two -- the
    promotion policy's redaction -- has to see the row before it is shaped.
    Yields (row, error) with exactly one of the two set; a malformed line is
    skipped rather than fatal, since the file is appended to by a live
    session and one bad line must not cost the rest of the history.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue

            try:
                yield json.loads(line), None
            except ValueError as exc:
                yield None, f"line {index + 1}: {exc}"


def episodes_from_file(path, *, system="", session_id=None, tool_schemas=None):
    """Every convertible row of a trajectories.jsonl, in order.

    A malformed row is skipped rather than fatal: the file is appended to by a
    live session, and one bad line must not cost the rest of the history.
    Yields (episode, error) with exactly one of the two set.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
                yield episode_from_row(row, system=system,
                                       session_id=session_id, index=index,
                                       tool_schemas=tool_schemas), None
            except (ValueError, TrajectoryConversionError) as exc:
                yield None, f"line {index + 1}: {exc}"
