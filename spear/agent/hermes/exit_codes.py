# Copied from Hermes Agent (https://github.com/NousResearch/hermes-agent),
# tools/terminal_tool.py lines 2530-2670 at commit 0cbc6e37 (v0.21.0),
# unmodified below this header (the exit-code interpretation block).
# Copyright (c) 2025 Nous Research. MIT License -- see
# THIRD_PARTY_NOTICES.md at the repository root.

import re
import shlex

# Exit Code Context for Common CLI Tools
# =============================================================================
# Many Unix commands use non-zero exit codes for informational purposes, not
# to indicate failure.  The model sees a raw exit_code=1 from `grep` and
# wastes a turn investigating something that just means "no matches".
# This lookup adds a human-readable note so the agent can move on.

# Signal-death notes for the lethal signals seen in practice. Keyed by
# signum; used for both the ``-signum`` (subprocess) and ``128+signum``
# (shell) encodings. Curated rather than exhaustive so we never mislabel a
# legitimate application exit code (e.g. 130/SIGINT is handled by the
# executor's interrupt-marker path and excluded here).
_SIGNAL_EXIT_NOTES: dict[int, str] = {
    3:  "SIGQUIT (quit from keyboard)",
    4:  "SIGILL (illegal instruction — corrupt binary or wrong architecture)",
    6:  "SIGABRT (abort — assertion failure, fatal runtime error, or glibc abort)",
    7:  "SIGBUS (bus error — misaligned or unmapped memory access)",
    8:  "SIGFPE (fatal arithmetic error, e.g. integer division by zero)",
    9:  "SIGKILL — often the kernel OOM killer on memory exhaustion, "
        "or an explicit kill -9",
    11: "SIGSEGV (segmentation fault — the program crashed)",
    13: "SIGPIPE (wrote to a closed pipe — e.g. output piped to a reader that exited)",
    15: "SIGTERM (terminated — kill/timeout or shutdown requested it to stop)",
    24: "SIGXCPU (CPU time limit exceeded)",
    25: "SIGXFSZ (file size limit exceeded)",
}


def _interpret_signal_exit(exit_code: int) -> str | None:
    """Map signal-termination exit codes to a human-readable note.

    Returns None when ``exit_code`` does not look like a signal death.
    Negative codes are Python ``subprocess`` semantics (definite); codes in
    the 128+signum band are the shell convention (very likely but not
    guaranteed, so those notes hedge with "usually").
    """
    if exit_code < 0:
        signum = -exit_code
        if signum == 2:  # SIGINT — executor's interrupt-marker path owns it
            return None
        note = _SIGNAL_EXIT_NOTES.get(signum)
        if note:
            return f"Command terminated by signal {signum}: {note}"
        try:
            import signal as _signal
            name = _signal.Signals(signum).name
        except (ValueError, ImportError):
            name = f"signal {signum}"
        return f"Command terminated by {name} (signal {signum})"

    if exit_code > 128:
        signum = exit_code - 128
        note = _SIGNAL_EXIT_NOTES.get(signum)
        if note:
            return (
                f"Exit code {exit_code} usually means the command was "
                f"terminated by signal {signum}: {note}"
            )

    return None


def _interpret_exit_code(command: str, exit_code: int) -> str | None:
    """Return a human-readable note when a non-zero exit code is non-erroneous.

    Returns None when the exit code is 0 or genuinely signals an error.
    The note is appended to the tool result so the model doesn't waste
    turns investigating expected exit codes.
    """
    if exit_code == 0:
        return None

    # Signal terminations (ported from Kilo-Org/kilocode#12698, adapted to
    # Python semantics). Two shapes reach the model:
    #   * negative codes — subprocess.Popen reports a signal-killed process
    #     as ``-signum`` (definite signal death), and
    #   * 128+signum — the conventional shell encoding when bash reports a
    #     signal-killed child (heuristic: a program *can* ``exit 139``, so
    #     these notes say "usually").
    # Without a note the model sees a bare ``exit_code=-9`` or ``137`` and
    # burns turns re-running or mis-diagnosing (137 = OOM kill is the big
    # one). 130/SIGINT is deliberately absent: the executor has bespoke
    # interrupt-marker handling for rc=130.
    signal_note = _interpret_signal_exit(exit_code)
    if signal_note is not None:
        return signal_note

    # Extract the last command in a pipeline/chain — that determines the
    # exit code.  Handles  `cmd1 && cmd2`, `cmd1 | cmd2`, `cmd1; cmd2`.
    # Deliberately simple: split on shell operators and take the last piece.
    segments = re.split(r'\s*(?:\|\||&&|[|;])\s*', command)
    last_segment = (segments[-1] if segments else command).strip()

    # Get base command name (first word), stripping env var assignments
    # like  VAR=val cmd ...
    words = last_segment.split()
    base_cmd = ""
    for w in words:
        if "=" in w and not w.startswith("-"):
            continue  # skip VAR=val
        base_cmd = w.split("/")[-1]  # handle /usr/bin/grep -> grep
        break

    if not base_cmd:
        return None

    # Command-specific semantics
    semantics: dict[str, dict[int, str]] = {
        # grep/rg/ag/ack: 1=no matches found (normal), 2+=real error
        "grep":  {1: "No matches found (not an error)"},
        "egrep": {1: "No matches found (not an error)"},
        "fgrep": {1: "No matches found (not an error)"},
        "rg":    {1: "No matches found (not an error)"},
        "ag":    {1: "No matches found (not an error)"},
        "ack":   {1: "No matches found (not an error)"},
        # diff: 1=files differ (expected), 2+=real error
        "diff":  {1: "Files differ (expected, not an error)"},
        "colordiff": {1: "Files differ (expected, not an error)"},
        # find: 1=some dirs inaccessible but results may still be valid
        "find":  {1: "Some directories were inaccessible (partial results may still be valid)"},
        # test/[: 1=condition is false (expected)
        "test":  {1: "Condition evaluated to false (expected, not an error)"},
        "[":     {1: "Condition evaluated to false (expected, not an error)"},
        # curl: common non-error codes
        "curl":  {
            6: "Could not resolve host",
            7: "Failed to connect to host",
            22: "HTTP response code indicated error (e.g. 404, 500)",
            28: "Operation timed out",
        },
        # git: 1 is context-dependent but often normal (e.g. git diff with changes)
        "git":   {1: "Non-zero exit (often normal — e.g. 'git diff' returns 1 when files differ)"},
    }

    cmd_semantics = semantics.get(base_cmd)
    if cmd_semantics and exit_code in cmd_semantics:
        return cmd_semantics[exit_code]

    return None


