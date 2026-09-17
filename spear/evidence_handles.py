"""Evidence handles belong to the turn that retrieved them.

A source id is a handle: `standard.fetch` takes one and returns the unit it
names. That is exactly what makes it dangerous to write down. The live tool
transcript is persisted into the conversation so a follow-up can refer back to
what happened -- and the transcript carries the handles, so a LATER turn can
read them and fetch that evidence again.

It did. Asked a fresh question about acknowledgements, a real session made no
search at all: it fetched five source ids it had found in the previous turn's
persisted transcript. The clauses it then reasoned from were the earlier
question's, and the central ones for the question actually being asked were
never retrieved. The guard withheld the answer, correctly, but by then the
turn had been answered from someone else's evidence.

So the transcript keeps its words and loses its handles. What a reader wants
from it -- which tool ran, what it was asked, roughly what came back -- is
prose and survives. What only a machine can use, and only to re-fetch stale
evidence, does not.

This is deliberately not a guard. Nothing here decides whether an answer is
grounded; it decides what a later turn is able to reach for.
"""

from __future__ import annotations

import re

#: A retrieval handle as the standard tools issue and accept one.
_HANDLE = re.compile(r"\bstd-[0-9a-f]{32}\b")

#: What replaces it. Says that evidence was read and that naming it again is
#: not how to read it, which is the whole of what a later turn should know.
REDACTED = "[a source read in an earlier turn]"


def redact(text):
    """The same transcript, with retrieval handles made unusable."""
    return _HANDLE.sub(REDACTED, text or "")


def handles_in(text):
    """Every retrieval handle a piece of text still carries."""
    return _HANDLE.findall(text or "")


# ── ownership ────────────────────────────────────────────────────────
#
# Redaction keeps handles out of transcripts written from now on. It does
# nothing for the transcripts already on disk, and a conversation that still
# carries live handles still hands them to the model: a replay of one real
# contaminated history reproduced the original failure exactly, down to the
# same wrong clause set.
#
# So the reading side is guarded too. A handle is usable only if THIS turn
# issued it -- if it came back from a retrieval the model actually performed
# now. Where a handle was copied from is then beside the point: an old
# transcript, a memory, a paste, or the model's own recollection all fail the
# same test, and the way through is the same in every case, which is to search.

#: The per-turn set, kept in the tool cache. That dict is created once per
#: turn by the runtime and is already how "already executed this turn" is
#: decided, so it is the scope that exists rather than a second one invented
#: for this.
_ISSUED = "standard_evidence_handles_issued"


def issued_this_turn(cache):
    """The handles this turn has put in front of the model."""
    if cache is None:
        return set()

    return cache.setdefault(_ISSUED, set())


def issue(cache, text):
    """Record every handle a standard tool result just showed the model.

    Taken from the RESULT TEXT, not from a list the caller assembles: what
    makes a handle fetchable is that the model saw it, so the thing to read is
    what the model was given -- search hits, a parent carried alongside a
    fetched unit, a companion reached by structural completion. One rule
    covers all of them and cannot fall behind a tool that starts returning one
    more id.
    """
    if cache is None:
        return set()

    found = set(handles_in(text))
    issued_this_turn(cache).update(found)

    return found


def owns(cache, handle):
    """Whether this turn may fetch that handle."""
    return handle in issued_this_turn(cache)
