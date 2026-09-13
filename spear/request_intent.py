"""Whether the operator asked for the code to change — judged, not matched.

The harness decides a great deal from this one question. If the answer is
yes, a turn that writes nothing is sent back to the code, the tools are
narrowed to the ones that edit, and the build is checked afterwards. If the
answer is no, every one of those gates stays silent.

It used to be a regular expression over a list of verbs, and the list was
written by hand -- so it knew "make changes", "implement" and "corrige", and
did not know "adapt". A whole turn passed with every write-side gate
disarmed because the operator had written "can you adapt the code
accordingly". Each missing verb is a symptom, and the list can only ever
cover what somebody thought to type into it.

The distinction the rest of the harness rests on is between FACTS and
JUDGEMENTS. Whether the tree compiles, whether a file changed, how many
packets went out -- those are facts, and they are measured deterministically
because a deterministic measure of a fact is never wrong. Whether a sentence
of ordinary French or English asks for the code to change is not a fact
about the tree; it is a reading of what someone meant. That is what a
language model is for.

So the two are combined, and deliberately not symmetrically:

    a write request  =  the pattern says so  OR  the model says so

The pattern stays as the floor. What it recognises is recognised whatever
the model answers, so the harness keeps a deterministic core that no reply
can talk it out of -- and the model only ever ADDS readings the list missed.

The asymmetry is what makes this safe. A false positive costs almost
nothing: the write-side gates only fire when nothing has been written, so on
a genuine question they stay quiet anyway. A false negative costs a whole
turn, which is what was measured.
"""

from __future__ import annotations

SYSTEM = (
    "You classify one message from a software operator to a coding "
    "assistant. Answer with exactly one word: WRITE or ASK.\n"
    "WRITE — the operator wants the code changed, fixed, adapted, extended "
    "or removed, however they phrase it, including indirectly ('this should "
    "follow the rule', 'that is wrong').\n"
    "ASK — the operator wants an explanation, an opinion, a review or an "
    "answer, and no change to the files."
)

# What the model may answer and be believed. Anything else is treated as no
# answer at all: a classifier that has to be interpreted is a second guess.
_YES = "write"
_NO = "ask"


def _verdict(text):
    """WRITE, ASK, or None when the reply is neither."""
    first = (text or "").strip().split()

    if not first:
        return None

    word = first[0].strip(".,:;!?*_`\"'").casefold()

    return True if word == _YES else False if word == _NO else None


def judge(backend, question, *, budget_manager=None, budget_kind=None):
    """Does this message ask for the code to change? None when unknown.

    None is a real answer and the caller must keep it distinct from False:
    it means nothing was learned -- the provider failed, the reply was not
    one of the two words -- and the caller should fall back to the pattern
    rather than conclude the operator asked for nothing.
    """
    if backend is None or not (question or "").strip():
        return None

    from model_backend import ConversationMessage, TextBlock

    if budget_manager is not None and budget_kind is not None:
        try:
            budget_manager.consume(budget_kind)
        except Exception:
            # A classification is not worth failing a turn over, and a spent
            # auxiliary budget is a good enough reason to fall back.
            return None

    try:
        turn = backend.complete(
            system=SYSTEM,
            conversation=(ConversationMessage(
                "user", (TextBlock(question[:2000]),)),),
            tools=(),
            use_tools=False,
        )
    except Exception:
        return None

    if getattr(turn, "error", None):
        return None

    return _verdict(getattr(turn, "text", ""))


def resolve(pattern_says, model_says) -> bool:
    """The floor, raised by the reading.

    Never lowered: a message the pattern recognises stays recognised even if
    the model disagrees, so the deterministic core cannot be argued away.
    """
    return bool(pattern_says) or model_says is True
