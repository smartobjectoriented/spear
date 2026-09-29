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

That asymmetry was argued as safe because "a false positive costs almost
nothing". It does not. Asked "Could we have this artifact in images/ instead
of board/<target>/?" -- a feasibility question, no verb of the list in it --
the model read WRITE, the harness then told the turn "the user asked you to
change the code, this turn is not finished until a file changes", and the
turn edited nine files in a tree whose owner had asked a question.

So there is a second deterministic floor, the other way: a message whose
sentences are ADVISORY -- could we, would it be possible, how could, should
we, what if -- and none of which asks for the change, is a question. It is
settled before the model is consulted, and the model cannot raise it. A
follow-up that accepts a proposal ("yes, do it", "go ahead") is a write in
its own right; the question before it never was one. When in doubt, read.
"""

from __future__ import annotations

import re

SYSTEM = (
    "You classify one message from a software operator to a coding "
    "assistant. Answer with exactly one word: WRITE or ASK.\n"
    "WRITE — the operator wants the code changed, fixed, adapted, extended "
    "or removed, however they phrase it, including indirectly ('this should "
    "follow the rule', 'that is wrong').\n"
    "ASK — the operator wants an explanation, an opinion, a review or an "
    "answer, and no change to the files. A question about whether or how "
    "something COULD be changed ('could we move X?', 'would it be possible "
    "to ...?', 'how could we ...?') is ASK: it asks for feasibility, not for "
    "the change. If unsure, answer ASK."
)

# ---------------------------------------------------------------------------
# The deterministic reading of the sentence, before any model is asked.

# A sentence that deliberates about a change rather than asking for it. The
# subject matters: "could WE move it?" weighs an option, "could YOU move it?"
# is a polite imperative and is not listed here.
_ADVISORY_OPENER = re.compile(
    r"^\W*(?:"
    r"(?:and|so|but|then|also|ok|okay|hmm|well)\W+)?(?:"
    r"(?:could|can|should|shall|may|might|would|must)\s+(?:we|i|one|it|this|"
    r"that|these|those|there|the\b)"
    r"|(?:would|will|is|was)\s+it\s+(?:be\s+)?(?:possible|feasible|better|"
    r"wise|reasonable|a\s+good\s+idea|make\s+sense|sensible|worth)"
    r"|(?:does|would|will)\s+it\s+make\s+sense"
    r"|is\s+there\s+(?:a|any)\s+(?:way|reason|chance|option)"
    r"|how\s+(?:could|can|would|should|might|shall)\s+(?:we|i|one|this|"
    r"that|it|these|those)\b"
    r"|how\s+(?:hard|difficult|easy|risky|much\s+work)\b"
    r"|what\s+(?:would|could|if|about|should\s+we|do\s+we\s+need|"
    r"(?:is|are)\s+(?:needed|required)\s+to|(?:would|does)\s+it\s+take)\b"
    r"|why\s+(?:not|don'?t\s+we|do\s+we)\b"
    r"|(?:i\s+)?wonder(?:ing)?\s+(?:if|whether)"
    r"|(?:pourrait|peut|devrait|faudrait)[-\s](?:on|il)\b"
    r"|(?:serait|est)[-\s]il\s+(?:possible|envisageable|faisable|utile)"
    r"|est[-\s]ce\s+qu"
    r"|comment\s+(?:pourrait|peut|faudrait|faire)"
    r"|et\s+si\b"
    r")", re.I)

# Accepting what the previous answer proposed. A write by itself: nothing in
# "yes, do it" is a question, and the proposal it accepts was the question.
_FOLLOW_UP = re.compile(
    r"^\W*(?:(?:yes|yep|yeah|ok(?:ay)?|sure|fine|good|great|right|oui|"
    r"d'accord|parfait|ok\s+go)\b[\s,.!]*)*(?:please\s+|s'il\s+te\s+pla[iî]t\s+)?"
    r"(?:(?:do|apply|make|implement|go\s+(?:for|with))\s+(?:it|that|this|them|"
    r"those|these|so)\b"
    r"|go\s+ahead|proceed|let'?s\s+(?:do|go)|ship\s+it|make\s+it\s+so"
    r"|vas[-\s]y|allez[-\s]y|fais[-\s](?:le|la|les)|applique[-\s](?:le|la)"
    r"|on\s+y\s+va)\b"
    r"|^\W*(?:yes|yep|yeah|oui)(?:\s+please|\s+s'il\s+te\s+pla[iî]t)?\W*$",
    re.I)

# A polite imperative: the assistant is asked to act, however softly.
_POLITE_REQUEST = re.compile(
    r"^\W*(?:please\s+)?(?:could|can|would|will)\s+you\s+(?:please\s+)?",
    re.I)

_SENTENCE = re.compile(r"[^.?!\n]+(?:[.?!]+|$)")


def sentences(text):
    """The message cut into sentences; a question keeps its mark."""
    return [item.strip() for item in _SENTENCE.findall(text or "")
            if item.strip(" \t.?!")]


def _advisory_sentence(sentence):
    return bool(_ADVISORY_OPENER.search(sentence))


def follow_up(text):
    """Does the message accept what the previous answer proposed?"""
    return any(_FOLLOW_UP.search(item) for item in sentences(text))


def mutation_intent(text, write_pattern, *, answered=False):
    """"write", "advisory" or "unknown", from the sentences alone.

    write      a sentence asks for the change: an imperative the pattern
               knows, a polite "could you <change>", or -- when there is a
               previous answer to accept (`answered`) -- an acceptance.
    advisory   no sentence asks for it, and at least one deliberates about
               it. Settled: nothing may raise it to a write.
    unknown    neither -- the old reading applies (pattern, then model).

    A sentence only counts toward "write" when it is not itself advisory:
    "Could we add a flag?" contains `add` and asks nothing to be added.
    """
    items = sentences(text)
    advisory = False

    for item in items:
        # "do it" with nothing before it accepts nothing.
        if answered and _FOLLOW_UP.search(item):
            return "write"

        polite = _POLITE_REQUEST.match(item)

        if polite:
            if write_pattern.search(item[polite.end():]):
                return "write"

            continue

        if _advisory_sentence(item):
            advisory = True
            continue

        if write_pattern.search(item):
            return "write"

    return "advisory" if advisory else "unknown"


def advisory(text, write_pattern):
    """A question about a change, and not a request for one."""
    return mutation_intent(text, write_pattern) == "advisory"


def answered(conversation):
    """Did an answer precede the operator's latest message?

    What makes "yes, do it" an instruction: there is a proposal to accept.
    """
    seen_operator = False

    for message in reversed(list(conversation or ())):
        role = getattr(message, "role", None)

        # Tool results travel as user messages too; only text the operator
        # wrote is a turn of theirs.
        if role == "user" and getattr(message, "authored_by",
                                      "operator") == "operator" and any(
                getattr(block, "text", "")
                for block in getattr(message, "content", ()) or ()):
            if seen_operator:
                return False

            seen_operator = True
        elif role == "assistant" and seen_operator:
            return True

    return False

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
