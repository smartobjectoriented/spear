"""What a session carries about the bound standard from turn to turn."""

import re
from normative import requirement_set
from standard import standard_scope
from standard.standard_commands import StandardCommandError
from cli.chat_settings import STANDARD_OPERATOR
from cli.terminal_ui import C_DIM, C_RST


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
    from runtime.agent_notes import is_write_request

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
