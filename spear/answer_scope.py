"""What the CURRENT message asked for, and therefore what to go and look at.

A conversation accumulates subjects. Someone spends a morning on a codebase
and then asks a question about the standard it implements -- and the question
is answered with the codebase, because that is what the conversation has been
about. The reading was real and the identifiers were real, so nothing
downstream objects: the answer is grounded, it is just an answer to a larger
question than the one asked.

Measured on one real session: a self-contained normative question, asked after
implementation-heavy turns, drew nine to twenty-one shell calls over a
customer's source tree and came back describing functions nobody had asked
about.

So scope is read from the message in front of us. History still resolves what
the message MEANS -- "does that comply" is about whatever "that" was -- but it
does not add a second piece of work to a question that did not ask for one.
That distinction is the whole of this module: a turn inherits the previous
turn's subject only when it cannot be understood without it.

Enforcement is by what is offered rather than by instruction, the same way the
rest of the runtime handles this: for a turn that asked about the document, the
tools that read a working tree are simply not on the table, and they come back
the moment a turn asks about one.
"""

from __future__ import annotations

import re

from tool_registry import ToolCategory

#: What the turn is asking to be told about.
NORMATIVE = "NORMATIVE"
IMPLEMENTATION = "IMPLEMENTATION"
MIXED = "MIXED"
GENERAL = "GENERAL"

#: Something local: a working tree, its files, the things in them. Deliberately
#: ordinary words -- no project's own symbol names, which would make this
#: module true of one codebase and false of the next.
_LOCAL_SUBJECT = re.compile(
    r"\b(?:implementation|implementations|code|codebase|source|sources|"
    r"repository|repositories|repo|file|files|module|modules|function|"
    r"functions|method|methods|class|classes|script|scripts|program|"
    r"build|tests?|driver|firmware|encoder|decoder|parser|"
    r"implémentation|impl[ée]menter|fichiers?|fonctions?|"
    r"our\s+\w+|this\s+project|the\s+project)\b", re.I)

#: The document, as the thing being asked about.
_DOCUMENT_SUBJECT = re.compile(
    r"\b(?:standard|specification|spec|clause|rule|rules|provision|"
    r"provisions|requirement|requirements|normative|norme|exigences?)\b", re.I)

#: A message that cannot be read on its own. Only counted at the START of the
#: message, where a referent stands in for the thing just discussed: "does that
#: comply", "what about its error handling". Mid-sentence "that" is usually a
#: conjunction ("the rule that says ..."), which refers to nothing.
_REFERENT = re.compile(
    r"^\s*(?:and\s+|but\s+|so\s+|then\s+)?"
    r"(?:what\s+about|how\s+about|does\s+(?:that|it|this)|"
    r"is\s+(?:that|it|this)|are\s+(?:those|these)|"
    r"(?:that|it|this|these|those|its|their)\b)", re.I)

#: Asking for something to be changed is asking about a working tree, whatever
#: else the sentence mentions.
_CHANGE = re.compile(
    r"\b(?:implement|implements?|add|change|fix|modify|update|refactor|"
    r"rewrite|patch|correct|migrate|rename|delete|remove)\b", re.I)

#: Tools that reach a working tree. By category, so a tool added later is
#: covered by what it is rather than by being remembered here.
LOCAL_CATEGORIES = frozenset({ToolCategory.COMMAND, ToolCategory.FILE_WRITE})


def of(message, *, prior=None, standard_bound=False):
    """The scope of this turn, from its own words.

    `prior` is the previous turn's scope, consulted only where this message
    refers back to something instead of naming it. `standard_bound` says a
    document is bound to the session, which is what makes a question that
    names no subject at all a question about that document -- the same
    principle the normative policy already works on, where activation is the
    binding and never the phrasing.
    """
    text = (message or "").strip()
    local = bool(_LOCAL_SUBJECT.search(text)) or bool(_CHANGE.search(text))
    document = bool(_DOCUMENT_SUBJECT.search(text))
    refers_back = bool(prior and _REFERENT.match(text))

    # A message that points at the last turn is about whatever that was, even
    # when it also names the document: "does that comply with the standard"
    # asks about both, and the thing it points at is not in this sentence.
    if refers_back and prior in (IMPLEMENTATION, MIXED):
        return MIXED if document else prior

    if local and document:
        return MIXED

    if local:
        return IMPLEMENTATION

    if document:
        return NORMATIVE

    if refers_back:
        return prior

    # Names no subject of its own. With a document bound, that is a question
    # about the document; without one there is nothing for this to mean.
    return NORMATIVE if standard_bound else GENERAL


def withholds_local_tools(scope, *, standard_bound):
    """Whether this turn should be offered the tools that read a working tree.

    Only on a turn that asked about the document and nothing else, and only
    where a document is bound -- otherwise there is nothing for "normative" to
    mean and the session behaves as it always did.
    """
    return bool(standard_bound) and scope == NORMATIVE


def offered(tools, scope, *, standard_bound):
    """The tools this turn puts on the table."""
    if not withholds_local_tools(scope, standard_bound=standard_bound):
        return tools

    return type(tools)(tool for tool in tools
                       if getattr(tool, "category", None) not in LOCAL_CATEGORIES)


def operator_messages(conversation):
    """What the operator actually typed, oldest first.

    Harness-authored turns carry the user role because the transport has no
    other one, and reading them back as the request would let a nudge decide
    the scope of a question nobody asked.
    """
    said = []

    for message in conversation or ():
        if getattr(message, "role", None) != "user":
            continue

        if getattr(message, "authored_by", "operator") != "operator":
            continue

        parts = [getattr(block, "text", "") for block in
                 getattr(message, "content", ()) if getattr(block, "text", "")]

        if parts:
            said.append("\n".join(parts))

    return said


def prior_scope(conversation, *, standard_bound=False):
    """The scope of the turn before this one, for a message that refers back.

    Read from the conversation rather than remembered between turns: the
    transcript is already the record of what was asked, and a second copy of
    that record is a second thing to keep true.
    """
    said = operator_messages(conversation)

    if len(said) < 2:
        return None

    return of(said[-2], standard_bound=standard_bound)
