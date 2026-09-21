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
    r"provisions|requirement|requirements|normative|norme|exigences?|"
    # Measuring something AGAINST the document names the document as surely
    # as saying its name does: "does that comply" is a question about both
    # sides, and used to be a question about the implementation alone.
    r"compl(?:y|ies|iant|iance)|conform(?:s|ant|ance|e)?)\b", re.I)

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


#: The machine payload a finished turn appends to what the operator typed:
#: the transcript of what its tools returned. Conversation is what a person
#: said and what the model concluded; this is neither, and carrying it forward
#: hands a later turn the raw output of an earlier turn's reading as though it
#: were something said.
_TOOL_TRANSCRIPT = re.compile(r"\n\n\[Tools executed during this turn.*",
                              re.S)

#: What a withheld exchange is replaced by. It says that an exchange happened
#: and what kind of thing it was -- enough for the next message to read as the
#: next message in a conversation -- and it quotes nothing, so there is no
#: content in it for an answer to be built out of.
ELIDED_ASK = "[an earlier turn about this project's own code]"
ELIDED_REPLY = ("[the answer to it, not repeated here: it described this "
                "project's code and is not evidence about the document]")


def self_contained(message):
    """Whether this message can be read without the turn before it.

    The same referent test `of` uses, asked directly. A message that points at
    something instead of naming it needs what it points at; a message that
    names its own subject does not.
    """
    return not bool(_REFERENT.match((message or "").strip()))


def spoken_part(text):
    """What was actually said, without the appended tool transcript."""
    return _TOOL_TRANSCRIPT.sub("", text or "")


def withholds_history_payload(scope, *, standard_bound, self_contained=True):
    """Whether the prior turns' payload may reach this turn's generation.

    The same condition that takes the local tools off the table, with one
    addition: the message must stand on its own. A turn that points back at
    the last one is asking about the last one, and cannot be answered from a
    conversation it has been cut out of.
    """
    return (bool(standard_bound) and scope == NORMATIVE
            and bool(self_contained))


def carried(history, scope, *, standard_bound, self_contained=True):
    """The prior exchange, as far as THIS turn may use it.

    Withholding the local tools stopped a normative turn from GOING and
    reading a working tree. It did not stop it from using a working tree it
    had already been shown: the conversation is re-injected whole, and an
    earlier turn of implementation work carries its file paths, its symbols
    and the transcript of every command it ran. Measured on the session this
    exists for: a self-contained normative question, asked after a turn of
    code changes, produced a draft naming a function that occurs seven times
    in that turn's answer and nowhere in the document. No tool call had to be
    made for that, and none was.

    So the same distinction the rest of this module makes, applied to the
    generation context rather than to the tool table:

        history for interpretation is not history for evidence

    On a turn that stands on its own words and asks about the document, what
    is carried is the shape of the conversation and what was said in it about
    the document. What is withheld is the project's own material: the turns
    that were about its code, and the tool transcripts on every turn, which
    are machine output rather than anything a person said.

    Nothing here looks at an identifier or at any project's vocabulary. A turn
    is withheld for what it was ABOUT, which `of` already decides, and the
    referent it may be needed for is exactly the case this does not apply to.
    """
    if not withholds_history_payload(scope, standard_bound=standard_bound,
                                     self_contained=self_contained):
        return list(history or ())

    kept = []
    # Whether the exchange currently being read was about the project's code.
    # An assistant message belongs to the question it answered.
    local_exchange = False

    for message in history or ():
        role = message.get("role")
        text = message.get("content") or ""

        if role == "user":
            said = spoken_part(text)
            local_exchange = of(said, standard_bound=standard_bound) in (
                IMPLEMENTATION, MIXED)
            kept.append({"role": role,
                         "content": ELIDED_ASK if local_exchange else said})
        else:
            kept.append({"role": role,
                         "content": ELIDED_REPLY if local_exchange else text})

    return kept


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
