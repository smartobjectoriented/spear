.. _context:

Conversation context and history
################################

A session accumulates subjects. Someone spends a morning in a codebase and then
asks a question about the specification that codebase implements — and gets an
answer about the codebase, because that is what the conversation has been about.

The reading was real and the names were real, so nothing in the answer looks
wrong. It is simply an answer to a larger question than the one asked.

SPEAR draws one distinction to prevent this:

.. rubric:: The rule

**History for interpretation is not history for evidence.** Earlier turns
resolve what the current message *means*. They do not supply what the current
answer is *made of*.

Scope is read from the current message
**************************************

Every turn is classified from its own words:

.. list-table::
   :header-rows: 1
   :widths: 22 44 34

   * - Scope
     - The question
     - What is on the table
   * - ``NORMATIVE``
     - about the specification — what it defines, requires or means
     - the authoritative tools; the working-tree tools are withheld
   * - ``IMPLEMENTATION``
     - about this project's code
     - the ordinary tool set; no normative retrieval is forced
   * - ``MIXED``
     - about both — a comparison, a compliance question
     - both
   * - ``GENERAL``
     - neither, with no document bound
     - the ordinary tool set

"How does our parser handle this?" is an implementation question and is
answered from the code. "What does the specification require here?" is a
normative one and is answered from the document. "Does our parser comply?" asks
about both and gets both.

Self-contained and referring turns
**********************************

A message that names its own subject is *self-contained*. A message that points
at something instead of naming it — "does **that** comply", "what does **that**
function do" — needs what it points at, and keeps the whole conversation. That
is what makes the referring form work.

For a self-contained normative turn, the generation context is built from:

* the current question;
* the authoritative evidence retrieved for it;
* the conversation, as continuity rather than as material.

Concretely, prior exchanges that were about the project's own code are carried
as a marker saying an exchange happened and what kind it was, rather than as
their contents. The appended transcripts of what earlier turns' tools returned
are not carried at all: they are machine output, and a clause this turn needs
is a clause this turn fetches.

What was said about the *document* in earlier turns is carried whole. The
withholding is of the project's material, not of the conversation.

.. note::

   This is why a prior turn's function names do not appear in an answer about
   the specification — not because they are filtered out of the text, but
   because they were never in the turn's evidence to begin with. Filtering
   names out of a finished draft would hide the symptom; keeping them out of
   the context removes the cause.

Why prior implementation detail is not authoritative
****************************************************

Two independent reasons, and both hold:

#. it is not in the context of a self-contained normative turn;
#. it was never evidence — what grounds a name is what a tool returned *this*
   turn, and a normative turn is offered no code tools, so there is nothing to
   ground with.

A name a conversation carries was never a citation. It was a name somebody
mentioned.

Practical consequences
**********************

* Ask normative questions in a self-contained form. "Explain how X works" gets
  the document; "and what about that other thing we saw" gets the conversation.
* To compare, say so. A comparison is a MIXED question and is given both sides.
* ``/clear`` starts a fresh conversation when a session has drifted far from
  what you now want to ask.
* ``/history`` shows what the session is carrying.

.. seealso::

   :ref:`Authoritative standards <standards>` · :ref:`Evidence and guards
   <guards>`
