.. _capabilities:

External capabilities
#####################

A turn can be given operations that reach outside its workspace: an issue
tracker's search, a service's status, a note created somewhere else. SPEAR
calls these *external capabilities*. They come from *providers* the deployment
registers; the Model Context Protocol (MCP) is the first kind of provider.

SPEAR implements the part of MCP that tools need, over the stdio transport:
``initialize``, ``tools/list`` (following its pagination) and ``tools/call``,
in protocol version ``2025-06-18``. A server that answers with another version
is refused. Resources, server prompts, the HTTP transport and server
notifications are not supported.

Nothing is discovered: a provider SPEAR has not been told about does not exist
to it, and a registered provider applies only where its entry says.

Registering a provider
**********************

Providers are listed in ``capabilities.json`` beside ``projects.json``, or in
the file ``SPEAR_CAPABILITIES_FILE`` names. Like the project registry it is the
deployment's own file: commands and server names are not public material.

.. code-block:: json

   {"providers": [
     {"id": "tracker",
      "command": ["/opt/tracker-mcp/bin/tracker-mcp", "--readonly-token-file", "/etc/t"],
      "scope": "corpus firmware-a",
      "tasks": ["implementation", "mixed"],
      "read": ["search_issues", "get_issue"],
      "write": "refuse",
      "timeout": 10}
   ]}

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Key
     - Meaning
   * - ``id``
     - lower-case name; a capability is ``<id>/<tool>``
   * - ``command``
     - the server to start, as an argument list; it gets only ``PATH``,
       ``HOME`` and the variables ``env`` names
   * - ``scope``
     - where the provider applies, in the rule header's syntax
       (:ref:`context_selection`): ``global``, ``corpus <project>[, ...]`` or
       ``path <dir>``. Without one it applies nowhere
   * - ``tasks``
     - the request classes it serves: ``implementation``, ``mixed``,
       ``general`` (default: implementation and mixed)
   * - ``read``
     - the tools that only read. Every other tool is a WRITE, whatever the
       server says about it
   * - ``write``
     - ``refuse`` (default), ``confirm`` (asked through the session's
       permission mode) or ``allow``
   * - ``enabled``, ``timeout``
     - ``false`` to keep an entry without using it; seconds one request may take

What a turn is offered
**********************

A provider is a candidate like a rule: its scope and task classes decide which
turns it reaches, and the decision is audited with the others. A provider for
one project never reaches another, and an unregistered tree inherits none.
External capabilities reach implementation turns, general questions (read
capabilities only) and the implementation pass of a MIXED change -- never a
question about a bound standard, the normative pre-pass or the compliance
check.

The coding tools stay the six they are. A turn that may use external
capabilities is shown them in its context and runs them with one command
through its terminal, which SPEAR answers itself and never hands to a shell::

   spear-capability invoke <id> '<arguments as one JSON object>'

The command is recognised where a shell would run it: as the first word, or
after ``;``, ``&``, ``|``, a newline, ``(``, a backquote or ``$(``. Once
recognised it is answered by SPEAR -- with a result or a refusal -- whatever
is wrong with it, and nothing of it is run: it must stand alone on one line,
its words are split but never expanded, and its arguments are parsed as JSON.
The name anywhere else, such as an argument to ``grep``, is ordinary text.

A change in a standard-bound session that does not concern the standard runs
on SPEAR's earlier runtime, which has no capability gateway: such a turn is not
offered external capabilities, says so, and records it.

How much it is shown depends on the size of the family:

* up to eight capabilities, and about 1500 tokens of descriptions and schemas,
  are shown whole and can be invoked at once;
* a larger family is shown as an index -- one line per capability, with its
  READ or WRITE class -- and a capability's arguments are shown when the turn
  asks for them with ``spear-capability describe <id>``. One that has not been
  described cannot be invoked.

A call is checked before it reaches the provider: the capability must be one
this turn admits, its arguments must be ones its schema declares, and a WRITE
runs only if the provider's ``write`` policy and the session's permission mode
both allow it.

What a provider says is data
****************************

Names, descriptions, schemas and results all come from the external system.
Descriptions are shown as plain bounded text, and every result is framed as
external content. None of it can widen what the turn may do: an instruction in
a result to call some other tool finds no such capability, and nothing from a
provider is ever read as a rule, a standard or an approval.

A provider that cannot start, times out or fails is left out of the turn and
recorded; the task carries on without it. The audit trail records each
provider's registration, the family a turn was offered and how
(``capability_family_selected``, ``capability_index_exposed``), every
description and call, and every failure.
