.. _coding_conventions:

SPEAR coding conventions
########################

These are descriptions, not aspirations: each one was measured on the tree as
it stands, and the figure is given so that a later reader can check whether
it still holds rather than take it on trust. There is no linter configuration
in the repository — no ``.flake8``, no ``pyproject.toml``, no pre-commit hook
— which is deliberate. A convention that only a tool enforces is one nobody
has to understand.

Line length
***********

Seventy-nine columns. Of the 62 000 lines across the 110 modules, 95.1 % are
within it; the exceptions are almost all long string literals and table rows
where breaking would cost more than it buys.

Rationale: two files side by side on a laptop, and a diff in a terminal, both
fit without wrapping. Wrapped diffs are where review stops being careful.

Comments
********

Leave a blank line between a multi-line comment block and the statement it
introduces.

.. code-block:: python

    # The floor guarantees a way to work; it does not grant one the request
    # refused. A turn told not to edit keeps reading, searching and bash --
    # and bash itself runs without workspace write.

    if role == AgentRole.MAIN and not read_only:
        return cls._READ_FLOOR + cls._WRITE_FLOOR

The blank line is what makes the block read as a paragraph about the code
rather than as a label glued to one line.

Write comments about the *decision*, not about the mechanism. The mechanism
is visible in the code underneath; why it was chosen, and what went wrong
when it was not, is not. A comment that says what a line does earns nothing
and rots at the first edit; one that records the failure a line prevents
survives being moved.

Docstrings
**********

Triple double quotes, always: ``"""`` appears 2342 times in the tree,
``'''`` twenty. A module docstring says what the module is for in one
sentence; a function docstring says what it guarantees, not how.

Typing
******

``from __future__ import annotations`` at the top of every module — 105 of
the 110 have it — and return annotations wherever the return type is not
obvious from the name (1217 of them). The point is not type checking, which
nothing in CI runs; it is that a signature should answer "what comes back"
without the reader opening the body.

File headers
************

**An existing header is never rewritten.** Whoever it names stays named, in
the form it already has — ``Copyright (C) 2014-2017 Daniel Rossier``, ``REDS
Institute from HEIG-VD``, anything else. The only permitted change is the
year, and only when you actually changed the file: extend the range to the
current year (``2014-2017`` → ``2014-2026``), or turn a single year into a
range (``2017`` → ``2017-2026``).

Never replace the holder, never drop a name, never modernise the wording.
Attribution is a record of who did the work, and a record that gets tidied is
no longer a record.

Naming
******

Modules are nouns for what they hold (``tool_registry``, ``answer_scope``,
``evidence_handles``). A predicate returns a boolean and reads as one at the
call site — ``retains_raw_pdf(root)``, ``withholds_local_tools(scope)`` — so
that a condition can be read aloud.

Fail-closed, in code
********************

The rule that governs the harness governs its source too. A branch that
cannot apply a confinement raises; it does not log and continue. If you find
yourself writing a fallback that is *almost* as safe, the fallback is the
bug: see :ref:`the security model <security_model>` for what that costs when
it is got wrong.
