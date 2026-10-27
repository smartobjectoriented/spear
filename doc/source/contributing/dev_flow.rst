.. _dev_flow:

Development flow
################

An overview of the flow used in SPEAR. It is deliberately short. Development
happens on ``main``; how a version is cut from it and maintained afterwards is
the subject of :ref:`the release process <release_process>`.

Branches
********

* The **current** state is ``main``, on
  `github.com/smartobjectoriented/spear <https://github.com/smartobjectoriented/spear>`_.
* Work happens on a branch, never directly on ``main``.
* A finished change becomes a pull request. It is reviewed, then merged into
  ``main``.

Material that cannot be public — a corpus derived from a licensed
standard, customer material, evaluation matrices — lives in a separate,
private repository. The split is not a matter of taste: the public repository
can carry nothing confidential, so the private one does not have to sort what
is. Nothing from an engagement belongs in a commit here.

Commits
*******

A commit message says **why**, in prose. The subject line is imperative and
short — the median across the history is 56 characters, the longest 80 — and
the body explains the decision: what was wrong before, what was measured, and
what was rejected.

.. code-block:: text

    Let a turn dereference only the evidence it retrieved itself

    A handle printed in an earlier turn stayed usable in every later one,
    so an answer could cite a clause this turn never read and be right by
    accident. [...]

This is not a style preference. The subject tells a reader scanning
``git log`` whether a change concerns them; the body is the only place the
reasoning survives, because the code that results from it cannot state what
it *isn't* doing. A body runs to well over ten lines on average across the
history, which is the ratio to aim for.

A commit that changes behaviour and a commit that renames things are two
commits. When one file carries both, stage only the change that belongs, and
commit the rest separately — the working tree is not the unit of work.

Tests
*****

Run the modules you touched while you work; run the whole suite before you
open the pull request, not during.  The commands, what the suite covers and,
more usefully, what it deliberately does not, are in :ref:`testing <testing>`.

Documentation
*************

A change that alters what a page states updates the page in the same pull
request.  How the documentation is built, checked and illustrated is in
:ref:`documentation`.
