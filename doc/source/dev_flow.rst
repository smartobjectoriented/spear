.. _dev_flow:

Development flow
################

An overview of the flow used in SPEAR. It is deliberately short: the project
has one long-lived branch and no release ceremony, and pretending otherwise
in a document would not make it true.

Branches
********

* The **current** state is ``main``, on
  `github.com/smartobjectoriented/spear <https://github.com/smartobjectoriented/spear>`_.
* Work happens on a branch, never directly on ``main``.
* A finished change becomes a pull request. It is reviewed, then merged into
  ``main``.

The private counterpart — the corpus derived from a licensed standard,
customer material, evaluation matrices — lives in a separate repository on
``gitlab.edgemtech.ch``. The split is not a matter of taste: the public
repository can carry nothing confidential, so the private one does not have
to sort what is. Nothing from an engagement belongs in a commit here.

Commits
*******

A commit message says **why**, in prose. The subject line is imperative and
short — the median across the history is 53 characters, the longest 77 — and
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
it *isn't* doing. Roughly a thousand lines of the history are message bodies,
which is the ratio to aim for.

A commit that changes behaviour and a commit that renames things are two
commits. When one file carries both, stage only the change that belongs, and
commit the rest separately — the working tree is not the unit of work.

Tests
*****

The suite runs from the ``spear`` directory:

.. code-block:: console

    $ PYTHONPATH=. ./bin/python -m unittest discover -s tests -p "test_*.py"

Run the modules you touched while you work; run the whole suite before you
open the pull request, not during. See :ref:`testing <testing>` for what the
suite covers and, more usefully, what it deliberately does not.

Documentation
*************

Every project in this family uses the same layout: ``doc/Makefile``, sources
under ``doc/source/*.rst``, configuration in ``doc/source/conf.py``, output in
``doc/build/`` — which is generated, never edited and never indexed.

To check for warnings reliably, clean first:

.. code-block:: console

    $ cd doc && make clean && make html

An incremental build skips the consistency check, so existing warnings do not
reappear without ``make clean``. Continuous integration builds with ``-W
--keep-going``, which turns every warning into an error: an extension that
cannot reach its native tool only logs a warning and drops the node, so
without ``-W`` the job stays green while the diagrams are missing from the
published pages.
