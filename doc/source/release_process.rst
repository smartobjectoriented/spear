.. _release_process:

Release process
###############

SPEAR follows the branch-per-release model of its sibling projects, SO3 first
among them: development happens on ``main``, and every minor version gets a
long-lived maintenance branch on which patch releases are tagged. A deployment
— a workstation, a GPU host, a container image — pins a line, not a commit,
and a fix to the line it runs can ship without everything ``main`` has gained
since.

Overview
********

Three git objects work together, and GitHub surfaces them in different places:

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Object
     - Example
     - Role
   * - ``main`` branch
     - ``main``
     - Continuous development (the next, unreleased version).
   * - ``release/vX.Y`` branch
     - ``release/v0.2``
     - Long-lived maintenance line for a minor version. Patch fixes land here
       and are tagged.
   * - Tag ``vX.Y.Z``
     - ``v0.2.0``
     - Immutable point marking a delivered version. Release candidates use the
       ``-rc`` suffix (``v0.2.1-rc``).
   * - GitHub Release
     - "SPEAR v0.2.0"
     - The release page (notes + assets), attached to a tag. Exactly one is
       flagged *Latest*; ``-rc`` tags are published as *pre-release*.

Versioning
**********

Versions follow semantic versioning ``vMAJOR.MINOR.PATCH``. What a deployment
relies on is SPEAR's operator interface — the command line of ``spear-chat``
and the other entry points, the ``projects.json`` registry, ``machine.env``
and the other configuration files, the layout of the state directory, and the
format of the standard store — so that is what the numbers are about:

* **MAJOR** — breaking changes to that interface: an option removed or with a
  new meaning, a configuration key renamed, a state or store format that an
  existing installation can no longer read.
* **MINOR** — new features, backward compatible (a new command, backend,
  guard, or configuration key with a default). Opens a new ``release/vX.Y``
  branch.
* **PATCH** — bug fixes only, cut *on* an existing ``release/vX.Y`` branch.

The served model is not part of the version: it is chosen per deployment
(``spear/active-model.conf``, ``machine.env``) and changes without a release.

Release candidates append ``-rc`` (optionally ``-rcN`` for successive
candidates), e.g. ``v0.3.0-rc``.

Which release is running?
*************************

Every entry point prints the release it belongs to when it starts, on stderr,
once per invocation — also when it runs another one (``spear-image`` runs the
scripts under ``scripts/docker``, ``spear-chat`` re-executes itself under
``systemd-run``)::

   [spear v0.2.0] spear-chat --reds --auto

The version comes from ``scripts/spearversion.sh``, which reads the git release
tag (``git describe``) and keeps only the base version: a tagged commit and
development on top of ``v0.2.0`` both report ``0.2.0``, while ``-rc`` is kept.
A tree without git metadata — a tarball, a container image — falls back to
the ``SPEAR_VERSION_FALLBACK`` constant of that script.
``spearversion.sh`` can also be run on its own. The documentation derives its
version from the same helper.

Branch layout
*************

::

   main ──●──●──●──●──●──●───────►   development (next version)
           \
   release/v0.2  ●──●──●             maintenance line for 0.2.x
                 │  │  └─ v0.2.1     (tags live on the branch)
                 │  └──── v0.2.1-rc
                 └─────── v0.2.0

While a minor line has not diverged from ``main`` yet (no work started on the
next minor), its ``release/vX.Y`` branch and ``main`` may point at the same
commit — that is expected.

Which fixes go on a release branch?
***********************************

Being a bug fix is *not* what sends a change to a release branch. ``main`` is
the continuous development line and carries both features **and** fixes as they
land; anything committed there simply ships in the next version. There are two
kinds of fix:

* **A fix for unreleased code, or one that can wait for the next version.**
  Nothing special — it is an ordinary commit on ``main`` and ships in the next
  ``vX.Y.0``. Do *not* touch any release branch.
* **A fix for an already-published version** (e.g. a guard that lets through
  what it should refuse in ``v0.2.0``) that must ship *before* the next
  version. Only this case uses the patch-release procedure below: the fix lands
  on ``release/v0.2`` (tagged ``v0.2.1``) and is also carried to ``main`` so it
  is not lost at the next minor.

In other words, what sends a change to ``release/vX.Y`` is the need to patch a
*live, already-released* version — never the mere fact that it is a fix rather
than a feature. A documentation-only change does not justify a patch release
either: it rides the next version.

Cutting a patch release (``vX.Y.Z``)
************************************

Patch fixes are committed **on the release branch**, then tagged. If a fix was
first merged into ``main``, cherry-pick it onto the branch rather than
fast-forwarding.

.. code-block:: sh

   git checkout release/v0.2
   git cherry-pick <sha>          # or commit the fix directly
   # bump SPEAR_VERSION_FALLBACK in scripts/spearversion.sh to 0.2.1

   # optional: publish a candidate first
   git tag -a v0.2.1-rc -m "spear v0.2.1-rc"
   git push origin release/v0.2 v0.2.1-rc
   gh release create v0.2.1-rc --title "SPEAR v0.2.1-rc" \
       --target release/v0.2 --prerelease --generate-notes

   # final release
   git tag -a v0.2.1 -m "spear v0.2.1"
   git push origin release/v0.2 v0.2.1
   gh release create v0.2.1 --title "SPEAR v0.2.1" \
       --target release/v0.2 --latest --notes-file <notes>

Cutting a new minor release (``vX.Y.0``)
****************************************

When ``main`` is ready for a new minor version, prepare it on ``main`` first (a
commit carrying the ``CHANGELOG`` entry, the *Maintained versions* table and
``SPEAR_VERSION_FALLBACK`` set to the version being cut), then branch off and
tag:

.. code-block:: sh

   git checkout main
   git checkout -b release/v0.3
   git push -u origin release/v0.3

   git tag -a v0.3.0 -m "spear v0.3.0"
   git push origin v0.3.0
   gh release create v0.3.0 --title "SPEAR v0.3.0" \
       --target release/v0.3 --latest --notes-file <notes>

The release notes are the version's ``CHANGELOG`` entry.

After tagging: propagate the release to ``main``
************************************************

A release is not finished when the tag is pushed. The references to the current
version that live on ``main`` must be bumped right after every release (they
drift silently otherwise):

* the *Maintained versions* table in ``README.md`` (the *Latest release*
  column of the line, and the *Status* column when a new minor line starts);
* a ``CHANGELOG`` entry summarizing the release (same content as the GitHub
  Release notes, kept in the repository for offline reference);
* ``SPEAR_VERSION_FALLBACK`` in ``scripts/spearversion.sh``, used only when the
  tree carries no git metadata.

Two version strings need **no** action: the entry points' release banner and
the documentation version both derive from the release tag
(``scripts/spearversion.sh``, reused by ``doc/source/conf.py``).

Before tagging: check the tree
******************************

The ``documentation`` workflow runs on every push to ``main`` and on pull
requests, and must be green on the commit the tag will point at::

   gh run list --workflow documentation --branch main

SPEAR has no test workflow of its own — the suite needs the project's virtual
environment and, for its sandbox tests, a host with bubblewrap and delegated
cgroups — so it is run by hand, on the commit being tagged, from ``spear/``:

.. code-block:: sh

   PYTHONPATH=. ./bin/python -m unittest discover -s tests -p "test_*.py"

It must pass in full; a test that only fails because an optional host (the
remote embedding server) is unreachable is re-run once that host is back, not
waved through. The strict documentation build must be clean as well:

.. code-block:: sh

   cd doc && make clean && \
       sphinx-build -W --keep-going -b html -d build/doctrees source build/html

Check ``SPEAR_VERSION_FALLBACK`` too: it must already read the version being
tagged, since the tagged tree is what a gitless copy reports. It appears twice
in this page on purpose — bump it **in the release commit set** so the tag is
right, and again when propagating to ``main`` so the next release does not
start a version behind.

Rules of thumb
**************

* One ``release/vX.Y`` branch **per minor**, not per patch — patches are tags
  *on* the branch.
* Tags are immutable: never move or delete a published ``vX.Y.Z`` tag. To
  correct a release, cut the next patch.
* Exactly one GitHub Release carries the *Latest* flag; every ``-rc`` Release is
  a *pre-release* so it never shadows the latest stable version.
* Once a ``release/vX.Y`` branch has diverged from ``main``, backport fixes with
  ``git cherry-pick`` — do not fast-forward the branch onto ``main``.
