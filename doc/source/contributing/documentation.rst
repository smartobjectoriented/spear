.. _documentation:

Documentation
#############

How this manual is laid out, built, published and illustrated.

Layout
******

::

   doc/
     README                         the shared configuration and what differs
     Makefile                       make html | latexpdf | clean | …
     requirements.txt               Sphinx toolchain (local and CI)
     source/
       conf.py                      Sphinx configuration
       rstFlatTable.py              the ``flat-table`` directive
       index.rst                    the landing page
       <section>/*.rst              the chapters, one directory per section
       _static/theme_overrides.css  small readability overrides
       img/
         spear.drawio               every diagram, one page each (source)
         SPEAR-<Page>.drawio.png    the pages exported for the HTML build
         261008_SPEAR_Overview.png  the overview figure, a standalone image

``doc/build/`` is generated: never edit it, never index it.

The Sphinx configuration is the one shared by the sibling projects' manuals,
taken verbatim so they build and look the same; ``doc/README`` lists the few
places where SPEAR's differs.  Do not add project-specific styling or options.

Writing a page
**************

* Every page carries a label (``.. _name:`` above its title), so a reference
  to it can be a ``:ref:`` that breaks the build if its target disappears.
* Cross-references are absolute (``:doc:`/harness/sandbox```), and so are
  image paths (``/img/SPEAR-Layout.drawio.png``): a relative reference changes
  meaning with the directory of the page that writes it.
* ``index.rst`` follows the shared landing page: the logotype, one hidden
  numbered toctree, a centred title, and sections that list ``:ref:`` links.
  Prose belongs in the chapters.

Building
********

.. code-block:: console

   $ cd ~/spear/doc
   $ make html                 # -> build/html/index.html
   $ make latexpdf             # -> build/latex/SPEAR.pdf
   $ xdg-open build/html/index.html

Sphinx and the RTD theme come from the system Python (``/usr/bin/sphinx-build``),
not from the ``client/`` virtualenv, which carries only the application's
runtime dependencies.  ``doc/requirements.txt`` lists the toolchain for anyone
who prefers an isolated environment:

.. code-block:: console

   $ pip install -r doc/requirements.txt

The documentation imports no project module, so a Sphinx toolchain and a
checkout are all a build needs.

Warnings are errors
===================

An incremental build skips the consistency check, so existing warnings do not
reappear without a clean build.  The strict build — the one CI runs, and the
one a release must pass — is:

.. code-block:: console

   $ cd doc && make clean && \
         sphinx-build -W --keep-going -b html -d build/doctrees source build/html

``-W`` turns every warning into an error: an extension that cannot reach its
native tool only logs a warning and drops the node, so without ``-W`` the
build succeeds while the diagrams are missing from the pages.  ``--keep-going``
reports every warning of the run instead of stopping at the first.

Publishing
==========

``.github/workflows/docs.yml`` installs ``doc/requirements.txt``, runs the
strict build on every push to ``main`` and on every pull request, and
publishes the result to GitHub Pages from ``main`` only: the published site is
the documentation of the current ``main``, and a pull request builds the
documentation without publishing it.

The version string
==================

``conf.py`` takes ``release`` from ``scripts/spearversion.sh``, the helper
behind the entry points' release banner, so the documentation is never bumped
by hand (:ref:`release_process`).

.. _doc-diagrams:

Diagrams
********

``source/img/spear.drawio`` is the source of every diagram but the overview,
and is edited directly in draw.io or the VS Code extension.  Each page the
documentation uses is exported to ``SPEAR-<Page>.drawio.png`` (one file per
page, named after the page), and the exported PNG is committed next to the
``.drawio`` in the same change.  The overview figure of the introduction is a
standalone raster image, not exported from ``spear.drawio``, and is replaced as
a whole.

Export with the drawio snap.  The snap is confined: it cannot read ``/opt``
nor any hidden directory in ``$HOME`` (``~/.cache`` included), so the file is
staged under ``~/snap/drawio/common`` and the PNG copied back:

.. code-block:: console

   $ cd doc/source/img
   $ mkdir -p ~/snap/drawio/common/x && cp spear.drawio ~/snap/drawio/common/x/
   $ xvfb-run -a drawio -x -f png --scale 1.5 --border 10 -p 6 \
         -o ~/snap/drawio/common/x/SPEAR-Layout.drawio.png \
         ~/snap/drawio/common/x/spear.drawio --no-sandbox --disable-gpu
   $ cp ~/snap/drawio/common/x/SPEAR-Layout.drawio.png .

``-p`` is the 1-based page index, in the order of the pages in
``spear.drawio``:

.. list-table::
   :header-rows: 1
   :widths: 10 40 10 40

   * - ``-p``
     - Page
     - ``-p``
     - Page
   * - 1
     - Harness
     - 8
     - Architecture_dark
   * - 2
     - Security
     - 9
     - Workflow
   * - 3
     - Sandbox
     - 10
     - Evidence
   * - 4
     - Network
     - 11
     - Final_architecture
   * - 5
     - Cgroup
     - 12
     - Infrastructure
   * - 6
     - Layout
     - 13
     - Requests
   * - 7
     - Testing
     - 14
     - Mixed
   * - 15
     - Deployment
     - 
     - 

A page added or moved in ``spear.drawio`` shifts the indices after it.

A non-snap ``drawio`` exporting headless can drop the labels and render boxes
without their text; check the exported PNG before committing it.
