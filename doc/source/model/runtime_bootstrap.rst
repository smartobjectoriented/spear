.. _runtime_bootstrap:

Rebuilding the runtime
======================

The server side of a SPEAR deployment is a directory, not a package:

.. code-block:: text

   <runtime>/
     llama.cpp/        a pinned revision of llama.cpp, built
     models/gguf/      the served weights
     embed/            an isolated virtualenv and its launcher
     config/           server.conf and gpu.conf
     log/              empty; whatever supervises the server owns the output

Nothing in it is in Git.  It is 85 GB of weights, a CUDA build tree and a
5 GB virtualenv — a repository that held them is a repository nobody can
clone.  So the contents are *described* in the checkout and reconstructed
on the machine:

.. code-block:: sh

   ./scripts/bootstrap-runtime.sh --root ~/spear-runtime

What makes that reconstruction meaningful is not the script.  It is
``server/runtime/manifest.json``.


The manifest is the contract
----------------------------

Every version the bootstrap installs comes from the manifest, and every one
of them is immutable:

============================  ==================================================
llama.cpp                     a full 40-character commit, never a branch
model shards                  exact byte sizes and the publisher's sha256
embedding stack               ``constraints-reds-tested.txt``, pinned
serving defaults              the numbers the profile was measured at
============================  ==================================================

The llama.cpp pin is the one worth dwelling on.  ``git clone`` with no
revision gives whatever ``master`` holds on the day it runs, so a runtime
rebuilt next year would serve from a different binary than the one that was
measured — and nothing would record that it had changed.  A moving reference
is not a reproducibility contract, and a test refuses one:
``test_llama_cpp_is_pinned_to_an_immutable_commit``.

The manifest describes software.  It never names a host, an account, a card
or a filesystem path.  Those arrive as arguments:

.. code-block:: sh

   ./scripts/bootstrap-runtime.sh --root /srv/spear-runtime \
       --gpu GPU-00000000-0000-0000-0000-000000000000

``--gpu`` (or ``SPEAR_GPU_UUID``) is written into ``config/gpu.conf`` on the
machine that owns the card.  A card is an allocation, granted and revoked by
whoever administers the host; it is not a property of the project.


What it does not rebuild
------------------------

A long-lived deployment accumulates sediment: rollback trees from past
migrations, evidence directories, staging areas, logs.  The manifest lists
these under ``layout.excluded`` and the bootstrap reconstructs none of them.
They are history, not runtime, and a bootstrap that recreated them would be
asserting that they matter.

It also touches nothing outside ``--root``.  No Chroma collection, no
training tree, no path the manifest does not name.


Reusing what already exists
---------------------------

The bootstrap orchestrates; it does not reimplement.

* the model download is ``server/scripts/fetch-model.sh`` — which already
  resolves a shard set from any one member's name and resumes with ``-c``;
* the build is ``server/inference/install-llamacpp.sh`` — which already
  picks a CUDA compiler new enough for the target architecture;
* the serving command is ``server/inference/serve.sh``.

That last one matters for verification.  ``--verify`` does not describe what
it thinks the command line would be: it runs the real launcher against a
stub binary that prints its own argv, so precedence, the mixture-of-experts
guess and the fixed flags are *exercised* rather than assumed.  This is the
same trick the deployment's own dry-run used, and it exists because three
implementations of that command line once disagreed — a running server at
98304 tokens of context against a restart script that would have brought it
back at 65536.


Idempotency
-----------

The expensive things are an 85 GB download and a 5 GB virtualenv, so the
property that matters is not "it can run twice" but "it recognises what is
already correct and leaves it alone".

Before replacing anything, the bootstrap asks whether what is there already
satisfies the manifest:

===================  =============================================================
llama.cpp            HEAD equals the pinned commit, **and** the built binary
                     reports that build number — the only evidence the checkout
                     was not moved after the build
model shards         each present at exactly the manifest's byte size
embedding venv       imports the stack and speaks the manifest's protocol version
configuration        present — and then left alone entirely
===================  =============================================================

Configuration is never overwritten.  ``server.conf`` is generated from the
manifest defaults the first time and is the deployment's thereafter: a
second run that reset an operator's context size to the profile's would be
exactly the drift this file exists to end.  Logs are never touched.


Integrity
---------

Size is the cheap check and runs every time.  It catches a truncated or
half-resumed download, which is the common failure, and it costs one
``stat`` per shard.

It does not catch corruption that preserves length.  For that the manifest
carries the publisher's sha256 for every shard, and:

.. code-block:: sh

   ./scripts/bootstrap-runtime.sh --root ~/spear-runtime --verify --checksum

reads all 85 GB and proves them.  It is off by default because hashing 85 GB
is not something a routine check should do, and on demand because "probably
fine" is not an integrity claim.

A shard that fails ``--checksum`` should be deleted before refetching: the
downloader resumes, which is right for an interrupted transfer and wrong for
a file that is the correct length and the wrong bytes.


Proving the embedder
--------------------

Importing the stack proves the virtualenv.  It does not prove the worker.

.. code-block:: sh

   ./scripts/bootstrap-runtime.sh --root ~/spear-runtime --verify --probe

runs the real launcher, sends one short text through the real wire protocol,
and checks that the answer has the dimension the profile says it must —
1024, for ``BAAI/bge-m3``.  Nothing is reimplemented: the request is built by
``server/embed/protocol.py``, the same module both ends already use.

It is opt-in because it loads the model, which on a cold cache is a
download.  On the deployment this profile was measured against it returns in
seconds.


The three modes
---------------

.. code-block:: sh

   ./scripts/bootstrap-runtime.sh --root PATH              # build it
   ./scripts/bootstrap-runtime.sh --root PATH --dry-run    # what would happen
   ./scripts/bootstrap-runtime.sh --root PATH --verify     # what is true now

``--dry-run`` and ``--verify`` write nothing whatsoever — asserted by
running them against a temporary root and comparing the directory tree
before and after, because a test that only read the source could not tell.

``--verify`` exits 0 when the runtime satisfies the manifest and 2 when it
does not, so it can gate a deployment.  ``--skip-model``, ``--skip-llama``
and ``--skip-embed`` narrow any of the three to the components you care
about.

A fresh ``--verify`` reports, in order: the layout, the llama.cpp revision
and build, every shard, the embedding venv and its protocol, the
configuration, and the full serving command line the launcher would produce.
