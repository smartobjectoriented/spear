.. _backends:

Model backends
##############

SPEAR talks to a model through a provider-neutral interface. Everything below
the application layer — the runtime, the tool lifecycle, the guards — imports
no provider and behaves identically whichever backend is selected.

Two providers are supported:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - ``--provider``
     - Talks to
   * - ``openai-compatible``
     - any OpenAI-compatible chat-completions endpoint (the default)
   * - ``anthropic``
     - the Anthropic API

The OpenAI-compatible path covers a locally served model, a model served on
another machine, and hosted endpoints that speak the same protocol. What
changes between them is a URL.

Selecting a backend
*******************

Launched without a backend flag, an interactive session lists the configured
choices and preselects the one used last — remembered in
``spear/active-backend.conf``. A flag skips the picker; off a terminal, the
last choice is reused silently.

.. code-block:: console

   $ spear-chat --local                      # the locally served model
   $ spear-chat --remote                     # a remote endpoint over an SSH tunnel
   $ spear-chat --reds                       # the configured institute server
   $ spear-chat --provider anthropic --model claude-sonnet-5

The remote flags open an SSH tunnel and point the client at the local end of
it. They do **not** start anything on the far side: the server there is yours
to start.

Pointing at an endpoint directly
********************************

Any endpoint can be addressed without touching a configuration file:

.. code-block:: console

   $ spear-chat --api-base https://inference.example.org/v1 \
                --model qwen3 \
                --ctx 32768

or through the environment, which is what ``spear/machine.env`` is for:

.. code-block:: bash

   # spear/machine.env — untracked, this machine only
   export SPEAR_API_BASE="http://localhost:8080/v1"
   export SPEAR_MODEL_NAME="qwen3"
   export SPEAR_CTX=32768

.. note::

   ``SPEAR_MODEL_NAME`` matters for some servers and not others. A
   llama.cpp server ignores the model name in the request; a vLLM server
   matches it against its own ``--served-model-name`` and rejects a
   mismatch.

The context window
******************

``--ctx`` (``SPEAR_CTX``) tells the client how much room it has. It is not a
request to the server — it is what the client budgets against, and setting it
above what the server actually serves produces requests the server refuses
whole.

Where the endpoint reports its own window, the launcher reads it and uses that.

Two settings interact with it:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Effect
   * - ``--max-tokens`` (``SPEAR_MAX_TOKENS``)
     - cap on one reply; reserved out of the window so a long edit completes
       instead of being truncated mid-call
   * - ``SPEAR_CONTEXT_FRACTION``
     - the fraction of the window the prompt may occupy before compaction

Sampling
********

``--temp`` (``SPEAR_TEMP``) defaults to ``0.25``. A low temperature gives more
conservative, more consistent edits; a model that falls into repetition at a
low temperature wants it raised rather than lowered.

Recording and replaying
***********************

Two flags make a backend optional:

.. code-block:: console

   $ spear-chat --record turns.jsonl        # write down every model turn
   $ spear-chat --replay turns.jsonl        # answer from the recording

On replay, the tools, the files and the gates all run for real; only the model
is a recording. A change to the platform can therefore be judged in minutes and
at no inference cost. A turn that needs more rounds than were recorded simply
ends.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Setting
     - Meaning
   * - ``--api-base`` / ``SPEAR_API_BASE``
     - OpenAI-compatible endpoint URL
   * - ``--provider``
     - ``openai-compatible`` or ``anthropic``
   * - ``--model``
     - model id sent with the request
   * - ``--ctx`` / ``SPEAR_CTX``
     - context window in tokens (default 32768)
   * - ``--temp`` / ``SPEAR_TEMP``
     - sampling temperature (default 0.25)
   * - ``--max-tokens`` / ``SPEAR_MAX_TOKENS``
     - cap on one reply
   * - ``ANTHROPIC_API_KEY``
     - credential for the Anthropic provider

.. warning::

   Credentials belong in the environment or in an untracked
   ``spear/machine.env``, never in a tracked file.

.. seealso::

   :ref:`Configuration reference <configuration>` · :ref:`Model serving
   <model_serving>`
