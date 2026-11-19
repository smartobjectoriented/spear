.. _backends:

Model backends
##############

SPEAR talks to a model through a provider-neutral interface. Everything below
the application layer — the runtime, the tool lifecycle, the guards — imports
no provider and behaves identically whichever backend is selected. This page
is the client side: which endpoint the client talks to, on which port, and
what it sends. The server behind the endpoint is :ref:`model_serving`.

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

.. note::

   The coding core needs an OpenAI-compatible endpoint: it keeps its own
   OpenAI-format history and calls the endpoint directly. The Anthropic
   backend serves normative and general questions only.

Selecting a backend
*******************

Four backends, mutually exclusive, are chosen at launch. Launched without a
backend flag, an interactive session lists them and preselects the one used
last — remembered in ``active-backend.conf`` in the state directory
(:doc:`/operations/directory_layout`). A flag skips the picker; off a
terminal, the last choice is reused silently. Listing never probes — no SSH,
no ``curl`` — so a server that is down is still selectable.

.. code-block:: console

   $ spear-chat --local                      # the locally served model
   $ spear-chat --remote                     # a rented pod over an SSH tunnel
   $ spear-chat --reds                       # a configured shared GPU server
   $ spear-chat --provider anthropic --model claude-sonnet-5

.. list-table::
   :header-rows: 1
   :widths: 16 14 34 36

   * - Backend
     - Flag
     - Endpoint
     - Prepared by the launcher
   * - ``local``
     - ``--local``
     - ``127.0.0.1:8080``
     - starts ``llama-server`` if absent
   * - ``remote``
     - ``--remote``
     - ``127.0.0.1:8081`` → pod
     - opens the tunnel, uploads and starts the serve script
   * - ``reds``
     - ``--reds``
     - ``127.0.0.1:8082`` → REDS
     - **tunnel only** — see below
   * - ``anthropic``
     - ``--provider anthropic``
     - api.anthropic.com
     - nothing local

``reds`` is deliberately not the pod path. The pod is ephemeral — RunPod
reassigns its SSH port on every restart, which is why that branch re-prompts
for host and port, uploads a serve script and starts vLLM over ``tmux``. The
REDS server is a stable host, so none of that applies, and the launcher
**installs, uploads and starts nothing there**: it opens the tunnel, checks
that something answers, and otherwise says so and stops. Start the inference
server on that machine yourself. ``reds.conf`` carries the host (an
``~/.ssh/config`` alias is preferred — it already holds the user, key and auth
quirks), the remote port (``SPEAR_SERVER_PORT`` on that host; the profile uses
8010) and the model id to advertise.

.. _backend_ports:

Ports and endpoints
*******************

The local ports differ so a laptop model, the pod tunnel and the REDS tunnel
can be up at once.

.. list-table::
   :header-rows: 1
   :widths: 10 34 56

   * - Port
     - What answers there
     - Where it is set
   * - 8080
     - on the client machine, the ``llama-server`` that ``spear-chat
       --local`` starts or reuses
     - ``client/spear-chat.sh`` (``SPEAR_API_BASE`` and the
       ``SPEAR_SERVER_PORT`` it passes to ``serve.sh``); also the default
       ``SPEAR_API_BASE`` of ``client/cli/chat_settings.py`` and the port
       fallback of ``server/inference/serve.sh``
   * - 8081
     - on the client machine, the local end of the tunnel to a rented pod,
       whose vLLM listens on 8080
     - ``client/spear-chat.sh``, ``--remote`` branch
   * - 8082
     - on the client machine, the local end of the SSH tunnel to the
       inference host
     - ``client/spear-chat.sh``, ``--reds`` branch;
       ``scripts/docker/spear-docker.sh`` (default ``SPEAR_API_BASE``, and
       its own ``--reds`` tunnel)
   * - 8010
     - on the inference host, ``llama-server`` bound to ``127.0.0.1``,
       reached only through the tunnel
     - host side: ``SPEAR_SERVER_PORT`` in ``<runtime>/config/server.conf``,
       written by ``scripts/bootstrap-runtime.sh`` from the
       ``server_defaults`` of ``server/runtime/manifest.json``; client side:
       ``REDS_PORT`` in ``client/reds.conf`` (8010 when unset)

How the inference host is built and its tunnel opened by hand is
:ref:`inference_host`.

Pointing at an endpoint directly
********************************

Any endpoint can be addressed without touching a configuration file:

.. code-block:: console

   $ spear-chat --api-base https://inference.example.org/v1 \
                --model qwen3 \
                --ctx 32768

or through the environment, which is what ``client/machine.env`` is for:

.. code-block:: bash

   # client/machine.env — untracked, this machine only
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

When ``SPEAR_CTX`` is not set, the client asks the endpoint at startup —
llama.cpp's ``/props`` (``n_ctx``, per slot) or vLLM's ``/v1/models``
(``max_model_len``) — and uses what it reports. If neither answers, it falls
back to 32768 and says so. The startup line names the source::

   context window: 524288 (server /props)
   context window: 32768 (default: server did not report a window — /props: URLError; ...)

Two settings interact with it. The reply cap (``--max-tokens``,
``SPEAR_MAX_TOKENS``) is reserved out of the window, so a long edit completes
instead of being truncated mid-call. ``SPEAR_CONTEXT_FRACTION`` is the
fraction of the window the prompt may occupy before compaction (default
0.95).

These two, and the sampling settings below, govern the normative and general
runtime. The :term:`coding core` (:doc:`/reasoning/implementation`) reserves its own
output budget, sends no sampling parameters of its own, and does not compact:
it stops at half the window and asks for a summary. One coding-core response
is capped at ``SPEAR_RESPONSE_MAX_TOKENS`` (:ref:`configuration`), so a reply
that degenerates into a repeating tool call ends as truncated instead of
running to the full reservation.

Sampling
********

Sampling is a client-side decision. ``--temp`` (``SPEAR_TEMP``) defaults to
``0.25``. A low temperature gives more conservative, more consistent edits; a
model that falls into repetition at a low temperature wants it raised rather
than lowered, and a stream circuit-breaker truncates a loop when it happens
anyway.

Alongside the temperature, the normative and general runtime sends
``top_p`` 0.8, ``top_k`` 20, a repetition penalty of 1.05 and disables the
model's thinking mode. ``SPEAR_SAMPLING=server`` sends none of them, so the
server's own configuration applies — which is what the coding core always
does.

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

Every flag and variable above, with its default, is listed in the
:ref:`configuration reference <configuration>`.

.. warning::

   Credentials belong in the environment or in an untracked
   ``client/machine.env``, never in a tracked file.

The provider interface
**********************

``client/models/model_backend.py`` normalises providers behind one interface,
so ``rag_chat`` never branches on which model is answering.

``OpenAICompatibleBackend``
   The local ``llama-server``, and any OpenAI-compatible endpoint including
   the remote vLLM pod.

``AnthropicBackend``
   Optional, selected with ``--provider anthropic``.  Thinking is explicitly
   disabled.  It serves the normative and general runtime only: the coding
   core needs an OpenAI-compatible endpoint.  See `Authenticating against
   Anthropic`_ below.

For the normative and general runtime, both produce a ``ModelTurn`` (the
coding core reads the OpenAI-compatible reply directly, with its
``finish_reason``, as a ``RawTurn`` from ``complete_messages``):

.. code-block:: python

   @dataclass(frozen=True)
   class ModelTurn:
       text: str
       tool_calls: tuple[ModelToolCall, ...]
       stop_reason: StopReason        # TOOL_USE | END_TURN | MAX_TOKENS | REFUSAL | ERROR | OTHER
       usage: Mapping[str, int] | None = None
       error: str | None = None

Prompt caching on the Anthropic path
************************************

Tokens are free on the local server and charged on the remote one, so only
``AnthropicBackend`` caches.  The adapter marks what it has *observed* to
repeat rather than what a caller promises, which is why the canonical turn did
not have to grow a field for it:

``tools``
   The final schema carries a breakpoint once the whole list arrives unchanged
   from the previous call.  One mark caches every schema before it.

``system``
   The longest common prefix of this call's system text and the previous
   one's, cut back to a line break, becomes a cached first block; the rest
   follows uncached.  ``ContextEngine`` composes the stable layers first --
   system rules, project rules, selected memory -- so the shared opening is a
   real prefix and not a coincidence.

A first call marks nothing: there is no prior turn to compare against, and a
cache write costs more than an uncached read.  Below roughly 1500 tokens
nothing is marked either, because a breakpoint under the model's minimum
cacheable prefix is silently ignored.  When the provider reports them,
``cache_creation_input_tokens`` and ``cache_read_input_tokens`` join
``ModelTurn.usage``, which is what makes the mechanism checkable in a trace
rather than believed.

Roles sharing one backend instance simply miss: an Explorer turn between two
Main turns shares no long prefix with either, the comparison yields nothing,
and the call goes out uncached.

**The ceiling is deliberate.** The conversation is never cached, because the
system block ends with the layers that are rebuilt every turn -- the
``WorkingState`` projection, the retrieved chunks, the summaries -- and a
prefix cache stops at the first byte that moved.  Caching the messages too
would mean moving those layers out of ``system`` and behind the conversation,
which changes what *every* backend sends, including the local one all the
published measurements were taken on.  That trade has not been measured, so it
has not been made.

Authenticating against Anthropic
********************************

Two credentials work, and the harness never chooses between them itself — it
asks the SDK, whose resolution chain is the authority:

.. code-block:: text

   ANTHROPIC_API_KEY → ANTHROPIC_AUTH_TOKEN → OAuth profile (~/.config/anthropic)
                     → workload identity federation

An exported ``ANTHROPIC_API_KEY`` is passed through explicitly.  With no key,
the client is constructed **bare** so the SDK resolves and refreshes an OAuth
profile on its own — a key passed as ``None`` would short-circuit that chain.

When nothing resolves and the run is interactive, the harness *offers* to open
a browser session:

.. code-block:: console

   $ spear-chat --provider anthropic
   No Anthropic credential found (no ANTHROPIC_API_KEY, no `ant` session).
   Open a browser to sign in now?
     ❯ 1. Yes, run `ant auth login` (Enter)
       2. No, abort

Three properties matter here.  It is a **proposal**, never automatic: opening a
browser is an outward-facing side effect, and this harness does not take those
implicitly.  It requires a **terminal** — a piped or cron run fails closed with
the manual instructions instead of blocking on a prompt nobody can answer.  And
the login runs as an ordinary supervisor-side subprocess, deliberately **outside
the tool sandbox**: it needs the network, a browser, and write access to
``~/.config/anthropic``, none of which a tool call is ever granted.  It belongs
to the same class of setup actions as starting the model server.

Every failure is closed: declining, a missing ``ant`` binary (the message
carries the install URL), and a login that ends without a credential all raise
``ModelBackendConfigurationError`` rather than continuing unauthenticated.

The ``ant`` CLI is not part of the venv; install it from
https://github.com/anthropics/anthropic-cli/releases and put the binary on
``PATH``.  ``ant auth status`` shows which source is active — worth checking
first when a profile appears to be ignored, since a stale exported
``ANTHROPIC_API_KEY`` silently outranks it.

The ``stop_reason`` contract
****************************

On the normative and general runtime this contract is enforced twice — once in
the backend, once in the agent loop — and it is deliberately unforgiving, because a tool call that survives an
ambiguous stop reason is a tool call nobody authorized.

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Backend observation
     - Result
   * - ``tool_use`` **and** at least one valid call
     - ``TOOL_USE``
   * - ``tool_use`` with no calls
     - ``ERROR``
   * - calls present but ``end_turn``
     - ``ERROR``, and **no** tool call is exposed
   * - calls present but ``max_tokens``
     - ``ERROR``, and no tool call is exposed
   * - calls present but ``refusal``
     - ``ERROR``, and no tool call is exposed
   * - ``pause_turn``
     - ``ERROR`` — SPEAR uses client tools only
   * - ``model_context_window_exceeded``
     - ``ERROR``

On the loop side, only ``TOOL_USE`` carrying at least one call can reach tool
execution.  ``END_TURN`` and ``REFUSAL`` require zero tool calls;
``MAX_TOKENS``, ``ERROR`` and ``OTHER`` execute nothing at all.  Even the
forced final synthesis turn refuses a fresh tool request.

The coding core applies the equivalent rule to ``finish_reason``: a tool call
cut by the output limit is dropped and asked for again, never executed, and an
invalid tool-call payload is never guessed at.

.. _backends_smoke_test:

Smoke test
**********

From the ``client/`` directory of a checkout, with its virtualenv, against
the inference host's tunnel:

.. code-block:: console

   $ cd ~/spear/client
   $ ./bin/python -c "
   from openai import OpenAI
   from models.model_backend import OpenAICompatibleBackend, ConversationMessage, TextBlock
   b = OpenAICompatibleBackend(OpenAI(base_url='http://127.0.0.1:8082/v1', api_key='not-needed'))
   t = b.complete(system='Reply with exactly the token requested, nothing else.',
                  conversation=[ConversationMessage(role='user',
                      content=(TextBlock(text='Reply with exactly: SPEAR-BACKEND-OK'),))],
                  tools=[], use_tools=False)
   print(t.stop_reason, '|', (t.text or '').strip(), '|', t.tool_calls)"
   end_turn | SPEAR-BACKEND-OK | ()

``python -c`` puts the current directory on the import path, which is why it
runs from ``client/``.  That three-part answer — ``end_turn``, the exact
token, an empty tool tuple — is the quickest confirmation that serving and the
backend contract are both healthy.

.. seealso::

   :ref:`Configuration reference <configuration>` · :ref:`Model serving
   <model_serving>` · :ref:`Setting up an inference host <inference_host>`
