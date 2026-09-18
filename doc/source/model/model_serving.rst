.. _model_serving:

=============
Model serving
=============

The model runs locally.  ``llama-server`` exposes an OpenAI-compatible API on
``127.0.0.1:8080`` and nothing else on the machine talks to it.

Current profile
===============

.. code-block:: console

   $ cat /opt/llm/spear/spear/active-model.conf
   /srv/spear-runtime/models/gguf/Qwen3-Coder-Next-UD-Q4_K_XL.gguf

   $ cat /opt/llm/spear/spear/active-lora.conf
   none

The running command line, as launched by ``server/inference/serve.sh``:

.. code-block:: console

   llama-server
       --model /srv/spear-runtime/models/gguf/Qwen3-Coder-Next-UD-Q4_K_XL.gguf
       --ctx-size 32768
       --n-gpu-layers 99
       --n-cpu-moe 99
       --threads 16
       --flash-attn on
       --cache-type-k q8_0  --cache-type-v q8_0
       --host 127.0.0.1  --port 8080
       --jinja
       --parallel 1

The flags that matter
=====================

``--jinja``
   Required.  It enables the model's chat template, and with it native tool
   calling.  Without it the assistant loses structured tool calls entirely.

``--n-gpu-layers 99`` with ``--n-cpu-moe 99``
   The hybrid offload.  Attention and the KV cache go to the GPU; the
   mixture-of-experts weights stay in CPU RAM.  This is what makes a large MoE
   model usable on a laptop-class GPU: VRAM holds the part that is latency
   critical, system RAM holds the part that is merely large.

``--cache-type-k q8_0 --cache-type-v q8_0``
   Quantised KV cache.  Buys context length at a small quality cost.

``--ctx-size 32768``
   The usable context.  The harness is aware of it: long tool outputs are
   truncated at ``max_output_chars`` (10 000 by default) precisely so a single
   verbose build log cannot evict the conversation.

``--parallel 1``
   One slot.  This is a single-user assistant; concurrency would only fragment
   the KV cache.

The launcher holds no flag values
=================================

``server/inference/serve.sh`` contains no context size, no model path, no port
and no thread count.  Those come from ``server/config/server.conf``, or from
``SPEAR_SERVER_*`` in the environment; ``server/config/server.conf.example``
documents every key.  The script itself contributes only what is a property of
the *build* rather than of a deployment:

``--cache-type-k q8_0 --cache-type-v q8_0``
   the quantised KV cache;

``--flash-attn``
   unconditional;

``--jinja``
   required for the chat template and therefore for tool calling;

``--n-cpu-moe``
   for a mixture-of-experts model, unless a deployment says otherwise.

Context size has no built-in default at all.  It is the one value two earlier
implementations disagreed about, so it must be stated by the deployment rather
than inherited from whichever launcher happened to run.

Sampling
========

Sampling is a client-side decision and lives in ``rag_chat.py``: temperature
0.7, ``top_p`` 0.8, ``top_k`` 20, ``repeat_penalty`` 1.05.  Lower temperatures
are not safer here — they drive this model into repetition death-loops.  A
stream circuit-breaker truncates them when they happen anyway.

To survive a session, run the server as a transient unit:

.. code-block:: console

   $ sudo systemd-run --unit=spear-llm server/inference/serve.sh --lora <adapter>

Model and adapter switching
===========================

Resolution order is the same for both, and both are persisted so a restart
keeps the choice:

.. code-block:: text

   SPEAR_MODEL env   >   active-model.conf   >   built-in default
   SPEAR_LORA  env   >   active-lora.conf    >   none

``spear-model`` writes those files.  ``server/inference/serve.sh`` validates the result:
an unreadable model path falls back to the default, and a stale or unreadable
adapter path is silently ignored rather than failing the launch.

Service placement
=================

``spear-server`` runs as the transient unit ``spear-llm.service``:

.. code-block:: console

   $ cat /proc/$(pgrep -x llama-server)/cgroup
   0::/system.slice/spear-llm.service

This placement is load-bearing for the security model.  The model server sits
in ``system.slice``; sandboxed tool commands sit in transient scopes under
``user.slice/…/app.slice``.  The two trees are disjoint, so no memory limit,
OOM kill or task-limit hit inside a tool call can reach the model server.
:doc:`/harness/resource_control` verifies this explicitly.

Backends
========

Four, mutually exclusive, chosen at launch.  Without a flag an interactive
run lists them and preselects the one used last (kept in
``active-backend.conf``); a flag skips the picker, and off a terminal the last
choice is reused silently.  Listing never probes — no SSH, no ``curl`` — so a
server that is down is still selectable.

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

The three tunnel ports differ so a laptop model, the pod and the REDS server
can be up at once; a test asserts they never collide.

``reds`` is deliberately not the pod path.  The pod is ephemeral — RunPod
reassigns its SSH port on every restart, which is why that branch re-prompts
for host and port, uploads a serve script and starts vLLM over ``tmux``.  The
REDS server is a stable host, so none of that applies, and the launcher
**installs, uploads and starts nothing there**: it opens the tunnel, checks
that something answers, and otherwise says so and stops.  Start the inference
server on that machine yourself.  ``reds.conf`` carries the host (an
``~/.ssh/config`` alias is preferred — it already holds the user, key and auth
quirks), the remote port (8000 for vLLM, 8080 for llama-server) and the model
id to advertise.

``model_backend.py`` normalises providers behind one interface, so
``rag_chat`` never branches on which model is answering.

``OpenAICompatibleBackend``
   The local ``llama-server``, and any OpenAI-compatible endpoint including
   the remote vLLM pod.

``AnthropicBackend``
   Optional, selected with ``--provider anthropic``.  Thinking is explicitly
   disabled.  See `Authenticating against Anthropic`_ below.

Both produce a ``ModelTurn``:

.. code-block:: python

   @dataclass(frozen=True)
   class ModelTurn:
       stop_reason: StopReason        # TOOL_USE | END_TURN | MAX_TOKENS | REFUSAL | ERROR | OTHER
       text: str | None
       tool_calls: tuple[ModelToolCall, ...]

Prompt caching on the Anthropic path
====================================

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
================================

Two credentials work, and the harness never chooses between them itself — it
asks the SDK, whose resolution chain is the authority:

.. code-block:: text

   ANTHROPIC_API_KEY → ANTHROPIC_AUTH_TOKEN → OAuth profile (~/.config/anthropic)
                     → workload identity federation

An exported ``ANTHROPIC_API_KEY`` is passed through explicitly.  With no key,
the client is constructed **bare** so the SDK resolves and refreshes an OAuth
profile on its own — a key passed as ``None`` would short-circuit that chain,
which is exactly the bug this replaced.

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
============================

This contract is enforced twice — once in the backend, once in the agent loop
— and it is deliberately unforgiving, because a tool call that survives an
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

Smoke test
==========

.. code-block:: console

   $ cd /opt/llm/spear/spear
   $ ./bin/python -c "
   from openai import OpenAI
   from model_backend import OpenAICompatibleBackend, ConversationMessage, TextBlock
   b = OpenAICompatibleBackend(OpenAI(base_url='http://127.0.0.1:8080/v1', api_key='not-needed'))
   t = b.complete(system='Reply with exactly the token requested, nothing else.',
                  conversation=[ConversationMessage(role='user',
                      content=(TextBlock(text='Reply with exactly: EDGEM-BACKEND-OK'),))],
                  tools=[], use_tools=False)
   print(t.stop_reason, '|', (t.text or '').strip(), '|', t.tool_calls)"
   end_turn | EDGEM-BACKEND-OK | ()

That three-part answer — ``end_turn``, the exact token, an empty tool tuple —
is the quickest confirmation that serving and the backend contract are both
healthy.
