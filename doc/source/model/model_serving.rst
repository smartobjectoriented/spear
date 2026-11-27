.. _model_serving:

=============
Model serving
=============

The model runs on one GPU host.  ``llama-server`` exposes an OpenAI-compatible
API on ``127.0.0.1`` only, and clients reach it through an SSH tunnel.  How to
build that host is :ref:`inference_host`; this page explains what it serves
and why.  What the client does with the endpoint — the backend it selects,
the ports, the sampling it sends, the smoke test — is :ref:`backends`.

Current profile
===============

The profile is ``server/runtime/manifest.json``: Qwen3.8-27B, a dense model,
in Q8_0, one 28.6 GB file, on a 96 GB card.  ``server/inference/serve.sh``
turns the deployment's ``config/server.conf`` into this command line:

.. code-block:: console

   llama-server
       --model <runtime>/models/gguf/Qwen3.8-27B-Q8_0.gguf
       --ctx-size 262144
       --parallel 1
       --n-gpu-layers 99
       --flash-attn on
       --cache-type-k q8_0  --cache-type-v q8_0
       --host 127.0.0.1  --port 8010
       --jinja
       --threads 16

The reference deployment points ``SPEAR_SERVER_LLAMA_BIN`` at a wrapper that
adds the model card's reasoning and sampling settings: ``--reasoning-format
auto --no-reasoning-preserve``, medium reasoning effort, ``--temp 1.0
--top-p 0.95 --top-k 20 --min-p 0``.

The flags that matter
=====================

``--jinja``
   Required.  It enables the model's chat template, and with it native tool
   calling.  Without it the assistant loses structured tool calls entirely.

``--n-gpu-layers 99``
   Everything on the card.

``--cache-type-k q8_0 --cache-type-v q8_0``
   Quantised KV cache.  Buys context length at a small quality cost.

``--ctx-size 262144``
   The length the model was trained at.  A window past it is served with YaRN:
   ``serve.sh`` adds the RoPE scaling and overrides the GGUF's
   ``context_length`` when ``SPEAR_SERVER_CTX`` exceeds
   ``SPEAR_SERVER_NATIVE_CTX``, because ``llama-server`` otherwise caps each
   slot at the trained length.  YaRN is static and would touch every short
   prompt too, so it is switched on only past that length.  The
   :term:`harness` still truncates long tool outputs at ``max_output_chars``
   (10 000 by default) so a single verbose build log cannot evict the
   conversation.

``--parallel 1``
   One slot.  This is a single-user assistant; concurrency would only fragment
   the KV cache.

The launcher holds no flag values
=================================

``serve.sh`` contains no context size, no model path and no thread count,
and its port fallback (8080) is not the profile's.  Those come from ``<runtime>/config/server.conf``, or from
``SPEAR_SERVER_*`` in the environment; ``server/config/server.conf.example``
documents every key.  The script itself contributes only what is a property of
the *build* rather than of a deployment:

``--cache-type-k q8_0 --cache-type-v q8_0``
   the quantised KV cache;

``--flash-attn``
   unconditional;

``--jinja``
   required for the chat template and therefore for tool calling;

the YaRN flags
   derived from ``SPEAR_SERVER_CTX``, ``SPEAR_SERVER_NATIVE_CTX`` and
   ``SPEAR_SERVER_ARCH`` as above.

Context size has no built-in default at all: it must be stated by the
deployment rather than inherited from whichever launcher happens to run.

Changing the model or the adapter
=================================

Both are keys of ``server.conf`` — ``SPEAR_SERVER_MODEL`` and
``SPEAR_SERVER_LORA`` (``none`` for the base model) — followed by a restart of
the service.  ``serve.sh`` refuses to start on an unreadable model, and serves
the base model, saying so, when the adapter is unreadable.

Service placement
=================

On the host the server is ``spear-inference.service``, a system unit
(:ref:`inference_host`, step 6), so it sits in ``system.slice``:

.. code-block:: console

   $ cat /proc/$(systemctl show spear-inference -p MainPID --value)/cgroup
   0::/system.slice/spear-inference.service

This placement is load-bearing for the security model when the harness runs on
the same machine.  Sandboxed tool commands sit in transient scopes under
``user.slice/…/app.slice``.  The two trees are disjoint, so no memory limit,
OOM kill or task-limit hit inside a tool call can reach the model server.
:doc:`/harness/resource_control` verifies this explicitly.

Checking it from a client
=========================

The :ref:`smoke test <backends_smoke_test>` sends one request through the
client's backend and confirms that serving and the backend contract are both
healthy.
