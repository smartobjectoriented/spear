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

The profile is ``server/runtime/manifest.json``: former-model, Q8_0, in
four shards (85 GB, 79 GiB), on a 96 GB card.  ``server/inference/serve.sh`` turns the
deployment's ``config/server.conf`` into this command line:

.. code-block:: console

   llama-server
       --model <runtime>/models/gguf/former-model-Q8_0-00001-of-00004.gguf
       --ctx-size 524288
       --parallel 1
       --n-gpu-layers 99
       --flash-attn on
       --cache-type-k q8_0  --cache-type-v q8_0
       --host 127.0.0.1  --port 8010
       --jinja
       --threads 16
       --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144
       --override-kv former-arch.context_length=int:524288

The flags that matter
=====================

``--jinja``
   Required.  It enables the model's chat template, and with it native tool
   calling.  Without it the assistant loses structured tool calls entirely.

``--n-gpu-layers 99``, and no ``--n-cpu-moe``
   Everything on the card.  On a smaller GPU, ``SPEAR_SERVER_NCPUMOE`` keeps
   the mixture-of-experts weights in system RAM while attention and the KV
   cache stay on the GPU — the hybrid offload that makes this model usable on
   a laptop-class card, at a large cost in speed.

``--cache-type-k q8_0 --cache-type-v q8_0``
   Quantised KV cache.  Buys context length at a small quality cost.

``--ctx-size 524288`` and the YaRN flags
   Twice the 262 144 tokens the model was trained at.  Past that length the
   RoPE positions are rescaled with YaRN, and the GGUF's ``context_length`` is
   overridden because ``llama-server`` otherwise caps each slot at it.  Both
   are added by ``serve.sh`` only when the window exceeds the trained length,
   since YaRN is static and would otherwise touch every short prompt too.
   Measured: a fact 40 % into a 311 744-token prompt was recalled exactly, and
   generation speed did not change.  The :term:`harness` still truncates long tool
   outputs at ``max_output_chars`` (10 000 by default) so a single verbose
   build log cannot evict the conversation.

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

``--n-cpu-moe``
   for a mixture-of-experts model, unless a deployment says otherwise;

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
