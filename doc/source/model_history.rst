==============================
Model history and what it cost
==============================

This page is the project's own record of which model has been served, why it
was changed, and what the fine-tuning experiments actually measured.  It is
kept because the reasoning that led to the expensive decisions looks
convincing and was wrong, and a summary that omitted that would be the least
useful part of this documentation.

Four learning loops, on different clocks
========================================

The assistant is not "RAG plus a fine-tune".  Four independent loops feed it,
and they run at different speeds::

                      ┌────────────────────────────────────────────────┐
    real usage ──/good──► dataset ──QLoRA on a rented pod──► adapter ───┤
         │            (+ teacher-student distillation: the model's      │
         │             answers are graded against the real source)      │
         │                                                              ▼
         └── /remember · save_skill ──► instant, local, no retraining ───► better
                                                                        assistant

**Retrieval** brings *content* on the fly — indexed code and documentation,
refreshed by a reindex, no retraining (:doc:`retrieval`).

**A QLoRA adapter** brings *behaviour* — domain style, vocabulary, reflexes —
frozen into a small adapter served with ``--lora`` on top of the frozen base.
Only about 0.1 % of the weights are trained, which is why a run costs a few
dollars (:doc:`training`).

**Memory and skills** are immediate, local context that the model writes
itself, with confirmation, and recalls across sessions (:doc:`retrieval`).

The contextual-learning loop — skills plus memory — is comparable to what Nous
Research's Hermes Agent does.  What is unusual here is that the same system
also learns in the weights, and runs entirely offline.

Which model is served
=====================

The project started on Qwen3-Coder-30B-A3B with a homemade QLoRA adapter
(v1 → v3, retrained on a rented pod).  It migrated on 2026-06-08 to the stock
**Qwen3.6-35B-A3B**: the newer base beat the fine-tuned Coder outright, so the
adapter was dropped.  Since 2026-06-23 the served model is **Qwen3-Coder-Next
(80B-A3B)**, chosen for native tool calls and *targeted* edits where the dense
Coder-32B rewrote whole files.  The inference host serves it at roughly
157 tokens/s in Q8_0.

Q8 was chosen over Q4 for accuracy: measured, it costs about 7 % of throughput
to double the precision of the weights.  :doc:`model_serving` describes the
serving profile and how to override the model or load an adapter for one run.

No fine-tune is served today
============================

The first full-size runs were done in bf16 on a rented B200, because the
weights then occupy 152 GB and no single card here holds them.

**That was the expensive way, and it was avoidable.**  4-bit had been ruled out
by reading the *installed* ``transformers`` 5.x, where the routed experts are
fused into 3-D parameters that bitsandbytes — which replaces ``nn.Linear`` —
cannot reach.  But the published checkpoint stores those experts separately,
and the pod was pinned to ``transformers`` 4.57.6, which loads them as ordinary
``nn.Linear``, where bitsandbytes reaches every one.  A 4-bit run would very
likely have fitted a 48–80 GB card.  The measured layouts, and the loader flag
that fixes this under ``transformers`` 5.x, are in :ref:`the-moe-lesson`.

Those runs did validate the chain end to end — dataset, pod, adapter, GGUF,
``--lora``, audit — which is what they were for.

Measured, 2026-08-24: the adapter does not pay
==============================================

Two adapters were trained on a rented B200 and served on the same Q8_0 base.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Configuration
     - Without retrieval
     - With retrieval
   * - base, no adapter
     - 18.2 %
     - 89.6 %
   * - usage-shaped adapter
     - 22.1 %
     - 83.1 %
   * - descriptive adapter
     - 16.9 %
     - —

Without retrieval the usage-shaped corpus helps a little and the descriptive
one hurts.  With retrieval — the way the harness actually runs — the adapter
*costs* six complete answers out of 37.  It does not complement retrieval, it
competes with it.

So no adapter is served.  The pipeline is kept for the thing that would justify
one: **behaviour, not facts**.  What a governed dataset looks like, and what it
takes to run a job on the card that also serves the model, is :doc:`training`.
