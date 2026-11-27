.. _model_history:

==============================
Model history and what it cost
==============================

This page is a record of past choices, not current instructions: which model
has been served, why it was changed, and what the fine-tuning experiments
actually measured.  What is served today, and how, is :ref:`model_serving` and
:ref:`inference_host`.  The record is kept because the reasoning that led to the expensive decisions looks
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
refreshed by a reindex, no retraining (:doc:`/using/retrieval`).

**A QLoRA adapter** brings *behaviour* — domain style, vocabulary, reflexes —
frozen into a small adapter served with ``--lora`` on top of the frozen base.
Only about 0.1 % of the weights are trained, which is why a run costs a few
dollars (:doc:`/model/training`).

**Workspace knowledge and skills** are immediate, local context recalled
across sessions: ``/remember`` records a fact about the workspace, and what the
model offers stays a proposal until the operator accepts it
(:doc:`/using/retrieval`).

The contextual-learning loop — skills plus knowledge — is comparable to what Nous
Research's Hermes Agent does.  What is unusual here is that the same system
also learns in the weights, and runs entirely offline.

Which model is served
=====================

The project started on Qwen3-Coder-30B-A3B with a homemade QLoRA adapter
(v1 → v3, retrained on a rented pod).  It migrated on 2026-06-08 to the stock
**Qwen3.6-35B-A3B**: the newer base beat the fine-tuned Coder outright, so the
adapter was dropped.  Since 2026-11-11 the served model is the dense
**Qwen3.8-27B**, in Q8_0, at its native 262144-token context.

Q8 is chosen over Q4 for accuracy.  :doc:`/model/model_serving` describes the
serving profile and how to override the model or load an adapter for one run.

No fine-tune is served today
============================

No adapter is served.  The pipeline is kept for the thing that would justify
one: **behaviour, not facts**.  What a governed dataset looks like, and what it
takes to run a job on the card that also serves the model, is :doc:`/model/training`.
