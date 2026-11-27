.. _training:

========================
Training and fine-tuning
========================

The :term:`harness` produces its own training data.  Every task it runs leaves a
trajectory behind, and this subsystem is what turns that exhaust into a
governed dataset, freezes it into an auditable bundle, decides whether the
hardware can hold the run, and — only on an explicit operator command — starts
it.

Two things it deliberately is **not**.  It is not model-visible: no tool
exposes any of it, and nothing here can be reached by something the model
writes.  And freezing a bundle executes no training.  ``freeze`` writes files
and stops; the command that would train is printed in the bundle's runbook for
a human to run.

The five stages
===============

The stage names appear in the module docstrings and in bundle metadata, so
they are worth learning once.

.. list-table::
   :header-rows: 1
   :widths: 8 26 66

   * - Stage
     - Modules
     - What it owns
   * - FT0
     - ``training_data`` · ``training_store``
     - Canonical, provider-neutral episode capture.  Evidence, not examples.
   * - FT1
     - ``sft_dataset`` · ``training_export``
     - Turn-level SFT curation: one selected assistant target per sample.
   * - FT2
     - ``preference_dataset`` · ``preference_export``
     - Paired and unpaired preference mining from observed outcomes.
   * - FT3
     - ``training_bundle`` · ``training_readiness`` · ``training_governance``
       · ``training_splits``
     - Readiness, governance, deterministic splits, and the frozen bundle.
   * - FT4
     - ``training_controller`` · ``training_launcher`` · ``training_jobs``
       · ``training_handoff`` · ``finetune_commands``
     - The operator control plane: preflight, launch, job records, and the
       single-GPU handoff between serving and training.

Capture is wider than judging
=============================

A turn is recorded whether or not anything judged it.  A turn the project's
bench judged is recorded ``pass`` or ``fail``; a turn nothing judged is
recorded ``unrated`` rather than dropped.

That width is the point, not an oversight.  Gating the recording on a verdict
would keep only the turns of projects that declare a bench, and every other
session would produce nothing.  ``/good``
promotes what was right after the fact, which is a curation act on a record
that already exists.

Records are provider-neutral by construction: no private reasoning, no wire
payloads.  What is captured is what any trainer could consume, not what one
provider happened to return.

Governance: where a sample may come from
========================================

Every episode carries a ``DataOrigin``:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Origin
     - Meaning
   * - ``NORMAL_USAGE``
     - Real work.  Admitted by default.
   * - ``BENCHMARK``
     - Produced while measuring.  Training on it would train on the exam.
   * - ``TEST_FIXTURE``
     - Manufactured by the test suite.
   * - ``SYNTHETIC``
     - Generated rather than observed.  Smoke tests only.
   * - ``MANUAL_IMPORT``
     - Hand-added, and marked as such.  Admitted by default.
   * - ``LICENSED_STANDARD``
     - Work done against a bound licensed standard, whatever the episode
       claims.  Not admitted by default.

An origin nobody can classify becomes ``UNKNOWN``, which no profile admits.

A bundle built from a prohibited origin is not a warning — it is a bundle that
declares ``smoke_test_only`` and whose runbook refuses to print a training
command.  The permanent holdout is enforced at the same layer: a task family
reserved for evaluation never appears in a training split, and the split is
derived from a hash of the task family rather than from a shuffle, so the same
episode lands in the same side on every machine and every rebuild.

Readiness is evidence, not a feeling
====================================

``TrainingReadinessEvaluator`` answers three separate questions, and keeping
them separate is what makes the answer usable:

``TrainingReadinessState``
   ``EMPTY`` · ``COLLECTING`` · ``READY`` · ``BLOCKED`` · ``NEEDS_REVIEW`` —
   what the *data* is.

``ReadinessLevel``
   ``NOT_READY`` · ``READY_FOR_SMOKE`` · ``READY_FOR_EXPERIMENT`` — what may
   be *run* with it.  Nothing in this codebase claims a level above
   "experiment".

``TrainingStrategyRecommendation``
   ``COLLECT_MORE_DATA`` · ``SFT_ONLY`` · ``SFT_THEN_UNPAIRED_PREFERENCE`` ·
   ``SFT_THEN_PAIRED_PREFERENCE`` · ``NEEDS_MANUAL_DATA_REVIEW`` — what to do
   next.

Preference pairing is stricter than recovery linkage on purpose: an
automatically approved pair needs two observed completions for the *exact same*
model-visible decision context.  A recovery after the context changed is two
independent outcome-labelled observations, not a preference.

The frozen bundle
=================

.. code-block:: console

   $ cd ~/spear/client
   $ ./bin/python -m training readiness --source audit/training-data
   $ ./bin/python -m training freeze --source audit/training-data --output /tmp/out
   $ ./bin/python -m training validate-bundle /tmp/out/bundles/<id>

``--source`` is the episode store, ``audit/training-data`` under ``STATE_DIR``.
It is required rather than defaulted: freezing the wrong store is not a mistake
worth making convenient.

A freeze is deterministic and immutable.  The same store freezes to the same
``bundle_id`` and the same path; adding an episode produces a *new* bundle and
leaves the previous manifest byte-identical.  Every bundle carries:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - File
     - Content
   * - ``manifest.json``
     - Schema version, source model profile, dataset counts, lineage schema.
   * - ``datasets/{sft,unpaired,paired}/``
     - The frozen JSONL, split as governance decided.
   * - ``configs/axolotl-{sft,kto,dpo}.yml``
     - The training configs — written, never run.
   * - ``governance.json`` · ``readiness.json`` · ``readiness.md``
     - Why this data was admissible and what it supports.
   * - ``evaluation-plan.json``
     - Held-out only, including a general-control regression.
   * - ``RUNBOOK.md``
     - The human procedure.
   * - ``checksums.sha256``
     - What ``validate-bundle`` re-checks.

Loss masking is validated rather than assumed.  A sample trains exactly one
assistant message; every earlier assistant message is context with
``train: false``.  A record that would train a previous assistant turn is a
hard error — that is how a model learns to reproduce the mistake it later
recovered from.

Will it fit?
============

``TrainingHardwareReport.classify()`` gives a coarse verdict for a 4-bit QLoRA
run of the dense 27B source: ``LIKELY_FITS`` from 96 GiB, the card the FT-4 and
FT-5 runs completed on; ``MAY_FIT`` from 48 GiB, which means *try a short run
first*, not *go*; below that, ``LIKELY_DOES_NOT_FIT``.

The operator surface
====================

``/finetune`` inside the chat, operator-only and never model-visible:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Command
     - Effect
   * - ``/finetune status`` · ``list``
     - Stage readiness and known jobs.
   * - ``/finetune doctor [--remote]``
     - What is missing locally, and optionally on the training host.
   * - ``/finetune prepare [sft|kto|dpo]``
     - Freeze a bundle.  Nothing runs.
   * - ``/finetune start [sft|kto|dpo] [--force]``
     - Preflight, then launch.  ``--force`` is the only way past a hard check.
   * - ``/finetune inspect <job-id>`` · ``logs`` · ``stop``
     - Durable job records, bounded log tail, cancellation.

``SSHTrainingLauncher`` runs the job on the training host.  ``training_handoff``
owns the awkward part: the card that trains is the card that serves, so
starting a run takes the inference service down and giving the card back brings
it up, with a lease file so two operators cannot both believe they hold it.

Where it runs today
===================

The training host is the inference host (:ref:`inference_host`), whose reserved card is an
RTX PRO 6000 Blackwell with 96 GiB.  The objective is that this card is
sufficient and no GPU is rented.

The bf16 shards of the base model must land on disk before bitsandbytes
quantizes them at load: 4-bit does not reduce the disk a run needs.

.. note::

   The remote stack is pinned by Axolotl, not by us: installing
   ``axolotl 0.18.0`` on that host moved ``transformers`` to 5.14.1, ``peft``
   to 0.19.1 and ``bitsandbytes`` to 0.49.1.  Freeze the venv
   (``pip freeze``) before installing it, so its previous state can be
   restored.

Related material
================

``qwen3-finetune/`` holds the experiment side: the LoRA trainers, the pod
scripts and the corpus tooling (``scripts/merge_corpora.py``; the builders that
read a deployment's own sources are deployment-specific and not shipped).  It
is a working area, not a
product surface — the governed path is the one described above.
