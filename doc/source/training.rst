========================
Training and fine-tuning
========================

The harness produces its own training data.  Every task it runs leaves a
trajectory behind, and this subsystem is what turns that exhaust into a
governed dataset, freezes it into an auditable bundle, decides whether the
hardware can hold the run, and — only on an explicit operator command — starts
it.

Two things it deliberately is **not**.  It is not model-visible: no tool
exposes any of it, and nothing here can be reached by something the model
writes.  And freezing a bundle executes no training.  ``freeze`` writes files
and stops; the command that would train is printed in the bundle's runbook for
a human to run.

.. figure:: img/spear_components.svg
   :width: 100%
   :alt: Component map, including the training band

   The training modules are the band the rest of the harness feeds.

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
is what once left the dataset holding a single trajectory — exactly one project
declares a bench — while every other session produced nothing.  ``/good``
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
     - Real work.  The only origin a real bundle admits by default.
   * - ``BENCHMARK``
     - Produced while measuring.  Training on it would train on the exam.
   * - ``TEST_FIXTURE``
     - Manufactured by the test suite.
   * - ``SYNTHETIC``
     - Generated rather than observed.  Smoke tests only.
   * - ``MANUAL_IMPORT``
     - Hand-added, and marked as such.

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

   $ cd /opt/llm/spear/spear
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
     - The human procedure, preflight included.
   * - ``checksums.sha256``
     - What ``validate-bundle`` re-checks.

Loss masking is validated rather than assumed.  A sample trains exactly one
assistant message; every earlier assistant message is context with
``train: false``.  A record that would train a previous assistant turn is a
hard error — that is how a model learns to reproduce the mistake it later
recovered from.

.. _the-moe-lesson:

The MoE lesson
==============

The bundle emits QLoRA configs for an 80B mixture-of-experts model, and the
single most expensive thing this project learned is that **"4-bit" is not a
property of a config, it is a property of a load**.

``load_in_4bit`` makes bitsandbytes replace ``nn.Linear`` modules.  Whether the
routed experts *are* ``nn.Linear`` depends on the installed ``transformers``:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - ``transformers``
     - Routed expert layout
     - ``load_in_4bit`` reaches them
   * - 4.57.6
     - ``per_expert_modules (512,)``
     - yes
   * - 5.9.0
     - ``fused_parameter (512, 1024, 2048)``
     - **no**
   * - 5.14.1 / 5.16.0.dev0
     - ``fused_parameter (512, 1024, 2048)``
     - **no**

Both rows are measured, not read: ``LAYOUT_ONLY=1`` in
``qwen3-finetune/cloud/load_preflight.py`` materialises the model on the meta
device from ``config.json`` alone — no weights, no GPU, no disk — and reports
the layout and the exact parameter counts.  Over the same 77.3 B expert
parameters that is **145 GiB left in bf16** against **39 GiB in NF4**.

So the emitted config sets ``quantize_moe_experts: true``, which is Axolotl's
patch of the *loader*: it quantizes every ≥3-D CUDA parameter as it lands and
disables ``caching_allocator_warmup``, which would otherwise pre-reserve the
bf16 size and eat the saving.  Two consequences the config generator encodes:

* routed experts are targeted through ``lora_target_parameters``, never
  ``lora_target_modules`` — they are parameters, not modules;
* ``lora_target_linear`` must stay false, which Axolotl itself enforces.

Asking for 4-bit without the flag is how a run ended up holding 148 GiB of
bf16 weights on a rented 179 GiB card, for a job that fits a far smaller one.

Will it fit?  Measure it
========================

``TrainingHardwareReport.classify()`` takes the load mode, because without it
the question has no answer: for this model 96 GiB is insufficient or ample
depending on one flag.  It returns ``LIKELY_FITS`` only on the bf16 path at
≥160 GiB.  On the quantized path it returns at most ``MAY_FIT``, and
``MAY_FIT`` means *provision it and run the preflight*, not *go*.  Those
thresholds are Axolotl's documentation, and this project has already been bitten
once by a documented figure that did not survive contact with the stack.

The preflight is the measurement:

.. code-block:: console

   $ CONFIG_YAML=configs/axolotl-sft.yml bash qwen3-finetune/cloud/load_preflight.sh

It refuses on disk first, naming the exact number of GB to add; stops the
inference server, because 5 GiB free out of 97 answers nothing; loads the real
checkpoint with the real config; reports how many expert parameters were
actually quantized on load; runs one forward + backward at the real sequence
length; prints the peak and the headroom; and restarts the server from a
``trap``, including after the OOM it exists to provoke.  It writes
``load-preflight.json`` and exits non-zero on anything but ``FITS``.

It reads its numbers from the bundle's own ``axolotl-sft.yml`` when given one,
otherwise from ``AxolotlProfile`` in ``training_bundle.py``, parsed with
``ast`` rather than imported — the harness's runtime dependencies have no
business on a training host, and a preflight that cannot run where the training
runs is decoration.  It names which source it used, so a verdict cannot be
mistaken for one covering a config it never saw.

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

The training host is the REDS machine (``reds-ml``), whose reserved card is an
RTX PRO 6000 Blackwell with 96 GiB.  The objective is that this card is
sufficient and no GPU is rented.

The binding constraint there is **disk, not VRAM**.  Measured 2026-08-25:
``/home`` holds 97 GB free of 1399, and the bf16 shards are 159 GB.  4-bit does
not reduce that — bitsandbytes quantizes at load, from shards that must land on
disk first.  Growing that filesystem is the cheap fix, and it is why the
preflight's first check is a disk check that prints the deficit rather than a
suggestion to rent something.

.. note::

   The remote stack is pinned by Axolotl, not by us: installing
   ``axolotl 0.18.0`` on that host moved ``transformers`` to 5.14.1, ``peft``
   to 0.19.1 and ``bitsandbytes`` to 0.49.1.  The venv's previous state is kept
   in ``qwen3-finetune/.venv-freeze-before-axolotl.txt``.

Related material
================

``qwen3-finetune/`` holds the experiment side: the LoRA trainers, the pod
scripts, the corpus builders (deployment-specific, not shipped:
``scripts/merge_corpora.py``) and the preflight.  It is a working area, not a
product surface — the governed path is the one described above.
