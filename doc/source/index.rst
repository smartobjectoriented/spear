.. SPEAR documentation master file.

.. image:: img/REDS-HEIG-VD.png
   :align: center
   :scale: 30%
   :target: https://reds.heig-vd.ch

.. toctree::
   :maxdepth: 5
   :numbered:
   :hidden:

   introduction
   Getting started <getting_started>
   Using the assistant <usage>
   directory_layout
   architecture
   retrieval
   harness/harness
   model/model
   testing
   operations
   Container <container>
   Coding conventions <coding_conventions>
   Development flow <dev_flow>
   glossary

|


.. rst-class:: center

SPEAR — Specification-driven Platform for Embedded Agentic Reasoning
####################################################################

.. rst-class:: left

Setup and environment
*********************

- :ref:`Getting started <getting_started>` — four commands, and the one
  measurement that explains why the retrieval index is not optional
- :ref:`Using the assistant <usage>` — the command line, the in-chat
  commands and the guards
- :ref:`Containerised deployment <container>` — what the image carries, what
  it expects mounted, and the two run flags without which the harness
  refuses every command

.. rst-class:: left

What runs where
***************

- :ref:`Introduction <introduction>` — what this is, and what it is not
- :ref:`Directory layout <directory_layout>` — the trees, and which one owns
  what
- :ref:`Harness architecture <architecture>` — the Python core and the paths
  through it
- :ref:`Retrieval <retrieval>` — the corpus, the index, and what retrieval
  measurably buys

.. rst-class:: left

The confined execution path
***************************

The reason this documentation exists. One rule governs all of it:
**fail-closed** — a confinement that cannot be applied is an error, never a
silent downgrade to a less confined execution.

- :ref:`The tool harness <tool_harness>` — what a tool is, and what is known
  about a call before it runs
- :ref:`The security model <security_model>` — the rules that decide whether
  it runs at all
- :ref:`Sandbox <sandbox>`, :ref:`network <network>` and
  :ref:`resource control <resource_control>` — the three confinement layers,
  in increasing order of subtlety
- :ref:`Final harness audit <final_harness_audit>` — what was measured
  against the whole path, rather than what it was designed to do

.. rst-class:: left

The model and its weights
*************************

- :ref:`Model serving <model_serving>` and :ref:`runtime bootstrap
  <runtime_bootstrap>` — the quantised weights, the GPU/CPU split, and what
  must be true before the first token
- :ref:`Training <training>` — how the harness turns its own usage into a
  governed dataset
- :ref:`Model history <model_history>` — what the earlier attempts measured,
  including the one that cost a rented B200

.. rst-class:: left

Verification and operation
**************************

- :ref:`Testing <testing>` — what the suite covers, and what it deliberately
  does not
- :ref:`Operations <operations>` — the observable states and where each one
  is reported

.. rst-class:: left

Conventions and development flow
********************************

- :ref:`Our coding conventions <coding_conventions>`
- :ref:`Our development flow <dev_flow>`
- :ref:`Glossary <glossary>`

To edit the documentation and to use the correct underlying policy, you can
read `this documentation style guide
<https://documentation-style-guide-sphinx.readthedocs.io/en/latest/style-guide.html>`_.
