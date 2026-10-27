.. SPEAR documentation master file.

.. image:: img/spear-logo.svg
   :align: center
   :width: 336px
   :alt: SPEAR

.. toctree::
   :maxdepth: 3
   :numbered:
   :hidden:

   start/index
   overview/index
   using/index
   reasoning/index
   harness/harness
   model/model
   operations/index
   contributing/index

|


.. rst-class:: center

Specification-driven Platform for Embedded Agentic Reasoning
############################################################

.. image:: img/REDS-HEIG-VD.png
   :align: center
   :width: 190px
   :target: https://reds.heig-vd.ch

|

SPEAR is an engineering agent for specified systems. It keeps apart the agent
that changes code, the :term:`control plane` that confines it, the evidence of what it
did, the normative reasoning about a bound standard, and the compliance
evidence that judges a change — and computes its verdicts from that evidence on
the :term:`final source state`. Each turn is given the context of its own workspace —
rules, skills, knowledge, project metadata and external capabilities — selected
deterministically by workspace and request class.

.. rst-class:: left

Getting started
***************

- :ref:`Getting started <part_start>`: :ref:`Quick start <getting_started>`, :ref:`Running the public image <container_run>`, :ref:`Installation <installation>`, :ref:`Containerised environment <container>`

.. rst-class:: left

Overview
********

- :ref:`Overview <part_overview>`: :ref:`Introduction <introduction>`, :ref:`Architecture <architecture>`, :ref:`Glossary <glossary>`

.. rst-class:: left

Using SPEAR
***********

- :ref:`Using SPEAR <part_using>`: :ref:`spear-chat <usage>`, :ref:`Projects and corpora <projects>`, :ref:`Retrieval <retrieval>`, :ref:`Context selection <context_selection>`, :ref:`Context <context>`, :ref:`Workspace knowledge <knowledge>`, :ref:`External capabilities <capabilities>`, :ref:`Configuration reference <configuration>`

.. rst-class:: left

Request modes and evidence
**************************

- :ref:`Request modes and evidence <part_reasoning>`: :ref:`Implementation mode <implementation_mode>`, :ref:`Normative mode <standards>`, :ref:`Mixed mode <mixed_mode>`, :ref:`Evidence and verdicts <evidence_verdicts>`, :ref:`Normative answer guards <guards>`, :ref:`The bound-session change workflow <workflow>`

.. rst-class:: left

The confined execution path
***************************

- :ref:`The confined execution path <harness>`: :ref:`Tool execution harness <tool_harness>`, :ref:`Security model <security_model>`, :ref:`Sandbox <sandbox>`, :ref:`Network backend <network>`, :ref:`Resource control <resource_control>`, :ref:`Final harness audit <final_harness_audit>`, :ref:`Runtime internals <harness_internals>`

.. rst-class:: left

The model and its weights
*************************

- :ref:`The model and its weights <model>`: :ref:`Setting up an inference host <inference_host>`, :ref:`Model serving <model_serving>`, :ref:`Model backends <backends>`, :ref:`Rebuilding the runtime <runtime_bootstrap>`, :ref:`Training and fine-tuning <training>`, :ref:`Model history and what it cost <model_history>`

.. rst-class:: left

Operations and reference
************************

- :ref:`Operations and reference <part_operations>`: :ref:`Operations <operations>`, :ref:`Troubleshooting <troubleshooting>`, :ref:`Testing <testing>`, :ref:`Directory layout <directory_layout>`

.. rst-class:: left

Contributing
************

- :ref:`Contributing <part_contributing>`: :ref:`Coding conventions <coding_conventions>`, :ref:`Development flow <dev_flow>`, :ref:`Release process <release_process>`


To edit the documentation and to use the correct underlying policy, you can read `this documentation style guide <https://documentation-style-guide-sphinx.readthedocs.io/en/latest/style-guide.html>`_.
