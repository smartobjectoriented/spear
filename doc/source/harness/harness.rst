.. _harness:

============================
The confined execution path
============================

The reason this documentation exists. Everything else here serves a model
that answers questions; this part is what happens when the model stops
answering and starts *doing* — running a command, writing a file, opening a
socket on the machine you are reading this on.

One rule governs all of it: **fail-closed** — a confinement the
:term:`harness` cannot apply is an error, never a silent downgrade
(:ref:`the list <fail_closed>`).

Read the pages in this order.  Each one assumes the one before it.

.. toctree::
   :maxdepth: 2

   tool_harness
   security_model
   sandbox
   network
   resource_control
   final_harness_audit
   internals

:doc:`tool_harness` is the spine: what a tool is, how one is selected, the
control plane and the decision, boundary and execution layers a call passes,
and what the harness knows about a call before it runs. :doc:`security_model`
holds the authorization rules that decide whether it runs at all.

The three that follow are the execution layer's mechanisms, in increasing
order of subtlety: :doc:`sandbox` for the filesystem and the process,
:doc:`network` for what it may reach, :doc:`resource_control` for what it may
consume. Subtlety is the ordering because the failure modes get quieter:
a blocked path is obvious, a blocked connection less so, and a process that
merely runs slower than it should is the hardest to notice at all.

:doc:`final_harness_audit` closes the section with the whole harness on one
sheet: what is on by default, what is optional and what is experimental.
:doc:`internals` is for contributors: who owns what inside the runtime.
