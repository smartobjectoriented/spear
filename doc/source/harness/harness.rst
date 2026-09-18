.. _harness:

============================
The confined execution path
============================

The reason this documentation exists. Everything else here serves a model
that answers questions; this part is what happens when the model stops
answering and starts *doing* — running a command, writing a file, opening a
socket on the machine you are reading this on.

One rule governs all of it, and it is worth stating before any of the
mechanisms: **fail-closed**. A confinement that cannot be applied is an
error, never a silent downgrade. A model that asks for something the harness
cannot confine gets a refusal, not a shortcut. Every page below is, in the
end, an account of how one mechanism holds that line and what it costs when
it cannot.

Read them in this order. Each layer assumes the one before it.

.. toctree::
   :maxdepth: 2

   tool_harness
   security_model
   sandbox
   network
   resource_control
   final_harness_audit

:doc:`tool_harness` is the spine: what a tool is, how one is selected, and
what the harness knows about a call before it runs. :doc:`security_model`
holds the authorization rules that decide whether it runs at all.

The three that follow are the confinement layers themselves, in increasing
order of subtlety: :doc:`sandbox` for the filesystem and the process,
:doc:`network` for what it may reach, :doc:`resource_control` for what it may
consume. Subtlety is the ordering because the failure modes get quieter:
a blocked path is obvious, a blocked connection less so, and a process that
merely runs slower than it should is the hardest to notice at all.

:doc:`final_harness_audit` closes the section with what was actually measured
against the whole path, rather than what it was designed to do.
