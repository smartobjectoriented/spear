.. _model:

=========================
The model and its weights
=========================

Where the answers come from: which model is served, how it is brought up,
how it is trained on the harness's own usage, and — the page that exists
because the others would otherwise repeat its lessons — what the earlier
attempts measured before this one was chosen.

.. toctree::
   :maxdepth: 2

   model_serving
   runtime_bootstrap
   training
   model_history

:doc:`model_serving` covers the serving side: the quantised weights, the
GPU/CPU split and the flags that decide throughput. :doc:`runtime_bootstrap`
is what has to be true before the first token, and what the harness does when
it is not.

:doc:`training` is the fine-tuning side, and it stands on its own: how the
harness turns its own usage into a governed dataset, and what it takes to run
a job on the card that is also serving the model.

:doc:`model_history` is the record of what was tried and what it cost —
including the experiment that rented a B200 to learn something a smaller run
would have shown. It is kept because a choice whose alternatives are
forgotten gets re-litigated every year.
