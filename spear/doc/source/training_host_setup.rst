SPEAR training host setup
============================

This is an operator runbook for a future ``reds-ml`` Qwen3-Coder-Next
experiment. FT4B diagnostics found that the current SPEAR execution account
cannot authenticate through the bare ``reds-ml`` target, while the explicit
operator destination/identity works. No setup command below has been executed by
SPEAR. Do not use these instructions to modify the inference service or its
environment.

Latest operator-supplied read-only probe
----------------------------------------

Using the configured ``host`` with the operator-provided
SSH identity succeeded. It reported Python 3.12.3, driver 595.84,
an RTX A4000, and two RTX PRO 6000 Blackwell Max-Q GPUs. The assigned
``GPU-00000000-0000-0000-0000-000000000000`` has about 14.3 GiB free at probe
time. Axolotl was not on PATH and ``/srv/edgem-training`` was not usable, so
remote readiness remains BLOCKED. The probe was read-only and did not inspect
or alter the other GPUs.

Provisioned FT4C environment
----------------------------

The safe operator-owned root is whatever ``remote_root`` names, because
``/srv/edgem-training`` was unavailable to the training account. It contains a
private uv-managed Python 3.12 environment at ``env/``, confined caches,
bundles, jobs, probes, models, and outputs. The trusted wrapper is
the configured ``axolotl_executable``; it sets the assigned
``CUDA_VISIBLE_DEVICES`` UUID and cache locations before executing the pinned
environment's Axolotl binary.

Installed versions are Axolotl 0.18.0, PyTorch 2.12.1+cu130, Transformers
5.14.1, PEFT 0.19.1, bitsandbytes 0.49.1, flash-linear-attention 0.4.1, and
Cut Cross Entropy 25.1.1. Tokenizer/config/chat-template-only assets for
``Qwen/Qwen3-Coder-Next`` were downloaded at revision
``a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb``; no model weights were fetched.

Capacity and availability are separate: the assigned GPU has 97,887 MiB total
(suitable for the approximately 47 GiB no-expert QLoRA reference profile), but
14,599 MiB free while the existing llama-server uses 82,628 MiB. Current
availability is therefore ``BUSY``; FT4C does not stop or signal that workload.
Future operator-controlled training may stop the known inference service and
re-check free VRAM before launch.

The training filesystem currently has roughly 100,937,000 KiB free. The
configured coarse model-storage reserve is 160 GiB, so doctor reports
``MODEL_STORAGE_CAPACITY=INSUFFICIENT`` and the full Qwen snapshot remains
deferred. No weights are downloaded while data is collecting.

Inference management discovery
------------------------------

Read-only inspection found no ``llama-server.service`` in system or user
systemd and no container owns the assigned-GPU process. The running process is
a manually detached ``llama-server`` launched through
the configured inference wrapper. Its fixed control identity is the
configured binary, GGUF path, port 8010, and assigned UUID; readiness is
``http://127.0.0.1:8010/health``. FT4D does not migrate or restart it.

The future ``SSHInferenceServiceController`` may signal only the verified
process group matching that identity. It refuses unknown GPU owners and never
accepts an arbitrary PID or shell. One filesystem lease prevents overlapping
training handoffs. The empty-data gate runs before this controller, so the
current workflow cannot interrupt inference.

Separate inference and training
-------------------------------

The inference service previously exposed by ``reds-ml:8010`` is not a training
environment. Use a dedicated training account/environment and a dedicated root,
for example ``/srv/training`` (replace this with an operator-approved
absolute path). Do not stop inference, reuse its process identity, overwrite its
model files, or install Axolotl into its Python environment.

SSH operator configuration
--------------------------

From the account that runs SPEAR, configure the existing OpenSSH alias and
verify both commands non-interactively::

  ssh reds-ml true
  ssh reds-ml hostname

Use the normal key/agent and host-key configuration managed by the operator.
Do not add passwords, private keys, or custom authentication logic to SPEAR.
The application never edits ``~/.ssh/config``. If this check fails, run
``/finetune doctor --remote`` after the operator fixes SSH and review the
classified reason (alias, connectivity, host-key, authentication, or shell).

Dedicated Axolotl environment
-----------------------------

Create an isolated environment on the training host using the host's approved
CUDA/Python combination. Do not guess a CUDA version when hardware inspection
is unavailable. Current Axolotl documentation requires Python >=3.11, PyTorch
>=2.11, and a supported NVIDIA Ampere+ or AMD GPU. The documented uv-first
pattern is conceptually::

  uv venv --python 3.12
  source .venv/bin/activate
  uv pip install torch
  uv pip install --no-build-isolation 'axolotl[deepspeed]'

Choose the CUDA/PyTorch matrix appropriate for the actual GPU and approved
host image. Keep the environment isolated from inference. For Qwen3-Next,
install only the extra performance dependencies that the operator elects to
use after reviewing the current Axolotl model guide (for example Cut Cross
Entropy and the documented FLA version); do not enable optional kernels by
default.

Trusted execution contract
--------------------------

Expose a stable operator-managed executable or wrapper, such as
``/opt/spear-training/bin/axolotl``. It must execute the dedicated environment
and then ``exec`` Axolotl. Configure it in the gitignored
``training-execution.json`` along with an absolute remote root and Python
executable. For the currently assigned host GPU, set
``cuda_visible_devices`` to
``GPU-00000000-0000-0000-0000-000000000000``. The doctor, preprocessing
preflight, and future wrapper query/use only that UUID; they do not touch other
GPUs. Do not put shell activation syntax or arbitrary command prefixes in
slash-command arguments.

After setup, check::

  /finetune doctor --remote

The doctor performs read-only checks for Python, Axolotl, CUDA, GPU/VRAM,
PyTorch, Transformers, PEFT, bitsandbytes, disk, and root usability. It does
not install dependencies or create a directory. A future explicit bundle
preflight may create only its confined opaque job directory.

It reports four independent dimensions: training-environment readiness,
hardware capacity from TOTAL VRAM, current GPU availability from free VRAM and
known workload activity, and dataset readiness. A BUSY current GPU is not a
hardware-capacity failure.

Qwen3-Next planning notes
-------------------------

The source must be the Transformers/Hugging Face
``Qwen/Qwen3-Coder-Next`` checkpoint at an immutable commit revision. GGUF is
an inference artifact and is rejected as a training source. The first profile
is conservative QLoRA: 4-bit loading, attention-only LoRA targets, no expert
targeting, gradient checkpointing, small micro-batch, and no packing. Axolotl's
Qwen3-Next guide gives approximately 47 GiB for no-expert-targeting QLoRA and
approximately 71 GiB when targeting experts; these are coarse planning values,
not an admission guarantee. Existing GPU processes must not be killed to make
room. Configure a minimum free-VRAM policy if the operator needs one.

Non-training validation
-----------------------

Once SSH and the environment are ready, use a synthetic training-prohibited
fixture only for a bounded checksum/preprocessing smoke. The command shape is::

  axolotl preprocess <config.yml> --debug --debug-num-examples 3

Do not run ``axolotl train`` during FT4B. If preprocessing would download
large Qwen model assets that are not already cached, stop and report
``preprocessing_not_validated_model_assets_unavailable`` rather than downloading
the model merely for a smoke check.

FT4C completed this bounded synthetic preprocess with tokenizer-only assets.
Axolotl showed 50 ignored labels and 5 trainable labels, so only the selected
recovery assistant target received loss. No model weights were downloaded and
``axolotl train`` was not run.

References
----------

* https://docs.axolotl.ai/docs/installation.html
* https://docs.axolotl.ai/docs/models/qwen3-next.html
* https://docs.axolotl.ai/docs/cli.html
