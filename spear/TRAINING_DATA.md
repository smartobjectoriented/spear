# Continuous training-data workflow

SPEAR automatically captures canonical trajectories (FT0), curates
one-target SFT samples (FT1), and mines exact-context paired plus grounded
unpaired preference observations (FT2). It does **not** automatically train
itself. FT3 adds governed readiness decisions and immutable bundle freezing.

Run `python -m training readiness --source audit/training-data` to inspect the
real store. Run `python -m training freeze --source audit/training-data
--output audit/training-data` only when a snapshot is wanted. The bundle keeps
semantic OpenAI messages, source checksums, policies, governance, a held-out
evaluation plan, Axolotl configuration, and a human runbook.

The default source is the Transformers checkpoint
`Qwen/Qwen3-Coder-Next`, not the deployed GGUF inference artifact. The optional
`qwen3_coder_next_base` profile uses `Qwen/Qwen3-Coder-Next-Base`. GGUF paths
are rejected. The initial profile is conservative 4-bit QLoRA and targets only
attention projections; targeting MoE experts is explicit because it changes
memory requirements.

Every SFT message is annotated with a per-turn `train` flag. Context—including
prior failed assistant actions—is false; only FT1's selected last assistant
target is true. Axolotl uses the tokenizer's chat template and
`message_field_training`, never hard-coded Qwen tokens. Preprocessing with
`axolotl preprocess ... --debug` remains mandatory on the actual training host.

Production governance excludes benchmark, test-fixture, and synthetic origins,
supports permanent holdout task families and project allow/deny filters, and
fails bundle validation on leakage. Synthetic data is only for pipeline tests.
FT4 must evaluate an adapter against held-out and general control tasks before
any deployment or merge.

## Operator fine-tuning jobs (FT4A)

Fine-tuning is an operator control-plane operation. It is not registered in
`ToolRegistry`, is unavailable to `AgentRuntime`, and is handled before a chat
turn or model call exists. The model cannot prepare, start, stop, or inspect
training. Normal use only accumulates episodes and derived candidates.

```text
/finetune status
/finetune doctor [--remote]
/finetune prepare [sft|kto|dpo]
/finetune start [sft|kto|dpo] [--force]
/finetune stop [job-id]
/finetune logs [job-id] [lines]
/finetune list
/finetune inspect <job-id>
```

`prepare` performs local readiness, freeze, governance, leakage, checksum, and
configuration validation only. `start` repeats those checks, requires a pinned
Hugging Face revision, then performs remote environment/checksum and
`axolotl preprocess ... --debug` validation before launching detached
`axolotl train ...`. SFT is the default; KTO and DPO have independent gates.

Copy `training-execution.json.example` to the state directory as
`training-execution.json` and set the operator-owned SSH alias, remote root,
Axolotl executable/wrapper, and immutable model revision. The actual file is
gitignored. System SSH configuration supplies authentication; SPEAR stores
no passwords or keys. Remote paths are confined to
`<remote_root>/jobs/<opaque-job-id>`.

If the normal alias does not select the desired key, `ssh_identity_file` may
contain an operator-managed absolute path such as
the operator's SSH identity file. Only that path is recorded; key contents are
never read or copied. A host value such as `operator@gpu-host.example` is also
accepted when the operator intentionally targets that explicit SSH destination.
Set `cuda_visible_devices` to the operator-assigned UUID
`GPU-00000000-0000-0000-0000-000000000000` for this host. The remote doctor,
preprocessing preflight, and future detached training wrapper all pass that
selector through; they do not query or use other GPUs.

`--force` acknowledges a small experiment. It may lower operational thresholds
and acknowledge coarse hardware feasibility. It cannot bypass empty data,
governance/holdout exclusions, quarantine, leakage, checksums, malformed data,
GGUF rejection, missing revision, remote ownership/path validation, remote
checksum verification, or preprocessing failure.

Jobs persist at `audit/training-data/jobs/<job-id>/` using atomic snapshots and
an append-safe recovery journal. The detached remote wrapper writes bounded
logs and atomic terminal markers. Stop verifies root, job marker, and recorded
process group before SIGTERM and never automatically SIGKILLs. Outputs are only
lineage references for later evaluation: FT4A does not merge, convert,
quantize, evaluate, or deploy adapters.

## Remote host readiness (FT4B)

`/finetune doctor` checks local training-control configuration without SSH.
`/finetune doctor --remote` performs bounded, read-only diagnostics against
the configured `reds-ml` alias and caches a bounded machine manifest at
`audit/training-data/remote-readiness.json`. It never prints keys, passwords,
environment variables, or full package inventories. `/finetune status` reads
that cache and never performs a fresh SSH probe merely to display status.

Remote readiness is independent of dataset readiness. A host may be READY
while the real TrainingStore is still COLLECTING. A future SSH start requires
the cached environment state to be READY; current GPU BUSY is handled only by
the configured handoff and never by `--force`. SSH
failure, missing Axolotl, missing GPU/CUDA, invalid root, missing dependencies,
and missing source revision remain hard gates.

The expected environment is a dedicated operator-managed Python/Axolotl
environment, separate from the inference service on port 8010. Current
Axolotl guidance requires Python >=3.11, PyTorch >=2.11, and an NVIDIA Ampere+
or AMD GPU. Its Qwen3-Next guide describes conservative QLoRA at about 47 GiB
without expert targeting and about 71 GiB with expert targeting; these are
coarse planning values, not an EDGEM admission formula. Axolotl's documented
uv installation and Qwen3-Next extra dependencies are not installed
automatically.

If SSH authentication is unavailable, configure the normal OpenSSH alias so
`ssh reds-ml` succeeds non-interactively from the SPEAR account. Use an
operator-managed agent/key setup; SPEAR never edits SSH config, asks for
passwords, copies keys, or weakens host verification.

## Dedicated reds-ml environment (FT4C)

The provisioned operator-owned root is named by `remote_root`; the
original `/srv/edgem-training` path was unavailable. It contains a private uv
Python 3.12 environment and confined `cache`, `models`, `bundles`, `jobs`,
`probes`, and `outputs` directories. The trusted wrapper is
the configured `axolotl_executable`, which enforces the assigned GPU
UUID and cache paths before executing Axolotl.

The pinned environment currently contains Axolotl 0.18.0, PyTorch 2.12.1+cu130,
Transformers 5.14.1, PEFT 0.19.1, bitsandbytes 0.49.1,
flash-linear-attention 0.4.1, and Cut Cross Entropy 25.1.1. The source model is
`Qwen/Qwen3-Coder-Next` at immutable revision
`a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb`; only tokenizer/config/chat-template
assets were fetched, never model weights.

The assigned RTX PRO 6000 Blackwell has 97,887 MiB total VRAM, so hardware
capacity is `READY` for the coarse approximately 47 GiB no-expert QLoRA
profile. Current availability is `BUSY` because the known llama-server uses
82,628 MiB and 14,599 MiB is free. FT4C never stops or signals inference.
These are separate from data readiness, which remains `COLLECTING` with zero
real episodes. The bounded synthetic Axolotl preprocess succeeded and showed
50 ignored labels plus 5 trainable labels, preserving one-target masking;
`axolotl train` was never executed.

## Controlled inference handoff (FT4D)

Read-only inspection found the assigned-GPU llama-server is a manually
detached host process launched through the configured inference wrapper,
not a systemd unit or Docker container. Its fixed identity is the configured
llama-server binary, Qwen GGUF path, port 8010, and assigned GPU UUID. The
health probe is the fixed local `http://127.0.0.1:8010/health` endpoint.

`SSHInferenceServiceController` verifies that identity, including the process
environment and process group, before it can send SIGTERM to the configured
process group. It never accepts a PID, shell, service name, or command from a
slash command. Unknown or differently-owned GPU workloads fail closed. A
filesystem lease permits only one training job to own the assigned GPU
handoff.

Training environment readiness is independent of current GPU availability:
the environment and hardware capacity are `READY`, while current availability
is `BUSY_BY_MANAGED_INFERENCE`. The data gate runs first, so the empty real
TrainingStore prevents any inference stop, bundle transfer, preprocessing, or
training. Future jobs restore inference only when that job actually stopped
it; restore failure is recorded independently from the training outcome. The
full Qwen snapshot remains deferred until data is ready and storage is
explicitly sufficient.
