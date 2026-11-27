# Cloud fine-tuning

Fine-tuning a large base happens on a rented multi-GPU pod; the only thing
that comes home is the LoRA adapter, served on top of the same GGUF the
inference host already has.

| Script | Where |
|---|---|
| `runpod_train.sh` + `train_qlora_35b.py` | any 48 GB pod |
| `runpod_train_coder32.sh` + `train_qlora_coder32.py` | dense-coder QLoRA |

## What is not here

The run scripts that drive a particular corpus — they name a dataset, a base
and a run — are deployment-specific and live with that deployment.
