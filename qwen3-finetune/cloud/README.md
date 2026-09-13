# Cloud fine-tuning

Fine-tuning a large base happens on a rented multi-GPU pod; the only thing
that comes home is the LoRA adapter, served on top of the same GGUF the
inference host already has.

| Script | Where |
|---|---|
| `load_preflight.sh` / `load_preflight.py` | the training host — **run this first** |
| `runpod_train.sh` + `train_qlora_35b.py` | any 48 GB pod |
| `runpod_train_coder32.sh` + `train_qlora_coder32.py` | dense-coder QLoRA |
| `train_lora_qwen3next.py` | a pod with ~160 GB for the weights alone |

## Measure, do not deduce

`load_preflight.sh` answers one question empirically: does this card hold the
model, loaded exactly the way training will load it? Stop inference, load with
the real quantization config, measure the peak, unload, restart inference.

It exists because the answer was guessed twice and was wrong twice — once at
the cost of a rented B200 for a job that would have fitted a far smaller card.
A documentation figure said one thing and the installed library did another.

The same rule the harness applies to sandboxes and budgets: if it says FITS,
the local card does the run and no pod is booked; if it says OOM, it says so
in the ten minutes a load takes rather than in the third hour of a paid one.

## What is not here

The run scripts that drive a particular corpus — they name a dataset, a base
and a run — are deployment-specific and live with that deployment.
