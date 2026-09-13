# Fine-tuning tooling

Generic machinery for producing a LoRA adapter and serving it on top of the
base model. It trains nothing by itself: a run needs a dataset, and a dataset
comes from your own trees.

```
scripts/train_qlora.py          QLoRA fine-tune
scripts/merge_corpora.py        merge JSONL sample sets, keeping their system
                                prompts distinct
scripts/merge_and_convert.sh    merge the adapter and quantize to GGUF
scripts/build_edit_discipline.py  samples for whole-file edit behaviour
scripts/{wget,resilient}_download.sh   resumable weight downloads
cloud/load_preflight.{sh,py}    does this card actually hold the model, loaded
                                the way training will load it? Measured, not
                                deduced.
cloud/train_qlora_*.py          per-base trainers
cloud/runpod_train*.sh          drive a rented pod
setup.sh · activate.sh          environment
```

## What is not here

**Corpus builders.** Turning a project into training samples means harvesting
its trees, its documentation and its commit history, and emitting them under a
system prompt that describes that project. All of that belongs to whoever owns
the project, not to the platform, and it is not shipped here.

A builder writes JSONL lines of `{"messages": [...]}`; `merge_corpora.py`
combines several sets without unifying their system prompts, and
`train_qlora.py` trains on the result. That is the whole contract.

**The experience file.** `/good` in the chat appends validated exchanges to
`$SPEAR_STATE_DIR/experience.jsonl` (or `SPEAR_EXPERIENCE_FILE`). It is a
sample source like any other; nothing here reads it by default.
