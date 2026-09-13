"""LoRA fine-tune of Qwen3-Coder-Next (80B-A3B, arch qwen3_next) in bf16.

CLOUD script — a rented multi-GPU pod: this needs ~160 GB of
VRAM for the weights alone, more than a single-card host has, and more free
disk. Bring back only the LoRA adapter (~10-400 MB) and serve it on top of the
Q8_0 GGUF the server already has: `llama-server --lora`.

Why this loads bf16, and why that is not a recommendation. The argument was
that bitsandbytes quantizes nn.Linear while this model keeps its 512 routed
experts as fused 3-D nn.Parameter tensors, so 4-bit would buy nothing. That is
true of transformers 5.x. It is NOT true of the 4.57.6 this script pins, where
the experts load as an nn.ModuleList of ordinary nn.Linear -- nor of the
published checkpoint, which stores them separately (24 576 per-expert tensors,
no fused one). 4-bit was available and would very likely have halved the card
this needs. Kept as-is because the runs it produced are the ones measured in
cloud/README.md; read that file before choosing hardware.

What is adapted, and what must never be:
  - `self_attn.{q,k,v,o}_proj` — present only in the FULL-attention layers,
    one every `full_attention_interval` (4) of the 48, so 12 layers. The other
    36 are Gated-DeltaNet (linear attention) and have no such projections.
  - `mlp.shared_expert.{gate,up,down}_proj` — a real nn.Linear that runs on
    EVERY token in ALL 48 layers. It is the only always-on MLP path, so it can
    carry fact->symbol associations that 12 attention layers alone cannot.
  - NEVER `linear_attn.{in_proj_qkvz,in_proj_ba,out_proj}`. On the way to GGUF,
    Qwen3NextModel.modify_tensors permutes and splits in_proj_qkvz into
    attn_qkv + attn_gate; an adapter on those cannot be mapped back, and the
    run would produce weights that nothing can serve. The regex below excludes
    them and the guard after get_peft_model refuses to continue if one slipped
    through.
  - The routed experts are nn.Parameter, not nn.Linear: PEFT never touches
    them, which is also why VRAM stays predictable.

Both adapted families map cleanly to GGUF (`blk.N.attn_{q,k,v,output}`,
`blk.N.ffn_{gate,up,down}_shexp`) and llama.cpp's qwen3next graph runs both
through build_lora_mm / build_ffn — checked in llama.cpp-next before this
script was written, because an adapter that converts but is silently ignored
at inference looks exactly like a fine-tune that did not work.

Env knobs:
    BASE_MODEL     HF repo id or local snapshot dir of the BF16 base
    FT_DATA_DIR    dir holding the jsonl datasets
    FT_TRAIN_FILE  / FT_VAL_FILE   {"messages": [...]} jsonl
    FT_OUT_DIR     / FT_RUN_NAME   adapter output
    FT_TARGETS     attn | attn+shexp   (default attn+shexp)
    FT_RANK        LoRA rank         (default 32)
    FT_EPOCHS      epochs            (default 4)
    FT_MAX_LEN     max sample tokens (default 2560 — the ib corpus tops at
                   2364, so nothing is truncated)
    FT_BATCH / FT_ACCUM             micro-batch and accumulation
    FT_LR          learning rate     (default 2e-4)
    FT_MAX_STEPS   >0 = smoke test
    FT_GPU_RESERVE_GIB  headroom left per card for activations (default 10)
"""
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-Coder-Next")
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("FT_DATA_DIR", HERE.parent / "data"))
OUT = Path(os.environ.get("FT_OUT_DIR", HERE.parent / "checkpoints")) / \
    os.environ.get("FT_RUN_NAME", "qlora-spear-v1")
MAX_LEN = int(os.environ.get("FT_MAX_LEN", "2560"))
TRAIN_FILE = os.environ.get("FT_TRAIN_FILE", "ib_train.jsonl")
VAL_FILE = os.environ.get("FT_VAL_FILE", "ib_val.jsonl")

# Regexes, not bare module names: `gate_proj` alone would also be the right
# name in a dense Qwen MLP, and naming the full path is what documents that
# only the SHARED expert is meant here — never a routed one, never the router.
_ATTN = r".*\.self_attn\.(q|k|v|o)_proj"
_SHEXP = r".*\.mlp\.shared_expert\.(gate|up|down)_proj"
_PRESETS = {
    "attn": _ATTN,
    "attn+shexp": f"({_ATTN}|{_SHEXP})",
}


def _placement() -> dict:
    """Where the weights go, as kwargs for from_pretrained.

    On the pod: spread over every card, each capped below its real size.
    device_map="auto" otherwise fills each one to the brim and the first
    optimizer step OOMs on the logits — vocab 151936 x seq x fp32 is several
    GB on whichever card ends up holding the LM head.

    With no CUDA device at all it returns nothing and the model loads on the
    CPU. That is not a training mode: it is what lets this script be rehearsed
    against a miniature qwen3_next before a pod is rented by the hour.
    """
    if not torch.cuda.is_available():
        print("no CUDA device: loading on CPU (rehearsal only)")
        return {}
    reserve = float(os.environ.get("FT_GPU_RESERVE_GIB", "10"))
    budget = {}
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory / 2**30
        budget[i] = f"{max(total - reserve, 1.0):.0f}GiB"
    print(f"loading in bf16 across {len(budget)} card(s): {budget}")
    return {"device_map": "auto", "max_memory": budget}


def main() -> None:
    cfg = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    n_layers = getattr(cfg, "num_hidden_layers", 0)
    interval = getattr(cfg, "full_attention_interval", 0)
    print(f"base   : {BASE_MODEL}")
    print(f"arch   : {cfg.model_type}, {n_layers} layers, "
          f"full-attention every {interval}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        **_placement(),
    )
    # Accelerate falls back to CPU (or disk) offload when the weights do not
    # fit the budget, and offload does not fail — it just makes a one-hour run
    # a one-week run, on a card billed by the hour. Refuse instead.
    spilled = {d for d in getattr(model, "hf_device_map", {}).values()
               if d in ("cpu", "disk")}
    if spilled and torch.cuda.is_available():
        raise SystemExit(
            f"the base spilled to {sorted(spilled)}: the cards cannot hold it. "
            "Lower FT_GPU_RESERVE_GIB only if you know what is left fits the "
            "activations, otherwise take a bigger pod.")

    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    targets = _PRESETS[os.environ.get("FT_TARGETS", "attn+shexp")]
    rank = int(os.environ.get("FT_RANK", "32"))
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    ))
    model.print_trainable_parameters()

    # What PEFT actually matched, checked rather than assumed. A regex that
    # matches nothing trains an adapter full of zeros for two hours and reports
    # success; one that matches the linear-attention projections trains
    # something that convert_lora_to_gguf cannot map afterwards.
    adapted = sorted({n.rsplit(".lora_A", 1)[0]
                      for n, _ in model.named_modules() if n.endswith("lora_A")})
    forbidden = [n for n in adapted if "linear_attn" in n]
    if forbidden:
        raise SystemExit(
            "LoRA matched linear-attention projections, which cannot be "
            f"converted to GGUF: {forbidden[:3]} ... ({len(forbidden)} total)")
    n_attn = len([n for n in adapted if ".self_attn." in n])
    n_shexp = len([n for n in adapted if ".shared_expert." in n])
    print(f"adapted: {len(adapted)} modules "
          f"({n_attn} attention, {n_shexp} shared-expert)")
    if not adapted:
        raise SystemExit(f"LoRA matched no module with pattern {targets!r}")
    expected_attn = 4 * (n_layers // interval) if interval else 0
    if expected_attn and n_attn != expected_attn:
        print(f"WARNING: expected {expected_attn} attention modules for "
              f"{n_layers} layers / interval {interval}, matched {n_attn}")

    ds = load_dataset("json", data_files={
        "train": str(DATA / TRAIN_FILE),
        "validation": str(DATA / VAL_FILE),
    })

    def tokenize(batch):
        texts = [
            tok.apply_chat_template(m, tokenize=False,
                                    add_generation_prompt=False)
            for m in batch["messages"]
        ]
        return tok(texts, truncation=True, max_length=MAX_LEN)

    ds = ds.map(tokenize, batched=True,
                remove_columns=ds["train"].column_names)
    truncated = sum(1 for r in ds["train"] if len(r["input_ids"]) == MAX_LEN)
    if truncated:
        print(f"WARNING: {truncated} training samples hit FT_MAX_LEN="
              f"{MAX_LEN} and lost their tail")

    max_steps = int(os.environ.get("FT_MAX_STEPS", "-1"))
    smoke = max_steps > 0

    args = TrainingArguments(
        output_dir=str(OUT),
        # FT_BATCH defaults to 4, not 1. Measured on a B200: one 350-token
        # sample takes ~22 s, of which the arithmetic is ~6 ms — the cost is
        # kernel launches, dominated by the sequential Gated-DeltaNet
        # recurrence in 36 of the 48 layers (the fla + causal-conv1d fast path
        # is rarely installed, and transformers needs both). That recurrence
        # is walked once per FORWARD, not once per sample, so the batch rides
        # along nearly free. The ceiling is the loss, not the model: logits are
        # batch x seq x 151936, upcast to fp32 and backed by a gradient of the
        # same size, so 4 x 2560 tokens is what fits the ~30 GiB a B200 has
        # left after the weights.
        per_device_train_batch_size=int(os.environ.get("FT_BATCH", "4")),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(os.environ.get("FT_ACCUM", "2")),
        num_train_epochs=float(os.environ.get("FT_EPOCHS", "4")),
        max_steps=max_steps,
        learning_rate=float(os.environ.get("FT_LR", "2e-4")),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # adamw_torch, not paged_adamw_8bit: without bitsandbytes in the
        # picture there is no reason to pull it back in for the few million
        # LoRA parameters this trains.
        optim="adamw_torch",
        # Batches of samples of similar length. The corpus runs from 179 to
        # 2364 tokens with a median of 349, so a random batch is padded to its
        # longest member and most of what the card computes is padding. This
        # only reorders samples; the gradient is the same.
        group_by_length=os.environ.get("FT_GROUP_BY_LENGTH", "1") == "1",
        logging_steps=1 if smoke else 5,
        save_strategy="no" if smoke else "epoch",
        eval_strategy="no" if smoke else "epoch",
        # Keep every epoch and end on the best one. 311 samples against 20 M
        # trainable parameters overfits early, and the epoch that scores best
        # on the validation split is not reliably the last: `save_total_limit=2`
        # would already have thrown away epoch 1 and 2 by the time that is
        # visible. A LoRA checkpoint here is ~80 MB, so keeping all four costs
        # nothing worth having.
        save_total_limit=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    from transformers.trainer_utils import get_last_checkpoint
    last = get_last_checkpoint(str(OUT)) if OUT.is_dir() else None
    if last:
        print(f"Resuming from {last}")
    trainer.train(resume_from_checkpoint=last)
    if smoke:
        print("smoke test OK — nothing saved")
        return
    trainer.save_model(str(OUT))
    tok.save_pretrained(str(OUT))
    print(f"LoRA adapter saved -> {OUT}")


if __name__ == "__main__":
    main()
