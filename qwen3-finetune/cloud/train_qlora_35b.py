"""QLoRA fine-tune of Qwen3.6-35B-A3B on an instruct dataset you supply.

CLOUD script — needs a >= 48 GB GPU (L40S/A6000/A100-80G; A100/H100 ideal).
The 4-bit NF4 base alone is ~20 GB, and the qwen35moe forward is heavier
than the old Coder (256 experts + 1 ALWAYS-ON shared expert per token), so
this does NOT fit the 8 GB laptop GPU. Rent a pod (see runpod_train.sh) and
bring back only the LoRA adapter (~200 MB).

MoE note: LoRA targets the ATTENTION projections only. The 256 routed
experts, the shared expert and the routers are left frozen — adapting them
is memory-hungry and risks destabilizing the routing, while attention
adapters carry the style/domain signal just fine.

IMPORTANT — confirm the HF base repo id before launching. The local model
was downloaded as GGUF (Qwen3.6-35B-A3B-Q8_0.gguf); set BASE_MODEL to the
exact HF id it came from (e.g. an official Qwen/... or an Unsloth bf16 repo).
Override with:  BASE_MODEL=<hf-id> bash runpod_train.sh

Env knobs (same convention as the local 8B script):
    BASE_MODEL    HF repo id of the bf16 base (default Qwen/Qwen3.6-35B-A3B)
    FT_RUN_NAME   output dir name   (default qlora-spear-35b-v1)
    FT_EPOCHS     epochs            (default 3)
    FT_MAX_LEN    max sample tokens (default 1024 — we have VRAM here)
    FT_MAX_STEPS  >0 = smoke test
"""
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.6-35B-A3B")
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("FT_DATA_DIR", HERE.parent / "data"))
OUT = Path(os.environ.get("FT_OUT_DIR", HERE.parent / "checkpoints")) / \
    os.environ.get("FT_RUN_NAME", "qlora-spear-35b-v1")
MAX_LEN = int(os.environ.get("FT_MAX_LEN", "1024"))
# Same convention as the dense script: this file is generic over any
# {"messages": [...]} jsonl, and hardcoding one dataset name was the only
# thing tying it to the instruct corpus.
TRAIN_FILE = os.environ.get("FT_TRAIN_FILE", "instruct_train.jsonl")
VAL_FILE = os.environ.get("FT_VAL_FILE", "instruct_val.jsonl")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # MoE forward uses the fast grouped_mm path (torch._grouped_mm).
        # NOTE: this requires torch >= 2.9 on Blackwell (sm_120) — torch 2.8's
        # _grouped_mm is Hopper-only (CC == 9.0) and crashes here. The pod runs
        # torch 2.11 (see runpod_train.sh), so the default path is fine; force
        # experts_implementation="eager" only if you must run on torch 2.8.
    )
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
        if p.ndim == 1:
            p.data = p.data.to(torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # LoRA targets. Default = attention only. For knowledge injection (v3+),
    # also adapt the shared-expert MLP (gate/up/down_proj) — it is a real
    # nn.Linear that runs on EVERY token, so it can carry new fact->symbol
    # associations that attention-only adapters cannot. The 256 ROUTED experts
    # are nn.Parameter tensors (gate_up_proj), NOT nn.Linear, so PEFT never
    # touches them — VRAM stays modest. Override with FT_TARGETS / FT_RANK.
    #   FT_TARGETS=attn          -> q,k,v,o
    #   FT_TARGETS=attn+mlp      -> q,k,v,o + gate,up,down (default v3)
    _PRESETS = {
        "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "attn+mlp": ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    }
    targets = _PRESETS[os.environ.get("FT_TARGETS", "attn+mlp")]
    rank = int(os.environ.get("FT_RANK", "32"))
    lora = LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

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
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    max_steps = int(os.environ.get("FT_MAX_STEPS", "-1"))
    epochs = float(os.environ.get("FT_EPOCHS", "3"))
    smoke = max_steps > 0

    args = TrainingArguments(
        output_dir=str(OUT),
        # FT_BATCH x FT_ACCUM = 16 effective. Bigger micro-batches mean
        # fewer forward passes per step — decisive on pods with slow CPU
        # cores, since the HF MoE forward loops over experts in Python
        # (heavier here: 256 experts + 1 shared expert).
        per_device_train_batch_size=int(os.environ.get("FT_BATCH", "2")),
        # eval batch defaults to 8 → the loss over the large vocab OOMs a
        # 24 GB card during evaluation. Keep it as small as training.
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=int(os.environ.get("FT_ACCUM", "8")),
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=1 if smoke else 10,
        save_strategy="no" if smoke else "epoch",
        eval_strategy="no" if smoke else "epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        data_collator=collator,
    )
    from transformers.trainer_utils import get_last_checkpoint
    last = get_last_checkpoint(str(OUT)) if OUT.is_dir() else None
    if last:
        print(f"Resuming from {last}")
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(OUT))
    tok.save_pretrained(str(OUT))
    print(f"LoRA adapter saved -> {OUT}")


if __name__ == "__main__":
    main()
