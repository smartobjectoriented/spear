"""QLoRA fine-tune of Qwen3-8B on an instruct dataset you supply.

Tuned to fit an 8 GB GPU (RTX 4060 Laptop): 4-bit NF4 base + LoRA adapters,
gradient checkpointing, batch size 1 with accumulation, paged 8-bit optimizer.

Run inside the workspace venv:
    source /opt/llm/qwen3-finetune/activate.sh
    python scripts/train_qlora.py
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

FT_HOME = Path("/opt/llm/qwen3-finetune")
BASE_MODEL = os.environ.get("BASE_MODEL_DIR", str(FT_HOME / "models" / "Qwen3-8B"))
DATA = FT_HOME / "data"
OUT = FT_HOME / "checkpoints" / os.environ.get(
    "FT_RUN_NAME", "qlora-spear-v1")
# Keep short to bound activation memory on 8 GB. The loss over a 152k-token
# vocab is the main memory cost, so this scales peak usage almost linearly.
MAX_LEN = int(os.environ.get("FT_MAX_LEN", "256"))


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 4-bit NF4 quantization of the base weights. llm_int8_skip_modules=[]
    # forces the lm_head to be quantized too (it's kept in bf16 by default,
    # costing ~0.9 GB) — essential to fit an 8 GB GPU. No real quality loss
    # since the served GGUF quantizes the lm_head anyway.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=[],
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    # Minimal k-bit prep. peft's prepare_model_for_kbit_training upcasts the
    # large embedding/lm_head (~0.6 B params) to fp32 and OOMs an 8 GB GPU, so
    # do it by hand: freeze base weights, upcast ONLY the tiny 1-D params
    # (layernorms) for stability, and enable gradient checkpointing.
    for p in model.parameters():
        p.requires_grad_(False)
        if p.ndim == 1:
            p.data = p.data.to(torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Attention-only LoRA to fit 8 GB: the MLP projections (gate/up/down) are
    # the widest layers and dominate activation memory; adapting only attention
    # keeps peak memory in budget. Set FT_LORA_ALL=1 to also target the MLP
    # (needs more VRAM / a shorter MAX_LEN).
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if os.environ.get("FT_LORA_ALL") == "1":
        targets += ["gate_proj", "up_proj", "down_proj"]
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={
        "train": str(DATA / "instruct_train.jsonl"),
        "validation": str(DATA / "instruct_val.jsonl"),
    })

    def tokenize(batch):
        texts = [
            tok.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in batch["messages"]
        ]
        return tok(texts, truncation=True, max_length=MAX_LEN)

    ds = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # Optional overrides for a quick smoke test:
    #   FT_MAX_STEPS=2 python scripts/train_qlora.py
    max_steps = int(os.environ.get("FT_MAX_STEPS", "-1"))
    epochs = float(os.environ.get("FT_EPOCHS", "3"))
    smoke = max_steps > 0

    args = TrainingArguments(
        output_dir=str(OUT),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
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
    # Auto-resume from the last epoch checkpoint if one exists (e.g. after a
    # crash or a suspend that killed the CUDA context mid-run).
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
