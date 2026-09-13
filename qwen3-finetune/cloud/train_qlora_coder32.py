"""QLoRA fine-tune of the DENSE Qwen2.5-Coder-32B-Instruct.

Objective here: EDIT DISCIPLINE (answer "improve X" with targeted edit_file
calls that preserve all functionality), dataset built by
scripts/build_edit_discipline.py. But the script is generic — point
FT_TRAIN_FILE/FT_VAL_FILE at any {"messages": [...]} jsonl.

Dense model: standard qwen2 arch (no MoE, no remote code). 4-bit NF4 base is
~18 GB; LoRA on attn+mlp fits comfortably on a 48 GB+ GPU (trivial on the
96 GB RTX PRO 6000). Serve the adapter with vLLM --enable-lora (no GGUF).

Env knobs:
    BASE_MODEL     HF id (default Qwen/Qwen2.5-Coder-32B-Instruct)
    FT_RUN_NAME    output dir name (default qlora-coder32-edit-v1)
    FT_TRAIN_FILE  / FT_VAL_FILE   jsonl names under FT_DATA_DIR
    FT_TARGETS     attn | attn+mlp (default attn+mlp)
    FT_RANK FT_EPOCHS FT_MAX_LEN FT_BATCH FT_ACCUM FT_MAX_STEPS
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

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("FT_DATA_DIR", HERE.parent / "data"))
OUT = Path(os.environ.get("FT_OUT_DIR", HERE.parent / "checkpoints")) / \
    os.environ.get("FT_RUN_NAME", "qlora-coder32-edit-v1")
MAX_LEN = int(os.environ.get("FT_MAX_LEN", "4096"))
TRAIN_FILE = os.environ.get("FT_TRAIN_FILE", "edit_discipline_train.jsonl")
VAL_FILE = os.environ.get("FT_VAL_FILE", "edit_discipline_val.jsonl")

_PRESETS = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attn+mlp": ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"],
}


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
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
    )
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
        if p.ndim == 1:                       # keep norms in fp32 for stability
            p.data = p.data.to(torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    targets = _PRESETS[os.environ.get("FT_TARGETS", "attn+mlp")]
    rank = int(os.environ.get("FT_RANK", "16"))
    lora = LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={
        "train": str(DATA / TRAIN_FILE),
        "validation": str(DATA / VAL_FILE),
    })

    def tokenize(batch):
        texts = [tok.apply_chat_template(m, tokenize=False,
                                         add_generation_prompt=False)
                 for m in batch["messages"]]
        return tok(texts, truncation=True, max_length=MAX_LEN)

    ds = ds.map(tokenize, batched=True,
                remove_columns=ds["train"].column_names)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    max_steps = int(os.environ.get("FT_MAX_STEPS", "-1"))
    epochs = float(os.environ.get("FT_EPOCHS", "4"))
    smoke = max_steps > 0

    args = TrainingArguments(
        output_dir=str(OUT),
        per_device_train_batch_size=int(os.environ.get("FT_BATCH", "1")),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(os.environ.get("FT_ACCUM", "8")),
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=1 if smoke else 5,
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
