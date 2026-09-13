"""Measure what one training step of THIS config actually costs on THIS card.

Not a size estimate. It loads the real checkpoint with the real quantization
config, attaches the real LoRA adapter, and runs one forward + backward at the
real sequence length, then reports the peak. Weights-only numbers answer the
wrong question: at sequence_len 32768 the activations and the logits are a
large share of the peak, and they are exactly what a table cannot predict.

The config is not restated here -- it is imported from training_bundle, the
same object that writes the Axolotl YAML. If the two ever disagree, the
measurement is worthless, so there is only one of them.

Driven by cloud/load_preflight.sh, which owns the card and the inference
server. Env: BASE_MODEL, MOE_QUANT (1/0), LOAD_DTYPE (4bit/bf16).

LAYOUT_ONLY=1 answers the cheap half -- which expert layout the installed
transformers builds, and therefore whether load_in_4bit reaches the experts --
from config.json alone, no weights, no GPU, no disk.
"""
import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

_FIELDS = ("sequence_len", "micro_batch_size", "gradient_accumulation_steps",
           "learning_rate", "lora_r", "lora_alpha", "gradient_checkpointing")
_BUNDLE = Path(__file__).resolve().parents[2] / "spear" / "training_bundle.py"


def _from_yaml(path):
    """The config as SHIPPED: the bundle's own axolotl-sft.yml.

    Preferred when present, because it is the file the training command will
    actually read -- measuring anything else measures a different run.
    """
    text = Path(path).read_text()
    values = {}
    for field in _FIELDS:
        match = re.search(rf"^{field}:\s*(\S+)", text, re.M)
        if match:
            values[field] = ast.literal_eval(match.group(1)) if match.group(1)[0].isdigit() \
                else match.group(1)
    base = re.search(r"^base_model:\s*(\S+)", text, re.M)
    return SimpleNamespace(**values), base.group(1) if base else None, str(path)


# Last resort only -- see _from_source(). Kept in sync by hand with
# training_bundle.AxolotlProfile, which is exactly why it is the last resort.
_FALLBACK = dict(sequence_len=32768, micro_batch_size=1,
                 gradient_accumulation_steps=16, learning_rate=0.0001,
                 lora_r=16, lora_alpha=32, gradient_checkpointing="true")
_FALLBACK_BASE = "Qwen/Qwen3-Coder-Next"


def _from_source():
    """Otherwise the AxolotlProfile defaults, read from the dataclass SOURCE.

    Parsed with ast rather than imported: training_bundle drags in the whole
    harness (openai, chromadb, ...), none of which belongs on a training host,
    and a preflight that cannot run where the training runs is decoration. The
    values still come from the one file that writes the YAML, so the two cannot
    drift apart silently.

    A training host may carry only the qwen3-finetune tree, with no harness
    beside it. That is not a reason to refuse to measure -- it is a reason to
    say out loud which numbers are being measured, so nobody reads the verdict
    as covering a config it never saw.
    """
    if not _BUNDLE.is_file():
        return SimpleNamespace(**_FALLBACK), _FALLBACK_BASE, (
            f"built-in defaults ({_BUNDLE} absent) -- pass CONFIG_YAML=<bundle>"
            f"/configs/axolotl-sft.yml to measure the config you will train")
    tree = ast.parse(_BUNDLE.read_text())
    values, base = {}, None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in ("AxolotlProfile",
                                                            "SourceModelProfile"):
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and item.value is not None:
                    name = item.target.id
                    try:
                        literal = ast.literal_eval(item.value)
                    except ValueError:
                        continue
                    if name in _FIELDS:
                        values[name] = literal
                    elif name == "base_model":
                        base = literal
    return SimpleNamespace(**values), base, f"{_BUNDLE.name}:AxolotlProfile"


if os.environ.get("CONFIG_YAML"):
    PROFILE, _BASE, CONFIG_SOURCE = _from_yaml(os.environ["CONFIG_YAML"])
else:
    PROFILE, _BASE, CONFIG_SOURCE = _from_source()
BASE_MODEL = os.environ.get("BASE_MODEL") or _BASE
MOE_QUANT = os.environ.get("MOE_QUANT", "1") == "1"
LOAD_DTYPE = os.environ.get("LOAD_DTYPE", "4bit")
# The measurement is the point; a step that takes an hour is not one. Sequence
# length drives the activation peak, so it stays at the training value unless
# explicitly lowered -- and if it is lowered, the verdict says so.
SEQ_LEN = int(os.environ.get("SEQ_LEN", PROFILE.sequence_len))


def gib(n_bytes):
    return n_bytes / 1024 ** 3


def smi_used():
    """Peak as the driver sees it: torch's allocator misses cuBLAS and NCCL."""
    try:
        free, total = torch.cuda.mem_get_info()
        return gib(total - free)
    except Exception:
        return float("nan")


def expert_layout(model):
    """Per-expert nn.Linear, or one fused 3-D parameter?

    This is the whole reason quantize_moe_experts exists. bitsandbytes swaps
    nn.Linear modules; it cannot see a 3-D nn.Parameter. Under transformers
    4.57 the experts are Linears and plain load_in_4bit reaches them; under 5.x
    they are fused and it silently does not.
    """
    for name, module in model.named_modules():
        if name.endswith("mlp.experts"):
            for attr in ("gate_up_proj", "down_proj"):
                value = getattr(module, attr, None)
                if isinstance(value, torch.nn.Parameter):
                    return "fused_parameter", tuple(value.shape), str(value.dtype)
            children = [c for c in module.children()]
            if children:
                return "per_expert_modules", (len(children),), str(
                    next(module.parameters()).dtype)
    return "unknown", (), ""


def layout_only():
    """The pivotal question, answered for the price of one config.json.

    Whether load_in_4bit reaches the routed experts is a property of the
    installed transformers, not of the weights: it depends on whether
    Qwen3NextExperts builds per-expert nn.Linear or one fused 3-D
    nn.Parameter. Materialising the model on the meta device settles it with
    zero bytes downloaded and zero VRAM -- which is what makes it runnable
    before the disk is grown.

    The parameter counts that follow are exact (from the shapes), so the
    weight footprint is arithmetic, not an estimate. The training PEAK still
    is not: that needs the real load and the real step.
    """
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    print(f"== layout probe ==\n   {BASE_MODEL}, transformers {transformers.__version__}")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    layout, shape, dtype = expert_layout(model)
    expert_params = sum(p.numel() for name, p in model.named_parameters()
                        if ".mlp.experts." in name)
    total_params = sum(p.numel() for p in model.parameters())
    other = total_params - expert_params
    print(f"   routed experts: {layout} {shape or ''}")
    print(f"   {expert_params / 1e9:.1f}B parameters in routed experts, "
          f"{other / 1e9:.1f}B elsewhere ({total_params / 1e9:.1f}B total)")
    reachable = layout == "per_expert_modules"
    print(f"   bitsandbytes replaces nn.Linear, so load_in_4bit reaches them "
          f"on this stack: {reachable}")
    bf16_all = gib(total_params * 2)
    nf4_all = gib(total_params * 0.5 + total_params * 0.03)
    mixed = gib(expert_params * 2 + other * 0.53)
    print(f"== weight footprint (exact arithmetic, no activations) ==")
    print(f"   everything bf16:            {bf16_all:.0f} GiB")
    print(f"   everything NF4:             {nf4_all:.0f} GiB")
    print(f"   experts bf16, rest NF4:     {mixed:.0f} GiB   <- the failing config")
    if reachable:
        print("\nOn THIS transformers, a plain load_in_4bit is enough; "
              "quantize_moe_experts is the belt to that pair of braces and "
              "costs nothing.")
    else:
        print("\nOn THIS transformers, a plain load_in_4bit silently leaves "
              "the experts in bf16. quantize_moe_experts is mandatory.")
    return 0


def _apply_axolotl_moe_patch():
    """Install axolotl's on-load MoE quantization, or say plainly that it is not
    installed.

    Returns a callable that reports how many expert parameters were actually
    quantized -- the one honest check that the flag did something, rather than
    a layout guess made before the weights arrived.
    """
    try:
        from axolotl.monkeypatch.moe_quant import (
            get_moe_quantized_count, patch_moe_quantization_on_load,
            patch_peft_target_parameters_matching,
        )
    except ImportError as exc:
        print(f"   axolotl MoE quantization unavailable ({exc})")
        print("   -> the experts will load however this transformers builds "
              "them. On 4.x they are per-expert nn.Linear and bitsandbytes "
              "reaches them unaided; on 5.x they are fused and it does not. "
              "The layout check below decides, and refuses to call the result "
              "a measurement of the declared config if it was the other one.")
        return lambda: 0
    cfg = SimpleNamespace(load_in_8bit=False, load_in_4bit=True,
                          bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    patch_moe_quantization_on_load(cfg)
    patch_peft_target_parameters_matching()
    print("   axolotl MoE on-load quantization: patched")
    return get_moe_quantized_count


def main():
    if os.environ.get("LAYOUT_ONLY") == "1":
        return layout_only()
    if not torch.cuda.is_available():
        print("No CUDA device visible -- nothing to measure.", file=sys.stderr)
        return 2
    import transformers
    from transformers import AutoModelForCausalLM

    name = torch.cuda.get_device_name(0)
    total = gib(torch.cuda.get_device_properties(0).total_memory)
    print(f"== target ==\n   {name}, {total:.1f} GiB total, "
          f"{gib(torch.cuda.mem_get_info()[0]):.1f} GiB free")
    print(f"   transformers {transformers.__version__}, torch {torch.__version__}")
    print(f"== config (from {CONFIG_SOURCE}) ==")
    print(f"   {BASE_MODEL}  dtype={LOAD_DTYPE}  quantize_moe_experts={MOE_QUANT}")
    print(f"   seq_len={SEQ_LEN} micro_batch={PROFILE.micro_batch_size} "
          f"lora_r={PROFILE.lora_r} checkpointing={PROFILE.gradient_checkpointing}")
    if SEQ_LEN != PROFILE.sequence_len:
        print(f"   !! seq_len lowered from {PROFILE.sequence_len}: the peak below "
              f"UNDERSTATES the training peak")

    quantized_count = lambda: 0
    kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0},
              "attn_implementation": os.environ.get("ATTN_IMPL", "sdpa")}
    if LOAD_DTYPE == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if MOE_QUANT:
            # Axolotl's patch is what makes load_in_4bit reach fused experts:
            # it wraps transformers' set_param_for_module and quantizes every
            # >=3-D CUDA parameter as it lands, and disables the caching
            # allocator warmup that would otherwise pre-reserve the bf16 size
            # and eat the saving it just made. Reproduced by CALLING axolotl,
            # not by reimplementing it -- a preflight that quantizes
            # differently from the trainer measures a run nobody will make.
            quantized_count = _apply_axolotl_moe_patch()

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)
    load_seconds = time.time() - started
    layout, shape, dtype = expert_layout(model)
    weights_peak = gib(torch.cuda.max_memory_allocated())
    experts_quantized = quantized_count()
    print(f"== weights loaded in {load_seconds:.0f}s ==")
    print(f"   routed experts: {layout} {shape or ''} {dtype}")
    print(f"   expert parameters quantized on load: {experts_quantized}")
    print(f"   allocated {gib(torch.cuda.memory_allocated()):.1f} GiB, "
          f"peak {weights_peak:.1f} GiB, driver sees {smi_used():.1f} GiB")

    # "Did the flag do anything" is answered by the count when axolotl is
    # present, and by the layout when it is not. Either way the answer is
    # taken after the weights landed, not predicted before.
    honest = not (MOE_QUANT and LOAD_DTYPE == "4bit" and not experts_quantized
                  and layout == "fused_parameter" and "bfloat16" in dtype)
    if not honest:
        print("\n!! The experts loaded FUSED and in bf16, and nothing "
              "quantized them: this is the config that does not fit, not the "
              "one the bundle emits. Install axolotl and re-run.",
              file=sys.stderr)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if LOAD_DTYPE == "4bit":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=PROFILE.gradient_checkpointing != "false")
    model = get_peft_model(model, LoraConfig(
        r=PROFILE.lora_r, lora_alpha=PROFILE.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    if PROFILE.gradient_checkpointing != "false":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"== adapter ==\n   {trainable / 1e6:.1f}M trainable parameters")

    from torch.optim import AdamW
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=PROFILE.learning_rate)
    ids = torch.randint(0, 1000, (PROFILE.micro_batch_size, SEQ_LEN), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    model.train()
    step_failed = None
    started = time.time()
    try:
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as exc:
        step_failed = str(exc).splitlines()[0]
    step_seconds = time.time() - started
    step_peak = gib(torch.cuda.max_memory_allocated())
    driver_peak = smi_used()

    print(f"== one training step ({step_seconds:.0f}s) ==")
    if step_failed:
        print(f"   OUT OF MEMORY: {step_failed}")
    else:
        print(f"   loss {loss.item():.3f}")
    print(f"   step peak {step_peak:.1f} GiB allocated, driver {driver_peak:.1f} GiB "
          f"of {total:.1f} GiB")

    verdict = ("OOM" if step_failed else
               "FITS" if honest else "MEASURED_THE_WRONG_CONFIG")
    headroom = total - driver_peak
    if verdict == "FITS":
        print(f"\nVERDICT: FITS -- {headroom:.1f} GiB headroom at seq_len {SEQ_LEN}, "
              f"micro_batch {PROFILE.micro_batch_size}."
              + ("" if headroom > 8 else " Thin: raising sequence length, batch "
                 "size or LoRA targets will not survive."))
    else:
        print(f"\nVERDICT: {verdict}")

    report = {
        "base_model": BASE_MODEL, "gpu": name, "total_vram_gib": round(total, 1),
        "load_dtype": LOAD_DTYPE, "quantize_moe_experts": MOE_QUANT,
        "expert_layout": layout, "expert_dtype": dtype,
        "experts_quantized_on_load": experts_quantized,
        "sequence_len": SEQ_LEN, "micro_batch_size": PROFILE.micro_batch_size,
        "lora_r": PROFILE.lora_r,
        "weights_peak_gib": round(weights_peak, 1),
        "step_peak_gib": round(step_peak, 1),
        "driver_peak_gib": round(driver_peak, 1),
        "transformers": transformers.__version__, "torch": torch.__version__,
        "verdict": verdict, "measured_the_declared_config": honest,
    }
    out = Path(os.environ.get("PREFLIGHT_REPORT", "load-preflight.json"))
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"   report: {out}")

    del model, optimizer, ids
    torch.cuda.empty_cache()
    return 0 if verdict == "FITS" else 1


if __name__ == "__main__":
    sys.exit(main())
