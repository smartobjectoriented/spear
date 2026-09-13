# SPEAR on the RTX PRO 6000 (Blackwell, 96 GB) — deployment

Goal: run a **dense** code model entirely in VRAM for reliable agentic
editing (full-file rewrites, long tool-use), which the laptop's
Qwen3.6-35B-**A3B** MoE (~3B active params/token) cannot sustain.

**Target model: Qwen2.5-Coder-32B-Instruct (dense, ~32B active), Q8_0 GGUF
(~35 GB).** On 96 GB it loads fully on the GPU with huge context headroom.
NB: *Qwen3-Coder* is a MoE A3B (3B active) — it would NOT help; the dense
coder is the 2.5 line.

## What carries over from the laptop
- `spear/` — the whole harness (rag_chat.py, tool-guide.md, rules.d/,
  system-prompt.md) and the ChromaDB corpora (SO3 / lvgl-so3 / micropython-so3).
- The `~/soo/so3/.edgem-rules.md` per-corpus orientation map.
- ✅ RAG + rules + skills + history.
- ❌ NOT the QLoRA adapter (v2/v3) — it is tied to the Qwen3.6-A3B base.
  The SO3 knowledge still reaches the model via RAG. (Re-QLoRA the Coder-32B
  later if you want a parametric boost — same `qwen3-finetune/cloud` pipeline,
  new BASE_MODEL.)

## Steps on the GPU box
1. Build llama.cpp with CUDA (Blackwell = sm_120, needs CUDA ≥ 12.8):
   ```
   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
   cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
   cmake --build build -j --config Release
   ```
2. Copy `/opt/llm/spear/spear` (with its `chromadb/`) and `~/.local/bin/spear-*`
   and `~/soo/so3` (or re-index there). Put llama.cpp at `/opt/llm/llama.cpp`.
3. Fetch the model:  `server/scripts/fetch-model.sh`   (downloads the Q8_0 shards)
4. Activate it:      `spear-model coder32`
   - auto-detects DENSE → no `--n-cpu-moe`, every layer on the GPU
   - auto-disables the A3B adapter (incompatible base)
5. Bigger context (96 GB has room):  `SPEAR_CTX=65536 spear-model coder32`
   (or export SPEAR_CTX in the systemd unit).
6. Run as usual:  `spear-chat`

## Key server flags (handled by server/inference/serve.sh)
- Dense model (no `A3B`/`MoE` in the filename) → **`--n-gpu-layers 99`,
  no `--n-cpu-moe`** → 100 % on the GPU, fast.
- MoE model → keeps `--n-cpu-moe` (laptop behaviour).
- Override: `SPEAR_SERVER_NCPUMOE=0` forces all-GPU; `SPEAR_SERVER_NGL`,
  `SPEAR_SERVER_CTX`, `SPEAR_SERVER_THREADS` tune the rest. They live in
  `server/config/server.conf`; the environment overrides it.

## Expected gain
The dense 32B sustains long, exact generation, so the "improve ping.c" class
of task (full edits, multi-step agentic work) should complete reliably —
unlike the A3B which stops mid-output (~1.3k tokens) on big edits. Speed:
~30-60+ t/s fully in VRAM (vs ~7 t/s CPU-offloaded on the laptop).

## Quick local prep now (laptop, before the box arrives)
- `server/scripts/fetch-model.sh` can pre-stage the GGUF on the laptop (84 GB free,
  ~35 GB needed) to copy over later — OR just download on the box (faster
  pipes). The laptop's 8 GB GPU can't run it fast (it would CPU-offload), so
  only use the laptop to *prepare*, not to serve the 32B.
