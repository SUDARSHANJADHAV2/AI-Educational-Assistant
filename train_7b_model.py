"""
train_7b_model.py
=================
Fine-tune a 7B+ model using BOTH GPU VRAM and System RAM.

Your Setup:
  GPU  : NVIDIA RTX 4050  →  6 GB VRAM
  RAM  : System RAM       →  used for overflow layers

Strategy:
  - 4-bit NF4 quantization   → shrinks 7B model from ~14 GB → ~4 GB
  - device_map="auto"        → automatically splits layers across GPU + CPU RAM
  - max_memory limits        → prevents OOM crashes
  - LoRA rank 32             → more capacity than the old 3.8B script
  - gradient_checkpointing   → trades compute for VRAM savings

Supported base models (pick ONE in the config below):
  ┌─────────────────────────────────────────────────────┬────────────┬───────────┐
  │ Model                                               │ Params     │ Q4 Size   │
  ├─────────────────────────────────────────────────────┼────────────┼───────────┤
  │ unsloth/mistral-7b-instruct-v0.3-bnb-4bit           │ 7B         │ ~4.1 GB   │
  │ unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit         │ 8B         │ ~4.7 GB   │
  │ unsloth/gemma-3-4b-it-bnb-4bit                      │ 4B Gemma3  │ ~2.5 GB   │
  │ unsloth/gemma-3-12b-it-bnb-4bit  (needs 12+ GB)     │ 12B Gemma3 │ ~7.2 GB   │
  │ unsloth/Qwen2.5-7B-Instruct-bnb-4bit                │ 7B Qwen2.5 │ ~4.1 GB   │
  └─────────────────────────────────────────────────────┴────────────┴───────────┘

VRAM + RAM usage estimates for 6 GB GPU:
  7B  Q4  → ~4.1 GB VRAM  (fully GPU, fastest)
  8B  Q4  → ~4.7 GB VRAM  (fully GPU)
  12B Q4  → ~7.2 GB total → ~6 GB VRAM + ~1.2 GB RAM offload
  13B Q4  → ~7.8 GB total → ~6 GB VRAM + ~1.8 GB RAM offload
"""
import os
os.environ["UNSLOTH_DISABLE_FUSED_LOSS"] = "1"

import sys
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset

# =============================================================================
# ▶ CONFIGURATION — Edit this block only
# =============================================================================

# Pick your model from the table above
MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"

# Output folder for the merged model weights (safetensors)
OUTPUT_DIR = "my_7b_model"

# Output folder for the GGUF file
GGUF_DIR   = "my_7b_model_gguf"

# VRAM budget for your RTX 4050 (leave ~500 MB headroom)
GPU_MEMORY  = "5500MiB"

# RAM budget for CPU offload layers (set to how much RAM you have free)
CPU_MEMORY  = "16GiB"

# Max token sequence length per training sample
MAX_SEQ_LEN = 1024

# LoRA rank — higher = more capacity but more VRAM
# r=16 for 3.8B, r=32 for 7B+ (recommended)
LORA_RANK   = 16

# Training steps — increase to 200+ for a full fine-tune
MAX_STEPS   = 200

# Dataset — use Alpaca or point to your own JSON file
# To use your own: load_dataset("json", data_files="my_dataset.json", split="train")
DATASET_NAME = "yahma/alpaca-cleaned"

# =============================================================================
# STEP 1 — Detect hardware
# =============================================================================
print("\n" + "="*60)
print("  SYSTEM INFO")
print("="*60)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  GPU  : {gpu_name}")
    print(f"  VRAM : {vram_gb:.1f} GB")
else:
    print("  GPU  : NOT FOUND — training will be very slow on CPU only!")
    sys.exit("Please ensure CUDA is installed and your GPU is available.")

ram_gb = os.popen("wmic computersystem get TotalPhysicalMemory").read()
print(f"  Model: {MODEL_NAME}")
print(f"  VRAM budget : {GPU_MEMORY}")
print(f"  RAM  budget : {CPU_MEMORY}")
print("="*60 + "\n")

# =============================================================================
# STEP 2 — Load base model with 4-bit quantization + GPU/RAM split
# =============================================================================
print("Loading base model with 4-bit NF4 quantization...")
print(f"  Layers exceeding {GPU_MEMORY} VRAM will be offloaded to CPU RAM automatically.\n")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    dtype          = None,          # auto → bfloat16 on RTX 4050
    load_in_4bit   = True,          # NF4 quantization → fits 7B in 6GB VRAM
    # These two lines enable the GPU + RAM hybrid:
    max_memory     = {
        0:     GPU_MEMORY,          # GPU VRAM limit
        "cpu": CPU_MEMORY,          # CPU RAM limit
    },
    # device_map="auto" is set internally by Unsloth when max_memory is given
)

print(f"\n✅ Model loaded successfully!")

# Show which device each layer landed on
if hasattr(model, 'hf_device_map'):
    devices = list(set(model.hf_device_map.values()))
    gpu_layers = sum(1 for d in model.hf_device_map.values() if d != "cpu")
    cpu_layers = sum(1 for d in model.hf_device_map.values() if d == "cpu")
    print(f"   GPU layers : {gpu_layers}")
    print(f"   CPU layers : {cpu_layers}  (RAM offloaded)")

# =============================================================================
# STEP 3 — Attach LoRA adapters
# =============================================================================
print(f"\nAttaching LoRA adapters (rank={LORA_RANK})...")

model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_RANK,
    lora_alpha     = LORA_RANK,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout   = 0,
    bias           = "none",

    # CHANGE THIS LINE
    use_gradient_checkpointing = True,   # ← replace "unsloth"

    random_state   = 3407,
)

# Count trainable parameters
trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
total       = sum(p.numel() for p in model.parameters())
print(f"✅ LoRA ready | Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params "
      f"({100*trainable/total:.2f}%)\n")

# =============================================================================
# STEP 4 — Prepare dataset
# =============================================================================
print("Loading dataset...")

# Detect model family to use the correct chat template
model_lower = MODEL_NAME.lower()

if "gemma" in model_lower:
    # Gemma uses <start_of_turn> format
    def formatting_prompts_func(examples):
        texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            text = (
                f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
                f"<start_of_turn>model\n{output}<end_of_turn>"
                + tokenizer.eos_token
            )
            texts.append(text)
        return {"text": texts}

elif "llama" in model_lower or "qwen" in model_lower:
    # Llama 3 / Qwen2 use special tokens
    def formatting_prompts_func(examples):
        texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            messages = [
                {"role": "user",      "content": instruction},
                {"role": "assistant", "content": output},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

else:
    # Mistral / default Alpaca format
    ALPACA_PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n{output}"
    )
    def formatting_prompts_func(examples):
        texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            text = ALPACA_PROMPT.format(
                instruction=instruction, output=output
            ) + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)
print(f"✅ Dataset loaded: {len(dataset):,} examples\n")

# =============================================================================
# STEP 5 — Training
# =============================================================================
print("Configuring trainer...")

trainer = SFTTrainer(
    model            = model,
    tokenizer        = tokenizer,
    train_dataset    = dataset,
    dataset_text_field = "text",
    max_seq_length   = MAX_SEQ_LEN,
    dataset_num_proc = 2,
    args = TrainingArguments(
        # ── Batch & Accumulation ──────────────────────────────────────
        per_device_train_batch_size  = 1,   # smaller for 7B (vs 2 for 3.8B)
        gradient_accumulation_steps  = 4,   # effective batch = 1×4 = 4
        # ── Learning Rate ─────────────────────────────────────────────
        warmup_steps                 = 10,
        max_steps                    = MAX_STEPS,
        learning_rate                = 2e-4,
        lr_scheduler_type            = "cosine",  # cosine decay (better than linear for 7B)
        # ── Precision & Memory ────────────────────────────────────────
        fp16                         = not torch.cuda.is_bf16_supported(),
        bf16                         = torch.cuda.is_bf16_supported(),
        # ── Optimizer ─────────────────────────────────────────────────
        optim                        = "adamw_8bit",  # 75% less RAM than standard Adam
        weight_decay                 = 0.01,
        # ── Logging & Output ──────────────────────────────────────────
        logging_steps                = 5,
        output_dir                   = "outputs_7b",
        save_strategy                = "steps",
        save_steps                   = 30,
        # ── Reproducibility ───────────────────────────────────────────
        seed                         = 3407,
        # ── Gradient Clipping (important for stability with 7B) ───────
        max_grad_norm                = 0.3,
    ),
)

print("="*60)
print("  STARTING TRAINING")
print(f"  Model    : {MODEL_NAME}")
print(f"  Steps    : {MAX_STEPS}")
print(f"  Batch    : {1} × {8} accumulation = effective batch 8")
print(f"  LoRA rank: {LORA_RANK}")
print("="*60 + "\n")

trainer_stats = trainer.train()

print(f"\n✅ Training complete!")
print(f"   Total time   : {trainer_stats.metrics['train_runtime']:.0f} seconds")
print(f"   Samples/sec  : {trainer_stats.metrics['train_samples_per_second']:.2f}")

# =============================================================================
# STEP 6 — Save merged model + export to GGUF
# =============================================================================
print(f"\nSaving merged model to '{OUTPUT_DIR}'...")

model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")
print(f"✅ Full merged model saved to ./{OUTPUT_DIR}/")

print(f"\nExporting to GGUF Q4_K_M → '{GGUF_DIR}' ...")
print("  This will take 15–25 minutes...")

model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")

print(f"\n🎉 DONE! Your 7B model is ready.")
print(f"   GGUF file location: ./{GGUF_DIR}/")
print(f"   Load this file in LM Studio → Local Server → Start Server")
print(f"   Then run:  streamlit run app.py")
