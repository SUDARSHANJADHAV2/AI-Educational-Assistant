# """
# train_7b_model_guff.py
# =================
# ORIGINAL TRAINING CODE COMMENTED OUT
# (kept for backup/reference)
# """

# import os
# os.environ["UNSLOTH_DISABLE_FUSED_LOSS"] = "1"

# import sys
# import torch
# from unsloth import FastLanguageModel
# from trl import SFTTrainer
# from transformers import TrainingArguments
# from datasets import load_dataset

# =============================================================================
# NEW GGUF EXPORT ONLY SCRIPT
# =============================================================================

import os
os.environ["UNSLOTH_DISABLE_FUSED_LOSS"] = "1"

import torch
from unsloth import FastLanguageModel

# =============================================================================
# CONFIG
# =============================================================================

# Folder containing your already trained merged model
OUTPUT_DIR = "my_7b_model"

# Folder where GGUF will be saved
GGUF_DIR = "my_7b_model_gguf"

# Max sequence length
MAX_SEQ_LEN = 1024

# =============================================================================
# SYSTEM INFO
# =============================================================================

print("\n" + "=" * 60)
print("SYSTEM INFO")
print("=" * 60)

print("GPU Available :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name      :", torch.cuda.get_device_name(0))
    print(
        "VRAM          :",
        round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
            2,
        ),
        "GB",
    )

print("=" * 60)

# =============================================================================
# LOAD MERGED MODEL
# =============================================================================

print(f"\nLoading merged model from '{OUTPUT_DIR}'...\n")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=OUTPUT_DIR,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=False,
)

print("\n✅ Merged model loaded successfully!")

# =============================================================================
# EXPORT TO GGUF
# =============================================================================

print(f"\nExporting GGUF to '{GGUF_DIR}'...")
print("This can take 10–30 minutes depending on disk speed.\n")

model.save_pretrained_gguf(
    GGUF_DIR,
    tokenizer,
    quantization_method="q4_k_m",
)

print("\n🎉 GGUF EXPORT COMPLETE!")
print(f"GGUF location: ./{GGUF_DIR}/")
print("\nYou can now load this GGUF directly into LM Studio.")