"""
This script demonstrates how to fine-tune your own Custom AI Model
using your 6GB VRAM RTX 4050, and then export it for LM Studio.

We use 'Unsloth' and 'TRL', which specialize in fast, low-memory QLoRA fine-tuning.
"""

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

# ==========================================
# 1. LOAD BASE MODEL (Optmized for 6GB VRAM)
# ==========================================
max_seq_length = 2048 # Lower sequence length = less VRAM used
dtype = None # Auto-detects float16 for RTX 4050
load_in_4bit = True # CRITICAL for 6GB VRAM

# We use Phi-3 as it is incredibly smart but very small (3.8B parameters)
model_name = "unsloth/Phi-3-mini-4k-instruct" 

print("Loading Base Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# ==========================================
# 2. ADD LoRA ADAPTERS (The Fine-Tuning layer)
# ==========================================
# We don't train the whole model (Impossible on 6GB VRAM). 
# We only train 1-10% of the weights using LoRA (Low-Rank Adaptation).
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # Crucial for saving VRAM
)

# ==========================================
# 3. PREPARE YOUR DATASET
# ==========================================
# We use an example prompt format. You will format your own data like this!
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS token so the model learns when to stop

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# Example: Loading a public dataset. 
# REPLACE THIS with your own local JSONL file later: load_dataset("json", data_files="my_dataset.json")
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# ==========================================
# 4. TRAIN YOUR MODEL
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2, # Keep low for 6GB VRAM!
        gradient_accumulation_steps = 4, # Simulates a larger batch size
        warmup_steps = 5,
        max_steps = 60, # Increase this later for a full training run
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

print("Starting Training! 🚀")
trainer.train()

# ==========================================
# 5. EXPORT TO LM STUDIO (.gguf)
# ==========================================
print("Training complete! Exporting directly to LM Studio compatible GGUF format...")

# This automatically converts your custom LoRA into a standardized Q4_K_M GGUF file.
model.save_pretrained_gguf("my_custom_model", tokenizer, quantization_method="q4_k_m")

print("✅ Success! You can now drag the 'my_custom_model-unsloth.Q4_K_M.gguf' file straight into LM Studio.")
