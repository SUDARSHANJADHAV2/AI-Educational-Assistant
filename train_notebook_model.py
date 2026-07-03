"""
train_notebook_model.py
=======================
Fine-tunes a model specifically for NotebookLM-style tasks:
Given a context (sources), answer questions accurately and avoid hallucinations.
We'll use a dataset like 'squad_v2' or a RAG instruct dataset.

Instructions for LM Studio:
1. Run this script: `python train_notebook_model.py`
2. It will save the model and export it to GGUF format in the 'notebooklm_gguf' folder.
3. Open LM Studio.
4. Drag and drop the generated .gguf file into LM Studio.
5. Go to the Local Server tab (<-> icon).
6. Start the server (Port 1234).
7. Open `index.html` in your browser and start chatting!
"""

import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
OUTPUT_DIR = "notebooklm_model"
GGUF_DIR   = "notebooklm_gguf"
MAX_SEQ_LEN = 2048 # Increased for context window
LORA_RANK = 32

print("Loading model for NotebookLM fine-tuning...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = LORA_RANK,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# NotebookLM Prompt Template
prompt_template = """You are NotebookLM. Answer the question based ONLY on the provided context. If the answer is not in the context, say "I don't know based on the provided sources."

### Context:
{}

### Question:
{}

### Answer:
{}"""

def format_prompt(examples):
    texts = []
    # Using a subset of squad or a QA dataset
    for context, question, answer in zip(examples['context'], examples['question'], examples['answers']):
        # SQuAD answers are dicts
        ans_text = answer['text'][0] if len(answer['text']) > 0 else "I don't know based on the provided sources."
        text = prompt_template.format(context, question, ans_text) + tokenizer.eos_token
        texts.append(text)
    return { "text": texts }

print("Loading Context-QA Dataset...")
dataset = load_dataset("squad_v2", split="train[:5000]") # Using 5000 samples for quick training
dataset = dataset.map(format_prompt, batched=True)

print("Starting Training...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LEN,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 100,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        output_dir = "outputs_notebooklm",
        seed = 3407,
    ),
)

trainer.train()

print("Saving model and exporting to GGUF...")
# Save merged 16bit model
model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")

# Export to GGUF (Q4_K_M for LM Studio)
model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")

print(f"✅ DONE! Model exported to '{GGUF_DIR}'. Load it into LM Studio to run NotebookLM locally!")
