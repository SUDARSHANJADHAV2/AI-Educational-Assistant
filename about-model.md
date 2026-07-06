# About the AI Model — Notes Pathv

This document explains everything about the AI models used in this project — what they are, how they were built, fine-tuned, exported, and used.

---

## Architecture Overview

This project uses **two separate AI models** working together in a pipeline:

| Role | Model | Parameters | Type |
|------|-------|------------|------|
| Audio Transcription | `openai/whisper-tiny` | ~39M | Encoder-Decoder Transformer (ASR) |
| Note Generation & Chat | `Phi-3-mini-4k-instruct` (fine-tuned) | **3.8 Billion** | Decoder-only Transformer (LLM) |

---

## Part 1: The Speech Model — Whisper Tiny

### What it is
**OpenAI Whisper** is a general-purpose speech recognition model trained on 680,000 hours of multilingual audio. The `tiny` variant is the smallest and fastest version suitable for local/edge usage.

### How it works
Whisper is a **sequence-to-sequence encoder-decoder** Transformer:
1. The audio is converted into a **Mel Spectrogram** (a 2D visual representation of sound frequencies over time)
2. The **Encoder** (12 layers) reads the spectrogram and extracts audio features
3. The **Decoder** (12 layers) auto-regressively generates text tokens from those features

### Why `whisper-tiny`?
| Variant | Parameters | VRAM Required |
|---------|------------|---------------|
| tiny | 39M | ~1 GB |
| base | 74M | ~1 GB |
| small | 244M | ~2 GB |
| medium | 769M | ~5 GB |
| large-v3 | 1550M | ~10 GB |

We chose `tiny` to ensure it runs on any consumer hardware (including CPU-only machines) without requiring a GPU.

---

## Part 2: The Language Model — Phi-3 Mini (3.8B)

This is the core AI brain of the application. It generates the structured lecture notes and powers the chat interface.

---

### What is Phi-3 Mini?

**Microsoft Phi-3 Mini** (`Phi-3-mini-4k-instruct`) is a **3.8 billion parameter** language model developed by Microsoft Research. Despite being small, it punches far above its weight — rivaling models 3–5x its size in reasoning, instruction-following, and language understanding.

- **Full model name**: `unsloth/Phi-3-mini-4k-instruct`
- **Architecture type**: `MistralForCausalLM` (decoder-only)
- **Context window**: 4,096 tokens
- **License**: MIT (fully open source, commercial use allowed)

---

### Exact Model Architecture Parameters (from `config.json`)

These are the raw architectural parameters of the base model:

| Parameter | Value | What it means |
|-----------|-------|----------------|
| `hidden_size` | **3072** | Width of each layer — size of every token's embedding vector |
| `intermediate_size` | **8192** | Inner size of the Feed-Forward Network (FFN) inside each Transformer block |
| `num_hidden_layers` | **32** | Total number of stacked Transformer blocks |
| `num_attention_heads` | **32** | Number of parallel attention heads per layer |
| `num_key_value_heads` | **32** | Used for Grouped Query Attention (GQA) — full MHA here |
| `head_dim` | **96** | Dimension per attention head (`hidden_size / num_heads = 3072 / 32 = 96`) |
| `max_position_embeddings` | **4096** | Maximum sequence/context length |
| `sliding_window` | **2048** | Window size for efficient sliding window attention |
| `vocab_size` | **32,064** | Number of unique tokens the model knows |
| `hidden_act` | `silu` | Activation function — SiLU (Sigmoid Linear Unit) |
| `rms_norm_eps` | `1e-05` | Numerical stability epsilon for RMSNorm layers |
| `rope_theta` | `10000.0` | Base frequency for Rotary Positional Embeddings (RoPE) |
| `torch_dtype` | `bfloat16` | 16-bit Brain Float precision for computation |
| `attention_dropout` | `0.0` | No attention dropout (inference stable) |

---

### How it is 3.8 Billion Parameters

The total parameter count comes from the sum of all weight matrices across the 32 layers:

```
Embedding Layer:
   vocab_size × hidden_size = 32,064 × 3,072 ≈ 98M params

🔁 × 32 Transformer Layers, each containing:
   ├── Self-Attention:
   │    Q, K, V, O projections = 4 × (3072 × 3072) ≈ 37.7M
   ├── Feed-Forward Network (FFN):
   │    gate_proj + up_proj + down_proj = 3 × (3072 × 8192) ≈ 75.5M
   └── RMSNorm (2 per layer, negligible)

Output Head (LM Head):
   hidden_size × vocab_size ≈ 98M params

Total ≈ 98M + 32 × (37.7M + 75.5M) + 98M
      ≈ 98M + 32 × 113.2M + 98M
      ≈ 98M + 3,622M + 98M
      ≈ 3,818M ≈ 3.8 Billion Parameters
```

---

### Fine-Tuning with QLoRA (How We Trained It)

We did **not** train the whole 3.8B model from scratch — that would require ~100+ GPUs and weeks of compute. Instead, we used **QLoRA (Quantized Low-Rank Adaptation)** via the `Unsloth` library to fine-tune it on a consumer RTX 4050 (6GB VRAM).

#### Step 1: Load Base Model in 4-bit Quantization

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-3-mini-4k-instruct",
    max_seq_length = 2048,
    dtype = None,          # auto-detects float16 for RTX 4050
    load_in_4bit = True,   # CRITICAL — compresses model from ~7GB to ~2.5GB
)
```

- `load_in_4bit = True` uses **NF4 (NormalFloat4)** quantization — each weight, normally stored as 16-bit, is compressed to 4-bit, reducing memory by ~4x
- This allows a 3.8B model (which would normally need ~7.5GB VRAM) to fit into 6GB VRAM

#### Step 2: Attach LoRA Adapters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                    # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)
```

**What is LoRA?** Instead of updating all 3.8B weights directly, LoRA injects tiny **trainable rank-decomposition matrices** alongside the frozen original weights. 

Mathematically: instead of updating weight matrix `W` (which is `3072×3072 = 9.4M numbers per layer`), LoRA adds two small matrices `A` (rank 16 × 3072) and `B` (3072 × rank 16), and only trains `A` and `B`.

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `r = 16` | Rank 16 | LoRA inserts 2 matrices of rank 16 per attention layer, training only ~1.6M params instead of 9.4M |
| `lora_alpha = 16` | Scaling factor | `alpha/r = 1.0` — output is scaled equally to original weights |
| `lora_dropout = 0` | No dropout | Optimized for speed; no regularization dropout |
| `bias = "none"` | Skip bias | Don't train bias terms — saves memory |
| `gradient_checkpointing = "unsloth"` | Unsloth's custom checkpointing | Recomputes activations on backward pass instead of storing them → saves ~30% VRAM |

**Result**: We only trained **~1.6% of the total model weights** while keeping the other 98.4% frozen.

#### Step 3: The Dataset & Prompt Format

```
### Instruction:
{your instruction}

### Response:
{expected answer}
```

The dataset used was **`yahma/alpaca-cleaned`** — a cleaned version of the Stanford Alpaca dataset with 52K instruction-following examples. This teaches the model to follow instructions clearly and concisely.

#### Step 4: Training Hyperparameters

| Hyperparameter | Value | Why |
|----------------|-------|-----|
| `per_device_train_batch_size` | 2 | Low to fit within 6GB VRAM |
| `gradient_accumulation_steps` | 4 | Effectively simulates batch size of 8 |
| `learning_rate` | 2e-4 | Standard for LoRA fine-tuning |
| `max_steps` | 60 | Quick fine-tune run |
| `warmup_steps` | 5 | Gradual LR warmup to avoid early instability |
| `optim` | `adamw_8bit` | 8-bit AdamW → uses 75% less memory than standard Adam |
| `lr_scheduler_type` | `linear` | Linearly decays LR to 0 |
| `fp16 / bf16` | auto-detected | Uses BF16 on RTX 4050 for better numeric range |
| `weight_decay` | 0.01 | L2 regularization to prevent overfitting |
| `seed` | 3407 | Reproducibility |

---

### Quantization & Export — GGUF Format

After fine-tuning, the model was exported to **GGUF format** for use in LM Studio:

```python
model.save_pretrained_gguf("lm_studio_model", tokenizer, quantization_method="q4_k_m")
```

**What is Q4_K_M?**
It's a specific GGUF quantization scheme:
- `Q4` = 4-bit weights
- `K` = "k-quants" — mixed precision, where important layers keep higher precision
- `M` = "Medium" size/accuracy tier (balances speed vs quality)

**Result:**

| Format | Size | Notes |
|--------|------|-------|
| Original bfloat16 (safetensors) | ~7.6 GB | Full precision, 2 shard files |
| Q4_K_M GGUF | **~2.15 GB** | 72% smaller, runs on CPU or any GPU |

The final GGUF file is at:
`my_custom_model_gguf/MyBrand/EducationalAssistant/my_custom_model.Q4_K_M.gguf`

---

## How the Two Models Work Together in the App

```
User uploads audio file
        │
        ▼
┌───────────────────────────────┐
│   Whisper Tiny (39M params)   │
│   Converts audio → raw text   │
│   (transcript)                │
└──────────────┬────────────────┘
               │  transcript text
               ▼
┌───────────────────────────────┐
│  Phi-3 Mini 3.8B (fine-tuned) │
│  Takes transcript + prompt    │
│  Generates structured notes   │
│  Also powers the chat UI      │
└───────────────────────────────┘
        │
        ▼
  Structured Markdown Notes
  displayed to user
```

---

## Hardware Requirements

| Mode | Minimum | Recommended |
|------|---------|-------------|
| Transcription (Whisper Tiny) | Any CPU | Any GPU |
| Note Generation (Phi-3 3.8B via Transformers) | 8GB RAM, CPU | 6GB VRAM GPU (RTX 3060+) |
| Fine-tuning training | RTX 4050 6GB VRAM | RTX 3090 / A100 |
| GGUF inference (LM Studio) | 8GB RAM | 8GB VRAM GPU |

---

## Key Libraries & Tools

| Tool | Version | Role |
|------|---------|------|
| `unsloth` | 2026.4.4 | Fast QLoRA fine-tuning (2–5x faster than baseline HuggingFace) |
| `transformers` | 4.57.3 | Model loading, tokenization, inference pipelines |
| `trl` | latest | `SFTTrainer` for supervised fine-tuning |
| `torch` | 2.9.1 | Core deep learning backend |
| `torchaudio` | 2.11.0 | Audio processing utilities |
| `soundfile` | 0.13.1 | Audio file reading |
| `imageio-ffmpeg` | 0.6.0 | Bundled ffmpeg binary for audio decoding |
| `accelerate` | 1.13.0 | Distributed training + mixed precision |
| `streamlit` | 1.53.0 | Web application UI |

---

## References

- [Phi-3 Technical Report — Microsoft](https://arxiv.org/abs/2404.14219)
- [LoRA: Low-Rank Adaptation of LLMs — Hu et al.](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning — Dettmers et al.](https://arxiv.org/abs/2305.14314)
- [Whisper: Robust Speech Recognition — OpenAI](https://arxiv.org/abs/2212.04356)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
