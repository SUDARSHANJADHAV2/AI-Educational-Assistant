# AI Educational Assistant
### Lecture to Structured Notes — Powered by a Custom Local AI Model

---

## Slide 1 — The Problem

Students today record hours of lectures but rarely have time to review them.

- A 1-hour lecture recording → hours spent writing notes manually
- Online AI tools require internet, subscriptions, and share your data
- Generic AI chat assistants are not specialized for education
- No offline, privacy-first, student-focused solution existed

> **The question:** Can we build an AI that listens to a lecture and instantly writes structured, exam-ready notes — running entirely on a student's own laptop?

---

## Slide 2 — Our Solution

**AI Educational Assistant** — a two-component local AI system that:

1. **Listens** to any audio or video lecture (uploaded file or live microphone)
2. **Transcribes** speech to text using a local Whisper model on the GPU
3. **Generates** beautifully structured study notes using a custom-trained language model
4. **Chats** with the student about their notes or any educational topic
5. **Works 100% offline** — no internet, no API keys, no subscriptions, no data sharing

---

## Slide 3 — Live Demo Flow

```
Student uploads lecture audio (MP3 / WAV / M4A)
              │
              ▼
   Whisper STT runs on GPU
   "Today we'll cover Newton's laws of motion..."
              │
              ▼
   Custom Phi-3 Model generates:

   # Newton's Laws of Motion
   ## Key Points
   - First Law: An object at rest stays at rest...
   ## Explanation
   - Force, mass and acceleration relate as F = ma...
   ## Examples
   - A hockey puck sliding on ice...
   ## Summary
   - Newton's three laws define classical mechanics...
              │
              ▼
   Student downloads notes as .md file
   Student chats: "What does F = ma mean?"
   Model replies instantly — fully offline
```

---

## Slide 4 — System Architecture

```
┌─────────────────────────────────────────────────────┐
│              Streamlit Web UI  (app.py)              │
│                                                      │
│   ┌──────────────┐   ┌────────────┐   ┌──────────┐  │
│   │ Upload Audio │   │ Record Mic │   │ YouTube  │  │
│   │ MP3/WAV/M4A  │   │ (live)     │   │ URL      │  │
│   └──────┬───────┘   └─────┬──────┘   └────┬─────┘  │
│          └─────────────────┴───────────────┘         │
│                            │                         │
│                            ▼                         │
│   ┌─────────────────────────────────────────────┐    │
│   │  faster-whisper (Whisper Small, float16)    │    │
│   │  Runs on NVIDIA RTX 4050 GPU                │    │
│   │  → Converts audio to raw text transcript    │    │
│   └─────────────────────┬───────────────────────┘    │
│                         │   transcript text           │
│                         ▼                            │
│   ┌─────────────────────────────────────────────┐    │
│   │  LM Studio Local Server  (localhost:1234)   │    │
│   │  Loads: my_custom_model.Q4_K_M.gguf         │    │
│   │  → Generates structured notes               │    │
│   │  → Powers the chat interface                │    │
│   └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

**Key principle:** Every component runs locally. Zero cloud calls. Zero data leaves your machine.

---

## Slide 5 — The Two AI Models

| | Model 1 | Model 2 |
|---|---|---|
| **Name** | OpenAI Whisper Small | Microsoft Phi-3 Mini (custom fine-tuned) |
| **Role** | Speech → Text | Text → Structured Notes + Chat |
| **Parameters** | 244 Million | **3.8 Billion** |
| **Library** | `faster-whisper` | LM Studio (GGUF format) |
| **Precision** | float16 | Q4_K_M (4-bit quantized) |
| **Runs on** | RTX 4050 GPU | RTX 4050 GPU (via LM Studio) |
| **Internet required?** | No | No |
| **Custom trained?** | No (used as-is) | **Yes — fine-tuned by us** |

---

## Slide 6 — Model 1: Whisper Small

**What it is:**
OpenAI Whisper is trained on 680,000 hours of multilingual audio. The `small` variant provides the best balance of speed and accuracy for a local setup.

**How it works:**

```
Audio File
    │
    ▼
Mel Spectrogram (2D visual of sound frequencies over time)
    │
    ▼
Encoder (12 Transformer layers) → extracts audio features
    │
    ▼
Decoder (12 Transformer layers) → generates text tokens one by one
    │
    ▼
"Today we will cover Newton's laws of motion..."
```

**In code:**
```python
from faster_whisper import WhisperModel

stt_model = WhisperModel("small", device="cuda", compute_type="float16")
segments, _ = stt_model.transcribe(audio_file_path, beam_size=5)
transcript  = " ".join(seg.text for seg in segments)
```

---

## Slide 7 — Model 2: Phi-3 Mini (Base Architecture)

**Microsoft Phi-3 Mini** is a 3.8B parameter decoder-only Transformer built on the Mistral architecture.

### Architecture Parameters (from `config.json`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `hidden_size` | 3072 | Width of each token's embedding vector |
| `intermediate_size` | 8192 | Size of Feed-Forward Network inside each block |
| `num_hidden_layers` | **32** | 32 stacked Transformer blocks |
| `num_attention_heads` | 32 | Parallel attention heads per layer |
| `head_dim` | 96 | Dimension per head (3072 ÷ 32) |
| `max_position_embeddings` | 4096 | Maximum context window (tokens) |
| `sliding_window` | 2048 | Efficient attention window size |
| `vocab_size` | 32,064 | Unique tokens the model understands |
| `hidden_act` | SiLU | Sigmoid Linear Unit activation |
| `torch_dtype` | bfloat16 | Brain Float 16 precision |

---

## Slide 8 — Why it's 3.8 Billion Parameters

The total comes from summing all weight matrices across 32 layers:

```
Embedding Layer:
   32,064 tokens × 3,072 dims  =  98M parameters

× 32 Transformer Blocks:
   Self-Attention  (Q, K, V, O projections):
      4 × (3072 × 3072)         =  37.7M per layer
   Feed-Forward Network (gate, up, down):
      3 × (3072 × 8192)         =  75.5M per layer
   Total per layer              = 113.2M

Output Head (LM Head):
   3,072 × 32,064               =  98M parameters

─────────────────────────────────────────────────
Grand Total:
   98M  +  (32 × 113.2M)  +  98M
=  98M  +  3,622M          +  98M
=  3,818M  ≈  3.8 Billion Parameters  ✅
```

---

## Slide 9 — Fine-Tuning the Model (QLoRA)

We trained the model on a single **RTX 4050 (6GB VRAM)** using a technique called **QLoRA**.

**The challenge:** A 3.8B model normally needs ~7.5GB VRAM. We only have 6GB.

**The solution — three tricks combined:**

| Trick | How it helps |
|-------|-------------|
| **4-bit NF4 Quantization** | Compresses model from 7.5GB → 2.5GB in VRAM |
| **LoRA Adapters** | Only trains 1.6% of weights — rest are frozen |
| **8-bit AdamW Optimizer** | Reduces optimizer memory by 75% |

Together these make training a 3.8B model possible on a consumer laptop GPU.

---

## Slide 10 — LoRA: How Fine-Tuning Works

Instead of updating all 3.8B weights (impossible), LoRA inserts tiny trainable matrices:

```
Standard training (IMPOSSIBLE on 6GB):
   Update W  →  W_new  (huge, 3.8B weights change)

LoRA training (POSSIBLE on 6GB):
   Freeze W  →  unchanged forever
   Train A × B  →  two tiny rank-16 matrices
   Output = W + (B × A)  →  combined at inference
```

**Only these 7 projection layers per block get LoRA adapters:**

```
q_proj  →  Query projection (attention)
k_proj  →  Key projection (attention)
v_proj  →  Value projection (attention)
o_proj  →  Output projection (attention)
gate_proj  →  FFN gate
up_proj    →  FFN up projection
down_proj  →  FFN down projection
```

**Result:** ~50M parameters trained out of 3.8B = **1.6% of the model**

---

## Slide 11 — Training Parameters

### LoRA Settings
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `r` (rank) | 16 | Size of trainable matrices A and B |
| `lora_alpha` | 16 | Scaling (alpha/r = 1.0 = full influence) |
| `lora_dropout` | 0 | No dropout — optimized for speed |
| `bias` | none | Skip bias terms to save memory |
| `gradient_checkpointing` | unsloth | Saves ~30% VRAM by recomputing activations |

### Training Hyperparameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `batch_size` | 2 | Small for 6GB VRAM |
| `gradient_accumulation` | 4 | Effective batch = 2×4 = 8 |
| `max_steps` | 60 | Short targeted training run |
| `learning_rate` | 2e-4 | Standard LoRA rate |
| `lr_scheduler` | linear | Smoothly decays LR to 0 |
| `optimizer` | adamw_8bit | 75% less memory than standard Adam |
| `weight_decay` | 0.01 | Prevents overfitting |
| `precision` | bf16 | Brain Float 16 on RTX 4050 |
| `seed` | 3407 | Reproducibility |

---

## Slide 12 — The Dataset

**Dataset:** `yahma/alpaca-cleaned` — 52,000 instruction-following examples

**Format used:**
```
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
Explain Newton's second law of motion.

### Response:
Newton's second law states that force equals mass
times acceleration, written as F = ma...<EOS>
```

The `<EOS>` (End of Sequence) token is critical — it teaches the model **when to stop generating**, preventing it from rambling endlessly.

---

## Slide 13 — Step-by-Step Training Process

```
STEP 1 — Download Base Model  (~5-10 min, one time)
   unsloth/Phi-3-mini-4k-instruct downloaded from HuggingFace
   Loaded in 4-bit NF4 quantization → fits in 6GB VRAM

STEP 2 — Inject LoRA Adapters  (<1 second)
   Tiny A × B matrices attached to 7 projections × 32 layers
   Original weights completely frozen

STEP 3 — Load & Format Dataset  (~2-3 min)
   52,000 Alpaca examples wrapped in instruction template
   EOS tokens appended to every example

STEP 4 — Training Loop  (~5-15 min, 60 steps)
   For each step:
   ① Pick batch of 2 examples
   ② Forward pass → predict next tokens
   ③ Compute cross-entropy loss (how wrong were predictions?)
   ④ Backward pass → gradients flow only through LoRA A & B
   ⑤ Accumulate 4 batches, then update weights
   ⑥ AdamW nudges A & B in direction of lower loss
   ⑦ Linear LR scheduler reduces learning rate gradually

STEP 5 — Merge & Export to GGUF  (~15-20 min)
   LoRA adapters merged: W_final = W_frozen + (B × A)
   Re-quantized to Q4_K_M format for LM Studio
   Final size: ~2.15 GB
```

---

## Slide 14 — Export & Quantization

After training, we export to **GGUF format** for LM Studio:

```python
model.save_pretrained_gguf(
    "my_custom_model",
    tokenizer,
    quantization_method = "q4_k_m"
)
```

**What is Q4_K_M?**

| Part | Meaning |
|------|---------|
| `Q4` | 4-bit quantization — each weight stored in 4 bits instead of 16 |
| `K` | K-Quants — mixed precision; important layers keep higher precision |
| `M` | Medium tier — best balance of file size vs output quality |

**Result:**

| Format | Size | Use |
|--------|------|-----|
| Safetensors (bfloat16) | ~7.6 GB | Training output, 2 shard files |
| GGUF Q4_K_M | **~2.15 GB** | Production — load into LM Studio |

72% size reduction with minimal quality loss.

---

## Slide 15 — How LM Studio Connects Everything

LM Studio loads the `.gguf` file and runs an **OpenAI-compatible local server** at `http://localhost:1234/v1`.

The app talks to it exactly like it would talk to ChatGPT:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

stream = client.chat.completions.create(
    model  = "local-model",       # LM Studio routes this to your .gguf
    messages = [
        {"role": "system", "content": "You are an expert educational assistant."},
        {"role": "user",   "content": f"Generate notes from:\n{transcript}"}
    ],
    temperature = 0.7,
    max_tokens  = 1024,
    stream      = True            # live token streaming → typing effect in UI
)
```

**Result:** Tokens stream back in real time, giving the user a live "AI is writing" experience — 100% offline.

---

## Slide 16 — Full Pipeline End-to-End

```
Student speaks or uploads lecture
           │
           ▼
┌──────────────────────┐
│  faster-whisper STT  │  244M params, Whisper Small, float16, GPU
│  beam_size = 5       │  beam search for accuracy
└──────────┬───────────┘
           │ "Today in physics we will cover..."
           ▼
┌──────────────────────┐
│  Prompt Engineering  │  Wraps transcript in structured notes prompt
│  (app.py)            │  Includes: Title, Key Points, Explanation,
└──────────┬───────────┘  Examples, Summary sections  
           │
           ▼
┌──────────────────────┐
│  Phi-3 Mini 3.8B     │  Q4_K_M GGUF in LM Studio
│  (custom fine-tuned) │  OpenAI-compatible API at localhost:1234
│  temperature = 0.7   │  Streams tokens live to UI
└──────────┬───────────┘
           │
           ▼
    Structured .md Notes displayed
    Download button (.md file)
    Chat interface for Q&A
```

---

## Slide 17 — Project File Structure

```
d:\mega_project\
│
├── app.py                    ← Main Streamlit web application
├── train_model.py            ← Fine-tuning script (QLoRA + Unsloth)
├── export_gguf.py            ← Convert trained weights → GGUF for LM Studio
├── requirements.txt          ← Python dependencies
│
├── my_custom_model/          ← Saved model after training (~7.6 GB)
│   ├── model-00001-of-00002.safetensors  (4.99 GB)
│   ├── model-00002-of-00002.safetensors  (2.65 GB)
│   ├── config.json           ← Architecture config
│   ├── tokenizer.json        ← Vocabulary (32,064 tokens)
│   └── tokenizer_config.json
│
├── my_custom_model_gguf/     ← LM Studio-ready model
│   └── MyBrand/
│       └── EducationalAssistant/
│           └── my_custom_model.Q4_K_M.gguf  (2.15 GB)
│
├── outputs/                  ← Training checkpoints
├── unsloth_compiled_cache/   ← Unsloth compiled cache
│
├── README.md                 ← Setup & usage guide
├── about-model.md            ← Model architecture documentation
├── model-training.md         ← Training process documentation
└── presentation.md           ← This file
```

---

## Slide 18 — Hardware & System Requirements

| Component | Minimum | Used in This Project |
|-----------|---------|---------------------|
| GPU | NVIDIA RTX 3060 (6GB) | **RTX 4050 (6GB VRAM)** |
| RAM | 16 GB | Recommended 16–32 GB |
| Disk Space | 20 GB free | ~25 GB used |
| OS | Windows 10/11 | Windows 11 |
| Python | 3.10 | Python 3.10 |
| CUDA | 12.1 | CUDA 12.1 |
| LM Studio | v0.3.x | Latest |

**VRAM Breakdown during inference:**

```
Whisper Small (float16):   ~500 MB
Phi-3 Q4_K_M (GGUF):      ~3.5 GB
OS + other:                ~1.5 GB
─────────────────────────────────
Total used:                ~5.5 GB  ✅  (fits in 6GB)
```

---

## Slide 19 — Technology Stack

| Layer | Technology | Version | Role |
|-------|-----------|---------|------|
| **UI** | Streamlit | 1.53.0 | Web interface |
| **STT** | faster-whisper | latest | Audio transcription |
| **LLM Server** | LM Studio | latest | Local model server |
| **LLM Client** | openai (Python) | latest | API calls to LM Studio |
| **Training** | Unsloth | 2026.4.4 | Fast QLoRA training |
| **Training** | TRL / SFTTrainer | latest | Supervised fine-tuning |
| **Training** | PEFT | latest | LoRA adapter management |
| **Deep Learning** | PyTorch | 2.9.1 | Core ML framework |
| **Models** | Transformers | 4.57.3 | Model loading & tokenization |
| **Acceleration** | accelerate | 1.13.0 | Mixed precision + GPU |
| **Audio** | pydub / soundfile | latest | Audio processing |

---

## Slide 20 — Training Timeline

```
Day 1 — Environment Setup
   ├── Install Python 3.10, CUDA 12.1
   ├── Install CMake, OpenSSL
   ├── Create virtual environment
   ├── Install PyTorch (CUDA build)
   └── Install Unsloth, TRL, PEFT

Day 1/2 — Model Training  (~30–40 mins total)
   ├── Download Phi-3-mini base          5–10 min
   ├── Inject LoRA adapters              <1 sec
   ├── Load Alpaca dataset               2–3 min
   ├── 60 training steps on RTX 4050     5–15 min
   ├── Merge LoRA into weights           2 min
   └── Export to Q4_K_M GGUF            15–20 min

Day 2 — App Development
   ├── Build Streamlit UI
   ├── Integrate faster-whisper STT
   ├── Connect to LM Studio API
   ├── Add note generation prompt
   ├── Add streaming chat interface
   └── Add YouTube URL support

Result: Fully working offline AI Educational Assistant ✅
```

---

## Slide 21 — Key Differentiators

| Feature | This Project | ChatGPT / Gemini |
|---------|-------------|-----------------|
| Works offline | ✅ Yes | ❌ No |
| Data privacy | ✅ Stays on your machine | ❌ Sent to cloud |
| Cost | ✅ Free forever | ❌ Subscription |
| Custom trained | ✅ Your own model | ❌ Generic |
| Audio transcription | ✅ Built-in (Whisper) | ❌ Extra plugin |
| YouTube support | ✅ Paste URL → Notes | ❌ No |
| Downloadable notes | ✅ .md file | ❌ Copy-paste only |
| Runs on laptop GPU | ✅ 6GB VRAM | ❌ Cloud GPU |

---

## Slide 22 — Results & Output Quality

The model generates notes in this structured format for every lecture:

```markdown
# Title
[AI-generated title based on lecture content]

## Key Points
- Bullet point 1
- Bullet point 2

## Explanation
Detailed paragraphs explaining the concepts...

## Examples
Real-world examples discussed in the lecture...

## Summary
A concise paragraph summarizing everything...
```

Notes are delivered **token-by-token in real time** (streaming), giving a live "AI writing" visual effect in the UI using `st.write_stream()`.

---

## Slide 23 — Future Improvements

1. **Larger base model** — Upgrade from Phi-3 Mini (3.8B) to Phi-3 Medium (14B) with more VRAM
2. **Custom educational dataset** — Fine-tune on textbooks, lecture notes, and exam papers instead of Alpaca
3. **PDF / slide support** — Extract text from lecture PDFs and PowerPoint files
4. **Multi-language** — Whisper already supports 99 languages; extend notes generation too
5. **Flashcard generation** — Auto-generate Anki-compatible flashcards from notes
6. **Speaker diarization** — Identify "Professor:" vs "Student:" in recorded lectures
7. **Longer context** — Upgrade to Phi-3 Medium 128K for full lecture transcripts without truncation
8. **Web deployment** — Package as a Docker container for university lab deployment

---

## Slide 24 — How to Run (Quick Start)

```powershell
# 1. Clone / navigate to project
cd D:\mega_project

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open LM Studio
#    → Load my_custom_model.Q4_K_M.gguf
#    → Local Server tab → Start Server

# 5. Run the app
streamlit run app.py

# 6. Open browser at:
#    http://localhost:8501
```

Download pre-trained model files from Google Drive:
`https://drive.google.com/drive/folders/1zm-9FvhZLmkV5y2qgzqNP-RqCLe1h8kZ`

---

## Slide 25 — Summary

```
Problem     Students waste hours writing notes from lecture recordings

Solution    Two-model local AI pipeline: Whisper → Phi-3 Mini

Innovation  Fine-tuned 3.8B model on a 6GB consumer GPU using QLoRA

Pipeline    Audio → Whisper STT → Phi-3 LLM → Structured Markdown Notes

Privacy     100% offline. Zero cloud. Zero subscriptions. Your data stays yours.

Impact      1-hour lecture → structured notes in under 2 minutes
```

---

## References

- [Phi-3 Technical Report — Microsoft Research](https://arxiv.org/abs/2404.14219)
- [LoRA: Low-Rank Adaptation of Large Language Models — Hu et al. 2021](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs — Dettmers et al. 2023](https://arxiv.org/abs/2305.14314)
- [Robust Speech Recognition via LWS — OpenAI Whisper 2022](https://arxiv.org/abs/2212.04356)
- [Unsloth — Fast QLoRA Fine-tuning](https://github.com/unslothai/unsloth)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [LM Studio — Local LLM Server](https://lmstudio.ai)
- [Alpaca Dataset — yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)
- [Streamlit Documentation](https://docs.streamlit.io)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
