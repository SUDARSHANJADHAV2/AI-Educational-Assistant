# 🎙️ AI Educational Assistant

A AI assistant that converts audio lectures into structured, exam-ready study notes and lets you chat with your own custom-trained AI model — all running locally on your machine without any external API calls.

---

## Download Pre-trained Model Files (Google Drive)

> **These large files are NOT included in this GitHub repository** due to GitHub's 100MB file size limit.
> You must download them separately and place them in the correct folders before running the app.

**[Click here to download all model files from Google Drive](https://drive.google.com/drive/folders/1zm-9FvhZLmkV5y2qgzqNP-RqCLe1h8kZ?usp=sharing)**

The Google Drive folder contains:

| Folder / File | Size | Purpose |
|---|---|---|
| `my_custom_model/` | ~7.6 GB | Fine-tuned model weights (safetensors) — used by `export_gguf.py` |
| `my_custom_model_gguf/` | ~2.3 GB | Ready-to-use `.gguf` file — load directly into LM Studio |
| `outputs/` | ~500 MB | Training checkpoints — only needed if you want to resume training |
| `unsloth_compiled_cache/` | ~50 MB | Compiled Unsloth trainer cache — speeds up re-training |

### How to use the downloaded files:

1. Download the folders from the Google Drive link above.
2. Place each folder directly inside your project directory:
   ```
   D:\mega_project\
   ├── my_custom_model\          ← paste here
   ├── my_custom_model_gguf\     ← paste here
   ├── outputs\                  ← paste here (optional)
   └── unsloth_compiled_cache\   ← paste here (optional)
   ```
3. **If you only want to run the app**, you just need `my_custom_model_gguf/`. Skip the rest.
4. Follow **[Part 5](#-part-5--load-model-in-lm-studio)** to load the `.gguf` into LM Studio.

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [System Requirements](#-system-requirements)
3. [Project Structure](#-project-structure)
4. [Architecture](#-architecture)
5. [Part 1 — Environment Setup](#-part-1--environment-setup)
6. [Part 2 — Install Dependencies](#-part-2--install-dependencies)
7. [Part 3 — Train Your Own Custom Model](#-part-3--train-your-own-custom-model)
8. [Part 4 — Export Model to GGUF](#-part-4--export-model-to-gguf-for-lm-studio)
9. [Part 5 — Load Model in LM Studio](#-part-5--load-model-in-lm-studio)
10. [Part 6 — Run the Streamlit App](#-part-6--run-the-streamlit-app)
11. [How to Use the App](#-how-to-use-the-app)
12. [Troubleshooting](#-troubleshooting)
13. [Known Issues & Fixes](#-known-issues--fixes)

---

## Project Overview

This application is a **two-component system**:

| Component | Technology | Role |
|-----------|-----------|------|
| **Speech-to-Text (STT)** | `faster-whisper` (small, float16) | Converts audio into raw transcript on your GPU |
| **Language Model (LLM)** | Your custom fine-tuned model via LM Studio | Generates structured notes & answers chat questions |

Both components run **100% locally on your NVIDIA RTX 4050 (6GB VRAM)** using smart quantization (4-bit / float16) to fit within memory constraints.

---

## System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| **GPU** | NVIDIA RTX 3060 (6GB VRAM) | NVIDIA RTX 4050+ (6GB VRAM) |
| **RAM** | 16 GB | 32 GB |
| **Disk Space** | 20 GB free | 40 GB free |
| **OS** | Windows 10/11 | Windows 11 |
| **Python** | 3.10 | 3.10 |
| **CUDA Toolkit** | 12.1 | 12.1 |
| **CMake** | 3.x | Latest |
| **OpenSSL** | Any | Latest |
| **LM Studio** | v0.3.x | Latest |

---

## Project Structure

```
d:\mega_project\
│
├── app.py                          ← Main Streamlit application
├── train_model.py                  ← Script to fine-tune your custom model
├── export_gguf.py                  ← Script to convert trained model to .gguf
├── requirements.txt                ← Python dependencies
├── README.md                       ← This file
│
├── my_custom_model/                ← Saved model weights after training (safetensors)
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   ├── tokenizer.json
│   └── config.json
│
├── my_custom_model_gguf/           ← GGUF export directory
│   └── my_custom_model.Q4_K_M.gguf  ← ← ← This file goes into LM Studio
│
├── outputs/                        ← Training checkpoints (can be deleted after export)
└── unsloth_compiled_cache/         ← Unsloth cache (auto-generated)
```

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│               Streamlit Web UI (app.py)           │
│  ┌────────────────┐    ┌────────────────────────┐ │
│  │  Audio Upload  │    │   Audio Recorder       │ │
│  │  (mp3/wav/m4a) │    │   (microphone)         │ │
│  └───────┬────────┘    └──────────┬─────────────┘ │
│          └──────────┬─────────────┘               │
│                     ▼                             │
│  ┌──────────────────────────────────────────────┐ │
│  │  faster-whisper STT Model (GPU, float16)     │ │
│  │  → Transcribes audio to raw text             │ │
│  └─────────────────────┬────────────────────────┘ │
│                        ▼                          │
│  ┌──────────────────────────────────────────────┐ │
│  │  LM Studio Local Server (localhost:1234)     │ │
│  │  → Runs your custom .gguf model              │ │
│  │  → Generates structured notes & chat replies │ │
│  └──────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

---

## Part 1 — Environment Setup

### 1.1 — Install Python 3.10

Download Python 3.10 from [python.org](https://www.python.org/downloads/release/python-3100/) and install it.
Make sure to check **"Add Python to PATH"** during installation.

### 1.2 — Install CUDA Toolkit 12.1

Download it from [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-12-1-0-download-archive) and install it.

### 1.3 — Install CMake (Required for GGUF export)

```powershell
winget install -e --id Kitware.CMake
```

Verify it works:
```powershell
cmake --version
```

If this says `INFO: Could not find files`, add it to PATH manually:
- Open **Environment Variables** → **System Variables** → **Path** → **New**
- Add: `C:\Program Files\CMake\bin`
- **Restart your terminal.**

### 1.4 — Install OpenSSL (Required for GGUF export)

```powershell
winget install -e --id ShiningLight.OpenSSL.Dev
```

Add to PATH if needed:
- Add: `C:\Program Files\OpenSSL-Win64\bin`

### 1.5 — Install LM Studio

Download and install from [lmstudio.ai](https://lmstudio.ai/).

---

## Part 2 — Install Dependencies

### 2.1 — Create and activate a Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2.2 — Install GPU version of PyTorch (CUDA 12.1)

> Do NOT use `pip install torch` alone — that installs the CPU-only version.

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2.3 — Install Unsloth and Training Libraries

```powershell
pip install unsloth
pip install trl peft bitsandbytes accelerate
```

Fix a known `torchao` version conflict:
```powershell
pip install "torchao<=0.7.0"
```

### 2.4 — Install App Dependencies

```powershell
pip install -r requirements.txt
```

`requirements.txt` contains:
```
streamlit
torch
torchaudio
faster-whisper
openai
streamlit-audiorecorder
pydub
```

---

## Part 3 — Train Your Own Custom Model

> Training takes approximately **50–60 minutes** on an RTX 4050.

### 3.1 — (Optional) Prepare your own dataset

Create a file called `my_dataset.json`:
```json
[
  {"instruction": "What is photosynthesis?", "output": "Photosynthesis is the process by which plants use sunlight to produce food."},
  {"instruction": "Explain Newton's Second Law.", "output": "F = ma. Force equals mass times acceleration."}
]
```

Then, update line 56 in `train_model.py`:
```python
# Replace this line:
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")

# With this line to use your own file:
dataset = load_dataset("json", data_files="my_dataset.json", split="train")
```

### 3.2 — Run Training

```powershell
python train_model.py
```

When complete, your trained brain is saved in the `my_custom_model/` folder as `.safetensors` files.

### 3.3 — What `train_model.py` does internally

| Step | What happens |
|------|-------------|
| Load base model | Downloads `Phi-3-mini-4k-instruct` in 4-bit quantization (fits in 6GB VRAM) |
| Apply LoRA adapters | Attaches trainable "adapters" (~1–5% of model weights) |
| Train | Feeds your dataset through the GPU for learning |
| Save weights | Saves the merged model to `my_custom_model/` |

---

## Part 4 — Export Model to GGUF (for LM Studio)

LM Studio requires models in the `.gguf` format. This script converts your trained weights.

### 4.1 — Run the export script

CMake and OpenSSL must be in your PATH. If you just installed them, first run this in your terminal to apply them to the current session:

```powershell
$env:PATH += ";C:\Program Files\CMake\bin;C:\Program Files\OpenSSL-Win64\bin"
python export_gguf.py
```

> This takes approximately **15–20 minutes**.

### 4.2 — Expected output

When complete, you will see:
```
DONE! Look for 'lm_studio_model.Q4_K_M.gguf' in your folder.
```

The file will be at:
```
D:\mega_project\my_custom_model_gguf\my_custom_model.Q4_K_M.gguf
```

> **Disk Space Warning**: You need at least **15 GB free** on the drive where your project lives. The export creates a ~7.6 GB temporary file that shrinks to ~2.5 GB after quantization.

---

## Part 5 — Load Model in LM Studio

### 5.1 — Add your GGUF to LM Studio

LM Studio requires a two-level folder structure to detect models:

1. Open File Explorer and go to `D:\mega_project\my_custom_model_gguf\`
2. Create a folder inside called `MyBrand`
3. Inside `MyBrand`, create another folder called `EducationalAssistant`
4. Move `my_custom_model.Q4_K_M.gguf` into `EducationalAssistant`

Final path:
```
my_custom_model_gguf/MyBrand/EducationalAssistant/my_custom_model.Q4_K_M.gguf
```

5. In LM Studio, click the **"My Models"** tab, then **"Check for new models"**.
6. Your model will now appear in the list!

### 5.2 — Start the Local Server

1. Click the **↔ Local Server** tab on the left sidebar.
2. Select your model from the dropdown at the top.
3. Set **GPU Offload** to **Max** (to use your RTX 4050 fully).
4. Click **"Start Server"**.

The server will start at: **`http://localhost:1234`**

> Keep LM Studio running in the background whenever you use the app.

---

## Part 6 — Run the Streamlit App

Make sure:
- [x] LM Studio server is started (Step 5.2)
- [x] You are in the project directory
- [x] Your virtual environment is activated

Then run:
```powershell
streamlit run app.py
```

Open your browser at: **`http://localhost:8501`**

---

## How to Use the App

### Upload Audio
1. Click **"Upload Audio"** and select an MP3, WAV, or M4A file.
2. Click **"Transcribe & Generate Notes"**.
3. The app will:
   - Transcribe the audio using Whisper (on GPU).
   - Send the transcript to your custom model in LM Studio.
   - Display structured notes with Title, Key Points, Explanation, Examples, and Summary.
4. Click **"Download Notes (.md)"** to save your notes.

### Record Audio
1. Click **"Record Audio"** → Click to Record → speak → Click to Stop.
2. Click **"Transcribe & Generate Notes"**.

### Chat with the AI
- Type any question in the chat box at the bottom.
- Your model will respond using the context of previous messages (last 10 messages remembered).

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `No module named 'unsloth'` | Unsloth not installed | `pip install unsloth` |
| `No module named 'torch'` | PyTorch not installed | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `torch has no attribute 'int1'` | `torchao` version mismatch | `pip install "torchao<=0.7.0"` |
| `Cannot find any torch accelerator` | CPU-only PyTorch installed | Reinstall with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `Failed to install Kitware.CMake` | CMake not in PATH | Add `C:\Program Files\CMake\bin` to system PATH |
| `OSError: X requested and 0 written` | Disk full | Free up at least 15 GB of space |
| `LM Studio server offline` | LM Studio not running | Open LM Studio → Local Server → Start Server |
| `No model appeared in LM Studio` | Wrong folder structure | Use two-level folder: `Brand/ModelName/file.gguf` |
| `Connection refused localhost:1234` | Server not started | Click "Start Server" in LM Studio |

---

## Known Issues & Fixes

### Issue: `winget install Kitware.CMake` fails during GGUF export
CMake IS installed on your system but not registered in the `PATH` that Python processes see.

**Fix**: Run this before running `export_gguf.py`:
```powershell
$env:PATH += ";C:\Program Files\CMake\bin;C:\Program Files\OpenSSL-Win64\bin"
python export_gguf.py
```

### Issue: Flash Attention 2 warning
```
Unsloth: Your Flash Attention 2 installation seems to be broken. Using Xformers instead.
```
This is a harmless warning on Windows. Performance is not impacted.

### Issue: Out of VRAM during generation
Reduce context length. In `app.py`, lower the value `messages[-10:]` to `messages[-4:]` to reduce the conversation window sent to the model.

---

<!-- ## License

This project is for personal, educational. Model weights derived from Phi-3 are subject to [Microsoft's Phi-3 license](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct). -->
