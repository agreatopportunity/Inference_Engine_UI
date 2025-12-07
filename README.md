```markdown
# 🌌 Universal AI Interface

**The "Holo-Glass" Cognitive Interface for your custom LLaMA-3 models.**
This interface is **Hardware Agnostic**, meaning it auto-detects your GPU (Titan V vs. RTX 4090) and adjusts precision (`float16` vs `bfloat16`) and attention mechanisms automatically to prevent crashes and maximize speed.

---

## 📦 Step 1: Installation & Setup

### 1\. The Core Essentials (Run this first)

This covers your custom LLaMA-3 model, the training scripts, and the new Holo-Glass UI.

```bash
# Core AI & Math libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy

# Data processing & Tokenization
pip install tiktoken transformers datasets sentencepiece

# The User Interface (Holo-Glass)
pip install gradio

# Training Monitoring (Optional but recommended)
pip install wandb
```

### 2\. Speed Optimization (Highly Recommended)

This makes your 5090/4090/3090 / Titan V run about 20% faster.

```bash
pip install flash-attn --no-build-isolation
```

### 3\. GGUF Support (Optional)

If you want to load downloaded models (like `Llama-3-8B.gguf`) alongside your custom one, you need this.

**Important:** You must install this with CUDA support enabled, or it will be slow (CPU only).

```bash
# Force it to compile with CUDA support
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

-----

### 📦 The "One-Liner" Installer

You can copy and paste this entire block to set up a fresh machine instantly:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install tiktoken transformers datasets sentencepiece numpy gradio wandb
pip install flash-attn --no-build-isolation
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

### 1. Activate your Environment
```bash
conda activate myLLM
```

## 📂 Step 2: Verify File Structure

Ensure your project folder contains these files:

  * **`app.py`**: The main UI script ( `app.py`).
  * **`model_llama3.py`**: The architecture file (Required so the script knows how to build the model).
  * **`checkpoints/latest.pt`**: Your trained model file.

-----

## 🚀 Step 3: Launching the Interface

You can run this script on **ANY** machine. It will automatically diagnose your hardware.

### Scenario A: The Modern Rig (RTX 3090 / 4090 / 5090)

*Use this if you are running on your main machine's primary GPU.*

```bash
python3 app.py
```

**Result:**

  * **Precision:** Auto-selects `bfloat16` (Max Speed/Stability).
  * **Attention:** Enables Flash Attention 2.
  * **Hardware:** Defaults to GPU 0.

### Scenario B: The "Titan V" Setup (Multi-GPU)

*Use this if you are training on GPU 0 and want to chat on GPU 1 (Titan V).*

```bash
python3 app.py --device_id 1
```

**Result:**

  * **Precision:** Auto-switches to `float16` (Prevents Titan V crash).
  * **Attention:** Disables Flash Attention (Prevents Titan V crash).
  * **Hardware:** Forces execution on GPU 1.

### Scenario C: Single Legacy GPU

*Use this if you move the Titan V to its own separate machine.*

```bash
python3 app.py
```

**Result:**

  * Detects "Volta" architecture on GPU 0.
  * Auto-downgrades settings to "Safe Mode" (`float16` / Efficient Attention).

-----

## 🧠 Interface Features

Once the UI launches (at `http://0.0.0.0:7860`), you will see:

1.  **Hardware Link:** Displays which GPU is active and what precision mode is running.
2.  **Neural Sync:** A visual representation of the AI's "Awareness" level.
3.  **Live Metrics:** Real-time bars showing the AI's **Focus** and **Energy** levels (Emotional Decay system).
4.  **Data Ingest:** Drag-and-drop zone to upload text/images (Future proofing).

-----

## ⚠️ Troubleshooting

**Error: `ModuleNotFoundError: No module named 'model_llama3'`**

  * **Fix:** You are missing `model_llama3.py` in the same folder as the script. Copy it from your training folder.

**Error: `CUDA Out of Memory`**

  * **Fix:** You might be trying to load the model on a GPU that is already training. Use `--device_id 1` to switch to the free GPU.

**Error: `FlashAttention only supports Ampere GPUs`**

  * **Fix:** The auto-detection failed. Manually force the Titan V mode by editing the script or ensuring you are using the latest `app.py`.

<!-- end list -->

