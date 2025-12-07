# 🌌 Universal AI Interface

**The "Holo-Glass" Cognitive Interface for your custom LLaMA-3 models.**
This interface is **Hardware Agnostic**, meaning it auto-detects your GPU (OLDER GPU vs. NEWER GPU) and adjusts precision (`float16` vs `bfloat16`) and attention mechanisms automatically to prevent crashes and maximize speed.

---


## Step 1: Installation & Setup

To use your machine properly, you should install **Miniconda**. It is the industry standard for deep learning on Linux because it is lightweight and avoids the bloat of the full Anaconda distribution.

### Install Miniconda

This block downloads the installer, runs it silently (no need to press "Enter" 50 times), and initializes it for your shell.

```bash
# 1. Create a directory for the installer
mkdir -p ~/miniconda3

# 2. Download the latest Linux x86_64 installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

# 3. Run the installer (Silent Mode)
# -b = Batch mode (no questions)
# -u = Update existing installation if found
# -p = Install path
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

# 4. Cleanup the installer file
rm -rf ~/miniconda3/miniconda.sh

# 5. Initialize Conda for your shell (Bash)
~/miniconda3/bin/conda init bash

# 6. Reload your shell to make the 'conda' command visible
source ~/.bashrc
```

### Verify Installation

Run this to make sure Conda is working. It should return a version number (e.g., `conda 24.x.x`).

```bash
conda --version
```

### Run Your Planned Commands

```bash
You must manually accept the TOS for the main channels one by one using these exact commands. Run them in your terminal. Once you have run those two commands successfully, try creating your environment again.

# Accept TOS for the main channel
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main

# Accept TOS for the R channel (also required by default)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create your environment
conda create -n myLLM python=3.11 -y

# Activate it
conda activate myLLM
```

Video Explanation: [Miniconda Installation](https://www.google.com/search?q=https://www.youtube.com/watch%3Fv%3D2K4Vaf9aN80)
*This video visually demonstrates the Miniconda installation process on Linux if you prefer to watch the steps.*


### 1\. The Core Essentials (Run this first)
```markdown
This covers your custom LLaMA-3 model, the training scripts, and the new Holo-Glass UI.

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

This makes your 5090/4090/3090 / Titan V or OLDER GPU run about 20% faster.

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

### Scenario B: The "Titan V/OLDER GPU" Setup (Multi-GPU)

*Use this if you are training on GPU 0 and want to chat on GPU 1 (Titan V).*

```bash
python3 app.py --device_id 1
```

**Result:**

  * **Precision:** Auto-switches to `float16` (Prevents OLDER GPU crash).
  * **Attention:** Disables Flash Attention (Prevents OLDER GPU crash).
  * **Hardware:** Forces execution on GPU 1.

### Scenario C: Single Legacy GPU

*Use this if you move the Titan V or OLDER GPU  to its own separate machine.*

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

