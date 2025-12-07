"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  SOVRA OMNI v2.0 - Neural Interface for LLM Inference                         ║
║  ─────────────────────────────────────────────────────────────────────────────║
║  Supports:                                                                    ║
║    • Native Models (.pt) - Custom LLaMA-3 architecture                        ║
║    • GGUF Models (.gguf) - DeepSeek, Llama, Mistral, etc.                     ║
║                                                                               ║
║  Features:                                                                    ║
║    • Real-time streaming generation                                           ║
║    • GPU memory monitoring                                                    ║
║    • Multi-GPU support                                                        ║
║    • Automatic precision selection (FP16/BF16)                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
import gradio as gr
import torch
import torch.nn.functional as F  # FIX: This was missing!
import tiktoken
import contextlib
import argparse
import os
import sys
import time
import threading
import subprocess
import urllib.request
from pathlib import Path
from datetime import datetime

# Optional: huggingface_hub for better downloads
try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("⚠️  huggingface_hub not installed. Using direct downloads.")

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND LINE ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  SOVRA OMNI v2.0 - Command Line Arguments                                     ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  --device_id  : GPU Index to use (Default: 0)                                 ║
║                 Use 'nvidia-smi' to check your GPU indices                    ║
║                                                                               ║
║  --port       : Port to run the UI on (Default: 7860)                         ║
║                                                                               ║
║  --share      : Create a public Gradio share link                             ║
║                                                                               ║
║  Example: python3 app.py --device_id 0 --port 7860 --share                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
        """)
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    ARGS_DEVICE = args.device_id
else:
    ARGS_DEVICE = 0
    args = type('obj', (object,), {'port': 7860, 'share': False})()

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from model_llama3 import GPT, GPTConfig
    NATIVE_AVAILABLE = True
except ImportError:
    print("⚠️  model_llama3.py not found. Native models disabled.")
    NATIVE_AVAILABLE = False

try:
    from llama_cpp import Llama
    GGUF_AVAILABLE = True
except ImportError:
    print("⚠️  llama-cpp-python not installed. GGUF models disabled.")
    GGUF_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════════
CURRENT_MODEL = None
CURRENT_ENGINE = None  # "native" or "gguf"
MODEL_INFO = {}
ENC = None
STOP_GENERATION = False
CURRENT_TEMPLATE = "None (Raw)"  # Track current chat template
DOWNLOAD_PROGRESS = {"status": "", "progress": 0}  # Track download progress

# Models directory (same folder as app.py)
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CATALOG - Popular GGUF models for download
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_CATALOG = {
    "── General Chat Models ──": None,  # Section header
    "Llama-3.2-3B-Instruct (Q5_K_M, 2.3GB)": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q5_K_M.gguf",
        "size": "2.3 GB",
        "type": "chat"
    },
    "Llama-3.2-1B-Instruct (Q8_0, 1.3GB)": {
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q8_0.gguf",
        "size": "1.3 GB",
        "type": "chat"
    },
    "Mistral-7B-Instruct-v0.3 (Q5_K_M, 5.1GB)": {
        "repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "file": "Mistral-7B-Instruct-v0.3-Q5_K_M.gguf",
        "size": "5.1 GB",
        "type": "chat"
    },
    "Qwen2.5-7B-Instruct (Q5_K_M, 5.4GB)": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "file": "qwen2.5-7b-instruct-q5_k_m.gguf",
        "size": "5.4 GB",
        "type": "chat"
    },
    "Qwen2.5-3B-Instruct (Q5_K_M, 2.4GB)": {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "file": "qwen2.5-3b-instruct-q5_k_m.gguf",
        "size": "2.4 GB",
        "type": "chat"
    },
    "Phi-3.5-mini-instruct (Q5_K_M, 2.8GB)": {
        "repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "file": "Phi-3.5-mini-instruct-Q5_K_M.gguf",
        "size": "2.8 GB",
        "type": "chat"
    },
    "Gemma-2-9B-Instruct (Q5_K_M, 6.6GB)": {
        "repo": "bartowski/gemma-2-9b-it-GGUF",
        "file": "gemma-2-9b-it-Q5_K_M.gguf",
        "size": "6.6 GB",
        "type": "chat"
    },
    "── Code Models ──": None,  # Section header
    "DeepSeek-Coder-V2-Lite (Q5_K_M, 11GB)": {
        "repo": "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        "file": "DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf",
        "size": "11 GB",
        "type": "code"
    },
    "Qwen2.5-Coder-7B-Instruct (Q5_K_M, 5.4GB)": {
        "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "file": "qwen2.5-coder-7b-instruct-q5_k_m.gguf",
        "size": "5.4 GB",
        "type": "code"
    },
    "Qwen2.5-Coder-3B-Instruct (Q8_0, 3.6GB)": {
        "repo": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        "file": "qwen2.5-coder-3b-instruct-q8_0.gguf",
        "size": "3.6 GB",
        "type": "code"
    },
    "── Small/Fast Models ──": None,  # Section header
    "SmolLM2-1.7B-Instruct (Q8_0, 1.8GB)": {
        "repo": "bartowski/SmolLM2-1.7B-Instruct-GGUF",
        "file": "SmolLM2-1.7B-Instruct-Q8_0.gguf",
        "size": "1.8 GB",
        "type": "chat"
    },
    "TinyLlama-1.1B-Chat (Q8_0, 1.2GB)": {
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        "size": "1.2 GB",
        "type": "chat"
    },
}

def get_model_list():
    """Get list of model names for dropdown"""
    return list(MODEL_CATALOG.keys())

def download_model_hf(repo_id: str, filename: str, progress_callback=None):
    """Download model using huggingface_hub"""
    global DOWNLOAD_PROGRESS
    
    if not HF_HUB_AVAILABLE:
        return None, "huggingface_hub not installed. Run: pip install huggingface_hub"
    
    try:
        DOWNLOAD_PROGRESS["status"] = f"Downloading {filename}..."
        DOWNLOAD_PROGRESS["progress"] = 10
        
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False
        )
        
        DOWNLOAD_PROGRESS["status"] = "Complete!"
        DOWNLOAD_PROGRESS["progress"] = 100
        
        return local_path, None
    except Exception as e:
        DOWNLOAD_PROGRESS["status"] = f"Error: {str(e)}"
        return None, str(e)

def download_model_direct(url: str, filename: str, progress_callback=None):
    """Download model using urllib with progress"""
    global DOWNLOAD_PROGRESS
    
    output_path = MODELS_DIR / filename
    
    try:
        DOWNLOAD_PROGRESS["status"] = f"Connecting to {url[:50]}..."
        DOWNLOAD_PROGRESS["progress"] = 5
        
        # Get file size
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress
        downloaded = 0
        block_size = 1024 * 1024  # 1MB chunks
        
        with urllib.request.urlopen(url, timeout=60) as response:
            with open(output_path, 'wb') as out_file:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    
                    if total_size > 0:
                        progress = int((downloaded / total_size) * 100)
                        size_mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        DOWNLOAD_PROGRESS["status"] = f"Downloading: {size_mb:.0f}/{total_mb:.0f} MB"
                        DOWNLOAD_PROGRESS["progress"] = progress
        
        DOWNLOAD_PROGRESS["status"] = "Complete!"
        DOWNLOAD_PROGRESS["progress"] = 100
        return str(output_path), None
        
    except Exception as e:
        DOWNLOAD_PROGRESS["status"] = f"Error: {str(e)}"
        if output_path.exists():
            output_path.unlink()  # Clean up partial download
        return None, str(e)

def ui_download_model(model_selection, custom_repo, custom_file):
    """Handle model download from UI"""
    global DOWNLOAD_PROGRESS
    DOWNLOAD_PROGRESS = {"status": "Starting...", "progress": 0}
    
    # Check if using custom or catalog
    if custom_repo.strip() and custom_file.strip():
        repo_id = custom_repo.strip()
        filename = custom_file.strip()
        model_name = filename
    elif model_selection and model_selection in MODEL_CATALOG:
        model_info = MODEL_CATALOG[model_selection]
        if model_info is None:  # Section header
            return "❌ Please select a model, not a section header", ""
        repo_id = model_info["repo"]
        filename = model_info["file"]
        model_name = model_selection
    else:
        return "❌ Please select a model or enter custom repo/file", ""
    
    # Check if already exists
    local_path = MODELS_DIR / filename
    if local_path.exists():
        return f"✅ Model already exists!\n\n📁 Path: {local_path}", str(local_path)
    
    # Download
    yield f"⏳ Downloading: {model_name}\n📦 Repo: {repo_id}\n📄 File: {filename}\n\nThis may take several minutes...", ""
    
    if HF_HUB_AVAILABLE:
        path, error = download_model_hf(repo_id, filename)
    else:
        # Construct direct URL
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        path, error = download_model_direct(url, filename)
    
    if error:
        yield f"❌ Download failed: {error}", ""
    else:
        final_path = MODELS_DIR / filename
        yield f"✅ Download complete!\n\n📁 Path: {final_path}\n💡 Click 'Use This Model' or copy path to Model Path field", str(final_path)

def ui_use_downloaded(model_path):
    """Set the downloaded model path in the model_path field"""
    if model_path:
        return model_path
    return ""

def ui_list_local_models():
    """List models in the models directory"""
    if not MODELS_DIR.exists():
        return "No models directory found"
    
    models = list(MODELS_DIR.glob("*.gguf"))
    if not models:
        return "No GGUF models found in models/ directory"
    
    result = "📁 Local Models:\n" + "━" * 40 + "\n"
    for m in sorted(models):
        size_gb = m.stat().st_size / (1024**3)
        result += f"• {m.name} ({size_gb:.1f} GB)\n"
    
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# CHAT TEMPLATES - Auto-wrap prompts for different model types
# ═══════════════════════════════════════════════════════════════════════════════
CHAT_TEMPLATES = {
    "None (Raw)": {
        "format": "{prompt}",
        "description": "No template - raw prompt",
        "detect": []  # No auto-detection patterns
    },
    "Llama-2/Mistral": {
        "format": "[INST] {prompt} [/INST]",
        "description": "Llama-2-Chat, Mistral-Instruct",
        "detect": ["llama-2", "mistral", "mixtral", "inst"]
    },
    "Llama-3": {
        "format": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "description": "Llama-3-Instruct models",
        "detect": ["llama-3", "llama3"]
    },
    "ChatML (Qwen/Yi)": {
        "format": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "description": "Qwen, Yi, OpenHermes",
        "detect": ["qwen", "yi-", "chatml", "hermes"]
    },
    "DeepSeek": {
        "format": "### Instruction:\n{prompt}\n\n### Response:\n",
        "description": "DeepSeek, DeepSeek-Coder",
        "detect": ["deepseek"]
    },
    "DeepSeek-V2/V3": {
        "format": "<|begin▁of▁sentence|><|User|>{prompt}<|Assistant|>",
        "description": "DeepSeek-V2, V3 models",
        "detect": ["deepseek-v2", "deepseek-v3", "deepseek-r1"]
    },
    "Alpaca": {
        "format": "### Instruction:\n{prompt}\n\n### Response:\n",
        "description": "Alpaca-style models",
        "detect": ["alpaca"]
    },
    "Vicuna": {
        "format": "USER: {prompt}\nASSISTANT:",
        "description": "Vicuna models",
        "detect": ["vicuna"]
    },
    "Phi-3": {
        "format": "<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
        "description": "Microsoft Phi-3 models",
        "detect": ["phi-3", "phi3"]
    },
    "Gemma": {
        "format": "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
        "description": "Google Gemma models",
        "detect": ["gemma"]
    },
    "Command-R": {
        "format": "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
        "description": "Cohere Command-R models",
        "detect": ["command-r"]
    },
    "Zephyr": {
        "format": "<|user|>\n{prompt}</s>\n<|assistant|>\n",
        "description": "Zephyr models",
        "detect": ["zephyr"]
    },
}

def detect_template(model_path: str) -> str:
    """Auto-detect chat template from model filename"""
    filename = os.path.basename(model_path).lower()
    
    # Priority order: Check specific patterns first, generic patterns last
    # This prevents "inst" from matching before "deepseek"
    priority_order = [
        "DeepSeek-V2/V3",    # Check deepseek-v2/v3/r1 first
        "DeepSeek",          # Then regular deepseek
        "Llama-3",           # Check llama-3 before generic llama
        "Phi-3",             # Check phi-3 before generic patterns
        "ChatML (Qwen/Yi)",  # Qwen, Yi
        "Gemma",
        "Command-R",
        "Zephyr",
        "Vicuna",
        "Alpaca",
        "Llama-2/Mistral",   # Generic inst/mistral patterns LAST
    ]
    
    for template_name in priority_order:
        if template_name not in CHAT_TEMPLATES:
            continue
        for pattern in CHAT_TEMPLATES[template_name]["detect"]:
            if pattern in filename:
                return template_name
    
    return "None (Raw)"

def apply_template(prompt: str, template_name: str) -> str:
    """Apply chat template to prompt"""
    if template_name not in CHAT_TEMPLATES:
        return prompt
    template = CHAT_TEMPLATES[template_name]["format"]
    return template.format(prompt=prompt)

# ═══════════════════════════════════════════════════════════════════════════════
# HARDWARE DETECTION & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
def get_gpu_info():
    """Get detailed GPU information"""
    if not torch.cuda.is_available():
        return [{"id": -1, "name": "CPU Only", "memory": 0, "compute": "N/A"}]
    
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            "id": i,
            "name": props.name,
            "memory": props.total_memory / 1e9,
            "compute": f"{props.major}.{props.minor}",
            "is_modern": props.major >= 8
        })
    return gpus

def get_gpu_stats(device_id=0):
    """Get real-time GPU statistics"""
    if not torch.cuda.is_available():
        return "CPU Mode - No GPU Stats"
    
    try:
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(device_id) / 1e9
        reserved = torch.cuda.memory_reserved(device_id) / 1e9
        total = torch.cuda.get_device_properties(device_id).total_memory / 1e9
        
        usage_pct = (allocated / total) * 100
        bar_len = 20
        filled = int(bar_len * usage_pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        return f"VRAM: [{bar}] {allocated:.1f}/{total:.1f} GB ({usage_pct:.0f}%)"
    except Exception as e:
        return f"Stats Error: {e}"

def get_device_config(device_id):
    """Configure device with optimal settings"""
    if not torch.cuda.is_available():
        return "cpu", "float32", contextlib.nullcontext()
    
    device = f"cuda:{device_id}"
    try:
        props = torch.cuda.get_device_properties(device)
        
        # Auto-select precision based on GPU architecture
        if props.major < 8:  # Pre-Ampere (Titan V, 1080, 2080, etc.)
            dtype = "float16"
            # Use newer API if available (PyTorch 2.2+), fallback to deprecated
            try:
                from torch.nn.attention import sdpa_kernel, SDPBackend
                attn = sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION])
            except ImportError:
                # Fallback for older PyTorch versions
                import warnings
                warnings.filterwarnings("ignore", message=".*sdp_kernel.*deprecated.*")
                attn = torch.backends.cuda.sdp_kernel(
                    enable_flash=False, 
                    enable_math=True, 
                    enable_mem_efficient=True
                )
        else:  # Ampere+ (3090, 4060 Ti, 4090, etc.)
            dtype = "bfloat16"
            attn = contextlib.nullcontext()
            
        return device, dtype, attn
    except Exception as e:
        print(f"❌ GPU {device_id} Error: {e}")
        return "cpu", "float32", contextlib.nullcontext()

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ═══════════════════════════════════════════════════════════════════════════════
def load_native(path, device_idx):
    """Load native PyTorch model (.pt)"""
    global CURRENT_MODEL, ENC, CURRENT_ENGINE, MODEL_INFO
    
    if not NATIVE_AVAILABLE:
        return "❌ Native engine unavailable (model_llama3.py not found)"
    
    device, dtype, _ = get_device_config(device_idx)
    
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        
        # Extract config
        if isinstance(ckpt['model_config'], dict):
            conf = GPTConfig(**ckpt['model_config'])
        else:
            conf = ckpt['model_config']
        
        # Build model
        model = GPT(conf)
        
        # Clean state dict (remove torch.compile prefix)
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
        model.load_state_dict(state_dict)
        
        # Apply precision
        if dtype == 'float16':
            model.half()
        elif dtype == 'bfloat16':
            model.bfloat16()
        
        model.to(device)
        model.eval()
        
        # Calculate parameters
        params = sum(p.numel() for p in model.parameters())
        
        CURRENT_MODEL = model
        ENC = tiktoken.get_encoding("gpt2")
        CURRENT_ENGINE = "native"
        MODEL_INFO = {
            "name": os.path.basename(path),
            "params": f"{params/1e6:.1f}M",
            "layers": conf.n_layer,
            "heads": conf.n_head,
            "ctx": conf.block_size,
            "dtype": dtype,
            "device": device
        }
        
        return f"""✅ NEURAL LINK ESTABLISHED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model:      {MODEL_INFO['name']}
Parameters: {MODEL_INFO['params']}
Layers:     {MODEL_INFO['layers']}
Heads:      {MODEL_INFO['heads']}
Context:    {MODEL_INFO['ctx']}
Precision:  {dtype.upper()}
Device:     {device}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
    except Exception as e:
        return f"❌ LOAD FAILED: {str(e)}"

def load_gguf(path, device_idx, n_ctx):
    """Load GGUF model via llama.cpp"""
    global CURRENT_MODEL, CURRENT_ENGINE, MODEL_INFO
    
    if not GGUF_AVAILABLE:
        return "❌ GGUF engine unavailable (pip install llama-cpp-python)"
    
    try:
        model = Llama(
            model_path=path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,  # Offload all layers to GPU
            main_gpu=device_idx,
            verbose=False
        )
        
        CURRENT_MODEL = model
        CURRENT_ENGINE = "gguf"
        MODEL_INFO = {
            "name": os.path.basename(path),
            "params": "Unknown",
            "ctx": n_ctx,
            "device": f"cuda:{device_idx}"
        }
        
        return f"""✅ GGUF CORE INITIALIZED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model:   {MODEL_INFO['name']}
Context: {n_ctx}
GPU:     {device_idx}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
    except Exception as e:
        return f"❌ GGUF LOAD FAILED: {str(e)}"

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def stop_generation():
    """Signal to stop generation"""
    global STOP_GENERATION
    STOP_GENERATION = True
    return "⏹️ Stop signal sent..."

def generate(prompt, max_tokens, temperature, top_k, top_p, repeat_penalty, template_name, device_idx):
    """Stream tokens from loaded model"""
    global STOP_GENERATION
    STOP_GENERATION = False
    
    if not CURRENT_MODEL:
        yield "⚠️ NO MODEL LOADED\n\nPlease initialize a model first."
        return
    
    if not prompt.strip():
        yield "⚠️ Empty prompt"
        return
    
    # Apply chat template
    formatted_prompt = apply_template(prompt, template_name)
    
    start_time = time.time()
    token_count = 0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GGUF ENGINE
    # ═══════════════════════════════════════════════════════════════════════════
    if CURRENT_ENGINE == "gguf":
        try:
            stream = CURRENT_MODEL(
                formatted_prompt,  # Use formatted prompt with template
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                repeat_penalty=float(repeat_penalty),  # From slider - prevents repetition!
                frequency_penalty=0.0,     # Additional repetition control
                presence_penalty=0.0,      # Additional repetition control
                stream=True,
                stop=["</s>", "<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<end_of_turn>", "</s>", "\n\n\n"]  # Common stop tokens
            )
            
            partial = ""
            for output in stream:
                if STOP_GENERATION:
                    partial += "\n\n⏹️ [GENERATION STOPPED]"
                    yield partial
                    return
                    
                token = output['choices'][0]['text']
                partial += token
                token_count += 1
                yield partial
                
        except Exception as e:
            yield f"❌ GGUF Generation Error: {str(e)}"
            return
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NATIVE ENGINE
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        device, _, attn_ctx = get_device_config(int(device_idx))
        
        try:
            # Encode prompt (with template applied)
            tokens = ENC.encode(formatted_prompt)
            idx = torch.tensor([tokens], dtype=torch.long, device=device)
            
            partial = formatted_prompt
            
            CURRENT_MODEL.eval()
            with torch.no_grad(), attn_ctx:
                for _ in range(int(max_tokens)):
                    if STOP_GENERATION:
                        partial += "\n\n⏹️ [GENERATION STOPPED]"
                        yield partial
                        return
                    
                    # Crop to context window
                    block_size = CURRENT_MODEL.config.block_size
                    idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
                    
                    # Forward pass
                    logits, _ = CURRENT_MODEL(idx_cond)
                    logits = logits[:, -1, :] / float(temperature)
                    
                    # Top-K filtering
                    if top_k > 0:
                        v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
                        logits[logits < v[:, [-1]]] = float('-inf')
                    
                    # Top-P (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative prob above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                        )
                        logits[indices_to_remove] = float('-inf')
                    
                    # Sample
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    
                    # Decode and yield
                    token_str = ENC.decode([idx_next.item()])
                    partial += token_str
                    token_count += 1
                    
                    idx = torch.cat((idx, idx_next), dim=1)
                    yield partial
                    
        except Exception as e:
            yield f"❌ Native Generation Error: {str(e)}"
            return
    
    # Final stats
    elapsed = time.time() - start_time
    tps = token_count / elapsed if elapsed > 0 else 0
    yield partial + f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n⚡ {token_count} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)"

# ═══════════════════════════════════════════════════════════════════════════════
# UI HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════
def ui_load(engine, path, gpu, ctx):
    """Handle model loading from UI - returns (status_msg, detected_template)"""
    global CURRENT_TEMPLATE
    
    if not path.strip():
        return "❌ No path specified", "None (Raw)"
    if not os.path.exists(path):
        return f"❌ File not found: {path}", "None (Raw)"
    
    # Auto-detect chat template from filename
    detected = detect_template(path)
    CURRENT_TEMPLATE = detected
    template_msg = f"\n💬 Auto-detected template: {detected}" if detected != "None (Raw)" else ""
    
    # Auto-detect engine based on file extension (prevents mismatched loading)
    path_lower = path.lower()
    if path_lower.endswith('.gguf'):
        if engine == "Native (.pt)":
            msg = load_gguf(path, int(gpu), int(ctx)) + "\n⚠️ Auto-switched to GGUF engine" + template_msg
            return msg, detected
        return load_gguf(path, int(gpu), int(ctx)) + template_msg, detected
    elif path_lower.endswith('.pt') or path_lower.endswith('.pth'):
        if engine == "GGUF (.gguf)":
            msg = load_native(path, int(gpu)) + "\n⚠️ Auto-switched to Native engine" + template_msg
            return msg, detected
        return load_native(path, int(gpu)) + template_msg, detected
    else:
        # Fall back to user selection for unknown extensions
        if engine == "Native (.pt)":
            return load_native(path, int(gpu)) + template_msg, detected
        else:
            return load_gguf(path, int(gpu), int(ctx)) + template_msg, detected

def ui_get_stats(gpu):
    """Update GPU stats display"""
    return get_gpu_stats(int(gpu))

def get_system_info():
    """Get system information for display"""
    gpus = get_gpu_info()
    info = "╔═══════════════════════════════════════╗\n"
    info += "║        SYSTEM DIAGNOSTICS             ║\n"
    info += "╠═══════════════════════════════════════╣\n"
    
    for gpu in gpus:
        if gpu['id'] == -1:
            info += f"║  CPU Mode Active                      ║\n"
        else:
            status = "⚡" if gpu.get('is_modern', False) else "⚠️"
            info += f"║  GPU {gpu['id']}: {gpu['name'][:25]:<25} ║\n"
            info += f"║    Memory: {gpu['memory']:.1f} GB | SM: {gpu['compute']:<5} {status}  ║\n"
    
    info += "╚═══════════════════════════════════════╝"
    return info

# ═══════════════════════════════════════════════════════════════════════════════
# CYBERPUNK CSS THEME (FIXED RADIO BUTTONS)
# ═══════════════════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

:root {
    --neon-cyan: #00f3ff;
    --neon-magenta: #ff00ff;
    --neon-yellow: #f3ff00;
    --dark-bg: #0a0e17;
    --panel-bg: rgba(15, 23, 42, 0.95);
    --border-glow: rgba(0, 243, 255, 0.3);
}

body {
    background: linear-gradient(135deg, #0a0e17 0%, #1a1a2e 50%, #0a0e17 100%) !important;
    background-attachment: fixed !important;
}

.gradio-container {
    max-width: 1400px !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* HEADER */
.title-banner {
    background: linear-gradient(90deg, transparent, var(--panel-bg), transparent);
    border: 1px solid var(--border-glow);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.title-text {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 2.5rem !important;
    font-weight: 900 !important;
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-magenta), var(--neon-cyan));
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-shift 3s ease infinite;
    margin: 0;
}

@keyframes gradient-shift {
    0% { background-position: 0% center; }
    50% { background-position: 100% center; }
    100% { background-position: 0% center; }
}

.subtitle {
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.9rem;
    margin-top: 8px;
}

/* PANELS */
.panel-container {
    background: var(--panel-bg) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 20px rgba(0, 243, 255, 0.1);
}

/* SECTION HEADERS */
.section-header {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--neon-cyan) !important;
    font-size: 1rem !important;
    font-weight: 700;
    border-bottom: 1px solid var(--border-glow);
    padding-bottom: 10px;
    margin-bottom: 15px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* INPUTS */
.gradio-textbox textarea, .gradio-textbox input {
    background: rgba(0, 10, 20, 0.8) !important;
    border: 1px solid var(--border-glow) !important;
    color: #e0e0e0 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* RADIO BUTTONS & LABELS (FIXED VISIBILITY) */
.gradio-radio {
    background: transparent !important;
}
/* The radio circle itself */
.gradio-radio input[type="radio"] {
    accent-color: var(--neon-cyan) !important;
}
/* The text label next to the radio button */
.gradio-radio label span {
    color: var(--neon-cyan) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    text-shadow: 0 0 10px rgba(0, 243, 255, 0.3);
}
/* General Labels */
label span {
    color: rgba(255, 255, 255, 0.9) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* BUTTONS */
.primary-btn {
    background: linear-gradient(135deg, rgba(0, 243, 255, 0.2), rgba(255, 0, 255, 0.2)) !important;
    border: 1px solid var(--neon-cyan) !important;
    color: var(--neon-cyan) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
}
.primary-btn:hover {
    box-shadow: 0 0 30px rgba(0, 243, 255, 0.5) !important;
}

.stop-btn {
    background: rgba(255, 50, 50, 0.2) !important;
    border: 1px solid #ff3232 !important;
    color: #ff3232 !important;
}

/* SLIDERS */
.gradio-slider input[type="range"] {
    accent-color: var(--neon-cyan) !important;
}

/* STATUS & OUTPUT */
.status-display textarea {
    background: rgba(0, 20, 40, 0.9) !important;
    color: var(--neon-cyan) !important;
    border: 1px solid rgba(0, 243, 255, 0.3) !important;
}
.output-display textarea {
    background: rgba(0, 5, 15, 0.9) !important;
    color: #00ff88 !important;
    border: 1px solid rgba(0, 243, 255, 0.2) !important;
    font-size: 1rem !important;
}

/* Accordion styling */
.gradio-accordion {
    background: var(--panel-bg) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 8px !important;
    margin-bottom: 15px;
}
.gradio-accordion summary {
    color: var(--neon-cyan) !important;
    font-family: 'Share Tech Mono', monospace !important;
}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
# Create theme that uses web-safe fonts (prevents 404s for ui-sans-serif, system-ui)
cyberpunk_theme = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="purple", 
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Share Tech Mono"),
    font_mono=gr.themes.GoogleFont("Share Tech Mono"),
)

with gr.Blocks(css=CSS, theme=cyberpunk_theme, title="SOVRA OMNI") as demo:
    
    # HEADER
    gr.HTML("""
        <div class="title-banner">
            <h1 class="title-text">SOVRA OMNI</h1>
            <p class="subtitle">Neural Interface v2.1 │ LLM Inference Engine</p>
        </div>
    """)
    
    with gr.Row():
        # ═══════════════════════════════════════════════════════════════════════
        # LEFT PANEL - CONTROLS
        # ═══════════════════════════════════════════════════════════════════════
        with gr.Column(scale=1):
            gr.HTML('<div class="section-header">⚡ System Control</div>')
            
            # Model Selection
            with gr.Group(elem_classes="panel-container"):
                engine_radio = gr.Radio(
                    ["Native (.pt)", "GGUF (.gguf)"],
                    label="Engine Type",
                    value="Native (.pt)"
                )
                
                model_path = gr.Textbox(
                    label="Model Path",
                    value="checkpoints/latest.pt",
                    placeholder="/path/to/model"
                )
                
                with gr.Row():
                    gpu_dropdown = gr.Dropdown(
                        choices=[str(i) for i in range(torch.cuda.device_count() or 1)],
                        value=str(ARGS_DEVICE),
                        label="GPU"
                    )
                    ctx_slider = gr.Slider(
                        512, 8192, value=2048, step=256,
                        label="Context (GGUF)"
                    )
                
                load_btn = gr.Button(
                    "⚡ INITIALIZE NEURAL LINK",
                    elem_classes="primary-btn"
                )
            
            # Status Display
            gr.HTML('<div class="section-header">📊 System Status</div>')
            with gr.Group(elem_classes="panel-container"):
                status_box = gr.Textbox(
                    label="",
                    lines=10,
                    interactive=False,
                    elem_classes="status-display",
                    value=get_system_info()
                )
                
                gpu_stats = gr.Textbox(
                    label="GPU Memory",
                    interactive=False,
                    elem_classes="stats-display",
                    value=get_gpu_stats(ARGS_DEVICE)
                )
                
                refresh_btn = gr.Button("🔄 Refresh Stats", size="sm")
            
            # Model Downloader Section
            gr.HTML('<div class="section-header">📥 Model Downloader</div>')
            with gr.Accordion("Download GGUF Models from HuggingFace", open=False):
                with gr.Group(elem_classes="panel-container"):
                    model_select = gr.Dropdown(
                        choices=get_model_list(),
                        label="Select Model",
                        info="Popular pre-configured models"
                    )
                    
                    gr.Markdown("**Or enter custom HuggingFace repo:**")
                    with gr.Row():
                        custom_repo = gr.Textbox(
                            label="Repo ID",
                            placeholder="username/repo-name",
                            scale=2
                        )
                        custom_file = gr.Textbox(
                            label="Filename",
                            placeholder="model.gguf",
                            scale=2
                        )
                    
                    with gr.Row():
                        download_btn = gr.Button("⬇️ Download Model", variant="primary")
                        list_local_btn = gr.Button("📁 List Local", size="sm")
                    
                    download_status = gr.Textbox(
                        label="Download Status",
                        lines=4,
                        interactive=False,
                        value=ui_list_local_models()
                    )
                    
                    downloaded_path = gr.Textbox(visible=False)  # Hidden field to store path
                    
                    use_model_btn = gr.Button("✅ Use This Model", visible=True, size="sm")
        
        # ═══════════════════════════════════════════════════════════════════════
        # RIGHT PANEL - GENERATION
        # ═══════════════════════════════════════════════════════════════════════
        with gr.Column(scale=2):
            gr.HTML('<div class="section-header">🧠 Neural Output</div>')
            
            with gr.Group(elem_classes="panel-container"):
                output_box = gr.Textbox(
                    label="",
                    lines=18,
                    interactive=False,
                    elem_classes="output-display",
                    placeholder="Awaiting neural transmission..."
                )
            
            gr.HTML('<div class="section-header">📝 Input Terminal</div>')
            with gr.Group(elem_classes="panel-container"):
                input_box = gr.Textbox(
                    label="",
                    lines=3,
                    placeholder="Enter prompt for neural processing...",
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        temp_slider = gr.Slider(
                            0.1, 2.0, value=0.8, step=0.05,
                            label="Temperature"
                        )
                    with gr.Column(scale=1):
                        max_tokens_slider = gr.Slider(
                            10, 2000, value=256, step=10,
                            label="Max Tokens"
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        top_k_slider = gr.Slider(
                            0, 500, value=50, step=5,
                            label="Top-K (0=disabled)"
                        )
                    with gr.Column(scale=1):
                        top_p_slider = gr.Slider(
                            0.0, 1.0, value=0.95, step=0.01,
                            label="Top-P (nucleus)"
                        )
                    with gr.Column(scale=1):
                        repeat_penalty_slider = gr.Slider(
                            1.0, 2.0, value=1.1, step=0.05,
                            label="Repeat Penalty"
                        )
                
                with gr.Row():
                    template_dropdown = gr.Dropdown(
                        choices=list(CHAT_TEMPLATES.keys()),
                        value="None (Raw)",
                        label="💬 Chat Template (auto-detected on load)",
                        info="Wraps your prompt in the correct format"
                    )
                
                with gr.Row():
                    generate_btn = gr.Button(
                        "🚀 TRANSMIT",
                        elem_classes="primary-btn"
                    )
                    stop_btn = gr.Button(
                        "⏹️ HALT",
                        elem_classes="stop-btn"
                    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════
    load_btn.click(
        ui_load,
        inputs=[engine_radio, model_path, gpu_dropdown, ctx_slider],
        outputs=[status_box, template_dropdown]  # Now also updates template dropdown
    )
    
    refresh_btn.click(
        ui_get_stats,
        inputs=[gpu_dropdown],
        outputs=gpu_stats
    )
    
    # Generation with chat template support
    generate_btn.click(
        generate,
        inputs=[input_box, max_tokens_slider, temp_slider, top_k_slider, top_p_slider, repeat_penalty_slider, template_dropdown, gpu_dropdown],
        outputs=output_box
    )
    
    input_box.submit(
        generate,
        inputs=[input_box, max_tokens_slider, temp_slider, top_k_slider, top_p_slider, repeat_penalty_slider, template_dropdown, gpu_dropdown],
        outputs=output_box
    )
    
    stop_btn.click(stop_generation, outputs=status_box)
    
    # Download handlers
    download_btn.click(
        ui_download_model,
        inputs=[model_select, custom_repo, custom_file],
        outputs=[download_status, downloaded_path]
    )
    
    list_local_btn.click(
        ui_list_local_models,
        outputs=download_status
    )
    
    use_model_btn.click(
        ui_use_downloaded,
        inputs=[downloaded_path],
        outputs=model_path
    )
    
    # Note: demo.load() removed - use Refresh Stats button instead
    # (Older Gradio versions don't support inputs in demo.load)

# ═══════════════════════════════════════════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  SOVRA OMNI v2.1 - Initializing...                                            ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Port:   {args.port:<6}                                                       ║
║  Share:  {str(args.share):<6}                                                 ║
║  GPU:    {ARGS_DEVICE:<6}                                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )
