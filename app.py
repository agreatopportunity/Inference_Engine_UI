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
from datetime import datetime

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

def generate(prompt, max_tokens, temperature, top_k, top_p, repeat_penalty, device_idx):
    """Stream tokens from loaded model"""
    global STOP_GENERATION
    STOP_GENERATION = False
    
    if not CURRENT_MODEL:
        yield "⚠️ NO MODEL LOADED\n\nPlease initialize a model first."
        return
    
    if not prompt.strip():
        yield "⚠️ Empty prompt"
        return
    
    start_time = time.time()
    token_count = 0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GGUF ENGINE
    # ═══════════════════════════════════════════════════════════════════════════
    if CURRENT_ENGINE == "gguf":
        try:
            stream = CURRENT_MODEL(
                prompt,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                repeat_penalty=float(repeat_penalty),  # From slider - prevents repetition!
                frequency_penalty=0.0,     # Additional repetition control
                presence_penalty=0.0,      # Additional repetition control
                stream=True,
                stop=["</s>", "<|endoftext|>", "<|im_end|>", "\n\n\n"]  # Common stop tokens
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
            # Encode prompt
            tokens = ENC.encode(prompt)
            idx = torch.tensor([tokens], dtype=torch.long, device=device)
            
            partial = prompt
            
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
    """Handle model loading from UI"""
    if not path.strip():
        return "❌ No path specified"
    if not os.path.exists(path):
        return f"❌ File not found: {path}"
    
    if engine == "Native (.pt)":
        return load_native(path, int(gpu))
    else:
        return load_gguf(path, int(gpu), int(ctx))

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
            <p class="subtitle">Neural Interface v2.0 │ LLM Inference Engine</p>
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
        outputs=status_box
    )
    
    refresh_btn.click(
        ui_get_stats,
        inputs=[gpu_dropdown],
        outputs=gpu_stats
    )
    
    # FIX: Now passing correct parameters including repeat_penalty
    generate_btn.click(
        generate,
        inputs=[input_box, max_tokens_slider, temp_slider, top_k_slider, top_p_slider, repeat_penalty_slider, gpu_dropdown],
        outputs=output_box
    )
    
    input_box.submit(
        generate,
        inputs=[input_box, max_tokens_slider, temp_slider, top_k_slider, top_p_slider, repeat_penalty_slider, gpu_dropdown],
        outputs=output_box
    )
    
    stop_btn.click(stop_generation, outputs=status_box)
    
    # Note: demo.load() removed - use Refresh Stats button instead
    # (Older Gradio versions don't support inputs in demo.load)

# ═══════════════════════════════════════════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  SOVRA OMNI v2.0 - Initializing...                                            ║
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
