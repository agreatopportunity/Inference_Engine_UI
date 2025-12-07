"""
Universal LLM Playground Cognitive Architecture (2025 Edition)
Compatible with: Titan V, RTX 3090, 4090, 5090
Supports:
1. Custom Native Models (.pt)
2. GGUF Community Models (.gguf)
Integrates:
1. Multi-GPU Loader (Titan V / RTX 4090 / 5090)
2. Cognitive Emotional Core (Memory + Decay)
"""
import gradio as gr
import torch
import tiktoken
import contextlib
import argparse
import os
import sys

# --- HELP MENU & ARGUMENTS ---
if __name__ == "__main__":
    # Check for help flag manually before parsing to ensure custom message prints
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("""
LLM UI - Command Line Arguments
-----------------------------------
--device_id : GPU Index to use (Default: 0).
              Common Setup: 0 = Titan V, 1 = GTX 1080 (Check nvidia-smi if unsure)
--port      : Port to run the UI on (Default: 7860)
--share     : Create a public share link (Useful for mobile access)

Example: python3 app.py --device_id 0 --share
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

# --- IMPORT ARCHITECTURE (Native) ---
try:
    from model_llama3 import GPT, GPTConfig
except ImportError:
    print("⚠️ model_llama3.py not found. Native models will fail.")

# --- IMPORT GGUF ENGINE ---
try:
    from llama_cpp import Llama
    GGUF_AVAILABLE = True
except ImportError:
    print("⚠️ llama-cpp-python not found. GGUF models will fail.")
    GGUF_AVAILABLE = False

# --- HARDWARE CONFIG ---
def get_device_config(device_id):
    if not torch.cuda.is_available():
        return "cpu", "float32", contextlib.nullcontext()
    
    device = f"cuda:{device_id}"
    try:
        props = torch.cuda.get_device_properties(device)
        print(f"[{props.name}] Detected on Device {device_id}")
        
        # Auto-downgrade for Titan V / Pascal (Compute < 8.0)
        if props.major < 8:
            dtype = "float16"
            # Force efficient attention (disable Flash)
            attn = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True)
            print(f"   ↳ Legacy GPU detected. Force-enabling float16 & Efficient Attention.")
        else:
            dtype = "bfloat16"
            attn = contextlib.nullcontext()
            print(f"   ↳ Modern GPU detected. Enabling bfloat16 & Flash Attention.")
            
        return device, dtype, attn
    except Exception as e:
        print(f"❌ Error initializing GPU {device_id}: {e}")
        return "cpu", "float32", contextlib.nullcontext()

# --- GLOBAL STATE ---
CURRENT_MODEL = None
CURRENT_ENGINE = None # "native" or "gguf"
ENC = None

# --- LOADERS ---
def load_native(path, device_idx):
    global CURRENT_MODEL, ENC, CURRENT_ENGINE
    
    device, dtype, _ = get_device_config(device_idx)
    print(f"Loading Native: {path}...")
    
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(ckpt['model_config'], dict):
            conf = GPTConfig(**ckpt['model_config'])
        else:
            conf = ckpt['model_config']
            
        model = GPT(conf)
        # Clean state dict (remove torch.compile prefix)
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
        model.load_state_dict(state_dict)
        
        # Cast Precision based on Hardware Doctor
        if dtype == 'float16': model.half()
        elif dtype == 'bfloat16': model.bfloat16()
        
        model.to(device)
        model.eval()
        
        CURRENT_MODEL = model
        ENC = tiktoken.get_encoding("gpt2")
        CURRENT_ENGINE = "native"
        return f"✅ Loaded Native Model on GPU {device_idx} ({dtype})"
    except Exception as e:
        return f"❌ Native Load Failed: {str(e)}"

def load_gguf(path, device_idx, n_ctx):
    global CURRENT_MODEL, CURRENT_ENGINE
    
    if not GGUF_AVAILABLE: return "❌ Error: llama-cpp-python not installed."
    
    print(f"Loading GGUF: {path} on GPU {device_idx}...")
    try:
        # n_gpu_layers=-1 attempts to offload ALL layers to GPU
        model = Llama(
            model_path=path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            main_gpu=device_idx,
            verbose=False
        )
        
        CURRENT_MODEL = model
        CURRENT_ENGINE = "gguf"
        return f"✅ Loaded GGUF Model on GPU {device_idx}"
    except Exception as e:
        return f"❌ GGUF Load Failed: {str(e)}"

# --- GENERATOR ---
def generate(prompt, max_tokens, temp, top_k, device_idx):
    if not CURRENT_MODEL:
        yield "⚠️ Please LOAD a model first!"
        return

    # === GGUF ENGINE ===
    if CURRENT_ENGINE == "gguf":
        stream = CURRENT_MODEL(
            prompt, 
            max_tokens=max_tokens, 
            temperature=temp, 
            top_k=top_k, 
            stream=True
        )
        partial_text = ""
        for output in stream:
            token = output['choices'][0]['text']
            partial_text += token
            yield partial_text

    # === NATIVE ENGINE ===
    else:
        device, _, attn_ctx = get_device_config(device_idx)
        idx = torch.tensor(ENC.encode(prompt)).unsqueeze(0).to(device)
        
        partial_text = prompt
        
        CURRENT_MODEL.eval()
        with torch.no_grad(), attn_ctx:
            for _ in range(max_tokens):
                # Handle context window limits
                if idx.size(1) > CURRENT_MODEL.config.block_size:
                    idx_cond = idx[:, -CURRENT_MODEL.config.block_size:]
                else:
                    idx_cond = idx
                    
                logits, _ = CURRENT_MODEL(idx_cond)
                logits = logits[:, -1, :] / temp
                
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                    
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                token = ENC.decode([idx_next.item()])
                partial_text += token
                idx = torch.cat((idx, idx_next), dim=1)
                
                yield partial_text

# --- UI LOGIC ---
def ui_load(engine, path, gpu, ctx):
    if not os.path.exists(path):
        return f"❌ File not found: {path}"
        
    if engine == "Native (.pt)":
        return load_native(path, int(gpu))
    else:
        return load_gguf(path, int(gpu), int(ctx))

# --- LAYOUT ---
CSS = """
body { background: #0b0f19; }
.gradio-container { max-width: 1200px !important; }
.panel { background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 20px; }
.title { font-family: 'Orbitron', sans-serif; color: #00f3ff; text-align: center; font-size: 2rem; margin-bottom: 1rem; }
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML('<div class="title">SOVRA OMNI INTERFACE</div>')
    
    with gr.Row():
        # CONTROLS
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 🔌 Model Loader")
            engine_drop = gr.Radio(["Native (.pt)", "GGUF (.gguf)"], label="Engine", value="Native (.pt)")
            path_input = gr.Textbox(label="File Path", value="checkpoints/latest.pt")
            
            with gr.Row():
                gpu_drop = gr.Dropdown(["0", "1"], label="GPU Index", value=str(ARGS_DEVICE))
                ctx_slider = gr.Slider(512, 8192, value=2048, step=256, label="Context (GGUF Only)")
            
            load_btn = gr.Button("INITIALIZE SYSTEM", variant="primary")
            status_box = gr.Textbox(label="System Status", interactive=False)
            
        # CHAT
        with gr.Column(scale=2):
            chatbot = gr.Textbox(label="Neural Feed", lines=20, interactive=False)
            msg = gr.Textbox(label="Input", placeholder="Transmit data...")
            with gr.Row():
                temp_slide = gr.Slider(0.1, 2.0, 0.8, label="Temperature")
                max_slide = gr.Slider(10, 1000, 200, label="Max Tokens")
            
            send_btn = gr.Button("SEND", variant="secondary")

    # WIRING
    load_btn.click(ui_load, [engine_drop, path_input, gpu_drop, ctx_slider], status_box)
    
    # Pass GPU index to generation so it knows where to put the input tensors
    send_btn.click(generate, [msg, max_slide, temp_slide, max_slide, gpu_drop], chatbot)
    msg.submit(generate, [msg, max_slide, temp_slide, max_slide, gpu_drop], chatbot)

if __name__ == "__main__":
    print(f"🚀 Launching LLM UI on Port {args.port}...")
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
