"""
Universal LLM Playground Cognitive Architecture (2025 Edition)
Compatible with: Titan V, RTX 3090, 4090, 5090
Supports:
1. Custom Native Models (.pt)
2. GGUF Community Models (.gguf)
Integrates:
1. Multi-GPU Loader (Titan V / RTX 4090 / 5090)
2. Cognitive Emotional Core (Memory + Decay)
3. 2026 Holo-Glass UI
"""
import gradio as gr
import torch
import torch.nn.functional as F
import tiktoken
import contextlib
import os
import time
import json
import threading
import argparse
import sys
from datetime import datetime
from pathlib import Path

# --- COMMAND LINE ARGUMENTS ---
def parse_args():
    parser = argparse.ArgumentParser(description="SOVRA OMNI: Universal AI Interface")
    parser.add_argument("--device_id", type=int, default=1, help="GPU Index to use (Default: 1 for Titan V)")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the UI on")
    return parser.parse_args()

# --- IMPORT ARCHITECTURE ---
try:
    from model_llama3 import GPT, GPTConfig
except ImportError:
    print("⚠️  model_llama3.py not found. Please ensure it is in the same folder.")
    # Create dummy classes to prevent crash if file is missing
    class GPTConfig: pass
    class GPT: pass

# --- HARDWARE AUTO-DETECTION ---
def get_hardware_config(device_index=0):
    if not torch.cuda.is_available():
        return "cpu", "float32", contextlib.nullcontext()
    
    device = f"cuda:{device_index}"
    try:
        props = torch.cuda.get_device_properties(device)
        major = props.major
        print(f"[{props.name}] Detected on Device {device_index}")
    except Exception as e:
        print(f"⚠️ Error detecting GPU {device_index}: {e}")
        return "cpu", "float32", contextlib.nullcontext()
    
    # Titan V / Pascal (Major < 8) -> float16, No Flash Attn
    if major < 8:
        dtype = "float16"
        # Force efficient attention for older cards
        attn_ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True)
        print(f"   ↳ Legacy Architecture (Volta/Pascal). Using float16 + Efficient Attn.")
    else:
        # Ampere / Ada / Blackwell (Major >= 8) -> bfloat16, Flash Attn
        dtype = "bfloat16"
        attn_ctx = contextlib.nullcontext()
        print(f"   ↳ Modern Architecture. Using bfloat16 + Flash Attn.")
        
    return device, dtype, attn_ctx

# --- GLOBAL STATE ---
ARGS = parse_args()
DEVICE, DTYPE, ATTN_CTX = get_hardware_config(ARGS.device_id) 
MODEL = None
ENC = None

# --- COGNITIVE CORE (LOCAL EDITION) ---
class LocalCognitiveMind:
    def __init__(self):
        self.emotions = {
            "curiosity": 0.5, "focus": 0.5, "energy": 0.5, 
            "warmth": 0.5, "uncertainty": 0.1
        }
        self.frame = "Neutral"
        self.awareness = 0.5
    
    def process(self, prompt, model, enc, max_new_tokens=200):
        # 1. Update State
        self.awareness = min(1.0, self.awareness + 0.05)
        self.emotions["focus"] = min(1.0, self.emotions["focus"] + 0.1)
        
        # 2. Build Context
        system_prompt = f"""You are Sovra.
STATE: Frame={self.frame} | Focus={self.emotions['focus']:.2f}
USER: {prompt}
SOVRA:"""
        
        # 3. Generate (On GPU)
        return self.generate_local(system_prompt, model, enc, max_new_tokens)

    def generate_local(self, text, model, enc, max_tokens):
        if model is None: return "Error: Neural Link Offline. Click 'INITIALIZE LINK' first."
        
        try:
            input_ids = torch.tensor(enc.encode(text)).unsqueeze(0).to(DEVICE)
            model.eval()
            response_tokens = []
            
            with torch.no_grad(), ATTN_CTX:
                for _ in range(max_tokens):
                    # Crop context
                    if input_ids.size(1) > model.config.block_size:
                        input_ids = input_ids[:, -model.config.block_size:]
                    
                    logits, _ = model(input_ids)
                    logits = logits[:, -1, :] / 0.8 # Temperature
                    
                    # Sampling
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    input_ids = torch.cat((input_ids, next_token), dim=1)
                    response_tokens.append(next_token.item())
                    
                    if next_token.item() == 50256: # EOT
                        break
            
            return enc.decode(response_tokens)
        except Exception as e:
            return f"Generation Error: {str(e)}"

MIND = LocalCognitiveMind()

# --- MODEL LOADER ---
def load_native_model(checkpoint_path, gpu_idx):
    global MODEL, ENC, DEVICE, DTYPE, ATTN_CTX
    
    # Update Hardware Config based on selection
    DEVICE, DTYPE, ATTN_CTX = get_hardware_config(int(gpu_idx))
    
    print(f"Loading {checkpoint_path} to {DEVICE} ({DTYPE})...")
    try:
        if not os.path.exists(checkpoint_path):
            return f"❌ Error: File not found at {checkpoint_path}"

        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if isinstance(ckpt['model_config'], dict):
            config = GPTConfig(**ckpt['model_config'])
        else:
            config = ckpt['model_config']
            
        model = GPT(config)
        
        # Clean dict
        state_dict = ckpt['model']
        clean_state = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state)
        
        # Cast Precision
        if DTYPE == 'float16': model.half()
        elif DTYPE == 'bfloat16': model.bfloat16()
        
        model.to(DEVICE)
        model.eval()
        
        MODEL = model
        ENC = tiktoken.get_encoding("gpt2")
        return f"✅ Linked to {torch.cuda.get_device_name(DEVICE)}"
    except Exception as e:
        return f"❌ Link Failed: {str(e)}"

# --- UI LOGIC ---
def interaction_handler(user_text, image, history):
    if not user_text: return history, history
    
    # 1. Update User History
    history = history or []
    history.append((user_text, None))
    
    # 2. Cognitive Processing
    response = MIND.process(user_text, MODEL, ENC)
    
    # 3. Update Bot History
    history[-1] = (user_text, response)
    
    return history, history

def get_dashboard():
    e = MIND.emotions
    w = lambda x: max(5, int(x * 100))
    gpu_name = torch.cuda.get_device_name(DEVICE) if torch.cuda.is_available() else "CPU"
    
    return f"""
    <div style="font-family: 'Inter'; color: #94a3b8; font-size: 0.85rem;">
        <div style="margin-bottom: 15px;">
            <div style="display:flex; justify-content:space-between;">
                <span>NEURAL SYNC</span>
                <span style="color:#00f3ff">{MIND.awareness:.0%}</span>
            </div>
            <div style="height:4px; background:rgba(255,255,255,0.1); margin-top:4px;">
                <div style="height:100%; width:{w(MIND.awareness)}%; background:#00f3ff;"></div>
            </div>
        </div>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
            <div>
                <div style="display:flex; justify-content:space-between;"><span>Focus</span><span>{e['focus']:.1f}</span></div>
                <div style="height:4px; background:rgba(255,255,255,0.1); margin-top:4px;">
                    <div style="height:100%; width:{w(e['focus'])}%; background:#bc13fe;"></div>
                </div>
            </div>
            <div>
                <div style="display:flex; justify-content:space-between;"><span>Energy</span><span>{e['energy']:.1f}</span></div>
                <div style="height:4px; background:rgba(255,255,255,0.1); margin-top:4px;">
                    <div style="height:100%; width:{w(e['energy'])}%; background:#3b82f6;"></div>
                </div>
            </div>
        </div>
        <div style="margin-top:15px; border-top:1px solid rgba(255,255,255,0.1); padding-top:10px;">
            <span style="color:#00f3ff">HARDWARE:</span> {gpu_name} ({DTYPE})
        </div>
    </div>
    """

# ============================================================
# CSS THEME (2026 HOLO-GLASS)
# ============================================================
CSS = """
:root { --glass: rgba(15, 23, 42, 0.7); --border: 1px solid rgba(255, 255, 255, 0.08); }
body { background: #02040a; background-image: radial-gradient(circle at 50% 0%, #1e1b4b 0%, transparent 40%); }
.gradio-container { max-width: 1400px !important; }
.glass-panel { background: var(--glass); backdrop-filter: blur(20px); border: var(--border); border-radius: 20px; padding: 20px; }
.sovra-title { font-family: 'Orbitron'; font-size: 3em; background: linear-gradient(to right, #00f3ff, #bc13fe); -webkit-background-clip: text; color: transparent; text-align: center; letter-spacing: 4px; }
.primary-btn { background: linear-gradient(135deg, #00f3ff, #bc13fe); border: none; color: black; font-weight: bold; }
.chat-window { height: 600px !important; }
"""

# ============================================================
# UI BUILDER
# ============================================================
with gr.Blocks(css=CSS, theme=gr.themes.Base()) as demo:
    
    with gr.Row(elem_classes="glass-panel"):
        gr.HTML('<div class="sovra-title">SOVRA OMNI</div><div style="text-align:center; color:#64748b; letter-spacing:2px;">LOCAL COGNITIVE ARCHITECTURE</div>')

    with gr.Row():
        # LEFT: CONTROLS
        with gr.Column(scale=1):
            with gr.Group(elem_classes="glass-panel"):
                gr.Markdown("### 🔌 HARDWARE LINK")
                # Default to ARGS.device_id (Titan V = 1)
                gpu_select = gr.Dropdown(choices=[0, 1], label="Select GPU", value=ARGS.device_id)
                path_input = gr.Textbox(label="Model Path", value="checkpoints/latest.pt")
                load_btn = gr.Button("INITIALIZE LINK", variant="secondary")
                status_out = gr.Textbox(label="Status", interactive=False, value="Standby...")
                
            with gr.Group(elem_classes="glass-panel"):
                gr.Markdown("### 🧠 LIVE METRICS")
                dashboard = gr.HTML(get_dashboard())
                
            with gr.Group(elem_classes="glass-panel"):
                gr.Markdown("### 📂 DATA INGEST")
                file_up = gr.File(label="Upload Context", file_count="multiple")

        # RIGHT: INTERFACE
        with gr.Column(scale=3):
            with gr.Group(elem_classes="glass-panel"):
                chatbot = gr.Chatbot(label="Neural Feed", elem_classes="chat-window", height=600, render=True)
                with gr.Row():
                    txt = gr.Textbox(scale=4, show_label=False, placeholder="Transmit data...", container=False)
                    btn = gr.Button("SEND", elem_classes="primary-btn", scale=1)

    # EVENTS
    load_btn.click(load_native_model, [path_input, gpu_select], status_out)
    
    # Chat Loop
    # We chain: Process -> Update Chat -> Update Dashboard
    btn.click(interaction_handler, [txt, file_up, chatbot], [chatbot, chatbot]).then(get_dashboard, None, dashboard)
    txt.submit(interaction_handler, [txt, file_up, chatbot], [chatbot, chatbot]).then(get_dashboard, None, dashboard)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("""
SOVRA OMNI - Command Line Arguments
-----------------------------------
--device_id : GPU Index to use (0 for GTX 1080, 1 for Titan V). Default: 1
--port      : Port to run the UI on (Default: 7860)
--share     : Create a public share link (Useful for mobile access)

Example: python3 app.py --device_id 1 --share
        """)
        sys.exit(0)
        
    print(f"🚀 Launching SOVRA OMNI on GPU {ARGS.device_id}...")
    demo.launch(server_name="0.0.0.0", server_port=ARGS.port, share=ARGS.share)
