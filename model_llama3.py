"""
LLaMA-3 Style Architecture (2025 Edition)
Optimized for Consumer GPUs (RTX 4060 Ti / 4090)

Key LLaMA-3 Features Implemented:
1. RMSNorm (Pre-normalization)
2. SwiGLU Activation in MLP
3. RoPE (Rotary Positional Embeddings) with high theta (500k)
4. Grouped Query Attention (GQA) with correct repetition logic
5. QK-Norm (Normalization of Queries/Keys before Attention)
6. Weight Tying (Optimized for VRAM efficiency)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configurations
# =============================================================================

@dataclass
class GPTConfig:
    block_size: int = 2048        # Context length (can go to 8192+ if VRAM allows)
    vocab_size: int = 50304       # GPT-2 vocab size (rounded to multiple of 64)
    n_layer: int = 12             # Layers
    n_head: int = 12              # Query heads
    n_kv_head: Optional[int] = 4  # KV heads (If < n_head, GQA is active)
    n_embd: int = 768             # Embedding dimension
    dropout: float = 0.0
    bias: bool = False            # False = LLaMA style (no bias in Linears)
    rope_theta: float = 500000.0  # LLaMA-3 standard theta


MODEL_CONFIGS = {
    # 4060 Ti Friendly (High speed, good for debugging)
    "tiny":   GPTConfig(n_layer=4, n_head=8, n_kv_head=4, n_embd=256),
    "small":  GPTConfig(n_layer=6, n_head=12, n_kv_head=4, n_embd=384),
    
    # Solid Performance (Requires ~8-10GB VRAM)
    "base":   GPTConfig(n_layer=12, n_head=12, n_kv_head=4, n_embd=768),
    
    # LLaMA-3 1B equivalent (Requires ~16GB+ VRAM or Accumulation)
    "large":  GPTConfig(n_layer=16, n_head=16, n_kv_head=8, n_embd=1024),
}


# =============================================================================
# Core Components (Norms & RoPE)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x * rsqrt(x^2 + eps)
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 500000.0):
        super().__init__()
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len: int):
        # Update cache if sequence length exceeds current cache
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            # Different from standard RoPE: LLaMA uses polar form
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos().to(x.dtype)
            self.sin_cached = emb.sin().to(x.dtype)

        return (
            self.cos_cached[:seq_len, ...],
            self.sin_cached[:seq_len, ...]
        )


def apply_rope(q, k, cos, sin):
    # q, k: [B, H, T, D]
    # cos, sin: [T, D] -> reshape to [1, 1, T, D]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Rotate half trick
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the GQA fix.
    If we have 4 KV heads and 32 Query heads, we must repeat the KV heads 8 times.
    (B, n_kv, T, D) -> (B, n_head, T, D)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# =============================================================================
# Attention & MLP
# =============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        
        # Projections
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # LLaMA-3 QK Normalization
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # RoPE
        self.rotary = RotaryEmbedding(self.head_dim, theta=config.rope_theta)

    def forward(self, x):
        B, T, C = x.size()

        # 1. Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape for heads: [B, T, H, D] -> [B, H, T, D]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # 3. Apply QK Norm (LLaMA-3 specific)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4. Apply RoPE
        cos, sin = self.rotary(v, T)
        q, k = apply_rope(q, k, cos, sin)

        # 5. Handle GQA (Repeat KV heads to match Q heads)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # 6. Flash Attention
        # is_causal=True handles the masking automatically
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # 7. Reassemble
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 8. Output projection
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Hidden dimension size logic from LLaMA
        # int(2/3 * 4 * d_model) -> rounded to multiple of 256 for efficiency
        hidden_dim = int(2 * 4 * config.n_embd / 3)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False) # Gate
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False) # Down
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False) # Up

    def forward(self, x):
        # SiLU(Gate) * Up
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# =============================================================================
# Main Model
# =============================================================================

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        
        # Output Head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight Tying (Optional in LLaMA, but vital for 16GB VRAM training)
        self.transformer.wte.weight = self.lm_head.weight

        # Init
        self.apply(self._init_weights)
        
        # Scale residuals (GPT-2 style init trick, helps convergence)
        for pn, p in self.named_parameters():
            if pn.endswith('w2.weight') or pn.endswith('o_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        # Input Embeddings
        x = self.transformer.wte(idx)
        
        # Transformer Blocks
        for block in self.transformer.h:
            x = block(x)
            
        # Final Norm
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            # FIX: Explicitly cast targets to long for torch.compile compatibility
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1).long())
        else:
            # Inference optimization: only compute logits for last token
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # Crop to block_size if context is full
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
