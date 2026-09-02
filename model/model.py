import math
import os
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module): 
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FIX.md 25: fp32 accumulation for the reduction — strictly better
        # numerics over 128 bf16 residual additions; the cast back keeps the
        # stream in bf16. (Tiny numeric delta vs the old bf16 reduction.)
        xf = x.float()
        norm = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)
        return norm * self.weight

def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0,
                      device=None, dtype=torch.float32):
    """Precompute cos/sin tables for RoPE, shape (seq_len, head_dim)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)          # (seq_len, head_dim/2)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)
 
 
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)
 
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (batch, heads, seq_len, head_dim); cos/sin: (seq_len, head_dim)."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
 
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
 
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
 
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
 
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
 
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
 
        # FIX.md H3: rope output is non-contiguous (broadcast mul + chunk/cat),
        # which can make SDPA silently fall back to the math backend inside the
        # gradient-checkpoint recompute. The math backend materializes the full
        # B x H x T x T attention matrix: ~17 GB at B=8/T=8192 (measured peak
        # 38.8 GB in the probe vs ~25 expected). Contiguous q/k/v + pinning the
        # kernel to flash/efficient makes the workspace linear in T.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        if q.is_cuda:
            sdp_ctx = torch.nn.attention.sdpa_kernel([
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            ])
        else:
            sdp_ctx = contextlib.nullcontext()
        with sdp_ctx:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 8 / 3, ffn_hidden: int | None = None):
        super().__init__()
        hidden = ffn_hidden if ffn_hidden is not None else int(dim * hidden_mult)
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_hidden: int | None = None):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden=ffn_hidden)
        # FIX.md 24: LayerScale — learnable per-branch gains, init 1.0 so step 0
        # is exactly function-preserving. 1.0 * x is bitwise identical in bf16.
        self.ls_attn = nn.Parameter(torch.ones(dim))
        self.ls_ffn = nn.Parameter(torch.ones(dim))
 
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.ls_attn * self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ls_ffn * self.ffn(self.ffn_norm(x))
        return x

def _run_blocks(blocks, x, cos, sin):
    """Forward through a segment of blocks (used by activation checkpointing).

    FIX.md 19 LayerDrop: in training only, each block is skipped with prob p
    (LOCLLM_LAYERDROP, default 0.05). The skip decision uses TORCH rng — the
    checkpoint recompute replays the saved rng state, so forward and recompute
    drop the SAME blocks (correct gradients). Eval never drops.
    """
    p = 0.0
    if len(blocks) and blocks[0].training:
        p = float(os.environ.get("LOCLLM_LAYERDROP", "0.05"))
    for block in blocks:
        if p > 0 and torch.rand(1).item() < p:
            continue
        x = block(x, cos, sin)
    return x


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 512, n_layers: int = 6,
                 n_heads: int = 8, max_seq_len: int = 1024, rope_base: float = 10000.0,
                 ffn_hidden: int | None = None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.head_dim = dim // n_heads
        self.rope_base = rope_base
        self._rope_cache = None  # (cos, sin) full-length tables, built lazily per device
 
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads, ffn_hidden=ffn_hidden) for _ in range(n_layers)])
        self.final_norm = RMSNorm(dim)
        # FIX.md 23: UNTIED output head. Fresh models start tied-equivalent
        # (copy below), old tied checkpoints load unchanged because their state
        # dicts already contain lm_head.weight.
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
 
        self.apply(self._init_weights)
        # FIX.md 26: depth-scaled init for the residual-output projections.
        # Loaded checkpoints overwrite these; only fresh weights are affected.
        depth_std = 0.02 / math.sqrt(2 * n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=depth_std)
            nn.init.normal_(block.ffn.w_down.weight, mean=0.0, std=depth_std)
        self.lm_head.weight.data.copy_(self.tok_emb.weight.data)
        # FIX.md 34: torch.compile each Block once (gated — verify interaction
        # with the non-reentrant checkpointing before enabling in training).
        self._compiled = False
        self._enable_compile()

    def _enable_compile(self):
        if os.environ.get("LOCLLM_COMPILE") == "1" and not self._compiled:
            try:
                for block in self.blocks:
                    block.forward = torch.compile(block.forward)
                self._compiled = True
                print("torch.compile enabled for all blocks (LOCLLM_COMPILE=1)", flush=True)
            except Exception as e:
                print(f"WARNING: torch.compile failed ({e}) — running eager", flush=True)
 
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
 
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, return_hidden: bool = False):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        if self._rope_cache is None or self._rope_cache[0].device != idx.device:
            cos_full, sin_full = build_rope_cache(self.max_seq_len, self.head_dim,
                                                  base=self.rope_base, device=idx.device,
                                                  dtype=self.tok_emb.weight.dtype)
            self._rope_cache = (cos_full, sin_full)
        cos, sin = self._rope_cache[0][:T], self._rope_cache[1][:T]

        x = self.tok_emb(idx)
        # FIX.md H1: activation checkpointing in segments of CKPT_SEG blocks.
        # Non-reentrant checkpointing saves the segment INPUT (B x T x DIM) per
        # segment for backward: 128 per-block segments = ~17 GB at B=8/T=8192
        # (OOMs a 48 GB card). Segments of 4 cut that buffer to ~4.3 GB and
        # recompute cost stays one extra forward pass — the math is identical.
        ckpt_seg = int(os.environ.get("LOCLLM_CKPT_SEG", "4"))
        for i in range(0, len(self.blocks), ckpt_seg):
            x = torch.utils.checkpoint.checkpoint(
                _run_blocks, self.blocks[i:i + ckpt_seg], x, cos, sin,
                use_reentrant=False)
        x = self.final_norm(x)

        if return_hidden:
            return x

        logits = self.lm_head(x)
 
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
 
        return logits, loss
 
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 stop_tokens=None):
        stop = set(stop_tokens) if stop_tokens else None
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            if stop is not None and int(next_tok.item()) in stop:
                break
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    vocab_size = 1000
    model = Transformer(vocab_size=vocab_size, dim=256, n_layers=4, n_heads=4,
                               max_seq_len=128).to(device)
 
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}")
    print(f"params: {n_params / 1e6:.2f}M")
 
    x = torch.randint(0, vocab_size, (2, 32), device=device)
    y = torch.randint(0, vocab_size, (2, 32), device=device)
 
    logits, loss = model(x, y)
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"loss: {loss.item():.4f}")
 
    loss.backward()
    print("backward pass OK")
 
    out = model.generate(x[:, :4], max_new_tokens=10)
    print(f"generated shape: {tuple(out.shape)}")