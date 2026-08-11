import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module): 
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
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
 
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
 
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 8 / 3):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim)
 
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 512, n_layers: int = 6,
                 n_heads: int = 8, max_seq_len: int = 1024, rope_base: float = 10000.0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.head_dim = dim // n_heads
        self.rope_base = rope_base
 
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads) for _ in range(n_layers)])
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
 
        # weight tying: input embedding and output projection share weights
        self.lm_head.weight = self.tok_emb.weight
 
        self.apply(self._init_weights)
 
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
 
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, return_hidden: bool = False):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        cos, sin = build_rope_cache(T, self.head_dim, base=self.rope_base,
                                     device=idx.device, dtype=self.tok_emb.weight.dtype)

        x = self.tok_emb(idx)
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(block, x, cos, sin, use_reentrant=False)
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