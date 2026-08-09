import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time

import sentencepiece as spm
import torch
import torch.nn.functional as F

from model import Transformer, apply_rope, build_rope_cache
import chatml

BASE = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_MODEL_PATH = os.path.join(BASE, "..", "tok", "tokenize", "tokenizer_models", "tokenizer.model")
CKPT_DIR = os.path.join(BASE, "checkpoints")

DIM = 1024
N_LAYERS = 26
N_HEADS = 16
BLOCK_SIZE = 4096
HEAD_DIM = DIM // N_HEADS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def latest_checkpoint(ckpt_dir=CKPT_DIR):
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not files:
        return None
    def key(f):
        m = re.search(r"\d+", f)
        return (int(m.group()) if m else 0, 0 if "vocab" in f else 1)
    files.sort(key=key)
    return os.path.join(ckpt_dir, files[-1])


def _make_pre(block, k_cache, v_cache):
    def pre(x, cos_p, sin_p, pos):
        h = block.attn_norm(x)
        qkv = block.attn.qkv_proj(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(1, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(1, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(1, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
        q = apply_rope(q, cos_p, sin_p)
        k = apply_rope(k, cos_p, sin_p)
        k_cache.index_copy_(2, pos, k)
        v_cache.index_copy_(2, pos, v)
        return q, k, v
    return pre


def _make_post_pre(block_a, block_b, k_cache_b, v_cache_b):
    def f(x, attn_out, cos_p, sin_p, pos):
        out = attn_out.reshape(1, 1, DIM)
        x = x + block_a.attn.out_proj(out)
        x = x + block_a.ffn(block_a.ffn_norm(x))
        h = block_b.attn_norm(x)
        qkv = block_b.attn.qkv_proj(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(1, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(1, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(1, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
        q = apply_rope(q, cos_p, sin_p)
        k = apply_rope(k, cos_p, sin_p)
        k_cache_b.index_copy_(2, pos, k)
        v_cache_b.index_copy_(2, pos, v)
        return x, q, k, v
    return f


def _make_final(block, lm_head, final_norm):
    def f(x, attn_out):
        out = attn_out.reshape(1, 1, DIM)
        x = x + block.attn.out_proj(out)
        x = x + block.ffn(block.ffn_norm(x))
        return lm_head(final_norm(x))
    return f


class InferenceEngine:
    def __init__(self, ckpt_path=None, device=DEVICE, dtype=DTYPE):
        ckpt_path = ckpt_path or latest_checkpoint()
        if ckpt_path is None:
            raise FileNotFoundError("no checkpoint found in " + CKPT_DIR)
        self.device = device
        self.dtype = dtype
        self.sp = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL_PATH)
        base = self.sp.get_piece_size()
        self.fim_pre, self.fim_suf, self.fim_mid, self.fim_end = base, base + 1, base + 2, base + 3
        state = torch.load(ckpt_path, map_location="cpu", mmap=True)["model"]
        self.vocab_size = state["tok_emb.weight"].shape[0]
        self.n_layers = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("blocks."))
        self.model = Transformer(
            vocab_size=self.vocab_size, dim=DIM, n_layers=self.n_layers,
            n_heads=N_HEADS, max_seq_len=BLOCK_SIZE,
        )
        self.model.load_state_dict(state)
        self.model.to(device=device, dtype=dtype)
        self.model.eval()
        self.k_cache = torch.zeros(self.n_layers, 1, N_HEADS, BLOCK_SIZE, HEAD_DIM, device=device, dtype=dtype)
        self.v_cache = torch.zeros(self.n_layers, 1, N_HEADS, BLOCK_SIZE, HEAD_DIM, device=device, dtype=dtype)
        self.cos, self.sin = build_rope_cache(BLOCK_SIZE, HEAD_DIM, device=device, dtype=dtype)
        self.pre0 = torch.compile(_make_pre(self.model.blocks[0], self.k_cache[0], self.v_cache[0]))
        self.post_pre = [
            torch.compile(_make_post_pre(self.model.blocks[i], self.model.blocks[i + 1],
                                         self.k_cache[i + 1], self.v_cache[i + 1]))
            for i in range(self.n_layers - 1)
        ]
        self.final = torch.compile(_make_final(self.model.blocks[-1], self.model.lm_head, self.model.final_norm))
        list(self.generate(self.sp.encode("def f():\n    pass"), max_new_tokens=1, temperature=0.0))

    def _prefill(self, tokens):
        T = tokens.shape[1]
        x = self.model.tok_emb(tokens)
        for i, block in enumerate(self.model.blocks):
            h = block.attn_norm(x)
            qkv = block.attn.qkv_proj(h)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(1, T, N_HEADS, HEAD_DIM).transpose(1, 2)
            k = k.view(1, T, N_HEADS, HEAD_DIM).transpose(1, 2)
            v = v.view(1, T, N_HEADS, HEAD_DIM).transpose(1, 2)
            q = apply_rope(q, self.cos[:T], self.sin[:T])
            k = apply_rope(k, self.cos[:T], self.sin[:T])
            self.k_cache[i][:, :, :T] = k
            self.v_cache[i][:, :, :T] = v
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).contiguous().view(1, T, DIM)
            x = x + block.attn.out_proj(out)
            x = x + block.ffn(block.ffn_norm(x))
        x = self.model.final_norm(x)
        return self.model.lm_head(x)

    def _decode_step(self, token, pos):
        x = self.model.tok_emb(token)
        cos_p = self.cos[pos:pos + 1]
        sin_p = self.sin[pos:pos + 1]
        pos_t = torch.tensor([pos], dtype=torch.long, device=self.device)
        q, k, v = self.pre0(x, cos_p, sin_p, pos_t)
        out = F.scaled_dot_product_attention(
            q, self.k_cache[0][:, :, :pos + 1], self.v_cache[0][:, :, :pos + 1])
        for i in range(N_LAYERS - 1):
            x, q, k, v = self.post_pre[i](x, out, cos_p, sin_p, pos_t)
            out = F.scaled_dot_product_attention(
                q, self.k_cache[i + 1][:, :, :pos + 1], self.v_cache[i + 1][:, :, :pos + 1])
        return self.final(x, out)

    @staticmethod
    def _sample(logits, temperature, top_k, top_p, gen):
        logits = logits[:, -1, :]
        if temperature <= 0:
            return logits.argmax().item()
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        if top_k > 0:
            vals, idx = torch.topk(probs, min(top_k, probs.size(-1)))
            filt = torch.full_like(probs, float("-inf"))
            filt.scatter_(1, idx, vals)
            probs = filt
        if top_p < 1.0:
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_p, dim=-1)
            keep = cum - sorted_p <= top_p
            sorted_p = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
            probs = torch.zeros_like(probs).scatter_(1, sorted_i, sorted_p)
        total = probs.sum(dim=-1, keepdim=True)
        if total.item() <= 0:
            return logits.argmax().item()
        probs = probs / total
        return torch.multinomial(probs, num_samples=1, generator=gen).item()

    def _decode_loop(self, last, pos, max_new_tokens, temperature, top_k, top_p, gen):
        for _ in range(max_new_tokens):
            logits = self._decode_step(last.unsqueeze(0), pos)
            tok = self._sample(logits, temperature, top_k, top_p, gen)
            yield tok
            last = torch.tensor([tok], dtype=torch.long, device=self.device)
            pos += 1

    def decode(self, tokens):
        return chatml.safe_decode(
            tokens, self.sp, chatml.reserved_ids(self.sp.get_piece_size()),
            extra={self.fim_pre: "", self.fim_suf: "",
                   self.fim_mid: "", self.fim_end: ""})

    def generate(self, prompt_ids, max_new_tokens=256, temperature=1.0, top_k=0, top_p=1.0, seed=None):
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed)
        prompt = torch.as_tensor(prompt_ids, dtype=torch.long, device=self.device)
        if prompt.numel() == 0:
            return
        limit = max(1, BLOCK_SIZE - max_new_tokens)
        prompt = prompt[-limit:]
        with torch.inference_mode():
            self._prefill(prompt.unsqueeze(0))
            last = prompt[-1:].clone()
            pos = prompt.numel() - 1
            yield from self._decode_loop(last, pos, max_new_tokens, temperature, top_k, top_p, gen)

    def generate_fim(self, prefix_ids, suffix_ids, max_new_tokens=256, temperature=1.0,
                     top_k=0, top_p=1.0, seed=None):
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed)
        pre = torch.tensor([self.fim_pre], dtype=torch.long, device=self.device)
        suf = torch.tensor([self.fim_suf], dtype=torch.long, device=self.device)
        mid = torch.tensor([self.fim_mid], dtype=torch.long, device=self.device)
        prefix = torch.as_tensor(prefix_ids, dtype=torch.long, device=self.device)
        suffix = torch.as_tensor(suffix_ids, dtype=torch.long, device=self.device)
        prompt = torch.cat([pre, prefix, suf, suffix, mid])
        limit = max(1, BLOCK_SIZE - max_new_tokens)
        prompt = prompt[-limit:]
        with torch.inference_mode():
            self._prefill(prompt.unsqueeze(0))
            last = prompt[-1:].clone()
            pos = prompt.numel() - 1
            for _ in range(max_new_tokens):
                logits = self._decode_step(last.unsqueeze(0), pos)
                tok = self._sample(logits, temperature, top_k, top_p, gen)
                if tok == self.fim_end:
                    break
                yield tok
                last = torch.tensor([tok], dtype=torch.long, device=self.device)
                pos += 1


if __name__ == "__main__":
    engine = InferenceEngine()
    print(f"params: {sum(p.numel() for p in engine.model.parameters()) / 1e6:.1f}M")
    text = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return "
    ids = engine.sp.encode(text)
    t0 = time.time()
    tokens = list(engine.generate(ids, max_new_tokens=64, temperature=0.8, top_k=40))
    dt = time.time() - t0
    print(engine.decode(tokens))
    print(f"{len(tokens)} tokens in {dt:.2f}s = {len(tokens) / dt:.0f} tok/s")
    t0 = time.time()
    tokens = list(engine.generate(ids, max_new_tokens=64, temperature=0.0))
    dt = time.time() - t0
    print(engine.decode(tokens))
    print(f"{len(tokens)} tokens in {dt:.2f}s = {len(tokens) / dt:.0f} tok/s")
    pre_ids = engine.sp.encode("def add(a, b):\n    return ")
    suf_ids = engine.sp.encode("print(add(1, 2))")
    tokens = list(engine.generate_fim(pre_ids, suf_ids, max_new_tokens=32, temperature=0.8, top_k=40))
    print(engine.decode(tokens))
