"""Phase 0 diagnostics (FIX.md): per-block contribution analysis.

Run:  python diag_blocks.py <checkpoint.pt> [n_samples] [max_tokens]
Prints per block: angular distance (input vs output hidden), attn/ffn branch
norm ratios, sorted by angular distance. Gates the Phase 8 depth-pruning decision.
"""
from __future__ import annotations
import base64
import struct
import sys

import requests
import torch

from model import Transformer, build_rope_cache

TOKENIZER = "../tok/tokenize/tokenizer_models/tokenizer.model"
API = "http://91.98.145.193:8823"


def get_samples(n):
    res = requests.get(API + "/api/get-next-samples-random", params={"count": n}, timeout=30)
    out = []
    for raw in res.json().get("samples", []):
        data = base64.b64decode(raw)
        size = struct.unpack_from("<Q", data, 0)[0]
        ntok = (size - 1) // 2
        out.append(list(struct.unpack_from(f"<{ntok}H", data, 9)))
    return out


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints/step_big_fim_211500.pt"
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    max_tok = int(sys.argv[3]) if len(sys.argv) > 3 else 2048

    state = torch.load(ckpt_path, map_location="cpu", mmap=True)["model"]
    vocab = state["tok_emb.weight"].shape[0]
    n_layers = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("blocks."))
    ffn_hidden = state["blocks.0.ffn.w_gate.weight"].shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(vocab_size=vocab, dim=1024, n_layers=n_layers, n_heads=16,
                        max_seq_len=max_tok, ffn_hidden=ffn_hidden).to(device)
    model.load_state_dict(state, strict=False)  # old ckpts predate LayerScale
    model.eval()
    print(f"loaded {ckpt_path}: layers={n_layers} ffn={ffn_hidden} vocab={vocab}")

    samples = []
    tries = 0
    while len(samples) < n_samples and tries < 8:
        tries += 1
        try:
            samples.extend(get_samples(n_samples - len(samples)))
        except Exception as e:
            print(f"fetch failed: {e}")
    if not samples:
        print("no samples from server — aborting")
        return

    cos_acc = torch.zeros(n_layers, dtype=torch.float64)
    attn_ratio = torch.zeros(n_layers, dtype=torch.float64)
    ffn_ratio = torch.zeros(n_layers, dtype=torch.float64)

    with torch.no_grad():
        for si, tokens in enumerate(samples):
            x = torch.tensor([tokens[:max_tok]], device=device)
            T = x.shape[1]
            cos, sin = build_rope_cache(T, model.head_dim, base=model.rope_base,
                                        device=device, dtype=model.tok_emb.weight.dtype)
            h = model.tok_emb(x)
            for i, block in enumerate(model.blocks):
                a = block.attn(block.attn_norm(h), cos, sin)
                h_mid = h + a
                f = block.ffn(block.ffn_norm(h_mid))
                h_out = h_mid + f
                hf, ho = h.float(), h_out.float()
                dot = (hf * ho).sum(dim=-1)
                cosim = (dot / (hf.norm(dim=-1) * ho.norm(dim=-1) + 1e-8)).mean().item()
                cos_acc[i] += 1.0 - cosim
                attn_ratio[i] += (a.float().norm(dim=-1).mean() / (h.float().norm(dim=-1).mean() + 1e-8)).item()
                ffn_ratio[i] += (f.float().norm(dim=-1).mean() / (h_mid.float().norm(dim=-1).mean() + 1e-8)).item()
                h = h_out
            if (si + 1) % 16 == 0:
                print(f"  {si + 1}/{len(samples)} samples ...")

    n = max(len(samples), 1)
    cos_acc /= n
    attn_ratio /= n
    ffn_ratio /= n
    order = torch.argsort(cos_acc, descending=True)
    print("\nblock | 1-cosim (in->out) | attn ratio | ffn ratio")
    for idx in order.tolist():
        print(f"{idx:5d} | {cos_acc[idx].item():.5f} | {attn_ratio[idx].item():.4f} | {ffn_ratio[idx].item():.4f}")
    print(f"\nblocks with (1-cosim) < 0.05 (near-identity): "
          f"{int((cos_acc < 0.05).sum().item())}/{n_layers}")

    import json
    out_path = ckpt_path.rsplit(".", 1)[0] + "_diag.json"
    with open(out_path, "w") as f:
        json.dump({
            "ckpt": ckpt_path, "n_samples": len(samples), "max_tok": max_tok,
            "n_layers": n_layers, "ffn_hidden": ffn_hidden,
            "order": order.tolist(),
            "one_minus_cosim": [round(v, 6) for v in cos_acc.tolist()],
            "attn_ratio": [round(v, 5) for v in attn_ratio.tolist()],
            "ffn_ratio": [round(v, 5) for v in ffn_ratio.tolist()],
        }, f, indent=1)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
