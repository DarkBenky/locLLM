"""Net2Wider dim expansion (FIX.md 37, function-preserving): 1024 -> 1536.

Zero-padding the new columns/rows keeps step-0 output bitwise identical:
  - tok_emb (V, old) -> (V, new): zero columns
  - lm_head (V, old) -> (V, new): zero columns (nn.Linear weight is (out, in))
  - qkv/out/gate/up/down weights: top-left copy
  - norms + LayerScale: ones
The new capacity starts learning from step 1. Optimizer state is omitted
(restarts fresh — the wake-up heal covers it, same precedent as FFN widen).

Run:  python widen_dim.py <src.pt> <dst.pt> [--dim 1536]
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")


def widen_state(old_sd, new_dim, new_ffn=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    new_sd = {}
    for k, v in old_sd.items():
        if k.startswith("blocks."):
            parts = k.split(".")
            if parts[-1].startswith("ls_"):
                new_sd[k] = torch.ones(new_dim, dtype=v.dtype)
                continue
            if parts[-2] in ("attn_norm", "ffn_norm"):
                # RMSNorm is not invariant to zero-padding: rms scales by
                # sqrt(old/new), so old weights scale by sqrt(old/new) to keep
                # old-dim outputs unchanged. New dims start at 1.0.
                s = (v.shape[0] / new_dim) ** 0.5
                w = torch.ones(new_dim, dtype=v.dtype)
                w[: v.shape[0]] = v.to(torch.float32) * s
                new_sd[k] = w
                continue
            if v.ndim != 2:
                continue
            kind = parts[-1]
            if kind == "weight":
                layer = parts[-2]
                if layer == "qkv_proj":
                    # row layout is [q(dim), k(dim), v(dim)] — with a larger dim
                    # each block moves, so map old q/k/v rows separately
                    in_d = v.shape[1]
                    w = torch.zeros(3 * new_dim, new_dim, dtype=v.dtype)
                    w[:in_d, :in_d] = v[:in_d]                       # q
                    w[new_dim:new_dim + in_d, :in_d] = v[in_d:2 * in_d]   # k
                    w[2 * new_dim:2 * new_dim + in_d, :in_d] = v[2 * in_d:]  # v
                elif layer == "out_proj":
                    w = torch.zeros(new_dim, new_dim, dtype=v.dtype)
                    w[: v.shape[0], : v.shape[1]] = v
                elif layer in ("w_gate", "w_up"):
                    old_ffn = v.shape[0]
                    w = torch.zeros(new_ffn, new_dim, dtype=v.dtype)
                    w[:old_ffn, : v.shape[1]] = v
                    # fresh-init the new rows: w_down's new columns are zero, so
                    # step-0 output is unchanged, but gradients flow immediately
                    w[old_ffn:] = torch.randn(w[old_ffn:].shape, generator=g,
                                              dtype=v.dtype) * 0.02
                elif layer == "w_down":
                    old_ffn = v.shape[1]
                    w = torch.zeros(new_dim, new_ffn, dtype=v.dtype)
                    w[: v.shape[0], :old_ffn] = v  # new cols + new rows stay 0
                else:
                    raise ValueError(f"unknown weight {k}")
                new_sd[k] = w
        elif k == "tok_emb.weight":
            w = torch.zeros(v.shape[0], new_dim, dtype=v.dtype)
            w[:, : v.shape[1]] = v
            new_sd[k] = w
        elif k == "lm_head.weight":
            # nn.Linear weight is (out=vocab, in=dim) — pad the input columns
            w = torch.zeros(v.shape[0], new_dim, dtype=v.dtype)
            w[:, : v.shape[1]] = v
            new_sd[k] = w
        elif k.endswith("norm.weight"):  # final_norm
            s = (v.shape[0] / new_dim) ** 0.5
            w = torch.ones(new_dim, dtype=v.dtype)
            w[: v.shape[0]] = v.to(torch.float32) * s
            new_sd[k] = w
        elif k.endswith("weight"):
            new_sd[k] = torch.ones(new_dim, dtype=v.dtype)
        else:
            new_sd[k] = v
    return new_sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--dim", type=int, default=1536)
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", mmap=True)
    old_sd = ckpt["model"]
    old_dim = old_sd["tok_emb.weight"].shape[1]
    old_ffn = old_sd["blocks.0.ffn.w_gate.weight"].shape[0]
    old_heads = old_sd["blocks.0.attn.qkv_proj.weight"].shape[0] // 3 // 64
    new_heads = args.dim // 64  # keep head_dim 64: extra heads are zero-init
    new_ffn = round(args.dim * old_ffn / old_dim)
    print(f"widening dim {old_dim} -> {args.dim}, heads {old_heads} -> {new_heads}, "
          f"ffn {old_ffn} -> {new_ffn} ({new_ffn / args.dim:.1f}x)")
    new_sd = widen_state(old_sd, args.dim, new_ffn)
    new_sd["__meta_n_heads__"] = torch.tensor(new_heads)  # ignored by strict=False loads
    # store weights in bf16 (training dtype): halves the checkpoint file and
    # matches what main_big casts to at load anyway
    for k, v in list(new_sd.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            new_sd[k] = v.to(torch.bfloat16)
    out = {"model": new_sd, "step": ckpt.get("step", 0),
           "pruned_from": ckpt.get("pruned_from"), "widened_from": old_dim}
    torch.save(out, args.dst)
    n_params = sum(v.numel() for v in new_sd.values() if isinstance(v, torch.Tensor))
    print(f"new params: {n_params / 1e9:.3f}B (lm_head+tok_emb counted separately)")
    print(f"saved: {args.dst} (no optimizer state — starts fresh, wake-up heal covers it)")


if __name__ == "__main__":
    main()
