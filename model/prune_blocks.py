"""Phase 8 (FIX.md 36): depth pruning — drop the dead top blocks of the stack.

The Phase 0 diagnostic (diag_blocks.py) measured blocks 52..127 of the 128-layer
model as near-exact identity (attn/ffn branch outputs ~0.002 of the residual).
Pruning them keeps the loss essentially unchanged at step 0 and halves the
params. The freed VRAM goes back into bigger batches (the FFN stays 6.5x wide).

Run:  python prune_blocks.py <src.pt> <dst.pt> [--keep 52]
Saves a checkpoint for the NEW architecture (untied lm_head + LayerScale) with
the optimizer moments carried over for every surviving parameter.
"""
import argparse
import copy
import sys

import torch

sys.path.insert(0, ".")
import main_big as mb  # noqa: E402  (for _opt_state_is_8bit etc.)
from model import Transformer  # noqa: E402

DIM = 1024
N_HEADS = 16


def build_new_model(old_sd, keep):
    vocab = old_sd["tok_emb.weight"].shape[0]
    ffn = old_sd["blocks.0.ffn.w_gate.weight"].shape[0]
    m = Transformer(vocab_size=vocab, dim=DIM, n_layers=keep, n_heads=N_HEADS,
                    max_seq_len=8192, ffn_hidden=ffn)
    new_sd = {}
    for k, v in old_sd.items():
        if k.startswith("blocks."):
            idx = int(k.split(".")[1])
            if idx >= keep:
                continue
        new_sd[k] = v
    new_sd.setdefault("lm_head.weight", old_sd["tok_emb.weight"])
    m.load_state_dict(new_sd, strict=False)  # ls params keep init 1.0
    return m, ffn


def build_new_optimizer_state(old_sd, model, tied_emb, keep):
    """Old layout: decay = tok_emb(0) + 5 params/block x128 (no lm_head); the
    no-decay group (base 641) = 2 norms/block x128 + final_norm.
    New layout: decay = tok_emb + 5/block x keep + lm_head; no-decay = 4/block
    (attn_norm, ls_attn, ffn_norm, ls_ffn) x keep + final_norm."""
    old_state = old_sd["state"]
    n_decay = 1 + 5 * keep + 1
    n_no_decay = 4 * keep + 1
    new_state = {}
    if 0 in old_state:
        new_state[0] = old_state[0]                      # tok_emb
    for b in range(keep):
        for k in range(5):
            old_i = 1 + 5 * b + k
            if old_i in old_state:
                new_state[old_i] = old_state[old_i]      # block params keep idx
    if tied_emb and 0 in old_state:
        new_state[n_decay - 1] = copy.deepcopy(old_state[0])  # lm_head <- tok_emb
    base = n_decay
    old_base = 641
    for b in range(keep):
        for k in range(2):  # attn_norm, ffn_norm
            old_i = old_base + 2 * b + k
            new_i = base + 4 * b + (0 if k == 0 else 2)
            if old_i in old_state:
                new_state[new_i] = old_state[old_i]
    if old_base + 256 in old_state:
        new_state[base + 4 * keep] = old_state[old_base + 256]  # final_norm
    sd = dict(old_sd)
    sd["state"] = new_state
    pg0 = dict(old_sd["param_groups"][0])
    pg0["params"] = list(range(n_decay))
    if len(old_sd["param_groups"]) > 1:
        pg1 = dict(old_sd["param_groups"][1])
        pg1["params"] = list(range(n_decay, n_decay + n_no_decay))
        sd["param_groups"] = [pg0, pg1]
    else:
        sd["param_groups"] = [pg0]
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--keep", type=int, default=52)
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", mmap=True)
    old_sd = ckpt["model"]
    n_old_layers = 1 + max(int(k.split(".")[1]) for k in old_sd if k.startswith("blocks."))
    keep = min(args.keep, n_old_layers)
    print(f"pruning {n_old_layers} -> {keep} layers")
    tied_emb = torch.equal(old_sd["tok_emb.weight"], old_sd.get("lm_head.weight", old_sd["tok_emb.weight"]))

    model, ffn = build_new_model(old_sd, keep)
    model.eval()
    print(f"new model: {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B params, ffn={ffn}")

    out = {"model": model.state_dict(), "step": ckpt.get("step", 0), "pruned_from": n_old_layers}

    if "optimizer" in ckpt and mb._opt_state_is_8bit(ckpt["optimizer"]):
        import bitsandbytes as bnb
        decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
        no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
        opt = bnb.optim.AdamW8bit(
            [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
            lr=1e-4, betas=(0.9, 0.98))
        new_opt_sd = build_new_optimizer_state(ckpt["optimizer"], model, tied_emb, keep)
        opt.load_state_dict(new_opt_sd)  # round-trip check inside the tool
        print("optimizer state spliced and loadable")
        out["optimizer"] = new_opt_sd
    else:
        print("no usable 8-bit optimizer state in source — optimizer omitted")

    torch.save(out, args.dst)
    print(f"saved: {args.dst}")


if __name__ == "__main__":
    main()
