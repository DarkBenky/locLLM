import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sentencepiece as spm
import torch

from model import Transformer
import chatml

TOKENIZER_MODEL_PATH = "../tok/tokenize/tokenizer_models/tokenizer.model"


def resize_state(old_sd, old_vocab, new_vocab, dim):
    new_sd = {}
    for k, v in old_sd.items():
        if k in ("tok_emb.weight", "lm_head.weight"):
            w = torch.zeros(new_vocab, dim, dtype=v.dtype)
            w[:old_vocab] = v
            w[old_vocab:] = v.mean(dim=0, keepdim=True)
            new_sd[k] = w
        else:
            new_sd[k] = v
    return new_sd


def resize_optimizer(old_opt_sd, emb_idx, old_vocab, new_vocab):
    sd = copy.deepcopy(old_opt_sd)
    for key in ("exp_avg", "exp_avg_sq"):
        t = sd["state"][emb_idx][key]
        padded = torch.zeros(new_vocab, *t.shape[1:], dtype=t.dtype)
        padded[:old_vocab] = t
        sd["state"][emb_idx][key] = padded
    return sd


def main():
    ap = argparse.ArgumentParser(description="Resize a locLLM checkpoint to the current reserved-token vocab (FIM + ChatML + context + lang)")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=8192)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL_PATH)
    new_vocab = sp.get_piece_size() + chatml.reserved_total()

    ckpt = torch.load(args.src, map_location="cpu", mmap=True)
    old_sd = ckpt["model"]
    old_vocab = old_sd["tok_emb.weight"].shape[0]
    if old_vocab >= new_vocab:
        print(f"no resize needed: {old_vocab} >= {new_vocab}")
        return

    new_sd = resize_state(old_sd, old_vocab, new_vocab, args.dim)
    out = {"model": new_sd, "step": ckpt.get("step", 0)}

    if "optimizer" in ckpt:
        model = Transformer(vocab_size=new_vocab, dim=args.dim, n_layers=args.n_layers,
                            n_heads=args.n_heads, max_seq_len=args.block_size)
        model.load_state_dict(new_sd, strict=False)
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            (no_decay if p.ndim < 2 else decay).append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.1},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=1e-4, betas=(0.9, 0.95))
        grouped = []
        for g in opt.param_groups:
            grouped.extend(g["params"])
        emb_idx = next(i for i, p in enumerate(grouped) if p is model.tok_emb.weight)
        out["optimizer"] = resize_optimizer(ckpt["optimizer"], emb_idx, old_vocab, new_vocab)

    torch.save(out, args.dst)
    print(f"resized {args.src} (vocab {old_vocab}) -> {args.dst} (vocab {new_vocab})")


if __name__ == "__main__":
    main()
