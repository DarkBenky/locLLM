import os
import time
import math
import struct
import base64

import requests
import sentencepiece as spm
import torch
import torch.nn.functional as F
import wandb

from model import Transformer

API = "http://localhost:8823"
TOKENIZER_MODEL_PATH = "../tok/tokenize/tokenizer_models/tokenizer.model"

VOCAB_SIZE = 32000
BLOCK_SIZE = 512
BATCH_SIZE = 32
DIM = 512
N_LAYERS = 8
N_HEADS = 8

MAX_STEPS = 20_000
WARMUP_STEPS = 500
MAX_LR = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

LOG_EVERY = 10
CKPT_EVERY = 1000
CKPT_DIR = "checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sp = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL_PATH)
PAD_ID = sp.pad_id() if sp.pad_id() >= 0 else 0


def decode_record(data: bytes) -> tuple[int, list[int]]:
    if len(data) < 8:
        raise ValueError(f"record too short: {len(data)} bytes")
    record_size = struct.unpack_from("<Q", data, 0)[0]
    category = data[8]
    token_count = (record_size - 1) // 2
    tokens = []
    offset = 9
    for _ in range(token_count):
        tokens.append(struct.unpack_from("<H", data, offset)[0])
        offset += 2
    return category, tokens


def get_next_samples(count: int) -> list[list[int]]:
    res = requests.get(API + "/api/get-next-samples", params={"sample_count": count})
    raw_samples = res.json().get("samples", [])
    return [decode_record(base64.b64decode(raw))[1] for raw in raw_samples]


def make_batch(batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    token_lists = get_next_samples(batch_size)
    if not token_lists:
        raise RuntimeError("no samples returned — check the data server / cursor position")

    padded = torch.full((len(token_lists), block_size + 1), PAD_ID, dtype=torch.long)
    for i, tokens in enumerate(token_lists):
        tokens = tokens[: block_size + 1]
        if tokens:
            padded[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)

    x = padded[:, :-1].to(DEVICE)
    y = padded[:, 1:].to(DEVICE)
    return x, y


def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step > MAX_STEPS:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


if __name__ == "__main__":
    tokens = get_next_samples(1)[0]
    print(tokens[:20])
    print(sp.decode(tokens)[:200])

    model = Transformer(vocab_size=VOCAB_SIZE, dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS,
                         max_seq_len=BLOCK_SIZE).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {DEVICE} | params: {n_params / 1e6:.1f}M")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)

    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=MAX_LR, betas=(0.9, 0.95),
    )

    os.makedirs(CKPT_DIR, exist_ok=True)

    wandb.login()
    wandb.init(project="locLMM", entity="locLMM", config={
        "vocab_size": VOCAB_SIZE, "block_size": BLOCK_SIZE, "batch_size": BATCH_SIZE,
        "dim": DIM, "n_layers": N_LAYERS, "n_heads": N_HEADS,
        "max_lr": MAX_LR, "max_steps": MAX_STEPS, "params": n_params,
    })

    model.train()
    for step in range(MAX_STEPS):
        t0 = time.time()

        x, y = make_batch(BATCH_SIZE, BLOCK_SIZE)

        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), ignore_index=PAD_ID)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        tok_per_sec = (BATCH_SIZE * BLOCK_SIZE) / dt

        if step % LOG_EVERY == 0:
            print(f"step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                  f"grad_norm {grad_norm:.2f} | {tok_per_sec:.0f} tok/s")
            wandb.log({"loss": loss.item(), "lr": lr, "grad_norm": grad_norm.item(),
                        "tok_per_sec": tok_per_sec}, step=step)

        if step > 0 and step % CKPT_EVERY == 0:
            ckpt_path = f"{CKPT_DIR}/step_{step}.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step}, ckpt_path)
            print(f"saved checkpoint: {ckpt_path}")