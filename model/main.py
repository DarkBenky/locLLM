import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

print(f"Running {os.path.basename(__file__)}")

import random
import time
import math
import struct
import base64

import requests
import sentencepiece as spm
import torch
import torch.nn.functional as F
import wandb

try:
    import visualtorch
    HAS_VISUALTORCH = True
except ImportError:
    HAS_VISUALTORCH = False

from model import Transformer

# API = "http://localhost:8823"  # for local testing
API = "http://91.98.145.193:8823"

TOKENIZER_MODEL_PATH = "../tok/tokenize/tokenizer_models/tokenizer.model"

BLOCK_SIZE = 4096
BATCH_SIZE = 8
DIM = 1024
N_LAYERS = 26
N_HEADS = 16

RESUME_FROM_CHECKPOINT = True
KEEP_CHECKPOINTS_COUNT = 2 # -1 or 0 to keep all, >0 to keep last N checkpoints, <0 to keep none
RANDOM_SAMPLING = True

MAX_STEPS = 150_000
WARMUP_STEPS = 100
MAX_LR = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

LOG_EVERY = 10
CKPT_EVERY = 250
# CKPT_DIR = "/media/user/sda1/CKPT_DIR"
CKPT_DIR = "./checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sp = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL_PATH)
VOCAB_SIZE = sp.get_piece_size() + 4  # +4 reserved FIM sentinel slots

# FIM sentinel token IDs (reserved, not in SentencePiece vocab — inserted manually during batch construction)
FIM_PRE, FIM_SUF, FIM_MID, FIM_END = sp.get_piece_size(), sp.get_piece_size() + 1, sp.get_piece_size() + 2, sp.get_piece_size() + 3


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


def get_next_samples(count: int) -> list[tuple[int, list[int]]]:
    endpoint = "/api/get-next-samples-random" if RANDOM_SAMPLING else "/api/get-next-samples"
    res = requests.get(API + endpoint, params={"sample_count": count})
    raw_samples = res.json().get("samples", [])
    return [decode_record(base64.b64decode(raw)) for raw in raw_samples]


def get_code_category_ids() -> set[int]:
    """Fetch category index from server and return IDs belonging to code."""
    res = requests.get(API + "/api/get-category-index")
    cat_map = res.json()
    code_names = {
        "Python", "JavaScript", "C++", "Java", "C", "Go", "TypeScript",
        "Ruby", "Rust", "PHP", "Swift", "C#", "Kotlin", "Scala", "Dart",
        "Objective-C", "Perl", "Lua", "SQL", "HTML", "CSS", "JSON",
        "YAML", "Markdown", "XML", "OtherLanguage",
    }
    return {cid for name, cid in cat_map.items() if name in code_names}


CODE_CATEGORY_IDS = set()

_leftover_cache = []
MAX_CACHE_SIZE = 1024


def make_batch(batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    global _leftover_cache

    samples = []
    while _leftover_cache and len(samples) < batch_size:
        samples.append(_leftover_cache.pop())

    need = batch_size - len(samples)
    if need > 0:
        max_retries = 300
        for attempt in range(max_retries):
            fresh = get_next_samples(need)
            if fresh:
                break
            if attempt == 0:
                print("WARNING: no samples returned — cursor may be exhausted, retrying...")
            time.sleep(2)
        else:
            raise RuntimeError(
                f"no samples returned after {max_retries} retries — "
                "check that the data server is running and data exists"
            )
        samples.extend(fresh)

    categories = []
    token_lists = []
    for cat, tokens in samples:
        if len(tokens) > block_size + 1:
            if len(_leftover_cache) < MAX_CACHE_SIZE:
                _leftover_cache.append((cat, tokens[block_size:]))
            else:
                print(f"WARNING: leftover cache full ({MAX_CACHE_SIZE}), discarding tail of sample")
            tokens = tokens[:block_size + 1]
        if len(tokens) < 4:
            continue
        token_lists.append(tokens)
        categories.append(cat)

    if len(token_lists) == 0:
        tok = torch.zeros(1, block_size, dtype=torch.long)
        return tok.to(DEVICE), torch.full_like(tok, -100)

    x = torch.full((len(token_lists), block_size), 0, dtype=torch.long)
    y = torch.full((len(token_lists), block_size), -100, dtype=torch.long)
    for i, (tokens, cat_id) in enumerate(zip(token_lists, categories)):
        seq = torch.tensor(tokens, dtype=torch.long)
        n = min(len(tokens) - 1, block_size)

        is_code = cat_id in CODE_CATEGORY_IDS
        if is_code and random.random() < 0.5 and len(tokens) >= 64:
            pre_end = random.randint(len(tokens) // 4, 7 * len(tokens) // 10)
            mid_max = min(len(tokens) - pre_end - 4, len(tokens) // 4)
            mid_len = random.randint(4, max(5, mid_max))
            mid_end = min(pre_end + mid_len, len(tokens))

            prefix = seq[:pre_end]
            middle = seq[pre_end:mid_end]
            suffix = seq[mid_end:]

            # Assemble: FIM_PRE | prefix | FIM_SUF | suffix | FIM_MID | middle | FIM_END
            fim_seq = torch.cat([
                torch.tensor([FIM_PRE], dtype=torch.long),
                prefix,
                torch.tensor([FIM_SUF], dtype=torch.long),
                suffix,
                torch.tensor([FIM_MID], dtype=torch.long),
                middle,
                torch.tensor([FIM_END], dtype=torch.long),
            ])
            fim_n = min(len(fim_seq) - 1, block_size)
            x[i, :fim_n] = fim_seq[:fim_n]
            mid_pos = len(prefix) + len(suffix) + 2
            if fim_n > mid_pos:
                end = min(fim_n - mid_pos, len(fim_seq) - mid_pos - 1)
                y[i, mid_pos:mid_pos + end] = fim_seq[mid_pos + 1:mid_pos + 1 + end]
        else:
            x[i, :n] = seq[:n]
            y[i, :n] = seq[1:n + 1]

    return x.to(DEVICE), y.to(DEVICE)


def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step > MAX_STEPS:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


if __name__ == "__main__":
    CODE_CATEGORY_IDS = get_code_category_ids()
    print(f"Code categories for FIM: {len(CODE_CATEGORY_IDS)} IDs")

    samples = get_next_samples(1)
    if samples:
        _, tokens = samples[0]
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

    start_step = 0
    os.makedirs(CKPT_DIR, exist_ok=True)
    if RESUME_FROM_CHECKPOINT:
        ckpts = sorted(
            [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt")],
            key=lambda x: int(x.replace("step_", "").replace(".pt", "")),
        )
        if ckpts:
            latest = os.path.join(CKPT_DIR, ckpts[-1])
            print(f"Loading checkpoint: {latest}")
            ckpt = torch.load(latest, map_location=DEVICE)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"] + 1
            print(f"Resuming from step {start_step}")
        else:
            print("No checkpoint found, starting from scratch")

    wandb.login()
    wandb.init(project="locLMM", config={
        "vocab_size": VOCAB_SIZE, "block_size": BLOCK_SIZE, "batch_size": BATCH_SIZE,
        "dim": DIM, "n_layers": N_LAYERS, "n_heads": N_HEADS,
        "max_lr": MAX_LR, "max_steps": MAX_STEPS, "params": n_params,
    })

    if HAS_VISUALTORCH:
        try:
            dummy = torch.zeros(1, BLOCK_SIZE, dtype=torch.long).to(DEVICE)
            graph = visualtorch.flow.flow_view(model, dummy)
            wandb.log({"model_architecture": wandb.Image(graph)})
            print("Logged model architecture diagram to wandb")
        except Exception as e:
            print(f"visualtorch graph failed: {e}")

    model.train()
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    for step in range(start_step, MAX_STEPS):
        t0 = time.time()

        x, y = make_batch(BATCH_SIZE, BLOCK_SIZE)

        if (y == -100).all():
            continue

        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))

        ppl = math.exp(loss.item())
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = y != -100
            acc = (preds[mask] == y[mask]).float().mean().item()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        tok_per_sec = (BATCH_SIZE * BLOCK_SIZE) / dt

        if step % LOG_EVERY == 0:
            print(f"step {step:6d} | loss {loss.item():.4f} | ppl {ppl:.1f} | "
                  f"acc {acc:.3f} | lr {lr:.2e} | "
                  f"grad_norm {grad_norm:.2f} | {tok_per_sec:.0f} tok/s")
            wandb.log({"loss": loss.item(), "ppl": ppl, "acc": acc, "lr": lr,
                        "grad_norm": grad_norm.item(),
                        "tok_per_sec": tok_per_sec}, step=step)

        if step > 0 and step % CKPT_EVERY == 0:
            ckpt_path = f"{CKPT_DIR}/step_{step}.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step}, ckpt_path)
            print(f"saved checkpoint: {ckpt_path}")

            if KEEP_CHECKPOINTS_COUNT == 0 or KEEP_CHECKPOINTS_COUNT == -1:
                pass
            elif KEEP_CHECKPOINTS_COUNT < 0:
                for f in os.listdir(CKPT_DIR):
                    if f.endswith(".pt"):
                        os.remove(os.path.join(CKPT_DIR, f))
                print(f"removed all checkpoints (KEEP_CHECKPOINTS_COUNT={KEEP_CHECKPOINTS_COUNT})")
            else:
                ckpts = sorted(
                    [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt")],
                    key=lambda x: int(x.replace("step_", "").replace(".pt", "")),
                )
                while len(ckpts) > KEEP_CHECKPOINTS_COUNT:
                    old = ckpts.pop(0)
                    os.remove(os.path.join(CKPT_DIR, old))
                    print(f"removed old checkpoint: {old}")