from __future__ import annotations
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

print(f"Running {os.path.basename(__file__)}")

import random
import re
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
from checkpoint_sample import record_raw_tokens, run_checkpoint_sample

API = "http://91.98.145.193:8823"

TOKENIZER_MODEL_PATH = "../tok/tokenize/tokenizer_models/tokenizer.model"

BLOCK_SIZE = 4096
BATCH_SIZE = 6
GRAD_ACCUM = 3
DIM = 1024
N_LAYERS = 128
OLD_N_LAYERS = 26
N_HEADS = 16

RESUME_FROM_CHECKPOINT = True
UPSCALE_ON_RESUME = True
KEEP_CHECKPOINTS_COUNT = 1
RANDOM_SAMPLING = True

MAX_STEPS = 500_000
WARMUP_STEPS = 100
WAKEUP_STEPS = 3000
WAKEUP_LR = 1e-4
MAX_LR = 5e-5
MIN_LR = 1e-5
LR_DECAY_STEPS = 250_000
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

LOG_EVERY = 10
CKPT_EVERY = 250
EVAL_EVERY = 250
EVAL_SAMPLES = 8
CKPT_DIR = "./checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRECISION = "bf16"
MODEL_DTYPE = torch.bfloat16 if PRECISION == "bf16" else torch.float32

if DEVICE == "cuda" and PRECISION == "bf16":
    AUTOCAST_DTYPE = torch.bfloat16
    USE_SCALER = False
elif DEVICE == "cuda":
    AUTOCAST_DTYPE = torch.float16
    USE_SCALER = True
else:
    AUTOCAST_DTYPE = torch.float32
    USE_SCALER = False

sp = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL_PATH)
VOCAB_SIZE = sp.get_piece_size() + 4

FIM_PRE, FIM_SUF, FIM_MID, FIM_END = sp.get_piece_size(), sp.get_piece_size() + 1, sp.get_piece_size() + 2, sp.get_piece_size() + 3

UPSCALED = False
CKPT_PREFIX = "step_big_"


def _step_from_name(f):
    return int(re.search(r"\d+", f).group())


def upscale_into(model, old_sd, old_n_layers):
    big_sd = {}
    for k, v in old_sd.items():
        if k.startswith("blocks."):
            parts = k.split(".")
            idx = int(parts[1])
            big_sd[f"blocks.{2 * idx}." + ".".join(parts[2:])] = v
        else:
            big_sd[k] = v
    missing, unexpected = model.load_state_dict(big_sd, strict=False)
    old_slots = 2 * old_n_layers
    for i in range(N_LAYERS):
        if i % 2 == 1 or i >= old_slots:
            blk = model.blocks[i]
            blk.attn.out_proj.weight.data.zero_()
            blk.ffn.w_down.weight.data.zero_()
    return missing, unexpected


def decode_record(data: bytes) -> tuple[int, list[int]]:
    if len(data) < 8:
        raise ValueError(f"record too short: {len(data)} bytes")
    record_size = struct.unpack_from("<Q", data, 0)[0]
    category = data[8]
    token_count = (record_size - 1) // 2
    if 9 + 2 * token_count > len(data):
        raise ValueError(f"record size mismatch: header={record_size} data={len(data)}")
    tokens = []
    offset = 9
    for _ in range(token_count):
        tokens.append(struct.unpack_from("<H", data, offset)[0])
        offset += 2
    return category, tokens


REQUEST_TIMEOUT = 30
MAX_REQUEST_RETRIES = 5
_session = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=4, max_retries=0,
        )
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
    return _session


def _request_with_retry(url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
    last_exc = None
    for attempt in range(MAX_REQUEST_RETRIES):
        try:
            return _get_session().get(url, **kwargs)
        except requests.exceptions.Timeout:
            last_exc = f"timeout after {kwargs['timeout']}s"
        except requests.exceptions.ConnectionError as e:
            last_exc = str(e)
        except requests.exceptions.RequestException as e:
            last_exc = str(e)
        if attempt < MAX_REQUEST_RETRIES - 1:
            delay = 2 ** attempt
            print(f"  request failed ({last_exc}), retrying in {delay}s (attempt {attempt + 2}/{MAX_REQUEST_RETRIES})...")
            time.sleep(delay)
    raise RuntimeError(f"request failed after {MAX_REQUEST_RETRIES} retries: {last_exc}")


def get_next_samples(count: int) -> list[tuple[int, list[int]]]:
    endpoint = "/api/get-next-samples-random" if RANDOM_SAMPLING else "/api/get-next-samples"
    res = _request_with_retry(API + endpoint, params={"sample_count": count})
    raw_samples = res.json().get("samples", [])
    return [decode_record(base64.b64decode(raw)) for raw in raw_samples]


def get_code_category_ids() -> set[int]:
    res = _request_with_retry(API + "/api/get-category-index")
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
            try:
                fresh = get_next_samples(need)
            except RuntimeError as e:
                if attempt == 0:
                    print(f"WARNING: get_next_samples failed: {e}")
                if attempt < max_retries - 1:
                    delay = min(60, 2 + 2 ** min(attempt, 6))
                    time.sleep(delay)
                    continue
                raise
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
        record_raw_tokens(tokens)
        token_lists.append(tokens)
        categories.append(cat)

    if len(token_lists) == 0:
        tok = torch.zeros(1, block_size, dtype=torch.long)
        return tok.to(DEVICE), torch.full_like(tok, -100), [], []

    x = torch.full((len(token_lists), block_size), 0, dtype=torch.long)
    y = torch.full((len(token_lists), block_size), -100, dtype=torch.long)
    fim_flags = []
    for i, (tokens, cat_id) in enumerate(zip(token_lists, categories)):
        is_code = cat_id in CODE_CATEGORY_IDS
        do_fim = is_code and random.random() < 0.5 and len(tokens) >= 64

        if do_fim:
            fim_cap = block_size - 3
            if len(tokens) > fim_cap:
                tail = tokens[fim_cap:]
                if len(tail) >= 16:
                    if len(_leftover_cache) < MAX_CACHE_SIZE:
                        _leftover_cache.append((cat_id, tail))
                    else:
                        print(f"WARNING: leftover cache full ({MAX_CACHE_SIZE}), discarding tail of sample")
                tokens = tokens[:fim_cap]
            seq = torch.tensor(tokens, dtype=torch.long)
            pre_end = random.randint(len(tokens) // 4, 7 * len(tokens) // 10)
            mid_max = min(len(tokens) - pre_end - 4, len(tokens) // 4)
            if mid_max < 16:
                do_fim = False

        if not do_fim:
            seq = torch.tensor(tokens, dtype=torch.long)
            n = min(len(tokens) - 1, block_size)
            x[i, :n] = seq[:n]
            y[i, :n] = seq[1:n + 1]
            fim_flags.append(False)
            continue

        mid_len = random.randint(min(64, mid_max), mid_max)
        mid_end = min(pre_end + mid_len, len(tokens) - 4)

        prefix = seq[:pre_end]
        middle = seq[pre_end:mid_end]
        suffix = seq[mid_end:]

        fim_seq = torch.cat([
            torch.tensor([FIM_PRE], dtype=torch.long),
            prefix,
            torch.tensor([FIM_SUF], dtype=torch.long),
            suffix,
            torch.tensor([FIM_MID], dtype=torch.long),
            middle,
            torch.tensor([FIM_END], dtype=torch.long),
        ])
        fim_n = len(fim_seq) - 1
        x[i, :fim_n] = fim_seq[:fim_n]
        mid_pos = len(prefix) + len(suffix) + 2
        if fim_n > mid_pos:
            end = min(fim_n - mid_pos, len(fim_seq) - mid_pos - 1)
            y[i, mid_pos:mid_pos + end] = fim_seq[mid_pos + 1:mid_pos + 1 + end]
        fim_flags.append(True)

    return x.to(DEVICE), y.to(DEVICE), categories, fim_flags


def get_lr(step: int, step0: int = 0) -> float:
    s = step - step0
    if UPSCALED and s < WAKEUP_STEPS:
        return WAKEUP_LR * (s + 1) / WARMUP_STEPS
    s = s - WAKEUP_STEPS
    if s < WARMUP_STEPS:
        return MAX_LR * (s + 1) / WARMUP_STEPS
    if s >= LR_DECAY_STEPS:
        return MIN_LR
    decay_ratio = (s - WARMUP_STEPS) / (LR_DECAY_STEPS - WARMUP_STEPS)
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
    if MODEL_DTYPE != torch.float32:
        model.to(MODEL_DTYPE)
    print(f"model dtype: {MODEL_DTYPE}")

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
        all_pt = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt")]
        big_ckpts = sorted([f for f in all_pt if f.startswith(CKPT_PREFIX)], key=_step_from_name)
        normal_ckpts = sorted([f for f in all_pt if f.startswith("step_") and not f.startswith(CKPT_PREFIX)], key=_step_from_name)
        if big_ckpts:
            latest = os.path.join(CKPT_DIR, big_ckpts[-1])
            print(f"Loading big checkpoint: {latest}")
            ckpt = torch.load(latest, map_location="cpu", mmap=True)
            sd = ckpt["model"]
            ckpt_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
            if ckpt_layers != N_LAYERS:
                raise RuntimeError(f"big checkpoint {latest} has {ckpt_layers} layers, expected {N_LAYERS}")
            model.load_state_dict(sd)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"] + 1
            print(f"Resuming from step {start_step}")
        elif normal_ckpts:
            latest = os.path.join(CKPT_DIR, normal_ckpts[-1])
            print(f"Loading normal checkpoint: {latest}")
            ckpt = torch.load(latest, map_location="cpu", mmap=True)
            sd = ckpt["model"]
            ckpt_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
            if UPSCALE_ON_RESUME and ckpt_layers < N_LAYERS:
                print(f"Upscaling {ckpt_layers} -> {N_LAYERS} layers (identity init)")
                upscale_into(model, sd, ckpt_layers)
                UPSCALED = True
                start_step = ckpt["step"] + 1
                big_path = f"{CKPT_DIR}/{CKPT_PREFIX}{start_step - 1}.pt"
                new_sd = model.state_dict()
                del ckpt, sd
                torch.save({"model": new_sd, "step": start_step - 1}, big_path)
                print(f"Saved upscaled big checkpoint: {big_path}")
            else:
                model.load_state_dict(sd)
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
                start_step = ckpt["step"] + 1
                print(f"Resuming from step {start_step}")
        else:
            print("No checkpoint found, starting from scratch")

    if UPSCALED:
        for i in range(0, 2 * OLD_N_LAYERS, 2):
            for p in model.blocks[i].parameters():
                p.requires_grad = False
        print(f"Wake-up phase: old blocks frozen for {WAKEUP_STEPS} steps")

    wandb.login()
    wandb.init(project="locLMM", config={
        "vocab_size": VOCAB_SIZE, "block_size": BLOCK_SIZE, "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM, "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "dim": DIM, "n_layers": N_LAYERS, "n_heads": N_HEADS, "upscaled_from": OLD_N_LAYERS,
        "wakeup_steps": WAKEUP_STEPS, "wakeup_lr": WAKEUP_LR,
        "max_lr": MAX_LR, "min_lr": MIN_LR, "max_steps": MAX_STEPS,
        "lr_decay_steps": LR_DECAY_STEPS, "params": n_params,
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
    scaler = torch.amp.GradScaler(enabled=USE_SCALER)

    EMA_BETA = 0.01
    ema_loss = ema_ppl = ema_acc = ema_grad = None

    def _ema_update(ema, cur, beta=EMA_BETA):
        if cur is None or not math.isfinite(cur):
            return ema
        if ema is None or not math.isfinite(ema):
            return cur
        return beta * cur + (1 - beta) * ema

    def _micro_step():
        x, y, cats, fim_flags = make_batch(BATCH_SIZE, BLOCK_SIZE)
        if (y == -100).all():
            return 0.0, 0.0, 0.0, 0, {}
        with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE, enabled=(DEVICE == "cuda")):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        with torch.no_grad():
            mask = y != -100
            acc = (logits.argmax(dim=-1)[mask] == y[mask]).float().mean().item()
            per_row = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="none",
            ).view_as(y)
            row_counts = mask.sum(dim=-1).clamp(min=1).float()
            row_loss = per_row.sum(dim=-1) / row_counts
            cat_stats = {}
            for j, cat in enumerate(cats):
                key = f"fim/{cat}" if fim_flags[j] else f"lm/{cat}"
                wl, nt = cat_stats.get(key, (0.0, 0))
                rc = int(row_counts[j].item())
                cat_stats[key] = (wl + row_loss[j].item() * rc, nt + rc)
        scaler.scale(loss / GRAD_ACCUM).backward()
        loss_val = loss.item()
        n_tok = mask.sum().item()
        del logits, loss, mask, x, y
        return loss_val, math.exp(min(loss_val, 20)), acc, n_tok, cat_stats

    eval_set = []

    def build_eval_set(n: int = EVAL_SAMPLES):
        eval_set.clear()
        attempts = 0
        while len(eval_set) < n and attempts < 5:
            attempts += 1
            try:
                fresh = get_next_samples(n - len(eval_set))
            except RuntimeError:
                break
            if not fresh:
                break
            for cat, tokens in fresh:
                if len(tokens) >= 64:
                    eval_set.append((cat, tokens[:BLOCK_SIZE]))
        print(f"Cached eval set: {len(eval_set)} samples (fixed for this run)")

    @torch.no_grad()
    def run_eval(step: int):
        if not eval_set:
            return
        x = torch.full((len(eval_set), BLOCK_SIZE), 0, dtype=torch.long, device=DEVICE)
        y = torch.full((len(eval_set), BLOCK_SIZE), -100, dtype=torch.long, device=DEVICE)
        for i, (cat, tokens) in enumerate(eval_set):
            seq = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
            n = min(len(seq) - 1, BLOCK_SIZE)
            x[i, :n] = seq[:n]
            y[i, :n] = seq[1:n + 1]
        model.eval()
        with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE, enabled=(DEVICE == "cuda")):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        model.train()
        val = loss.item()
        print(f"  eval @ step {step}: loss {val:.4f} | ppl {math.exp(min(val, 20)):.1f}")
        wandb.log({"eval_loss": val, "eval_ppl": math.exp(min(val, 20))}, step=step)

    build_eval_set()

    for step in range(start_step, MAX_STEPS):
        if UPSCALED and step - start_step == WAKEUP_STEPS:
            for i in range(0, 2 * OLD_N_LAYERS, 2):
                for p in model.blocks[i].parameters():
                    p.requires_grad = True
            print(f"Wake-up done @ step {step}: unfreezing old blocks")

        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)
        accum_loss_w = accum_acc = 0.0
        accum_tokens = 0
        cat_stats_accum = {}

        for micro in range(GRAD_ACCUM):
            l, _, a, n, cat_stats = _micro_step()
            if n == 0:
                continue
            accum_loss_w += l * n
            accum_acc += a * n
            accum_tokens += n
            for key, (wl, nt) in cat_stats.items():
                aw, at = cat_stats_accum.get(key, (0.0, 0))
                cat_stats_accum[key] = (aw + wl, at + nt)

        if accum_tokens == 0:
            continue

        lr = get_lr(step, start_step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

        cur_loss = accum_loss_w / max(accum_tokens, 1)
        cur_ppl = math.exp(min(cur_loss, 20))
        cur_acc = accum_acc / max(accum_tokens, 1)
        cur_grad = grad_norm.item()
        if not math.isfinite(cur_grad):
            print(f"WARNING: step {step}: non-finite grad_norm={cur_grad} — "
                  f"GradScaler likely skipped this step; ema_grad update skipped")
        tok_per_sec = accum_tokens / max(dt, 1e-6)
        tokens_seen = accum_tokens

        ema_loss = _ema_update(ema_loss, cur_loss)
        ema_ppl = _ema_update(ema_ppl, cur_ppl)
        ema_acc = _ema_update(ema_acc, cur_acc)
        ema_grad = _ema_update(ema_grad, cur_grad)

        if step % LOG_EVERY == 0:
            cat_metrics = {}
            for key, (wl, nt) in cat_stats_accum.items():
                if nt > 0:
                    cat_metrics[f"loss/{key}"] = wl / nt
                cat_metrics[f"tokens/{key}"] = nt
            print(f"step {step:6d} | loss {cur_loss:.4f} | ppl {cur_ppl:.1f} | "
                  f"acc {cur_acc:.3f} | lr {lr:.2e} | "
                  f"grad_norm {cur_grad:.2f} | {tok_per_sec:.0f} tok/s | "
                  f"ema_loss {ema_loss:.4f} | ema_ppl {ema_ppl:.1f} | "
                  f"ema_acc {ema_acc:.3f} | ema_grad {ema_grad:.2f}")
            wandb.log({"loss": cur_loss, "ppl": cur_ppl, "acc": cur_acc, "lr": lr,
                        "grad_norm": cur_grad,
                        "tok_per_sec": tok_per_sec, "tokens_seen": tokens_seen,
                        "ema_loss": ema_loss, "ema_ppl": ema_ppl, "ema_acc": ema_acc,
                        "ema_grad": ema_grad, **cat_metrics}, step=step)

        if step > 0 and step % CKPT_EVERY == 0:
            ckpt_path = f"{CKPT_DIR}/{CKPT_PREFIX}{step}.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step}, ckpt_path)
            print(f"saved checkpoint: {ckpt_path}")
            run_checkpoint_sample(step, model, sp, BLOCK_SIZE, DEVICE)

            if KEEP_CHECKPOINTS_COUNT == 0 or KEEP_CHECKPOINTS_COUNT == -1:
                pass
            elif KEEP_CHECKPOINTS_COUNT < 0:
                for f in os.listdir(CKPT_DIR):
                    if f.endswith(".pt"):
                        os.remove(os.path.join(CKPT_DIR, f))
                print(f"removed all checkpoints (KEEP_CHECKPOINTS_COUNT={KEEP_CHECKPOINTS_COUNT})")
            else:
                bigs = sorted(
                    [f for f in os.listdir(CKPT_DIR) if f.startswith(CKPT_PREFIX)],
                    key=_step_from_name,
                )
                while len(bigs) > KEEP_CHECKPOINTS_COUNT:
                    old = bigs.pop(0)
                    os.remove(os.path.join(CKPT_DIR, old))
                    print(f"removed old checkpoint: {old}")
                for f in os.listdir(CKPT_DIR):
                    if f.endswith(".pt") and not f.startswith(CKPT_PREFIX):
                        os.remove(os.path.join(CKPT_DIR, f))
                        print(f"removed superseded normal checkpoint: {f}")

        if step > 0 and step % EVAL_EVERY == 0:
            run_eval(step)
