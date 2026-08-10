from __future__ import annotations
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
from checkpoint_sample import record_raw_tokens, run_checkpoint_sample
import chatml

# API = "http://localhost:8823"  # for local testing
API = "http://91.98.145.193:8823"

TOKENIZER_MODEL_PATH = "../tok/tokenize/tokenizer_models/tokenizer.model"

BLOCK_SIZE = 4096
BATCH_SIZE = 7
DIM = 1024
N_LAYERS = 26
N_HEADS = 16

RESUME_FROM_CHECKPOINT = True
KEEP_CHECKPOINTS_COUNT = 2 # -1 or 0 to keep all, >0 to keep last N checkpoints, <0 to keep none
RANDOM_SAMPLING = True

MAX_STEPS = 500_000
WARMUP_STEPS = 300
MAX_LR = 1.5e-4
MIN_LR = 3e-5
LR_DECAY_STEPS = 250_000  # cosine decays to MIN_LR by this step (faster than MAX_STEPS)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
CHATML_MASK_PROB = 0.8
FIM_RATIO = 0.8
GRAD_ACCUM = 3

LOG_EVERY = 10
CKPT_EVERY = 250
EVAL_EVERY = 250
EVAL_SAMPLES = 8
# CKPT_DIR = "/media/user/sda1/CKPT_DIR"
CKPT_DIR = "./checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRECISION = "bf16"

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
VOCAB_SIZE = sp.get_piece_size() + 4  # +4 reserved FIM sentinel slots

# FIM sentinel token IDs (reserved, not in SentencePiece vocab — inserted manually during batch construction)
FIM_PRE, FIM_SUF, FIM_MID, FIM_END = sp.get_piece_size(), sp.get_piece_size() + 1, sp.get_piece_size() + 2, sp.get_piece_size() + 3

_CHATML = chatml.ChatMLDetector(sp)


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


REQUEST_TIMEOUT = 30  # seconds
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
    """GET request with timeout and retry on transient network errors."""
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
    """Fetch category index from server and return IDs belonging to code."""
    res = _request_with_retry(API + "/api/get-category-index")
    cat_map = res.json()
    code_names = {
        "python", "javascript", "c++", "java", "c", "go", "typescript",
        "ruby", "rust", "php", "swift", "c#", "kotlin", "scala", "dart",
        "objective-c", "perl", "lua", "sql", "html", "css", "json",
        "yaml", "markdown", "xml", "otherlanguage",
        "star_coder",
    }
    return {cid for name, cid in cat_map.items() if name.strip().lower() in code_names}


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
        do_fim = is_code and random.random() < FIM_RATIO and len(tokens) >= 64

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
                do_fim = False  # not enough room for a meaningful middle+suffix → plain LM

        if not do_fim:
            seq = torch.tensor(tokens, dtype=torch.long)
            n = min(len(tokens) - 1, block_size)
            x[i, :n] = seq[:n]
            y[i, :n] = seq[1:n + 1]
            if CHATML_MASK_PROB > 0.0:
                mt = _CHATML.mask_targets(tokens, n)
                if mt is not None and random.random() < CHATML_MASK_PROB:
                    y[i, :n] = torch.where(
                        mt.to(seq.device), seq[1:n + 1],
                        torch.full_like(seq[1:n + 1], -100))
            fim_flags.append(False)
            continue

        mid_len = random.randint(min(64, mid_max), mid_max)
        mid_end = min(pre_end + mid_len, len(tokens) - 4)

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
        fim_n = len(fim_seq) - 1
        x[i, :fim_n] = fim_seq[:fim_n]
        mid_pos = len(prefix) + len(suffix) + 2
        if fim_n > mid_pos:
            end = min(fim_n - mid_pos, len(fim_seq) - mid_pos - 1)
            y[i, mid_pos:mid_pos + end] = fim_seq[mid_pos + 1:mid_pos + 1 + end]
        fim_flags.append(True)

    return x.to(DEVICE), y.to(DEVICE), categories, fim_flags


def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step >= LR_DECAY_STEPS:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / (LR_DECAY_STEPS - WARMUP_STEPS)
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
        "grad_accum": GRAD_ACCUM, "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "dim": DIM, "n_layers": N_LAYERS, "n_heads": N_HEADS,
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
            # per-row losses for diagnostics (no extra forward pass, transient only)
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
        """Cache a small fixed eval set from the server at startup (RAM only)."""
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
            mt = _CHATML.mask_targets(tokens, n)
            if mt is not None:
                y[i, :n] = torch.where(
                    mt.to(seq.device), seq[1:n + 1],
                    torch.full_like(seq[1:n + 1], -100))
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
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)
        accum_loss_w = accum_acc = 0.0
        accum_tokens = 0
        cat_stats_accum = {}  # {mode/cat: (weighted_loss_sum, n_tokens)}

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

        lr = get_lr(step)
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
            ckpt_path = f"{CKPT_DIR}/step_{step}.pt"
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
                ckpts = sorted(
                    [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt")],
                    key=lambda x: int(x.replace("step_", "").replace(".pt", "")),
                )
                while len(ckpts) > KEEP_CHECKPOINTS_COUNT:
                    old = ckpts.pop(0)
                    os.remove(os.path.join(CKPT_DIR, old))
                    print(f"removed old checkpoint: {old}")

        if step > 0 and step % EVAL_EVERY == 0:
            run_eval(step)
