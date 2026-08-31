from __future__ import annotations
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

print(f"Running {os.path.basename(__file__)} (torch {torch.__version__})")

import random
import re
import time
import math
import struct
import base64
import queue
import threading
import sys
import json
import subprocess
import tempfile

import requests
import sentencepiece as spm
import torch.nn.functional as F
import wandb
import questionary
from questionary import Style

from model import Transformer
from checkpoint_sample import record_raw_tokens, record_fim_sample, run_checkpoint_sample, run_fim_checkpoint_sample
import chatml

API = "http://91.98.145.193:8823"
FIM_API = "http://localhost:8823"
FIM_SEARCH_API = "http://localhost:8234/search"
CONTEXT_MAX_TOKENS = 1024

TOKENIZER_MODEL_PATH = "../tok/tokenize/tokenizer_models/tokenizer.model"

BLOCK_SIZE = 8192
BATCH_SIZE = 2
GRAD_ACCUM = 6  # effective batch = BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM (2*8192*6 ≈ 98k tokens/step)
LOSS_SKIP_THRESHOLD = 5.0  # per-micro-step loss above this = corrupted batch -> skip it
MIN_SAMPLE_TOKENS = 64  # drop degenerate/short samples (loss over a few tokens is pure noise)
DIM = 1024
N_LAYERS = 128
OLD_N_LAYERS = 26
N_HEADS = 16

RESUME_FROM_CHECKPOINT = True
UPSCALE_ON_RESUME = True
VOCAB_RESIZE_ON_RESUME = True
KEEP_CHECKPOINTS_COUNT = 1
RANDOM_SAMPLING = True

MAX_STEPS = 500_000
WARMUP_STEPS = 300
WAKEUP_STEPS = 1500  # wake-up phase (LR burst + freeze) after a layer upscale
VOCAB_WAKEUP_STEPS = 1500  # wake-up phase when only the vocab was resized
WAKEUP_LR = 1e-4
MAX_LR = 1e-4
MIN_LR = 1e-5
LR_DECAY_STEPS = MAX_STEPS  # decay spans the full MAX_STEPS horizon (was 250k -> LR floor for the last half)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
CHATML_MASK_PROB = 0.8
FIM_RATIO = 0.95
FIM_VARIANTS = 1  # FIM samples generated per code sample
FIM_MAX_SAMPLE_TOKENS = 0  # 0 = FULL window (up to BLOCK_SIZE=8192) per sample:
                           # maximum context for every sample / matches 8K.
                           # Set e.g. 1536/4096 to cap windows and go faster.
NO_CONTEXT_PROB = 0.5  # RAG-only knob: 50/50 with/without context when LOCLLM_RAG_TRAIN=1, unused in plain FIM mode
# SAFIM-style short-span emphasis: benchmark completions are mostly one-liners
# (e.g. api calls, control-flow expressions), not long chunks. ~75% of FIM
# middles are line-level and only ~5% up to the full window budget.
SHORT_MID_CUM = 0.75    # cumulative prob for line-level spans (<= ~128 tok)
MEDIUM_MID_CUM = 0.95   # cumulative prob for short/medium spans (16..160 tok)
ROPE_BASE = 10000.0  # test e.g. 100000 for longer extrapolation at 8192
LOSS_CHUNK = 1024  # sequence-chunk size for head+loss (bounds logits memory;
                    # 1024 keeps peak logits ~0.4GB at B=8; 2048 is faster but +~0.4GB)

LOG_EVERY = 10
CKPT_EVERY = 75
EVAL_EVERY = 125
EVAL_FIRST_AFTER = 10  # run one early eval after N steps (fail-fast sanity check)
EVAL_SAMPLES = 64
FIM_EVAL_SAMPLES = 32
FIM_EVAL_GEN_SAMPLES = 4
FIM_EVAL_GEN_TOKENS = 32
CKPT_DIR = "./checkpoints"

PRECISION = "bf16"
MODEL_DTYPE = torch.bfloat16 if PRECISION == "bf16" else torch.float32

DEVICE = "cpu"
AUTOCAST_DTYPE = torch.float32
USE_SCALER = False

sp = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL_PATH)
BASE_VOCAB = sp.get_piece_size()
VOCAB_SIZE = BASE_VOCAB + chatml.reserved_total()

FIM_PRE, FIM_SUF, FIM_MID, FIM_END = BASE_VOCAB, BASE_VOCAB + 1, BASE_VOCAB + 2, BASE_VOCAB + 3
CONTEXT_IDS = chatml.context_ids(BASE_VOCAB)
CONTEXT_START, CONTEXT_END = CONTEXT_IDS["start"], CONTEXT_IDS["end"]
LANG_IDS = chatml.lang_ids(BASE_VOCAB)
LANG_OPEN, LANG_CLOSE = LANG_IDS["open"], LANG_IDS["close"]
# Worst-case tokens for the three <lang>...</lang> blocks (context + prefix + suffix):
# 3 blocks x (2 marker tokens + up to 6 tokens for the language name).
LANG_OVERHEAD = 24
NEWLINE_IDS = set(sp.encode("\n", out_type=int))

_CHATML = chatml.ChatMLDetector(sp)
CHATML_IDS = chatml.reserved_ids(BASE_VOCAB)
CHATML_IDS["start_core"] = sp.encode("<|im_start|>", out_type=int)[1:]
CHATML_IDS["end_core"] = sp.encode("<|im_end|>", out_type=int)[1:]
IM_START, IM_END = CHATML_IDS["im_start"], CHATML_IDS["im_end"]
ROLE_ASSISTANT = CHATML_IDS["roles"]["assistant"]

UPSCALED = False
VOCAB_RESIZED = False
CKPT_PREFIX = "step_big_"


def _select_gpu_interactive():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    tmp_path = tmp.name
    tmp.close()
    code = (
        "import json\n"
        "from gpuSeletor.main import select_gpus_with_config\n"
        f"result = select_gpus_with_config()\n"
        f"with open({tmp_path!r}, 'w') as f:\n"
        "    json.dump(result, f)\n"
    )
    try:
        subprocess.run([sys.executable, "-c", code], cwd=script_dir, check=True)
        with open(tmp_path) as f:
            result = json.load(f)
    except Exception as e:
        print(f"WARNING: GPU selection failed ({e}); falling back to defaults")
        result = []
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return result if isinstance(result, list) else []


FIM_MODE = False
# RAG conditioning is OFF in FIM training. Opt in only for a separate
# RAG-context experiment: LOCLLM_RAG_TRAIN=1 python main_big.py
RAG_TRAIN_MODE = os.environ.get("LOCLLM_RAG_TRAIN") == "1"

def _select_fim_mode_enable():
    global FIM_MODE
    custom_style = Style([
        ("qmark", "fg:#00d7ff bold"),
        ("question", "bold"),
        ("pointer", "fg:#00d7ff bold"),
        ("highlighted", "fg:#00d7ff bold"),
        ("selected", "fg:#00d7af"),
    ])
    choice = questionary.select(
        "Enable FIM mode",
        choices=[
            questionary.Choice(title="Yes", value=True),
            questionary.Choice(title="No", value=False),
        ],
        default=False,
        style=custom_style,
    ).ask()
    FIM_MODE = bool(choice)


def _step_from_name(f):
    return int(re.search(r"\d+", f).group())


def _load_ckpt(path):
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def resize_vocab_embeddings(model, old_sd, old_vocab):
    new_sd = {}
    for k, v in old_sd.items():
        if k in ("tok_emb.weight", "lm_head.weight"):
            w = torch.zeros(VOCAB_SIZE, DIM, dtype=v.dtype)
            w[:old_vocab] = v
            w[old_vocab:] = v.mean(dim=0, keepdim=True)
            new_sd[k] = w
        else:
            new_sd[k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    return missing, unexpected


def _splice_optimizer(old_opt_sd, model, optimizer, old_vocab):
    grouped = []
    for g in optimizer.param_groups:
        grouped.extend(g["params"])
    emb_idx = next(i for i, p in enumerate(grouped) if p is model.tok_emb.weight)
    state = dict(old_opt_sd["state"])
    emb_state = dict(state[emb_idx])
    for key in ("exp_avg", "exp_avg_sq"):
        t = state[emb_idx][key]
        padded = torch.zeros(VOCAB_SIZE, *t.shape[1:], dtype=t.dtype)
        padded[:old_vocab] = t
        emb_state[key] = padded
    state[emb_idx] = emb_state
    sd = dict(old_opt_sd)
    sd["state"] = state
    return sd


def _is_8bit_optimizer(opt) -> bool:
    return opt.__class__.__name__ == "AdamW8bit"


def _opt_state_is_8bit(opt_sd) -> bool:
    for v in opt_sd.get("state", {}).values():
        if isinstance(v, dict):
            return "state1" in v or "qmap1" in v
    return False


def _load_optimizer_state(optimizer, ckpt, model, old_vocab=None) -> bool:
    if "optimizer" not in ckpt:
        return False
    old_sd = ckpt["optimizer"]
    want_8bit = _is_8bit_optimizer(optimizer)
    if _opt_state_is_8bit(old_sd) != want_8bit:
        print("  WARNING: saved optimizer state (8-bit vs fp32) does not match the "
              "selected optimizer — starting optimizer fresh (model weights unchanged)")
        return False
    try:
        if old_vocab is not None and not want_8bit:
            optimizer.load_state_dict(_splice_optimizer(old_sd, model, optimizer, old_vocab))
        else:
            optimizer.load_state_dict(old_sd)
        return True
    except Exception as e:
        print(f"  WARNING: optimizer state not loaded ({e.__class__.__name__}: {e}) — "
              f"starting optimizer fresh (model weights unchanged)")
        return False


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
        t = struct.unpack_from("<H", data, offset)[0]
        if t >= VOCAB_SIZE:
            raise ValueError(f"token id {t} out of range (vocab {VOCAB_SIZE})")
        tokens.append(t)
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


def _request_with_retry(url: str, method: str = "get", **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
    last_exc = None
    for attempt in range(MAX_REQUEST_RETRIES):
        try:
            return getattr(_get_session(), method)(url, **kwargs)
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


RAG_BATCH_MAX = 8
RAG_QUERY_MAX_TOKENS = 1536
RAG_QUERY_MODE = "prefix_suffix"
RAG_CACHE_MAX = 8192
_rag_cache = {}


def _search_texts(texts: list) -> list:
    try:
        res = _get_session().post(
            FIM_SEARCH_API, json={"texts": texts, "top_k": 1}, timeout=10)
        data = res.json()
        return [d["results"][0] if d.get("results") else None for d in data]
    except Exception as e:
        print(f"  WARNING: RAG search failed: {e}")
        return [None] * len(texts)


def search_rag_batch(queries: list) -> list:
    # FIM training NEVER queries RAG unless explicitly enabled (LOCLLM_RAG_TRAIN=1).
    if not RAG_TRAIN_MODE:
        return [None] * len(queries)
    if len(_rag_cache) > RAG_CACHE_MAX:
        _rag_cache.clear()
    out = [None] * len(queries)
    misses = []
    for i, q in enumerate(queries):
        if q in _rag_cache:
            out[i] = _rag_cache[q]
        else:
            misses.append(i)
    for j in range(0, len(misses), RAG_BATCH_MAX):
        chunk = misses[j:j + RAG_BATCH_MAX]
        records = _search_texts([queries[i] for i in chunk])
        for i, record in zip(chunk, records):
            if record is not None:
                _rag_cache[queries[i]] = record
            out[i] = record
    return out


def search_rag(query_text: str) -> str | None:
    return search_rag_batch([query_text])[0]


def get_next_samples(count: int) -> list[tuple[int, list[int]]]:
    endpoint = "/api/get-next-samples-random" if RANDOM_SAMPLING else "/api/get-next-samples"
    res = _request_with_retry(API + endpoint, params={"sample_count": count})
    raw_samples = res.json().get("samples", [])
    samples = []
    dropped = 0
    for raw in raw_samples:
        try:
            samples.append(decode_record(base64.b64decode(raw)))
        except ValueError as e:
            dropped += 1
            if dropped <= 5:
                print(f"  dropped invalid record: {e}")
    if dropped:
        print(f"  dropped {dropped}/{len(raw_samples)} invalid records")
    return samples


def get_category_index() -> tuple[set[int], dict[int, str]]:
    res = _request_with_retry(API + "/api/get-category-index")
    cat_map = res.json()
    code_names = {
        "python", "javascript", "c++", "java", "c", "go", "typescript",
        "ruby", "rust", "php", "swift", "c#", "kotlin", "scala", "dart",
        "objective-c", "perl", "lua", "sql", "html", "css", "yaml", "tsx", "shell", "otherlanguage",
        # data/doc formats (markdown/json/xml) train as plain LM, not FIM
        # extra raw-code categories emitted by dataSets.py generators
        "golang",     # stack_v3_gen (Go files tagged "Golang")
        "star_coder", # star_coder_gen (bigcode/starcoderdata)
        # stack-v3 real programming languages (CODE_LANGS in tok/loadData/dataSets.py)
        "objective-c++", "r", "julia", "matlab", "powershell", "groovy", "tcl",
        "zig", "nim", "crystal", "d", "v", "odin", "gleam", "mojo", "carbon",
        "hare", "jai", "pony", "chapel", "vala", "cython",
        "haskell", "ocaml", "f#", "erlang", "elixir", "clojure", "elm",
        "purescript", "rescript", "reason", "racket", "scheme", "common lisp",
        "emacs lisp", "apl", "j", "raku",
        "assembly", "cuda", "glsl", "hlsl", "opencl", "wgsl", "webassembly",
        "verilog", "systemverilog", "vhdl",
        "fortran", "fortran free form", "pascal", "ada", "cobol", "visual basic .net",
        "solidity", "vyper", "move", "cairo", "noir",
        "q#", "starlark", "ballerina",
    }
    code_ids = set()
    name_by_id = {}
    for name, cid in cat_map.items():
        norm = normalize_lang(name)
        name_by_id[cid] = norm
        if name.strip().lower() in code_names:
            code_ids.add(cid)
    return code_ids, name_by_id


LANG_ALIASES = {"golang": "go", "cpp": "c++"}


def normalize_lang(name) -> str:
    n = (name or "").strip().lower()
    return LANG_ALIASES.get(n, n)


def lang_block_ids(name) -> list[int] | None:
    """<lang>name</lang> token block, or None if the language is unknown/generic."""
    if not name:
        return None
    n = normalize_lang(name)
    if n in ("star_coder", "otherlanguage", "other language", "code"):
        return None
    return [LANG_OPEN, *sp.encode(n, out_type=int), LANG_CLOSE]


CODE_CATEGORY_IDS = set()
CAT_NAME_BY_ID: dict[int, str] = {}

MAX_CACHE_SIZE = 1024

FETCH_BULK = 512
POOL_MIN = 128
BATCH_QUEUE_MAX = 4

_sample_pool: list[tuple[int, list[int]]] = []
_batch_queue = queue.Queue(maxsize=BATCH_QUEUE_MAX)
_prefetch_thread: "threading.Thread | None" = None

# --- lightweight instrumentation counters (read by the trainer loop) ---
_stats_lock = threading.Lock()
_stats = {
    "fim_eligible": 0,      # FIM-eligible code samples planned
    "fim_trunc": 0,         # ... truncated to fim_cap
    "hist_1k": 0, "hist_2k": 0, "hist_4k": 0, "hist_8k": 0,  # pre-truncation length buckets
    "rag_q": 0,             # RAG queries issued
    "rag_miss": 0,          # ... that produced no context
    "ctx_n": 0, "ctx_len": 0, "ctx_clip": 0,  # context built / tokens used / clipped at cap
}


def _stat(key: str, n: int = 1):
    with _stats_lock:
        _stats[key] += n


def _drain_stats() -> dict:
    with _stats_lock:
        s = dict(_stats)
        for k in _stats:
            _stats[k] = 0
    return s


def _fetch_samples_with_retry(count: int) -> list[tuple[int, list[int]]]:
    max_retries = 300
    for attempt in range(max_retries):
        try:
            fresh = get_next_samples(count)
        except RuntimeError as e:
            if attempt == 0:
                print(f"WARNING: get_next_samples failed: {e}")
            if attempt < max_retries - 1:
                delay = min(60, 2 + 2 ** min(attempt, 6))
                time.sleep(delay)
                continue
            raise
        if fresh:
            return fresh
        if attempt == 0:
            print("WARNING: no samples returned — cursor may be exhausted, retrying...")
        time.sleep(2)
    raise RuntimeError(
        f"no samples returned after {max_retries} retries — "
        "check that the data server is running and data exists"
    )


def _sample_mid_len(mid_max: int) -> int:
    """Middle-span mixture biased toward short, line-level completions:
    ~75% 1-few lines (<=128 tok, incl. tiny 2-3 token expressions),
    ~20% short-medium (16-160 tok), ~5% up to the window budget."""
    if mid_max <= 1:
        return 1
    r = random.random()
    if r < SHORT_MID_CUM:
        return min(mid_max, random.choice((2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)))
    if r < MEDIUM_MID_CUM:
        return random.randint(min(16, mid_max), min(160, mid_max))
    return random.randint(min(64, mid_max), mid_max)


def _snap_newline(tokens: list, pos: int, window: int = 96) -> int:
    """Snap a split position to the start of the next line (newline token + 1).
    Falls back to the raw offset when no newline is nearby (single-line code)."""
    for j in range(pos, min(len(tokens), pos + window)):
        if tokens[j] in NEWLINE_IDS:
            return j + 1
    return pos


def _fim_splits(tokens: list, n: int):
    L = len(tokens)
    cut = _snap_newline(tokens, random.randint(L // 4, 7 * L // 10))
    splits = []
    for i in range(n):
        remaining = L - cut
        reserve = 16 * (n - i - 1) + 4
        mid_max = min(L // 4, remaining - reserve)
        if mid_max < 4:
            break
        mid_len = _sample_mid_len(mid_max)
        mid_end = _snap_newline(tokens, cut + mid_len)
        if mid_end - cut < 4:
            mid_end = cut + mid_len
        mid_end = min(mid_end, L)
        if mid_end - cut < 4:
            break
        splits.append((cut, mid_end))
        cut = mid_end
    return splits


def _fim_variant(seq, pre_end: int, mid_end: int, context=None, sample_lang=None):
    prefix = seq[:pre_end]
    middle = seq[pre_end:mid_end]
    suffix = seq[mid_end:]
    parts = []
    if context is not None:
        parts.append(torch.tensor(context, dtype=torch.long))
    s_lang = lang_block_ids(sample_lang)
    parts.append(torch.tensor([FIM_PRE], dtype=torch.long))
    if s_lang is not None:
        parts.append(torch.tensor(s_lang, dtype=torch.long))
    parts.append(prefix)
    parts.append(torch.tensor([FIM_SUF], dtype=torch.long))
    if s_lang is not None:
        parts.append(torch.tensor(s_lang, dtype=torch.long))
    parts.append(suffix)
    parts.append(torch.tensor([FIM_MID], dtype=torch.long))
    parts.append(middle)
    parts.append(torch.tensor([FIM_END], dtype=torch.long))
    fim_seq = torch.cat(parts)
    n = len(fim_seq) - 1
    mt = torch.zeros(n, dtype=torch.bool)
    # <fim_middle> appears exactly once; everything from it on is trainable
    # (middle + <fim_end>); context, lang tags, prefix and suffix are masked.
    mid_idx = int((fim_seq == FIM_MID).nonzero()[0].item())
    if mid_idx < n:
        mt[mid_idx:] = True
    return fim_seq, mt


def _fim_query(tokens: list, pre_end: int, mid_end: int):
    if RAG_QUERY_MODE == "prefix":
        query = sp.decode(tokens[:pre_end])
    elif RAG_QUERY_MODE == "suffix":
        query = sp.decode(tokens[mid_end:])
    else:
        query = sp.decode(tokens[:pre_end]) + "\n" + sp.decode(tokens[mid_end:])
    query = query.strip()
    if not query:
        return None
    qids = sp.encode(query, out_type=int)
    if len(qids) > RAG_QUERY_MAX_TOKENS:
        qids = qids[:RAG_QUERY_MAX_TOKENS]
        query = sp.decode(qids)
    return query


def _make_context(tokens: list, record, block_size: int):
    if not record:
        return None
    code = record.get("code") if isinstance(record, dict) else None
    if not code:
        return None
    block = [CONTEXT_START]
    lang_block = lang_block_ids(record.get("lang"))
    if lang_block is not None:
        block.extend(lang_block)
    ctx = sp.encode(code, out_type=int)
    if len(ctx) < 8:
        return None
    overhead = len(block) + 1  # context markers + lang block; +1 for CONTEXT_END
    room = max(0, block_size - len(tokens) - 6 - LANG_OVERHEAD)
    if room < 8 + overhead:
        return None
    limit = min(CONTEXT_MAX_TOKENS, room - overhead)
    if len(ctx) > limit:
        _stat("ctx_clip")
    ctx = ctx[:limit]
    if len(ctx) < 8:
        return None
    block.extend(ctx)
    block.append(CONTEXT_END)
    _stat("ctx_n")
    _stat("ctx_len", len(ctx))
    return block


def _flip_no_context() -> bool:
    return NO_CONTEXT_PROB > 0.0 and random.random() < NO_CONTEXT_PROB


def _fim_context(tokens: list, pre_end: int, mid_end: int, block_size: int):
    query = _fim_query(tokens, pre_end, mid_end)
    if not query:
        return None
    record = search_rag(query)
    return _make_context(tokens, record, block_size)


def _plan_fim(tokens: list, cat_id: int, block_size: int):
    global _sample_pool
    is_code = cat_id in CODE_CATEGORY_IDS
    do_fim = is_code and random.random() < FIM_RATIO and len(tokens) >= MIN_SAMPLE_TOKENS
    use_ctx = False
    if do_fim:
        # RAG context is NOT part of FIM training by default. Enable it only
        # for a separate experiment: LOCLLM_RAG_TRAIN=1 (pairs with RAG server).
        use_ctx = FIM_MODE and RAG_TRAIN_MODE and not _flip_no_context()
        _stat("fim_eligible")
        n0 = len(tokens)
        if n0 < 1024:
            _stat("hist_1k")
        elif n0 < 2048:
            _stat("hist_2k")
        elif n0 < 4096:
            _stat("hist_4k")
        else:
            _stat("hist_8k")
        if FIM_MODE:
            if use_ctx:
                fim_cap = block_size - CONTEXT_MAX_TOKENS - 6 - LANG_OVERHEAD
            else:
                fim_cap = block_size - 3 - LANG_OVERHEAD
            if FIM_MAX_SAMPLE_TOKENS > 0:
                fim_cap = min(fim_cap, FIM_MAX_SAMPLE_TOKENS)
        else:
            fim_cap = block_size - 3 - LANG_OVERHEAD
        if len(tokens) > fim_cap:
            _stat("fim_trunc")
            tail = tokens[fim_cap:]
            if len(tail) >= 16:
                if len(_sample_pool) < MAX_CACHE_SIZE:
                    _sample_pool.append((cat_id, tail))
                else:
                    print(f"WARNING: pool full ({MAX_CACHE_SIZE}), discarding tail of sample")
            tokens = tokens[:fim_cap]
        L = len(tokens)
        splits = _fim_splits(tokens, FIM_VARIANTS)
        if not splits:
            splits = _fim_splits(tokens, 1)
        if not splits:
            do_fim = False
    elif FIM_MODE and FIM_MAX_SAMPLE_TOKENS > 0:
        # Non-FIM (plain LM / non-code) samples: cap them too. Without this,
        # a single 8k-token sample in a batch pads every row to BLOCK_SIZE,
        # ~10GB+ of extra activations even at small batch sizes (the OOM).
        tokens = tokens[:FIM_MAX_SAMPLE_TOKENS]
    return tokens, splits if do_fim else [], use_ctx


def _process_sample(tokens: list, cat_id: int, block_size: int, splits=None, context_map=None, use_ctx=None):
    if splits is None:
        tokens, splits, use_ctx = _plan_fim(tokens, cat_id, block_size)
        context_map = None
    if not splits:
        headers, ends = _CHATML.analyze(tokens)
        if headers:
            toks = chatml.replace_markers(tokens, headers, ends, CHATML_IDS)
            mt = torch.tensor(chatml.mask_from_ids(toks, CHATML_IDS)[1:], dtype=torch.bool)
        else:
            toks = tokens
            mt = None
        return [(torch.tensor(toks, dtype=torch.long), mt, False)]

    seq = torch.tensor(tokens, dtype=torch.long)
    sample_lang = CAT_NAME_BY_ID.get(cat_id)
    variants = []
    for s in splits:
        context = None
        if FIM_MODE and use_ctx:
            if context_map is not None:
                context = context_map.get(s)
            else:
                context = _fim_context(tokens, s[0], s[1], block_size)
        vseq, vmt = _fim_variant(seq, s[0], s[1], context, sample_lang)
        variants.append((vseq, vmt, True))
    return variants


def _assemble_batch(processed: list, block_size: int):
    if not processed:
        return (torch.zeros(1, 1, dtype=torch.long),
                torch.full((1, 1), -100, dtype=torch.long),
                [], [])
    seqs = [p[0] for p in processed]
    ns = [min(len(s) - 1, block_size) for s in seqs]
    max_n = max(ns)
    bs = len(processed)
    x = torch.zeros((bs, max_n), dtype=torch.long)
    y = torch.full((bs, max_n), -100, dtype=torch.long)
    fim_flags = []
    for i, (seq, mt, fim_flag) in enumerate(processed):
        n = ns[i]
        x[i, :n] = seq[:n]
        y[i, :n] = seq[1:n + 1]
        if mt is not None:
            apply = fim_flag or (CHATML_MASK_PROB > 0.0 and random.random() < CHATML_MASK_PROB)
            if apply:
                y[i, :n] = torch.where(
                    mt[:n], seq[1:n + 1], torch.full_like(seq[1:n + 1], -100))
        fim_flags.append(fim_flag)
    return x, y, fim_flags


def _build_batch_from_pool(batch_size: int, block_size: int):
    global _sample_pool
    while len(_sample_pool) < POOL_MIN:
        _sample_pool.extend(_fetch_samples_with_retry(FETCH_BULK))

    trimmed = []
    tails = []
    for cat, tokens in _sample_pool:
        if len(tokens) > block_size + 1:
            if len(tails) < MAX_CACHE_SIZE:
                tails.append((cat, tokens[block_size:]))
            else:
                print(f"WARNING: pool full ({MAX_CACHE_SIZE}), discarding tail of sample")
            tokens = tokens[:block_size + 1]
        if len(tokens) >= MIN_SAMPLE_TOKENS:
            trimmed.append((cat, tokens))

    trimmed.sort(key=lambda c: len(c[1]))
    start = random.randrange(0, max(1, len(trimmed) - batch_size + 1))
    chosen = trimmed[start:start + batch_size]
    del trimmed[start:start + batch_size]
    _sample_pool = tails + trimmed

    if not chosen:
        return (torch.zeros(1, 1, dtype=torch.long),
                torch.full((1, 1), -100, dtype=torch.long),
                [], [])

    processed = []
    cats = []
    plans = []
    context_maps = [{} for _ in chosen]
    for cat, tokens in chosen:
        record_raw_tokens(tokens)
        pt, splits, use_ctx = _plan_fim(tokens, cat, block_size)
        plans.append((pt, cat, splits, use_ctx))
    pending = []
    queried = []
    if FIM_MODE and RAG_TRAIN_MODE:
        for i, (pt, cat, splits, use_ctx) in enumerate(plans):
            for s in splits:
                if not use_ctx:
                    context_maps[i][s] = None
                    continue
                q = _fim_query(pt, s[0], s[1])
                if q:
                    pending.append((i, s))
                    queried.append(q)
        records = search_rag_batch(queried)
        _stat("rag_q", len(queried))
        for (i, s), record in zip(pending, records):
            ctx_block = _make_context(plans[i][0], record, block_size)
            if ctx_block is None:
                _stat("rag_miss")
            context_maps[i][s] = ctx_block
    for i, (pt, cat, splits, use_ctx) in enumerate(plans):
        outs = _process_sample(pt, cat, block_size, splits=splits, context_map=context_maps[i], use_ctx=use_ctx)
        processed.extend(outs)
        cats.extend([cat] * len(outs))
    for seq, mt, fim_flag in reversed(processed):
        if fim_flag:
            lst = seq.tolist()
            record_fim_sample(lst, lst.index(FIM_MID))
            break
    x, y, fim_flags = _assemble_batch(processed, block_size)
    return x, y, cats, fim_flags


def _data_worker():
    while True:
        try:
            batch = _build_batch_from_pool(BATCH_SIZE, BLOCK_SIZE)
            _batch_queue.put(batch)
        except Exception as e:
            print(f"WARNING: data worker error: {e}")
            time.sleep(5)


def make_batch(batch_size: int, block_size: int):
    global _prefetch_thread
    if _prefetch_thread is None:
        _prefetch_thread = threading.Thread(target=_data_worker, daemon=True, name="data-worker")
        _prefetch_thread.start()
    x, y, cats, fim_flags = _batch_queue.get()
    return x.to(DEVICE), y.to(DEVICE), cats, fim_flags


def get_lr(step: int, step0: int = 0) -> float:
    # Wake-up phase (fresh on each resume, relative to start_step): new/upscaled
    # weights get a short burst at WAKEUP_LR before the main schedule takes over.
    s_rel = step - step0
    wake = VOCAB_WAKEUP_STEPS if VOCAB_RESIZED else WAKEUP_STEPS
    if (UPSCALED or VOCAB_RESIZED) and s_rel < wake:
        return WAKEUP_LR * min((s_rel + 1) / WARMUP_STEPS, 1.0)

    # Main cosine schedule is based on the ABSOLUTE step so it never restarts at
    # MAX_LR on resume (previously the schedule reset every run, pinning LR ~max).
    s = step
    if s < WARMUP_STEPS:
        lr = MAX_LR * (s + 1) / WARMUP_STEPS
    elif s >= LR_DECAY_STEPS:
        lr = MIN_LR
    else:
        decay_ratio = (s - WARMUP_STEPS) / (LR_DECAY_STEPS - WARMUP_STEPS)
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        lr = MIN_LR + coeff * (MAX_LR - MIN_LR)
    return max(0.0, lr)


if __name__ == "__main__":
    selected = _select_gpu_interactive()
    _select_fim_mode_enable()
    if FIM_MODE:
        API = FIM_API
        CKPT_PREFIX = "step_big_fim_"
    print(f"FIM mode: {FIM_MODE} | data API: {API} | ckpt prefix: {CKPT_PREFIX}")
    if len(selected) > 1:
        print(f"NOTE: multi-GPU training is not supported yet — using only the "
              f"first selected GPU ({selected[0]['name']})")
        selected = selected[:1]
    if not selected:
        print("No GPU selected — falling back to defaults "
              f"(GPU 0, batch_size={BATCH_SIZE}, accum={GRAD_ACCUM})")
        selected = [{"name": "GPU 0", "vram_size": 0.0, "index": 0,
                     "batch_size": BATCH_SIZE, "accumulation_steps": GRAD_ACCUM,
                     "optimizer": "fp32"}]

    gpu = selected[0]
    # Env overrides for speed tuning without touching the GPU selector:
    #   LOCLLM_BATCH_SIZE=8 LOCLLM_GRAD_ACCUM=6 python main_big.py
    BATCH_SIZE = int(os.environ.get("LOCLLM_BATCH_SIZE", gpu["batch_size"]))
    GRAD_ACCUM = int(os.environ.get("LOCLLM_GRAD_ACCUM", gpu["accumulation_steps"]))
    if "LOCLLM_BATCH_SIZE" in os.environ or "LOCLLM_GRAD_ACCUM" in os.environ:
        print(f"NOTE: env overrides active -> batch_size={BATCH_SIZE} accum={GRAD_ACCUM}")
    OPTIMIZER = gpu.get("optimizer", "fp32")
    print(f"\nTraining on: {gpu['name']} ({gpu.get('vram_size', 0):.1f} GB) | "
          f"batch_size={BATCH_SIZE} | accum={GRAD_ACCUM} | optimizer={OPTIMIZER} | "
          f"effective batch ≈ {BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM:,} tokens/step")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda" and PRECISION == "bf16":
        AUTOCAST_DTYPE = torch.bfloat16
        USE_SCALER = False
    elif DEVICE == "cuda":
        AUTOCAST_DTYPE = torch.float16
        USE_SCALER = True
    else:
        AUTOCAST_DTYPE = torch.float32
        USE_SCALER = False

    CODE_CATEGORY_IDS, CAT_NAME_BY_ID = get_category_index()
    print(f"Code categories for FIM: {len(CODE_CATEGORY_IDS)} IDs")

    samples = get_next_samples(1)
    if samples:
        _, tokens = samples[0]
        print(tokens[:20])
        print(sp.decode(tokens)[:200])

    print("building model (128 layers, ~1.6B params) ...", flush=True)
    model = Transformer(vocab_size=VOCAB_SIZE, dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS,
                         max_seq_len=BLOCK_SIZE, rope_base=ROPE_BASE).to(DEVICE)
    if MODEL_DTYPE != torch.float32:
        model.to(MODEL_DTYPE)
    print(f"model dtype: {MODEL_DTYPE}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {DEVICE} | params: {n_params / 1e6:.1f}M", flush=True)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)

    if OPTIMIZER == "8bit":
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise SystemExit(
                "AdamW 8-bit selected, but 'bitsandbytes' is not installed.\n"
                "Install it with: pip install bitsandbytes"
            )
        optimizer = bnb.optim.AdamW8bit(
            [{"params": decay, "weight_decay": WEIGHT_DECAY},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=MAX_LR, betas=(0.9, 0.95),
        )
        print("Using 8-bit AdamW optimizer (bitsandbytes)", flush=True)
    else:
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": WEIGHT_DECAY},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=MAX_LR, betas=(0.9, 0.95),
        )

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print(f"VRAM after model+optimizer: "
              f"{torch.cuda.memory_allocated() / 1e9:.2f} GB "
              f"(reserved {torch.cuda.memory_reserved() / 1e9:.2f} GB)", flush=True)

    start_step = 0
    os.makedirs(CKPT_DIR, exist_ok=True)
    if RESUME_FROM_CHECKPOINT:
        all_pt = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt")]
        fim_big = [f for f in all_pt if f.startswith("step_big_fim_")]
        base_big = [f for f in all_pt if f.startswith("step_big_") and not f.startswith("step_big_fim_")]
        big_ckpts = fim_big if FIM_MODE else base_big
        if not big_ckpts:
            big_ckpts = base_big
        big_ckpts = sorted(big_ckpts, key=_step_from_name)
        normal_ckpts = sorted([f for f in all_pt if f.startswith("step_") and not f.startswith("step_big_")], key=_step_from_name)
        if big_ckpts:
            latest = os.path.join(CKPT_DIR, big_ckpts[-1])
            print(f"Loading big checkpoint: {latest}", flush=True)
            ckpt = _load_ckpt(latest)
            sd = ckpt["model"]
            ckpt_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
            if ckpt_layers != N_LAYERS:
                raise RuntimeError(f"big checkpoint {latest} has {ckpt_layers} layers, expected {N_LAYERS}")
            ckpt_vocab = sd["tok_emb.weight"].shape[0]
            if ckpt_vocab != VOCAB_SIZE:
                if not VOCAB_RESIZE_ON_RESUME:
                    raise RuntimeError(f"checkpoint {latest} has vocab {ckpt_vocab}, expected {VOCAB_SIZE}")
                print(f"Resizing vocab {ckpt_vocab} -> {VOCAB_SIZE}", flush=True)
                resize_vocab_embeddings(model, sd, ckpt_vocab)
                VOCAB_RESIZED = True
                _load_optimizer_state(optimizer, ckpt, model, old_vocab=ckpt_vocab)
            else:
                model.load_state_dict(sd)
                if "optimizer" in ckpt:
                    _load_optimizer_state(optimizer, ckpt, model)
                else:
                    UPSCALED = True
            start_step = ckpt["step"] + 1
            print(f"Resuming from step {start_step}")
        elif normal_ckpts:
            latest = os.path.join(CKPT_DIR, normal_ckpts[-1])
            print(f"Loading normal checkpoint: {latest}", flush=True)
            ckpt = _load_ckpt(latest)
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
                    _load_optimizer_state(optimizer, ckpt, model)
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
    wandb.init(project="locLMM-FIM" if FIM_MODE else "locLMM", config={
        "vocab_size": VOCAB_SIZE, "block_size": BLOCK_SIZE, "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM, "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "dim": DIM, "n_layers": N_LAYERS, "n_heads": N_HEADS, "upscaled_from": OLD_N_LAYERS,
        "wakeup_steps": WAKEUP_STEPS, "wakeup_lr": WAKEUP_LR,
        "max_lr": MAX_LR, "min_lr": MIN_LR, "max_steps": MAX_STEPS,
        "lr_decay_steps": LR_DECAY_STEPS, "params": n_params,
        "optimizer": OPTIMIZER,
    })

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
            hidden = model(x, return_hidden=True)  # (B, T, DIM) — ~33 MB at T=4096, ~66 MB at T=8192
            if os.environ.get("LOCLLM_DEBUG_MEM") == "1":
                print(f"  [mem] B={x.shape[0]} T={x.shape[1]} "
                      f"alloc={torch.cuda.memory_allocated() / 1e9:.2f} GB "
                      f"reserved={torch.cuda.memory_reserved() / 1e9:.2f} GB", flush=True)
            hidden = hidden.to(MODEL_DTYPE)        # final_norm promotes to fp32 under autocast; normalize
            seq_len = hidden.shape[1]

            def _head_ce(hchunk, ychunk):
                logits = model.lm_head(hchunk)
                return F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    ychunk.reshape(-1),
                    reduction="sum",
                )

            losses = []
            counts = []
            for s in range(0, seq_len, LOSS_CHUNK):
                e = min(s + LOSS_CHUNK, seq_len)
                yc = y[:, s:e]
                losses.append(torch.utils.checkpoint.checkpoint(
                    _head_ce, hidden[:, s:e], yc, use_reentrant=False))
                counts.append(int((yc != -100).sum()))
            loss = sum(losses) / sum(counts)
        loss_val = loss.item()
        if not math.isfinite(loss_val) or loss_val > LOSS_SKIP_THRESHOLD:
            # corrupted/degenerate batch: don't backprop or accumulate it
            print(f"  skipped micro-step: loss {loss_val:.2f} (threshold {LOSS_SKIP_THRESHOLD})")
            del hidden, loss, x, y
            return 0.0, 0.0, 0.0, 0, {}
        scaler.scale(loss / GRAD_ACCUM).backward()
        with torch.no_grad():
            mask = y != -100
            n_tok = mask.sum().item()
            correct = 0
            per_row = torch.zeros_like(y, dtype=torch.float32)
            # Metrics from the first chunk only: saves a redundant full lm_head pass
            s = 0
            e = min(LOSS_CHUNK, seq_len)
            logits_c = model.lm_head(hidden[:, s:e])
            yc = y[:, s:e]
            mc = mask[:, s:e]
            correct += (logits_c.argmax(dim=-1)[mc] == yc[mc]).sum().item()
            pr = F.cross_entropy(
                logits_c.view(-1, logits_c.size(-1)), yc.reshape(-1), reduction="none",
            ).view_as(yc)
            per_row[:, s:e] = pr
            acc = correct / max(n_tok, 1)
            row_real = mask.sum(dim=-1).float()
            row_counts = row_real.clamp(min=1)
            row_loss = per_row.sum(dim=-1) / row_counts
            cat_stats = {}
            for j, cat in enumerate(cats):
                key = f"fim/{cat}" if fim_flags[j] else f"lm/{cat}"
                wl, nt = cat_stats.get(key, (0.0, 0))
                rc = int(row_real[j].item())
                cat_stats[key] = (wl + row_loss[j].item() * rc, nt + rc)
        del hidden, loss, mask, per_row, x, y
        return loss_val, math.exp(min(loss_val, 20)), acc, n_tok, cat_stats

    eval_set = []
    fim_eval_set = []

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
                if len(tokens) >= MIN_SAMPLE_TOKENS:
                    eval_set.append((cat, tokens[:BLOCK_SIZE]))
        print(f"Cached eval set: {len(eval_set)} samples (fixed for this run)")

    @torch.no_grad()
    def run_eval(step: int):
        if not eval_set:
            return
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        step_chunk = min(4, max(1, len(eval_set) // 2))  # bound logits memory at 8192
        for s0 in range(0, len(eval_set), step_chunk):
            batch = eval_set[s0:s0 + step_chunk]
            x = torch.full((len(batch), BLOCK_SIZE), 0, dtype=torch.long, device=DEVICE)
            y = torch.full((len(batch), BLOCK_SIZE), -100, dtype=torch.long, device=DEVICE)
            for i, (cat, tokens) in enumerate(batch):
                headers, ends = _CHATML.analyze(tokens)
                if headers:
                    toks = chatml.replace_markers(tokens, headers, ends, CHATML_IDS)
                    mt = torch.tensor(chatml.mask_from_ids(toks, CHATML_IDS)[1:], dtype=torch.bool)
                else:
                    toks = tokens
                    mt = None
                seq = torch.tensor(toks, dtype=torch.long, device=DEVICE)
                n = min(len(seq) - 1, BLOCK_SIZE)
                x[i, :n] = seq[:n]
                y[i, :n] = seq[1:n + 1]
                if mt is not None:
                    y[i, :n] = torch.where(
                        mt[:n].to(seq.device), seq[1:n + 1],
                        torch.full_like(seq[1:n + 1], -100))
            with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE, enabled=(DEVICE == "cuda")):
                hidden = model(x, return_hidden=True)
                hidden = hidden.to(MODEL_DTYPE)
                losses = []
                for c in range(0, hidden.shape[1], LOSS_CHUNK):
                    logits_c = model.lm_head(hidden[:, c:c + LOSS_CHUNK])
                    losses.append(F.cross_entropy(
                        logits_c.reshape(-1, logits_c.size(-1)),
                        y[:, c:c + LOSS_CHUNK].reshape(-1),
                        reduction="sum"))
                loss = sum(losses)
            total_loss += loss.item()
            total_tokens += int((y != -100).sum().item())
            del hidden, loss, x, y
        model.train()
        val = total_loss / max(total_tokens, 1)
        print(f"  eval @ step {step}: loss {val:.4f} | ppl {math.exp(min(val, 20)):.1f}")
        wandb.log({"eval_loss": val, "eval_ppl": math.exp(min(val, 20))}, step=step)

    def build_fim_eval_set(n: int = FIM_EVAL_SAMPLES):
        """Deterministic FIM eval set: fixed split + optional RAG context, built once per run."""
        fim_eval_set.clear()
        # Reserve context room ONLY when RAG is actually enabled; otherwise the
        # FIM eval samples use the FULL window like training does.
        fim_cap = (BLOCK_SIZE - CONTEXT_MAX_TOKENS - 6 - LANG_OVERHEAD
                   if (FIM_MODE and RAG_TRAIN_MODE)
                   else BLOCK_SIZE - 3 - LANG_OVERHEAD)
        attempts = 0
        while len(fim_eval_set) < n and attempts < 6:
            attempts += 1
            try:
                fresh = get_next_samples(n - len(fim_eval_set) + 8)
            except RuntimeError:
                break
            if not fresh:
                break
            for cat, tokens in fresh:
                if len(fim_eval_set) >= n:
                    break
                if cat not in CODE_CATEGORY_IDS or len(tokens) < 200:
                    continue
                tokens = tokens[:fim_cap]
                L = len(tokens)
                pre_end = _snap_newline(tokens, max(32, L // 2))
                mid_end = _snap_newline(tokens, pre_end + max(16, min(L // 4, L - pre_end - 1)))
                if mid_end <= pre_end or mid_end >= L:
                    continue
                if FIM_MODE and RAG_TRAIN_MODE:
                    query = _fim_query(tokens, pre_end, mid_end)
                    record = search_rag(query) if query else None
                    context = _make_context(tokens, record, BLOCK_SIZE)
                else:
                    context = None
                fim_eval_set.append((cat, tokens, pre_end, mid_end, context))
        print(f"Cached FIM eval set: {len(fim_eval_set)} samples (fixed for this run)")

    @torch.no_grad()
    def run_fim_eval(step: int):
        """FIM loss/ppl on the fixed eval set (context + lang tags masked out)."""
        if not fim_eval_set:
            return
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        step_chunk = min(4, len(fim_eval_set))  # bound logits memory at 8192
        for s0 in range(0, len(fim_eval_set), step_chunk):
            batch = fim_eval_set[s0:s0 + step_chunk]
            prepared = []
            for cat, tokens, pre_end, mid_end, context in batch:
                seq = torch.tensor(tokens, dtype=torch.long)
                vseq, vmt = _fim_variant(seq, pre_end, mid_end, context,
                                         CAT_NAME_BY_ID.get(cat))
                prepared.append((vseq, vmt))
            n_max = max(len(s) - 1 for s, _ in prepared)
            x = torch.zeros((len(prepared), n_max), dtype=torch.long, device=DEVICE)
            y = torch.full((len(prepared), n_max), -100, dtype=torch.long, device=DEVICE)
            for i, (seq, mt) in enumerate(prepared):
                n = min(len(seq) - 1, n_max)
                x[i, :n] = seq[:n]
                y[i, :n] = torch.where(mt[:n], seq[1:n + 1],
                                       torch.full_like(seq[1:n + 1], -100))
            with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE, enabled=(DEVICE == "cuda")):
                hidden = model(x, return_hidden=True)
                hidden = hidden.to(MODEL_DTYPE)
                losses = []
                for c in range(0, hidden.shape[1], LOSS_CHUNK):
                    logits_c = model.lm_head(hidden[:, c:c + LOSS_CHUNK])
                    losses.append(F.cross_entropy(
                        logits_c.reshape(-1, logits_c.size(-1)),
                        y[:, c:c + LOSS_CHUNK].reshape(-1),
                        reduction="sum"))
                loss = sum(losses)
            total_loss += loss.item()
            total_tokens += int((y != -100).sum().item())
            del hidden, loss, x, y
        model.train()
        val = total_loss / max(total_tokens, 1)
        print(f"  fim eval @ step {step}: loss {val:.4f} | ppl {math.exp(min(val, 20)):.1f}")
        wandb.log({"eval_fim_loss": val, "eval_fim_ppl": math.exp(min(val, 20))}, step=step)

    @torch.no_grad()
    def run_fim_gen_eval(step: int):
        """Greedy FIM generation vs reference middle: prefix-match accuracy + exact-match."""
        if not fim_eval_set or DEVICE == "cpu":
            return
        model.eval()
        accs = 0.0
        exacts = 0
        count = 0
        for cat, tokens, pre_end, mid_end, context in fim_eval_set[:FIM_EVAL_GEN_SAMPLES]:
            seq = torch.tensor(tokens, dtype=torch.long)
            vseq, _ = _fim_variant(seq, pre_end, mid_end, context, CAT_NAME_BY_ID.get(cat))
            mid_idx = int((vseq == FIM_MID).nonzero()[0].item())
            prompt = vseq[:mid_idx + 1].unsqueeze(0).to(DEVICE)
            ref = vseq[mid_idx + 1:-1].tolist()  # middle (without <fim_end>)
            gen = model.generate(prompt, max_new_tokens=FIM_EVAL_GEN_TOKENS,
                                 temperature=0.0,
                                 stop_tokens={FIM_END, CONTEXT_END, IM_END})
            gen_ids = gen[0, prompt.shape[1]:].tolist()
            k = min(len(gen_ids), len(ref), FIM_EVAL_GEN_TOKENS)
            if k > 0:
                pref = 0
                for j in range(k):
                    if gen_ids[j] == ref[j]:
                        pref += 1
                    else:
                        break
                accs += pref / k
                if gen_ids[:k] == ref[:k]:
                    exacts += 1
                count += 1
        model.train()
        if count:
            print(f"  fim gen eval @ step {step}: prefix_acc {accs / count:.3f} | "
                  f"exact@{FIM_EVAL_GEN_TOKENS} {exacts / count:.2f}")
            wandb.log({"eval_fim_gen_prefix_acc": accs / count,
                       "eval_fim_gen_exact": exacts / count}, step=step)

    build_eval_set()
    build_fim_eval_set()
    torch.cuda.empty_cache()

    first_eval_done = False

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

        cur_loss = accum_loss_w / max(accum_tokens, 1)
        cur_ppl = math.exp(min(cur_loss, 20))
        cur_acc = accum_acc / max(accum_tokens, 1)

        lr = get_lr(step, start_step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if not math.isfinite(cur_loss):
            print(f"WARNING: step {step}: non-finite loss {cur_loss} — zeroing grads, skipping step")
            optimizer.zero_grad(set_to_none=True)
            if USE_SCALER:
                scaler.update()
            continue

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

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
            stats = _drain_stats()
            if stats["fim_eligible"] > 0:
                wandb.log({
                    "fim/trunc_rate": stats["fim_trunc"] / stats["fim_eligible"],
                    "fim/len_lt1k": stats["hist_1k"], "fim/len_1_2k": stats["hist_2k"],
                    "fim/len_2_4k": stats["hist_4k"], "fim/len_ge4k": stats["hist_8k"],
                    "rag/query_count": stats["rag_q"],
                    "rag/miss_rate": stats["rag_miss"] / max(1, stats["rag_q"]),
                    "rag/context_len": stats["ctx_len"] / max(1, stats["ctx_n"]),
                    "rag/context_clip_rate": stats["ctx_clip"] / max(1, stats["ctx_n"]),
                }, step=step)
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
            if FIM_MODE:
                run_fim_checkpoint_sample(step, model, sp, BLOCK_SIZE, DEVICE)
            else:
                run_checkpoint_sample(step, model, sp, BLOCK_SIZE, DEVICE)
            torch.cuda.empty_cache()

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
                    if f.endswith(".pt") and f.startswith("step_") and not f.startswith("step_big_"):
                        os.remove(os.path.join(CKPT_DIR, f))
                        print(f"removed superseded normal checkpoint: {f}")

        if step > 0 and (step % EVAL_EVERY == 0 or (not first_eval_done and step >= start_step + EVAL_FIRST_AFTER)):
            run_eval(step)
            run_fim_eval(step)
            run_fim_gen_eval(step)
            first_eval_done = True
            torch.cuda.empty_cache()
