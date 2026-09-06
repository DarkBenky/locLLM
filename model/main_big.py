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
from synth_eval_cases import SYNTH_EVAL_CASES, load_eval_items, save_eval_items

API = "http://91.98.145.193:8823"
FIM_API = "http://91.98.145.193:8823"
FIM_SEARCH_API = "http://91.98.145.193:8234/search"
CONTEXT_MAX_TOKENS = 1024

TOKENIZER_MODEL_PATH = "../tok/tokenize/tokenizer_models/tokenizer.model"

BLOCK_SIZE = int(os.environ.get("LOCLLM_BLOCK_SIZE", "8192"))
BATCH_SIZE = 2
GRAD_ACCUM = 6  # effective batch = BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM (2*8192*6 ≈ 98k tokens/step)
# Fixed token normalizer for the backward pass (FIX.md D1): every micro-step
# gradient is scaled by 1/TOKEN_NORM instead of 1/GRAD_ACCUM, so token-rich
# micro-batches contribute proportionally to their supervised token count.
TOKEN_NORM = float(os.environ.get("LOCLLM_TOKEN_NORM", "12000"))
# FIX.md 10: dynamic accumulation — keep adding micro-steps until the supervised
# token target is met; MAX_MICRO caps the worst case (tiny/skipped micros).
TOKEN_TARGET = int(os.environ.get("LOCLLM_TOKEN_TARGET", str(int(TOKEN_NORM))))
MAX_MICRO = int(os.environ.get("LOCLLM_MAX_MICRO", str(GRAD_ACCUM)))
# z-loss coefficient (PaLM/Chinchilla) for bf16 logit stability (FIX.md 17).
Z_LOSS_COEF = float(os.environ.get("LOCLLM_Z_LOSS", "1e-4"))
LOSS_SKIP_THRESHOLD = float(os.environ.get("LOCLLM_SKIP_TOK", "5.0"))  # PER-TOKEN loss above this = corrupted micro-step -> skip it
MIN_SAMPLE_TOKENS = int(os.environ.get("LOCLLM_MIN_SAMPLE_TOKENS", "256"))  # drop short samples (skip-micro diag 2026-09-04: 64 -> 256; revert via env)
DIM = int(os.environ.get("LOCLLM_DIM", "1024"))
N_LAYERS = int(os.environ.get("LOCLLM_N_LAYERS", "128"))
OLD_N_LAYERS = 26
N_HEADS = int(os.environ.get("LOCLLM_N_HEADS", "16"))
PRUNED = False  # set when resuming a depth-pruned checkpoint (FIX.md 36)

# Non-destructive FFN widening: target hidden dim = DIM * ratio (3584 at 3.5x).
# Override with LOCLLM_FFN_RATIO (e.g. 3.0 -> 3072, ~1.73B, tighter on VRAM).
# Trained weights are kept; only the newly added FFN rows/cols are fresh.
FFN_EXPAND_RATIO = float(os.environ.get("LOCLLM_FFN_RATIO", "3.5"))
NEW_FFN_HIDDEN = int(DIM * FFN_EXPAND_RATIO)

RESUME_FROM_CHECKPOINT = True
UPSCALE_ON_RESUME = True
VOCAB_RESIZE_ON_RESUME = True
KEEP_CHECKPOINTS_COUNT = 1
RANDOM_SAMPLING = True

MAX_STEPS = 750_000
WARMUP_STEPS = 300
WAKEUP_STEPS = 1500  # wake-up phase (LR burst + freeze) after a layer upscale
VOCAB_WAKEUP_STEPS = 1500  # wake-up phase when only the vocab was resized
WAKEUP_LR = 1e-4
MAX_LR = 1e-4
MIN_LR = 1e-5
LR_DECAY_STEPS = MAX_STEPS  # decay spans the full MAX_STEPS horizon (was 250k -> LR floor for the last half)
WEIGHT_DECAY = 0.1
GRAD_CLIP = float(os.environ.get("LOCLLM_GRAD_CLIP", "3.0"))  # pre-clip grad_norm p90 ~4.5 in logs
CHATML_MASK_PROB = 0.8
# FIM/NTP mixture (grounding fix): 60% FIM infill, 40% classical next-token
# prediction. Plain LM teaches exact left-to-right continuation (variable usage,
# control flow) that FIM-only training with masked prefix/suffix never sees.
FIM_RATIO = float(os.environ.get("LOCLLM_FIM_RATIO", "0.6"))
# Classical-NTP samples get a <lang>...</lang> header so language conditioning
# is learned in both modes (matches the FIM prompt format). 0 disables.
LM_HEADER = os.environ.get("LOCLLM_LM_HEADER", "1") != "0"
FIM_VARIANTS = 1  # FIM samples generated per code sample
FIM_MAX_SAMPLE_TOKENS = 0  # 0 = FULL window (up to BLOCK_SIZE=8192) per sample:
                           # maximum context for every sample / matches 8K.
                           # Set e.g. 1536/4096 to cap windows and go faster.
NO_CONTEXT_PROB = 0.5  # RAG-only knob: 50/50 with/without context when LOCLLM_RAG_TRAIN=1, unused in plain FIM mode
# FIM middle-span mixture (FIX.md Phase 3): ~80% spans of 32-512 tokens,
# ~20% longer spans up to LONG_MID_MAX. The old 75% line-level bias starved
# per-step supervision (sup tok/step swung 200-30k).
SHORT_MID_CUM = 0.8     # cumulative prob for the 32..512-token span class
LONG_MID_MAX = 2048     # cap for the long-span class
ROPE_BASE = 10000.0  # test e.g. 100000 for longer extrapolation at 8192
LOSS_CHUNK = 1024  # sequence-chunk size for head+loss (bounds logits memory;
                    # 1024 keeps peak logits ~0.4GB at B=8; 2048 is faster but +~0.4GB)

LOG_EVERY = 10
CKPT_EVERY = 125
EVAL_EVERY = 125
EVAL_FIRST_AFTER = 10  # run one early eval after N steps (fail-fast sanity check)
EVAL_SAMPLES = 64
FIM_EVAL_SAMPLES = 32
FIM_EVAL_GEN_SAMPLES = 32
FIM_EVAL_GEN_TOKENS = 32
CKPT_DIR = os.environ.get("LOCLLM_CKPT_DIR", "./checkpoints")

# --- deterministic eval instrumentation --------------------------------------
# Eval-set paths; the synthetic cases + persist helpers live in
# model/synth_eval_cases.py (kept out of this file on purpose).
EVAL_LM_PATH = os.path.join(CKPT_DIR, "eval_set_lm.json")
EVAL_FIM_PATH = os.path.join(CKPT_DIR, "eval_set_fim.json")

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
FFN_WIDENED = False
CKPT_PREFIX = "step_big_"


def _select_gpu_interactive():
    # Server/headless-friendly: bypass the interactive selector entirely.
    #   LOCLLM_GPU=<cuda-index> LOCLLM_BATCH_SIZE=3 LOCLLM_GRAD_ACCUM=10 \
    #   LOCLLM_OPTIMIZER=8bit python main_big.py
    if "LOCLLM_GPU" in os.environ:
        gpu_idx = int(os.environ["LOCLLM_GPU"])
        return [{
            "name": os.environ.get("LOCLLM_GPU_NAME", f"GPU {gpu_idx}"),
            "vram_size": float(os.environ.get("LOCLLM_GPU_VRAM", "0.0")),
            "index": gpu_idx,
            "batch_size": int(os.environ.get("LOCLLM_BATCH_SIZE", BATCH_SIZE)),
            "accumulation_steps": int(os.environ.get("LOCLLM_GRAD_ACCUM", GRAD_ACCUM)),
            "optimizer": os.environ.get("LOCLLM_OPTIMIZER", "fp32"),
        }]
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
    # Headless/server-friendly: LOCLLM_FIM=1 (or true/yes) enables FIM mode
    # without the interactive question (which crashes on py3.8 questionary).
    if "LOCLLM_FIM" in os.environ:
        FIM_MODE = os.environ.get("LOCLLM_FIM").lower() not in ("0", "false", "no")
        print(f"FIM mode from env: {FIM_MODE}")
        return
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


def expand_ffn(model, old_sd, old_hidden):
    """Non-destructive FFN widening. Old w_gate/w_up rows and w_down columns keep
    their trained values; new w_gate/w_up rows keep the model's fresh init; new
    w_down columns are ZEROED so the residual stream (and the loss) is unchanged
    at step 0 — output-neutral by construction."""
    new_sd = model.state_dict()
    for k, v in old_sd.items():
        if k not in new_sd or v.ndim != new_sd[k].ndim:
            continue
        if k.endswith(".ffn.w_gate.weight") or k.endswith(".ffn.w_up.weight"):
            new_sd[k][:old_hidden] = v
        elif k.endswith(".ffn.w_down.weight"):
            new_sd[k][:, :old_hidden] = v
        elif new_sd[k].shape == v.shape:
            new_sd[k] = v
    for k in list(new_sd):
        if k.endswith(".ffn.w_down.weight"):
            new_sd[k][:, old_hidden:] = 0
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
            return ("state1" in v or "qmap1" in v
                    or "__bnb_optimizer_quant_state__" in v)
    return False


def _splice_optimizer_for_arch(old_sd, optimizer, model, tied_emb):
    """Rebuild a saved optimizer state dict for the current architecture
    (FIX.md 23/24). Old checkpoints: tied lm_head (not a separate param) and no
    LayerScale. New: lm_head appended to the decay group, ls_attn/ls_ffn in the
    no-decay group (2 new params per block).

    bnb's load_state_dict maps saved state keys to the CURRENT params by
    POSITION (zip over the group param lists) and validates group sizes, so we
    rewrite both the group "params" lists (new sizes) and the state keys
    (new global indices). New LayerScale params get no state entry — bnb
    initializes them lazily on the first step. lm_head clones tok_emb's state:
    the weights are identical at step 0, so the copied moments are exactly right.
    """
    import copy as _copy
    old_state = old_sd.get("state", {})
    decay = optimizer.param_groups[0]["params"]
    no_decay = optimizer.param_groups[1]["params"] if len(optimizer.param_groups) > 1 else []
    n_decay = len(decay)
    n_no_decay = len(no_decay)
    n_old_decay = n_decay - 1  # lm_head is the appended param

    name_of = {p: n for n, p in model.named_parameters()}
    old_decay_names = [name_of[p] for p in decay][:n_old_decay]
    old_no_decay_names = [name_of[p] for p in no_decay
                          if not (name_of[p].endswith("ls_attn") or name_of[p].endswith("ls_ffn"))]
    if len(old_state) != n_old_decay + len(old_no_decay_names):
        return old_sd  # unexpected old architecture — let the loader fail/fallback

    new_state = {}
    for i in range(n_old_decay):
        if i in old_state:
            new_state[i] = old_state[i]
    if tied_emb and 0 in old_state:
        new_state[n_decay - 1] = _copy.deepcopy(old_state[0])  # lm_head <- tok_emb
    old_base = n_old_decay  # first no-decay param's old global index
    for j, p in enumerate(no_decay):
        nm = name_of[p]
        if nm.endswith("ls_attn") or nm.endswith("ls_ffn"):
            continue  # new LayerScale params — left uninitialized
        old_rel = old_no_decay_names.index(nm)
        if old_base + old_rel in old_state:
            new_state[old_base + j] = old_state[old_base + old_rel]
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


def _load_optimizer_state(optimizer, ckpt, model, old_vocab=None, tied_emb=False) -> bool:
    if "optimizer" not in ckpt:
        return False
    old_sd = ckpt["optimizer"]
    want_8bit = _is_8bit_optimizer(optimizer)
    if _opt_state_is_8bit(old_sd) != want_8bit:
        print("  WARNING: saved optimizer state (8-bit vs fp32) does not match the "
              "selected optimizer — starting optimizer fresh (model weights unchanged)")
        return False
    try:
        n_new = sum(len(g["params"]) for g in optimizer.param_groups)
        if len(old_sd.get("state", {})) != n_new:
            old_sd = _splice_optimizer_for_arch(old_sd, optimizer, model, tied_emb)
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
POOL_MIN = 512  # was 128 — a bigger pool is needed for real length/category mixing (FIX.md 14)
BATCH_QUEUE_MAX = 4

_sample_pool: list[tuple[int, list[int]]] = []
_batch_queue = queue.Queue(maxsize=BATCH_QUEUE_MAX)
_prefetch_thread: "threading.Thread | None" = None

# --- lightweight instrumentation counters (read by the trainer loop) ---
_stats_lock = threading.Lock()
_stats = {
    "fim_eligible": 0,      # FIM-eligible code samples planned
    "lm_eligible": 0,       # classical-NTP code samples planned (grounding mix)
    "fim_trunc": 0,         # ... truncated to fim_cap
    "hist_1k": 0, "hist_2k": 0, "hist_4k": 0, "hist_8k": 0,  # pre-truncation length buckets
    "rag_q": 0,             # RAG queries issued
    "rag_miss": 0,          # ... that produced no context
    "ctx_n": 0, "ctx_len": 0, "ctx_clip": 0,  # context built / tokens used / clipped at cap
    "skip_micro": 0,        # micro-steps skipped (corrupted batches)
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
    """Middle-span mixture: ~80% spans of 32-512 tokens, ~20% longer (up to 2048).
    Replaces the old line-level bias that starved per-step supervision."""
    if mid_max <= 1:
        return 1
    r = random.random()
    if r < SHORT_MID_CUM:
        return random.randint(min(32, mid_max), min(512, mid_max))
    return random.randint(min(256, mid_max), min(LONG_MID_MAX, mid_max))


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
    if is_code and not do_fim and len(tokens) >= MIN_SAMPLE_TOKENS:
        _stat("lm_eligible")  # classical-NTP grounding mix
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
            # Skip-micro diagnosis (2026-09-04): do NOT re-add the truncated tail
            # to _sample_pool. Tail re-addition turns the pool into a slow-draining
            # reservoir where degenerate long samples (star_coder redactions,
            # archive/SEO filler, foreign prose) accumulate and re-circulate — the
            # source of the ~27% skip rate. Same model + fresh API data -> 0/200
            # skips (diag_skip_micro.py). Truncate and discard the tail.
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
            # FIX.md follow-up: classical NTP grounding — prepend <lang>name
            # </lang> so LM samples learn language conditioning too. Only for
            # code categories; chatml-marked samples keep their masked form.
            if LM_HEADER and cat_id is not None and cat_id in CODE_CATEGORY_IDS:
                lb = lang_block_ids(CAT_NAME_BY_ID.get(cat_id))
                if lb:
                    toks = lb + toks
                    if len(toks) > block_size:
                        toks = toks[:block_size]
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
    for cat, tokens in _sample_pool:
        if len(tokens) > block_size + 1:
            # tail discarded (same rationale as _plan_fim — no pool reservoir)
            tokens = tokens[:block_size + 1]
        if len(tokens) >= MIN_SAMPLE_TOKENS:
            trimmed.append((cat, tokens))

    random.shuffle(trimmed)
    # FIX.md 13/14: round-robin selection across length buckets and categories.
    # The old sort(key=len) + contiguous window made every batch length-uniform
    # (the regime swings). Draw one sample per length bucket per round, and
    # prefer categories not yet seen in this batch (stratification).
    buckets: dict[int, list] = {}
    for item in trimmed:
        buckets.setdefault(len(item[1]) // 1024, []).append(item)
    for lst in buckets.values():
        random.shuffle(lst)
    chosen = []
    while len(chosen) < batch_size and any(buckets.values()):
        seen_cats = {c for c, _ in chosen}
        progressed = False
        for key in sorted(buckets.keys()):
            lst = buckets[key]
            if not lst:
                continue
            idx = next((i for i, (c, _) in enumerate(lst) if c not in seen_cats), 0)
            chosen.append(lst.pop(idx))
            seen_cats.add(chosen[-1][0])
            progressed = True
            if len(chosen) >= batch_size:
                break
        if not progressed:
            break
    _sample_pool = [item for lst in buckets.values() for item in lst]

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


def _abs_lr(step: int) -> float:
    """Absolute cosine schedule (based on the global step count)."""
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step >= LR_DECAY_STEPS:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / (LR_DECAY_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


def get_lr(step: int, step0: int = 0) -> float:
    # Wake-up phase: only active after an actual structural change (layer
    # upscale, vocab resize or FFN widen, flagged UPSCALED/VOCAB_RESIZED/
    # FFN_WIDENED). A normal resume of a regular checkpoint leaves all flags
    # False -> pure absolute schedule.
    s_rel = step - step0
    wake = VOCAB_WAKEUP_STEPS if VOCAB_RESIZED else WAKEUP_STEPS
    if (UPSCALED or VOCAB_RESIZED or FFN_WIDENED) and s_rel < wake:
        lr = WAKEUP_LR * min((s_rel + 1) / WARMUP_STEPS, 1.0)
        # never exceed the scheduled LR at this step: prevents a surprise 1e-4
        # burst on a mid-training resume from disturbing already-trained weights
        return min(lr, _abs_lr(step))

    return max(0.0, _abs_lr(step))


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
    BATCH_SIZE = int(os.environ.get("LOCLLM_BATCH_SIZE", gpu.get("batch_size", BATCH_SIZE)))
    GRAD_ACCUM = int(os.environ.get("LOCLLM_GRAD_ACCUM", gpu.get("accumulation_steps", GRAD_ACCUM)))
    if "LOCLLM_BATCH_SIZE" in os.environ or "LOCLLM_GRAD_ACCUM" in os.environ:
        print(f"NOTE: env overrides active -> batch_size={BATCH_SIZE} accum={GRAD_ACCUM}")
    OPTIMIZER = gpu.get("optimizer", "fp32")
    print(f"\nTraining on: {gpu['name']} ({gpu.get('vram_size', 0):.1f} GB) | "
          f"batch_size={BATCH_SIZE} | accum={GRAD_ACCUM} | optimizer={OPTIMIZER} | "
          f"context batch ≈ {BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM:,} tokens/step "
          f"(SUPERVISED tokens/step are far fewer — see the 'sup' column in the log)")

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
    CAT_ID_BY_NAME = {name: cid for cid, name in CAT_NAME_BY_ID.items()}

    # Adopt the FFN width of the latest migrated checkpoint when no explicit
    # ratio is given (LOCLLM_FFN_RATIO overrides). Prevents the "checkpoint is
    # wider than this run's model" crash on resume.
    _ffn_meta = os.path.join(CKPT_DIR, "ffn_hidden.json")
    if "LOCLLM_FFN_RATIO" not in os.environ and os.path.exists(_ffn_meta):
        with open(_ffn_meta) as f:
            _meta = json.load(f)
        NEW_FFN_HIDDEN = int(_meta["ffn_hidden"])
        FFN_EXPAND_RATIO = float(_meta.get("ratio", NEW_FFN_HIDDEN / DIM))
        print(f"FFN width from checkpoint meta: {NEW_FFN_HIDDEN} "
              f"(set LOCLLM_FFN_RATIO to override)", flush=True)

    samples = get_next_samples(1)
    if samples:
        _, tokens = samples[0]
        print(tokens[:20])
        print(sp.decode(tokens)[:200])

    print(f"building model ({N_LAYERS} layers, dim {DIM}, FFN {NEW_FFN_HIDDEN} hidden) ...", flush=True)
    model = Transformer(vocab_size=VOCAB_SIZE, dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS,
                         max_seq_len=BLOCK_SIZE, rope_base=ROPE_BASE,
                         ffn_hidden=NEW_FFN_HIDDEN).to(DEVICE)
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
            lr=MAX_LR, betas=(0.9, 0.98),
        )
        print("Using 8-bit AdamW optimizer (bitsandbytes)", flush=True)
    else:
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": WEIGHT_DECAY},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=MAX_LR, betas=(0.9, 0.98),
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
            # tied-embedding checkpoints may lack the duplicate lm_head key
            sd.setdefault("lm_head.weight", sd["tok_emb.weight"])
            tied_emb = torch.equal(sd["tok_emb.weight"], sd["lm_head.weight"])
            ckpt_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
            if ckpt_layers != N_LAYERS:
                raise RuntimeError(f"big checkpoint {latest} has {ckpt_layers} layers, expected {N_LAYERS}")
            ckpt_vocab = sd["tok_emb.weight"].shape[0]
            ckpt_ffn = sd["blocks.0.ffn.w_gate.weight"].shape[0]
            migrated = False
            if ckpt_vocab != VOCAB_SIZE:
                if not VOCAB_RESIZE_ON_RESUME:
                    raise RuntimeError(f"checkpoint {latest} has vocab {ckpt_vocab}, expected {VOCAB_SIZE}")
                print(f"Resizing vocab {ckpt_vocab} -> {VOCAB_SIZE}", flush=True)
                resize_vocab_embeddings(model, sd, ckpt_vocab)
                VOCAB_RESIZED = True
                migrated = True
            if ckpt_ffn != NEW_FFN_HIDDEN:
                if ckpt_ffn > NEW_FFN_HIDDEN:
                    raise RuntimeError(
                        f"checkpoint {latest} FFN {ckpt_ffn} is WIDER than this run's target "
                        f"{NEW_FFN_HIDDEN} — set LOCLLM_FFN_RATIO={ckpt_ffn / DIM} "
                        f"(or unset it to adopt the checkpoint width)")
                print(f"Widening FFN {ckpt_ffn} -> {NEW_FFN_HIDDEN} (zero-init w_down, output-neutral)",
                      flush=True)
                expand_ffn(model, sd, ckpt_ffn)
                FFN_WIDENED = True
                migrated = True
            if not migrated:
                # strict=False: LayerScale params keep their init of 1.0
                # (exactly function-preserving, FIX.md 24)
                model.load_state_dict(sd, strict=False)
                if "optimizer" in ckpt:
                    _load_optimizer_state(optimizer, ckpt, model, tied_emb=tied_emb)
                else:
                    print("  no optimizer state in checkpoint — starting optimizer fresh", flush=True)
            elif VOCAB_RESIZED and not FFN_WIDENED:
                # vocab-only resize: splice the optimizer state for the new rows
                _load_optimizer_state(optimizer, ckpt, model, old_vocab=ckpt_vocab, tied_emb=tied_emb)
            else:
                # FFN widen (or both): optimizer state shapes no longer match ->
                # start the optimizer fresh. Safe: zero-init keeps the output
                # identical at step 0.
                print("  optimizer restarted fresh (FFN shapes changed)", flush=True)
            start_step = ckpt["step"] + 1
            if ckpt.get("pruned_from") or ckpt.get("widened_from"):
                PRUNED = True
                info = []
                if ckpt.get("pruned_from"):
                    info.append(f"pruned {ckpt['pruned_from']}->{N_LAYERS} layers")
                if ckpt.get("widened_from"):
                    info.append(f"widened dim {ckpt['widened_from']}->{DIM}")
                print(f"Structural checkpoint detected ({', '.join(info)}) "
                      f"— wake-up heal active for {WAKEUP_STEPS} steps", flush=True)
            if migrated:
                big_path = f"{CKPT_DIR}/{CKPT_PREFIX}{start_step - 1}_widen{NEW_FFN_HIDDEN}.pt"
                torch.save({"model": model.state_dict(), "step": start_step - 1}, big_path)
                os.replace(latest, latest + ".orig")
                with open(os.path.join(CKPT_DIR, "ffn_hidden.json"), "w") as f:
                    json.dump({"ffn_hidden": NEW_FFN_HIDDEN, "ratio": FFN_EXPAND_RATIO}, f)
                print(f"Saved migrated big checkpoint: {big_path}")
                print(f"Original kept as backup: {latest}.orig")
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
                sd.setdefault("lm_head.weight", sd["tok_emb.weight"])
                tied_emb = torch.equal(sd["tok_emb.weight"], sd["lm_head.weight"])
                model.load_state_dict(sd, strict=False)
                if "optimizer" in ckpt:
                    _load_optimizer_state(optimizer, ckpt, model, tied_emb=tied_emb)
                start_step = ckpt["step"] + 1
                print(f"Resuming from step {start_step}")
        else:
            print("No checkpoint found, starting from scratch")

    if UPSCALED:
        for i in range(0, 2 * OLD_N_LAYERS, 2):
            for p in model.blocks[i].parameters():
                p.requires_grad = False
        print(f"Wake-up phase: old blocks frozen for {WAKEUP_STEPS} steps")

    # saved optimizer state carries the old betas in its param groups — force
    # the new ones after any resume (FIX.md 15). Moments decay within ~50 steps.
    for _group in optimizer.param_groups:
        _group["betas"] = (0.9, 0.98)

    # LR resume check: a NORMAL resume (same layers, same vocab, optimizer state
    # present) must NOT enter the wake-up burst. Print what is active so a silent
    # LR restart can never masquerade as "no improvement".
    _wake_active = UPSCALED or VOCAB_RESIZED or FFN_WIDENED or PRUNED
    _wake_n = VOCAB_WAKEUP_STEPS if VOCAB_RESIZED else WAKEUP_STEPS
    _lr_at_resume = get_lr(start_step, start_step)
    if _wake_active:
        print(f"LR CHECK: WAKE-UP ACTIVE (upscaled={UPSCALED}, vocab_resized={VOCAB_RESIZED}, "
              f"ffn_widened={FFN_WIDENED}, pruned={PRUNED}) | "
              f"LR phase for {_wake_n} steps from {start_step} (until ~{start_step + _wake_n}) | "
              f"lr now {_lr_at_resume:.2e} (capped at scheduled LR)")
    else:
        print(f"LR CHECK: no wake-up burst on this resume (upscaled={UPSCALED}, "
              f"vocab_resized={VOCAB_RESIZED}, ffn_widened={FFN_WIDENED}, pruned={PRUNED}) | absolute cosine schedule | "
              f"lr @ {start_step} = {_lr_at_resume:.2e}")

    # FIX.md 18: weight-EMA (CPU, bf16), used ONLY for eval/export. Training
    # weights are untouched; evals run on the EMA copy for smoother curves.
    EMA_DECAY = float(os.environ.get("LOCLLM_EMA_DECAY", "0.999"))
    EMA_EVERY = int(os.environ.get("LOCLLM_EMA_EVERY", "100"))
    EMA_ENABLED = os.environ.get("LOCLLM_EMA", "1") != "0"
    ema_state = None

    def _ema_sync():
        global ema_state
        if not EMA_ENABLED:
            return
        if ema_state is None:
            ema_state = {n: p.data.detach().to("cpu", torch.bfloat16)
                         for n, p in model.named_parameters()}
        else:
            for n, p in model.named_parameters():
                ema_state[n].mul_(EMA_DECAY).add_(
                    p.data.detach().to("cpu", torch.bfloat16), alpha=1.0 - EMA_DECAY)

    def _ema_swap_in():
        """Swap EMA weights into the model; returns a stash to restore from."""
        if not ema_state:
            return None
        stash = {n: p.data.detach().to("cpu") for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            p.data.copy_(ema_state[n].to(p.device, p.dtype))
        return stash

    def _ema_restore(stash):
        if stash is None:
            return
        for n, p in model.named_parameters():
            p.data.copy_(stash[n].to(p.device, p.dtype))

    wandb.login()
    wandb.init(project="locLMM-FIM" if FIM_MODE else "locLMM", config={
        "vocab_size": VOCAB_SIZE, "block_size": BLOCK_SIZE, "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM, "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "dim": DIM, "n_layers": N_LAYERS, "n_heads": N_HEADS, "upscaled_from": OLD_N_LAYERS,
        "ffn_hidden": NEW_FFN_HIDDEN, "ffn_expand_ratio": FFN_EXPAND_RATIO,
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
                maskc = ychunk != -100
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    ychunk.reshape(-1),
                    reduction="sum",
                )
                correct = int((logits.argmax(dim=-1)[maskc] == ychunk[maskc]).sum())
                per_tok = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), ychunk.reshape(-1), reduction="none",
                ).float().view_as(ychunk)
                row_loss = torch.where(maskc, per_tok, torch.zeros_like(per_tok)).sum(dim=-1)
                row_cnt = maskc.sum(dim=-1)
                zsum = torch.logsumexp(logits.float(), dim=-1).pow(2).sum()
                return loss, correct, row_loss, row_cnt, zsum

            loss_sum = torch.zeros((), device=x.device, dtype=torch.float32)
            z_sum = torch.zeros((), device=x.device, dtype=torch.float32)
            n_tok = 0
            correct = 0
            row_loss_acc = None
            row_cnt_acc = None
            for s in range(0, seq_len, LOSS_CHUNK):
                e = min(s + LOSS_CHUNK, seq_len)
                yc = y[:, s:e]
                # FIX.md H2: reentrant checkpointing for the head chunks.
                # Non-reentrant checkpointing retains every tensor saved during
                # each chunk's recompute (frame.recomputed) until the graph
                # dies: the fp32 logits/logsumexp/per_tok copies of 8 chunks
                # are ~13-17 GB at B=8, held through the whole transformer
                # backward (measured: 3.2 GB at B=2, gone with reentrant).
                # Reentrant has no retention mechanism and _head_ce is a small
                # dropout-free function, so it is safe and costs the same.
                l_, c_, rl_, rc_, z_ = torch.utils.checkpoint.checkpoint(
                    _head_ce, hidden[:, s:e], yc, use_reentrant=True)
                loss_sum = loss_sum + l_
                z_sum = z_sum + z_
                n_tok += int((yc != -100).sum())
                correct += c_
                row_loss_acc = rl_ if row_loss_acc is None else row_loss_acc + rl_
                row_cnt_acc = rc_ if row_cnt_acc is None else row_cnt_acc + rc_
            loss = (loss_sum + Z_LOSS_COEF * z_sum) / max(n_tok, 1)
        loss_val = loss.item()
        if not math.isfinite(loss_val) or loss_val > LOSS_SKIP_THRESHOLD:
            # corrupted/degenerate batch: don't backprop or accumulate it
            _stat("skip_micro")
            _cat_names = [CAT_NAME_BY_ID.get(c, str(c)) for c in cats]
            if row_loss_acc is not None and row_cnt_acc is not None:
                _row_means = [f"{float(w.detach() / max(rc.detach(), 1)):.1f}"
                              for w, rc in zip(row_loss_acc, row_cnt_acc)]
            else:
                _row_means = ["?"] * len(cats)
            print(f"  skipped micro-step: loss {loss_val:.2f} n_tok {n_tok} "
                  f"row_losses [{','.join(_row_means)}] cats {_cat_names} "
                  f"(per-token threshold {LOSS_SKIP_THRESHOLD})")
            del hidden, loss, x, y
            return 0.0, 0.0, 0.0, 0, {}
        # FIX.md D1: fixed token normalizer — micro-batches contribute to the
        # gradient in proportion to their supervised token count.
        scaler.scale((loss_sum + Z_LOSS_COEF * z_sum) / TOKEN_NORM).backward()
        with torch.no_grad():
            acc = correct / max(n_tok, 1)
            row_cnt_safe = row_cnt_acc.clamp(min=1)
            row_loss_mean = row_loss_acc / row_cnt_safe
            cat_stats = {}
            for j, cat in enumerate(cats):
                key = f"fim/{cat}" if fim_flags[j] else f"lm/{cat}"
                wl, nt = cat_stats.get(key, (0.0, 0))
                rc = float(row_cnt_acc[j].item())
                cat_stats[key] = (wl + row_loss_acc[j].item(), nt + rc)
        del hidden, loss, x, y
        return loss_val, math.exp(min(loss_val, 20)), acc, n_tok, cat_stats

    eval_set = []
    fim_eval_set = []

    def _synth_lm_eval_items():
        """Synthetic eval cases (go/c/python/opencl) as (cat_id, tokens)."""
        items = []
        for lang, code in SYNTH_EVAL_CASES:
            cid = CAT_ID_BY_NAME.get(normalize_lang(lang))
            if cid is None:
                print(f"  WARNING: synth eval lang '{lang}' not in category index — skipping")
                continue
            tokens = sp.encode(code, out_type=int)
            if len(tokens) >= MIN_SAMPLE_TOKENS:
                items.append((cid, tokens[:BLOCK_SIZE]))
        return items

    def build_eval_set(n: int = EVAL_SAMPLES):
        """Deterministic LM eval set: synthetic + persisted server samples.
        Identical on every resume (previously fresh random samples per run)."""
        eval_set.clear()
        synth = _synth_lm_eval_items()
        eval_set.extend(synth)
        cached = load_eval_items(EVAL_LM_PATH)
        if cached is None:
            items = []
            need = max(0, n - len(eval_set))
            attempts = 0
            while len(items) < need and attempts < 5:
                attempts += 1
                try:
                    fresh = get_next_samples(need - len(items))
                except RuntimeError:
                    break
                if not fresh:
                    break
                for cat, tokens in fresh:
                    if len(tokens) >= MIN_SAMPLE_TOKENS:
                        items.append((cat, tokens[:BLOCK_SIZE]))
            save_eval_items(EVAL_LM_PATH, [{"cat": c, "tokens": list(t)} for c, t in items])
            cached = [{"cat": c, "tokens": list(t)} for c, t in items]
        eval_set.extend((d["cat"], list(d["tokens"])) for d in cached)
        print(f"Eval set: {len(eval_set)} samples ({len(synth)} synthetic + {len(cached)} cached server) "
              f"— persistent: {EVAL_LM_PATH}")

    def _fim_eval_split(tokens, fim_cap):
        """Deterministic FIM split for an eval sample, or None if unusable."""
        tokens = tokens[:fim_cap]
        L = len(tokens)
        if L < 200:
            return None
        pre_end = _snap_newline(tokens, max(32, L // 2))
        mid_end = _snap_newline(tokens, pre_end + max(16, min(L // 4, L - pre_end - 1)))
        if mid_end <= pre_end or mid_end >= L:
            return None
        return tokens, pre_end, mid_end

    def _fim_eval_item(cat, tokens, pre_end, mid_end, lang):
        """Build the (cat, tokens, pre_end, mid_end, context, lang) eval item."""
        context = None
        if FIM_MODE and RAG_TRAIN_MODE:
            query = _fim_query(tokens, pre_end, mid_end)
            record = search_rag(query) if query else None
            context = _make_context(tokens, record, BLOCK_SIZE)
        return (cat, tokens, pre_end, mid_end, context, lang)

    @torch.no_grad()
    def run_eval(step: int):
        if not eval_set:
            return
        model.eval()
        stash = _ema_swap_in()  # FIX.md 18: eval on the EMA weights
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
        _ema_restore(stash)
        model.train()
        val = total_loss / max(total_tokens, 1)
        print(f"  eval @ step {step}: loss {val:.4f} | ppl {math.exp(min(val, 20)):.1f}")
        wandb.log({"eval_loss": val, "eval_ppl": math.exp(min(val, 20))}, step=step)

    def build_fim_eval_set(n: int = FIM_EVAL_SAMPLES):
        """Fixed FIM eval set: synthetic + persisted server samples, same on every resume."""
        fim_eval_set.clear()
        # Reserve context room ONLY when RAG is actually enabled; otherwise the
        # FIM eval samples use the FULL window like training does.
        fim_cap = (BLOCK_SIZE - CONTEXT_MAX_TOKENS - 6 - LANG_OVERHEAD
                   if (FIM_MODE and RAG_TRAIN_MODE)
                   else BLOCK_SIZE - 3 - LANG_OVERHEAD)
        # 1) synthetic cases first — always present, fully deterministic
        for lang, code in SYNTH_EVAL_CASES:
            cid = CAT_ID_BY_NAME.get(normalize_lang(lang))
            if cid is None:
                print(f"  WARNING: synth eval lang '{lang}' not in category index — skipping")
                continue
            out = _fim_eval_split(sp.encode(code, out_type=int), fim_cap)
            if out is None:
                continue
            tokens, pre_end, mid_end = out
            fim_eval_set.append(_fim_eval_item(cid, tokens, pre_end, mid_end, lang))
        # 2) server-derived samples — fetched once, cached to disk.
        # A too-small legacy cache (n=8 quantizes the gen metrics to 0.125
        # steps) is discarded and rebuilt with a stratified fetch.
        cached = load_eval_items(EVAL_FIM_PATH)
        if cached is not None and len(cached) < 16:
            cached = None
        if cached is None:
            items = []
            need = max(0, n - len(fim_eval_set))
            lang_counts = {}
            attempts = 0
            while len(items) < need and attempts < 40:
                attempts += 1
                try:
                    fresh = get_next_samples(need - len(items) + 16)
                except RuntimeError:
                    break
                if not fresh:
                    break
                for cat, tokens in fresh:
                    if len(items) >= need:
                        break
                    if cat not in CODE_CATEGORY_IDS or len(tokens) < 200:
                        continue
                    name = CAT_NAME_BY_ID.get(cat)
                    # language stratification: soft cap per language so the set
                    # is not dominated by one language of the data stream
                    if lang_counts.get(name, 0) >= max(3, need // 8):
                        continue
                    out = _fim_eval_split(tokens, fim_cap)
                    if out is None:
                        continue
                    tok, pre_end, mid_end = out
                    items.append((cat, tok, pre_end, mid_end, name))
                    lang_counts[name] = lang_counts.get(name, 0) + 1
            save_eval_items(EVAL_FIM_PATH, [
                {"cat": c, "tokens": list(t), "pre_end": p, "mid_end": m, "lang": lg}
                for c, t, p, m, lg in items])
            cached = [{"cat": c, "tokens": list(t), "pre_end": p, "mid_end": m, "lang": lg}
                      for c, t, p, m, lg in items]
        for d in cached:
            fim_eval_set.append(_fim_eval_item(
                d["cat"], list(d["tokens"]), int(d["pre_end"]), int(d["mid_end"]), d.get("lang")))
        print(f"FIM eval set: {len(fim_eval_set)} samples — persistent: {EVAL_FIM_PATH}")

    @torch.no_grad()
    def run_fim_eval(step: int):
        """FIM loss/ppl on the fixed eval set (context + lang tags masked out)."""
        if not fim_eval_set:
            return
        model.eval()
        stash = _ema_swap_in()  # FIX.md 18
        total_loss = 0.0
        total_tokens = 0
        step_chunk = min(4, len(fim_eval_set))  # bound logits memory at 8192
        for s0 in range(0, len(fim_eval_set), step_chunk):
            batch = fim_eval_set[s0:s0 + step_chunk]
            prepared = []
            for cat, tokens, pre_end, mid_end, context, lang in batch:
                seq = torch.tensor(tokens, dtype=torch.long)
                sample_lang = lang or CAT_NAME_BY_ID.get(cat)
                vseq, vmt = _fim_variant(seq, pre_end, mid_end, context, sample_lang)
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
        _ema_restore(stash)
        model.train()
        val = total_loss / max(total_tokens, 1)
        print(f"  fim eval @ step {step}: loss {val:.4f} | ppl {math.exp(min(val, 20)):.1f}")
        wandb.log({"eval_fim_loss": val, "eval_fim_ppl": math.exp(min(val, 20))}, step=step)

    @torch.no_grad()
    def run_fim_gen_eval(step: int):
        """FIM middle-completion quality (FIX.md 21):
        - teacher-forced middle top-1 (batched, span-only: padding excluded)
        - greedy generation: prefix_acc, exact@k, first-line exact match,
          token edit similarity (SAFIM-style ES), micro + per-sample macro
        prefix_acc is de-emphasized in favor of the sharper metrics."""
        if not fim_eval_set or DEVICE == "cpu":
            return
        model.eval()
        stash = _ema_swap_in()
        items = fim_eval_set[:FIM_EVAL_GEN_SAMPLES]

        def _lev(a, b):
            if not a or not b:
                return max(len(a), len(b))
            prev = list(range(len(b) + 1))
            for i, x in enumerate(a, 1):
                cur = [i]
                for j, y in enumerate(b, 1):
                    cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
                prev = cur
            return prev[-1]

        def _first_line(ids):
            line = []
            for t in ids:
                if t in NEWLINE_IDS:
                    break
                line.append(t)
            return line

        # ---- teacher-forced middle top-1 (batched) ----
        # Only SUPERVISED positions count (the middle span, masked exactly like
        # training). The old scan scored every position up to the batch's padded
        # width, so ~89% of the denominator was zero-padding (constant) — that
        # froze top1_tf at 0.278 while real quality moved.
        top1_correct = top1_total = 0
        step_chunk = min(4, len(items))
        for s0 in range(0, len(items), step_chunk):
            batch = items[s0:s0 + step_chunk]
            prepared = []
            masks = []
            for cat, tokens, pre_end, mid_end, context, lang in batch:
                seq = torch.tensor(tokens, dtype=torch.long)
                vseq, mt = _fim_variant(seq, pre_end, mid_end, context,
                                        lang or CAT_NAME_BY_ID.get(cat))
                prepared.append(vseq)
                masks.append(mt)
            n_max = max(len(s) - 1 for s in prepared)
            x = torch.zeros((len(prepared), n_max), dtype=torch.long, device=DEVICE)
            for i, s in enumerate(prepared):
                n = min(len(s) - 1, n_max)
                x[i, :n] = s[:n]
            with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE, enabled=(DEVICE == "cuda")):
                hidden = model(x, return_hidden=True)
                hidden = hidden.to(MODEL_DTYPE)
                for c in range(0, hidden.shape[1], LOSS_CHUNK):
                    logits_c = model.lm_head(hidden[:, c:c + LOSS_CHUNK])  # (B, chunk, V)
                    preds = logits_c.argmax(dim=-1)
                    for i in range(len(prepared)):
                        n = len(prepared[i]) - 1
                        for j in range(c, min(c + LOSS_CHUNK, n_max - 1)):
                            if j + 1 >= n or not masks[i][j]:
                                continue  # padding / prefix / suffix / markers
                            tgt = x[i, j + 1].item()
                            if tgt == FIM_END:
                                continue
                            top1_total += 1
                            if preds[i, j - c] == tgt:
                                top1_correct += 1
            del hidden, logits_c, x
        top1_tf = top1_correct / max(top1_total, 1)

        # ---- greedy generation vs reference middle ----
        accs = 0.0
        exacts = 0
        fl_exacts = 0
        es_sum = 0.0
        count = 0
        # per-sample (macro) arrays — with a small eval set the token-weighted
        # micro aggregates quantize to 1/n steps; macro means move smoothly.
        pref_list, es_list, fl_list, exact_list = [], [], [], []
        for cat, tokens, pre_end, mid_end, context, lang in items:
            seq = torch.tensor(tokens, dtype=torch.long)
            sample_lang = lang or CAT_NAME_BY_ID.get(cat)
            vseq, _ = _fim_variant(seq, pre_end, mid_end, context, sample_lang)
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
                pref_r = pref / k
                es_i = 1.0 - _lev(gen_ids[:k], ref[:k]) / max(k, 1)
                fl_i = 1.0 if _first_line(gen_ids) == _first_line(ref) else 0.0
                ex_i = 1.0 if gen_ids[:k] == ref[:k] else 0.0
                accs += pref_r
                es_sum += es_i
                exacts += ex_i
                fl_exacts += fl_i
                pref_list.append(pref_r)
                es_list.append(es_i)
                fl_list.append(fl_i)
                exact_list.append(ex_i)
                count += 1
        _ema_restore(stash)
        model.train()
        if count:
            es = es_sum / count
            macro_exact = sum(exact_list) / count
            macro_fl = sum(fl_list) / count
            macro_es = sum(es_list) / count
            print(f"  fim gen eval @ step {step}: top1_tf {top1_tf:.3f} (span-only) | prefix_acc {accs / count:.3f} | "
                  f"exact@{FIM_EVAL_GEN_TOKENS} {exacts / count:.2f} (macro {macro_exact:.2f}) | "
                  f"first-line {fl_exacts / count:.2f} (macro {macro_fl:.2f}) | "
                  f"edit-sim {es:.3f} (macro {macro_es:.3f}) | n={count}")
            wandb.log({"eval_fim_gen_top1_tf": top1_tf,
                       "eval_fim_gen_prefix_acc": accs / count,
                       "eval_fim_gen_exact": exacts / count,
                       "eval_fim_gen_firstline": fl_exacts / count,
                       "eval_fim_gen_edit_sim": es,
                       "eval_fim_gen_exact_macro": macro_exact,
                       "eval_fim_gen_firstline_macro": macro_fl,
                       "eval_fim_gen_edit_sim_macro": macro_es,
                       "eval_fim_gen_n": count}, step=step)

    build_eval_set()
    build_fim_eval_set()
    torch.cuda.empty_cache()

    if os.environ.get("LOCLLM_EVAL_ONLY") == "1":
        # Measurement mode: run the fixed eval sets on the loaded checkpoint
        # BEFORE any training step (no optimizer perturbation).
        run_eval(start_step)
        run_fim_eval(start_step)
        run_fim_gen_eval(start_step)
        torch.cuda.empty_cache()
        print("EVAL-ONLY: exiting before training (LOCLLM_EVAL_ONLY=1)")
        raise SystemExit(0)

    first_eval_done = False
    win_sup, win_loss = [], []

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
        micro_n = 0
        cat_stats_accum = {}

        # FIX.md 10: dynamic accumulation — stop once the supervised token
        # target is met (token-rich micros end the step early), capped by
        # MAX_MICRO so tiny/skipped micros can't stall the step.
        for micro in range(MAX_MICRO):
            l, _, a, n, cat_stats = _micro_step()
            if n == 0:
                continue
            micro_n += 1
            accum_loss_w += l * n
            accum_acc += a * n
            accum_tokens += n
            for key, (wl, nt) in cat_stats.items():
                aw, at = cat_stats_accum.get(key, (0.0, 0))
                cat_stats_accum[key] = (aw + wl, at + nt)
            if accum_tokens >= TOKEN_TARGET:
                break

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

        if EMA_ENABLED and (step - start_step) % EMA_EVERY == 0:
            _ema_sync()  # FIX.md 18: CPU-side EMA, eval/export only

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
        win_sup.append(tokens_seen)
        win_loss.append(cur_loss)

        if step % LOG_EVERY == 0:
            cat_metrics = {}
            for key, (wl, nt) in cat_stats_accum.items():
                if nt > 0:
                    cat_metrics[f"loss/{key}"] = wl / nt
                cat_metrics[f"tokens/{key}"] = nt
            stats = _drain_stats()
            n_fim = stats.get("fim_eligible", 0)
            n_lm = stats.get("lm_eligible", 0)
            n_mix = n_fim + n_lm
            wandb.log({"train/micro_steps": micro_n,
                       "train/accum_tokens": accum_tokens,
                       "train/ntp_samples": n_lm,
                       "train/fim_samples": n_fim,
                       "train/ntp_ratio": (n_lm / n_mix) if n_mix else 0.0}, step=step)
            if win_sup:
                w_sup_mean = sum(win_sup) / len(win_sup)
                w_sup_std = (sum((s - w_sup_mean) ** 2 for s in win_sup) / len(win_sup)) ** 0.5
                w_loss_mean = sum(win_loss) / len(win_loss)
                w_loss_std = (sum((l - w_loss_mean) ** 2 for l in win_loss) / len(win_loss)) ** 0.5
                print(f"  window(n={len(win_sup)}): sup {w_sup_mean:.0f} +/- {w_sup_std:.0f} "
                      f"[{min(win_sup)}..{max(win_sup)}] tok/step | loss {w_loss_mean:.3f} +/- {w_loss_std:.3f} | "
                      f"skipped {stats['skip_micro']}")
                wandb.log({
                    "window/sup_mean": w_sup_mean, "window/sup_std": w_sup_std,
                    "window/sup_min": min(win_sup), "window/sup_max": max(win_sup),
                    "window/loss_mean": w_loss_mean, "window/loss_std": w_loss_std,
                    "skip/micro_steps": stats["skip_micro"],
                }, step=step)
                win_sup.clear()
                win_loss.clear()
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
                  f"sup {tokens_seen} tok/step | "
                  f"ema_loss {ema_loss:.4f} | ema_ppl {ema_ppl:.1f} | "
                  f"ema_acc {ema_acc:.3f} | ema_grad {ema_grad:.2f}")
            wandb.log({"loss": cur_loss, "ppl": cur_ppl, "acc": cur_acc, "lr": lr,
                        "grad_norm": cur_grad,
                        "tok_per_sec": tok_per_sec, "tokens_seen": tokens_seen,
                        "sup_tokens_per_step": tokens_seen,
                        "ema_loss": ema_loss, "ema_ppl": ema_ppl, "ema_acc": ema_acc,
                        "ema_grad": ema_grad, **cat_metrics}, step=step)

        if step > 0 and step % CKPT_EVERY == 0:
            ckpt_path = f"{CKPT_DIR}/{CKPT_PREFIX}{step}.pt"
            # LOCLLM_OPT_EVERY=4 -> optimizer state saved only every 4th ckpt
            # (halves most checkpoint files: model-only ~6.4GB vs 12.9GB).
            _opt_every = int(os.environ.get("LOCLLM_OPT_EVERY", "1"))
            _save_opt = _opt_every <= 1 or (step // CKPT_EVERY) % _opt_every == 0
            ckpt_data = {"model": model.state_dict(), "step": step}
            if _save_opt:
                ckpt_data["optimizer"] = optimizer.state_dict()
            torch.save(ckpt_data, ckpt_path)
            print(f"saved checkpoint: {ckpt_path}" + ("" if _save_opt else " (model only)"))
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
            if os.environ.get("LOCLLM_STOP_AFTER_FIRST_EVAL") == "1":
                print("TEST MODE: stopping after first eval (LOCLLM_STOP_AFTER_FIRST_EVAL=1)")
                break
