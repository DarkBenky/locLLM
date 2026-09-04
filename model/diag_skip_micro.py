#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_skip_micro.py — diagnose "skipped micro-step" micro-batches in main_big.py FIM training.

Hypothesis under test
=====================
The ~27% of micro-batches skipped by LOSS_SKIP_THRESHOLD=5.0 are degenerate
DATA (star_coder <NAME>/<reponame> redaction artifacts, archive/SEO filler,
foreign prose, tiny samples), not a model/optimizer problem. Expected signature:
  * skipped per-token losses form a tight band ~7.5-10.5 (kept batches ~1.0)
  * skipped batches have more redaction markers + higher non-ASCII ratio
  * skips are uncorrelated with supervised-token count / batch loss
  * skips cluster in specific categories

What it does
============
  Phase 1 (data-only, no GPU): scan the live sample pool + assemble micro-batches
    with the EXACT training pipeline (main_big._build_batch_from_pool) and log
    per-row content stats (category, length, redaction markers, non-ASCII).
  Phase 2 (model, GPU): load the latest checkpoint (same migration logic as
    main()), run the EXACT forward+loss computation of _micro_step (chunked head
    CE + z-loss, bf16 autocast, NO backward, no optimizer) on N micro-batches,
    and classify each with the exact skip rule. Full per-row detail is dumped
    for every skipped batch. Includes a synthetic-case smoke check to verify
    the weights loaded correctly (should print loss ~1.2-1.4, like eval).

Outputs (--out dir, default ./diag_skip_out):
  report.json             all aggregates (written at the end)
  batches.jsonl           one record per micro-batch (appended LIVE as it runs)
  skipped_detail.jsonl    full text snippets per skipped batch (appended LIVE)
  run.log                 full stdout of the run (tee'd from the start)

Usage (on the training box, same cwd as training):
  cd ~/locLLM/model
  python -u diag_skip_micro.py --batches 200                 # full diagnosis (~7-10 min)
  python -u diag_skip_micro.py --batches 100                 # quicker (~4-5 min)
  python -u diag_skip_micro.py --batches 200 --skip-phase1   # straight to model scan
  python -u diag_skip_micro.py --data-only --data-batches 40 # no GPU needed (~1 min)

Config env vars are read the same way as training (LOCLLM_FIM, LOCLLM_DIM,
LOCLLM_N_LAYERS, LOCLLM_N_HEADS, LOCLLM_FFN_RATIO, LOCLLM_BLOCK_SIZE,
LOCLLM_BATCH_SIZE). Defaults match the 4090 run (52L x dim1536 x 24H, B=7).
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Pre-import setup: must match the training run BEFORE main_big is imported.
# ---------------------------------------------------------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # cwd = model/ (tokenizer + ckpt paths are relative)
os.environ.setdefault("LOCLLM_FIM", "1")
os.environ.setdefault("LOCLLM_DIM", "1536")        # 4090 run: 52L x dim1536 x 24H x FFN9984
os.environ.setdefault("LOCLLM_N_LAYERS", "52")
os.environ.setdefault("LOCLLM_N_HEADS", "24")
os.environ.setdefault("LOCLLM_BLOCK_SIZE", "8192")
# LOCLLM_FFN_RATIO is resolved later from env > checkpoints/ffn_hidden.json > 6.5
# (mirrors main()); do not set it here.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("LOCLLM_GPU", "0"))

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

import chatml
import main_big as MB
from model import Transformer


# ---------------------------------------------------------------------------
# Decoding + content scanners
# ---------------------------------------------------------------------------

def _decode_extra() -> dict:
    base = MB.BASE_VOCAB
    ctx = chatml.context_ids(base)
    lang = chatml.lang_ids(base)
    return {
        base + 0: "", base + 1: "", base + 2: "", base + 3: "",   # FIM markers
        ctx["start"]: "<context_start>", ctx["end"]: "</context_end>",
        lang["open"]: "<lang>", lang["close"]: "</lang>",
    }


_EXTRA = None  # lazy


def safe_decode(tokens) -> str:
    global _EXTRA
    if _EXTRA is None:
        _EXTRA = _decode_extra()
    try:
        return chatml.safe_decode(list(tokens), MB.sp, chatml.reserved_ids(MB.BASE_VOCAB), extra=_EXTRA)
    except Exception:
        # fallback: drop virtual ids, plain decode
        ids = {MB.BASE_VOCAB, MB.BASE_VOCAB + 1, MB.BASE_VOCAB + 2, MB.BASE_VOCAB + 3}
        return MB.sp.decode([t for t in tokens if t < MB.BASE_VOCAB and t not in ids])


_LANG_BLOCK_RE = re.compile(r"</?lang>")
_HTML_TAG_RE = re.compile(
    r"<(?:p|div|span|a|li|ul|ol|td|tr|th|table|h[1-6]|html|head|body|meta|"
    r"script|style|title|br|img|href)\b", re.I)
_GEN_TAG_RE = re.compile(r"<([A-Za-z][A-Za-z0-9_.\-]{0,20})>")
_NAME_RE = re.compile(r"<name>", re.I)
_REPONAME_RE = re.compile(r"<reponame>", re.I)


def scan_text(text: str) -> dict:
    """Redaction/filler signals in a decoded string."""
    text = _LANG_BLOCK_RE.sub("", text)  # strip <lang> blocks (LM headers)
    n = len(text)
    non_ascii = (sum(1 for c in text if ord(c) > 127) / n) if n else 0.0
    html_hits = len(_HTML_TAG_RE.findall(text))
    generic = len(_GEN_TAG_RE.findall(text))
    name_hits = len(_NAME_RE.findall(text))
    repo_hits = len(_REPONAME_RE.findall(text))
    other = max(0, generic - html_hits - name_hits - repo_hits)
    return {
        "chars": n,
        "non_ascii": non_ascii,
        "html_hits": html_hits,
        "name_hits": name_hits,
        "repo_hits": repo_hits,
        "other_tag_hits": other,
        "any_hit": (html_hits + name_hits + repo_hits + other) > 0,
    }


# ---------------------------------------------------------------------------
# Small stats helpers
# ---------------------------------------------------------------------------

def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(q * len(xs))))
    return xs[k]


def pearson(a, b):
    n = len(a)
    if n < 2:
        return None
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if va == 0.0 or vb == 0.0:
        return None
    return cov / (va * vb)


# ---------------------------------------------------------------------------
# Config resolution (mirrors main())
# ---------------------------------------------------------------------------

def resolve_ffn_hidden() -> int:
    if "LOCLLM_FFN_RATIO" in os.environ:
        return int(MB.DIM * float(os.environ["LOCLLM_FFN_RATIO"]))
    meta = os.path.join(MB.CKPT_DIR, "ffn_hidden.json")
    if os.path.exists(meta):
        try:
            with open(meta) as f:
                return int(json.load(f)["ffn_hidden"])
        except Exception:
            pass
    return int(MB.DIM * 6.5)


def find_latest_ckpt():
    if not os.path.isdir(MB.CKPT_DIR):
        return None
    pts = [f for f in os.listdir(MB.CKPT_DIR) if f.endswith(".pt")]
    big = sorted([f for f in pts if f.startswith("step_big_")], key=MB._step_from_name)
    return os.path.join(MB.CKPT_DIR, big[-1]) if big else None


# ---------------------------------------------------------------------------
# Per-row analysis of an assembled micro-batch (x, y already built by
# _build_batch_from_pool, exactly like training sees them)
# ---------------------------------------------------------------------------

def analyze_rows(x, y, cats, fim_flags, row_loss=None, row_cnt=None, name_by_id=None):
    rows = []
    for i in range(x.shape[0]):
        mask = y[i] != -100
        n_tok = int(mask.sum())
        sup_ids = x[i][mask].tolist()
        text = safe_decode(sup_ids)
        s = scan_text(text)
        cat = int(cats[i]) if cats is not None else -1
        r = {
            "cat": cat,
            "cat_name": (name_by_id or MB.CAT_NAME_BY_ID).get(cat, f"cat{cat}"),
            "fim": bool(fim_flags[i]) if fim_flags is not None else False,
            "n_tok": n_tok,
            "chars": s["chars"],
            "name_hits": s["name_hits"],
            "repo_hits": s["repo_hits"],
            "html_hits": s["html_hits"],
            "other_tag_hits": s["other_tag_hits"],
            "any_hit": s["any_hit"],
            "non_ascii": round(s["non_ascii"], 4),
        }
        if row_loss is not None and row_cnt is not None:
            rc = max(float(row_cnt[i].item()), 1.0)
            r["row_loss"] = round(float(row_loss[i].item()) / rc, 3)
            r["loss_sum"] = round(float(row_loss[i].item()), 1)
        r["text_head"] = text[:600]
        r["text_tail"] = text[-300:] if len(text) > 600 else ""
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Model construction + checkpoint loading (same logic/order as main())
# ---------------------------------------------------------------------------

def build_and_load_model(ckpt_path: str, device: torch.device):
    # Load the checkpoint FIRST (mmap — cheap) so the actual width of the model
    # being diagnosed drives the in-memory architecture. The stale
    # checkpoints/ffn_hidden.json (old 6656 model) must never narrow the scan.
    ckpt = MB._load_ckpt(ckpt_path)
    sd = ckpt["model"]
    sd.setdefault("lm_head.weight", sd["tok_emb.weight"])
    ckpt_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    if ckpt_layers != MB.N_LAYERS:
        raise RuntimeError(
            f"checkpoint {ckpt_path} has {ckpt_layers} layers but LOCLLM_N_LAYERS={MB.N_LAYERS}. "
            f"Set LOCLLM_N_LAYERS={ckpt_layers} to match the training run.")
    ckpt_vocab = sd["tok_emb.weight"].shape[0]
    ckpt_ffn = sd["blocks.0.ffn.w_gate.weight"].shape[0]

    ffn_hidden = resolve_ffn_hidden()
    if ckpt_ffn > ffn_hidden:
        if "LOCLLM_FFN_RATIO" in os.environ:
            print(f"  note: LOCLLM_FFN_RATIO gives {ffn_hidden} but checkpoint is wider "
                  f"({ckpt_ffn}) — adopting checkpoint width (read-only)", flush=True)
        else:
            print(f"  note: ffn_hidden.json/meta says {ffn_hidden} but checkpoint is wider "
                  f"({ckpt_ffn}) — adopting checkpoint width (read-only)", flush=True)
        ffn_hidden = ckpt_ffn

    print(f"building model: {MB.N_LAYERS} layers | dim {MB.DIM} | heads {MB.N_HEADS} | "
          f"FFN {ffn_hidden} | vocab {MB.VOCAB_SIZE} | block {MB.BLOCK_SIZE}", flush=True)
    model = Transformer(vocab_size=MB.VOCAB_SIZE, dim=MB.DIM, n_layers=MB.N_LAYERS,
                        n_heads=MB.N_HEADS, max_seq_len=MB.BLOCK_SIZE,
                        rope_base=MB.ROPE_BASE, ffn_hidden=ffn_hidden).to(device)
    if MB.MODEL_DTYPE != torch.float32:
        model = model.to(MB.MODEL_DTYPE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params {n_params / 1e9:.2f}B | dtype {MB.MODEL_DTYPE} | device {device}", flush=True)

    if ckpt_vocab != MB.VOCAB_SIZE:
        print(f"  resizing vocab {ckpt_vocab} -> {MB.VOCAB_SIZE} (mean-init)", flush=True)
        MB.resize_vocab_embeddings(model, sd, ckpt_vocab)
    if ckpt_ffn != ffn_hidden:
        if ckpt_ffn > ffn_hidden:
            raise RuntimeError(
                f"checkpoint FFN {ckpt_ffn} is wider than target {ffn_hidden}. "
                f"Set LOCLLM_FFN_RATIO={ckpt_ffn / MB.DIM}.")
        print(f"  widening FFN {ckpt_ffn} -> {ffn_hidden} (output-neutral)", flush=True)
        MB.expand_ffn(model, sd, ckpt_ffn)
    else:
        model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"  loaded {os.path.basename(ckpt_path)} (saved step {ckpt.get('step')})", flush=True)
    return model, ffn_hidden, ckpt.get("step")


def compute_loss(model, x, y):
    """Forward + loss exactly like _micro_step (chunked head CE + z-loss,
    bf16 autocast on CUDA). No autograd, no checkpoint wrapper, no backward."""
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(x.device.type == "cuda")):
            hidden = model(x, return_hidden=True)
        hidden = hidden.to(MB.MODEL_DTYPE)
        seq_len = hidden.shape[1]
        loss_sum = torch.zeros((), device=x.device, dtype=torch.float32)
        z_sum = torch.zeros((), device=x.device, dtype=torch.float32)
        n_tok = 0
        row_loss_acc = None
        row_cnt_acc = None
        for s in range(0, seq_len, MB.LOSS_CHUNK):
            e = min(s + MB.LOSS_CHUNK, seq_len)
            logits = model.lm_head(hidden[:, s:e])
            yc = y[:, s:e]
            maskc = yc != -100
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   yc.reshape(-1), reduction="sum")
            per_tok = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                      yc.reshape(-1), reduction="none").float().view_as(yc)
            row_loss = torch.where(maskc, per_tok, torch.zeros_like(per_tok)).sum(dim=-1)
            row_cnt = maskc.sum(dim=-1)
            zsum = torch.logsumexp(logits.float(), dim=-1).pow(2).sum()
            loss_sum = loss_sum + loss
            z_sum = z_sum + zsum
            n_tok += int(maskc.sum())
            row_loss_acc = row_loss if row_loss_acc is None else row_loss_acc + row_loss
            row_cnt_acc = row_cnt if row_cnt_acc is None else row_cnt_acc + row_cnt
        loss_val = ((loss_sum + MB.Z_LOSS_COEF * z_sum) / max(n_tok, 1)).item()
    return loss_val, n_tok, row_loss_acc, row_cnt_acc


def smoke_check(model, device):
    """One synthetic LM eval sample: loss should be ~1.2-1.4 (training eval
    ~1.27). If it is ~10, the weights did not load (fresh model)."""
    try:
        lang, code = MB.SYNTH_EVAL_CASES[0]
        toks = MB.sp.encode(code, out_type=int)[:MB.BLOCK_SIZE]
        if len(toks) < 16:
            return None
        x = torch.tensor([toks], dtype=torch.long, device=device)
        y = torch.tensor([toks[1:] + [-100]], dtype=torch.long, device=device)
        loss_val, n_tok, _, _ = compute_loss(model, x, y)
        return loss_val
    except Exception as e:
        print(f"  WARNING: smoke check failed: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Fetch patch: training retries 300x silently — bad for an interactive diag
# ---------------------------------------------------------------------------

def patch_fetch_for_diag():
    """Training's fetch retries 300x silently — fine for a trainer, terrible for
    an interactive diagnostic (nothing prints, Ctrl+C feels dead). Replace it
    with a visible, bounded version that fails fast."""
    MB.REQUEST_TIMEOUT = 15
    MB.MAX_REQUEST_RETRIES = 3

    def _diag_fetch(count, max_retries=8):
        delay = 2.0
        for attempt in range(max_retries):
            try:
                fresh = MB.get_next_samples(count)
            except RuntimeError as e:
                print(f"  [diag] fetch failed (attempt {attempt + 1}/{max_retries}): {e}", flush=True)
            else:
                if fresh:
                    return fresh
                print(f"  [diag] API returned 0 samples (attempt {attempt + 1}/{max_retries}) "
                      f"— cursor may be exhausted", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 10.0)
        raise RuntimeError(f"[diag] no samples after {max_retries} attempts — is the data API up?")

    MB._fetch_samples_with_retry = _diag_fetch
    print(f"[diag] patched sample fetch: timeout {MB.REQUEST_TIMEOUT}s, "
          f"{MB.MAX_REQUEST_RETRIES} request retries, max 8 fetch attempts "
          f"(visible progress)", flush=True)


# ---------------------------------------------------------------------------
# Phase 1: data-only scan (no model, no GPU)
# ---------------------------------------------------------------------------

def phase1_data_scan(args, batch_size, name_by_id):
    print("\n" + "=" * 70, flush=True)
    print(f"PHASE 1 — data-only scan (pool {args.pool} samples, "
          f"{args.data_batches} assembled batches)", flush=True)
    print("=" * 70, flush=True)

    # 1a. raw pool composition
    t0 = time.time()
    fresh = MB._fetch_samples_with_retry(args.pool)
    lens = [len(t) for _, t in fresh]
    cat_counts = Counter(name_by_id.get(c, f"cat{c}") for c, _ in fresh)
    redacted = 0
    nonascii_fracs = []
    short64 = sum(1 for l in lens if l < 64)
    short512 = sum(1 for l in lens if l < 512)
    for _, toks in fresh:
        s = scan_text(safe_decode(toks))
        if s["any_hit"]:
            redacted += 1
        nonascii_fracs.append(s["non_ascii"])
    n = len(lens)
    print(f"pool scan: {n} samples fetched in {time.time() - t0:.0f}s", flush=True)
    print(f"  len: p10 {pct(lens, .10)} p50 {pct(lens, .50)} p90 {pct(lens, .90)} "
          f"min {min(lens) if lens else '-'} max {max(lens) if lens else '-'}", flush=True)
    print(f"  <64 tok: {short64 / n * 100:.1f}% | <512 tok: {short512 / n * 100:.1f}%", flush=True)
    print(f"  redaction-marker samples (any <tag> hit): {redacted / n * 100:.1f}%", flush=True)
    print(f"  non-ASCII ratio: mean {sum(nonascii_fracs) / n * 100:.2f}% | "
          f">5% in {sum(1 for f in nonascii_fracs if f > 0.05) / n * 100:.1f}%", flush=True)
    print(f"  top categories: {dict(cat_counts.most_common(12))}", flush=True)

    # 1b. assembled-batch stats (exact training pipeline)
    stats = {"n_batches": 0, "n_tok": [], "redacted_rows": 0, "total_rows": 0,
             "any_hit_rows": 0, "thin": 0, "cat_rows": Counter()}
    for i in range(args.data_batches):
        x, y, cats, fim_flags = MB._build_batch_from_pool(batch_size, MB.BLOCK_SIZE)
        if not cats:
            print(f"  batch {i}: empty (pool drained)", flush=True)
            continue
        rows = analyze_rows(x, y, cats, fim_flags, name_by_id=name_by_id)
        n_tok = sum(r["n_tok"] for r in rows)
        stats["n_batches"] += 1
        stats["n_tok"].append(n_tok)
        for r in rows:
            stats["total_rows"] += 1
            stats["any_hit_rows"] += int(r["any_hit"])
            stats["cat_rows"][(r["fim"], r["cat_name"])] += 1
        if n_tok < 400:
            stats["thin"] += 1
    if stats["n_batches"]:
        print(f"assembly scan: {stats['n_batches']} batches | rows/batch {batch_size}", flush=True)
        print(f"  sup tokens/batch: p10 {pct(stats['n_tok'], .10)} p50 {pct(stats['n_tok'], .50)} "
              f"p90 {pct(stats['n_tok'], .90)}", flush=True)
        print(f"  thin batches (<400 sup tok): {stats['thin'] / stats['n_batches'] * 100:.1f}%", flush=True)
        print(f"  rows with redaction markers: {stats['any_hit_rows'] / stats['total_rows'] * 100:.1f}%", flush=True)
        print(f"  row categories: {dict(stats['cat_rows'].most_common(12))}", flush=True)
    return {"pool": {"n": n, "len_p10": pct(lens, .10), "len_p50": pct(lens, .50),
                     "len_p90": pct(lens, .90),
                     "frac_lt64": short64 / n if n else None,
                     "frac_lt512": short512 / n if n else None,
                     "redacted_frac": redacted / n if n else None,
                     "nonascii_mean": sum(nonascii_fracs) / n if n else None,
                     "top_categories": dict(cat_counts.most_common(12))},
            "assemblies": {"n_batches": stats["n_batches"],
                           "n_tok_p10": pct(stats["n_tok"], .10),
                           "n_tok_p50": pct(stats["n_tok"], .50),
                           "n_tok_p90": pct(stats["n_tok"], .90),
                           "thin_frac": stats["thin"] / stats["n_batches"] if stats["n_batches"] else None,
                           "redacted_row_frac": (stats["any_hit_rows"] / stats["total_rows"]
                                                 if stats["total_rows"] else None),
                           "row_categories": dict(stats["cat_rows"].most_common(12))}}


# ---------------------------------------------------------------------------
# Phase 2: real-model loss scan with skip classification
# ---------------------------------------------------------------------------

def phase2_loss_scan(args, batch_size, name_by_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device available — phase 2 will run on CPU "
              "(very slow for a 3B model)", flush=True)
    batches_path = os.path.join(args.out, "batches.jsonl")
    skips_path = os.path.join(args.out, "skipped_detail.jsonl")

    ckpt_path = args.ckpt or find_latest_ckpt()
    if not ckpt_path:
        raise SystemExit("no step_big_*.pt checkpoint found in "
                         f"{MB.CKPT_DIR} — pass --ckpt explicitly")
    model, ffn_hidden, ckpt_step = build_and_load_model(ckpt_path, device)

    sm = smoke_check(model, device)
    if sm is not None:
        flag = "" if 0.3 < sm < 4.0 else "  <-- WEIGHTS LOOK WRONG (fresh/random model?)"
        print(f"  smoke check (synthetic {MB.SYNTH_EVAL_CASES[0][0]} sample): "
              f"loss {sm:.3f} (training eval ~1.27){flag}", flush=True)

    threshold = args.threshold if args.threshold is not None else MB.LOSS_SKIP_THRESHOLD
    print("\n" + "=" * 70, flush=True)
    print(f"PHASE 2 — loss scan: {args.batches} micro-batches | B={batch_size} "
          f"T={MB.BLOCK_SIZE} | skip threshold {threshold} (per-token)", flush=True)
    print("=" * 70, flush=True)

    records = []
    cat_rows = []  # (fim, cat_name, n_tok, loss_sum, skipped)
    t0 = time.time()
    try:
        for i in range(args.batches):
            x, y, cats, fim_flags = MB._build_batch_from_pool(batch_size, MB.BLOCK_SIZE)
            if not cats:
                print(f"  batch {i}: empty (pool drained) — excluded from stats", flush=True)
                continue
            n_masked = int((y != -100).sum())
            if n_masked == 0:
                print(f"  batch {i}: all-masked supervision — excluded from stats", flush=True)
                continue
            loss_val, n_tok, row_loss, row_cnt = compute_loss(
                model, x.to(device), y.to(device))
            skipped = (not math.isfinite(loss_val)) or (loss_val > threshold)
            rows = analyze_rows(x.cpu(), y.cpu(), cats, fim_flags,
                                row_loss.cpu(), row_cnt.cpu(), name_by_id=name_by_id)
            rec = {"batch": i, "loss": round(loss_val, 4), "n_tok": n_tok,
                   "skipped": bool(skipped), "rows": rows}
            records.append(rec)
            for r in rows:
                cat_rows.append((r["fim"], r["cat_name"], r["n_tok"],
                                 r.get("loss_sum", 0.0), bool(skipped)))
            # append live so a crash/kill never loses already-computed batches
            with open(batches_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if skipped:
                with open(skips_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            tag = "SKIP" if skipped else "keep"
            if skipped:
                hit = sum(1 for r in rows if r["any_hit"])
                na = max(r["non_ascii"] for r in rows)
                cc = Counter(r["cat_name"] for r in rows)
                top = ", ".join(f"{k}({v})" for k, v in cc.most_common(3))
                print(f"  [{i:4d}] {tag} | LOSS {loss_val:6.2f} | n_tok {n_tok:5d} | "
                      f"redacted rows {hit}/{len(rows)} | max non-ascii {na:.2f} | {top}",
                      flush=True)
            elif i % 25 == 0 or args.verbose:
                print(f"  [{i:4d}] {tag} | LOSS {loss_val:6.2f} | n_tok {n_tok:5d}", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted — saving partial results", flush=True)
    dt = time.time() - t0
    print(f"phase 2 done: {len(records)} valid batches in {dt:.0f}s "
          f"({dt / max(1, len(records)):.1f}s/batch)", flush=True)
    return records, cat_rows, {
        "ckpt": os.path.basename(ckpt_path), "ckpt_step": ckpt_step,
        "ffn_hidden": ffn_hidden, "threshold": threshold,
        "smoke_loss": (round(sm, 3) if sm is not None else None), "elapsed_s": round(dt, 1)}


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------

def build_report(args, batch_size, records, cat_rows, phase2_meta):
    valid = [r for r in records]
    skips = [r for r in valid if r["skipped"]]
    keeps = [r for r in valid if not r["skipped"]]
    n = len(valid)
    skip_rate = len(skips) / n if n else None

    skip_loss = [r["loss"] for r in skips]
    keep_loss = [r["loss"] for r in keeps]
    all_ntok = [r["n_tok"] for r in valid]

    # redaction / non-ascii: rows grouped by batch skip flag
    def row_stats(flag):
        rows = [rr for r in valid if r["skipped"] == flag for rr in r["rows"]]
        if not rows:
            return None
        return {
            "n_rows": len(rows),
            "any_hit_frac": sum(1 for rr in rows if rr["any_hit"]) / len(rows),
            "name_hits_per_row": sum(rr["name_hits"] for rr in rows) / len(rows),
            "repo_hits_per_row": sum(rr["repo_hits"] for rr in rows) / len(rows),
            "html_hits_per_row": sum(rr["html_hits"] for rr in rows) / len(rows),
            "non_ascii_mean": sum(rr["non_ascii"] for rr in rows) / len(rows),
            "n_tok_mean": sum(rr["n_tok"] for rr in rows) / len(rows),
        }

    # category table
    cat_tbl = defaultdict(lambda: [0, 0, 0, 0.0])  # n_rows, n_skipped_rows, n_tok, loss_sum
    for fim, name, n_tok, loss_sum, skipped in cat_rows:
        e = cat_tbl[(fim, name)]
        e[0] += 1
        e[1] += int(skipped)
        e[2] += n_tok
        e[3] += loss_sum
    cat_table = [{"fim": fim, "cat": name, "n_rows": e[0], "n_rows_in_skips": e[1],
                  "n_tok": int(e[2]), "mean_row_loss": round(e[3] / max(1, e[2]), 3)}
                 for (fim, name), e in cat_tbl.items()]
    cat_table.sort(key=lambda d: (-d["n_rows_in_skips"], -d["n_tok"]))

    # bursts
    runs, cur = [], 0
    for r in valid:
        if r["skipped"]:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)

    # correlations
    r_loss_ntok = pearson([r["loss"] for r in valid], [r["n_tok"] for r in valid])
    r_skip_ntok = pearson([1.0 if r["skipped"] else 0.0 for r in valid],
                          [float(r["n_tok"]) for r in valid])

    # threshold sweep
    sweep = {}
    for t in (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0):
        sweep[str(t)] = (sum(1 for r in valid if r["loss"] > t) / n) if n else None

    tok_all = sum(r["n_tok"] for r in valid)
    tok_skipped = sum(r["n_tok"] for r in skips)
    all_loss = [r["loss"] for r in valid]

    return {
        "config": {
            "batch_size": batch_size, "block_size": MB.BLOCK_SIZE,
            "dim": MB.DIM, "n_layers": MB.N_LAYERS, "n_heads": MB.N_HEADS,
            "vocab": MB.VOCAB_SIZE,
            "fim_ratio": MB.FIM_RATIO, "fim_variants": MB.FIM_VARIANTS,
            "short_mid_cum": MB.SHORT_MID_CUM, "long_mid_max": MB.LONG_MID_MAX,
            "min_sample_tokens": MB.MIN_SAMPLE_TOKENS,
            "pool_min": MB.POOL_MIN, "fetch_bulk": MB.FETCH_BULK,
            "loss_chunk": MB.LOSS_CHUNK, "z_loss_coef": MB.Z_LOSS_COEF,
            "fim_mode": MB.FIM_MODE, "rag_train_mode": MB.RAG_TRAIN_MODE,
            **phase2_meta,
        },
        "n_batches": n,
        "n_skipped": len(skips),
        "skip_rate": skip_rate,
        "loss_all": {"min": min(all_loss) if all_loss else None,
                     "p10": pct(all_loss, .10), "p50": pct(all_loss, .50),
                     "p90": pct(all_loss, .90), "max": max(all_loss) if all_loss else None},
        "loss_skipped": {"n": len(skip_loss), "min": pct(skip_loss, 0),
                         "p25": pct(skip_loss, .25), "p50": pct(skip_loss, .50),
                         "p75": pct(skip_loss, .75), "max": pct(skip_loss, 1)},
        "loss_kept": {"n": len(keep_loss), "p10": pct(keep_loss, .10),
                      "p50": pct(keep_loss, .50), "p90": pct(keep_loss, .90)},
        "n_tok_all": {"p10": pct(all_ntok, .10), "p50": pct(all_ntok, .50),
                      "p90": pct(all_ntok, .90)},
        "n_tok_skipped": {"p10": pct([r["n_tok"] for r in skips], .10),
                          "p50": pct([r["n_tok"] for r in skips], .50)},
        "n_tok_kept": {"p50": pct([r["n_tok"] for r in keeps], .50)},
        "thin_skips_frac_lt400": (sum(1 for r in skips if r["n_tok"] < 400) / len(skips)
                                  if skips else None),
        "bursts": {"n_runs": len(runs), "max_run": max(runs) if runs else 0,
                   "runs": runs[:20]},
        "corr": {"loss_vs_ntok": (round(r_loss_ntok, 4) if r_loss_ntok is not None else None),
                 "skip_vs_ntok": (round(r_skip_ntok, 4) if r_skip_ntok is not None else None)},
        "rows_skipped": row_stats(True),
        "rows_kept": row_stats(False),
        "supervised_tokens": {"total": tok_all, "in_skipped": tok_skipped,
                              "skipped_share": tok_skipped / tok_all if tok_all else None},
        "threshold_sweep": sweep,
        "category_table": cat_table[:25],
    }


def print_verdict(report):
    c = report["config"]
    print("\n" + "=" * 70, flush=True)
    print("HYPOTHESIS CHECK", flush=True)
    print("=" * 70, flush=True)
    rate = report["skip_rate"]
    print(f"skip rate reproduced: {rate * 100:.1f}% of micro-batches "
          f"(log shows ~27%) — {'consistent' if rate and 0.08 < rate < 0.6 else 'DEVIATES'}", flush=True)
    sl = report["loss_skipped"]
    if sl["n"]:
        band = sl["max"] - sl["min"]
        print(f"skipped-loss band: [{sl['min']:.2f}, {sl['max']:.2f}] p50 {sl['p50']:.2f} "
              f"({'tight (degenerate population)' if band < 3.5 else 'wide — mixed causes'}); "
              f"kept p50 {report['loss_kept']['p50']:.2f} — "
              f"{'bimodal: data, not stats' if sl['min'] > 5.0 and report['loss_kept']['p50'] < 2.0 else 'check'}", flush=True)
    rs, rk = report["rows_skipped"], report["rows_kept"]
    if rs and rk:
        print(f"degenerate-content signature: redaction-marker rows {rs['any_hit_frac'] * 100:.1f}% "
              f"(skipped) vs {rk['any_hit_frac'] * 100:.1f}% (kept) | non-ASCII {rs['non_ascii_mean'] * 100:.2f}% "
              f"vs {rk['non_ascii_mean'] * 100:.2f}% | <NAME>/<reponame> per row {rs['name_hits_per_row']:.2f}/{rs['repo_hits_per_row']:.2f} "
              f"vs {rk['name_hits_per_row']:.2f}/{rk['repo_hits_per_row']:.2f}", flush=True)
    print(f"sup tokens per batch: p50 {report['n_tok_all']['p50']} | thin-skip frac (<400 tok) "
          f"{report['thin_skips_frac_lt400'] if report['thin_skips_frac_lt400'] is not None else '-'}", flush=True)
    corr = report["corr"]
    print(f"independence: corr(loss, n_tok) = {corr['loss_vs_ntok']} | "
          f"corr(skip, n_tok) = {corr['skip_vs_ntok']} "
          f"({'skips are content events, not batch-size effects' if abs(corr['loss_vs_ntok'] or 1) < 0.3 else 'skips correlate with batch composition'})", flush=True)
    st = report["supervised_tokens"]
    print(f"training cost: {st['skipped_share'] * 100:.1f}% of supervised tokens sit in skipped batches "
          f"({st['in_skipped']}/{st['total']}) — that much gradient signal is dropped", flush=True)
    b = report["bursts"]
    print(f"burst structure: {b['n_runs']} skip runs, max {b['max_run']} consecutive "
          f"(pool-refill clustering => data-side, not optimizer)", flush=True)
    print("\ntop categories by rows inside skipped batches:", flush=True)
    for d in report["category_table"][:10]:
        kind = "fim" if d["fim"] else "lm"
        print(f"  {d['cat']:<18} [{kind}] rows {d['n_rows']:>3} in_skips {d['n_rows_in_skips']:>3} "
              f"n_tok {d['n_tok']:>7} mean_row_loss {d['mean_row_loss']}", flush=True)
    print(f"\nthreshold sweep (skip rate per threshold): "
          + "  ".join(f"{t}:{report['threshold_sweep'][t] * 100:.0f}%" for t in report["threshold_sweep"]), flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class _Tee:
    """Duplicate every print to a log file so the run survives SSH drops."""

    def __init__(self, primary, *extra):
        self.primary = primary
        self.extra = extra

    def write(self, s):
        self.primary.write(s)
        for st in self.extra:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        try:
            self.primary.flush()
        except Exception:
            pass
        for st in self.extra:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", type=int, default=150,
                    help="phase 2: number of micro-batches to run through the model (default 150)")
    ap.add_argument("--data-only", action="store_true",
                    help="run phase 1 only (no model, no GPU)")
    ap.add_argument("--skip-phase1", action="store_true",
                    help="skip the data-only scan and go straight to the model loss scan")
    ap.add_argument("--data-batches", type=int, default=30,
                    help="phase 1: number of assembled micro-batches to analyze (default 30)")
    ap.add_argument("--pool", type=int, default=512,
                    help="phase 1: number of raw samples to scan from the API (default 512)")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="explicit checkpoint path (default: latest step_big_*.pt in CKPT_DIR)")
    ap.add_argument("--out", type=str, default="diag_skip_out",
                    help="output directory (default ./diag_skip_out)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch-size", type=int,
                    default=int(os.environ.get("LOCLLM_BATCH_SIZE", "7")),
                    help="rows per micro-batch — must match training (default 7, the 4090 run)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the skip threshold (default: LOSS_SKIP_THRESHOLD / LOCLLM_SKIP_TOK)")
    ap.add_argument("--verbose", action="store_true",
                    help="print every batch line, not just skips")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # output dir + stdout log, before anything long runs
    os.makedirs(args.out, exist_ok=True)
    for _name in ("batches.jsonl", "skipped_detail.jsonl"):
        open(os.path.join(args.out, _name), "w").close()  # clear previous runs
    sys.stdout = _Tee(sys.stdout, open(os.path.join(args.out, "run.log"), "w", buffering=1))
    sys.stderr = sys.stdout
    print(f"stdout tee'd to {os.path.join(args.out, 'run.log')}", flush=True)

    MB._select_fim_mode_enable()
    if MB.RAG_TRAIN_MODE:
        print("WARNING: LOCLLM_RAG_TRAIN is set — training runs with RAG off. "
              "Unset it to match the production FIM pipeline.", flush=True)

    print("=" * 70, flush=True)
    print(f"diag_skip_micro: B={args.batch_size} T={MB.BLOCK_SIZE} "
          f"fim_mode={MB.FIM_MODE} seed={args.seed}", flush=True)
    print("=" * 70, flush=True)

    try:
        code_ids, name_by_id = MB.get_category_index()
    except Exception as e:
        raise SystemExit(f"cannot fetch category index from data API ({MB.API}): {e}\n"
                         "is the data server reachable from this box?")
    MB.CODE_CATEGORY_IDS = code_ids
    MB.CAT_NAME_BY_ID = name_by_id
    print(f"code categories for FIM: {len(code_ids)} IDs", flush=True)

    patch_fetch_for_diag()

    report = {"config": {"batch_size": args.batch_size, "block_size": MB.BLOCK_SIZE}}
    if args.skip_phase1:
        print("\nphase 1 skipped (--skip-phase1)", flush=True)
    else:
        try:
            report["phase1"] = phase1_data_scan(args, args.batch_size, name_by_id)
        except KeyboardInterrupt:
            print("\ninterrupted during phase 1 — exiting (no report written)", flush=True)
            raise SystemExit(1)

    if not args.data_only:
        records, cat_rows, meta = phase2_loss_scan(args, args.batch_size, name_by_id)
        if not records:
            raise SystemExit("no valid batches recorded in phase 2 — nothing to report")
        report["phase2"] = build_report(args, args.batch_size, records, cat_rows, meta)

        with open(os.path.join(args.out, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nfiles written: {args.out}/{{report.json, batches.jsonl, skipped_detail.jsonl, run.log}}",
              flush=True)
        print_verdict(report["phase2"])
    else:
        with open(os.path.join(args.out, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"\ndata-only mode: wrote {args.out}/report.json", flush=True)


if __name__ == "__main__":
    main()
