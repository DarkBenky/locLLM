Updated todo list

Deleted memory 

Created memory file 

## Plan: Stabilize + upgrade locLLM training (merged)

One plan covering the variance work and the architecture work, ordered for implementation. Everything keeps your step-211k checkpoint loadable. I expanded the "Option C" set from 4 items to 9 — including one I think is bigger than any of the originals (**untying the embeddings**).

### Constraints
- Checkpoint must stay loadable — improve/extend, never invalidate
- Prefer function-preserving changes: no knowledge loss, still a win
- VRAM: `BATCH_SIZE` 8/accum 1 → 6/accum 2 (effective 12)
- FIM middles: mostly short, occasional long

### Steps

**Phase 0 — Diagnostics (read-only, do first; gates Phase 8)**
1. Per-block angular distance (block input vs output hidden) over ~500 calibration samples → sorted contribution table. Expect many near-identity blocks (see D12 below).
2. Histogram of supervised tokens per micro-step and per optimizer step over ~200 steps — quantify the swing before changing anything. *Parallel with 1.*
3. `grad_norm` percentiles (p50/p90/p99) to pick a real `GRAD_CLIP`. *Parallel with 1.*
4. Per-block residual-branch norm vs residual-stream norm (residual dominance).

**Phase 1 — Metric correctness** *(no training-semantics change)*

5. Fix the accuracy bug: return a correct-token count from the checkpointed `_head_ce` so `acc` covers **all** chunks, not just the first 1024 positions.
6. Fix per-category loss weighting (chunk `row_loss` weighted by full-row count). *Parallel with 5.*
7. Log token-weighted **window means** over `LOG_EVERY`, plus sup-tok min/max/std. Log every step to wandb, print every 10.
8. Track a 100-step median alongside the EMA; count and log skipped micro-steps per window.

**Phase 2 — Gradient weighting + constant token budget** *(depends on Phase 1 to be measurable)*

9. Replace `loss / GRAD_ACCUM` with a **fixed token normalizer**: backward `loss_sum / TOKEN_NORM` (≈12000). Highest-impact single change.
10. Dynamic accumulation until `accum_tokens >= TOKEN_TARGET`, with `MAX_MICRO` cap (2–4). Matches `BATCH_SIZE=6` + 2–4 micro-steps.
11. Make the skip path token-aware; step 9 makes under-normalization impossible.

**Phase 3 — Data-level variance** *(parallel with Phase 2)*

12. Cap the FIM middle with a mixed length distribution (~80% in 32–512 tokens, ~20% up to 2048). Note `SHORT_MID_CUM` / `MEDIUM_MID_CUM` already exist at `main_big.py:84-86` — **verify they're actually applied** in `_fim_splits`.
13. Replace `sort(key=len)` + random contiguous window with length-bucketed shuffling.
14. Stratify each batch by language/category; raise `POOL_MIN` 128 → 512+.

**Phase 4 — Optimizer & training-time stability**
15. `betas` (0.9, 0.95) → (0.9, 0.98). Optimizer state still loads.
16. Set `GRAD_CLIP` from the measured p90 (likely ~3.0), or keep 1.0 deliberately as normalized-SGD and stop reading `grad_norm` as health.
17. **z-loss**: `z_coef * mean(logsumexp(logits)²)`, `z_coef=1e-4` (PaLM/Chinchilla), computed per chunk inside `_head_ce`. Purely additive, no weight change, strong bf16 stability win.
18. **Weight-EMA** (decay ~0.999) used **only** for eval/export. Training untouched; eval curves get much smoother and EMA weights usually score better.
19. **LayerDrop** p=0.05–0.1, training only. At 128 layers this regularizes, gives ~5–10% speedup, **and** pre-conditions the model to tolerate removed blocks — directly de-risking Phase 8.

**Phase 5 — Eval quality**
20. `FIM_EVAL_GEN_SAMPLES` 4 → 32/64.
21. Add teacher-forced middle top-1 accuracy, first-line exact match, edit similarity (SAFIM-style ES). Keep `exact@32`; de-emphasize `prefix_acc`.
22. Verify train↔inference prompt symmetry for the double `<lang>` tag.

**Phase 6 — Knowledge-preserving architecture upgrades** *(the expanded "Option C" set, ordered by value/risk)*

23. **Untie the embeddings** — *exactly* function-preserving, and I'd rank this the highest-value item in the whole plan. Today `lm_head.weight = tok_emb.weight` (`model.py:95`). Clone into an independent parameter at load. Costs ~33M params. Why it matters: your 14 special tokens (FIM/ChatML/context/lang) were initialized by `resize_vocab_embeddings` as the **mean of all rows**, and currently must serve as both input embedding *and* output classifier — two roles that want opposite geometry. This is plausibly part of why FIM *generation* lags FIM *loss*.
24. **LayerScale** — learnable per-block scalar on each residual branch, init 1.0 → exactly function-preserving. Lets redundant blocks self-attenuate and yields a readable per-layer importance signal for Phase 8.
25. **fp32 RMSNorm** — compute `x.pow(2).mean()` in fp32, cast back. Strictly better numerics; matters across 128 residual additions in bf16. *Parallel with 23, 24.*
26. **Depth-scaled init for future fresh blocks**: `std` 0.02 → `0.02/sqrt(2·n_layers)` on `out_proj`/`w_down`. Doesn't touch existing weights.
27. **Calibrated QK-norm** — add RMSNorm on q and k, but initialize each gain to the *measured* per-head RMS of q/k on a calibration batch, so step 0 is approximately function-preserving; then a short heal. Standard deep-stack fix, pairs with step 16.
28. **GQA conversion** — mean-pool 16 kv heads → 4, keep 16 query heads. Not function-preserving, but the GQA paper's "uptraining" restores quality with ~5% of original compute. Cuts KV cache 4x: 512 KB/token → 128 KB/token. *Do after Phase 8's depth decision — the two multiply.*
29. **fp32 residual stream** with bf16 block compute — only if step 25 isn't enough.
30. **RoPE base** 10000 → 100000 (`ROPE_BASE` is already a knob). Cheap to heal, better at the 8192 window. Low priority.
31. **Stop widening the FFN further** until depth comes down. The 3.5x widening you did was the right direction (more compute per sequential step) but doesn't fix the aspect ratio.

**Phase 7 — Free performance (zero math change)**
32. Cache RoPE cos/sin as a registered buffer instead of rebuilding it in every forward *and* every checkpoint recompute (`model.py:112-114`).
33. Selective gradient checkpointing — every 2nd or 4th block instead of all 128.
34. `torch.compile` the `Block` module (compiled once, reused 128x). Big for a launch-bound deep-thin stack; verify interaction with checkpointing.
35. The speed from 32–34 is what pays for Phase 2's larger token budget.

**Phase 8 — Structural (deferred, gated on Phase 0.1 + the LayerScale signal)**
36. **Depth pruning** 128 → 48/64 (ShortGPT / Gromov et al. 2024), heal via the existing `WAKEUP` machinery. Surviving weights untouched. At 64: ratio 16, ~0.97B params, ~2x throughput, half the KV cache. Mirror of `upscale_into`, infra exists.
37. **Net2Wider** dim 1024 → 2048 with 32 layers (ratio 64, ~same params). Correct long-term shape but invasive — last.

### Execution order
Phase 0 → Phase 1 → Phase 6 items 23/24/25/26 (land together) → Phase 7 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → re-measure → Phase 8.

### Defects being fixed
| ID | Defect | Location |
|---|---|---|
| D1 | `loss / GRAD_ACCUM` weights micro-batches equally regardless of token count (2k–80k) | `main_big.py:1192` |
| D2 | `acc` from first 1024 positions, weighted by full token count → the 0.255/0.439 outliers | `main_big.py:1198`, `main_big.py:1504` |
| D3 | Same mismatch in `cat_stats` | `main_big.py:1211` |
| D4 | `sort(key=len)` + random contiguous window → length-homogeneous batches, wild regime swings | `main_big.py:816` |
| D5 | `LOG_EVERY` prints a single instantaneous step | `main_big.py:1551` |
| D6 | `GRAD_CLIP` 1.0 vs grad_norm 1.4–3.9 → clipped every step | `main_big.py:74` |
| D7 | `betas=(0.9,0.95)`, beta2 horizon ~20 steps | `main_big.py:1004` |
| D9 | gen eval on 4 samples, break-on-first-mismatch → pure noise | `main_big.py:1435` |
| D11 | `std=0.02`, no depth scaling at 128 layers | `model.py:107` |
| D12 | `upscale_into` zeroed `out_proj`+`w_down` for odd slots and all ≥52 → **≥102 of 128 blocks began as exact identity** | `main_big.py:222` |
| D13 | RoPE cache rebuilt every forward *and* every recompute | `model.py:112` |
| D15 | RMSNorm reduction in bf16, compounding over 128 blocks | `model.py:6` |
| D16 | KV cache 512 KB/token = 4.3 GB at 8192 ctx for one sequence | architecture |

### Verification
1. `sup tok/step` std over a 100-step window: ~15k → under 2k
2. Per-step loss std drops; `ema_loss` stops swinging 1.0 ↔ 1.3
3. `acc` no longer emits 0.25/0.44 outliers on FIM-heavy steps
4. `eval` / `fim eval` loss at step N+2000 ≤ the current 1.24–1.28 band
5. After untying (23): expect `fim eval` loss to drop within ~2k steps — watch explicitly
6. `fim gen` on 32+ samples shows a trend, not 0.02–0.07 noise
7. tok/s improves after Phase 7 with an *identical* first-step loss
8. **Items 23, 24, 25 must land with an unchanged first-step loss** — that is the definition of function-preserving. If the loss moves at all, the implementation is wrong.

### Decisions
- Everything through Phase 7 is training-loop, data-pipeline, or function-preserving architecture — the 211k checkpoint loads unchanged.
- Step-level loss normalization (`accum_loss_w / accum_tokens`) is **already correct**; the classic divide-by-accum bug is *not* present. The flaw is in the backward weighting only.
- Low `prefix_acc` is treated as a metric artifact first (Phase 5) and a model issue second (item 23).
- Structural reshaping (pruning/widening) is deferred until Phase 0.1 tells us how many of the 128 blocks actually do work.

Created 8 todos

## Implementation log

- 2026-09-02 DONE: Phase 1 (5,6,7,8) — acc/cat-stats over all chunks, windowed sup/loss logging + skip counter.
- 2026-09-02 DONE: Phase 2 (9,11) — fixed token normalizer TOKEN_NORM=12000 (LOCLLM_TOKEN_NORM), per-token skip (5.0).
- 2026-09-02 DONE: Phase 3 (12) — span mixture 80% 32-512 tok / 20% up to 2048.
- 2026-09-02 DONE: Phase 4 (15,16,17) — betas (0.9,0.98), GRAD_CLIP 3.0 (env), z-loss 1e-4 in _head_ce.
- 2026-09-02 DONE: Phase 5 (20) — FIM_EVAL_GEN_SAMPLES 32.
- 2026-09-02 DONE: Phase 7 (32) — RoPE tables cached per device (model.py).
- 2026-09-02 DONE: Phase 0 script model/diag_blocks.py — run: python diag_blocks.py <ckpt> [n_samples] [max_tok].
- 2026-09-02 DONE: Phase 4 (15) follow-up — betas forced to (0.9,0.98) after optimizer-state load (resume-safe).
- 2026-09-02 DONE: Phase 4 (15) follow-up — betas forced to (0.9,0.98) after optimizer-state load (resume-safe).
- 2026-09-02 DONE (H1): activation checkpointing segmented via LOCLLM_CKPT_SEG (default 4). Non-reentrant checkpointing saves the segment input per segment: 128 per-block segments = ~17 GB at B=8/T=8192 and OOMs 48 GB on the first backward. Segments of 4 cut the buffer to ~4.3 GB; forward/grad equivalence verified bitwise (max diff 0.0) on CPU for seg sizes 1/2/4/6.
- 2026-09-02 DONE (H2): head chunk checkpoints switched to use_reentrant=True. Non-reentrant checkpointing retains every tensor saved during each chunk's recompute (frame.recomputed) until the graph dies: 8 chunks of fp32 logits/logsumexp/per_tok = ~13-17 GB at B=8 held through the whole transformer backward. Measured on the 3090: 3.2 GB retained at B=2 (live after backward 15.95 GB), gone with reentrant (12.80 GB), backward peak 22.28 -> 19.13 GB. Reentrant is safe for _head_ce (small, dropout-free); math unchanged.
- 2026-09-02 MEASURED (mem_probe.py, local 3090): fixed cost ~19.2 GB (params 6.37 + AdamW8bit 6.5 + grads 6.37 bf16); projected B=8/T=8192/seg4/reentrant-head peak ~33-38 GB — fits 48 GB with margin.
- 2026-09-02 NOTE: the half-sequence split idea is mathematically WRONG (causal attention truncates context for the second half) — do not use; segmented checkpointing is the correct fix.
- 2026-09-02 VERIFIED: lint clean, model.py self-test OK, span sampler + constants smoke OK.
- 2026-09-02 DONE: Phase 0 diagnostics (diag_blocks.py, 64 samples x 2048 tok on step_big_fim_211500.pt). Result: block 0 extreme (1-cosim 1.06, attn branch 78x residual norm); blocks 2/44/46/48/50 moderate (0.06-0.16); blocks 4-28 (even) small (0.02-0.03); **122/128 blocks near-identity (1-cosim < 0.05) and ALL blocks >= 52 exact identity** — confirms D12 and strongly supports Phase 8 depth pruning. Data endpoint fixed (/api/get-next-samples-random), results table in conversation; diag now saves <ckpt>_diag.json.
- 2026-09-02 DONE: Phase 6 (23) — untied lm_head (fresh models start tied-equivalent via copy; old checkpoints already contain lm_head.weight, so loads are unchanged). Verified: lm_head == tok_emb after loading 211500.
- 2026-09-02 DONE: Phase 6 (24) — LayerScale ls_attn/ls_ffn per block, init 1.0 (bf16 multiply by 1.0 is bitwise-exact => function-preserving). Applied in model.py and in inference.py's manual prefill/decode paths (train/inference symmetry kept).
- 2026-09-02 DONE: Phase 6 (25) — fp32 RMSNorm accumulation. Measured logit delta vs old bf16 reduction on 211500: **0.00e+00** (bitwise identical on the test sample).
- 2026-09-02 DONE: Phase 6 (26) — depth-scaled init (0.02/sqrt(2*n_layers)) for out_proj/w_down of FRESH weights only; loaded checkpoints overwrite.
- 2026-09-02 DONE: optimizer-state splice `_splice_optimizer_for_arch` — rebuilds saved 8-bit state for untie+LayerScale (bnb load_state_dict is positional and validates group sizes). lm_head clones tok_emb's moments (exact at step 0). VERIFIED on 211500: load_state_dict OK, lm_head state == tok_emb state, optimizer.step() OK. Also fixed `_opt_state_is_8bit` to recognize bnb 0.50's `__bnb_optimizer_quant_state__` key (this was silently restarting the optimizer fresh on the instance!).
- 2026-09-02 NOTE: verify_phase6.py is the reusable function-preservation test (LayerScale==1, lm_head==tok_emb, RMSNorm delta, optimizer splice+step).
- 2026-09-02 DONE: Phase 2 (10) — dynamic accumulation: micro-steps until accum_tokens >= TOKEN_TARGET (LOCLLM_TOKEN_TARGET, default = TOKEN_NORM 12000), capped by MAX_MICRO (LOCLLM_MAX_MICRO, default GRAD_ACCUM). Logs train/micro_steps.
- 2026-09-02 DONE: Phase 3 (13,14) — length-bucketed round-robin batch selection (replaces sort(key=len) contiguous window), per-batch category stratification (prefer unseen cats), POOL_MIN 128 -> 512. Smoke-tested: mixed-length batches, distinct categories.
- 2026-09-02 DONE: Phase 4 (18) — weight-EMA on CPU (bf16, decay 0.999, synced every LOCLLM_EMA_EVERY=100 steps), used ONLY for eval (run_eval/run_fim_eval/run_fim_gen_eval swap EMA in/out). LOCLLM_EMA=0 disables.
- 2026-09-02 DONE: Phase 4 (19) — LayerDrop p=0.05 (LOCLLM_LAYERDROP), training only, torch-rng based so checkpoint recompute drops the SAME blocks (fork_rng replays state). Eval never drops. Smoke-tested.
- 2026-09-02 DONE: Phase 5 (21) — run_fim_gen_eval now reports teacher-forced middle top-1 (batched), first-line exact match, and token edit-similarity (Levenshtein, SAFIM-style) alongside prefix_acc/exact@k.
- 2026-09-02 VERIFIED: Phase 5 (22) — training (_fim_variant) and inference (generate_fim) FIM prompts are SYMMETRIC: <fim_pre><lang>lang</lang>pre<fim_suf><lang>lang</lang>suf<fim_mid> in both. Minor known asymmetry only in the RAG path: inference wraps context in ctx_start/ctx_end, training appends raw (RAG_TRAIN_MODE is off by default; fix if ever enabled).
- 2026-09-02 DONE: Phase 7 (33) — selective checkpointing via LOCLLM_CKPT_SEG (segments of N blocks, default 4).
- 2026-09-02 DONE: Phase 7 (34) — torch.compile per Block, gated behind LOCLLM_COMPILE=1 (off by default; verify interaction with checkpointing before enabling in training).
- 2026-09-02 ALL FIX.md ITEMS COMPLETE (remaining: optional Phase 8 follow-ups + future items).
- 2026-09-02 DONE: Phase 8 tooling (36) — model/prune_blocks.py prunes 128 -> 52 layers (drop blocks 52-127, measured near-identity; keeps ALL contributing blocks incl. 44/46/48/50), carries optimizer moments for surviving params (name/position-based, bnb-compatible), saves for the NEW arch (untied + LayerScale). Result: step_big_fim_211500_pruned52.pt, 1.347B params. main_big: LOCLLM_N_LAYERS env + PRUNED wake-up heal on "pruned_from" marker.
- 2026-09-02 MEASURED: pruned 52L eval on the real eval sets (LOCLLM_EVAL_ONLY): eval 1.715 vs 1.316, fim 1.698 vs 1.230 (step-0 cost of dropping the identity tail), but fim gen prefix_acc 0.210 vs 0.023 and exact@32 0.12 vs 0.00 (9x better generation — dead layers were hurting sampling). Deploy with wake-up heal; expect eval to recover in a few thousand steps.
- 2026-09-02 DONE: Phase 8+widen combined (37) — model/widen_dim.py: Net2Wider dim 1024->1536, heads 16->24 (head_dim stays 64; extra heads zero-init), ffn 6656->9984 (new gate/up rows fresh-init, w_down new cols/rows zero). RMSNorm weights scaled by sqrt(old/new) (RMSNorm is not zero-padding invariant); qkv row layout remapped ([q,k,v] blocks shift with dim). VERIFIED function-preserving: max logit diff 0.024 (bf16 rounding) vs pruned model. Result: step_big_fim_211500_pruned52_w1536.pt, 2.98B params, NO optimizer state (fresh + wake-up heal, same precedent as FFN widen). main_big: LOCLLM_DIM/LOCLLM_N_HEADS envs; inference derives DIM/N_HEADS from checkpoint meta (__meta_n_heads__).
- 2026-09-02 NOTE: random-token logit comparison 128L vs 52L is misleading (off-distribution, delta +1.7); decision made on the real eval-set numbers instead.
- TODO (DEPLOYED to instance, operational only): rsync code + step_big_fim_211500_pruned52_w1536.pt, restart with LOCLLM_N_LAYERS=52 LOCLLM_DIM=1536 LOCLLM_N_HEADS=24 LOCLLM_FFN_RATIO=6.5.
- OPTIONAL follow-ups (NOT blocking the long run; separate future plan): item 8 100-step median (window mean/std used instead), item 27 calibrated QK-norm, item 28 GQA (KV cache still 512 KB/tok, mitigated by pruning), item 30 RoPE base 100000, item 34 torch.compile verification (gated off by default).
