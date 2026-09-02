#!/bin/bash
# locLLM long unattended training run on the rented RTX 4090 (48 GB).
# Config validated by smoke test (2026-09-02):
#   - 52 layers x dim 1536 x 24 heads x FFN 9984 (2.98B params, bf16)
#   - B=7 x T=8192 x accum=6; per-block activation checkpointing (CKPT_SEG=1)
#     is REQUIRED: seg=4 OOMs at ~46 GB. seg=1+B=7 peak fits with headroom.
#   - 8-bit AdamW, wake-up heal active for 1500 steps (pruned+widened marker)
#   - LayerDrop 0.05 (default), EMA eval-only, dynamic accumulation
cd ~/locLLM/model || exit 1
WANDB_KEY=$(grep -h "WANDB_API_KEY" ~/.bashrc | tail -1 | sed "s/^[^=]*=//; s/^[[:space:]]*//; s/^[\"'\`]//; s/[\"'\`][[:space:]]*$//")
exec env \
  WANDB_API_KEY="$WANDB_KEY" \
  LOCLLM_FIM=1 \
  LOCLLM_GPU=0 \
  LOCLLM_GPU_NAME="RTX 4090" \
  LOCLLM_OPTIMIZER=8bit \
  LOCLLM_N_LAYERS=52 \
  LOCLLM_DIM=1536 \
  LOCLLM_N_HEADS=24 \
  LOCLLM_FFN_RATIO=6.5 \
  LOCLLM_CKPT_SEG=1 \
  LOCLLM_BATCH_SIZE=7 \
  LOCLLM_GRAD_ACCUM=6 \
  LOCLLM_OPT_EVERY=4 \
  /root/miniconda3/bin/python -u main_big.py
