# Add this datasets

- [ ] Distill the current dataset only for code
  - [ ] create training script that will be used for fim training
    - [ ] add special token <context_start> and </context_end>
    - [ ] inject context to model fim input

- [x] https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1
- [ ] https://huggingface.co/datasets/rajpurkar/squad/
- [x] https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- [ ] https://huggingface.co/datasets/tatsu-lab/alpaca
- [x] https://huggingface.co/datasets/r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation
- [x] https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1
- [x] https://huggingface.co/datasets/HuggingFaceTB/smoltalk
- [x] https://huggingface.co/datasets/open-thoughts/AgentTrove
- [x] https://huggingface.co/datasets/bigcode/starcoderdata
- [X] https://huggingface.co/datasets/code-search-net/code_search_net
- [ ] https://huggingface.co/datasets/open-phi/programming_books_llama

- [x] increase the split of FIM to 80%
  - [x] Fim support for all samples from stack V3 (also code_search_net raw + starcoderdata)
- [X] increase LR ?
- [X] optimize training (tokens/sec)
- [X] threading for dataset uploading (dataSets.py)
  - [X] pre dataset checkpoint

## Training

`model/main_big.py` — **single-GPU** training with interactive GPU selection.

- On startup it launches `model/gpuSeletor/main.py` (in a subprocess) and lets you
  pick **one** GPU plus a per-GPU `batch_size` and `accumulation_steps`
  (effective batch = `batch_size × block_size × accumulation_steps`).
- Per-GPU optimizer choice: **AdamW (fp32)** (default) or **AdamW 8-bit**
  (`bitsandbytes`, saves ~7 GB VRAM — needed to fit the 1.6B model on the 3060).
  Selecting 8-bit requires `pip install bitsandbytes`. Note: switching the
  optimizer format between runs (fp32 ↔ 8-bit) resets optimizer momentum —
  model weights are always preserved on resume.
- VRAM guidance (1.6B model, 128 layers, block 4096): baseline (weights + grads +
  optimizer) ≈ **9.25 GiB** with 8-bit Adam (fp32 Adam needs ~20 GiB — does not
  fit the 3060 at all). The per-step backward spike adds ~1.7 GiB at batch 2.
  - **RTX 3060 (12 GB): use 8-bit Adam + `batch_size=1`.** Batch ≥ 2 OOMs.
  - **RTX 3090 (24 GB):** fits batch 2+ with fp32 or 8-bit Adam.
- The chosen GPU is applied via `CUDA_VISIBLE_DEVICES` **before** CUDA is
  initialized, so only that device is used for training.
- Cancelling the prompt or selecting nothing falls back to GPU 0 with the file's
  default `BATCH_SIZE` / `GRAD_ACCUM`.

### TODO: multi-GPU training (not implemented yet)

- The GPU selector (`model/gpuSeletor/main.py`) already returns a **list** of GPUs,
  each with its own `batch_size` / `accumulation_steps`, so the plumbing for
  per-GPU config exists.
- To add multi-GPU support, `model/main_big.py` needs to be refactored so the
  training loop becomes `train_worker(rank, selected)` launched with
  `torch.multiprocessing.spawn` (start method **spawn**, not fork) and
  `CUDA_VISIBLE_DEVICES` set to the chosen physical indices. Each rank would:
  - build its own `Transformer` and wrap it in `DistributedDataParallel`
  - use its own data-worker thread / sample pool with its own `batch_size`
  - run its own micro-steps inside `model.no_sync()` and average gradients with a
    single `dist.all_reduce(..., SUM)` per optimizer step (this is what allows
    different `accumulation_steps` per GPU)
  - have only rank 0 save checkpoints (`model.module.state_dict()`), run eval /
    checkpoint samples, and log to wandb
