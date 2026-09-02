"""Memory probe: reproduce _micro_step's memory profile on the local GPU.

Usage: CUDA_VISIBLE_DEVICES=1 python mem_probe.py B T [SEG]
Prints allocated/reserved at each phase and grad dtype, to explain the
instance's 46 GB OOM at B=8.
"""
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Transformer

GPU = int(os.environ.get("PROBE_GPU", "1"))  # physical GPU 1 = RTX 3090 (24GB)
torch.cuda.set_device(GPU)
torch.set_default_device(f"cuda:{GPU}")
torch.set_default_dtype(torch.bfloat16)

B = int(sys.argv[1]) if len(sys.argv) > 1 else 1
T = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
SEG = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("LOCLLM_CKPT_SEG", "4"))
LAYERS = int(os.environ.get("PROBE_LAYERS", "128"))
VOCAB = 32014
CHUNK = 1024

torch.manual_seed(0)


def mem(tag):
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    p = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{tag}] alloc={a:.2f}GB reserved={r:.2f}GB peak={p:.2f}GB", flush=True)


print(f"probe B={B} T={T} seg={SEG} layers={LAYERS} torch={torch.__version__} "
      f"gpu={torch.cuda.get_device_name(GPU)} total={torch.cuda.get_device_properties(GPU).total_memory/1e9:.1f}GB")

m = Transformer(vocab_size=VOCAB, dim=1024, n_layers=LAYERS, n_heads=16,
                max_seq_len=T, ffn_hidden=6656)
n_params = sum(p.numel() for p in m.parameters())
print(f"params {n_params/1e9:.3f}B  param dtype {next(m.parameters()).dtype}")
mem("model loaded")

import bitsandbytes as bnb
decay = [p for n, p in m.named_parameters() if p.ndim >= 2]
no_decay = [p for n, p in m.named_parameters() if p.ndim < 2]
opt = bnb.optim.AdamW8bit(
    [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
    lr=1e-4, betas=(0.9, 0.98))
mem("optimizer created")

x = torch.randint(0, VOCAB, (B, T))
y = torch.randint(0, VOCAB, (B, T))
y[:, :128] = -100
scaler = torch.amp.GradScaler(enabled=True)


def _head_ce(hchunk, ychunk):
    # EXACT replica of main_big._head_ce (5 outputs)
    logits = m.lm_head(hchunk)
    maskc = ychunk != -100
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), ychunk.reshape(-1), reduction="sum")
    if os.environ.get("PROBE_LEAN_HEAD") == "1":
        return loss
    correct = int((logits.argmax(dim=-1)[maskc] == ychunk[maskc]).sum())
    per_tok = F.cross_entropy(logits.view(-1, logits.size(-1)), ychunk.reshape(-1),
                              reduction="none").float().view_as(ychunk)
    row_loss = torch.where(maskc, per_tok, torch.zeros_like(per_tok)).sum(dim=-1)
    row_cnt = maskc.sum(dim=-1)
    zsum = torch.logsumexp(logits.float(), dim=-1).pow(2).sum()
    return loss, correct, row_loss, row_cnt, zsum


with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    hidden = m(x, return_hidden=True)
    mem("after forward")
    hidden = hidden.to(torch.bfloat16)
    loss_sum = torch.zeros((), dtype=torch.float32)
    z_sum = torch.zeros((), dtype=torch.float32)
    row_loss_acc = None
    lean = os.environ.get("PROBE_LEAN_HEAD") == "1"
    mode = os.environ.get("PROBE_MODE", "")
    if mode == "head":
        # head-only graph: detach so backward never enters the transformer
        hidden = hidden.detach().requires_grad_(True)
    for s in range(0, T, CHUNK):
        yc = y[:, s:s + CHUNK]
        if lean:
            l_ = ckpt.checkpoint(
                _head_ce, hidden[:, s:s + CHUNK], yc, use_reentrant=True)
            loss_sum = loss_sum + l_
        else:
            l_, c_, rl_, rc_, z_ = ckpt.checkpoint(
                _head_ce, hidden[:, s:s + CHUNK], yc, use_reentrant=True)
            loss_sum = loss_sum + l_
            z_sum = z_sum + z_
            row_loss_acc = rl_ if row_loss_acc is None else row_loss_acc + rl_
    mem("after loss")
    loss = (loss_sum + 1e-4 * z_sum) / 12000

if mode == "trans":
    # transformer-only backward: skip the head entirely
    torch.cuda.memory._record_memory_history(True)
    hidden.sum().backward()
elif mode == "head":
    scaler.scale(loss).backward()
else:
    scaler.scale(loss).backward()
mem("after backward")

if mode == "trans":
    # snapshot must be taken while recording is still enabled
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(False)
    # reconstruct peak live set from alloc/free traces (device_traces is a list
    # of lists of dicts)
    events = []
    for lst in snap.get("device_traces", []):
        for tr in lst:
            action = tr.get("action", "")
            if "alloc" in action or "free" in action:
                events.append((tr.get("time_us", 0), action, tr.get("addr", 0), tr.get("size", 0)))
    events.sort()
    live = {}
    # preload currently-active blocks (allocated before recording started)
    for seg in snap.get("segments", []):
        for blk in seg.get("blocks", []):
            if blk.get("state") == "active_allocated":
                live[blk.get("address", id(blk))] = blk.get("size", 0)
    peak_total = 0.0
    peak_live = []
    for _, action, addr, size in events:
        if "alloc" in action and "free" not in action:
            live[addr] = size
        elif "free" in action:
            live.pop(addr, None)
        total = sum(live.values()) / 1e9
        if total > peak_total:
            peak_total = total
            peak_live = sorted(live.values(), reverse=True)
    print(f"[trans peak live] total={peak_total:.2f}GB n={len(peak_live)} top10:")
    for sz in peak_live[:10]:
        print(f"    {sz / 1e9:6.2f}GB")
g0 = next(p.grad for p in m.parameters() if p.grad is not None)
print("grad dtype:", g0.dtype, "| grad bytes/param:", g0.numel() and (g0.element_size() * g0.numel()) / g0.numel())

# live-allocation composition right now (grads full, saved inputs freed)
snap = torch.cuda.memory._snapshot()
live = []
for seg in snap["segments"]:
    for blk in seg["blocks"]:
        if blk["state"] == "active_allocated":
            live.append((blk["size"] / 1e9, blk.get("allocator_name", "?")))
live.sort(reverse=True)
print(f"[live after backward] total={sum(s for s, _ in live):.2f}GB top6:")
for sz, name in live[:6]:
    print(f"    {sz:6.2f}GB  {name}")

opt.step()
mem("after optimizer step")
opt.zero_grad(set_to_none=True)
mem("after zero_grad")
