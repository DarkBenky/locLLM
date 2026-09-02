"""Phase 6 verification (FIX.md 23/24/25): load the 211500 checkpoint into the
new architecture and prove function preservation.

Checks:
  1. LayerScale params stay at 1.0 after a strict=False load.
  2. lm_head.weight == tok_emb.weight after load (tied-equivalent start).
  3. fp32 RMSNorm (item 25) numeric delta vs the old bf16 reduction.
  4. Optimizer-state splice (untie + LayerScale) loads and steps cleanly.
"""
import sys

import torch

sys.path.insert(0, ".")
import main_big as mb  # noqa: E402  (loads sentencepiece etc.)
import model as M  # noqa: E402

CKPT = "checkpoints/step_big_fim_211500.pt"

ckpt = torch.load(CKPT, map_location="cpu", mmap=True)
sd = dict(ckpt["model"])
sd.setdefault("lm_head.weight", sd["tok_emb.weight"])
vocab = sd["tok_emb.weight"].shape[0]
n_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
ffn = sd["blocks.0.ffn.w_gate.weight"].shape[0]

m = M.Transformer(vocab_size=vocab, dim=1024, n_layers=n_layers, n_heads=16,
                  max_seq_len=512, ffn_hidden=ffn)
m.load_state_dict(sd, strict=False)
m.eval()

# 1. LayerScale == 1.0
ls_all_one = all(torch.equal(b.ls_attn, torch.ones_like(b.ls_attn))
                 and torch.equal(b.ls_ffn, torch.ones_like(b.ls_ffn))
                 for b in m.blocks)
print("1. LayerScale init == 1.0 everywhere:", ls_all_one)

# 2. untied head starts tied-equivalent
print("2. lm_head == tok_emb after load:", torch.equal(m.lm_head.weight, m.tok_emb.weight))

# 3. fp32 RMSNorm delta: new forward vs monkeypatched old-bf16-norm forward
torch.manual_seed(0)
x = torch.randint(0, vocab, (1, 128))
with torch.no_grad():
    logits_new = m(x)[0]
    # old norm: bf16 reduction
    old_fwd = M.RMSNorm.forward

    def old_norm(self, xx):
        norm = xx * torch.rsqrt(xx.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

    M.RMSNorm.forward = old_norm
    logits_old = m(x)[0]
    M.RMSNorm.forward = old_fwd
    d = (logits_new.float() - logits_old.float()).abs()
    rel = (d / (logits_old.float().abs() + 1e-6)).max().item()
    print(f"3. fp32-RMSNorm delta: max|dlogit|={d.max().item():.2e} "
          f"max rel={rel:.2e} (expect tiny)")

# 4. optimizer splice: replicate main_big's optimizer construction
import bitsandbytes as bnb
decay = [p for n, p in m.named_parameters() if p.ndim >= 2]
no_decay = [p for n, p in m.named_parameters() if p.ndim < 2]
opt = bnb.optim.AdamW8bit(
    [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
    lr=1e-4, betas=(0.9, 0.98))
n_new = sum(len(g["params"]) for g in opt.param_groups)
old_sd = ckpt["optimizer"]
print("4a. old state entries:", len(old_sd["state"]), "| new param count:", n_new)
print("4b. 8bit detection now:", mb._opt_state_is_8bit(old_sd))
tied_emb = torch.equal(sd["tok_emb.weight"], sd["lm_head.weight"])
spliced = mb._splice_optimizer_for_arch(old_sd, opt, m, tied_emb)
print("4c. spliced state entries:", len(spliced["state"]))
opt.load_state_dict(spliced)
print("4e. load_state_dict OK")
# state actually landed: tok_emb vs lm_head quant state equal
import bitsandbytes.functional as bnbF  # noqa: E402
st = opt.state
print("4d. optimizer state entries after load:", len(st))
q_emb = None
q_head = None
for p, v in st.items():
    if p is m.tok_emb.weight:
        q_emb = v
    if p is m.lm_head.weight:
        q_head = v
if q_emb and q_head:
    print("4e2. state keys:", sorted(q_emb.keys()))
    same = all(torch.equal(q_emb[k], q_head[k]) for k in q_emb
               if isinstance(q_emb[k], torch.Tensor))
    print("4e3. lm_head state == tok_emb state:", same)
# one optimizer step on a dummy loss to prove step works
logits = m(x)[0]
import torch.nn.functional as F
loss = F.cross_entropy(logits[:, 1:].reshape(-1, vocab), x[:, 1:].reshape(-1))
loss.backward()
opt.step()
opt.zero_grad(set_to_none=True)
print("4f. optimizer.step() OK")
