from __future__ import annotations

import unicodedata

import torch
import wandb

import chatml

SAMPLE_PROMPT_LINES = 6
SAMPLE_GEN_TOKENS = 128
SAMPLE_TEMP = 0.8
TERM_WIDTH = 80

LAST_RAW_TOKENS = None
sample_table = wandb.Table(columns=["step", "sample"])


def record_raw_tokens(tokens):
    global LAST_RAW_TOKENS
    LAST_RAW_TOKENS = tokens


def _disp_width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _wrap(text: str, width: int = TERM_WIDTH) -> list[str]:
    out = []
    for line in text.split("\n"):
        line = line.rstrip()
        cur = ""
        for ch in line:
            if _disp_width(cur + ch) > width:
                out.append(cur)
                cur = ch
            else:
                cur += ch
        out.append(cur)
    return out


def _fmt_checkpoint_sample(step: int, prompt_text: str, gen_text: str) -> list[str]:
    block = [f"===== step {step} | checkpoint sample ====="]
    block.append(f"-- prompt (last {SAMPLE_PROMPT_LINES} lines) --")
    block.extend(_wrap(prompt_text))
    block.append(f"-- generated ({SAMPLE_GEN_TOKENS} tokens) --")
    block.extend(_wrap(gen_text))
    block.append("=" * (TERM_WIDTH - 4))
    return block


def run_checkpoint_sample(step: int, model, sp, block_size: int, device: str) -> None:
    global LAST_RAW_TOKENS
    if not LAST_RAW_TOKENS:
        print("  (checkpoint sample skipped: no raw sample captured)")
        return
    try:
        raw_text = sp.decode(LAST_RAW_TOKENS)
        lines = raw_text.split("\n")
        prompt_text = "\n".join(lines[-SAMPLE_PROMPT_LINES:])
        prompt_ids = sp.encode(prompt_text)[-(block_size - SAMPLE_GEN_TOKENS):]
        if not prompt_ids:
            prompt_ids = LAST_RAW_TOKENS[-min(len(LAST_RAW_TOKENS), block_size - SAMPLE_GEN_TOKENS):]
        if not prompt_ids:
            print("  (checkpoint sample skipped: empty prompt)")
            return
        prompt_text = sp.decode(prompt_ids)

        model.eval()
        with torch.no_grad():
            prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            base = sp.get_piece_size()
            stop_tokens = {base + 3, base + chatml.IM_END_OFF}
            gen = model.generate(prompt, max_new_tokens=SAMPLE_GEN_TOKENS,
                                 temperature=SAMPLE_TEMP, stop_tokens=stop_tokens)
        model.train()

        gen_ids = gen[0, prompt.shape[1]:].tolist()
        if gen_ids:
            gen_text = chatml.safe_decode(
                gen_ids, sp, chatml.reserved_ids(base),
                extra={base: "", base + 1: "", base + 2: "", base + 3: ""})
        else:
            gen_text = "<|im_end|>"
        block = _fmt_checkpoint_sample(step, prompt_text, gen_text)
        for line in block:
            print(line)
        print()
        sample_table.add_data(step, "\n".join(block))
        wandb.log({"checkpoint_samples": sample_table}, step=step)
    except Exception as e:
        print(f"WARNING: checkpoint sample generation failed: {e}")
        model.train()
