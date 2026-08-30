from __future__ import annotations

import re
import sys
import unicodedata

import torch
import wandb

import chatml

SAMPLE_PROMPT_LINES = 6
SAMPLE_GEN_TOKENS = 128
SAMPLE_TEMP = 0.8
TERM_WIDTH = 80

# --- terminal colors (ANSI; only when stdout is a tty) ---
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ANSI = {"ctx": "\033[90m", "prefix": "\033[32m", "gen": "\033[1;33m",
        "suffix": "\033[34m", "head": "\033[1;36m"}
USE_COLOR = sys.stdout.isatty()


def _color(code: str, text: str) -> str:
    if not USE_COLOR or not text:
        return text
    return f"{ANSI[code]}{text}\033[0m"


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


LAST_RAW_TOKENS = None
LAST_FIM_TOKENS = None
LAST_FIM_MID = None
sample_table = wandb.Table(columns=["step", "sample"])


def record_raw_tokens(tokens):
    global LAST_RAW_TOKENS
    LAST_RAW_TOKENS = tokens


def record_fim_sample(tokens, mid):
    global LAST_FIM_TOKENS, LAST_FIM_MID
    LAST_FIM_TOKENS = tokens
    LAST_FIM_MID = mid


def _disp_width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in _strip_ansi(s))


def _wrap(text: str, width: int = TERM_WIDTH) -> list[str]:
    """Wrap text to `width` visible columns, treating ANSI escapes as zero-width."""
    out = []
    for line in text.split("\n"):
        line = line.rstrip()
        cur = ""
        cur_w = 0
        i = 0
        while i < len(line):
            if line[i] == "\x1b":
                j = line.find("m", i)
                if j == -1:
                    cur += line[i:]
                    break
                cur += line[i:j + 1]
                i = j + 1
                continue
            ch = line[i]
            w = 2 if unicodedata.east_asian_width(ch) in "WF" else 1
            if cur_w + w > width and cur_w > 0:
                out.append(cur)
                cur = ""
                cur_w = 0
            cur += ch
            cur_w += w
            i += 1
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


def _fmt_fim_sample(step: int, context_text, prefix_text: str, suffix_text: str,
                    gen_text: str, final_text: str | None = None) -> list[str]:
    block = [f"===== step {step} | FIM checkpoint sample ====="]
    block.append(_color("head", "-- context --"))
    block.extend(_wrap(_color("ctx", context_text)) if context_text else ["(none)"])
    block.append(_color("head", "-- <fim_prefix> --"))
    block.extend(_wrap(_color("prefix", prefix_text)))
    block.append(_color("head", "-- <fim_suffix> --"))
    block.extend(_wrap(_color("suffix", suffix_text)))
    block.append(_color("head", "-- <fim_middle> | model output --"))
    block.extend(_wrap(_color("gen", gen_text)))
    if final_text:
        block.append(_color("head", "-- final code (prefix + generated + suffix) --"))
        block.extend(_wrap(final_text))
    block.append("=" * (TERM_WIDTH - 4))
    return block


def run_fim_checkpoint_sample(step: int, model, sp, block_size: int, device: str) -> None:
    global LAST_FIM_TOKENS, LAST_FIM_MID
    if not LAST_FIM_TOKENS or LAST_FIM_MID is None:
        print("  (FIM checkpoint sample skipped: no FIM sample captured)")
        return
    try:
        base = sp.get_piece_size()
        ctx = chatml.context_ids(base)
        lang = chatml.lang_ids(base)
        toks = LAST_FIM_TOKENS
        mid = LAST_FIM_MID
        fim_pre, fim_suf, fim_mid, fim_end = base, base + 1, base + 2, base + 3
        cs = toks.index(ctx["start"]) if ctx["start"] in toks else -1
        ce = toks.index(ctx["end"]) if ctx["end"] in toks else -1
        pi = toks.index(fim_pre)
        si = toks.index(fim_suf)
        extra = {fim_pre: "", fim_suf: "", fim_mid: "", fim_end: "",
                 ctx["start"]: "<context_start>", ctx["end"]: "</context_end>",
                 lang["open"]: "<lang>", lang["close"]: "</lang>"}
        context_text = None
        if cs >= 0 and ce > cs:
            context_text = chatml.safe_decode(toks[cs + 1:ce], sp, chatml.reserved_ids(base), extra=extra)
        prefix_full = chatml.safe_decode(toks[pi + 1:si], sp, chatml.reserved_ids(base), extra=extra)
        suffix_full = chatml.safe_decode(toks[si + 1:mid], sp, chatml.reserved_ids(base), extra=extra)
        prefix_text = prefix_full
        if len(prefix_text) > 600:
            prefix_text = "[...] " + prefix_text[-600:]
        suffix_text = suffix_full
        if len(suffix_text) > 600:
            suffix_text = suffix_text[:600] + " [...]"

        model.eval()
        with torch.no_grad():
            prompt_toks = toks[:mid + 1][-(block_size - SAMPLE_GEN_TOKENS):]
            prompt = torch.tensor([prompt_toks], dtype=torch.long, device=device)
            stop_tokens = {fim_end, base + chatml.IM_END_OFF, ctx["end"]}
            gen = model.generate(prompt, max_new_tokens=SAMPLE_GEN_TOKENS,
                                 temperature=SAMPLE_TEMP, stop_tokens=stop_tokens)
        model.train()

        gen_ids = gen[0, prompt.shape[1]:].tolist()
        if gen_ids:
            gen_text = chatml.safe_decode(
                gen_ids, sp, chatml.reserved_ids(base), extra=extra)
        else:
            gen_text = "<|fim_end|>"
        # final-code view: generated text inline between prefix and suffix
        pfx = prefix_full
        sfx = suffix_full
        if len(pfx) > 1500:
            pfx = "[...]\n" + pfx[-1500:]
        if len(sfx) > 1500:
            sfx = sfx[:1500] + "\n[...]"
        final_text = (_color("prefix", pfx) + _color("gen", gen_text)
                      + _color("suffix", sfx))
        block = _fmt_fim_sample(step, context_text, prefix_text, suffix_text, gen_text, final_text)
        for line in block:
            print(line)
        print()
        sample_table.add_data(step, "\n".join(_strip_ansi(l) for l in block))
        wandb.log({"checkpoint_samples": sample_table}, step=step)
    except Exception as e:
        print(f"WARNING: FIM checkpoint sample generation failed: {e}")
        model.train()


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
            ctx = chatml.context_ids(base)
            stop_tokens = {base + 3, base + chatml.IM_END_OFF, ctx["end"]}
            gen = model.generate(prompt, max_new_tokens=SAMPLE_GEN_TOKENS,
                                 temperature=SAMPLE_TEMP, stop_tokens=stop_tokens)
        model.train()

        gen_ids = gen[0, prompt.shape[1]:].tolist()
        if gen_ids:
            gen_text = chatml.safe_decode(
                gen_ids, sp, chatml.reserved_ids(base),
                extra={base: "", base + 1: "", base + 2: "", base + 3: "",
                       ctx["start"]: "<context_start>", ctx["end"]: "</context_end>"})
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
