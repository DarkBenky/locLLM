from __future__ import annotations

import torch

IM_START_OFF, IM_END_OFF = 4, 5
ROLE_OFFSETS = ("system", "user", "assistant", "tool")
EXTRA_FIM = 4
EXTRA_CHATML = 6
EXTRA_CONTEXT = 2
EXTRA_LANG = 2


def reserved_total():
    return EXTRA_FIM + EXTRA_CHATML + EXTRA_CONTEXT + EXTRA_LANG


def context_ids(base):
    off = base + EXTRA_FIM + EXTRA_CHATML
    return {"start": off, "end": off + 1}


def lang_ids(base):
    off = base + EXTRA_FIM + EXTRA_CHATML + EXTRA_CONTEXT
    return {"open": off, "close": off + 1}

CHATML_MASK_PROB = 0.8


def make_marker_tables(sp):
    start_full = sp.encode("<|im_start|>", out_type=int)
    end_full = sp.encode("<|im_end|>", out_type=int)
    im_start_core = start_full[1:]
    im_end_core = end_full[1:]
    b = sp.encode("x<|im_start|>", out_type=int)
    lead_after_char = b[len(b) - len(im_start_core) - 1]
    leads = {start_full[0], lead_after_char}
    roles = {}
    for r in ROLE_OFFSETS:
        hdr = sp.encode(f"<|im_start|>{r}", out_type=int)
        assert hdr[:len(start_full)] == start_full, r
        roles[r] = hdr[len(start_full):]
    return im_start_core, im_end_core, roles, leads


def find_markers(tokens, core, leads):
    n, L = len(tokens), len(core)
    return [i for i in range(n - L + 1)
            if tokens[i:i + L] == core and i > 0 and tokens[i - 1] in leads]


def analyze(tokens, im_start_core, im_end_core, roles, leads):
    starts = find_markers(tokens, im_start_core, leads)
    ends = find_markers(tokens, im_end_core, leads)
    headers = []
    for s in starts:
        e = s + len(im_start_core)
        for role, rseq in roles.items():
            if tokens[e:e + len(rseq)] == rseq:
                headers.append((role, s, e + len(rseq)))
                break
    return headers, ends


def build_mask(tokens, headers, ends, im_start_core, im_end_core):
    n = len(tokens)
    if not headers:
        return [True] * n
    m = [False] * n
    for role, s, _ in headers:
        if role != "assistant":
            continue
        nxt = next((e for e in ends if e > s), n)
        for j in range(s - 1, min(nxt + len(im_end_core), n)):
            m[j] = True
    return m


def reserved_ids(base):
    return {
        "im_start": base + IM_START_OFF,
        "im_end": base + IM_END_OFF,
        "roles": {r: base + ROLE_OFFSETS.index(r) + 6 for r in ROLE_OFFSETS},
    }


def replace_markers(tokens, headers, ends, ids):
    im_end_core = ids["end_core"]
    spans = []
    for role, s, e in headers:
        spans.append((s - 1, e, [ids["im_start"], ids["roles"][role]]))
    for e in ends:
        spans.append((e - 1, e + len(im_end_core), [ids["im_end"]]))
    spans.sort()
    out, i = [], 0
    for s, e, rep in spans:
        out.extend(tokens[i:s])
        out.extend(rep)
        i = e
    out.extend(tokens[i:])
    return out


def safe_decode(tokens, sp, ids, extra=None):
    lit = {ids["im_start"]: "<|im_start|>", ids["im_end"]: "<|im_end|>"}
    lit.update({v: r for r, v in ids["roles"].items()})
    if extra:
        lit.update(extra)
    out, buf = [], []
    for t in tokens:
        if t in lit:
            if buf:
                out.append(sp.decode(buf))
                buf = []
            out.append(lit[t])
        else:
            buf.append(t)
    if buf:
        out.append(sp.decode(buf))
    return "".join(out)


def mask_from_ids(tokens, ids):
    im_start, im_end = ids["im_start"], ids["im_end"]
    role_asst = ids["roles"]["assistant"]
    n = len(tokens)
    if im_start not in tokens:
        return [True] * n
    m = [False] * n
    i = 0
    while i < n:
        if tokens[i] == im_start and i + 1 < n and tokens[i + 1] == role_asst:
            j = i
            while j < n and tokens[j] != im_end:
                j += 1
            end = min(j + 1, n)
            for k in range(i, end):
                m[k] = True
            i = end
        else:
            i += 1
    return m


class ChatMLDetector:
    def __init__(self, sp):
        self.im_start_core, self.im_end_core, self.roles, self.leads = \
            make_marker_tables(sp)

    def analyze(self, tokens):
        return analyze(tokens, self.im_start_core, self.im_end_core,
                       self.roles, self.leads)

    def mask(self, tokens):
        headers, ends = self.analyze(tokens)
        return build_mask(tokens, headers, ends, self.im_start_core,
                          self.im_end_core)

    def mask_targets(self, tokens, n):
        headers, ends = self.analyze(tokens)
        if not headers:
            return None
        m = build_mask(tokens, headers, ends, self.im_start_core,
                       self.im_end_core)
        return torch.tensor(m[1:n + 1], dtype=torch.bool)
