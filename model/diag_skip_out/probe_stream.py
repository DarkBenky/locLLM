#!/usr/bin/env python3
"""Probe the live data API stream for junk content."""
import os
os.environ["LOCLLM_FIM"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import re
import main_big as MB

MB._select_fim_mode_enable()
code_ids, name_by_id = MB.get_category_index()
MB.CODE_CATEGORY_IDS, MB.CAT_NAME_BY_ID = code_ids, name_by_id

fresh = MB.get_next_samples(512)
print("fetched:", len(fresh))
rows = []
for cat, toks in fresh:
    try:
        txt = MB.sp.decode(toks)
    except Exception:
        txt = ""
    n = len(txt)
    nonascii = sum(1 for c in txt if ord(c) > 127) / max(1, n)
    nl = txt.count("\n") / max(1, n)
    name_hits = len(re.findall(r"<name>", txt, re.I))
    repo_hits = len(re.findall(r"<reponame>", txt, re.I))
    base64ish = len(re.findall(r"[A-Za-z0-9+/]{100,}={0,2}", txt))
    rows.append((cat, name_by_id.get(cat, str(cat)), len(toks), n, nonascii, nl,
                 name_hits, repo_hits, base64ish, txt))

toks_lens = sorted(r[2] for r in rows)
print("tok len: p10", toks_lens[51], "p50", toks_lens[256], "p90", toks_lens[460], "max", toks_lens[-1])
susp = [r for r in rows if r[4] > 0.05 or r[8] > 0 or (r[3] > 2000 and r[5] < 0.005) or r[6] + r[7] > 5]
print("suspicious:", len(susp), "/", len(rows))
susp.sort(key=lambda r: -(r[4] + r[8] / 10 + (r[6] + r[7]) / 100 + (0.0002 * len(r[9]) if r[5] < 0.003 else 0)))
for cat, name, tl, chars, na, nl, nh, rh, b64, txt in susp[:12]:
    head = txt[:180].replace("\n", " ")
    print(f"--- cat {name} tok={tl} chars={chars} nonascii={na:.2f} nl={nl:.4f} <NAME>={nh} <repo>={rh} b64={b64}")
    print("   ", head)
