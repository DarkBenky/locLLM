#!/usr/bin/env python3
"""Analyze diag_skip_micro batches.jsonl locally."""
import json
import statistics as st
from collections import Counter

recs = [json.loads(l) for l in open('batches.jsonl')]
losses = [r['loss'] for r in recs]
ntok = [r['n_tok'] for r in recs]
skips = [r for r in recs if r['skipped'] or r['loss'] > 5.0]

print(f"batches {len(recs)} | skips {len(skips)}")
print(f"loss: min {min(losses):.2f} p10 {sorted(losses)[20]:.2f} p50 {st.median(losses):.2f} "
      f"p90 {sorted(losses)[180]:.2f} max {max(losses):.2f}")
print(f"loss>3: {sum(1 for l in losses if l > 3)} | loss>5: {sum(1 for l in losses if l > 5)}")
print(f"n_tok: p10 {sorted(ntok)[20]} p50 {st.median(ntok):.0f} p90 {sorted(ntok)[180]}")
print(f"highest-loss batches: {sorted(((round(r['loss'],2), r['batch']) for r in recs), reverse=True)[:5]}")

# worst rows
rows = []
for b in recs:
    for rr in b['rows']:
        rows.append((b['loss'], b['n_tok'], b['batch'], rr['cat_name'], rr['fim'],
                     rr['n_tok'], rr['row_loss'], rr['name_hits'], rr['repo_hits'],
                     rr['html_hits'], rr['other_tag_hits'], rr['non_ascii'],
                     rr['text_head'][:130].replace('\n', ' ')))
rows.sort(reverse=True)
print("\nworst rows (bL=batch loss, bnt=batch n_tok, b=batch#, cat, fim, rt=row tok, rl=row loss, hits=NAME/repo/html/other, na=nonascii):")
for w in rows[:18]:
    print(f"  bL{w[0]:.2f} bnt{w[1]:5d} b{w[2]:3d} {w[3]:<13} fim={str(w[4]):5} rt{w[5]:5d} rl{w[6]:.2f} "
          f"hits={w[7]}/{w[8]}/{w[9]}/{w[10]} na={w[11]:.3f} | {w[12]}")

all_rows = [rr for b in recs for rr in b['rows']]
any_hit = sum(1 for rr in all_rows if rr['any_hit'])
name = sum(rr['name_hits'] for rr in all_rows)
repo = sum(rr['repo_hits'] for rr in all_rows)
html = sum(rr['html_hits'] for rr in all_rows)
na = [rr['non_ascii'] for rr in all_rows]
print(f"\nrows {len(all_rows)}: any_tag {any_hit} ({any_hit / len(all_rows) * 100:.1f}%) | "
      f"<NAME> {name} | <reponame> {repo} | html {html}")
print(f"non-ascii: mean {sum(na) / len(na) * 100:.2f}% max {max(na) * 100:.2f}% | "
      f">5%: {sum(1 for x in na if x > 0.05)} rows")
cc = Counter((rr['fim'], rr['cat_name']) for rr in all_rows)
print("\nrow categories (fim,cat -> n):")
for k, v in cc.most_common(15):
    print(f"  {k[0]!s:5} {k[1]:<14} {v}")
