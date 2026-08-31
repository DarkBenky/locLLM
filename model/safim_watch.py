#!/usr/bin/env python3
"""Lightweight progress dashboard for the SAFIM full benchmark run.

Usage:
    python model/safim_watch.py                    # compact counts every 60s
    python model/safim_watch.py --interval 10      # faster refresh
    python model/safim_watch.py --until api        # exit + banner when api (RAG) finishes
    python model/safim_watch.py --detail           # also print last PASS/FAIL cases (slower)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
TOTALS = {"block": 8781, "control": 8629, "api": 310}
MODES = [("on", "RAG"), ("off", "-")]


def counts(outdir: str) -> dict:
    out = {}
    for task in TOTALS:
        for mode, _ in MODES:
            tag = "_rag" if mode == "on" else ""
            path = os.path.join(outdir, f"outputs_{task}{tag}.jsonl")
            n = 0
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    n = sum(1 for _ in f)
            out[(task, mode)] = n
    return out


def results_done(outdir: str) -> list[str]:
    done = []
    for task in TOTALS:
        for mode, _ in MODES:
            tag = "_rag" if mode == "on" else ""
            p = os.path.join(outdir, f"results_{task}{tag}.jsonl")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    passed = sum(1 for line in f if json.loads(line).get("passed"))
                done.append(f"{task}{tag} ({passed} pass)")
    return done


def line(outdir: str, done: list[str]) -> str:
    c = counts(outdir)
    parts = [f"{t}[{m}] {c[(t, m)]}/{TOTALS[t]}" for t in TOTALS for m, _ in MODES]
    return f"{time.strftime('%H:%M:%S')} | " + " | ".join(parts) + \
        (f" | done: {', '.join(done)}" if done else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=os.path.join(BASE, "safim"))
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--until", choices=["api"], default=None,
                    help="exit when api (RAG mode) results are written")
    ap.add_argument("--detail", action="store_true",
                    help="also run the full --status every interval (slower)")
    args = ap.parse_args()

    done_prev = set()
    while True:
        done = results_done(args.outdir)
        print(line(args.outdir, done), flush=True)
        if args.detail:
            subprocess.run([sys.executable, os.path.join(BASE, "eval.py"),
                            "--status", "--rag", "both", "--show", "2",
                            "--outdir", args.outdir], check=False)
        if args.until == "api":
            c = counts(args.outdir)
            if c[("api", "on")] >= TOTALS["api"] and c[("api", "off")] >= TOTALS["api"]:
                stats = []
                for mode, tag in (("on", "_rag"), ("off", "")):
                    path = os.path.join(args.outdir, f"results_api{tag}.jsonl")
                    if os.path.exists(path):
                        with open(path, encoding="utf-8") as f:
                            rows = [json.loads(x) for x in f]
                        passed = sum(1 for r in rows if r.get("passed"))
                        stats.append(f"api{tag}: {passed}/{len(rows)} "
                                     f"({passed / max(1, len(rows)) * 100:.2f}%)")
                print(f"\n>>> API FINISHED (both modes): {', '.join(stats) or 'no results yet'}. "
                      f"Run --status for the detailed table.", flush=True)
                return
        # notify when a new results file appears
        if set(done) - done_prev:
            print(">>> new results:", ", ".join(sorted(set(done) - done_prev)),
                  flush=True)
        done_prev = set(done)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()