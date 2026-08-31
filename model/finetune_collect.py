#!/usr/bin/env python3
"""Collect autocomplete data for fine-tuning locLLM.

Merges two sources into model/log/finetune_pairs.jsonl:
  1. Continue's local dev-data events (~/.continue/dev_data/*/autocomplete.jsonl)
     — these carry `accepted: true/false` (accepted vs dismissed suggestions).
  2. The inference server log (model/log/completions.jsonl) — every completion
     the server generated (acceptance unknown -> accepted: null).

Usage:
    python model/finetune_collect.py                 # default
    python model/finetune_collect.py --out /path/x.jsonl
    python model/finetune_collect.py --min-completion 4   # drop tiny completions
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
CONTINUE_DEV_DIR = os.path.expanduser("~/.continue/dev_data")
SERVER_LOG = os.path.join(BASE, "log", "completions.jsonl")
DEFAULT_OUT = os.path.join(BASE, "log", "finetune_pairs.jsonl")


def normalize(row: dict, source: str) -> dict | None:
    completion = (row.get("completion") or "").strip()
    if not completion:
        return None
    prefix = row.get("prefix") or row.get("prompt") or ""
    suffix = row.get("suffix") or ""
    accepted = row.get("accepted")
    if isinstance(accepted, str):
        accepted = accepted.lower() in ("true", "1", "yes")
    return {
        "source": source,
        "prefix": prefix,
        "suffix": suffix,
        "completion": completion,
        "accepted": accepted if isinstance(accepted, bool) else None,
        "lang": row.get("lang"),
        "model": row.get("model") or row.get("modelTitle") or row.get("modelName"),
        "ts": row.get("ts") or row.get("timestamp"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--min-completion", type=int, default=0,
                    help="drop completions shorter than this many chars")
    ap.add_argument("--continue-dir", default=CONTINUE_DEV_DIR)
    ap.add_argument("--server-log", default=SERVER_LOG)
    args = ap.parse_args()

    rows = []

    # 1) Continue accept/reject events
    cont_files = sorted(glob.glob(os.path.join(args.continue_dir, "**", "*.jsonl"),
                                  recursive=True))
    for path in cont_files:
        if not os.path.basename(path).startswith("autocomplete"):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    n = normalize(row, "continue")
                    if n:
                        rows.append(n)
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: skipping {path}: {e}")

    # 2) server completions
    if os.path.exists(args.server_log):
        with open(args.server_log, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                n = normalize(row, "server")
                if n:
                    rows.append(n)

    # dedupe by (prefix, suffix, completion)
    seen = set()
    out = []
    for r in rows:
        if args.min_completion and len(r["completion"]) < args.min_completion:
            continue
        key = (r["prefix"], r["suffix"], r["completion"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    acc = sum(1 for r in out if r["accepted"] is True)
    rej = sum(1 for r in out if r["accepted"] is False)
    unk = sum(1 for r in out if r["accepted"] is None)
    cont = sum(1 for r in out if r["source"] == "continue")
    srv = sum(1 for r in out if r["source"] == "server")
    print(f"wrote {len(out)} samples -> {args.out}")
    print(f"  accepted={acc}  rejected={rej}  unknown(no accept signal)={unk}")
    print(f"  from Continue events: {cont} | from server log: {srv}")
    if cont == 0:
        print("NOTE: no Continue accept/reject events found yet. They are written by")
        print("  Continue to ~/.continue/dev_data/<version>/autocomplete.jsonl when")
        print("  tab-autocomplete suggestions are shown/accepted. Keep coding a bit,")
        print("  then re-run this script. The server log already captures everything.")


if __name__ == "__main__":
    main()
