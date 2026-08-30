#!/usr/bin/env python3
"""One-command SAFIM evaluation for locLLM.

Runs the full Syntax-Aware Fill-in-the-Middle benchmark (block / control / api)
against the locLLM model served by model/infrance.py, using the official
methodology (ExecEval daemon for execution-based scoring), and prints a
leaderboard-style pass@1 table.

Usage:
    ./model/eval.py                     # full run (block + control + api)
    ./model/eval.py --tasks api         # quick signal (310 samples)
    ./model/eval.py --tasks api --limit 5   # smoke test
    ./model/eval.py --tasks control --limit 100 --generate-only
    ./model/eval.py --setup             # check deps + daemon instructions

Generation settings (SAFIM official): temperature 0.2, top_p 0.95, 128 tokens.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import httpx
import tqdm
from datasets import load_dataset

from safim_helpers import (
    COMPLETION_PLACEHOLDER, LANG_MAP, POST_PROCESSORS,
    ExecEvalClient, apply_postprocessors, check_syntax, get_infilling_parts,
    syntax_match,
)

TASKS = ("block", "control", "api", "block_v2")
DEFAULT_OUTDIR = os.path.join(BASE, "safim")
DEFAULT_SERVER_URL = "http://localhost:8000"
DEFAULT_DAEMON_URL = "http://localhost:5000"

# Reference pass@1 (%) from the official leaderboard (for context only).
REFERENCE = {
    "DeepSeek-Coder-1.3B": {"block": 41.2, "control": 54.1, "api": 62.6},
    "StarCoder": {"block": 44.1, "control": 54.5, "api": 68.1},
    "WizardCoder-1B": {"block": 28.1, "control": 40.0, "api": 57.4},
    "CodeGen-2B": {"block": 23.5, "control": 32.9, "api": 32.3},
    "InCoder-1B": {"block": 21.1, "control": 22.9, "api": 43.9},
    "GPT-3.5": {"block": 31.2, "control": 37.5, "api": 53.9},
}


# ---------------------------------------------------------------------------
# Inference server client
# ---------------------------------------------------------------------------

class ServerClient:
    def __init__(self, server_url: str):
        self.url = server_url.rstrip("/")
        self._client = httpx.Client(timeout=httpx.Timeout(300.0, connect=5.0))

    def health(self, timeout: float = 2.0):
        try:
            r = self._client.get(f"{self.url}/health", timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def generate_fim(self, prefix: str, suffix: str, lang: str | None,
                     max_tokens: int, temperature: float, top_p: float,
                     seed: int | None, use_rag: bool = False,
                     retries: int = 3) -> str:
        body = {
            "prefix": prefix, "suffix": suffix, "lang": lang,
            "max_tokens": max_tokens, "temperature": temperature,
            "top_k": 0, "top_p": top_p, "seed": seed, "use_rag": use_rag,
        }
        last_err = None
        for attempt in range(retries):
            try:
                with self._client.stream("POST", f"{self.url}/generate_fim",
                                         json=body) as r:
                    r.raise_for_status()
                    text = ""
                    final = None
                    for line in r.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = json.loads(line[len("data: "):])
                        if payload.get("done"):
                            final = payload.get("text", text)
                        else:
                            text += payload.get("text", "")
                    return final if final is not None else text
            except Exception as e:  # noqa: BLE001 - retry on any transport error
                last_err = e
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"generate_fim failed after {retries} attempts: {last_err}")


def pick_gpu():
    """Pick the CUDA device (torch index) with the most free memory."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None
        best, best_free = None, -1.0
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            if free > best_free:
                best, best_free = i, free / 1024 ** 3
        return best, best_free
    except Exception:
        return None, None


def auto_start_server(server_url: str, outdir: str, use_cpu: bool,
                      gpu: int | None, log_path: str) -> tuple[subprocess.Popen | None, bool]:
    """Start model/infrance.py if the server is not already up.

    Returns (process, started_here)."""
    client = ServerClient(server_url)
    if client.health():
        return None, False

    env = dict(os.environ)
    if use_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        idx, free = pick_gpu() if gpu is None else (gpu, None)
        if idx is None:
            print("[eval] no CUDA GPU available; starting server on CPU")
            env["CUDA_VISIBLE_DEVICES"] = ""
        else:
            if free is not None and free < 6.0:
                print(f"[eval] WARNING: only {free:.1f} GB free on GPU {idx}; "
                      f"the server needs ~7 GB (bf16). It may OOM.")
            env["CUDA_VISIBLE_DEVICES"] = str(idx)
            print(f"[eval] starting inference server on CUDA device {idx} "
                  f"(torch index; see nvidia-smi for physical GPU)")

    log_f = open(log_path, "a", buffering=1)
    log_f.write(f"\n=== server start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"cwd={BASE} ===\n")
    log_f.flush()
    proc = subprocess.Popen(
        [sys.executable, os.path.join(BASE, "infrance.py")],
        cwd=BASE, env=env, stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"[eval] server pid={proc.pid}, waiting for {server_url}/health ...")
    deadline = time.time() + 300
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"[eval] server exited early (code {proc.returncode}). "
                  f"See {log_path}")
            return None, False
        if client.health():
            print(f"[eval] server is up")
            return proc, True
        time.sleep(2)
    print(f"[eval] timeout waiting for server; see {log_path}")
    return None, False


# ---------------------------------------------------------------------------
# Dataset + generation + evaluation
# ---------------------------------------------------------------------------

def load_task(task: str, limit: int = 0) -> list[dict]:
    ds = load_dataset("gonglinyuan/safim", task, split="test")
    rows = []
    for m in ds:
        m = dict(m)
        raw = m.get("unit_tests")
        m["unit_tests"] = json.loads(raw) if isinstance(raw, str) and raw.strip() else []
        rows.append(m)
        if limit and len(rows) >= limit:
            break
    return rows


def load_outputs(path: str) -> dict[str, str]:
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                out[d["task_id"]] = d["completion"]
    return out


def generate_task(client: ServerClient, task: str, rows: list[dict],
                  outdir: str, args, tag: str = "",
                  use_rag: bool = False) -> tuple[dict[str, str], int]:
    outputs_path = os.path.join(outdir, f"outputs_{task}{tag}.jsonl")
    errors_path = os.path.join(outdir, f"errors_{task}{tag}.log")
    done = load_outputs(outputs_path)
    if args.refresh:
        done = {}
        for p in (outputs_path, os.path.join(outdir, f"results_{task}{tag}.jsonl")):
            if os.path.exists(p):
                os.remove(p)
    n_err = 0
    with open(outputs_path, "a", encoding="utf-8") as fo, \
            open(errors_path, "a", encoding="utf-8") as fe:
        for row in tqdm.tqdm(rows, desc=f"[{task}{tag}] generate", unit="sample"):
            tid = row["task_id"]
            if tid in done:
                continue
            try:
                prefix, suffix = get_infilling_parts(row)
                lang = LANG_MAP.get(row["lang"])
                raw = client.generate_fim(
                    prefix, suffix, lang=lang,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    top_p=args.top_p, seed=args.seed, use_rag=use_rag,
                )
                completion = apply_postprocessors(raw, row, task, POST_PROCESSORS[task])
            except Exception as e:  # noqa: BLE001
                fe.write(f"{tid}\t{type(e).__name__}: {e}\n")
                fe.flush()
                n_err += 1
                continue
            fo.write(json.dumps({"task_id": tid, "completion": completion}) + "\n")
            fo.flush()
            done[tid] = completion
    return done, n_err


def evaluate_task(task: str, rows: list[dict], outputs: dict[str, str],
                  outdir: str, evalc: ExecEvalClient, tag: str = "") -> list[dict]:
    results_path = os.path.join(outdir, f"results_{task}{tag}.jsonl")
    ev = []
    for problem in tqdm.tqdm(rows, desc=f"[{task}] evaluate", unit="sample"):
        tid = problem["task_id"]
        completion = outputs.get(tid)
        if tid not in outputs:
            result, passed = "EMPTY", False
        else:
            if problem.get("unit_tests"):
                if completion == problem["ground_truth"]:
                    result, passed = "PASSED", True
                else:
                    result, passed = evalc.run_test(
                        problem, {"task_id": tid, "completion": completion})
            else:
                if syntax_match(completion, problem["ground_truth"], problem["lang"]):
                    result, passed = "EXACT_MATCH", True
                else:
                    result, passed = "WRONG_ANSWER", False
            if not completion.strip() and not passed:
                result, passed = "EMPTY", False
        if problem["lang"] == "python" and not passed:
            full_code = problem["eval_prompt"].replace(
                "{{completion}}", completion if completion is not None else "")
            if "unit_tests" in problem and not check_syntax(full_code):
                result = "COMPILATION_ERROR"
        ev.append({"task_id": tid, "result": result, "passed": passed,
                   "check_result": 0})

    with open(results_path, "w", encoding="utf-8") as f:
        for r in ev:
            f.write(json.dumps(r) + "\n")
    return ev


def summarize(task: str, rows: list[dict], ev: list[dict]):
    lang_of = {p["task_id"]: p["lang"] for p in rows}
    pass_cnt = defaultdict(int)
    total = defaultdict(int)
    fail_hist = defaultdict(int)
    for r in ev:
        lang = lang_of[r["task_id"]]
        for key in (lang, "all"):
            total[key] += 1
            if r["passed"]:
                pass_cnt[key] += 1
        if not r["passed"]:
            res = r["result"]
            if isinstance(res, str):
                if res != "EXACT_MATCH":
                    fail_hist[res] += 1
            elif isinstance(res, list):
                s = {o.get("exec_outcome") for o in res if o.get("exec_outcome") != "PASSED"}
                if len(s) > 1:
                    fail_hist["MIXED"] += 1
                elif s:
                    fail_hist[list(s)[0]] += 1
                else:
                    fail_hist["WRONG_ANSWER"] += 1
    pct = {k: (pass_cnt[k] / total[k] * 100.0 if total[k] else 0.0)
           for k in total}
    return pct, pass_cnt, total, fail_hist


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_task_row(task: str, pct: dict, total: dict, fail_hist: dict,
                   daemon_up: bool):
    langs = ["cpp", "java", "python", "csharp", "all"]
    row = ",".join(f"{pct.get(l, 0.0):.6f}" for l in langs)
    hist = ",".join(str(fail_hist.get(o, 0)) for o in
                    ["EMPTY", "COMPILATION_ERROR", "RUNTIME_ERROR",
                     "MEMORY_LIMIT_EXCEEDED", "TIME_LIMIT_EXCEEDED",
                     "WRONG_ANSWER", "MIXED"])
    print(f"\n=== {task} (n={total.get('all', 0)}) ===")
    print(f"task,{row}")
    print(f"lang pass@1: " + ", ".join(f"{l}={pct.get(l, 0.0):.2f}%" for l in langs))
    print(f"failed outcomes: {dict(fail_hist)}")
    if task in ("block", "block_v2", "control") and not daemon_up:
        print("NOTE: ExecEval daemon not reachable on :5000 -> block/control "
              "were scored by EXACT ground-truth match only, NOT execution. "
              "Those numbers are NOT directly comparable to the leaderboard.")
    if task == "block_v2":
        print("NOTE: block_v2 is the contamination-check split (no leaderboard entry).")


def print_comparison(results: dict[str, dict]):
    print("\n=== Comparison (pass@1 %, all languages) vs leaderboard ===")
    print(f"{'model':<22}{'block':>8}{'control':>9}{'api':>7}{'avg':>7}")
    ours = results
    print(f"{'locLLM (this run)':<22}"
          f"{ours.get('block', {}).get('all', float('nan')):>8.2f}"
          f"{ours.get('control', {}).get('all', float('nan')):>9.2f}"
          f"{ours.get('api', {}).get('all', float('nan')):>7.2f}"
          f"{'':>7}")
    for name, vals in REFERENCE.items():
        avg = (vals["block"] + vals["control"] + vals["api"]) / 3.0
        print(f"{name:<22}{vals['block']:>8.1f}{vals['control']:>9.1f}"
              f"{vals['api']:>7.1f}{avg:>7.1f}")


# ---------------------------------------------------------------------------
# Setup / checks
# ---------------------------------------------------------------------------

def cmd_setup(args):
    print("[setup] checking Python environment ...")
    required = ["datasets", "httpx", "requests", "tqdm", "tree_sitter",
                "tree_sitter_language_pack"]
    import importlib.util
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if missing:
        print(f"[setup] MISSING pip packages (install in the env you run "
              f"eval.py with): {', '.join(missing)}\n"
              f"  pip install {' '.join(missing)}")
    else:
        print("[setup] all eval dependencies present")

    if not os.path.exists(os.path.join(BASE, "..", "tok", "tokenize",
                                       "tokenizer_models", "tokenizer.model")):
        print("[setup] WARNING: tokenizer model not found at "
              "tok/tokenize/tokenizer_models/tokenizer.model")

    client = ServerClient(DEFAULT_SERVER_URL)
    health = client.health()
    print(f"[setup] inference server :8000 -> "
          f"{'UP ' + str(health.get('model')) if health else 'DOWN'}")
    daemon = ExecEvalClient(DEFAULT_DAEMON_URL)
    up = daemon.available()
    print(f"[setup] ExecEval daemon :5000 -> {'UP' if up else 'DOWN'}")

    docker = shutil.which("docker")
    if docker:
        print(f"[setup] docker: {docker}")
        if not up:
            print("[setup] start the daemon with:\n"
                  "  docker run -d -p 5000:5000 -e NUM_WORKERS=2 exec-eval:1.0")
    else:
        print("[setup] docker: NOT FOUND. To run the official execution-based "
              "scoring for block/control (leaderboard-comparable), install "
              "Docker CE, then:\n"
              "  git clone https://github.com/ntunlp/ExecEval\n"
              "  cd ExecEval && docker build . -t exec-eval:1.0\n"
              "  docker run -d -p 5000:5000 -e NUM_WORKERS=2 exec-eval:1.0\n"
              "Without it, eval.py falls back to exact-match scoring "
              "(api stays AST-scored).")
    return


def print_rag_delta(results_by_task: dict, tasks: list[str]):
    print("\n=== RAG vs no-RAG (pass@1 %, all languages) ===")
    print(f"{'task':<10}{'no-RAG':>10}{'RAG':>10}{'delta':>10}")
    for task in tasks:
        off = results_by_task.get((task, "off"), {}).get("all")
        on = results_by_task.get((task, "on"), {}).get("all")
        if off is None or on is None:
            continue
        print(f"{task:<10}{off:>10.2f}{on:>10.2f}{on - off:>+10.2f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Run the SAFIM benchmark against locLLM (one command).")
    ap.add_argument("--tasks", nargs="+", choices=TASKS + ("all",),
                    default=["block", "control", "api"],
                    help="subtasks to run (default: full benchmark block+control+api; "
                         "'all' = same as default)")
    ap.add_argument("--limit", type=int, default=0, help="max samples per task (0=all)")
    ap.add_argument("--refresh", action="store_true", help="ignore cached outputs")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--rag", choices=["off", "on", "both"], default="off",
                    help="condition on RAG-retrieved context: off = no RAG, "
                         "on = RAG context, both = run both and compare")
    ap.add_argument("--server-url", dest="server_url", default=DEFAULT_SERVER_URL)
    ap.add_argument("--no-start", action="store_true",
                    help="do NOT auto-start infrance.py; fail if server is down")
    ap.add_argument("--cpu", action="store_true", help="start server on CPU")
    ap.add_argument("--gpu", type=int, default=None,
                    help="torch CUDA index for the server (default: freest GPU)")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--setup", action="store_true")
    args = ap.parse_args()
    if "all" in args.tasks:
        args.tasks = ["block", "control", "api"]

    if args.setup:
        cmd_setup(args)
        return

    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "server.log")

    proc, started = None, False
    needs_server = not args.evaluate_only
    if needs_server and not args.no_start:
        proc, started = auto_start_server(args.server_url, args.outdir,
                                          args.cpu, args.gpu, log_path)
    client = ServerClient(args.server_url)
    if needs_server and not client.health():
        print(f"[eval] server not reachable at {args.server_url}. Start it with:\n"
              f"  cd {BASE} && python infrance.py\n"
              f"(or run with --cpu/--gpu to let eval.py start it)")
        sys.exit(1)

    evalc = ExecEvalClient(DEFAULT_DAEMON_URL)
    daemon_up = evalc.available()
    if not daemon_up:
        print("[eval] WARNING: ExecEval daemon (:5000) not reachable. "
              "block/control will use exact-match fallback (not comparable "
              "to the leaderboard). See ./model/eval.py --setup")

    results_by_task = {}
    modes = ["on", "off"] if args.rag == "both" else (
        ["on"] if args.rag == "on" else ["off"])
    if args.rag == "both":
        print(f"[eval] full benchmark: {', '.join(args.tasks)} — running both "
              f"modes (no-RAG then RAG); use --task/--limit/--refresh to narrow.")
    for mode in modes:
        tag = "_rag" if mode == "on" else ""
        print(f"\n########## mode: {'RAG' if mode == 'on' else 'no-RAG'} ##########")
        for task in args.tasks:
            print(f"\n=== {task} ===")
            rows = load_task(task, args.limit)
            print(f"[{task}] {len(rows)} samples")
            if not rows:
                continue
            if not args.evaluate_only:
                outputs, n_err = generate_task(client, task, rows, args.outdir,
                                               args, tag=tag, use_rag=(mode == "on"))
                if n_err:
                    print(f"[{task}] {n_err} generation errors logged to "
                          f"{os.path.join(args.outdir, f'errors_{task}{tag}.log')}")
            else:
                outputs = load_outputs(os.path.join(
                    args.outdir, f"outputs_{task}{tag}.jsonl"))
            if not args.generate_only and outputs:
                ev = evaluate_task(task, rows, outputs, args.outdir, evalc, tag=tag)
                pct, pass_cnt, total, fail_hist = summarize(task, rows, ev)
                print_task_row(task, pct, total, fail_hist, daemon_up)
                results_by_task[(task, mode)] = pct
            elif args.generate_only:
                print(f"[{task}] generation only: {len(outputs)} completions")

    if args.rag == "both":
        print_rag_delta(results_by_task, list(args.tasks))
    leaderboard_results = {}
    for mode in ("off", "on"):
        for task in args.tasks:
            if (task, mode) in results_by_task:
                leaderboard_results[task] = results_by_task[(task, mode)]
        if leaderboard_results:
            break
    if leaderboard_results:
        print_comparison(leaderboard_results)

    if started and proc is not None:
        print(f"\n[eval] inference server left running (pid {proc.pid}, log: {log_path}). "
              f"Stop it with: kill {proc.pid}")
        # NOTE: could add --stop-server later; leaving it hot helps re-runs.


if __name__ == "__main__":
    main()
