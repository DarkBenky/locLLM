import ast
from datasets import load_dataset
from pprint import pprint
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException
import os
import json
import time
import random

API = "http://localhost:8823"
CACHE_DIR = "data/"
CHECKPOINT_FILE = "checkpoint.json"


def send_with_retry(url, payload, max_retries=5, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            res = requests.post(url, json=payload, timeout=30)
            if res.status_code == 200:
                return res
            if res.status_code >= 500:
                raise RequestException(f"HTTP {res.status_code}")
            return res
        except (ConnectionError, Timeout, RequestException) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"[retry] attempt {attempt+1}/{max_retries} after {delay:.1f}s: {e}")
            time.sleep(delay)
    return None


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            ckpt = json.load(f)
            print(f"[checkpoint] resuming from iter={ckpt['iter']:_}, tokens={ckpt['tokens']:_}")
            return ckpt
    return None

def save_checkpoint(_iter, tokenCount):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"iter": _iter, "tokens": tokenCount}, f)

def to_chatml(messages):
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    return "\n".join(parts)

def from_chatml(text):
    messages = []
    blocks = text.split("<|im_start|>")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        block = block.replace("<|im_end|>", "").strip()
        role, _, content = block.partition("\n")
        messages.append({"role": role.strip(), "content": content.strip()})
    return messages

def getNextSample():
    COMMON_LANG = ["Python", "JavaScript", "C++", "Java", "C", "Go", "TypeScript", "Ruby", "Rust", "PHP", "Swift", "C#", "Kotlin", "Scala", "Dart", "Objective-C", "Perl", "Lua", "SQL", "HTML", "CSS", "JSON", "YAML", "Markdown", "XML"]

    def stack_v3_gen():
        ds = load_dataset("HuggingFaceCode/stack-v3-train", split="train", streaming=True, cache_dir=CACHE_DIR)
        for repo in ds:
            for f in repo["files"]:
                category = f["language"]
                if f["language"] not in COMMON_LANG:
                    category = "OtherLanguage"
                yield {
                    "category": category,
                    "text": f["content"],
                }

    def reasoning_gen():
        ds = load_dataset("SupraLabs/reasoning-corpus-4K-5M-v1", split="train", streaming=True, cache_dir=CACHE_DIR)
        for repo in ds:
            yield {
                "text": repo["ChatML"],
                "category": "reasoning",
            }

    def code_instruction_gen():
        ds = load_dataset("TokenBender/code_instructions_122k_alpaca_style", split="train", streaming=True, cache_dir=CACHE_DIR)
        for repo in ds:
            yield {
                "text": repo["text"],
                "category": "instruction_code",
            }

    def open_math_gen():
        ds = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True, cache_dir=CACHE_DIR)
        for repo in ds:
            problem = repo["problem"]
            solution = repo["generated_solution"]
            _text = f"Problem: {problem}\nSolution: {solution}"
            yield {
                "text": _text,
                "category": "math",
            }

    def manusagents_gen():
        ds = load_dataset(
            "Manusagents/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-5-Mythos-5-Qwen-3.7-Max-and-more-Distillation-Dataset",
            split="train", streaming=True, cache_dir=CACHE_DIR
        )
        for repo in ds:
            try:
                raw = repo.get("instruction")
                resp = repo.get("response")
                if not raw or not resp:
                    continue
    
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                    except (json.JSONDecodeError, ValueError):
                        try:
                            parsed = ast.literal_eval(raw)
                        except (ValueError, SyntaxError):
                            continue
                elif isinstance(raw, dict):
                    parsed = raw
                else:
                    continue
    
                msgs = parsed.get("messages")
                if not isinstance(msgs, list):
                    continue
    
                parts = []
                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    r = m.get("role") or m.get("from")
                    c = m.get("content") or m.get("value")
                    if isinstance(r, str) and isinstance(c, str) and r and c:
                        parts.append(f"<|im_start|>{r}\n{c}<|im_end|>")
    
                if not parts or not isinstance(resp, str) or not resp.strip():
                    continue
    
                parts.append(f"<|im_start|>assistant\n{resp}<|im_end|>")
                yield {"text": "\n".join(parts), "category": "instruction"}
            except:
                continue

    def fineweb_gen():
        ds = load_dataset("m-a-p/FineFineWeb", split="train", streaming=True, cache_dir=CACHE_DIR)
        for repo in ds:
            if repo["language_score"] < 0.75:
                continue
            yield {
                "text": repo["text"],
                "category": "web",
            }

    def code_feedback_gen():
        ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True, cache_dir=CACHE_DIR)
        for repo in ds:
            prompt = repo["query"]
            response = repo["answer"]
            lang = repo["lang"]
            msg = to_chatml([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
            yield {
                "text": msg,
                "category": f"{lang}_instruction_code",
            }

    def nemotron_codealpaca_gen():
        ds = load_dataset("JessieWei/GLM-5.2-FP8-nemotron-codealpaca", split="train", streaming=True, cache_dir=CACHE_DIR)
        for repo in ds:
            conversation = repo["conversations"]
            msg = to_chatml(conversation)
            yield {
                "text": msg,
                "category": "instruction_code_alpaca",
            }

    def code_gen():
        ds = load_dataset("pengyunie/codesearchnet-codegen", split="train", streaming=True, cache_dir=CACHE_DIR)
        for repo in ds:
            problem = repo["problem"]
            output = repo["output"]
            lang = repo["language"]
            msg = to_chatml([{"role": "user", "content": problem}, {"role": "assistant", "content": output}])
            yield {
                "text": msg,
                "category": f"{lang}_instruction_code",
            }

    def tiny_codes_gen():
        ds = load_dataset("nampdn-ai/tiny-codes", split="train", streaming=True, cache_dir=CACHE_DIR)
        for repo in ds:
            lang = repo["programming_language"]
            problem = repo["prompt"]
            response = repo["response"]
            msg = to_chatml([{"role": "user", "content": problem}, {"role": "assistant", "content": response}])
            yield {
                "text": msg,
                "category": f"{lang}_instruction_code",
            }

    def code_security_gen():
        ds = load_dataset("ayshajavd/code-security-vulnerability-dataset", split="train", streaming=True, cache_dir=CACHE_DIR)
        for repo in ds:
            code = repo["code"]
            is_vulnerable = repo["is_vulnerable"]
            lang = repo["language"]
            prompt = f"Fix the following {lang} code:\n{code}\n"
            if is_vulnerable:
                code_fix = repo["code_fixed"]
                response = f"Fixed code:\n{code_fix}"
                msg = to_chatml([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
                yield {
                    "text": msg,
                    "category": f"{lang}_instruction_code",
                }
            else:
                response = f"The following {lang} code is not vulnerable"
                msg = to_chatml([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
                yield {
                    "text": msg,
                    "category": f"{lang}_instruction_code",
                }

    # gens = [stack_v3_gen(), reasoning_gen(), manusagents_gen(), fineweb_gen(), code_instruction_gen(), open_math_gen(), code_feedback_gen(), nemotron_codealpaca_gen(), code_gen(), tiny_codes_gen(), code_security_gen()]
    # gens = [stack_v3_gen(), code_feedback_gen(), nemotron_codealpaca_gen(), code_gen(), tiny_codes_gen(), code_security_gen()]
    gens = [stack_v3_gen()]
    active = list(range(len(gens)))

    while active:
        for i in active[:]:
            try:
                yield next(gens[i])
            except StopIteration:
                active.remove(i)


if __name__ == "__main__":
    gen = getNextSample()

    ckpt = load_checkpoint()
    start_iter = ckpt["iter"] if ckpt else 0
    tokenCount = ckpt["tokens"] if ckpt else 0
    _iter = 0

    if start_iter > 0:
        print(f"[checkpoint] skipping {start_iter:_} records...")
        for _ in range(start_iter):
            try:
                next(gen)
            except StopIteration:
                break
        print("[checkpoint] skip complete")

    lastUpdate = time.time()
    lastTokenCount = tokenCount

    for rec in gen:
        try:
            res = send_with_retry(API + "/api/receive-data", rec)
        except Exception as e:
            print(f"[fatal] send failed after retries: {e}")
            save_checkpoint(_iter + start_iter, tokenCount)
            print(f"[checkpoint] saved at iter={_iter+start_iter:_}, tokens={tokenCount:_}")
            raise

        if res.status_code != 200:
            continue

        tokenCount += res.json()["token_count"]
        _iter += 1

        if _iter % 16 == 0:
            now = time.time()
            delta_tokens = tokenCount - lastTokenCount
            delta_time = now - lastUpdate
            tokensPerSec = delta_tokens / delta_time if delta_time > 0 else 0
            lastTokenCount = tokenCount
            lastUpdate = now
            print(f"iter {_iter+start_iter:_} tokenCount {tokenCount:_} tokensPerSec {tokensPerSec:_.0f}")

        if _iter % 1000 == 0:
            save_checkpoint(_iter + start_iter, tokenCount)

    save_checkpoint(_iter + start_iter, tokenCount)
    print(f"Done — total iter={_iter+start_iter:_}, tokens={tokenCount:_}")
