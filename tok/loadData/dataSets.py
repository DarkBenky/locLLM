import ast
from datasets import load_dataset
from pprint import pprint
import requests
import os
import json
import time

API = "http://localhost:8823"
CACHE_DIR = "data/"

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

    gens = [stack_v3_gen(), reasoning_gen(), manusagents_gen(), fineweb_gen(), code_instruction_gen(), open_math_gen(), code_feedback_gen(), nemotron_codealpaca_gen(), code_gen(), tiny_codes_gen(), code_security_gen()]
    active = list(range(len(gens)))

    while active:
        for i in active[:]:
            try:
                yield next(gens[i])
            except StopIteration:
                active.remove(i)


if __name__ == "__main__":
    gen = getNextSample()
    _iter = 0
    tokenCount = 0
    lastUpdate = time.time()
    for rec in gen:
        # pprint(rec)
        # os._exit(0)
        res = requests.post(API+"/api/receive-data", json=rec)
        if res.status_code != 200:
            continue
        tokenCount += res.json()["token_count"]
        _iter += 1
        if _iter % 16 == 0:
            now = time.time()
            tokensPerSec = tokenCount / (now - lastUpdate)
            lastUpdate = now
            print(f"iter {_iter:_} tokenCount {tokenCount:_} tokensPerSec {tokensPerSec:.2f}")
            # os._exit(0)
    print("Done")
