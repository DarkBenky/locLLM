import ast
from datasets import load_dataset
from pprint import pprint
import requests
import os
import json

API = "http://localhost:8823"
CACHE_DIR = "data/"

def getNextSample():
    COMMON_LANG = ["Python", "JavaScript", "C++", "Java", "C", "Go", "TypeScript", "Ruby", "Rust", "PHP", "Swift", "C#", "Kotlin", "Scala", "Dart", "Objective-C", "Perl", "Lua", "SQL", "HTML", "CSS", "JSON", "YAML", "Markdown", "XML", "XML"]

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
                    except:
                        parsed = ast.literal_eval(raw)
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
            if repo["language_score"] > 0.75:
                continue
            yield {
                "text": repo["text"],
                "category": "web",
            }

    gens = [stack_v3_gen(), reasoning_gen(), manusagents_gen(), fineweb_gen()]
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
    for rec in gen:
        # pprint(rec)
        # os._exit(0)
        res = requests.post(API+"/api/receive-data", json=rec)
        if res.status_code != 200:
            continue
        tokenCount += res.json()["token_count"]
        _iter += 1
        if _iter % 16 == 0:
            print(f"iter {_iter:_} tokenCount {tokenCount:_}")
            # os._exit(0)
    print("Done")
