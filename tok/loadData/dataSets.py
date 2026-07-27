import ast
from datasets import load_dataset
from pprint import pprint
import requests
import os

API = "http://localhost:8823"
CACHE_DIR = "data/"

def getNextSample():
    # ds = load_dataset("HuggingFaceCode/stack-v3-train", split="train", streaming=True, cache_dir=CACHE_DIR)
    # for repo in ds:
    #     for f in repo["files"]:
    #         yield {
    #             "category": f["language"],
    #             "text": f["content"],
    #         }

    # ds = load_dataset("SupraLabs/reasoning-corpus-4K-5M-v1", split="train", streaming=True, cache_dir=CACHE_DIR)
    # for repo in ds:
    #     yield {
    #         "text": repo["ChatML"],
    #         "category": "reasoning",
    #     }

    # ds = load_dataset("Manusagents/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-5-Mythos-5-Qwen-3.7-Max-and-more-Distillation-Dataset", split="train", streaming=True, cache_dir=CACHE_DIR)
    # for repo in ds:
    #     instructions = ast.literal_eval(repo["instruction"])["messages"]
    #     response = repo["response"]

    #     text = ""
    #     for msg in instructions:
    #         text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    #     text += f"<|im_start|>assistant\n{response}<|im_end|>"

    #     yield {
    #         "text": text,
    #         "category": repo["category"],
    #     }

    ds = load_dataset("m-a-p/FineFineWeb", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        if repo["language_score"] > 0.75:
            continue
        yield {
            "text": repo["text"],
            "category": "web",
        }


if __name__ == "__main__":
    gen = getNextSample()
    _iter = 0
    tokenCount = 0
    for rec in gen:
        pprint(rec)
        os._exit(0)
        res = requests.post(API+"/api/receive-data", json=rec)
        if res.status_code != 200:
            continue
        tokenCount += res.json()["token_count"]
        _iter += 1
        if _iter % 16 == 0:
            print(f"iter {_iter:_} tokenCount {tokenCount:_}")
            # os._exit(0)
    print("Done")
