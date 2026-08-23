from __future__ import annotations

import base64
import json
import os
import struct

import requests
import sentencepiece as spm

import time

FROM_API = "http://91.98.145.193:8823/api/"
TO_API = "http://localhost:8823/api/receive-data"

LANGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "langs.json")
TOKENIZER_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "tokenize", "tokenizer_models", "tokenizer.model",
)

def getSupportedLangs():
    langsDict = {}
    with open(LANGS_PATH, "r") as f:
        langs = json.load(f)
        for rec in langs:
            langsDict[rec["language"]] = rec["estimated_tokens"]
    return langsDict

def getCategories():
    res = requests.get(FROM_API+"get-category-index")
    return res.json()

LANG_ALIASES = {
    "cpp": "c++",
    "cxx": "c++",
    "cplusplus": "c++",
    "csharp": "c#",
    "js": "javascript",
    "ts": "typescript",
    "golang": "go",
    "py": "python",
}

_supported_map_cache = None


def _supported_map() -> dict[str, str]:
    global _supported_map_cache
    if _supported_map_cache is None:
        _supported_map_cache = {
            name.strip().lower(): name for name in getSupportedLangs()
        }
    return _supported_map_cache


def _resolve_language(category: str) -> str | None:
    key = category.strip().lower()
    if key in LANG_ALIASES:
        key = LANG_ALIASES[key]
    return _supported_map().get(key)


def parse_record(data: bytes) -> tuple[int, list[int]]:
    if len(data) < 9:
        raise ValueError(f"record too short: {len(data)} bytes")
    record_size = struct.unpack_from("<Q", data, 0)[0]
    category = data[8]
    token_count = (record_size - 1) // 2
    if 9 + 2 * token_count > len(data):
        raise ValueError(f"record size mismatch: header={record_size} data={len(data)}")
    tokens = list(struct.unpack_from(f"<{token_count}H", data, 9))
    return category, tokens


_sp = None


def _get_tokenizer():
    global _sp
    if _sp is None:
        _sp = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL_PATH)
    return _sp


def decode_tokens(tokens: list[int]) -> str:
    text = _get_tokenizer().decode(tokens)
    if text.endswith("</s>"):
        text = text[:-len("</s>")]
    return text


def getSamples(count: int, verbose: bool = False):
    cat_index = getCategories()
    id_to_name = {cid: name for name, cid in cat_index.items()}

    res = requests.get(FROM_API + "get-next-samples-random", params={"sample_count": count})
    res.raise_for_status()
    raw_samples = res.json().get("samples", [])

    samples = []
    dropped = 0
    for raw in raw_samples:
        try:
            data = base64.b64decode(raw)
            category_id, tokens = parse_record(data)
        except (ValueError, TypeError, struct.error):
            dropped += 1
            continue
        name = id_to_name.get(category_id)
        lang = _resolve_language(name) if name is not None else None
        if lang is None:
            dropped += 1
            continue
        samples.append({
            "category": lang,
            "tokens": tokens,
            "decoded_text": decode_tokens(tokens),
        })
    if verbose and dropped:
        print(f"getSamples: dropped {dropped}/{len(raw_samples)} samples (invalid/unsupported)")
    return samples


def submitData(data):
    for sample in data:
        request = {
            "category": sample["category"],
            "text": sample["decoded_text"],
        }
        res = requests.post(TO_API, json=request)
        res.raise_for_status()
        if res.status_code != 200:
            print(f"submitData: server returned status {res.status_code}: {res.text}")


if __name__ == "__main__":
    total_token_count = 0
    total_samples = 0
    start_all = time.time()
    while True:
        start = time.time()
        samples = getSamples(2048, verbose=True)
        batch_tokens = sum(len(s["tokens"]) for s in samples)
        total_token_count += batch_tokens
        total_samples += len(samples)

        submitData(samples)

        elapsed = time.time() - start
        elapsed_all = time.time() - start_all
        tps = batch_tokens / elapsed if elapsed > 0 else 0.0
        tps_all = total_token_count / elapsed_all if elapsed_all > 0 else 0.0
        print(
            f"sent {total_samples:_} samples, {total_token_count:_} total tokens, "
            f"{tps:_.0f} tok/s (avg {tps_all:_.0f} tok/s), batch {elapsed:.2f}s"
        )
    

   