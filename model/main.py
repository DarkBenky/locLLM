import requests
import struct
import base64
import sentencepiece as spm
from pprint import pprint
from model import Transformer
import wandb

wandb.login()
wandb.init(project="locLMM", entity="darkbenky")

API = "http://localhost:8823"

VOCAB_SIZE = 32000

TOKENIZER_MODEL_PATH = "../tok/tokenize/tokenizer_models/tokenizer.model"
sp = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL_PATH)


def decodeToIdx(data: bytes) -> tuple[int, list[int]]:
    if len(data) < 8:
        raise ValueError(f"record too short: {len(data)} bytes")

    record_size = struct.unpack_from("<Q", data, 0)[0]
    category = data[8]
    token_count = (record_size - 1) // 2

    tokens = []
    offset = 9
    for _ in range(token_count):
        token = struct.unpack_from("<H", data, offset)[0]
        tokens.append(token)
        offset += 2

    return category, tokens


def tokensToText(tokens: list[int]) -> str:
    """Convert token indices back to readable text using the SentencePiece model."""
    return sp.decode(tokens)


def getNextSamples(count: int):
    res = requests.get(API + "/api/get-next-samples", params={"sample_count": count})
    return res.json()

if __name__ == "__main__":
    response = getNextSamples(1)
    raw_samples = response.get("samples", [])

    if not raw_samples:
        print("No samples returned (cursor may be at EOF).")
    else:
        for i, raw in enumerate(raw_samples):
            data = base64.b64decode(raw)
            category, tokens = decodeToIdx(data)
            text = tokensToText(tokens)
            print(f"--- Sample {i} ---")
            print(f"Category: {category}")
            print(f"Token count: {len(tokens)}")
            print(f"First 20 tokens: {tokens[:20]}")
            print(f"Decoded text: {text[:200]}...")
            print()
