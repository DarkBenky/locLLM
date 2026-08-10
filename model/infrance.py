import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from inference import BLOCK_SIZE, InferenceEngine

engine = InferenceEngine()
_lock = threading.Lock()
_started = time.time()

app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None


class FIMRequest(BaseModel):
    prefix: str
    suffix: str = ""
    max_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None


def sse_stream(tokens):
    buf = []
    prev = ""
    for tok in tokens:
        buf.append(tok)
        text = engine.decode(buf)
        chunk = text[len(prev):]
        prev = text
        if chunk:
            yield f"data: {json.dumps({'text': chunk})}\n\n"
    yield f"data: {json.dumps({'done': True, 'text': engine.decode(buf)})}\n\n"


@app.post("/generate")
def generate(req: GenerateRequest):
    ids = engine.sp.encode(req.prompt)

    def stream():
        with _lock:
            yield from sse_stream(
                engine.generate(ids, req.max_tokens, req.temperature,
                                req.top_k, req.top_p, req.seed,
                                stop_tokens={engine.fim_end, engine.im_end})
            )

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/generate_fim")
def generate_fim(req: FIMRequest):
    prefix_ids = engine.sp.encode(req.prefix)
    suffix_ids = engine.sp.encode(req.suffix)

    def stream():
        with _lock:
            yield from sse_stream(
                engine.generate_fim(prefix_ids, suffix_ids, req.max_tokens,
                                    req.temperature, req.top_k, req.top_p, req.seed)
            )

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "locLLM",
        "params": sum(p.numel() for p in engine.model.parameters()),
        "device": str(engine.device),
        "dtype": str(engine.dtype).replace("torch.", ""),
        "vocab_size": engine.vocab_size,
        "max_seq_len": BLOCK_SIZE,
        "uptime": round(time.time() - _started, 1),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)