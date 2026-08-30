import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from inference import BLOCK_SIZE, InferenceEngine
import chatml

engine = InferenceEngine()
_lock = threading.Lock()
_started = time.time()

# RAG service (tok/fimData/server.py) — mirrors the training-time retrieval
# in main_big.py (_fim_query / _make_context).
RAG_URL = os.environ.get("LOCLLM_RAG_URL", "http://localhost:8234")
CONTEXT_MAX_TOKENS = 1024
RAG_QUERY_MAX_TOKENS = 1536
RAG_LANG_ALIASES = {"golang": "go", "cpp": "c++"}
_rag_session = requests.Session()
_rag_warned = False

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
    lang: str | None = None
    context: str | None = None
    max_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None
    # RAG-conditioned infilling: retrieve code from the RAG DB and inject it
    # as <context_start><lang>..</lang>ctx...</context_end> (training format).
    use_rag: bool = False
    rag_top_k: int = 1
    rag_query: str | None = None


def _normalize_rag_lang(name) -> str:
    n = (name or "").strip().lower()
    return RAG_LANG_ALIASES.get(n, n)


def _rag_lang_block(name):
    """<lang>..</lang> token block, or None for generic/unknown (matches lang_block_ids)."""
    n = _normalize_rag_lang(name)
    if not n or n in ("star_coder", "otherlanguage", "other language", "code"):
        return None
    return [engine.lang_open, *engine.sp.encode(n, out_type=int), engine.lang_close]


def _rag_search(query: str, top_k: int):
    """Query the RAG DB (port 8234) and return the best hit dict or None."""
    global _rag_warned
    try:
        resp = _rag_session.post(
            RAG_URL + "/search", json={"texts": [query], "top_k": top_k}, timeout=10)
        data = resp.json()
        for item in data:
            results = item.get("results") or []
            if results:
                return results[0]
    except Exception as e:  # noqa: BLE001 - RAG is an optional enhancement
        if not _rag_warned:
            print(f"WARNING: RAG search failed: {e}")
            _rag_warned = True
    return None


def _build_rag_context(prefix: str, suffix: str, top_k: int,
                       query_override: str | None = None,
                       max_new_tokens: int = 256):
    """Build the inner context token block for generate_fim (no ctx markers).

    Mirrors main_big.py _fim_query (prefix + suffix, 1536-token cap) and
    _make_context (<lang>..</lang> + code, capped at CONTEXT_MAX_TOKENS with
    room for the FIM prompt + max_new_tokens).
    """
    query = (query_override or (prefix + "\n" + suffix)).strip()
    if not query:
        return None
    qids = engine.sp.encode(query, out_type=int)
    if len(qids) > RAG_QUERY_MAX_TOKENS:
        qids = qids[:RAG_QUERY_MAX_TOKENS]
        query = engine.sp.decode(qids)

    record = _rag_search(query, top_k)
    if not record:
        return None
    code = record.get("code") or ""
    ctx = engine.sp.encode(code, out_type=int)
    if len(ctx) < 8:
        return None

    lang_block = _rag_lang_block(record.get("lang"))
    overhead = (len(lang_block) if lang_block else 0) + 1
    pref_ids = engine.sp.encode(prefix)
    suf_ids = engine.sp.encode(suffix)
    room = BLOCK_SIZE - max_new_tokens - len(pref_ids) - len(suf_ids) - 48
    limit = min(CONTEXT_MAX_TOKENS, room - overhead)
    if limit < 8:
        return None

    ids = []
    if lang_block:
        ids.extend(lang_block)
    ids.extend(ctx[:limit])
    return ids


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


def _run_fim(req: FIMRequest):
    prefix_ids = engine.sp.encode(req.prefix)
    suffix_ids = engine.sp.encode(req.suffix)
    if req.context is not None:
        context_ids = engine.sp.encode(req.context)
    elif req.use_rag:
        context_ids = _build_rag_context(req.prefix, req.suffix, req.rag_top_k,
                                         req.rag_query, req.max_tokens)
    else:
        context_ids = None

    def stream():
        with _lock:
            yield from sse_stream(
                engine.generate_fim(prefix_ids, suffix_ids, req.max_tokens,
                                    req.temperature, req.top_k, req.top_p, req.seed,
                                    lang=req.lang, context_ids=context_ids)
            )

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/generate_fim")
def generate_fim(req: FIMRequest):
    """FIM infilling without RAG (use_rag=false) or with RAG (use_rag=true)."""
    return _run_fim(req)


@app.post("/generate_fim_rag")
def generate_fim_rag(req: FIMRequest):
    """Convenience route: always conditions on RAG-retrieved context."""
    req.use_rag = True
    return _run_fim(req)


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