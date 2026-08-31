import asyncio
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.abspath(__file__))

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from inference import BLOCK_SIZE, InferenceEngine
import chatml
from lang_detect import infer_code_lang

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

# Repetition suppression defaults (used when clients send 0 / omit them).
# frequency: penalizes tokens by how often they were generated; presence:
# penalizes any token that already appeared. Tune FREQ up for stronger
# anti-repeat, PRES up to avoid echoing identifiers.
FREQ_PENALTY_DEFAULT = 0.8
PRESENCE_PENALTY_DEFAULT = 0.2

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
    # Repetition control (OpenAI-compatible). Values <= 0 fall back to the
    # server defaults (FREQ_PENALTY_DEFAULT / PRESENCE_PENALTY_DEFAULT) so
    # autocomplete stops repeating itself even when clients send 0.
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class OpenAICompletionRequest(BaseModel):
    """OpenAI-compatible /v1/completions (for Continue tab-autocomplete)."""
    model: str = "locllm-1.6b"
    prompt: str
    suffix: str | None = None
    max_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.95
    stream: bool = False
    stop: str | list[str] | None = None
    # locLLM extras: language conditioning (auto-inferred from input if unset)
    # and RAG context retrieval (like /generate_fim use_rag).
    lang: str | None = None
    use_rag: bool = False
    # Repetition control (OpenAI-compatible); <= 0 -> server defaults.
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class OpenAIChatRequest(BaseModel):
    """OpenAI-compatible /v1/chat/completions (chat test with the FIM model)."""
    model: str = "locllm-1.6b"
    messages: list[dict]
    max_tokens: int = 256
    temperature: float = 0.3
    top_p: float = 0.95
    stream: bool = False
    stop: str | list[str] | None = None


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


def _next_or_stop(gen):
    try:
        return "ok", next(gen)
    except StopIteration:
        return "stop", None


async def _iter_generation(gen):
    """Stream tokens from the engine's sync generator without buffering.

    The GPU lock is held for the WHOLE generation (the engine shares one KV
    cache, so runs must not interleave). Each token is pulled via a worker
    thread; if the client disconnects, the in-flight decode step is allowed
    to finish (bounded to ~1 step) before the lock is released, so cancelled
    streams can never wedge subsequent requests.
    """
    with _lock:
        while True:
            task = asyncio.create_task(asyncio.to_thread(_next_or_stop, gen))
            try:
                status, tok = await asyncio.shield(task)
            except asyncio.CancelledError:
                await asyncio.shield(task)
                raise
            if status == "stop":
                return
            yield tok


# --- completion logging (for later fine-tuning) -----------------------------

_COMPLETIONS_LOG = os.path.join(BASE, "log", "completions.jsonl")


def _log_completion(entry: dict):
    """Best-effort append to model/log/completions.jsonl."""
    try:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
        with open(_COMPLETIONS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001 - logging must never break serving
        print(f"WARNING: completion logging failed: {e}")


@app.post("/generate")
def generate(req: GenerateRequest):
    ids = engine.sp.encode(req.prompt)

    async def stream():
        t0 = time.time()
        buf, prev = [], ""
        async for tok in _iter_generation(engine.generate(
                ids, req.max_tokens, req.temperature,
                req.top_k, req.top_p, req.seed,
                stop_tokens={engine.fim_end, engine.im_end})):
            buf.append(tok)
            text = engine.decode(buf)
            chunk = text[len(prev):]
            prev = text
            if chunk:
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield f"data: {json.dumps({'done': True, 'text': engine.decode(buf)})}\n\n"
        _log_completion({
            "kind": "plain", "model": "locllm-1.6b", "stream": True,
            "prompt": req.prompt, "suffix": "", "lang": None, "use_rag": False,
            "completion": engine.decode(buf), "n_tokens": len(buf),
            "latency_s": round(time.time() - t0, 3), "endpoint": "/generate",
        })

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

    async def stream():
        t0 = time.time()
        buf, prev = [], ""
        freq, pres = _penalties(req)
        async for tok in _iter_generation(engine.generate_fim(
                prefix_ids, suffix_ids, req.max_tokens,
                req.temperature, req.top_k, req.top_p, req.seed,
                lang=req.lang, context_ids=context_ids,
                frequency_penalty=freq, presence_penalty=pres)):
            buf.append(tok)
            text = engine.decode(buf)
            chunk = text[len(prev):]
            prev = text
            if chunk:
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield f"data: {json.dumps({'done': True, 'text': engine.decode(buf)})}\n\n"
        _log_completion({
            "kind": "fim", "model": "locllm-1.6b", "stream": True,
            "prompt": req.prefix, "suffix": req.suffix,
            "lang": req.lang, "use_rag": req.use_rag,
            "completion": engine.decode(buf), "n_tokens": len(buf),
            "latency_s": round(time.time() - t0, 3), "endpoint": "/generate_fim",
        })

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


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints (for Continue tab-autocomplete, LM Studio, etc.)
# ---------------------------------------------------------------------------

def _penalties(req) -> tuple[float, float]:
    freq = req.frequency_penalty if req.frequency_penalty > 0 else FREQ_PENALTY_DEFAULT
    pres = req.presence_penalty if req.presence_penalty > 0 else PRESENCE_PENALTY_DEFAULT
    return freq, pres


def _openai_token_iter(req: OpenAICompletionRequest):
    """Lazy FIM generator for /v1/completions (used by true streaming).

    Always uses the trained FIM format (<fim_prefix><lang>..</lang>pre<fim_suffix>
    ..<fim_middle>); suffix is empty when the client didn't send one, so the
    model still gets its language tag based on the input code.
    """
    prompt_ids = engine.sp.encode(req.prompt)
    suffix_text = (req.suffix or "").strip()
    suffix_ids = engine.sp.encode(suffix_text) if suffix_text else []
    lang = req.lang or infer_code_lang(req.prompt)
    context_ids = None
    if req.use_rag:
        context_ids = _build_rag_context(req.prompt, suffix_text, 1, None,
                                         req.max_tokens)
    freq, pres = _penalties(req)
    yield from engine.generate_fim(prompt_ids, suffix_ids, req.max_tokens,
                                   req.temperature, 0, req.top_p, None,
                                   lang=lang, context_ids=context_ids,
                                   frequency_penalty=freq, presence_penalty=pres)


def _openai_tokens(req: OpenAICompletionRequest):
    return list(_openai_token_iter(req))


@app.get("/v1/models")
def v1_models():
    return {"object": "list",
            "data": [{"id": "locllm-1.6b", "object": "model",
                      "owned_by": "locllm"}]}


MAX_OAI_TOKENS = 128     # autocomplete hard cap
MAX_CHAT_TOKENS = 256    # chat hard cap


@app.post("/v1/completions")
def v1_completions(req: OpenAICompletionRequest):
    req.max_tokens = max(1, min(req.max_tokens, MAX_OAI_TOKENS))
    lang = req.lang or infer_code_lang(req.prompt)

    def _log(tokens, t0, streamed):
        _log_completion({
            "kind": "openai", "model": req.model, "stream": streamed,
            "prompt": req.prompt, "suffix": req.suffix or "",
            "lang": lang, "use_rag": req.use_rag,
            "completion": engine.decode(tokens), "n_tokens": len(tokens),
            "latency_s": round(time.time() - t0, 3), "endpoint": "/v1/completions",
        })

    if req.stream:
        async def stream():
            t0 = time.time()
            buf = []
            prev = ""
            async for tok in _iter_generation(_openai_token_iter(req)):
                buf.append(tok)
                text = engine.decode(buf)
                chunk = text[len(prev):]
                prev = text
                if chunk:
                    yield f"data: {json.dumps({'choices': [{'text': chunk, 'index': 0, 'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps({'choices': [{'text': '', 'index': 0, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
            _log_completion({
                "kind": "openai", "model": req.model, "stream": True,
                "prompt": req.prompt, "suffix": req.suffix or "",
                "lang": lang, "use_rag": req.use_rag,
                "completion": engine.decode(buf), "n_tokens": len(buf),
                "latency_s": round(time.time() - t0, 3), "endpoint": "/v1/completions",
            })
        return StreamingResponse(stream(), media_type="text/event-stream")

    with _lock:
        t0 = time.time()
        tokens = list(_openai_tokens(req))
        _log(tokens, t0, False)
    text = engine.decode(tokens)
    return {
        "id": f"cmpl-{int(time.time() * 1000)}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"text": text, "index": 0, "logprobs": None,
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(engine.sp.encode(req.prompt)),
                  "completion_tokens": len(tokens),
                  "total_tokens": len(engine.sp.encode(req.prompt)) + len(tokens)},
    }


def _chat_prompt(req: OpenAIChatRequest) -> str:
    """Flatten messages to a plain prompt (best-effort for a FIM code model)."""
    parts = []
    for m in req.messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, list):  # multimodal parts -> keep text only
            content = " ".join(str(c.get("text", "")) for c in content
                               if isinstance(c, dict))
        if content:
            parts.append(str(content))
    return "\n".join(parts).strip()


@app.post("/v1/chat/completions")
def v1_chat_completions(req: OpenAIChatRequest):
    """Chat-test endpoint: the FIM model isn't chat-tuned, but this makes
    Continue chat work end-to-end (completion-style generation)."""
    req.max_tokens = max(1, min(req.max_tokens, MAX_CHAT_TOKENS))

    def _chat_tokens(prompt: str):
        lang = infer_code_lang(prompt)
        return engine.generate_fim(engine.sp.encode(prompt), [], req.max_tokens,
                                   req.temperature, 0, req.top_p, None, lang=lang,
                                   frequency_penalty=FREQ_PENALTY_DEFAULT,
                                   presence_penalty=PRESENCE_PENALTY_DEFAULT)

    def _chat_response(tokens, t0):
        text = engine.decode(tokens)
        _log_completion({
            "kind": "chat", "model": req.model, "stream": req.stream,
            "prompt": _chat_prompt(req), "suffix": "", "lang": None,
            "use_rag": False, "completion": text, "n_tokens": len(tokens),
            "latency_s": round(time.time() - t0, 3),
            "endpoint": "/v1/chat/completions",
        })
        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(tokens),
                      "total_tokens": len(tokens)},
        }

    prompt = _chat_prompt(req)
    if not prompt:
        return {"error": "empty messages"}

    if req.stream:
        async def stream():
            t0 = time.time()
            buf, prev = [], ""
            async for tok in _iter_generation(_chat_tokens(prompt)):
                buf.append(tok)
                text = engine.decode(buf)
                chunk = text[len(prev):]
                prev = text
                if chunk:
                    yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
            _chat_response(buf, t0)
        return StreamingResponse(stream(), media_type="text/event-stream")

    with _lock:
        t0 = time.time()
        tokens = list(_chat_tokens(prompt))
        return _chat_response(tokens, t0)


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
    # LOCLLM_PORT overrides the default port (e.g. 8080 for a remote tunnel)
    uvicorn.run(app, host="0.0.0.0",
                port=int(os.environ.get("LOCLLM_PORT", "8000")))