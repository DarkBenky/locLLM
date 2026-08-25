import fastapi
from typing import List

from main import build
from model import embed_texts
from db import CodeDB
from search_index import InMemoryIndex
from pydantic import BaseModel
import uvicorn


MODEL = None
DB = None
INDEX = None
GPU_INDEX = None

DB_PATH = "/media/user/2TB Clear/codeDB/db.db"
MAX_BATCH_SIZE = 8

app = fastapi.FastAPI()


class SearchRequest(BaseModel):
    texts: List[str]
    top_k: int = 1


@app.post("/search")
def search(req: SearchRequest):
    texts = req.texts
    top_k = req.top_k
    if not texts:
        return []

    batch = min(MAX_BATCH_SIZE, len(texts))
    texts_list = texts[:MAX_BATCH_SIZE]

    embeddings = embed_texts(MODEL, texts_list, batch_size=batch)
    response = []
    for embedding, text in zip(embeddings, texts_list):
        hits = INDEX.search(embedding, k=top_k)
        results = []
        for rowid, distance in hits:
            item = DB.get_item(rowid)
            if item is None:
                continue
            results.append({
                "hash": item["hash"],
                "code": item["code"],
                "lang": item["lang"],
                "distance": distance,
            })
        response.append({
            "text": text,
            "embedding": embedding.tolist(),
            "results": results,
        })
    return response


@app.get("/")
def health():
    return {"status": "ok", "gpu_index": GPU_INDEX}


if __name__ == "__main__":
    MODEL, GPU_INDEX = build()
    DB = CodeDB(DB_PATH)
    INDEX = InMemoryIndex(DB.conn)

    uvicorn.run(app, host="0.0.0.0", port=8234)

    