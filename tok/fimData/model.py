import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "jinaai/jina-code-embeddings-1.5b"
MODEL_SAVE_DIR = Path("/media/user/2TB/models/jina-code-embeddings-1.5b")
MAX_SEQ_LENGTH = 2048


def get_model_path():
    return str(MODEL_SAVE_DIR) if MODEL_SAVE_DIR.exists() else MODEL_ID


def build_model(gpu_index):
    print(f"Loading model {MODEL_ID} ...")
    model = SentenceTransformer(
        get_model_path(),
        model_kwargs={"torch_dtype": torch.bfloat16, "device_map": {"": gpu_index}},
        processor_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    print(f"Model loaded on cuda:{gpu_index}")

    if not MODEL_SAVE_DIR.exists():
        MODEL_SAVE_DIR.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(MODEL_SAVE_DIR))
        print(f"Model saved to {MODEL_SAVE_DIR}")
    else:
        print(f"Model already saved at {MODEL_SAVE_DIR}")

    return model


def embed_texts(model, texts, batch_size=32, normalize_embeddings=False):
    if isinstance(texts, str):
        texts = [texts]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings
