
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from pprint import pprint

from db import CodeDB
from stackV3 import stack_v3_fim_gen

sys.path.append(str(Path(__file__).resolve().parents[2] / "model"))
from gpuSeletor.main import select_only_gpu

from model import build_model

MODEL = None
DB_PATH = "/media/user/2TB Clear/codeDB/db.db"
CHECKPOINT_PATH = "./checkpoint.json"
LOGGER_URL = "http://91.98.145.193:4242"
BATCH_SIZE = 64
ENCODE_BATCH_SIZE = 4

SEND_DATA_TO_DATASET = True
DATASET_URL = "http://localhost:8823/api/receive-data"


def get_gpu_temps():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        temps = []
        for line in out.splitlines():
            parts = line.split(", ")
            temps.append({"index": int(parts[0]), "name": parts[1], "temp": int(parts[2])})
        return temps
    except Exception:
        return []


def send_metrics(checkpoint, db_count, rate=None, gpu_temps=None):
    payload = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "step": checkpoint.get("step", 0),
        "db_count": db_count,
        "rate": rate,
        "gpu_temps": gpu_temps,
        "langs": {k: v for k, v in checkpoint.items() if k != "step" and not k.endswith("_char_count")},
        "char_counts": {k: v for k, v in checkpoint.items() if k.endswith("_char_count")},
    }
    req = urllib.request.Request(
        LOGGER_URL + "/metrics",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def send_to_dataset(text, category):
    payload = json.dumps({"text": text, "category": category}).encode()
    req = urllib.request.Request(
        DATASET_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"send_to_dataset failed: {e}")
        return None

def build():
    parser = argparse.ArgumentParser(description="FIM data embedding with GPU selection")
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU index to use (skips the interactive GPU selector)",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        gpu_index = args.gpu
        print(f"Using GPU {gpu_index} (from --gpu)")
    else:
        selected = select_only_gpu()
        if not selected:
            print("No GPU selected, exiting.")
            exit(1)
        gpu = selected[0]
        gpu_index = gpu["index"]
        print(f"Selected GPU {gpu_index}: {gpu['name']} ({gpu['vram_size']:.1f}GB)")

    model = build_model(gpu_index)

    # test
    embedding = model.encode("hello world")
    print(f"Embedding shape: {embedding.shape} | dtype: {embedding.dtype}")
    print("Test OK")

    return model, gpu_index

if __name__ == "__main__":
    MODEL, gpu_index = build()

    if not Path(CHECKPOINT_PATH).exists():
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump({"step": 0}, f)
    with open(CHECKPOINT_PATH) as f:
        checkpoint = json.load(f)
    step = checkpoint.get("step", 0)

    db = CodeDB(DB_PATH)
    start_time = time.time()
    send_metrics(checkpoint, db.count(), gpu_temps=get_gpu_temps())

    codeGen = stack_v3_fim_gen()
    for _ in range(step):
        try:
            next(codeGen)
        except StopIteration:
            break

    iteration = step
    batch = []
    try:
        while True:
            code = next(codeGen)
            if SEND_DATA_TO_DATASET:
                send_to_dataset(code["raw_text"], code["category"])
            lang = code["lang"]
            checkpoint[lang] = checkpoint.get(lang, 0) + 1
            checkpoint[lang + "_char_count"] = checkpoint.get(lang + "_char_count", 0) + len(code["text"])
            batch.append(code)
            iteration += 1
            checkpoint["step"] = iteration

            if len(batch) >= BATCH_SIZE:
                embeddings = MODEL.encode([c["text"] for c in batch], batch_size=ENCODE_BATCH_SIZE)
                db.add_batch([(c["text"], c["lang"], c["hash"], emb) for c, emb in zip(batch, embeddings)])
                batch = []

            if iteration % 256 == 0:
                pprint(checkpoint)
                with open(CHECKPOINT_PATH, "w") as f:
                    json.dump(checkpoint, f, indent=4)
                rate = round((iteration - step) / max(time.time() - start_time, 1e-9), 2)
                send_metrics(checkpoint, db.count(), rate, get_gpu_temps())
    except StopIteration:
        if batch:
            embeddings = MODEL.encode([c["text"] for c in batch], batch_size=ENCODE_BATCH_SIZE)
            db.add_batch([(c["text"], c["lang"], c["hash"], emb) for c, emb in zip(batch, embeddings)])
        print("dataset exhausted")
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint, f, indent=4)
        rate = round((iteration - step) / max(time.time() - start_time, 1e-9), 2)
        send_metrics(checkpoint, db.count(), rate, get_gpu_temps())


