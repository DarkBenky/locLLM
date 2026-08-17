
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "model"))
from gpuSeletor.main import select_only_gpu

from model import build_model

MODEL = None

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

    return model

if __name__ == "__main__":
    MODEL = build()

