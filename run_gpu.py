"""GPU版 実行ファイル（Windows WSL Ubuntu + NVIDIA GPU 用）.

    uv run python run_gpu.py
"""

from ocr_extract import run

if __name__ == "__main__":
    run(device="gpu")
