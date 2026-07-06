"""CPU版 実行ファイル（Mac mini M4 用）.

    uv run python run_cpu.py
"""

from ocr_extract import run

if __name__ == "__main__":
    run(device="cpu")
