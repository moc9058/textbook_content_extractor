"""PP-OCRv6 でdata/内の画像をOCRし、結果をJSONに変換する共通処理.

GPU版 (run_gpu.py) / CPU版 (run_cpu.py) から device を指定して呼び出す。
"""

from __future__ import annotations

from pathlib import Path

from paddleocr import PaddleOCR

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "output"

# 対象とする画像拡張子
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def run(device: str, lang: str = "japan") -> None:
    """data/内の全画像をPP-OCRv6で処理し、JSONをoutput/<device>/に保存する.

    Args:
        device: "gpu" もしくは "cpu"。
        lang:   認識言語。日本語教科書なら "japan"（英数字中心なら "en"）。
    """
    images = sorted(
        p for p in DATA_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise SystemExit(f"画像が見つかりません: {DATA_DIR}")

    output_dir = OUTPUT_ROOT / device
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{device}] PP-OCRv6 を初期化中 (lang={lang}) ...")
    # スクショなど平らな画像が対象のため、文書の歪み補正・向き補正は無効化する。
    # （これらは「カメラで撮った湾曲・傾いた紙」向けの機能で、平らな画像では
    #   レイアウトを歪めて検出漏れ=recall低下を招くため。実測でOFFの方が本文を
    #   取りこぼさないことを確認済み。）
    ocr = PaddleOCR(
        ocr_version="PP-OCRv6",
        lang=lang,
        device=device,
        use_doc_unwarping=False,
        use_doc_orientation_classify=False,
    )

    for img in images:
        print(f"[{device}] OCR 実行: {img.name}")
        results = list(ocr.predict(str(img)))
        for i, res in enumerate(results):
            # 1画像に複数結果が出た場合のみ連番を付ける
            stem = img.stem if len(results) == 1 else f"{img.stem}_{i}"
            json_path = output_dir / f"{stem}.json"
            res.save_to_json(str(json_path))
            print(f"[{device}]   -> {json_path.relative_to(PROJECT_ROOT)}")

            # 検出枠＋認識文字を描画した可視化画像も保存
            img_path = output_dir / f"{stem}.png"
            res.save_to_img(str(img_path))
            print(f"[{device}]   -> {img_path.relative_to(PROJECT_ROOT)}")

    print(f"[{device}] 完了。JSON 出力先: {output_dir.relative_to(PROJECT_ROOT)}/")
