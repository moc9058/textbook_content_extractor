"""PP-OCRv6 エンジンのウォームなラッパー。

(device, lang) の組ごとに PaddleOCR インスタンスを遅延生成してキャッシュする。
バッチ用の ocr_extract.py とは独立（あちらは無変更で残す）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paddleocr import PaddleOCR


def _to_list(v: Any) -> Any:
    """numpy 配列などを JSON 化可能なネストした list に変換する。"""
    return v.tolist() if hasattr(v, "tolist") else v


class OcrEngine:
    def __init__(self) -> None:
        self._engines: dict[tuple[str, str], PaddleOCR] = {}

    def warm(self, device: str, lang: str) -> None:
        self._get(device, lang)

    def _get(self, device: str, lang: str) -> PaddleOCR:
        key = (device, lang)
        if key not in self._engines:
            # スクショ等の平らな画像が対象のため歪み補正・向き補正は無効化
            # （ocr_extract.py と同じ判断。ONにするとrecall低下を実測済み）。
            self._engines[key] = PaddleOCR(
                ocr_version="PP-OCRv6",
                lang=lang,
                device=device,
                use_doc_unwarping=False,
                use_doc_orientation_classify=False,
            )
        return self._engines[key]

    def extract_texts(self, image_path: Path, device: str, lang: str) -> list[str]:
        """1画像をOCRし、認識テキスト行を読み順（上→下）で返す。"""
        texts: list[str] = []
        for res in self._get(device, lang).predict(str(image_path)):
            try:
                texts.extend(res["rec_texts"])
            except (TypeError, KeyError):
                texts.extend(res.json["res"]["rec_texts"])
        return texts

    def extract_detail(self, image_path: Path, device: str, lang: str) -> list[dict[str, Any]]:
        """1画像をOCRし、行ごとの詳細（テキスト・信頼度・矩形・ポリゴン）を返す。

        座標は元画像のピクセル基準。可視化UI用。
        """
        items: list[dict[str, Any]] = []
        for res in self._get(device, lang).predict(str(image_path)):
            try:
                data = res
                texts = data["rec_texts"]
            except (TypeError, KeyError):
                data = res.json["res"]
                texts = data["rec_texts"]
            scores = data["rec_scores"]
            boxes = data["rec_boxes"]
            polys = data["rec_polys"]
            for i, text in enumerate(texts):
                items.append(
                    {
                        "index": len(items),
                        "text": text,
                        "score": float(scores[i]),
                        "box": _to_list(boxes[i]),
                        "poly": _to_list(polys[i]),
                    }
                )
        return items
