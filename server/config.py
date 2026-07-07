"""設定の読み込み。プロジェクトルートの .env から環境変数をロードする。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    cloud_api_base_url: str
    cloud_ui_url: str
    ocr_device: str
    ocr_lang: str


def load_settings() -> Settings:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip().strip('"')
    if not api_key:
        raise SystemExit("OPENAI_API_KEY が .env に設定されていません")
    return Settings(
        openai_api_key=api_key,
        openai_model=os.environ.get("OPENAI_MODEL", "").strip().strip('"'),
        cloud_api_base_url=os.environ.get("CLOUD_API_BASE_URL", "http://localhost:3000").strip().rstrip("/"),
        cloud_ui_url=os.environ.get("CLOUD_UI_URL", "").strip().rstrip("/"),
        ocr_device=os.environ.get("OCR_DEVICE", "cpu").strip(),
        ocr_lang=os.environ.get("OCR_LANG", "ch").strip(),
    )
