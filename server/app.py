"""ローカルOCR→文法下書きツールの FastAPI サーバー.

起動:
    uv run uvicorn server.app:app --port 8100

フロー:
    画像アップロード → PP-OCRv6 → LLM構造化 → ブラウザで確認・編集 →
    クラウド vocab-trainer へ下書きとしてアップロード
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

from .config import load_settings
from .llm import structure_page
from .ocr import OcrEngine

settings = load_settings()
engine = OcrEngine()
openai_client = OpenAI(api_key=settings.openai_api_key, max_retries=3)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 初回アップロードを待たせないよう、デフォルトエンジンをプリウォーム
    print(f"PP-OCRv6 をプリウォーム中 (device={settings.ocr_device}, lang={settings.ocr_lang}) ...")
    engine.warm(settings.ocr_device, settings.ocr_lang)
    print("プリウォーム完了")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/config")
def get_config() -> dict[str, str]:
    return {
        "cloudApiBaseUrl": settings.cloud_api_base_url,
        "cloudUiUrl": settings.cloud_ui_url,
        "ocrDevice": settings.ocr_device,
        "ocrLang": settings.ocr_lang,
    }


@app.post("/api/extract")
async def extract(
    image: UploadFile = File(...),
    lang: str = Form(default=""),
    description_language: str = Form(default="ja"),
) -> dict[str, Any]:
    ocr_lang = lang or settings.ocr_lang
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        texts = engine.extract_texts(tmp_path, settings.ocr_device, ocr_lang)
        if not texts:
            raise HTTPException(status_code=422, detail="OCRでテキストを検出できませんでした")
        ocr_text = "\n".join(texts)
        candidates = structure_page(
            openai_client, settings.openai_model, ocr_text, description_language
        )
        return {
            "sourceImage": image.filename or "upload",
            "ocrText": ocr_text,
            "candidates": candidates,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/ocr-detail")
async def ocr_detail(
    image: UploadFile = File(...),
    lang: str = Form(default=""),
) -> dict[str, Any]:
    """LLM構造化を通さず、OCRの生の検出結果（テキスト・信頼度・座標）を返す。可視化UI用。"""
    ocr_lang = lang or settings.ocr_lang
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        items = engine.extract_detail(tmp_path, settings.ocr_device, ocr_lang)
        if not items:
            raise HTTPException(status_code=422, detail="OCRでテキストを検出できませんでした")
        return {
            "sourceImage": image.filename or "upload",
            "lang": ocr_lang,
            "count": len(items),
            "items": items,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/upload-drafts")
async def upload_drafts(body: dict[str, Any]) -> JSONResponse:
    language = body.get("language")
    drafts = body.get("drafts")
    if not language or not isinstance(drafts, list) or not drafts:
        raise HTTPException(status_code=400, detail="language と drafts (非空配列) が必要です")
    url = f"{settings.cloud_api_base_url}/api/grammar/{language}/drafts"
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(url, json={"drafts": drafts})
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"クラウドAPIへの接続に失敗: {e}") from e
    return JSONResponse(status_code=resp.status_code, content=resp.json())


# APIルートの後にマウント（/ は index.html）
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
