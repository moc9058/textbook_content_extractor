"""ローカルOCR→文法下書きツールの FastAPI サーバー.

起動:
    uv run uvicorn server.app:app --port 8100

フロー:
    画像アップロード → PP-OCRv6 → LLM構造化 → ブラウザで確認・編集 →
    クラウド vocab-trainer へ下書きとしてアップロード
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
import time
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
from .llm import BOOKS, DEFAULT_BOOK_BY_KIND, structure_page
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


@app.get("/api/books")
def get_books() -> dict[str, Any]:
    """登録済みの本の一覧（UIのセッション別・本選択用）。"""
    return {
        "books": [
            {"id": book_id, "kind": b["kind"], "title": b["title"]} for book_id, b in BOOKS.items()
        ]
    }


@app.post("/api/extract")
async def extract(
    image: UploadFile = File(...),
    lang: str = Form(default=""),
    description_language: str = Form(default="ja"),
    kind: str = Form(default="grammar"),
    book: str = Form(default=""),
) -> dict[str, Any]:
    """book=BOOKS のID。省略時は kind（grammar/word）ごとのデフォルトの本。"""
    book_id = book or DEFAULT_BOOK_BY_KIND.get(kind, "")
    if book_id not in BOOKS:
        raise HTTPException(status_code=400, detail=f"未登録の本です: {book or kind}")
    ocr_lang = lang or settings.ocr_lang
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        started_at = time.perf_counter()
        items = engine.extract_detail(tmp_path, settings.ocr_device, ocr_lang)
        if not items:
            raise HTTPException(status_code=422, detail="OCRでテキストを検出できませんでした")
        ocr_elapsed = time.perf_counter() - started_at
        ocr_text = "\n".join(it["text"] for it in items)
        result = structure_page(
            openai_client, settings.openai_model, book_id, items, description_language
        )
        total_elapsed = time.perf_counter() - started_at
        return {
            "sourceImage": image.filename or "upload",
            "book": book_id,
            "kind": BOOKS[book_id]["kind"],
            "ocrText": ocr_text,
            # 選択した本のページかどうかのLLM判定（違う本のスクショ検出用）
            "pageMatchesBook": result["page_matches_book"],
            "pageNote": result["page_note"],
            "candidates": result["candidates"],
            "ocrElapsedSeconds": round(ocr_elapsed, 2),
            "totalElapsedSeconds": round(total_elapsed, 2),
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


# gcloud のIDトークンキャッシュ（トークンは約1時間有効。リポジトリ・ファイルには保存しない）
_id_token_cache: dict[str, Any] = {"token": None, "expires_at": 0.0}


def _gcloud_identity_token() -> str | None:
    """ローカルPCの gcloud 認証情報からIDトークンを取得する。gcloud 未導入・未ログインなら None。"""
    if _id_token_cache["token"] and time.time() < _id_token_cache["expires_at"]:
        return _id_token_cache["token"]
    try:
        proc = subprocess.run(
            ["gcloud", "auth", "print-identity-token"],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    token = proc.stdout.strip()
    if not token:
        return None
    _id_token_cache.update(token=token, expires_at=time.time() + 45 * 60)
    return token


@app.post("/api/upload-drafts")
async def upload_drafts(body: dict[str, Any]) -> JSONResponse:
    language = body.get("language")
    drafts = body.get("drafts")
    kind = body.get("kind", "grammar")
    if not language or not isinstance(drafts, list) or not drafts:
        raise HTTPException(status_code=400, detail="language と drafts (非空配列) が必要です")
    if kind not in ("grammar", "word"):
        raise HTTPException(status_code=400, detail="kind は grammar か word を指定してください")
    # 文法: /api/grammar/... / 単語: /api/vocab/...（vocab-trainer 側のパス）
    api_path = "grammar" if kind == "grammar" else "vocab"
    url = f"{settings.cloud_api_base_url}/api/{api_path}/{language}/drafts"
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(url, json={"drafts": drafts})
            # Cloud Run がIAM保護されている場合はローカルの gcloud 認証で再試行
            if resp.status_code in (401, 403):
                token = await asyncio.to_thread(_gcloud_identity_token)
                if token:
                    resp = await client.post(
                        url, json={"drafts": drafts}, headers={"Authorization": f"Bearer {token}"}
                    )
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"クラウドAPIへの接続に失敗 ({url}): {e}") from e
    if resp.status_code in (401, 403):
        raise HTTPException(
            status_code=502,
            detail="クラウドAPIの認証に失敗しました。`gcloud auth login` 済みか確認してください",
        )
    try:
        content = resp.json()
    except ValueError:
        content = {"detail": resp.text[:500]}
    return JSONResponse(status_code=resp.status_code, content=content)


# APIルートの後にマウント（/ は index.html）
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
