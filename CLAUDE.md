# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                        # 依存インストール（paddle は sys_platform で GPU/CPU 自動切替）
uv run python run_gpu.py                       # バッチOCR（WSL + NVIDIA GPU）: data/ → output/gpu/
uv run python run_cpu.py                       # バッチOCR（Mac）: data/ → output/cpu/
uv run uvicorn server.app:app --port 8100      # 文法下書きツール（FastAPI + ブラウザUI）
uv run python -c "import paddle; paddle.utils.run_check()"   # Paddle 動作確認
```

テスト・lint は未設定。

## Architecture

PP-OCRv6（PaddleOCR、`external/PaddleOCR` の git submodule を editable インストール）で教科書ページ画像を OCR する2系統のツール:

1. **バッチOCR** — `ocr_extract.py` の `run(device, lang)` を `run_gpu.py`/`run_cpu.py` から呼ぶ。`data/` の全画像を OCR し `output/<device>/` に JSON + 可視化画像を保存。
2. **下書きツール** (`server/`) — 画像アップロード → OCR → LLM構造化 → vocab-trainer（別リポジトリ、Cloud Run にデプロイ）へ「下書き」アップロード、または JSON ダウンロード。中間ファイルなし、すべてメモリ上。
   - `server/app.py` — FastAPI。`GET /api/books`（登録済みの本一覧 `{id, kind, title}[]`）、`POST /api/extract`（multipart: image, lang, description_language, book=本ID（省略時は kind=`grammar`|`word` ごとのデフォルト） → `{sourceImage, book, kind, ocrText, pageMatchesBook, pageNote, candidates}`。`pageMatchesBook` は「選択した本のページか」のLLM判定で、false ならUIがそのページをスキップして報告）、`POST /api/ocr-detail`（LLMなしの生検出結果、可視化UI用）、`POST /api/upload-drafts`（`{language, drafts, kind}` を kind に応じて `CLOUD_API_BASE_URL/api/grammar/{language}/drafts` または `/api/vocab/{language}/drafts` へプロキシ）、`GET /api/config`。静的UIは最後に `/` へマウント。
   - `server/ocr.py` — `OcrEngine`: `(device, lang)` ごとに PaddleOCR インスタンスを遅延生成・キャッシュ。lifespan 起動時にデフォルトをプリウォーム。`use_doc_unwarping=False, use_doc_orientation_classify=False` は平面画像向けの実測ベスト設定（ocr_extract.py と同じ）。
   - `server/llm.py` — OpenAI structured outputs で座標付きOCR検出結果（`[y=.. x=.. h=..] text` 形式）を候補に構造化。**プロンプトは本ごとに分け、`BOOKS` レジストリに登録する**（本の追加 = プロンプト+スキーマを書いて `BOOKS` に1エントリ追加。`kind` が候補の形とUIセッション・アップロード先を決める）。`structure_page(client, model, book_id, items, description_language)` が入口:
     - 文法「HSK公認テキスト４級」（id: `hsk4-grammar`）→ `{statement, group, level, description, examples[]}`。`group` は Point 題名（大きい文字）から抽出。前提は `docs/book-structure.md`。
     - 単語「新HSK1~4級単語トレーニングブック」（id: `hsk1-4-word-training`）→ `{term, transliteration, level, definitions[], examples[]}`。例文の `segments` は本文中のピンイン分かち書きに沿って分割。前提は `docs/word-book-structure.md`。
     - **構造化のみ**: 4言語定義・見出し語ピンイン補完はアップロード先 vocab-trainer の smart-add が本登録時に行う。
   - `server/static/index.html` — ビルド不要の単一ファイルUI（vanilla JS、DOMはJSで構築）。**タブで文法/単語セッションを切替**（セッションごとに独立した状態: 本の選択（`/api/books` から kind でフィルタ）、言語設定、画像、候補。タブを切り替えても保持）。画像はドラッグ&ドロップ or クリック選択（複数可・サムネイル表示・個別削除）。**複数画像対応**: ファイル名の自然順（`numeric` localeCompare）で1枚ずつ `/api/extract` に投げ、進捗バー表示、全ページの候補を1リストに蓄積（候補ごとに `sourceImage` を保持、複数ページ時はカード右上に画像名を表示。失敗ページ・違う本のページはスキップして続行し、まとめて報告+サムネイルに!マーク）。候補カードで編集・トグルで取捨選択・全選択/全解除。カードの間（先頭・末尾含む）の挿入バー「＋ ここに候補を追加」で検出漏れを任意の位置に手入力追加できる（候補0件でも結果エリアを表示して追加可能）。画面下部の固定アクションバーから共通グループ名入力（カンマ区切り → 各draftの `groups`）、直接アップロード（文法・単語とも）or JSONダウンロード。ダウンロード名は単一ページ `<画像名>.<kind>.json`、複数ページ `<先頭画像名>-<末尾画像名>.<kind>.json`。
   - `server/config.py` — ルートの `.env` を読む: `OPENAI_API_KEY`, `OPENAI_MODEL`, `CLOUD_API_BASE_URL`, `CLOUD_UI_URL`, `OCR_DEVICE`, `OCR_LANG`。

## 下書きJSONフォーマット

`docs/draft-json-format.md` を参照（**正本は vocab-trainer/docs/draft-json-format.md** — 変更は正本側で行い、コピーを同期する）。要点:

- エンベロープ: `{version: 1, kind: "grammar-drafts" | "word-drafts", language, drafts[]}`
- grammar draft: `statement`（必須）、`descriptions: Meaning[]`（必須、1言語のみでよい）、`examples[]`、`level`、`tags`、`groups`（グループ**名**の配列 — 本登録時に解決・自動作成）、`sourceImage`
- word draft: `term`（必須）、`transliteration`、`definitions`、`examples`（各例文の `segments: string[]` が分かち書きチップ情報）、`level`、`topics`、`groups`、`sourceImage`。アップロード先は vocab-trainer `POST /api/vocab/:language/drafts`（UIは Browse の「JSONアップロード」）
- `id`/`createdAt` はサーバー付与のためファイルに含めない

## 注意

- 言語コードは vocab-trainer の backend フルネーム形式（`chinese` 等）。OCR言語は PaddleOCR 形式（`ch`/`japan`）で別物。
- vocab-trainer 本体（Cloud Run デプロイ対象）にはこのリポジトリのコードを混ぜないこと — 本ツールはローカル専用。
