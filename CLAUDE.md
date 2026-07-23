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
   - **アップロードは1つの章・パート単位（複数ページ）が原則**。構造化は全ページを1回のLLM呼び出しにまとめ、表などがページをまたいでも1候補に統合する（`docs/book-structure.md` / `docs/word-book-structure.md` 参照）。
   - `server/app.py` — FastAPI。`GET /api/books`（登録済みの本一覧 `{id, kind, title}[]`）、`POST /api/structure-batch`（**UI本流**。JSON: `{kind, book, description_language, pages:[{sourceImage, items}]}` → `{book, kind, pageMatchesBook, pageNote, unmatchedImages, candidates}`。フロントが各ページを `/api/ocr-detail` でOCRしてから items をまとめて渡し、章全体を1回で構造化。`unmatchedImages` は「選択した本と違うページ」のファイル名配列でUIがスキップ報告、`pageMatchesBook=false` はバッチ全体が別の本）、`POST /api/extract`（単一ページ版 multipart: image, lang, description_language, book → `{sourceImage, book, kind, ocrText, pageMatchesBook, pageNote, candidates}`。内部は `structure_page`＝1ページのバッチ）、`POST /api/ocr-detail`（LLMなしの生検出結果 `{sourceImage, items}`。可視化UI＋本流のOCRフェーズ用）、`POST /api/upload-drafts`（`{language, drafts, kind}` を kind に応じて `CLOUD_API_BASE_URL/api/grammar/{language}/drafts` または `/api/vocab/{language}/drafts` へプロキシ）、`GET /api/config`。静的UIは最後に `/` へマウント。
   - `server/ocr.py` — `OcrEngine`: `(device, lang)` ごとに PaddleOCR インスタンスを遅延生成・キャッシュ。lifespan 起動時にデフォルトをプリウォーム。`use_doc_unwarping=False, use_doc_orientation_classify=False` は平面画像向けの実測ベスト設定（ocr_extract.py と同じ）。
   - `server/llm.py` — OpenAI structured outputs で座標付きOCR検出結果（`[y=.. x=.. h=..] text` 形式）を候補に構造化。**プロンプトは本ごとに分け、`BOOKS` レジストリに登録する**（本の追加 = プロンプト+スキーマを書いて `BOOKS` に1エントリ追加。`kind` が候補の形とUIセッション・アップロード先を決める）。入口は `structure_pages(client, model, book_id, pages, description_language)`（`pages=[{sourceImage, items}]` を読む順に。`format_pages` が `=== PAGE: <名前> ===` マーカーで区切って1テキスト化 → 1回のLLM呼び出し。ページまたぎの表を1候補に統合し、各候補は由来ページを `sourceImages` に自己申告）。`structure_page` はその単一ページ薄ラッパ。各本の候補スキーマは全候補に `sourceImages`、トップレベルに `unmatched_images` を持つ:
     - 文法「HSK公認テキスト４級」（id: `hsk4-grammar`）→ `{statement, group, level, description, examples[]}`。`group` は Point 題名（大きい文字）から抽出。前提は `docs/book-structure.md`。
     - 単語「新HSK1~4級単語トレーニングブック」（id: `hsk1-4-word-training`）→ `{term, transliteration, level, definitions[], examples[]}`。例文の `segments` は本文中のピンイン分かち書きに沿って分割。前提は `docs/word-book-structure.md`。
     - **構造化のみ**: 4言語定義・見出し語ピンイン補完はアップロード先 vocab-trainer の smart-add が本登録時に行う。
   - `server/static/index.html` — ビルド不要の単一ファイルUI（vanilla JS、DOMはJSで構築）。**タブで文法/単語セッションを切替**（セッションごとに独立した状態: 本の選択（`/api/books` から kind でフィルタ）、言語設定、画像、候補。タブを切り替えても保持）。画像はドラッグ&ドロップ or クリック選択（複数可・サムネイル表示・個別削除）。**章・パート単位の一括処理**: ファイル名の自然順（`numeric` localeCompare）で並べ、フェーズ1で1枚ずつ `/api/ocr-detail` にOCR（進捗バーは全体の90%まで）、フェーズ2で全ページの items をまとめて `/api/structure-batch` に投げて章全体を1回で構造化（残り10%）。候補は `sourceImages`（ページまたぎ時は複数）を保持し、複数ページ時はカード右上に画像名を表示。OCR失敗ページ・`unmatchedImages`（違う本のページ）はスキップして報告+サムネイルに!マーク、`pageMatchesBook=false` は全ページスキップ。文法は取りこぼし対策に `mergeDuplicateGrammar` でクライアント側統合も併用。候補カードで編集・トグルで取捨選択・全選択/全解除。カードの間（先頭・末尾含む）の挿入バー「＋ ここに候補を追加」で検出漏れを任意の位置に手入力追加できる（候補0件でも結果エリアを表示して追加可能）。画面下部の固定アクションバーから共通グループ名入力（カンマ区切り → 各draftの `groups`）、直接アップロード（文法・単語とも）or JSONダウンロード。ダウンロード名は単一ページ `<画像名>.<kind>.json`、複数ページ `<先頭画像名>-<末尾画像名>.<kind>.json`。
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
