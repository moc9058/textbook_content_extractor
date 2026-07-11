# textbook-content-extractor

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) の **PP-OCRv6** モデルで教科書ページ画像を OCR するツール群です。2 つの使い方があります。

| モード | 用途 | 実行方法 |
|---|---|---|
| **バッチ OCR**（`run_gpu.py` / `run_cpu.py`） | `data/` 内の画像を一括 OCR し、JSON と可視化画像を `output/` に保存 | [使い方（バッチ OCR）](#使い方バッチ-ocr) |
| **文法下書きツール**（`server/`） | ブラウザで画像をアップロード → OCR → LLM で文法項目に構造化 → vocab-trainer へ「下書き」アップロード | [文法下書きツール（server/）](#文法下書きツールserver) |

- **Windows WSL Ubuntu（NVIDIA GPU）** → GPU 推論
- **Mac mini M4（Apple Silicon）** → CPU 推論

バッチ OCR は同じロジック（`ocr_extract.py`）を、環境ごとの実行ファイル（`run_gpu.py` / `run_cpu.py`）から `device` を切り替えて呼び出します。

---

## 必要環境

- [uv](https://docs.astral.sh/uv/)（パッケージ／仮想環境マネージャ）
- Python 3.13
- **GPU 版のみ**: NVIDIA GPU + ドライバ（CUDA 12.6 対応。WSL2 の GPU パススルーを含む）
- **Mac 版**: Apple Silicon（M シリーズ）。追加ドライバ不要

依存パッケージは `pyproject.toml` の `sys_platform` マーカーで自動的に切り替わります。

| 環境 | インストールされる PaddlePaddle |
|---|---|
| Windows WSL / Linux (x86_64) | `paddlepaddle-gpu`（CUDA 12.6 ビルド、Paddle 公式インデックスから） |
| macOS (arm64) | `paddlepaddle`（CPU、PyPI から） |

---

## セットアップ

```bash
# 1. リポジトリを取得
git clone <this-repo-url>
cd textbook_content_extractor

# 2. PaddleOCR サブモジュールを取得（external/PaddleOCR）
git submodule update --init --recursive

# 3. 依存をインストール（venv も自動作成される）
uv sync
```

### Ubuntu / WSL のみ: システムライブラリ

Paddle は OpenMP ランタイムに依存します。未インストールの場合は次を実行してください。

```bash
sudo apt update && sudo apt install -y libgomp1
```

> macOS では不要です（CPU ホイールに同梱されています）。

### 動作確認（任意）

```bash
uv run python -c "import paddle; paddle.utils.run_check()"
```

- GPU 環境: `PaddlePaddle works well on 1 GPU.` と表示されれば OK
- Mac 環境: `PaddlePaddle works well on CPU.` と表示されれば OK

> 実行時に出る `libcuda.so ... not configured` や `No ccache found` の警告は無害です（推論は正常に動作します）。

---

## 使い方（バッチ OCR）

`data/` に OCR したい画像を置いてから、環境に応じて実行します。

**Windows WSL Ubuntu（GPU）:**
```bash
uv run python run_gpu.py
```

**Mac mini M4（CPU）:**
```bash
uv run python run_cpu.py
```

対応画像形式: `.png` `.jpg` `.jpeg` `.bmp` `.tif` `.tiff` `.webp`

---

## 出力

`data/` 内の各画像につき、`output/<device>/` に 2 ファイルが生成されます。

| ファイル | 内容 |
|---|---|
| `output/gpu/<画像名>.json` | 認識テキスト（`rec_texts`）・信頼度（`rec_scores`）・座標（`rec_boxes` / `rec_polys`）など |
| `output/gpu/<画像名>.png` | 元画像に検出枠と認識文字を重ねた可視化画像 |

（Mac 版は `output/cpu/` に出力されます。）JSON は日本語をそのまま保持します（`ensure_ascii=False`）。

---

## 設定の変更

`ocr_extract.py` で調整できます。

- **認識言語**: 既定は `lang="japan"`（日本語教科書向け）。英数字中心なら `run(device, lang="en")` のように変更。
- **可視化画像が不要な場合**: `save_to_img` を呼んでいる行を削除すると JSON のみ出力になります。

---

## ファイル構成

```
textbook_content_extractor/
├── data/                 # 入力画像を置く（バッチOCR用）
├── output/               # バッチOCRの出力（gpu/ , cpu/ に分かれる）
├── external/PaddleOCR/   # PaddleOCR（git submodule）
├── ocr_extract.py        # バッチOCR共通処理（画像 → PP-OCRv6 → JSON + 画像）
├── run_gpu.py            # バッチOCR GPU 版 実行ファイル（WSL Ubuntu）
├── run_cpu.py            # バッチOCR CPU 版 実行ファイル（Mac mini M4）
├── server/               # 文法下書きツール（FastAPI + ブラウザUI）
│   ├── app.py            #   APIサーバー本体（/api/extract, /api/upload-drafts）
│   ├── ocr.py            #   PP-OCRv6 エンジンのウォームなラッパー
│   ├── llm.py            #   OCRテキスト → 文法項目候補への構造化（OpenAI）
│   ├── config.py         #   .env 読み込み
│   └── static/index.html #   ブラウザUI（ビルド不要の単一ファイル）
├── .env                  # 秘密情報・接続先設定（git管理外）
├── .env.example          # .env のテンプレート
└── pyproject.toml        # 依存定義（プラットフォーム別に paddle を切替）
```

---

## 文法下書きツール（server/）

中国語文法本のページ画像を **PP-OCRv6 → LLM構造化 → vocab-trainer へ「下書き」アップロード** まで一気に処理する、ローカル専用の frontend/backend です。中間 JSON ファイルは生成せず、すべてメモリ上で処理します。

### 1. 初回セットアップ

```bash
cd textbook_content_extractor
cp .env.example .env   # 下記の変数を設定
uv sync                # fastapi / uvicorn / openai などが入る
```

`.env` の設定項目:

| 変数 | 内容 | 例 |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI の API キー（vocab-trainer/.env からコピー） | `sk-...` |
| `OPENAI_MODEL` | 構造化に使うモデル（vocab-trainer の `OPENAI_MODEL_MINI` の値） | `gpt-5.5` |
| `CLOUD_API_BASE_URL` | 下書きのアップロード先 backend | ローカル: `http://localhost:3000` / 本番: Cloud Run backend の URL |
| `CLOUD_UI_URL` | アップロード成功後に表示するリンク先 frontend | ローカル: `http://localhost:5173` / 本番: Cloud Run frontend の URL |
| `OCR_DEVICE` | `gpu`（WSL + NVIDIA）または `cpu`（Mac） | `gpu` |
| `OCR_LANG` | OCR 言語の既定値（UI 側で毎回切替可） | `ch` |

### 2. サーバー起動

```bash
uv run uvicorn server.app:app --port 8100
```

起動時に PP-OCRv6 エンジンをプリウォームするため、初回は数十秒かかります。「プリウォーム完了」が出たら準備完了です。

### 3. ブラウザでの操作

`http://localhost:8100` を開き:

1. **ページ画像をアップロード** — OCR 言語（`ch`=中国語 / `japan`=日本語。日本語で書かれた文法書は `japan` の方が説明文の精度が出る場合あり）、説明言語（`ja`/`en`/`ko`）、アップロード先言語（通常 `chinese`）を選んで「抽出」。OCR + LLM で 10〜30 秒程度。
2. **候補を確認・編集** — 抽出された文法項目がカードで並ぶ。文法パターン（statement）・レベル・説明・例文をその場で修正でき、チェックを外した候補は送信されない。生の OCR テキストも折りたたみで確認可能。
3. **グループ名（任意）** — カンマ区切りで入力すると、選択中の全候補の下書きに `groups`（グループ**名**の配列）が付く。下書きを vocab-trainer 側で本登録した時点で該当の文法グループに追加され、同名グループが無ければ自動作成される。教科書の課ごとにまとめる用途を想定。
4. **「選択した候補を下書きとしてアップロード」** — `CLOUD_API_BASE_URL` の vocab-trainer backend に下書きとして登録される。
   **「JSONダウンロード」** — 直接アップロードする代わりに、選択した候補を `<画像名>.grammar-drafts.json` としてローカル保存できる。このファイルは vocab-trainer の文法画面の「JSONアップロード」ボタンから読み込むと同じく下書きが一括作成される。ファイル形式の仕様は [docs/draft-json-format.md](docs/draft-json-format.md)（正本は vocab-trainer/docs/ 側）を参照。単語用の `word-drafts` 形式も同ドキュメントで定義済み（アップロード先は未実装）。

### 4. 下書きの本登録（vocab-trainer 側）

アップロード後の編集は vocab-trainer の文法画面で行います。「下書き」パネル → **確認** で自動記入済みのフォームが開き、4 言語定義の自動補完・例文の単語 chip・分かち書き変更（例文にスペースを入れて区切る）を使って調整し、保存すると本登録されて下書きは消えます。**破棄** で下書きだけ削除できます。

> ローカルでの動作確認は、vocab-trainer 側を `cd backend && npm run dev`（+ UI を見るなら `cd frontend && npm run dev`）で起動してから行ってください。
