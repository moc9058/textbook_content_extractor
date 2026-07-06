# textbook-content-extractor

`data/` 内の画像を [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) の **PP-OCRv6** モデルで OCR し、認識結果を **JSON** と **可視化画像** として出力するツールです。

- **Windows WSL Ubuntu（NVIDIA GPU）** → GPU 推論
- **Mac mini M4（Apple Silicon）** → CPU 推論

同じロジック（`ocr_extract.py`）を、環境ごとの実行ファイル（`run_gpu.py` / `run_cpu.py`）から `device` を切り替えて呼び出します。

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

## 使い方

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
├── data/                 # 入力画像を置く
├── output/               # 出力（gpu/ , cpu/ に分かれる）
├── external/PaddleOCR/   # PaddleOCR（git submodule）
├── ocr_extract.py        # 共通処理（画像 → PP-OCRv6 → JSON + 画像）
├── run_gpu.py            # GPU 版 実行ファイル（WSL Ubuntu）
├── run_cpu.py            # CPU 版 実行ファイル（Mac mini M4）
└── pyproject.toml        # 依存定義（プラットフォーム別に paddle を切替）
```
