# 下書きJSONフォーマット仕様

textbook-content-extractor（ローカルOCRツール）のアウトプットとして生成し、vocab-trainer のUIにアップロードして下書きを一括作成するためのJSONファイル形式。

- 文字コード: UTF-8
- 拡張子: `.json`
- 1ファイル = 1言語。`kind` で文法（`grammar-drafts`）と単語（`word-drafts`）を区別する。

## 共通エンベロープ

```json
{
  "version": 1,
  "kind": "grammar-drafts | word-drafts",
  "language": "chinese | japanese | korean | english",
  "drafts": [ ... ]
}
```

| フィールド | 必須 | 説明 |
|---|---|---|
| `version` | 任意 | フォーマットバージョン。現在は `1`。省略時は `1` 扱い |
| `kind` | 必須 | `"grammar-drafts"` または `"word-drafts"` |
| `language` | 必須 | アップロード先言語（backendのフルネーム形式。ISOコードではない） |
| `drafts` | 必須 | 下書きの配列（1件以上） |

アップロードUIは、ファイルの `language` が現在表示中の言語と一致しない場合はエラーにする（誤アップロード防止）。

## kind: "grammar-drafts"（文法下書き）

`POST /api/grammar/:language/drafts` の入力にそのまま対応する。

```json
{
  "version": 1,
  "kind": "grammar-drafts",
  "language": "chinese",
  "drafts": [
    {
      "statement": "把+O+V",
      "transliteration": "ba+O+V",
      "descriptions": [
        { "partOfSpeech": "", "text": { "ja": "処置文。対象に処置を加えることを表す。" }, "pinyins": [] }
      ],
      "examples": [
        { "sentence": "我把书放在桌子上了。", "translation": "私は本を机の上に置いた。" }
      ],
      "level": "HSK4",
      "tags": [],
      "groups": ["第3課"],
      "sourceImage": "page_012.png"
    }
  ]
}
```

各draftのフィールド:

| フィールド | 必須 | 型 | 説明 |
|---|---|---|---|
| `statement` | 必須 | string | 文法パターン（中国語、`+`/`…` 記法） |
| `transliteration` | 任意 | string | `statement` のピンイン。**中国語部分だけ**をトーン無し小文字ピンインにし、英語略語・日本語スロット・`+`/`/`・括弧はそのまま（例: `能(/可以)+v`→`neng(/keyi)+v`）。中国語のみ有効 |
| `descriptions` | 必須 | Meaning[] | 説明。通常は1件・1言語のみ（`text` は ISOコード→説明文 のマップ）。残り言語は本登録時の smart-add が補完する |
| `examples` | 任意 | {sentence, translation}[] | 例文。ピンイン・分かち書きは本登録時に生成されるため含めない |
| `level` | 任意 | string | HSK等のレベル表記 |
| `tags` | 任意 | string[] | タグ |
| `groups` | 任意 | string[] | 追加先の文法グループ名（`grammar_groups.name`）。IDではなく**名前**で指定する。下書きを本登録（保存）した時点で解決され、同名グループが無ければ自動作成される |
| `sourceImage` | 任意 | string | 元画像ファイル名（トレーサビリティ用） |

## kind: "word-drafts"（単語下書き）

`POST /api/vocab/:language/drafts` の入力にそのまま対応する。単語一覧（Browse）の「JSONアップロード」ボタンから取り込み、下書きパネルで確認 → スマート追加モーダル（プレフィル済み）で本登録する。

```json
{
  "version": 1,
  "kind": "word-drafts",
  "language": "chinese",
  "drafts": [
    {
      "term": "后悔",
      "transliteration": "hòuhuǐ",
      "definitions": [
        { "partOfSpeech": "動詞", "text": { "ja": "後悔する" }, "pinyins": ["hòuhuǐ"] }
      ],
      "examples": [
        {
          "sentence": "后悔也来不及了。",
          "translation": "後悔しても間に合わない。",
          "segments": ["后悔", "也", "来不及", "了"]
        }
      ],
      "level": "HSK4",
      "topics": [],
      "groups": ["第3課"],
      "sourceImage": "page_012.png"
    }
  ]
}
```

各draftのフィールド:

| フィールド | 必須 | 型 | 説明 |
|---|---|---|---|
| `term` | 必須 | string | 見出し語 |
| `transliteration` | 任意 | string | 読み（中国語ならピンイン）。省略時は smart-add が生成 |
| `definitions` | 任意 | Meaning[] | 定義。1言語だけ埋めればよい（4言語補完は smart-add）。省略時は smart-add が全生成 |
| `examples` | 任意 | {sentence, translation, segments?}[] | 例文。`segments` はチップ（分かち書き）情報：文中の語セグメント**文字列**を文順に並べた配列（中国語のみ有効。`Example.segments` のオブジェクト形式とは別物）。確認モーダルでスペース区切りのチップ表示に復元され、本登録時に smart-add の `userSplits` として渡り `example_sentences.segments`（ピンイン・語ID付き）の生成に使われる。省略時はチップ分割なし（本登録時にLLMが分かち書きを生成） |
| `level` | 任意 | string | HSK/JLPT等 |
| `topics` | 任意 | string[] | `Word.topics`（`TOPICS` 定数のいずれか） |
| `groups` | 任意 | string[] | 追加先の単語グループ名（`word_groups.name`）。文法と同じく名前で指定し、本登録時に解決される（既存グループは選択済みで表示、無ければ保存時に自動作成） |
| `sourceImage` | 任意 | string | 元画像ファイル名 |

## 型の対応

- `Meaning` = `{ partOfSpeech: string, text: Record<string,string>, pinyins?: string[] }`（`backend/src/types.ts` / `frontend/src/types.ts`）
- 文法draftは `GrammarDraft`、単語draftは `WordDraft`（`id`/`language`/`createdAt` はサーバー付与のため**ファイルには含めない**）
