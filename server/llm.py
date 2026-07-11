"""OCR検出結果を文法項目/単語候補に構造化する（LLMは構造化のみ。

4言語への定義展開・ピンイン生成はクラウド側の smart-add が本登録時に行う）。
プロンプトは本ごとに分ける:
- 文法「HSK公認テキスト４級」 — ページ構造の前提は docs/book-structure.md
- 単語「新HSK1~4級単語トレーニングブック」 — docs/word-book-structure.md
"""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

# --- 文法本: HSK公認テキスト４級 ---

# statement 内の文法的役割の固定略語（小文字）。プロンプトに埋め込み、
# さらに normalize_statement() で出力後にも変換を保証する。
ELEMENT_ABBREVIATIONS: dict[str, str] = {
    "主語": "s",
    "動詞": "v",
    "述語": "v",
    "動作を表す述語": "v",
    "目的語": "o",
    "名詞": "n",
    "形容詞": "adj",
    "副詞": "adv",
    "動詞フレーズ": "vp",
    "量詞": "m",
    "数詞": "num",
}


def normalize_statement(statement: str) -> str:
    """statement の要素を正規化する: 全角＋→半角+、固定略語への変換、英字略語の小文字化。"""
    parts = re.split(r"\s*[+＋]\s*", statement)
    out = []
    for part in parts:
        p = part.strip()
        p = ELEMENT_ABBREVIATIONS.get(p, p)
        if re.fullmatch(r"[A-Za-z][A-Za-z/.…]*", p):
            p = p.lower()
        out.append(p)
    return "+".join(out)

GRAMMAR_CANDIDATES_SCHEMA: dict[str, Any] = {
    "name": "grammar_candidates",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["page_matches_book", "page_note", "candidates"],
        "properties": {
            "page_matches_book": {
                "type": "boolean",
                "description": "True if this page plausibly comes from the expected book described in the instructions; false if it clearly belongs to a different book or unrelated material.",
            },
            "page_note": {
                "type": ["string", "null"],
                "description": "When page_matches_book is false: a short Japanese note saying what the page appears to be instead. Null when true.",
            },
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["statement", "group", "level", "description", "examples"],
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": "The grammar pattern in normalized +/… notation: Chinese function words verbatim (从, 别, 把 …), grammatical roles as LOWERCASE English abbreviations (s, o, v, n, adj, adv …), semantic slots as concise Japanese nouns (人物, 場所, 時間 …). E.g. 别+v+了, 从+場所/時間+v",
                        },
                        "group": {
                            "type": ["string", "null"],
                            "description": "The Point title this pattern belongs to (the large-print heading near 'Point N', e.g. 中国語の基本的な語順). Null if no Point title is visible on the page.",
                        },
                        "level": {"type": ["string", "null"]},
                        "description": {
                            "type": "string",
                            "description": "Concise explanation in the requested description language, formatted as a bullet list (one point per line, each line starting with '- '), preserving the book's metaphors, analogies and paraphrases",
                        },
                        "examples": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["sentence", "translation"],
                                "properties": {
                                    "sentence": {"type": "string"},
                                    "translation": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            }
        },
    },
}

LANG_NAMES = {"ja": "Japanese", "en": "English", "ko": "Korean", "zh": "Chinese"}

GRAMMAR_SYSTEM_PROMPT = """You receive the raw OCR detections from one page of "HSK公認テキスト４級", a textbook that teaches Chinese grammar, with explanations written in {description_language}. Each input line has the form

[y=<top>-<bottom> x=<left>-<right> h=<box height>] <detected text>

Coordinates are pixels; y grows downward. Text fragments whose y-ranges overlap belong to the same printed line even when OCR split them into separate boxes (the page prints wide gaps inside patterns and between words) — merge them left-to-right by x. Box height h correlates with font size.

## Page layout

- The book has 4 Parts; "Part 1"…"Part 4" run down the left/right page margins (x near the page edge, often rotated/fragmentary). Ignore them.
- Each Part contains numbered Points. Near "Point N" is the Point title (e.g. 中国語の基本的な語順) printed noticeably LARGER (larger h) than body text. It is not necessarily at the top of the page. Set it as `group` for every candidate under that Point. Ignore the small-print introduction directly below the title.
- Section headings inside a Point (e.g. 文全体の述語を見つけよう!, 注意点) and their numbered items are organizational only — do not turn them into candidates and do not put them in `group`.

## What to extract

Each grammar point appears as a vertical 4-line set; extract one candidate per set:

1. Pattern line — a mix of grammar terms, "+" signs and Chinese (e.g. 从＋場所/時間＋動作を表す述語). Often split into several boxes at the same y (e.g. 从 / ＋ / 場所/時間 / ＋ / 動作を表す述語) — reassemble it, then NORMALIZE it into `statement`. Normalization rules:
   - Keep concrete Chinese function words verbatim (从, 别, 把, 了 …).
   - Grammatical roles become LOWERCASE abbreviations. Use this fixed mapping, always: {abbreviations}. For roles not in the mapping, coin a short lowercase English abbreviation in the same style. Never output uppercase abbreviations.
   - Semantic slots become concise Japanese nouns (人物, 場所, 時間, 抽象的な目標 …). Verbose printed descriptions must be shortened: 動作を表す述語 → v, 名詞（場所） → 場所.
   So 从＋場所/時間＋動作を表す述語 becomes 从+場所/時間+v. Join elements with half-width "+". Every element must be one of these three kinds — never leave a long descriptive phrase in `statement`. Sometimes the line is annotated with helper notes like （介詞フレーズ） or 結果補語 tying the pattern to the example — these are context, not separate candidates.
2. Pinyin line — romanization of line 3 (also fragmented). NEVER include pinyin in any output field; pinyin is regenerated downstream.
3. Chinese example sentence — may be split (e.g. 后悔 / 也来不及了。); reconstruct the full sentence and fix obvious OCR character errors.
4. {description_language} translation of line 3.

Lines 3+4 become one entry in `examples` ({{sentence, translation}}).

COMPLETENESS — do not miss patterns: virtually every line whose fragments are joined by "＋"/"+" is a pattern line and starts one of these 4-line sets. Before answering, re-scan ALL input lines for "＋"/"+" and verify that each such pattern line produced a candidate (or was merged into one deliberately). Missing a ＋-line is the most common failure.

3-LINE SETS — occasionally a set has NO pattern line: only pinyin / Chinese sentence / {description_language} translation. Read the surrounding explanation text to judge what it illustrates. If it demonstrates a DIFFERENT grammar element than the candidates already extracted, create a separate candidate for it and write a concise normalized `statement` yourself from the explanation (same notation rules). If it belongs to an existing pattern, append it to that candidate's `examples`.

Below a 4-line set there may be extra material — decide by content whether it belongs to the same pattern:
- Supplementary explanation in {description_language} (e.g. usage nuances of 也) → fold into `description`. Such explanations may embed additional Chinese sentences; their translation is often printed to the RIGHT (same y, larger x) rather than below — pair them and add to `examples` if they illustrate the same pattern.
- Word glosses like 「*网：ネットワーク（インターネット）」 → not a candidate; ignore or use only to inform `description`.
- If the material clearly describes a DIFFERENT grammar pattern, make it a separate candidate.

`description`: a concise explanation in {description_language} synthesized from the page text for that pattern, written as a BULLET LIST: one point per line, each line starting with "- " (meaning, usage conditions, nuances, notes as separate bullets). Keep each bullet brief, BUT never drop the metaphors, analogies or {description_language} paraphrases the book uses to explain the pattern — carry them over. Do NOT translate into other languages.
`level`: set only if a proficiency level (HSK, 中検 etc.) is printed for the point; otherwise null.

## Tables

The page may contain vocabulary tables (usually 4 columns: level/blank, word, meaning, example+notes; pinyin printed above words/sentences; cells wrap across boxes). These are word lists, not grammar points — do NOT emit candidates from them.

Drop page numbers, headers, exercise prompts and unrelated fragments. If the page contains no grammar points, return an empty array.

## Book check

Before extracting anything, judge whether this page really comes from THIS grammar textbook. Signals that it does: Part/Point structure, grammar-pattern lines with ＋ notation, 4-line sets (pattern / pinyin / Chinese sentence / {description_language} translation), section headings like 注意点. If the page instead looks like a different book — e.g. a vocabulary training book (numbered word rows in a 3-column table with □/★ marks and "UNIT N" margins) — or unrelated material, set `page_matches_book` to false, put a short Japanese note in `page_note` describing what the page appears to be, and return an empty `candidates` array. When the page matches, set `page_matches_book` to true and `page_note` to null."""

# 固定略語マッピングをプロンプトに焼き込む（{description_language} は structure_page で解決）
GRAMMAR_SYSTEM_PROMPT = GRAMMAR_SYSTEM_PROMPT.replace(
    "{abbreviations}", ", ".join(f"{ja} → {ab}" for ja, ab in ELEMENT_ABBREVIATIONS.items())
)


# --- 単語本: 新HSK1~4級単語トレーニングブック ---

WORD_CANDIDATES_SCHEMA: dict[str, Any] = {
    "name": "word_candidates",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["page_matches_book", "page_note", "candidates"],
        "properties": {
            "page_matches_book": {
                "type": "boolean",
                "description": "True if this page plausibly comes from the expected book described in the instructions; false if it clearly belongs to a different book or unrelated material.",
            },
            "page_note": {
                "type": ["string", "null"],
                "description": "When page_matches_book is false: a short Japanese note saying what the page appears to be instead. Null when true.",
            },
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["term", "transliteration", "level", "definitions", "examples"],
                    "properties": {
                        "term": {
                            "type": "string",
                            "description": "The Chinese headword exactly as printed (fix obvious OCR character errors)",
                        },
                        "transliteration": {
                            "type": ["string", "null"],
                            "description": "The pinyin printed next to the headword. Null if not readable.",
                        },
                        "level": {
                            "type": ["string", "null"],
                            "description": "HSK level only if explicitly printed for this word; otherwise null.",
                        },
                        "definitions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["partOfSpeech", "text"],
                                "properties": {
                                    "partOfSpeech": {
                                        "type": "string",
                                        "description": "Part-of-speech marker as printed (動, 名, 形, ...). Empty string if none.",
                                    },
                                    "text": {
                                        "type": "string",
                                        "description": "Japanese meaning as printed. May contain multiple senses and appended notes; newlines allowed.",
                                    },
                                },
                            },
                        },
                        "examples": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["segments", "translation"],
                                "properties": {
                                    "segments": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "The Chinese example sentence split into word segments following the pinyin word spacing. Punctuation attaches to the preceding segment.",
                                    },
                                    "translation": {
                                        "type": "string",
                                        "description": "Japanese translation of the sentence",
                                    },
                                },
                            },
                        },
                    },
                },
            }
        },
    },
}

WORD_SYSTEM_PROMPT = """You receive the raw OCR detections from one page of "新HSK1〜4級単語トレーニングブック", a Chinese vocabulary training book whose meanings and translations are written in Japanese. Each input line has the form

[y=<top>-<bottom> x=<left>-<right> h=<box height>] <detected text>

Coordinates are pixels; y grows downward. Text fragments whose y-ranges overlap belong to the same printed line even when OCR split them into separate boxes — merge them left-to-right by x. Box height h correlates with font size.

## Page layout

- The left/right page margins may show "UNIT N" (often rotated/fragmentary). Ignore it.
- Each unit contains numbered themes; the theme number (1, 2, 3, ...) is printed in (roughly) the LARGEST size on the page, followed by a theme title. A theme may contain sessions numbered like 2-3 or 14-1, each with its own title. All of these numbers and titles are organizational only — ignore them, never emit candidates from them.
- The vocabulary lives in a table of basically 3 columns, one row per word:
  1. Leftmost column: the row number, with a □ next to it and a ★ below the □. OCR may or may not pick up □/★ as characters. Ignore this column entirely; use the numbers only as hints for where a word row starts.
  2. Middle column: the headword cell — the Chinese word (largest h in the cell), its pinyin next to it, and the Japanese meaning below. When the meaning is too long for the cell width it wraps, so it may continue in a separate OCR box below. Meanings often carry part-of-speech markers such as 動, 名, 形 and may list multiple senses.
  3. Right column: one or MORE example-sentence cells for that word. Each cell contains the Chinese sentence (largest h in the cell), its pinyin (printed to the RIGHT of the sentence or BELOW it), and the Japanese translation (below the sentence and/or the pinyin).

## What to extract

One candidate per word row:

- `term`: the Chinese headword. Fix obvious OCR character errors.
- `transliteration`: the pinyin printed for the headword (null if unreadable). Do NOT invent pinyin.
- `definitions`: from the Japanese meaning text. Use `partOfSpeech` for markers like 動/名/形 (empty string if none); `text` holds the meaning as printed and may contain several senses. Reassemble wrapped meaning lines that OCR split across boxes.
- `examples`: one entry per example cell belonging to that word row (match by y-range against the headword cell; a single word often has several example cells stacked vertically). For each cell:
  - `segments`: the Chinese sentence split into word/grammar-element segments **strictly following the word spacing of its pinyin line** — the pinyin is space-separated per word/grammar element (it may be spread over several boxes, or space-separated inside one box; use both the spaces and the box boundaries). Attach punctuation to the preceding segment. Joining the segments must reproduce the printed sentence exactly.
  - `translation`: the Japanese translation.
  - NEVER output the pinyin of example sentences anywhere.

## Extra material

- Sometimes an annotation appears under a headword cell (e.g. 「学び合う」という意味がある) → append it to that word's definition `text` (newlines are allowed).
- Between word rows there may be comparison notes contrasting previously introduced words, as lines of the form （中国語単語）：日本語説明, possibly one per line and possibly wrapped across boxes. If the compared word is a headword on THIS page, append the explanation to that word's definition text; otherwise ignore the line. Never emit a comparison note as its own candidate.

Drop page numbers, headers, unit/theme/session numbers and titles, □/★ artifacts, and unrelated fragments. If the page contains no vocabulary table, return an empty array.

## Book check

Before extracting anything, judge whether this page really comes from THIS vocabulary training book. Signals that it does: numbered word rows in a 3-column table (number+□/★ column, headword+pinyin+Japanese meaning column, example-sentence column), "UNIT N" margins, theme/session numbering. If the page instead looks like a different book — e.g. a grammar textbook (Part/Point structure, grammar-pattern lines with ＋ notation, 4-line pattern/pinyin/sentence/translation sets) — or unrelated material, set `page_matches_book` to false, put a short Japanese note in `page_note` describing what the page appears to be, and return an empty `candidates` array. When the page matches, set `page_matches_book` to true and `page_note` to null."""


def format_ocr_items(items: list[dict[str, Any]]) -> str:
    """extract_detail の検出結果を座標注釈付きテキストに整形する。"""
    lines = []
    for it in items:
        x0, y0, x1, y1 = (int(v) for v in it["box"])
        lines.append(f"[y={y0}-{y1} x={x0}-{x1} h={y1 - y0}] {it['text']}")
    return "\n".join(lines)


def _structure(
    client: OpenAI,
    model: str,
    system_prompt: str,
    schema: dict[str, Any],
    ocr_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """{page_matches_book, page_note, candidates} を返す。"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": format_ocr_items(ocr_items)},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
    )
    content = response.choices[0].message.content or "{}"
    result = json.loads(content)
    return {
        "page_matches_book": result.get("page_matches_book", True),
        "page_note": result.get("page_note"),
        "candidates": result.get("candidates", []),
    }


# 本のレジストリ。新しい本を追加するときは、その本用のプロンプト+スキーマを書いてここに登録する。
# kind が候補の形（grammar/word）とUIのセッション・アップロード先を決める。
BOOKS: dict[str, dict[str, Any]] = {
    "hsk4-grammar": {
        "kind": "grammar",
        "title": "HSK公認テキスト４級",
        "prompt": GRAMMAR_SYSTEM_PROMPT,
        "schema": GRAMMAR_CANDIDATES_SCHEMA,
    },
    "hsk1-4-word-training": {
        "kind": "word",
        "title": "新HSK1~4級単語トレーニングブック",
        "prompt": WORD_SYSTEM_PROMPT,
        "schema": WORD_CANDIDATES_SCHEMA,
    },
}

DEFAULT_BOOK_BY_KIND = {"grammar": "hsk4-grammar", "word": "hsk1-4-word-training"}


def structure_page(
    client: OpenAI,
    model: str,
    book_id: str,
    ocr_items: list[dict[str, Any]],
    description_language: str = "ja",
) -> dict[str, Any]:
    book = BOOKS[book_id]
    prompt = book["prompt"]
    if "{description_language}" in prompt:
        lang_name = LANG_NAMES.get(description_language, description_language)
        prompt = prompt.format(description_language=lang_name)
    result = _structure(client, model, prompt, book["schema"], ocr_items)
    if book["kind"] == "grammar":
        # 略語の小文字化・固定マッピングをLLM出力後にも保証する
        for cand in result["candidates"]:
            if cand.get("statement"):
                cand["statement"] = normalize_statement(cand["statement"])
    return result
