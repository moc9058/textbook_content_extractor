"""OCRテキストを文法項目候補に構造化する（LLMは構造化のみ。

4言語への定義展開・ピンイン生成はクラウド側の smart-add が本登録時に行う）。
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

CANDIDATES_SCHEMA: dict[str, Any] = {
    "name": "grammar_candidates",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["statement", "level", "description", "examples"],
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": "The grammar pattern itself, in Chinese, using +/… notation as printed (e.g. 别+V+了, 把+O+V)",
                        },
                        "level": {"type": ["string", "null"]},
                        "description": {
                            "type": "string",
                            "description": "Concise explanation in the requested description language",
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

SYSTEM_PROMPT = """You receive raw OCR text lines (in top-to-bottom reading order) from one page of a textbook that teaches Chinese grammar, with explanations written in {description_language}.

Extract each distinct grammar point on the page as one candidate:
- `statement`: the grammar pattern itself, in Chinese, using the +/… notation as printed (e.g. 别+V+了, 把+O+V). If the page presents a structural pattern without a compact notation, write a concise pattern yourself.
- `description`: a concise explanation in {description_language}, synthesized from the page text. Do NOT translate into other languages.
- `examples`: the Chinese example sentences printed for that grammar point, each paired with its printed {description_language} translation. If the page gives no translation for a sentence, write a faithful translation yourself. Do not include pinyin/romanization lines in `sentence` — reconstruct the plain Chinese sentence.
- `level`: if a proficiency level (HSK, 中検 etc.) is printed for the point, set it; otherwise null.

Correct obvious OCR errors in Chinese sentences (wrong/similar-looking characters, split lines). Drop page headers, section numbers, exercise prompts, and unrelated fragments. If the page contains no grammar points, return an empty array."""


def structure_page(
    client: OpenAI, model: str, ocr_text: str, description_language: str
) -> list[dict[str, Any]]:
    lang_name = LANG_NAMES.get(description_language, description_language)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(description_language=lang_name)},
            {"role": "user", "content": ocr_text},
        ],
        response_format={"type": "json_schema", "json_schema": CANDIDATES_SCHEMA},
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content).get("candidates", [])
