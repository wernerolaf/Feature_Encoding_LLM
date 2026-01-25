"""
Generate an Excel file in the same column structure as data/LLM_nano.xlsx
from a JSON file of prompts and model generations.

Inputs expect a list of dicts shaped like artifacts/experiments/gender_clout/generations_*.json:
[
  {
    "prompt": "...",
    "baseline_generation": "...",
    "intervened_generation": "...",
    "tag": "...",
    "avg_logprob_delta": 0.0
  },
  ...
]

Dependencies (install if missing):
  pip install pandas openpyxl liwc
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from liwc_analysis import liwc_statistics_from_text, _tokenize

# Column order matching data/LLM_nano.xlsx
COLUMNS: List[str] = [
    "id",
    "level",
    "trait",
    "belief",
    "question",
    "type",
    "model",
    "gender",
    "pronoun",
    "response",
    "answered",
    "V1",
    "V2",
    "V3",
    "A1",
    "A2",
    "A3",
    "D1",
    "D2",
    "D3",
    "Hedge",
    "DAV",
    "IAV",
    "SV",
    "Language abstraction/concreteness",
    "WC",
    "analytic",
    "Clout",
    "Authentic",
    "Tone",
    "WPS",
    "BigWords",
    "pronoun_LIWC",
    "ppron",
    "i",
    "we",
    "you",
    "shehe",
    "they",
    "ipron",
    "det",
    "article",
    "number",
    "prep",
    "auxverb",
    "adverb",
    "conj",
    "negate",
    "verb",
    "adj",
    "Drives",
    "affiliation",
    "achieve",
    "power",
    "cognition",
    "allnone",
    "cogproc",
    "insight",
    "cause",
    "discrep",
    "tentat",
    "certitude",
    "differ",
    "memory",
    "affect",
    "tone_pos",
    "tone_neg",
    "emotion",
    "emo_pos",
    "emo_neg",
    "emo_anx",
    "emo_anger",
    "emo_sad",
    "swear",
    "Social",
    "socbehav",
    "prosocial",
    "polite",
    "conflict",
    "moral",
    "comm",
    "socrefs",
    "family",
    "friend",
    "female",
    "male",
    "Physical",
    "health",
    "illness",
    "wellness",
    "mental",
    "substances",
    "sexual",
    "food",
    "death",
    "need",
    "want",
    "acquire",
    "lack",
    "fulfill",
    "fatigue",
    "reward",
    "risk",
    "curiosity",
    "allure",
    "time",
    "focuspast",
    "focuspresent",
    "focusfuture",
    "Example",
    "unique_words_cnt",
]

# Default dictionary matches the LIWC-style column list above
DEFAULT_DICT = Path("data/llm_nano_liwc.dic")


def parse_prompt(prompt: str) -> Dict[str, Optional[str]]:
    """Pull metadata out of the prompt string (best-effort)."""
    level = trait = belief = question = qtype = gender = pronoun = None

    level_trait = re.search(r"(low|medium|high) level of ([^,]+)", prompt, re.IGNORECASE)
    if level_trait:
        level = level_trait.group(1).lower()
        trait = level_trait.group(2).strip()

    belief_q = re.search(r'answer\s+"([^"]+)"\s+to the question:\s+"([^"]+)"', prompt, re.IGNORECASE)
    if belief_q:
        belief = belief_q.group(1).strip()
        question = belief_q.group(2).strip()

    gender_match = re.search(r"\b(female|male)\b", prompt, re.IGNORECASE)
    if gender_match:
        gender = gender_match.group(1).lower()
        pronoun = "her" if gender == "female" else "him"

    type_match = re.search(r"Use\s+([A-Za-z]+)\s+arguments", prompt, re.IGNORECASE)
    if type_match:
        qtype = type_match.group(1).lower()

    return {
        "level": level,
        "trait": trait,
        "belief": belief,
        "question": question,
        "type": qtype,
        "gender": gender,
        "pronoun": pronoun,
    }


def basic_text_stats(text: str) -> Dict[str, float]:
    tokens = list(_tokenize(text))
    total = len(tokens)
    sentences = max(1, len([s for s in re.split(r"[.!?]+", text) if s.strip()]))
    big_words = sum(1 for t in tokens if len(t) > 6)
    unique_words = len(set(tokens))
    return {
        "total_tokens": total,
        "wps": (total / sentences) if total else 0.0,
        "big_words_pct": (100 * big_words / total) if total else 0.0,
        "unique_words_cnt": unique_words,
    }


def build_row(
    row_id: int, prompt: str, response: str, gen_type: str, dictionary_path: Path, model: Optional[str]
) -> Dict[str, Any]:
    row: Dict[str, Any] = {col: None for col in COLUMNS}
    meta = parse_prompt(prompt)
    row.update(meta)
    row["id"] = row_id
    # "type" reflects the argument style in the prompt (rational/emotional); fall back to variant.
    row["type"] = meta.get("type") or gen_type
    row["model"] = model
    row["response"] = response
    row["answered"] = meta.get("belief")

    liwc_stats = liwc_statistics_from_text(response, dictionary_path)
    text_stats = basic_text_stats(response)

    row["WC"] = text_stats["total_tokens"]
    row["WPS"] = text_stats["wps"]
    row["BigWords"] = text_stats["big_words_pct"]
    row["unique_words_cnt"] = text_stats["unique_words_cnt"]

    # Fill LIWC-like percentages (freq * 100)
    for category, freq in liwc_stats["category_frequencies"].items():
        if category in row:
            row[category] = freq * 100

    # Analytic/Clout/Authentic/Tone placeholders (require proprietary LIWC weights)
    for placeholder in ("analytic", "Clout", "Authentic", "Tone"):
        row.setdefault(placeholder, None)

    return row


def load_generations(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of dicts.")
    return data


def load_metadata(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def sanitize_sheet_name(name: str) -> str:
    """
    Excel sheet names must be <=31 chars and cannot contain: : \\ / ? * [ ]
    """
    invalid = set(':\\/?!*[]')
    cleaned = "".join(ch for ch in name if ch not in invalid)
    cleaned = cleaned.strip()
    return cleaned[:31] if cleaned else "sheet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an Excel file matching LLM_nano.xlsx structure.")
    parser.add_argument("--input", required=True, type=Path, help="Path to generations JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Output Excel path (e.g., output.xlsx).")
    parser.add_argument("--model", required=False, help="Model name to store in rows and sheet name (fallback to metadata).")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional path to metadata.json (defaults to alongside input).")
    parser.add_argument(
        "--dict",
        dest="dictionary",
        type=Path,
        default=DEFAULT_DICT,
        help="Path to LIWC dictionary (default=data/llm_nano_liwc.dic).",
    )
    parser.add_argument(
        "--include",
        choices=["baseline", "intervention", "both"],
        default="intervention",
        help="Which generations to include from each record.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="Optional path to write metadata JSON (model, dictionary, source).",
    )
    args = parser.parse_args()

    meta_path = args.metadata or args.input.parent / "metadata.json"
    meta = load_metadata(meta_path)
    model_name = args.model or meta.get("model_name")
    if not model_name:
        raise ValueError("Model name not provided and not found in metadata.")

    records = load_generations(args.input)
    rows: List[Dict[str, Any]] = []
    row_id = 1
    for record in records:
        prompt = record.get("prompt", "")
        if args.include in ("baseline", "both") and "baseline_generation" in record:
            rows.append(build_row(row_id, prompt, record["baseline_generation"], "baseline", args.dictionary, model_name))
            row_id += 1
        if args.include in ("intervention", "both") and "intervened_generation" in record:
            rows.append(
                build_row(row_id, prompt, record["intervened_generation"], "intervention", args.dictionary, model_name)
            )
            row_id += 1

    df = pd.DataFrame(rows, columns=COLUMNS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet_name = sanitize_sheet_name(model_name)
    with pd.ExcelWriter(args.output) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Wrote {len(df)} rows to {args.output} (sheet={sheet_name})")

    if args.metadata_out:
        meta_payload = {
            "model": model_name,
            "dictionary": str(args.dictionary),
            "source_json": str(args.input),
            "rows": len(df),
            "source_metadata": str(meta_path) if meta_path and meta_path.exists() else None,
        }
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_out.write_text(json.dumps(meta_payload, indent=2))
        print(f"Wrote metadata to {args.metadata_out}")


if __name__ == "__main__":
    main()
