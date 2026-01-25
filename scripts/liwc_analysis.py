"""
Utility helpers for running quick LIWC-style counts.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import liwc


def _tokenize(text: str) -> Iterable[str]:
    """Simple word tokenizer that lowercases and strips punctuation."""
    return re.findall(r"[A-Za-z']+", text.lower())


def load_dictionary(dictionary_path: Path) -> Tuple[Callable[[str], Iterable[str]], Tuple[str, ...]]:
    """Load a LIWC dictionary and return the parser function and category names."""
    parse, category_names = liwc.load_token_parser(str(dictionary_path))
    return parse, tuple(category_names)


def _summarize_counts(
    tokens: Iterable[str], parse: Callable[[str], Iterable[str]], categories: Tuple[str, ...]
) -> Tuple[int, Counter[str], Dict[str, float]]:
    counts: Counter[str] = Counter()
    total_tokens = 0

    for token in tokens:
        total_tokens += 1
        counts.update(parse(token))

    frequencies = {
        category: (counts[category] / total_tokens) if total_tokens else 0.0
        for category in categories
    }

    return total_tokens, counts, frequencies


def liwc_statistics_from_text(text: str, dictionary_path: Path) -> Dict[str, object]:
    """
    Calculate LIWC category counts and relative frequencies for a string.

    Returns a dict with total token count, raw counts per category, and
    frequency per category (count divided by total tokens).
    """
    parse, categories = load_dictionary(dictionary_path)
    tokens = _tokenize(text)
    total_tokens, counts, frequencies = _summarize_counts(tokens, parse, categories)

    return {
        "source": "string",
        "dictionary": str(dictionary_path),
        "total_tokens": total_tokens,
        "category_counts": {category: counts.get(category, 0) for category in categories},
        "category_frequencies": frequencies,
    }


def liwc_statistics(file_path: Path, dictionary_path: Path) -> Dict[str, object]:
    """
    Calculate LIWC category counts and relative frequencies for a text file.

    Returns a dict with total token count, raw counts per category, and
    frequency per category (count divided by total tokens).
    """
    parse, categories = load_dictionary(dictionary_path)
    with open(file_path, "r", encoding="utf-8") as handle:
        tokens = (token for line in handle for token in _tokenize(line))
        total_tokens, counts, frequencies = _summarize_counts(tokens, parse, categories)

    return {
        "file": str(file_path),
        "dictionary": str(dictionary_path),
        "total_tokens": total_tokens,
        "category_counts": {category: counts.get(category, 0) for category in categories},
        "category_frequencies": frequencies,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run LIWC-style counts for a text file.")
    parser.add_argument("file", type=Path, help="Path to the input text file.")
    parser.add_argument(
        "--dict",
        dest="dictionary",
        type=Path,
        default=Path("data/llm_nano_liwc.dic"),
        help="Path to a LIWC dictionary file.",
    )
    args = parser.parse_args()

    stats = liwc_statistics(args.file, args.dictionary)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    _cli()
