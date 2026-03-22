from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd

PROMPT_TEMPLATE = (
    "you want to convince your {gender} interlocutor with a {level} level of {trait}, "
    'and answer "{belief}" to the question: "{question}". '
    "Use {type} arguments to change {pronoun}'s mind.\n"
)


class SafeDict(dict):
    def __missing__(self, key):
        return ""


def build_prompt_from_row(
    row: pd.Series,
    template: str,
    template_fields: Iterable[str],
) -> str:
    values = {field: row.get(field, "") for field in template_fields}
    return template.format_map(SafeDict(values))


def build_prompt_series(
    df: pd.DataFrame,
    *,
    text_column: str | None,
    prompt_template: str | None,
    template_fields: Iterable[str],
) -> pd.Series:
    if text_column and text_column in df.columns:
        series = df[text_column]
    else:
        if not prompt_template:
            raise ValueError("Provide --text-column or --prompt-template")

        def render(row: pd.Series) -> str:
            return build_prompt_from_row(row, prompt_template, template_fields)

        series = df.apply(render, axis=1)

    cleaned, mask = _clean_text_series(series)
    return cleaned.loc[mask]


def build_text_from_row(
    row: pd.Series,
    *,
    text_column: str | None,
    prompt_template: str | None,
    template_fields: Iterable[str],
    response_column: str | None = None,
    prompt_response_sep: str = "",
) -> str:
    if text_column and text_column in row.index:
        return str(row.get(text_column, "")).strip()

    if not prompt_template:
        raise ValueError("Provide --text-column or --prompt-template")

    prompt = build_prompt_from_row(row, prompt_template, template_fields)
    response = ""
    if response_column and response_column in row.index:
        response = str(row.get(response_column, ""))
    if response:
        return f"{prompt}{prompt_response_sep}{response}".strip()
    return str(prompt).strip()


def build_text_series(
    df: pd.DataFrame,
    *,
    text_column: str | None,
    prompt_template: str | None,
    template_fields: Iterable[str],
    response_column: str | None = None,
    prompt_response_sep: str = "",
) -> Tuple[pd.Series, pd.DataFrame]:
    if text_column and text_column in df.columns:
        series = df[text_column]
    else:
        if not prompt_template:
            raise ValueError("Provide --text-column or --prompt-template")

        def render(row: pd.Series) -> str:
            return build_prompt_from_row(row, prompt_template, template_fields)

        prompts = df.apply(render, axis=1)
        if response_column and response_column in df.columns:
            responses = df[response_column].fillna("").astype(str)
            series = prompts + prompt_response_sep + responses
        else:
            series = prompts

    series, mask = _clean_text_series(series)
    return series.loc[mask], df.loc[mask].copy()


def _clean_text_series(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    cleaned = series.fillna("").astype(str).str.strip()
    mask = cleaned != ""
    return cleaned, mask
