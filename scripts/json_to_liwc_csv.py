#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable


DEFAULT_TEXT_FIELDS = ("baseline_generation", "intervened_generation")


EXPERIMENT_RE = re.compile(
    r".*?_L(?P<layer>\d+)_(?P<mode>increase|decrease|project)_(?P<strength>[\dpm]+)$"
)


def load_json_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"{path} does not contain a top-level JSON list.")

    records: list[dict] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{path} record {idx} is not a JSON object.")
        records.append(item)
    return records


def iter_liwc_rows(
    records: list[dict],
    *,
    source_file: Path,
    root: Path | None,
    text_fields: Iterable[str],
    include_empty: bool,
) -> Iterable[dict[str, object]]:
    stem = source_file.stem
    run_dir = source_file.parent
    experiment_dir = run_dir.parent
    experiment_name = experiment_dir.name
    version = run_dir.name
    rel_source_path = str(source_file)
    rel_run_dir = str(run_dir)
    if root is not None:
        try:
            rel_source_path = str(source_file.resolve().relative_to(root.resolve()))
            rel_run_dir = str(run_dir.resolve().relative_to(root.resolve()))
        except ValueError:
            pass

    parsed = EXPERIMENT_RE.match(experiment_name)
    intervention_layer = parsed.group("layer") if parsed else ""
    intervention_mode = parsed.group("mode") if parsed else ""
    intervention_strength = parsed.group("strength").replace("p", ".").replace("m", "-") if parsed else ""
    generation_condition = stem.replace("generations_", "")

    for record_index, record in enumerate(records):
        prompt = record.get("prompt", "")
        tag = record.get("tag", "")
        avg_logprob_delta = record.get("avg_logprob_delta")

        for text_field in text_fields:
            value = record.get(text_field, "")
            if value is None:
                value = ""
            if not isinstance(value, str):
                value = str(value)
            if not include_empty and not value.strip():
                continue

            yield {
                "source_file": source_file.name,
                "source_path": rel_source_path,
                "source_path_abs": str(source_file.resolve()),
                "run_dir": rel_run_dir,
                "experiment_name": experiment_name,
                "version": version,
                "generation_condition": generation_condition,
                "intervention_layer": intervention_layer,
                "intervention_mode": intervention_mode,
                "intervention_strength": intervention_strength,
                "record_index": record_index,
                "text_id": f"{stem}_{record_index}_{text_field}",
                "text_kind": text_field,
                "tag": tag,
                "avg_logprob_delta": avg_logprob_delta,
                "prompt": prompt,
                "text": value,
            }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert generation JSON files into a LIWC-friendly CSV."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more JSON files produced by the feature interaction pipeline.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination CSV file.",
    )
    parser.add_argument(
        "--text-fields",
        nargs="+",
        default=list(DEFAULT_TEXT_FIELDS),
        help=(
            "JSON fields to export as text rows. "
            "Default: baseline_generation intervened_generation"
        ),
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Keep rows whose text field is empty.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional root used to make source_path and run_dir relative.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_file",
        "source_path",
        "source_path_abs",
        "run_dir",
        "experiment_name",
        "version",
        "generation_condition",
        "intervention_layer",
        "intervention_mode",
        "intervention_strength",
        "record_index",
        "text_id",
        "text_kind",
        "tag",
        "avg_logprob_delta",
        "prompt",
        "text",
    ]

    row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for input_path in args.inputs:
            records = load_json_records(input_path)
            for row in iter_liwc_rows(
                records,
                source_file=input_path,
                root=args.root,
                text_fields=args.text_fields,
                include_empty=args.include_empty,
            ):
                writer.writerow(row)
                row_count += 1

    print(f"Wrote {row_count} rows to {output_path}")


if __name__ == "__main__":
    main()
