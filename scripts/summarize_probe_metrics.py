#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from utils import sanitize_model_name


def version_num(path: Path) -> int:
    match = re.match(r"ver_(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize probe metadata/metrics across layers and features.")
    parser.add_argument("--probe-root", default="artifacts/probes")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--latest-only", action="store_true", help="Keep only latest version per layer/feature.")
    parser.add_argument("--standardizer", default=None, help="Optional standardizer filter, e.g. identity.")
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.probe_root) / sanitize_model_name(args.model_name)
    rows: list[dict[str, object]] = []
    for metadata_path in sorted(root.glob("layer*/**/ver_*/metadata.json")):
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metrics = metadata.get("metrics") or {}
        standardizer = metadata.get("standardizer")
        if args.standardizer and standardizer != args.standardizer:
            continue
        version_dir = metadata_path.parent
        feature_dir = version_dir.parent
        layer_dir = feature_dir.parent
        rows.append(
            {
                "layer": metadata.get("layer"),
                "feature": metadata.get("label_column") or feature_dir.name,
                "version": version_dir.name,
                "version_num": version_num(version_dir),
                "task": metadata.get("task"),
                "probe_type": metadata.get("probe_type"),
                "standardizer": standardizer,
                "layer_indexing": metadata.get("layer_indexing"),
                "data_path": metadata.get("data_path"),
                "artifact_path": metadata.get("artifact_path"),
                "metadata_path": str(metadata_path),
                "train_r2": metrics.get("train_r2"),
                "test_r2": metrics.get("test_r2"),
                "train_mse": metrics.get("train_mse"),
                "test_mse": metrics.get("test_mse"),
                "train_accuracy": metrics.get("train_accuracy"),
                "test_accuracy": metrics.get("test_accuracy"),
                "train_auc": metrics.get("train_auc"),
                "test_auc": metrics.get("test_auc"),
                "train_f1_macro": metrics.get("train_f1_macro"),
                "test_f1_macro": metrics.get("test_f1_macro"),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"No probe metadata found under {root}")
    if args.latest_only:
        frame = (
            frame.sort_values(["layer", "feature", "version_num"])
            .groupby(["layer", "feature"], as_index=False, dropna=False)
            .tail(1)
            .sort_values(["layer", "feature"])
        )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Wrote {len(frame)} rows to {output_path}")
    key_cols = ["layer", "feature", "task", "standardizer", "test_r2", "test_accuracy", "test_auc"]
    print(frame[key_cols].to_string(index=False))


if __name__ == "__main__":
    main()
