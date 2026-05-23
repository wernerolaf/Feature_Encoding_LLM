#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from scripts.prompt_utils import PROMPT_TEMPLATE, build_text_series
from utils import load_json, load_table, sanitize_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate regression probe R2 from feature_interaction token stats "
            "and compare it with the probe metadata test_r2."
        )
    )
    parser.add_argument("--run-dir", required=True, help="feature_interaction ver_* directory.")
    parser.add_argument("--data-path", required=True, help="Same data file used for probe training and scoring.")
    parser.add_argument("--probe-root", default="artifacts/probes", help="Root containing trained probe artifacts.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--features", nargs="+", required=True, help="Regression label columns to validate.")
    parser.add_argument("--token-stats-file", default="token_stats_baseline_old.parquet")
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--prompt-template", default=PROMPT_TEMPLATE)
    parser.add_argument(
        "--template-fields",
        nargs="+",
        default=["gender", "level", "trait", "belief", "question", "type", "pronoun"],
    )
    parser.add_argument("--response-column", default="response")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--probe-version", default="latest_raw", choices=["latest_raw", "latest", "latest_autoencoder"])
    parser.add_argument("--model-filter-column", default="model_name")
    parser.add_argument("--model-filter-value", default=None)
    parser.add_argument("--no-model-filter", action="store_true")
    parser.add_argument("--skip-special", action="store_true", help="Ignore rows marked is_special in token stats.")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Fail if abs(recomputed-test_r2) exceeds this.")
    parser.add_argument("--output-csv", default=None, help="Optional path for validation summary CSV.")
    return parser.parse_args()


def canonical_model_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def filter_rows_for_model(
    df: pd.DataFrame,
    *,
    model_name: str,
    column: str,
    filter_value: str | None,
) -> pd.DataFrame:
    if column not in df.columns:
        return df
    target = filter_value or model_name
    target_keys = {canonical_model_key(target)}
    if "/" in model_name:
        target_keys.add(canonical_model_key(model_name.split("/")[-1]))
    keys = df[column].map(canonical_model_key)
    return df.loc[keys.isin(target_keys)].copy()


def version_number(path: Path) -> int:
    match = re.match(r"ver_(\d+)$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def probe_matches(metadata: dict, mode: str) -> bool:
    strategy = metadata.get("standardizer")
    autoencoder_artifact = metadata.get("autoencoder_artifact")
    uses_autoencoder = strategy == "autoencoder" or bool(autoencoder_artifact)
    if mode == "latest_raw":
        return not uses_autoencoder
    if mode == "latest_autoencoder":
        return uses_autoencoder
    return True


def latest_probe_metadata(
    *,
    probe_root: Path,
    model_name: str,
    layer: int,
    feature: str,
    version: str,
) -> tuple[Path | None, dict | None]:
    feature_dir = probe_root / sanitize_model_name(model_name) / f"layer{layer}" / feature
    if not feature_dir.exists():
        return None, None
    candidates: list[tuple[int, Path, dict]] = []
    for version_dir in sorted(feature_dir.glob("ver_*"), key=version_number):
        metadata_path = version_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = load_json(metadata_path)
        if probe_matches(metadata, version):
            candidates.append((version_number(version_dir), version_dir, metadata))
    if not candidates:
        return None, None
    _, version_dir, metadata = candidates[-1]
    return version_dir, metadata


def prepare_rows(df: pd.DataFrame, args: argparse.Namespace, features: list[str]) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).lower() for col in df.columns]
    features = [feature.lower() for feature in features]
    if not args.no_model_filter:
        df = filter_rows_for_model(
            df,
            model_name=args.model_name,
            column=args.model_filter_column.lower(),
            filter_value=args.model_filter_value,
        )
    _, filtered_df = build_text_series(
        df,
        text_column=args.text_column,
        prompt_template=args.prompt_template,
        template_fields=args.template_fields,
        response_column=args.response_column,
        prompt_response_sep="",
    )
    missing = [feature for feature in features if feature not in filtered_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in data: {missing}")
    mask = pd.Series(True, index=filtered_df.index)
    for feature in features:
        mask &= filtered_df[feature].notna()
    filtered_df = filtered_df.loc[mask].reset_index(drop=True)
    if args.max_samples is not None:
        filtered_df = filtered_df.iloc[: args.max_samples].reset_index(drop=True)
    return filtered_df


def reproduce_test_rows(df: pd.DataFrame, first_feature: str, args: argparse.Namespace) -> np.ndarray:
    row_indices = np.arange(len(df))
    y_rows_for_split = df[first_feature].to_numpy()
    task_for_split = "regression" if pd.api.types.is_float_dtype(df[first_feature]) else "classification"
    stratify_rows = y_rows_for_split if task_for_split == "classification" else None
    train_rows, test_rows = train_test_split(
        row_indices,
        test_size=args.test_size,
        stratify=stratify_rows,
        random_state=args.random_state,
    )
    if args.val_size and args.val_size > 0:
        if task_for_split == "classification":
            stratify_train = y_rows_for_split[train_rows]
        else:
            stratify_train = None
        train_test_split(
            train_rows,
            test_size=args.val_size,
            stratify=stratify_train,
            random_state=args.random_state,
        )
    return np.asarray(test_rows)


def read_token_stats(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported token stats format: {path}")


def finite_or_nan(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if np.isfinite(result) else math.nan


def main() -> None:
    args = parse_args()
    features = [feature.lower() for feature in args.features]
    df = prepare_rows(load_table(args.data_path, sheet=args.sheet), args, features)
    if df.empty:
        raise ValueError("No rows left after filtering.")
    test_rows = reproduce_test_rows(df, features[0], args)

    token_stats_path = Path(args.run_dir) / args.token_stats_file
    token_stats = read_token_stats(token_stats_path)
    if "prompt_index" not in token_stats.columns:
        raise ValueError(f"{token_stats_path} does not contain prompt_index.")
    if args.skip_special and "is_special" in token_stats.columns:
        token_stats = token_stats.loc[~token_stats["is_special"].astype(bool)].copy()
    token_stats = token_stats.loc[token_stats["prompt_index"].isin(test_rows)].copy()
    if token_stats.empty:
        raise ValueError("No token-stat rows matched the reproduced test split.")

    rows: list[dict[str, object]] = []
    for feature in features:
        pred_col = f"{feature}_L{args.layer}"
        if pred_col not in token_stats.columns:
            rows.append({"feature": feature, "status": f"missing token column {pred_col}"})
            continue
        probe_dir, metadata = latest_probe_metadata(
            probe_root=Path(args.probe_root),
            model_name=args.model_name,
            layer=args.layer,
            feature=feature,
            version=args.probe_version,
        )
        if metadata is None:
            rows.append({"feature": feature, "status": "missing probe metadata"})
            continue
        metadata_r2 = finite_or_nan((metadata.get("metrics") or {}).get("test_r2"))
        if not np.isfinite(metadata_r2):
            rows.append(
                {
                    "feature": feature,
                    "probe_dir": str(probe_dir),
                    "task": metadata.get("task"),
                    "status": "metadata has no finite test_r2",
                }
            )
            continue
        frame = token_stats[["prompt_index", pred_col]].copy()
        frame[pred_col] = pd.to_numeric(frame[pred_col], errors="coerce")
        frame = frame.loc[frame[pred_col].notna()].copy()
        y_true = df.loc[frame["prompt_index"].to_numpy(), feature].to_numpy(dtype=float)
        y_pred = frame[pred_col].to_numpy(dtype=float)
        recalculated_r2 = float(r2_score(y_true, y_pred))
        rows.append(
            {
                "feature": feature,
                "status": "ok",
                "probe_dir": str(probe_dir),
                "task": metadata.get("task"),
                "standardizer": metadata.get("standardizer"),
                "metadata_test_r2": metadata_r2,
                "recalculated_test_r2": recalculated_r2,
                "abs_diff": abs(recalculated_r2 - metadata_r2),
                "test_prompts": int(len(test_rows)),
                "test_tokens": int(len(frame)),
            }
        )

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"[validate] wrote {output_path}")

    bad_status = summary.loc[summary.get("status") != "ok"]
    r2_failures = summary.loc[
        (summary.get("status") == "ok")
        & (pd.to_numeric(summary.get("abs_diff"), errors="coerce") > args.tolerance)
    ]
    failures = pd.concat([bad_status, r2_failures], ignore_index=True)
    if not failures.empty:
        raise SystemExit(
            f"R2 validation failed for {len(failures)} feature(s); tolerance={args.tolerance}."
        )


if __name__ == "__main__":
    main()
