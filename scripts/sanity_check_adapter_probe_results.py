#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRED_COL_RE = re.compile(r"(?P<feature>.+)_L(?P<probe_layer>\d+)$")
VERSION_RE = re.compile(r"ver_(\d+)$")


@dataclass
class CheckResult:
    level: str
    message: str


class Reporter:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def ok(self, message: str) -> None:
        self.results.append(CheckResult("OK", message))

    def warn(self, message: str) -> None:
        self.results.append(CheckResult("WARN", message))

    def fail(self, message: str) -> None:
        self.results.append(CheckResult("FAIL", message))

    @property
    def failed(self) -> bool:
        return any(result.level == "FAIL" for result in self.results)

    def print(self) -> None:
        width = max(len(result.level) for result in self.results) if self.results else 4
        for result in self.results:
            print(f"[{result.level:<{width}}] {result.message}")


def version_number(path: Path) -> int:
    match = VERSION_RE.match(path.name)
    return int(match.group(1)) if match else -1


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(args.run_dir).expanduser().resolve()

    if args.slurm_log:
        log_path = Path(args.slurm_log).expanduser().resolve()
        text = log_path.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"(artifacts/experiments/[^\s]+/ver_\d+)", text)
        if matches:
            return (log_path.parent / matches[-1]).resolve()

    root = Path(args.root).expanduser().resolve()
    versions = sorted(
        [p for p in root.rglob("ver_*") if p.is_dir()],
        key=lambda p: (p.stat().st_mtime, version_number(p)),
        reverse=True,
    )
    if not versions:
        raise FileNotFoundError(f"No ver_* directories found below {root}")
    return versions[0]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).schema.names
    except Exception:
        return list(pd.read_parquet(path).columns)


def parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return int(len(pd.read_parquet(path, columns=[])))


def prediction_columns(columns: list[str]) -> list[str]:
    return [col for col in columns if PRED_COL_RE.match(col)]


def parse_prediction_column(col: str) -> tuple[str, int] | None:
    match = PRED_COL_RE.match(col)
    if not match:
        return None
    return match.group("feature"), int(match.group("probe_layer"))


def summarize_generation_file(path: Path, expected_prompts: int | None, reporter: Reporter) -> None:
    payload = read_json(path)
    if not isinstance(payload, list):
        reporter.fail(f"{path.name}: expected a JSON list")
        return
    if expected_prompts is not None and len(payload) != expected_prompts:
        reporter.fail(f"{path.name}: {len(payload)} rows, expected {expected_prompts}")
    else:
        reporter.ok(f"{path.name}: {len(payload)} rows")

    if not payload:
        return

    required = {"prompt", "baseline_generation", "intervened_generation", "tag", "avg_logprob_delta"}
    missing = required - set(payload[0])
    if missing:
        reporter.fail(f"{path.name}: first row missing keys {sorted(missing)}")

    prompt_lengths = [len(str(row.get("prompt", ""))) for row in payload if isinstance(row, dict)]
    base_lengths = [len(str(row.get("baseline_generation", ""))) for row in payload if isinstance(row, dict)]
    int_lengths = [len(str(row.get("intervened_generation", ""))) for row in payload if isinstance(row, dict)]
    avg_base = float(np.mean(base_lengths)) if base_lengths else math.nan
    avg_int = float(np.mean(int_lengths)) if int_lengths else math.nan
    empty_int = sum(length == 0 for length in int_lengths)
    if empty_int:
        reporter.warn(f"{path.name}: {empty_int} empty intervened_generation values")
    reporter.ok(
        f"{path.name}: prompt chars mean={np.mean(prompt_lengths):.1f}, "
        f"baseline chars mean={avg_base:.1f}, intervened chars mean={avg_int:.1f}"
    )


def summarize_token_stats(
    run_dir: Path,
    tag: str,
    expected_prompts: int | None,
    expected_layers: set[int] | None,
    expected_features: set[str] | None,
    reporter: Reporter,
    *,
    sample_columns: list[str] | None = None,
) -> pd.DataFrame | None:
    parquet_path = run_dir / f"{tag}.parquet"
    csv_path = run_dir / f"{tag}.csv"
    path = parquet_path if parquet_path.exists() else csv_path
    if not path.exists():
        reporter.fail(f"{tag}: missing .parquet/.csv")
        return None

    columns = parquet_columns(path) if path.suffix == ".parquet" else list(pd.read_csv(path, nrows=0).columns)
    pred_cols = prediction_columns(columns)
    missing_base = {"tag", "prompt_index", "token_index", "token_id", "logprob", "entropy"} - set(columns)
    if missing_base:
        reporter.fail(f"{path.name}: missing columns {sorted(missing_base)}")

    if expected_layers is not None or expected_features is not None:
        parsed = [parse_prediction_column(col) for col in pred_cols]
        parsed = [item for item in parsed if item is not None]
        got_features = {feature for feature, _ in parsed}
        got_layers = {layer for _, layer in parsed}
        if expected_features is not None and not expected_features <= got_features:
            reporter.fail(f"{path.name}: missing prediction features {sorted(expected_features - got_features)}")
        if expected_layers is not None and not expected_layers <= got_layers:
            reporter.fail(f"{path.name}: missing prediction layers {sorted(expected_layers - got_layers)}")

    rows = parquet_row_count(path) if path.suffix == ".parquet" else sum(1 for _ in path.open("r", encoding="utf-8")) - 1
    if rows <= 0:
        reporter.fail(f"{path.name}: no token rows")
        return None

    read_cols = [c for c in ["prompt_index", "logprob", "entropy"] + (sample_columns or []) if c in columns]
    df = pd.read_parquet(path, columns=read_cols) if path.suffix == ".parquet" else pd.read_csv(path, usecols=read_cols)
    prompt_count = int(df["prompt_index"].nunique()) if "prompt_index" in df.columns else 0
    if expected_prompts is not None and prompt_count != expected_prompts:
        reporter.fail(f"{path.name}: {prompt_count} prompts, expected {expected_prompts}")
    else:
        reporter.ok(f"{path.name}: {rows:,} rows, {prompt_count} prompts, {len(pred_cols)} prediction columns")

    for col in ["logprob", "entropy"]:
        if col not in df.columns:
            continue
        finite = np.isfinite(df[col].to_numpy(dtype=float))
        if not finite.any():
            reporter.warn(f"{path.name}: {col} has no finite values")
        else:
            reporter.ok(f"{path.name}: {col} mean={df[col].mean():.4g}, finite={finite.mean():.1%}")
    return df


def summarize_probe_deltas(run_dir: Path, split: str, features: list[str], layers: list[int], reporter: Reporter) -> None:
    base_path = run_dir / f"token_stats_baseline_{split}.parquet"
    int_path = run_dir / f"token_stats_intervention_{split}.parquet"
    adapter_path = run_dir / f"token_stats_adapter_{split}.parquet"
    if not base_path.exists() or not int_path.exists():
        reporter.warn(f"{split}: cannot summarize deltas; baseline/intervention token stats missing")
        return

    columns = parquet_columns(base_path)
    wanted = [f"{feature}_L{layer}" for feature in features for layer in layers]
    pred_cols = [col for col in wanted if col in columns]
    if not pred_cols:
        reporter.warn(f"{split}: none of requested delta columns were found")
        return

    read_cols = ["prompt_index"] + pred_cols
    base = pd.read_parquet(base_path, columns=read_cols)
    intervention = pd.read_parquet(int_path, columns=read_cols)
    rows = []
    for col in pred_cols:
        rows.append(
            {
                "column": col,
                "baseline_mean": float(base[col].mean()),
                "intervention_mean": float(intervention[col].mean()),
                "delta": float(intervention[col].mean() - base[col].mean()),
            }
        )
    delta_df = pd.DataFrame(rows).sort_values("delta", key=lambda s: s.abs(), ascending=False)
    reporter.ok(f"{split}: largest baseline->intervention probe deltas")
    print(delta_df.to_string(index=False, max_rows=20, float_format=lambda x: f"{x: .5f}"))

    if adapter_path.exists():
        adapter = pd.read_parquet(adapter_path, columns=read_cols)
        rows = []
        for col in pred_cols:
            rows.append(
                {
                    "column": col,
                    "adapter_mean": float(adapter[col].mean()),
                    "adapter_minus_baseline": float(adapter[col].mean() - base[col].mean()),
                    "adapter_minus_intervention": float(adapter[col].mean() - intervention[col].mean()),
                }
            )
        adapter_df = pd.DataFrame(rows).sort_values(
            "adapter_minus_intervention", key=lambda s: s.abs(), ascending=False
        )
        reporter.ok(f"{split}: adapter comparison deltas")
        print(adapter_df.to_string(index=False, max_rows=20, float_format=lambda x: f"{x: .5f}"))


def summarize_feature_deltas(run_dir: Path, split: str, feature: str, reporter: Reporter) -> None:
    base_path = run_dir / f"token_stats_baseline_{split}.parquet"
    int_path = run_dir / f"token_stats_intervention_{split}.parquet"
    adapter_path = run_dir / f"token_stats_adapter_{split}.parquet"
    if not base_path.exists() or not int_path.exists():
        reporter.warn(f"{split}: cannot summarize {feature} deltas; baseline/intervention token stats missing")
        return

    columns = parquet_columns(base_path)
    feature_cols = sorted(
        [col for col in columns if col.startswith(f"{feature}_L")],
        key=lambda col: parse_prediction_column(col)[1] if parse_prediction_column(col) else -1,
    )
    if not feature_cols:
        reporter.warn(f"{split}: no {feature} probe columns found")
        return

    read_cols = ["prompt_index"] + feature_cols
    base = pd.read_parquet(base_path, columns=read_cols)
    intervention = pd.read_parquet(int_path, columns=read_cols)
    rows = []
    adapter = pd.read_parquet(adapter_path, columns=read_cols) if adapter_path.exists() else None
    for col in feature_cols:
        parsed = parse_prediction_column(col)
        row = {
            "layer": parsed[1] if parsed else col,
            "baseline_mean": float(base[col].mean()),
            "intervention_mean": float(intervention[col].mean()),
            "intervention_minus_baseline": float(intervention[col].mean() - base[col].mean()),
        }
        if adapter is not None:
            row["adapter_mean"] = float(adapter[col].mean())
            row["adapter_minus_baseline"] = float(adapter[col].mean() - base[col].mean())
            row["adapter_minus_intervention"] = float(adapter[col].mean() - intervention[col].mean())
        rows.append(row)

    delta_df = pd.DataFrame(rows)
    reporter.ok(f"{split}: {feature} probe deltas by layer")
    print(delta_df.to_string(index=False, max_rows=80, float_format=lambda x: f"{x: .5f}"))


def summarize_activation_cosines(
    run_dir: Path,
    expected_prompts: int | None,
    expected_layers: set[int] | None,
    reporter: Reporter,
    *,
    expect_delta_pair: bool = False,
) -> None:
    parquet_path = run_dir / "activation_cosine_distances.parquet"
    csv_path = run_dir / "activation_cosine_distances.csv"
    path = parquet_path if parquet_path.exists() else csv_path
    if not path.exists():
        reporter.warn("activation_cosine_distances.parquet/csv missing")
        return

    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {
        "prompt_index",
        "pair",
        "layer",
        "tokens",
        "cosine_similarity_mean",
        "cosine_distance_mean",
    }
    missing = required - set(df.columns)
    if missing:
        reporter.fail(f"{path.name}: missing columns {sorted(missing)}")
        return

    if expected_prompts is not None and int(df["prompt_index"].nunique()) != expected_prompts:
        reporter.fail(f"{path.name}: {df['prompt_index'].nunique()} prompts, expected {expected_prompts}")
    if expected_layers is not None and not expected_layers <= set(df["layer"].unique()):
        reporter.fail(f"{path.name}: missing layers {sorted(expected_layers - set(df['layer'].unique()))}")

    bad_range = df[
        (df["cosine_similarity_mean"] < -1.0001)
        | (df["cosine_similarity_mean"] > 1.0001)
        | (df["cosine_distance_mean"] < -0.0001)
        | (df["cosine_distance_mean"] > 2.0001)
    ]
    if not bad_range.empty:
        reporter.fail(f"{path.name}: {len(bad_range)} cosine rows outside expected ranges")

    pairs = set(df["pair"].dropna().astype(str).unique())
    delta_pair = "delta_adapter_vs_delta_intervention"
    if expect_delta_pair and delta_pair not in pairs:
        reporter.warn(f"{path.name}: missing {delta_pair}; rerun activation cosine generation with updated script")

    reporter.ok(
        f"{path.name}: {len(df):,} rows, {df['prompt_index'].nunique()} prompts, "
        f"{df['layer'].nunique()} layers, pairs={sorted(df['pair'].unique())}"
    )
    summary = (
        df.groupby("pair", dropna=False)
        .agg(
            rows=("pair", "size"),
            mean_similarity=("cosine_similarity_mean", "mean"),
            median_similarity=("cosine_similarity_mean", "median"),
            mean_distance=("cosine_distance_mean", "mean"),
            median_distance=("cosine_distance_mean", "median"),
            max_distance=("cosine_distance_max", "max"),
            mean_tokens=("tokens", "mean"),
        )
        .reset_index()
        .sort_values("mean_distance", ascending=False)
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x: .5f}"))

    norm_columns = {
        "source_norm_mean",
        "target_norm_mean",
        "delta_norm_mean",
        "relative_delta_norm_mean",
    }
    if norm_columns <= set(df.columns):
        norm_summary = (
            df.groupby("pair", dropna=False)
            .agg(
                rows=("pair", "size"),
                source_norm_mean=("source_norm_mean", "mean"),
                target_norm_mean=("target_norm_mean", "mean"),
                delta_norm_mean=("delta_norm_mean", "mean"),
                relative_delta_norm_mean=("relative_delta_norm_mean", "mean"),
                relative_delta_norm_median=("relative_delta_norm_mean", "median"),
            )
            .reset_index()
            .sort_values("relative_delta_norm_mean", ascending=False)
        )
        reporter.ok(f"{path.name}: activation norm magnitudes")
        print(norm_summary.to_string(index=False, float_format=lambda x: f"{x: .5f}"))
    else:
        reporter.warn(f"{path.name}: norm magnitude columns missing; rerun activation cosine generation to create them")


def collect_expected_features(metadata: dict[str, Any]) -> set[str]:
    features = {item["label_column"] for item in metadata.get("features", []) if "label_column" in item}
    for spec in metadata.get("args", {}).get("collect_feature_specs", []) or []:
        features.add(str(spec).split(":", 1)[0].split("@", 1)[0])
    return features


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity-check adapter/probe comparison experiment outputs.")
    parser.add_argument("--run-dir", help="Path to a concrete ver_* result directory.")
    parser.add_argument("--slurm-log", help="Slurm output log; the checker will infer the result directory from it.")
    parser.add_argument(
        "--root",
        default="artifacts/experiments/adapter_probe_comparison/experiments",
        help="Root used when --run-dir and --slurm-log are omitted.",
    )
    parser.add_argument("--split", choices=["old", "new", "both"], default="both")
    parser.add_argument(
        "--delta-features",
        default="gender,female,shehe,clout,polite,prosocial,risk,differ,negate,i,we",
        help="Comma-separated features to include in compact delta summaries.",
    )
    parser.add_argument(
        "--delta-layers",
        default="0,8,16,24,31",
        help="Comma-separated probe layers to include in compact delta summaries.",
    )
    args = parser.parse_args()

    reporter = Reporter()
    run_dir = resolve_run_dir(args)
    print(f"run_dir: {run_dir}")
    if not run_dir.exists():
        reporter.fail(f"result directory does not exist: {run_dir}")
        reporter.print()
        return 1

    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        reporter.fail("metadata.json missing")
        reporter.print()
        return 1

    metadata = read_json(metadata_path)
    expected_prompts = metadata.get("max_samples")
    expected_layers = set(metadata.get("layers", []) or [])
    expected_features = collect_expected_features(metadata)
    reporter.ok(
        f"metadata: model={metadata.get('model_name')}, experiment={metadata.get('experiment_name')}, "
        f"max_samples={expected_prompts}, layers={len(expected_layers)}, features={sorted(expected_features)}"
    )

    for path in sorted(run_dir.glob("generations_*.json")):
        summarize_generation_file(path, expected_prompts, reporter)

    sample_cols = [f"{feature}_L{layer}" for feature in sorted(expected_features) for layer in sorted(expected_layers)[:1]]
    for tag_path in sorted(run_dir.glob("token_stats_*.*")):
        tag = tag_path.stem
        summarize_token_stats(
            run_dir,
            tag,
            expected_prompts,
            expected_layers,
            expected_features,
            reporter,
            sample_columns=sample_cols,
        )

    summarize_activation_cosines(
        run_dir,
        expected_prompts,
        expected_layers,
        reporter,
        expect_delta_pair=bool(metadata.get("args", {}).get("adapter_path")),
    )

    delta_features = [item.strip() for item in args.delta_features.split(",") if item.strip()]
    delta_layers = [int(item) for item in args.delta_layers.split(",") if item.strip()]
    splits = ["old", "new"] if args.split == "both" else [args.split]
    for split in splits:
        summarize_probe_deltas(run_dir, split, delta_features, delta_layers, reporter)
        summarize_feature_deltas(run_dir, split, "gender", reporter)

    reporter.print()
    return 1 if reporter.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
