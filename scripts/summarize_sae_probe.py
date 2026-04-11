#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize SAE autoencoder + probe performance into a table."
    )
    parser.add_argument(
        "--autoencoder-root",
        default="artifacts/autoencoders",
        help="Root directory containing autoencoder artifacts.",
    )
    parser.add_argument(
        "--probe-root",
        default="artifacts/probes",
        help="Root directory containing probe artifacts.",
    )
    parser.add_argument("--model-name", default=None, help="Filter by model name.")
    parser.add_argument("--layer", type=int, default=None, help="Filter by layer.")
    parser.add_argument("--label-filter", default=None, help="Filter by label column.")
    parser.add_argument("--output-csv", default=None, help="Write summary CSV to this path.")
    parser.add_argument("--output-md", default=None, help="Write summary markdown to this path.")
    return parser.parse_args()


def load_autoencoder_metadata(root: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for metadata_path in root.rglob("metadata.json"):
        try:
            metadata = load_json(metadata_path)
        except Exception:
            continue
        weights_path = metadata.get("weights_path")
        if not weights_path:
            continue
        mapping[str(weights_path)] = metadata
    return mapping


def load_probe_metadata(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metadata_path in root.rglob("metadata.json"):
        try:
            metadata = load_json(metadata_path)
        except Exception:
            continue
        if metadata.get("artifact_type") != "probe":
            continue
        rows.append(metadata)
    return rows


def select_score(row: dict[str, Any]) -> tuple[float, str]:
    if row.get("test_auc") is not None:
        return float(row["test_auc"]), "test_auc"
    if row.get("test_f1_macro") is not None:
        return float(row["test_f1_macro"]), "test_f1_macro"
    if row.get("test_accuracy") is not None:
        return float(row["test_accuracy"]), "test_accuracy"
    return -1.0, "none"


def main() -> None:
    args = parse_args()
    auto_root = Path(args.autoencoder_root)
    probe_root = Path(args.probe_root)

    ae_meta = load_autoencoder_metadata(auto_root)
    probe_rows = load_probe_metadata(probe_root)

    records: list[dict[str, Any]] = []
    for row in probe_rows:
        if args.model_name and row.get("model_name") != args.model_name:
            continue
        if args.layer is not None and int(row.get("layer", -999)) != args.layer:
            continue
        if args.label_filter and row.get("label_column") != args.label_filter:
            continue

        metrics = row.get("metrics", {}) or {}
        auto_artifact = row.get("autoencoder_artifact")
        ae_info = ae_meta.get(str(auto_artifact)) if auto_artifact else None
        ae_metrics = (ae_info or {}).get("metrics", {}) or {}

        record = {
            "model": row.get("model_name"),
            "layer": row.get("layer"),
            "label": row.get("label_column"),
            "standardizer": row.get("standardizer"),
            "latent_dim": (ae_info or {}).get("latent_dim"),
            "beta": (ae_info or {}).get("beta"),
            "lr": (ae_info or {}).get("lr"),
            "input_norm": (ae_info or {}).get("input_norm"),
            "avg_l0": ae_metrics.get("avg_l0"),
            "reconstruction_mse": ae_metrics.get("reconstruction_mse"),
            "dead_latent_fraction": ae_metrics.get("dead_latent_fraction"),
            "test_accuracy": metrics.get("test_accuracy"),
            "test_f1_macro": metrics.get("test_f1_macro"),
            "test_auc": metrics.get("test_auc"),
            "probe_weight_sparsity": metrics.get("probe_weight_sparsity"),
            "autoencoder_artifact": auto_artifact,
        }
        records.append(record)

    if not records:
        print("No probe metadata found for the given filters.")
        return

    df = pd.DataFrame(records)
    df = df.sort_values(["label", "layer", "standardizer", "latent_dim"], na_position="last")

    print(df.to_string(index=False))

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Wrote CSV to {args.output_csv}")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        md_text = df.to_markdown(index=False)
        Path(args.output_md).write_text(md_text, encoding="utf-8")
        print(f"Wrote markdown to {args.output_md}")

    # Highlight best SAE vs baseline per label/layer
    grouped = df.groupby(["label", "layer"])
    for (label, layer), group in grouped:
        baseline = group[group["standardizer"].eq("identity")]
        sae = group[group["standardizer"].eq("autoencoder")]
        if baseline.empty or sae.empty:
            continue
        base_score, base_metric = select_score(baseline.iloc[0].to_dict())
        best_sae_row = max((r.to_dict() for _, r in sae.iterrows()), key=lambda r: select_score(r)[0])
        best_sae_score, best_metric = select_score(best_sae_row)
        print(
            f"[{label} layer {layer}] baseline {base_metric}={base_score:.4f} | "
            f"best SAE {best_metric}={best_sae_score:.4f} "
            f"(latent={best_sae_row.get('latent_dim')}, beta={best_sae_row.get('beta')})"
        )


if __name__ == "__main__":
    main()
