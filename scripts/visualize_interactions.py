#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    print(f"[warn] Missing {path}")
    return None


def save_plot(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def merge_token_stats(baseline: pd.DataFrame, intervention: pd.DataFrame) -> pd.DataFrame:
    """
    Align baseline/intervention token stats on prompt/token indices and token string.
    Keeps per-feature prediction columns by suffixing with _base/_int.
    """
    pred_cols = [c for c in baseline.columns if c.startswith("pred_")]
    merged = baseline.merge(
        intervention,
        on=["prompt_index", "token_index"],
        suffixes=("_base", "_int"),
        how="inner",
    )
    merged["delta_logprob"] = merged["logprob_int"] - merged["logprob_base"]
    merged["delta_entropy"] = merged["entropy_int"] - merged["entropy_base"]

    for col in pred_cols:
        col_base = f"{col}_base"
        col_int = f"{col}_int"
        if col_base in merged.columns and col_int in merged.columns:
            merged[f"delta_{col}"] = merged[col_int] - merged[col_base]
    return merged


def plot_token_delta_hist(df: pd.DataFrame, out_dir: Path, tag: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    df["delta_logprob"].dropna().hist(bins=60, ax=ax)
    ax.set_title(f"Per-token Δlogprob ({tag})")
    ax.set_xlabel("intervention - baseline")
    ax.set_ylabel("count")
    ax.grid(alpha=0.3)
    save_plot(fig, out_dir / f"token_delta_hist_{tag}.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    df["delta_entropy"].dropna().hist(bins=60, ax=ax)
    ax.set_title(f"Per-token Δentropy ({tag})")
    ax.set_xlabel("intervention - baseline")
    ax.set_ylabel("count")
    ax.grid(alpha=0.3)
    save_plot(fig, out_dir / f"token_delta_entropy_{tag}.png")


def plot_prompt_bar(df: pd.DataFrame, out_dir: Path, tag: str) -> None:
    if df.empty:
        return
    prompt_delta = df.groupby("prompt_index")["delta_logprob"].mean().reset_index()
    prompt_delta = prompt_delta.sort_values("prompt_index")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(prompt_delta["prompt_index"], prompt_delta["delta_logprob"], width=0.8)
    ax.set_title(f"Mean Δlogprob per prompt ({tag})")
    ax.set_xlabel("prompt index")
    ax.set_ylabel("mean Δlogprob")
    ax.grid(alpha=0.3)
    save_plot(fig, out_dir / f"prompt_delta_bar_{tag}.png")


def plot_feature_pred_deltas(df: pd.DataFrame, out_dir: Path, tag: str, pred_cols: Sequence[str]) -> None:
    usable = [col for col in pred_cols if f"delta_{col}" in df.columns]
    for col in usable:
        fig, ax = plt.subplots(figsize=(7, 4))
        df[f"delta_{col}"].dropna().hist(bins=50, ax=ax)
        ax.set_title(f"Δ{col} ({tag})")
        ax.set_xlabel("intervention - baseline")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)
        save_plot(fig, out_dir / f"delta_{col}_{tag}.png")


def plot_gradient_cosines(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    df = df.drop_duplicates()
    df["pair"] = df.apply(lambda r: f"{r['feature_a']}|{r['feature_b']}", axis=1)
    pivot = df.pivot_table(index="pair", columns="layer", values="cosine", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Gradient cosine by pair/layer")
    fig.colorbar(im, ax=ax, label="cosine")
    save_plot(fig, out_dir / "gradient_cosines_heatmap.png")

    # Line plot per pair across layers
    fig, ax = plt.subplots(figsize=(8, 4))
    for pair, group in df.groupby("pair"):
        ax.plot(group["layer"], group["cosine"], marker="o", label=pair)
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine")
    ax.set_title("Gradient cosine per feature pair")
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_plot(fig, out_dir / "gradient_cosines_lines.png")


def _safe_sheet_name(name: str) -> str:
    """Excel limits sheet names to 31 characters."""
    return name[:31]


def process_split(
    exp_dir: Path,
    out_dir: Path,
    split: str,
    layers: Sequence[int] | None = None,
    excel_writer: pd.ExcelWriter | None = None,
) -> None:
    layer_list: list[int | None] = list(layers) if layers else [None]
    for layer in layer_list:
        suffix = f"_L{layer}" if layer is not None else ""
        base_path = exp_dir / f"token_stats_baseline_{split}{suffix}.csv"
        int_path = exp_dir / f"token_stats_intervention_{split}{suffix}.csv"
        base_df = load_csv(base_path)
        int_df = load_csv(int_path)
        if base_df is None or int_df is None:
            continue

        merged = merge_token_stats(base_df, int_df)
        merged_path = out_dir / f"token_stats_merged_{split}{suffix}.csv"
        merged.to_csv(merged_path, index=False)

        if excel_writer is not None:
            merged.to_excel(excel_writer, sheet_name=_safe_sheet_name(f"token_{split}{suffix}"), index=False)
            prompt_delta = merged.groupby("prompt_index")["delta_logprob"].mean().reset_index()
            prompt_delta = prompt_delta.sort_values("prompt_index")
            prompt_delta.to_excel(
                excel_writer,
                sheet_name=_safe_sheet_name(f"prompt_delta_{split}{suffix}"),
                index=False,
            )

        tag = f"{split}{suffix}"
        plot_token_delta_hist(merged, out_dir, tag=tag)
        plot_prompt_bar(merged, out_dir, tag=tag)
        pred_cols = [c for c in base_df.columns if c.startswith("pred_")]
        plot_feature_pred_deltas(merged, out_dir, tag=tag, pred_cols=pred_cols)


def plot_logprob_summary(
    exp_dir: Path, out_dir: Path, split: str, excel_writer: pd.ExcelWriter | None = None
) -> None:
    base_path = exp_dir / f"logprob_stats_baseline_{split}.csv"
    int_path = exp_dir / f"logprob_stats_intervention_{split}.csv"
    base_df = load_csv(base_path)
    int_df = load_csv(int_path)
    if base_df is None or int_df is None:
        return

    merged = base_df.merge(int_df, on="prompt_index", suffixes=("_base", "_int"))
    merged["delta_mean_logprob"] = merged["mean_logprob_int"] - merged["mean_logprob_base"]
    merged["delta_mean_entropy"] = merged["mean_entropy_int"] - merged["mean_entropy_base"]
    merged.to_csv(out_dir / f"logprob_stats_merged_{split}.csv", index=False)

    if excel_writer is not None:
        merged.to_excel(excel_writer, sheet_name=_safe_sheet_name(f"logprob_{split}"), index=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(merged["prompt_index"], merged["delta_mean_logprob"], width=0.8)
    ax.set_title(f"Δmean logprob per prompt ({split})")
    ax.set_xlabel("prompt index")
    ax.set_ylabel("Δmean logprob")
    ax.grid(alpha=0.3)
    save_plot(fig, out_dir / f"logprob_prompt_bar_{split}.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    merged["delta_mean_entropy"].hist(bins=50, ax=ax)
    ax.set_title(f"Δmean entropy distribution ({split})")
    ax.set_xlabel("intervention - baseline")
    ax.set_ylabel("count")
    ax.grid(alpha=0.3)
    save_plot(fig, out_dir / f"logprob_entropy_hist_{split}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize probe intervention outputs (new format).")
    parser.add_argument("--experiment-dir", required=True, help="Directory with token/logprob stats and gradient_cosines.csv")
    parser.add_argument("--output-dir", default=None, help="Where to write plots/merged CSVs (default=experiment dir)")
    parser.add_argument(
        "--excel-out",
        default=None,
        help="Optional path to write an Excel workbook containing merged stats (token/prompt/logprob).",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    out_dir = Path(args.output_dir) if args.output_dir else exp_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    excel_writer = pd.ExcelWriter(args.excel_out) if args.excel_out else None

    layers: list[int] = []
    meta_path = exp_dir / "metadata.json"
    if meta_path.exists():
        meta_text = meta_path.read_text()
        (out_dir / "metadata_copy.json").write_text(meta_text)
        try:
            meta = json.loads(meta_text)
            raw_layers = meta.get("layers", [])
            if isinstance(raw_layers, list):
                for l in raw_layers:
                    try:
                        layers.append(int(l))
                    except (TypeError, ValueError):
                        continue
        except Exception as exc:
            print(f"[warn] Could not parse layers from metadata: {exc}")

    for split in ("new", "old"):
        process_split(
            exp_dir,
            out_dir,
            split,
            layers=layers if layers else None,
            excel_writer=excel_writer,
        )
        plot_logprob_summary(exp_dir, out_dir, split, excel_writer=excel_writer)

    grad_path = exp_dir / "gradient_cosines.csv"
    grad_df = load_csv(grad_path)
    if grad_df is not None:
        plot_gradient_cosines(grad_df, out_dir)

    if excel_writer is not None:
        excel_writer.close()
        print(f"Wrote Excel workbook to {args.excel_out}")

    print(f"Saved visualizations to {out_dir}")


if __name__ == "__main__":
    main()
