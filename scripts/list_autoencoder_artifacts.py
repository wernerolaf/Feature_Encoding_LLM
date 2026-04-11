#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from utils import load_json, sanitize_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List autoencoder artifact weight files for a model/layer."
    )
    parser.add_argument("--model-name", required=True, help="Hugging Face model id.")
    parser.add_argument("--layer", type=int, required=True, help="Layer index.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/autoencoders",
        help="Base directory for saved autoencoders.",
    )
    parser.add_argument(
        "--label-filter",
        default=None,
        help="Optional substring filter applied to the label directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.output_dir)
    model_root = base / sanitize_model_name(args.model_name) / f"layer{args.layer}"
    if not model_root.exists():
        return
    artifacts = []
    for metadata_path in model_root.rglob("metadata.json"):
        try:
            metadata = load_json(metadata_path)
        except Exception:
            continue
        if metadata.get("model_name") != args.model_name:
            continue
        if int(metadata.get("layer", args.layer)) != args.layer:
            continue
        weights_path = metadata.get("weights_path")
        if not weights_path:
            continue
        if args.label_filter and args.label_filter not in str(metadata_path.parent):
            continue
        artifacts.append(weights_path)

    for path in sorted(set(artifacts)):
        print(path)


if __name__ == "__main__":
    main()
