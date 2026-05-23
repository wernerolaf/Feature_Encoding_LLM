#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import time
import gc
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_standardizer import ActivationStandardizer
from probes import DecisionTreeProbe, LinearProbe, ShallowNNProbe
from scripts.prompt_utils import PROMPT_TEMPLATE, build_text_series
from utils import (
    build_probe_metadata,
    ensure_dir,
    load_table,
    load_json,
    next_version_dir,
    resolve_device,
    save_json,
    set_seeds,
    sha256_file,
    sanitize_model_name,
)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train probes on language-model activations."
    )
    parser.add_argument("--model-name", required=True, help="Hugging Face model id.")
    parser.add_argument("--data-path", required=True, help="Path to the data file (csv/tsv/xlsx).")
    parser.add_argument("--sheet", default=None, help="Optional worksheet name (xlsx only).")
    parser.add_argument("--label-column", default=None, help="Column containing supervision labels.")
    parser.add_argument(
        "--label-columns",
        nargs="+",
        default=None,
        help="One or more label columns to train probes for (overrides --label-column).",
    )
    parser.add_argument("--layer", type=int, default=-1, help="Transformer block index to extract. Layer k means the output of model.layers[k].")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="One or more transformer block indices to extract (overrides --layer). Layer k means the output of model.layers[k].",
    )
    parser.add_argument("--text-column", default=None, help="Use this column directly if provided.")
    parser.add_argument(
        "--prompt-template",
        default=PROMPT_TEMPLATE,
        help="Optional str.format template used when --text-column is not supplied.",
    )
    parser.add_argument(
        "--template-fields",
        nargs="+",
        default=["gender", "level", "trait", "belief", "question", "type", "pronoun"],
        help="Columns consumed by the prompt template.",
    )
    parser.add_argument(
        "--response-column",
        default="response",
        help="Column appended after the prompt (if available).",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of rows used.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for forward passes.")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer max_length.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device spec or 'auto' (uses CUDA when available).",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_MAP),
        default="float32",
        help="Model compute dtype.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out fraction for evaluation.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Hold-out fraction for validation (from train).")
    parser.add_argument("--random-state", type=int, default=0, help="Seed for splitting.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--standardizer",
        choices=["identity", "standard", "autoencoder"],
        default="standard",
        help="Feature-normalisation strategy.",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "auto"],
        default="auto",
        help="Training objective. 'auto' picks regression for float labels; classification otherwise.",
    )
    parser.add_argument("--autoencoder-artifact", help="Path to a saved autoencoder (required when strategy=autoencoder).")
    parser.add_argument(
        "--autoencoder-map",
        action="append",
        default=None,
        help="Layer-to-autoencoder mapping in the form layer:path (repeatable).",
    )
    parser.add_argument(
        "--autoencoder-root",
        default="artifacts/autoencoders",
        help="Root directory for autoencoders when auto-resolving per layer.",
    )
    parser.add_argument(
        "--autoencoder-label-filter",
        default=None,
        help="Label directory name or substring to select autoencoder artifacts.",
    )
    parser.add_argument(
        "--autoencoder-version",
        default="latest",
        help="Version to use when resolving autoencoders (default: latest).",
    )
    parser.add_argument("--no-standard-mean", action="store_true", help="Disable mean centering when using the StandardScaler.")
    parser.add_argument("--no-standard-std", action="store_true", help="Disable variance scaling when using the StandardScaler.")
    parser.add_argument(
        "--probe-type",
        choices=["linear", "decision_tree", "shallow_nn"],
        default="linear",
        help="Probe architecture to train.",
    )
    parser.add_argument(
        "--logistic-penalty",
        choices=["l1", "l2", "elasticnet"],
        default="l1",
        help="LogisticRegression penalty (l1/l2/elasticnet).",
    )
    parser.add_argument("--logistic-C", type=float, default=None, help="Override LogisticRegression C.")
    parser.add_argument(
        "--logistic-l1-ratio",
        type=float,
        default=0.5,
        help="Elastic-net mixing parameter (only used when penalty=elasticnet).",
    )
    parser.add_argument("--logistic-max-iter", type=int, default=None, help="Override LogisticRegression max_iter.")
    parser.add_argument("--tree-max-depth", type=int, default=None, help="Decision tree maximum depth.")
    parser.add_argument("--tree-random-state", type=int, default=0, help="Tree probe random state.")
    parser.add_argument("--nn-hidden-dim", type=int, default=128, help="Hidden width for the shallow NN probe.")
    parser.add_argument("--nn-dropout", type=float, default=0.0, help="Dropout probability for the shallow NN probe.")
    parser.add_argument("--nn-epochs", type=int, default=50, help="Epochs for the shallow NN probe.")
    parser.add_argument("--nn-batch-size", type=int, default=64, help="Batch size for the shallow NN probe.")
    parser.add_argument("--nn-lr", type=float, default=1e-3, help="Learning rate for the shallow NN probe.")
    parser.add_argument("--nn-weight-decay", type=float, default=0.0, help="Weight decay for the shallow NN probe.")
    parser.add_argument(
        "--probe-log-dir",
        default=None,
        help="Directory to store TensorBoard logs for probe training (shallow_nn only).",
    )
    parser.add_argument(
        "--probe-log-interval",
        type=int,
        default=10,
        help="Log every N batches for probe training when --probe-log-dir is set.",
    )
    parser.add_argument(
        "--probe-track-history",
        action="store_true",
        help="Keep per-batch loss history of the probe in memory (useful for CSV export).",
    )
    parser.add_argument(
        "--probe-history-dir",
        default=None,
        help="If provided, save probe loss history CSVs here (shallow_nn only).",
    )
    parser.add_argument(
        "--save-path",
        help=(
            "Path to persist the trained probe artifact. "
            "Defaults to artifacts/probes/<model>_layer<idx>_<probe>.pkl."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts/probes",
        help="Base directory for saving probes when --save-path is not provided.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Optional explicit version number for the artifact. Defaults to the next available version.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving the trained probe artifact.",
    )
    parser.add_argument(
        "--tqdm",
        action="store_true",
        help="Enable tqdm progress bars during activation collection.",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bars during activation collection.",
    )
    return parser.parse_args()


def detect_task(label_series: pd.Series, user_choice: str) -> str:
    """
    Decide whether to run a classification or regression probe.
    - If the user specifies classification/regression, honour it.
    - If auto: treat float dtypes as regression; otherwise default to classification.
    """
    if user_choice in {"classification", "regression"}:
        return user_choice

    if pd.api.types.is_float_dtype(label_series):
        return "regression"
    return "classification"


def prepare_dataset(
    df: pd.DataFrame,
    args: argparse.Namespace,
    label_columns: list[str],
) -> tuple[list[str], dict[str, pd.Series]]:
    if "answered" in label_columns and "answered" not in df.columns:
        response_col = args.response_column
        if response_col in df.columns:
            answered = df[response_col].fillna("").astype(str).str.strip().ne("")
            df = df.copy()
            df["answered"] = answered.astype(int)
    text_series, filtered_df = build_text_series(
        df,
        text_column=args.text_column,
        prompt_template=args.prompt_template,
        template_fields=args.template_fields,
        response_column=args.response_column,
        prompt_response_sep="",
    )
    missing = [col for col in label_columns if col not in filtered_df.columns]
    if missing:
        raise ValueError(f"Missing label columns in data: {missing}")

    label_series_map: dict[str, pd.Series] = {}
    mask = pd.Series(True, index=filtered_df.index)
    for col in label_columns:
        series = filtered_df[col]
        label_series_map[col] = series
        mask &= series.notna()

    text_series = text_series.loc[mask]
    for col in label_columns:
        label_series_map[col] = label_series_map[col].loc[mask]

    if args.max_samples is not None:
        text_series = text_series.iloc[: args.max_samples]
        for col in label_columns:
            label_series_map[col] = label_series_map[col].iloc[: args.max_samples]

    return text_series.tolist(), label_series_map


def collect_layer_activations(
    texts: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: list[int],
    batch_size: int,
    device: torch.device,
    max_length: int,
    args_tqdm: bool,
) -> tuple[dict[int, np.ndarray], list[int]]:
    """
    Collect token activations using the same layer convention as feature_interaction.py.

    Requested layer k means the output of transformer block model.layers[k].
    Hugging Face output_hidden_states includes the embedding state at index 0,
    so block k output is hidden_states[k + 1]. This avoids training probes on
    the embedding stream or the previous block by accident.
    """
    if not texts:
        raise ValueError("No texts supplied for activation extraction.")

    ordered_layers = list(dict.fromkeys(layers))
    if not ordered_layers:
        raise ValueError("No layers requested.")
    buckets: dict[int, list[torch.Tensor]] = {layer: [] for layer in ordered_layers}
    resolved_layers: dict[int, int] | None = None
    token_counts: list[int] = []

    model.to(device)
    model.eval()

    indices = range(0, len(texts), batch_size)
    batch_iter = tqdm(indices, desc="Collect activations", disable=not args_tqdm)
    for start in batch_iter:
        batch_texts = texts[start: start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        if resolved_layers is None:
            total = len(hidden_states)
            resolved_layers = {}
            for layer in ordered_layers:
                resolved_idx = layer + 1 if layer >= 0 else total + layer
                if resolved_idx < 0 or resolved_idx >= total:
                    raise ValueError(
                        f"Layer {layer} resolves to hidden_states[{resolved_idx}], but only {total} hidden states are available."
                    )
                resolved_layers[layer] = resolved_idx
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            raise ValueError("Tokenizer did not return an attention_mask.")

        valid_tokens = attention_mask.bool()               # [batch, seq]
        token_counts.extend(valid_tokens.sum(dim=1).cpu().tolist())
        for layer in ordered_layers:
            idx = resolved_layers[layer]
            layer_states = hidden_states[idx]                 # [batch, seq, hidden], output of model.layers[layer]
            token_activations = layer_states[valid_tokens]    # [batch*seq_valid, hidden]
            buckets[layer].append(token_activations.float().cpu())

    return {layer: torch.cat(chunks, dim=0).numpy() for layer, chunks in buckets.items()}, token_counts


def make_standardizer(args: argparse.Namespace, device: torch.device) -> ActivationStandardizer:
    if args.standardizer == "identity":
        return ActivationStandardizer(strategy="identity", device=device)

    if args.standardizer == "standard":
        scaler_kwargs = dict(
            with_mean=not args.no_standard_mean,
            with_std=not args.no_standard_std,
        )
        return ActivationStandardizer(strategy="standard", scaler_kwargs=scaler_kwargs, device=device)

    if not args.autoencoder_artifact:
        raise ValueError("--autoencoder-artifact is required when strategy=autoencoder.")
    standardizer = ActivationStandardizer(strategy="autoencoder", device=device)
    standardizer.load_autoencoder_artifact(args.autoencoder_artifact, map_location=device)
    return standardizer


def parse_autoencoder_map(entries: list[str] | None) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if not entries:
        return mapping
    for item in entries:
        if ":" not in item:
            raise ValueError(f"Invalid --autoencoder-map entry '{item}'. Expected layer:path.")
        layer_str, path = item.split(":", 1)
        try:
            layer = int(layer_str)
        except ValueError as exc:
            raise ValueError(f"Invalid layer in --autoencoder-map: '{layer_str}'") from exc
        mapping[layer] = path
    return mapping


def resolve_autoencoder_paths(
    *,
    model_name: str,
    layers: list[int],
    root: str,
    label_filter: str | None,
    version: str,
) -> dict[int, str]:
    model_dir = Path(root) / sanitize_model_name(model_name)
    resolved: dict[int, str] = {}
    for layer in layers:
        layer_dir = model_dir / f"layer{layer}"
        if not layer_dir.exists():
            raise FileNotFoundError(f"Autoencoder layer directory not found: {layer_dir}")
        label_dirs = [p for p in layer_dir.iterdir() if p.is_dir()]
        if label_filter:
            label_dirs = [p for p in label_dirs if label_filter in p.name]
        if not label_dirs:
            raise FileNotFoundError(
                f"No autoencoder labels found for layer {layer} under {layer_dir} "
                f"with filter '{label_filter}'."
            )
        if len(label_dirs) > 1:
            names = ", ".join(p.name for p in label_dirs)
            raise ValueError(
                f"Multiple autoencoder labels match for layer {layer}: {names}. "
                "Refine --autoencoder-label-filter."
            )
        label_dir = label_dirs[0]
        if version == "latest":
            version_dirs = sorted(
                [p for p in label_dir.iterdir() if p.is_dir() and p.name.startswith("ver_")],
                key=lambda p: int(p.name.split("_")[1]),
            )
            if not version_dirs:
                raise FileNotFoundError(f"No version dirs found under {label_dir}")
            version_dir = version_dirs[-1]
        else:
            version_dir = label_dir / f"ver_{version}"
            if not version_dir.exists():
                raise FileNotFoundError(f"Version dir not found: {version_dir}")
        metadata_path = version_dir / "metadata.json"
        if metadata_path.exists():
            metadata = load_json(metadata_path)
            weights_path = metadata.get("weights_path")
            if weights_path:
                resolved[layer] = str(weights_path)
                continue
        # Fallback: find weights file in version_dir
        weights_files = sorted(version_dir.glob("*.pt"))
        if not weights_files:
            raise FileNotFoundError(f"No weights found in {version_dir}")
        resolved[layer] = str(weights_files[0])
    return resolved


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    if y_proba is not None:
        try:
            if y_proba.ndim == 1:
                metrics["auc"] = float(roc_auc_score(y_true, y_proba))
            elif y_proba.shape[1] == 2:
                metrics["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        except Exception:
            pass
    return metrics


def count_nonzero_weights(model: object, threshold: float = 1e-8) -> tuple[int, int]:
    coef = getattr(model, "coef_", None)
    if coef is None:
        return 0, 0
    arr = np.asarray(coef)
    total = arr.size
    nonzero = int(np.count_nonzero(np.abs(arr) > threshold))
    return nonzero, total


def make_probe(
    args: argparse.Namespace,
    standardizer: ActivationStandardizer,
    device: torch.device,
    *,
    task: str,
):
    probe_log_dir = None
    if args.probe_log_dir:
        probe_log_dir = Path(args.probe_log_dir) / (
            f"{sanitize_model_name(args.model_name)}_layer{args.layer}_{args.probe_type}"
        )

    track_history = args.probe_track_history or args.probe_history_dir is not None
    log_interval = max(1, args.probe_log_interval)

    if args.probe_type == "linear":
        logistic_kwargs: dict[str, object] = {}
        logistic_kwargs["penalty"] = args.logistic_penalty
        if args.logistic_C is not None:
            logistic_kwargs["C"] = args.logistic_C
        if args.logistic_max_iter is not None:
            logistic_kwargs["max_iter"] = args.logistic_max_iter
        if args.logistic_penalty in {"l1", "elasticnet"}:
            logistic_kwargs["solver"] = "saga"
        if args.logistic_penalty == "elasticnet":
            logistic_kwargs["l1_ratio"] = args.logistic_l1_ratio
        return LinearProbe(
            standardizer=standardizer,
            logistic_kwargs=logistic_kwargs or None,
            task=task,
        )

    if args.probe_type == "decision_tree":
        tree_kwargs: dict[str, object] = {"random_state": args.tree_random_state}
        if args.tree_max_depth is not None:
            tree_kwargs["max_depth"] = args.tree_max_depth
        return DecisionTreeProbe(standardizer=standardizer, tree_kwargs=tree_kwargs, task=task)

    return ShallowNNProbe(
        standardizer=standardizer,
        hidden_dim=args.nn_hidden_dim,
        dropout=args.nn_dropout,
        epochs=args.nn_epochs,
        batch_size=args.nn_batch_size,
        lr=args.nn_lr,
        weight_decay=args.nn_weight_decay,
        device=device,
        log_dir=str(probe_log_dir) if probe_log_dir is not None else None,
        log_interval=log_interval,
        track_history=track_history,
        task=task,
    )


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    device = resolve_device(args.device)
    if args.label_column is not None:
        args.label_column = args.label_column.lower()
    if args.text_column is not None:
        args.text_column = args.text_column.lower()
    if args.response_column is not None:
        args.response_column = args.response_column.lower()
    if args.template_fields:
        args.template_fields = [f.lower() for f in args.template_fields]
    args_tqdm = False if args.no_tqdm else True

    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    log(f"Loading tokenizer/model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=DTYPE_MAP[args.dtype],
    )
    log("Model loaded.")

    df = load_table(args.data_path, sheet=args.sheet)
    df.columns = [str(c).lower() for c in df.columns]
    data_hash = sha256_file(args.data_path)
    log(f"Loaded data: rows={len(df)} cols={len(df.columns)} hash={data_hash[:8]}")
    if args.label_columns:
        label_columns = [c.lower() for c in args.label_columns]
    elif args.label_column:
        label_columns = [args.label_column]
    else:
        raise ValueError("Provide --label-column or --label-columns.")
    if args.save_path and len(label_columns) > 1:
        raise ValueError("--save-path cannot be used with multiple label columns.")
    autoencoder_map = parse_autoencoder_map(args.autoencoder_map)
    if args.autoencoder_artifact and args.layers and len(args.layers) > 1 and not autoencoder_map:
        raise ValueError("--autoencoder-artifact does not support multiple --layers.")
    autoencoder_paths: dict[int, str] = {}

    texts, label_series_map = prepare_dataset(df, args, label_columns=label_columns)
    log(f"Prepared dataset: texts={len(texts)} labels={label_columns}")

    first_label = label_columns[0]
    task_for_split = detect_task(label_series_map[first_label], args.task)
    if task_for_split == "classification":
        label_encoder_for_split = LabelEncoder()
        y_rows_for_split = label_encoder_for_split.fit_transform(
            label_series_map[first_label].to_numpy()
        )
    else:
        y_rows_for_split = label_series_map[first_label].to_numpy(dtype=np.float32)

    texts_arr = np.asarray(texts, dtype=object)
    row_indices = np.arange(texts_arr.shape[0])
    stratify_rows = y_rows_for_split if task_for_split == "classification" else None
    train_rows, test_rows = train_test_split(
        row_indices,
        test_size=args.test_size,
        stratify=stratify_rows,
        random_state=args.random_state,
    )

    val_rows: np.ndarray | None = None
    if args.val_size and args.val_size > 0:
        stratify_train = y_rows_for_split[train_rows] if task_for_split == "classification" else None
        train_rows, val_rows = train_test_split(
            train_rows,
            test_size=args.val_size,
            stratify=stratify_train,
            random_state=args.random_state,
        )

    train_texts = texts_arr[train_rows].tolist()
    val_texts = texts_arr[val_rows].tolist() if val_rows is not None else None
    test_texts = texts_arr[test_rows].tolist()

    layers = args.layers if args.layers else [args.layer]

    log(f"Collecting activations for layers={layers} train/val/test")
    X_train_by_layer, train_token_counts = collect_layer_activations(
        texts=train_texts,
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        batch_size=args.batch_size,
        device=device,
        max_length=args.max_length,
        args_tqdm=args_tqdm,
    )
    X_val_by_layer: dict[int, np.ndarray] | None = None
    val_token_counts: list[int] | None = None
    if val_texts is not None:
        X_val_by_layer, val_token_counts = collect_layer_activations(
            texts=val_texts,
            model=model,
            tokenizer=tokenizer,
            layers=layers,
            batch_size=args.batch_size,
            device=device,
            max_length=args.max_length,
            args_tqdm=args_tqdm,
        )
    X_test_by_layer, test_token_counts = collect_layer_activations(
        texts=test_texts,
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        batch_size=args.batch_size,
        device=device,
        max_length=args.max_length,
        args_tqdm=args_tqdm,
    )

    # Free the LLM before probe training to save GPU memory.
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    torch.cuda.empty_cache()

    if args.standardizer == "autoencoder":
        if autoencoder_map:
            autoencoder_paths = dict(autoencoder_map)
        elif args.autoencoder_artifact and len(layers) == 1:
            autoencoder_paths = {layers[0]: args.autoencoder_artifact}
        else:
            if not args.autoencoder_label_filter:
                raise ValueError(
                    "Provide --autoencoder-label-filter (or --autoencoder-map) when "
                    "auto-resolving autoencoders for multiple layers."
                )
            autoencoder_paths = resolve_autoencoder_paths(
                model_name=args.model_name,
                layers=layers,
                root=args.autoencoder_root,
                label_filter=args.autoencoder_label_filter,
                version=str(args.autoencoder_version),
            )
        missing_layers = [l for l in layers if l not in autoencoder_paths]
        if missing_layers:
            raise ValueError(f"Autoencoder map missing layers: {missing_layers}")

    for layer in layers:
        args.layer = layer
        X_train = X_train_by_layer[layer]
        X_val = X_val_by_layer[layer] if X_val_by_layer is not None else None
        X_test = X_test_by_layer[layer]
        if args.standardizer == "autoencoder":
            args.autoencoder_artifact = autoencoder_paths[layer]
        log(f"Layer {layer}: X_train={X_train.shape} X_val={None if X_val is None else X_val.shape} X_test={X_test.shape}")

        for label_column in label_columns:
            log(f"Start label={label_column} layer={layer}")
            label_series = label_series_map[label_column]
            task = detect_task(label_series, args.task)
            label_encoder: LabelEncoder | None = None
            if task == "classification":
                label_encoder = LabelEncoder()
                y_rows = label_encoder.fit_transform(label_series.to_numpy())
            else:
                y_rows = label_series.to_numpy(dtype=np.float32)

            y_train = np.repeat(y_rows[train_rows], train_token_counts)
            y_val = (
                np.repeat(y_rows[val_rows], val_token_counts)
                if val_rows is not None and val_token_counts is not None
                else None
            )
            y_test = np.repeat(y_rows[test_rows], test_token_counts)

            metrics: dict[str, object] = {}

            if task == "classification":
                standardizer = make_standardizer(args, device=device)
                probe = make_probe(args, standardizer=standardizer, device=device, task=task)
                if y_val is not None:
                    X_train_full = np.concatenate([X_train, X_val], axis=0)
                    y_train_full = np.concatenate([y_train, y_val], axis=0)
                else:
                    X_train_full = X_train
                    y_train_full = y_train

                probe.fit(X_train_full, y_train_full)
                log(f"Label {label_column} layer {layer}: fit complete")
                train_pred = probe.predict(X_train_full)
                test_pred = probe.predict(X_test)
                train_proba = None
                test_proba = None
                try:
                    train_proba = probe.predict_proba(X_train_full)
                    test_proba = probe.predict_proba(X_test)
                except Exception:
                    pass

                train_metrics = compute_classification_metrics(y_train_full, train_pred, train_proba)
                test_metrics = compute_classification_metrics(y_test, test_pred, test_proba)

                print(f"[{label_column}] Train accuracy: {train_metrics['accuracy']:.4f}")
                print(f"[{label_column}] Test accuracy:  {test_metrics['accuracy']:.4f}")
                if "auc" in test_metrics:
                    print(f"[{label_column}] Test AUC:       {test_metrics['auc']:.4f}")
                print(f"[{label_column}] Test F1 (macro): {test_metrics['f1_macro']:.4f}")

                metrics.update(
                    {
                        "train_accuracy": train_metrics["accuracy"],
                        "test_accuracy": test_metrics["accuracy"],
                        "train_f1_macro": train_metrics["f1_macro"],
                        "test_f1_macro": test_metrics["f1_macro"],
                        "train_f1_weighted": train_metrics["f1_weighted"],
                        "test_f1_weighted": test_metrics["f1_weighted"],
                    }
                )
                if "auc" in train_metrics:
                    metrics["train_auc"] = train_metrics["auc"]
                if "auc" in test_metrics:
                    metrics["test_auc"] = test_metrics["auc"]

                nonzero, total = count_nonzero_weights(getattr(probe, "model", None))
                if total:
                    metrics["probe_weight_nonzero"] = nonzero
                    metrics["probe_weight_total"] = total
                    metrics["probe_weight_sparsity"] = float(1.0 - (nonzero / total))

                if label_encoder is not None:
                    class_labels = np.arange(len(label_encoder.classes_))
                    class_report = classification_report(
                        y_test,
                        test_pred,
                        labels=class_labels,
                        target_names=label_encoder.classes_,
                        output_dict=True,
                        zero_division=0,
                    )
                    metrics["classification_report"] = class_report
                    print(
                        classification_report(
                            y_test,
                            test_pred,
                            labels=class_labels,
                            target_names=label_encoder.classes_,
                            zero_division=0,
                        )
                    )

            else:
                standardizer = make_standardizer(args, device=device)
                probe = make_probe(args, standardizer=standardizer, device=device, task=task)
                if y_val is not None:
                    X_train_full = np.concatenate([X_train, X_val], axis=0)
                    y_train_full = np.concatenate([y_train, y_val], axis=0)
                else:
                    X_train_full = X_train
                    y_train_full = y_train
                probe.fit(X_train_full, y_train_full)
                log(f"Label {label_column} layer {layer}: fit complete")
                train_pred = probe.predict(X_train_full)
                test_pred = probe.predict(X_test)
                train_mse = mean_squared_error(y_train_full, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                train_r2 = r2_score(y_train_full, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                print(f"[{label_column}] Train MSE: {train_mse:.4f}  R2: {train_r2:.4f}")
                print(f"[{label_column}] Test  MSE: {test_mse:.4f}  R2: {test_r2:.4f}")
                metrics.update(
                    {
                        "train_mse": float(train_mse),
                        "test_mse": float(test_mse),
                        "train_r2": float(train_r2),
                        "test_r2": float(test_r2),
                    }
                )

            if args.probe_history_dir:
                history_dir = Path(args.probe_history_dir)
                if hasattr(probe, "get_history"):
                    history = probe.get_history()  # type: ignore[attr-defined]
                    if history:
                        history_dir.mkdir(parents=True, exist_ok=True)
                        history_path = history_dir / (
                            f"{sanitize_model_name(args.model_name)}_layer{args.layer}_{label_column}_{args.probe_type}_loss.csv"
                        )
                        pd.DataFrame(history).to_csv(history_path, index=False)
                        print(f"Saved probe loss history to {history_path}")
                    else:
                        print("Probe did not record loss history; nothing saved.")
                else:
                    print(
                        f"Probe type '{args.probe_type}' does not expose a loss history; skipping CSV export."
                    )

            artifact_dir: Path | None = None
            artifact_path: Path | None = None
            if not args.no_save:
                if args.save_path:
                    artifact_path = Path(args.save_path)
                    artifact_dir = artifact_path.parent
                else:
                    artifact_dir = next_version_dir(
                        kind="probes",
                        model_name=args.model_name,
                        layer=args.layer,
                        label=label_column,
                        base_dir=args.artifact_root,
                        version_override=args.version,
                    )
                    artifact_path = artifact_dir / "probe.pkl"

            if artifact_path and artifact_dir:
                ensure_dir(artifact_dir)

                model_attr = getattr(probe, "model", None)
                if model_attr is not None and hasattr(model_attr, "cpu"):
                    model_attr.cpu()

                standardizer_metadata = None
                standardizer_ref = getattr(probe, "standardizer", None)

                if standardizer_ref is not None:
                    if hasattr(standardizer_ref, "_autoencoder") and standardizer_ref._autoencoder is not None:
                        standardizer_ref._autoencoder.cpu()

                    standardizer_metadata = {
                        "strategy": standardizer_ref.strategy,
                        "autoencoder_artifact": args.autoencoder_artifact,
                    }

                    if standardizer_ref.strategy == "standard" and getattr(standardizer_ref, "_scaler", None) is not None:
                        standardizer_metadata["scaler_state"] = standardizer_ref._scaler

                try:
                    probe.standardizer = None

                    artifact = {
                        "probe_type": args.probe_type,
                        "model_name": args.model_name,
                        "layer": args.layer,
                        "standardizer": standardizer_metadata,
                        "label_encoder": label_encoder,
                        "classes": label_encoder.classes_ if label_encoder is not None else None,
                        "probe": probe,
                        "target_feature": label_column,
                        "task": task,
                    }
                    joblib.dump(artifact, artifact_path)
                    metrics_path = artifact_dir / "metrics.json"
                    save_json(metrics, metrics_path)

                    classes = label_encoder.classes_.tolist() if label_encoder is not None else None
                    metadata = build_probe_metadata(
                        args=args,
                        layer=args.layer,
                        label_column=label_column,
                        task=task,
                        standardizer_strategy=standardizer_metadata["strategy"] if standardizer_metadata else "identity",
                        autoencoder_artifact=args.autoencoder_artifact,
                        artifact_path=artifact_path,
                        data_hash=data_hash,
                        metrics=metrics,
                        classes=classes,
                    )
                    save_json(metadata, artifact_dir / "metadata.json")
                    print(f"Saved probe artifact to {artifact_path}")
                    log(f"Saved probe artifact to {artifact_path}")
                finally:
                    probe.standardizer = standardizer_ref

            # Free probe-related memory between labels.
            del probe
            del standardizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
