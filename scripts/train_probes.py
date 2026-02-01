#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_standardizer import ActivationStandardizer
from probes import DecisionTreeProbe, LinearProbe, ShallowNNProbe
from utils import (
    build_probe_metadata,
    ensure_dir,
    load_table,
    next_version_dir,
    resolve_device,
    save_json,
    set_seeds,
    sha256_file,
    sanitize_model_name,
)

PROMPT_TEMPLATE = (
    "you want to convince your {gender} interlocutor with a {level} level of {trait}, "
    'and answer "{belief}" to the question: "{question}". '
    "Use {type} arguments to change {pronoun}'s mind.\n"
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
    parser.add_argument("--data-path", required=True, help="Path to the Excel workbook.")
    parser.add_argument("--sheet", required=True, help="Worksheet name to read.")
    parser.add_argument("--label-column", required=True, help="Column containing supervision labels.")
    parser.add_argument("--layer", type=int, default=-1, help="Hidden-state index to extract.")
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
    parser.add_argument("--no-standard-mean", action="store_true", help="Disable mean centering when using the StandardScaler.")
    parser.add_argument("--no-standard-std", action="store_true", help="Disable variance scaling when using the StandardScaler.")
    parser.add_argument(
        "--probe-type",
        choices=["linear", "decision_tree", "shallow_nn"],
        default="linear",
        help="Probe architecture to train.",
    )
    parser.add_argument("--logistic-penalty", choices=["l1", "l2"], default=None, help="Override LogisticRegression penalty.")
    parser.add_argument("--logistic-C", type=float, default=None, help="Override LogisticRegression C.")
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


def build_text_series(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.Series, pd.DataFrame]:
    if args.text_column and args.text_column in df.columns:
        series = df[args.text_column]
    else:
        if not args.prompt_template:
            raise ValueError("Provide --text-column or --prompt-template")

        def render(row: pd.Series) -> str:
            values = {field: row.get(field, "") for field in args.template_fields}
            return args.prompt_template.format_map(values)

        prompts = df.apply(render, axis=1)
        if args.response_column and args.response_column in df.columns:
            responses = df[args.response_column].fillna("").astype(str)
            series = prompts + responses
        else:
            series = prompts

    series = series.fillna("").astype(str).str.strip()
    mask = series != ""
    return series.loc[mask], df.loc[mask].copy()


def prepare_dataset(df: pd.DataFrame, args: argparse.Namespace) -> tuple[list[str], np.ndarray]:
    text_series, filtered_df = build_text_series(df, args)
    if args.label_column not in filtered_df.columns:
        raise ValueError(f"Column '{args.label_column}' not found in the sheet.")

    label_series = filtered_df[args.label_column]
    mask = label_series.notna()
    text_series = text_series.loc[mask]
    label_series = label_series.loc[mask]

    if args.max_samples is not None:
        text_series = text_series.iloc[: args.max_samples]
        label_series = label_series.iloc[: args.max_samples]

    return text_series.tolist(), label_series


def collect_layer_activations(
    texts: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer: int,
    batch_size: int,
    device: torch.device,
    max_length: int,
) -> np.ndarray:
    if not texts:
        raise ValueError("No texts supplied for activation extraction.")

    buckets: list[torch.Tensor] = []
    resolved_idx: int | None = None
    token_counts: list[int] = []

    model.to(device)
    model.eval()

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
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

        if resolved_idx is None:
            total = len(hidden_states)
            resolved_idx = layer if layer >= 0 else total + layer
            if resolved_idx < 0 or resolved_idx >= total:
                raise ValueError(
                    f"Layer {layer} resolves to {resolved_idx}, but only {total} hidden states are available."
                )

        layer_states = hidden_states[resolved_idx]         # [batch, seq, hidden]
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            raise ValueError("Tokenizer did not return an attention_mask.")

        valid_tokens = attention_mask.bool()               # [batch, seq]
        token_counts.extend(valid_tokens.sum(dim=1).cpu().tolist())
        token_activations = layer_states[valid_tokens]     # [batch*seq_valid, hidden]
        buckets.append(token_activations.cpu())

    return torch.cat(buckets, dim=0).numpy(), token_counts


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
        if args.logistic_penalty:
            logistic_kwargs["penalty"] = args.logistic_penalty
        if args.logistic_C is not None:
            logistic_kwargs["C"] = args.logistic_C
        if args.logistic_max_iter is not None:
            logistic_kwargs["max_iter"] = args.logistic_max_iter
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=DTYPE_MAP[args.dtype],
    )

    df = load_table(args.data_path, sheet=args.sheet)
    data_hash = sha256_file(args.data_path)
    texts, label_series = prepare_dataset(df, args)

    task = detect_task(label_series, args.task)
    label_encoder: LabelEncoder | None = None
    if task == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(label_series.to_numpy())
    else:
        y = label_series.to_numpy(dtype=np.float32)

    activations, token_counts = collect_layer_activations(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        layer=args.layer,
        batch_size=args.batch_size,
        device=device,
        max_length=args.max_length,
    )

    y = np.repeat(y, token_counts)

    X_train, X_test, y_train, y_test = train_test_split(
        activations,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    standardizer = make_standardizer(args, device=device)
    probe = make_probe(args, standardizer=standardizer, device=device, task=task)

    probe.fit(X_train, y_train)
    train_pred = probe.predict(X_train)
    test_pred = probe.predict(X_test)
    metrics: dict[str, object] = {}

    if task == "classification":
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Test accuracy:  {test_acc:.4f}")
        metrics.update({"train_accuracy": float(train_acc), "test_accuracy": float(test_acc)})
        if label_encoder is not None:
            class_report = classification_report(y_test, test_pred, target_names=label_encoder.classes_, output_dict=True)
            metrics["classification_report"] = class_report
            print(classification_report(y_test, test_pred, target_names=label_encoder.classes_))
    else:
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        print(f"Train MSE: {train_mse:.4f}  R2: {train_r2:.4f}")
        print(f"Test  MSE: {test_mse:.4f}  R2: {test_r2:.4f}")
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
                    f"{sanitize_model_name(args.model_name)}_layer{args.layer}_{args.probe_type}_loss.csv"
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
                label=args.label_column,
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
                "target_feature": args.label_column,
                "task": task,
            }
            joblib.dump(artifact, artifact_path)
            metrics_path = artifact_dir / "metrics.json"
            save_json(metrics, metrics_path)

            classes = label_encoder.classes_.tolist() if label_encoder is not None else None
            metadata = build_probe_metadata(
                args=args,
                layer=args.layer,
                label_column=args.label_column,
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
        finally:
            probe.standardizer = standardizer_ref


if __name__ == "__main__":
    main()
