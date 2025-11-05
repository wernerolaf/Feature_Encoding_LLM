#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_standardizer import ActivationStandardizer
from probes import DecisionTreeProbe, LinearProbe, ShallowNNProbe

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
    parser.add_argument("--save-path", help="Optional path to persist the trained probe with metadata (joblib).")
    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


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

    return text_series.tolist(), label_series.to_numpy()


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

        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            raise ValueError("Tokenizer did not return an attention_mask.")
        last_idx = torch.clamp(attention_mask.sum(dim=1) - 1, min=0)
        batch_idx = torch.arange(last_idx.size(0), device=device)
        vectors = hidden_states[resolved_idx][batch_idx, last_idx]
        buckets.append(vectors.cpu())

    return torch.cat(buckets, dim=0).numpy()


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


def make_probe(args: argparse.Namespace, standardizer: ActivationStandardizer, device: torch.device):
    if args.probe_type == "linear":
        logistic_kwargs: dict[str, object] = {}
        if args.logistic_penalty:
            logistic_kwargs["penalty"] = args.logistic_penalty
        if args.logistic_C is not None:
            logistic_kwargs["C"] = args.logistic_C
        if args.logistic_max_iter is not None:
            logistic_kwargs["max_iter"] = args.logistic_max_iter
        return LinearProbe(standardizer=standardizer, logistic_kwargs=logistic_kwargs or None)

    if args.probe_type == "decision_tree":
        tree_kwargs: dict[str, object] = {"random_state": args.tree_random_state}
        if args.tree_max_depth is not None:
            tree_kwargs["max_depth"] = args.tree_max_depth
        return DecisionTreeProbe(standardizer=standardizer, tree_kwargs=tree_kwargs)

    return ShallowNNProbe(
        standardizer=standardizer,
        hidden_dim=args.nn_hidden_dim,
        dropout=args.nn_dropout,
        epochs=args.nn_epochs,
        batch_size=args.nn_batch_size,
        lr=args.nn_lr,
        weight_decay=args.nn_weight_decay,
        device=device,
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

    df = pd.read_excel(args.data_path, sheet_name=args.sheet)
    texts, labels = prepare_dataset(df, args)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    activations = collect_layer_activations(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        layer=args.layer,
        batch_size=args.batch_size,
        device=device,
        max_length=args.max_length,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        activations,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    standardizer = make_standardizer(args, device=device)
    probe = make_probe(args, standardizer=standardizer, device=device)

    probe.fit(X_train, y_train)
    train_pred = probe.predict(X_train)
    test_pred = probe.predict(X_test)

    print(f"Train accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"Test accuracy:  {accuracy_score(y_test, test_pred):.4f}")
    print(classification_report(y_test, test_pred, target_names=label_encoder.classes_))

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(probe, "model") and getattr(probe, "model", None) is not None:
            probe.model.cpu()
        if hasattr(probe.standardizer, "_autoencoder") and probe.standardizer._autoencoder is not None:
            probe.standardizer._autoencoder.cpu()

        artifact = {
            "probe_type": args.probe_type,
            "model_name": args.model_name,
            "layer": args.layer,
            "standardizer_strategy": probe.standardizer.strategy,
            "label_encoder": label_encoder,
            "classes": label_encoder.classes_,
            "probe": probe,
            "autoencoder_artifact": args.autoencoder_artifact,
        }
        joblib.dump(artifact, save_path)
        print(f"Saved probe artifact to {save_path}")


if __name__ == "__main__":
    main()
