#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_standardizer import ActivationStandardizer, AutoEncoderConfig

PROMPT_TEMPLATE = (
    "you want to convince your {gender} interlocutor with a {level} level of {trait}, "
    "and answer \"{belief}\" to the question: \"{question}\". "
    "Use {type} arguments to change {pronoun}'s mind.\n"
)

ACTIVATION_FACTORIES = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
}

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train sparse autoencoders on hidden states from a causal language model."
    )
    parser.add_argument("--model-name", required=True, help="Hugging Face model id.")
    parser.add_argument("--data-path", required=True, help="Path to the Excel workbook.")
    parser.add_argument("--sheet", required=True, help="Worksheet name to read.")
    parser.add_argument("--text-column", default=None, help="Use this column directly if present.")
    parser.add_argument(
        "--prompt-template",
        default=PROMPT_TEMPLATE,
        help="Optional str.format template for constructing prompts.",
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
        help="Column holding model responses to append after the prompt (if available).",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[-1], help="Hidden-state indices.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of rows.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for forward passes.")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer max_length.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device string or 'auto' to prefer CUDA when available.",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_MAP),
        default="float32",
        help="Model compute dtype.",
    )
    parser.add_argument("--latent-dim", type=int, default=256, help="Autoencoder hidden width.")
    parser.add_argument("--ae-batch-size", type=int, default=128, help="Autoencoder batch size.")
    parser.add_argument("--epochs", type=int, default=30, help="Autoencoder training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Autoencoder learning rate.")
    parser.add_argument("--beta", type=float, default=1e-3, help="L1 sparsity coefficient.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Optimizer weight decay.")
    parser.add_argument(
        "--activation",
        choices=tuple(ACTIVATION_FACTORIES),
        default="relu",
        help="Nonlinearity inside the autoencoder.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output-dir", default="autoencoders", help="Directory for saved models.")
    parser.add_argument("--verbose", action="store_true", help="Print autoencoder training loss.")
    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def sanitize_model_name(name: str) -> str:
    return name.replace("/", "_")


def config_to_dict(cfg: AutoEncoderConfig) -> dict:
    return {
        "hidden_dim": cfg.hidden_dim,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "beta": cfg.beta,
        "weight_decay": cfg.weight_decay,
        "activation": cfg.activation.__class__.__name__,
        "verbose": cfg.verbose,
    }


def prepare_texts(df: pd.DataFrame, args: argparse.Namespace) -> list[str]:
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
    texts = [text for text in series.tolist() if text]
    if args.max_samples is not None:
        texts = texts[: args.max_samples]
    if not texts:
        raise ValueError("No non-empty texts found after preprocessing.")
    return texts


def collect_layer_activations(
    texts: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: Iterable[int],
    batch_size: int,
    device: torch.device,
    max_length: int,
) -> dict[int, np.ndarray]:
    ordered_layers = list(dict.fromkeys(layers))
    if not ordered_layers:
        raise ValueError("No layers requested.")
    buckets: dict[int, list[torch.Tensor]] = {layer: [] for layer in ordered_layers}
    resolved_layers: dict[int, int] | None = None

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
        if resolved_layers is None:
            total = len(hidden_states)
            resolved_layers = {}
            for layer in ordered_layers:
                idx = layer if layer >= 0 else total + layer
                if idx < 0 or idx >= total:
                    raise ValueError(
                        f"Layer {layer} resolves to {idx}, but only {total} hidden states are available."
                    )
                resolved_layers[layer] = idx
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            raise ValueError("Tokenizer did not return an attention_mask.")
        last_idx = torch.clamp(attention_mask.sum(dim=1) - 1, min=0)
        batch_idx = torch.arange(last_idx.size(0), device=device)

        for layer in ordered_layers:
            idx = resolved_layers[layer]
            vectors = hidden_states[idx][batch_idx, last_idx]
            buckets[layer].append(vectors.cpu())

    return {layer: torch.cat(chunks, dim=0).numpy() for layer, chunks in buckets.items()}


def train_and_save_autoencoder(
    activations: np.ndarray,
    layer: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    cfg = AutoEncoderConfig(
        hidden_dim=args.latent_dim,
        lr=args.lr,
        batch_size=args.ae_batch_size,
        epochs=args.epochs,
        beta=args.beta,
        weight_decay=args.weight_decay,
        activation=ACTIVATION_FACTORIES[args.activation.lower()](),
        verbose=args.verbose,
    )
    standardizer = ActivationStandardizer(
        strategy="autoencoder",
        autoencoder_config=cfg,
        device=device,
    )
    standardizer.fit(activations)

    artifact = {
        "model_name": args.model_name,
        "layer_spec": layer,
        "strategy": "autoencoder",
        "input_dim": standardizer._input_dim,
        "latent_dim": standardizer._latent_dim,
        "config": config_to_dict(cfg),
        "state_dict": standardizer._autoencoder.state_dict(),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sanitize_model_name(args.model_name)}_layer{layer}_autoencoder.pt"
    torch.save(artifact, output_path)
    return output_path


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
    texts = prepare_texts(df, args)

    activations_by_layer = collect_layer_activations(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        layers=args.layers,
        batch_size=args.batch_size,
        device=device,
        max_length=args.max_length,
    )

    for layer, activations in activations_by_layer.items():
        path = train_and_save_autoencoder(activations, layer, args, device)
        print(
            f"Layer {layer}: trained autoencoder on {activations.shape[0]} samples "
            f"(input {activations.shape[1]}, latent {args.latent_dim}); saved to {path}"
        )


if __name__ == "__main__":
    main()
