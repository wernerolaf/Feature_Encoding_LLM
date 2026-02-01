from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch


# ---------- Generic helpers ----------
def sanitize_model_name(name: str) -> str:
    return name.replace("/", "_")


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------- Data loading ----------
def load_table(path: str | Path, sheet: str | None = None) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(file_path, sep=sep)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path, sheet_name=sheet)
    raise ValueError(f"Unsupported data format '{suffix}'. Use csv/tsv/xlsx.")


# ---------- JSON helpers ----------
def save_json(data: Any, path: Path) -> None:
    if is_dataclass(data):
        data = asdict(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Hashing / git ----------
def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


# ---------- Artifact paths ----------
def _artifact_root(kind: str, model_name: str, base_dir: str | Path = "artifacts") -> Path:
    base = Path(base_dir)
    # Avoid duplicating the kind if the user passes a path already ending with it.
    if base.name == kind:
        root = base
    else:
        root = base / kind
    return root / sanitize_model_name(model_name)


def _layer_path(root: Path, layer: int | None) -> Path:
    return root / f"layer{layer}" if layer is not None else root


def _label_path(root: Path, label: str | None) -> Path:
    return root / str(label) if label is not None else root


def _version_dirs(path: Path) -> list[Path]:
    pattern = re.compile(r"ver_(\d+)$")
    return sorted(
        (p for p in path.iterdir() if p.is_dir() and pattern.match(p.name)),
        key=lambda p: int(pattern.match(p.name).group(1)),  # type: ignore[arg-type]
    )


def next_version_dir(
    *,
    kind: str,
    model_name: str,
    layer: int | None = None,
    label: str | None = None,
    base_dir: str | Path = "artifacts",
    version_override: int | None = None,
) -> Path:
    root = _label_path(_layer_path(_artifact_root(kind, model_name, base_dir), layer), label)
    ensure_dir(root)
    if version_override is not None:
        return ensure_dir(root / f"ver_{version_override}")
    existing = _version_dirs(root)
    next_idx = int(existing[-1].name.split("_")[1]) + 1 if existing else 1
    return ensure_dir(root / f"ver_{next_idx}")


def latest_version_dir(
    *,
    kind: str,
    model_name: str,
    layer: int | None = None,
    label: str | None = None,
    base_dir: str | Path = "artifacts",
) -> Optional[Path]:
    root = _label_path(_layer_path(_artifact_root(kind, model_name, base_dir), layer), label)
    if not root.exists():
        return None
    existing = _version_dirs(root)
    return existing[-1] if existing else None


def artifact_version_dir(
    *,
    kind: str,
    model_name: str,
    layer: int | None = None,
    label: str | None = None,
    base_dir: str | Path = "artifacts",
    version: str | int | None = "latest",
) -> Optional[Path]:
    if isinstance(version, int):
        return (_label_path(_layer_path(_artifact_root(kind, model_name, base_dir), layer), label) / f"ver_{version}")
    if version == "latest":
        return latest_version_dir(kind=kind, model_name=model_name, layer=layer, label=label, base_dir=base_dir)
    return next_version_dir(kind=kind, model_name=model_name, layer=layer, label=label, base_dir=base_dir)


# ---------- Metadata builders ----------
def build_autoencoder_metadata(
    *,
    args: Any,
    layer: int,
    input_dim: int,
    latent_dim: int,
    weights_path: Path,
    loss_history_path: Path | None,
    data_hash: str | None,
    sample_count: int,
) -> dict[str, Any]:
    return {
        "artifact_type": "autoencoder",
        "model_name": args.model_name,
        "layer": layer,
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "beta": args.beta,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "ae_batch_size": args.ae_batch_size,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "seed": args.seed,
        "data_path": str(args.data_path),
        "data_hash": data_hash,
        "sheet": args.sheet,
        "text_column": args.text_column,
        "response_column": args.response_column,
        "template_fields": args.template_fields,
        "prompt_template": args.prompt_template,
        "max_samples": args.max_samples,
        "weights_path": str(weights_path),
        "loss_history_path": str(loss_history_path) if loss_history_path else None,
        "git_sha": git_sha(),
        "sample_count": sample_count,
    }


def build_probe_metadata(
    *,
    args: Any,
    layer: int,
    label_column: str,
    task: str,
    standardizer_strategy: str,
    autoencoder_artifact: str | None,
    artifact_path: Path,
    data_hash: str | None,
    metrics: dict[str, Any] | None,
    classes: list[str] | None,
) -> dict[str, Any]:
    return {
        "artifact_type": "probe",
        "model_name": args.model_name,
        "layer": layer,
        "label_column": label_column,
        "probe_type": args.probe_type,
        "task": task,
        "standardizer": standardizer_strategy,
        "autoencoder_artifact": autoencoder_artifact,
        "dtype": args.dtype,
        "seed": args.seed,
        "data_path": str(args.data_path),
        "data_hash": data_hash,
        "sheet": args.sheet,
        "text_column": args.text_column,
        "response_column": args.response_column,
        "template_fields": args.template_fields,
        "prompt_template": args.prompt_template,
        "max_samples": args.max_samples,
        "artifact_path": str(artifact_path),
        "metrics": metrics or {},
        "classes": classes,
        "git_sha": git_sha(),
        "args": vars(args),
    }


def build_debias_metadata(
    *,
    args: Any,
    output_dir: Path,
    data_hash: str | None,
    train_examples: int,
    eval_examples: int,
    dtype: str,
    tokenizer_padding_added: bool,
    eval_metrics: dict | None = None,
    removed_train: int = 0,
    removed_eval: int = 0,
) -> dict[str, Any]:
    return {
        "artifact_type": "debias_finetune",
        "model_name": args.model_name,
        "data_path": args.data_path,
        "data_hash": data_hash,
        "sheet": args.sheet,
        "text_column": args.text_column,
        "response_column": args.response_column,
        "prompt_template": args.prompt_template,
        "template_fields": args.template_fields,
        "swap_probability": args.swap_probability,
        "swap_response_text": args.swap_response_text,
        "gender_column": args.gender_column,
        "pronoun_column": args.pronoun_column,
        "max_samples": args.max_samples,
        "max_length": args.max_length,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "filtered_train_padding_only": removed_train,
        "filtered_eval_padding_only": removed_eval,
        "seed": args.seed,
        "dtype": dtype,
        "output_dir": str(output_dir),
        "git_sha": git_sha(),
        "tokenizer_padding_added": tokenizer_padding_added,
        "args": vars(args),
        "eval_metrics": eval_metrics or {},
    }
