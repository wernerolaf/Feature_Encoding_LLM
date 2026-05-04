#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

from activation_standardizer import ActivationStandardizer
from probes import DecisionTreeProbe, LinearProbe, ShallowNNProbe
from scripts.prompt_utils import PROMPT_TEMPLATE, build_prompt_series, build_text_series
from utils import (
    artifact_version_dir,
    load_table,
    next_version_dir,
    resolve_device,
    sanitize_model_name,
    save_json,
    set_seeds,
    sha256_file,
)

InterventionMode = Literal["increase", "decrease", "project"]
StrengthUnit = Literal["raw", "activation_pct"]


def tensor_to_numpy_float(tensor: torch.Tensor) -> np.ndarray:
    """Convert model activations/logits to NumPy in a dtype NumPy supports."""
    return tensor.detach().float().cpu().numpy()


def canonical_model_key(model_name: str) -> str:
    key = str(model_name).strip().split("/")[-1].lower().replace("_", "-")
    key = re.sub(r"-instruct.*$", "", key)
    key = re.sub(r"-deduped$", "", key)
    return key


def filter_rows_for_model(df: pd.DataFrame, *, model_name: str, column: str, filter_value: str | None = None) -> pd.DataFrame:
    if column not in df.columns:
        print(f"[data] model filter skipped: column '{column}' not found.")
        return df

    requested = filter_value if filter_value else model_name
    target = canonical_model_key(requested)
    keys = df[column].map(canonical_model_key)
    mask = keys == target
    filtered = df.loc[mask].copy()
    if filtered.empty:
        available = sorted(str(v) for v in df[column].dropna().unique())
        raise ValueError(
            f"No rows in '{column}' match model filter '{requested}' (key='{target}'). "
            f"Available values: {available}"
        )
    print(f"[data] model filter: column={column} value={requested} rows={len(filtered)}/{len(df)}")
    return filtered


@dataclass
class FeatureSpec:
    label_column: str
    layer: Optional[int] = None
    mode: InterventionMode = "increase"
    strength: float = 1.0
    probe_version: str | int | None = None


def detect_task(label_series: pd.Series, user_choice: str) -> str:
    """
    Decide classification vs regression.
    - honour explicit user choice
    - auto: float dtypes → regression; otherwise classification
    """
    if user_choice in {"classification", "regression"}:
        return user_choice
    if pd.api.types.is_float_dtype(label_series):
        return "regression"
    return "classification"


def build_generation_prompts(df: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    """
    Build prompts for generation only (never appending responses) so saved prompts
    do not include ground-truth answers.
    """
    series = build_prompt_series(
        df,
        text_column=args.text_column,
        prompt_template=args.prompt_template,
        template_fields=args.template_fields,
    )
    return series.tolist()


@dataclass
class FeatureProbeContext:
    name: str
    probe: LinearProbe | DecisionTreeProbe | ShallowNNProbe
    layer: int
    mode: InterventionMode
    strength: float
    class_names: list[str]
    task: str
    strength_unit: StrengthUnit = "raw"
    intervene: bool = True
    analytics: list[dict[str, object]] = field(default_factory=list)
    training_metrics: dict[str, float] = field(default_factory=dict)

    def log(self, step: int, probabilities: np.ndarray, *, tag: str) -> None:
        summary: dict[str, dict[str, float]] = {}
        if probabilities.ndim == 1 or probabilities.shape[1] == 1:
            col = probabilities.reshape(-1)
            key = self.class_names[0] if self.class_names else "value"
            summary[key] = {
                "mean": float(col.mean()),
                "std": float(col.std()),
                "min": float(col.min()),
                "max": float(col.max()),
            }
        else:
            for idx, class_name in enumerate(self.class_names):
                col = probabilities[:, idx]
                summary[str(class_name)] = {
                    "mean": float(col.mean()),
                    "std": float(col.std()),
                    "min": float(col.min()),
                    "max": float(col.max()),
                }
        self.analytics.append(
            {
                "step": step,
                "mode": self.mode,
                "strength": self.strength,
                "strength_unit": self.strength_unit,
                "intervene": self.intervene,
                "tokens_observed": int(probabilities.shape[0]),
                "class_stats": {tag: summary},
            }
        )


def parse_feature_specs(raw_specs: Sequence[str]) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for raw in raw_specs:
        parts = raw.split(":")
        head = parts[0]
        if "@" in head:
            label_column, layer_part = head.split("@", 1)
            layer_val: Optional[int] = int(layer_part)
        else:
            label_column = head
            layer_val = None
        mode: InterventionMode = "increase"
        strength = 1.0
        probe_version: str | int | None = None
        # parse optional mode / strength robustly
        if len(parts) > 1 and parts[1]:
            candidate = parts[1].lower()
            if candidate.replace(".", "", 1).isdigit():
                strength = float(candidate)
            else:
                mode = candidate  # type: ignore[assignment]
        if len(parts) > 2 and parts[2]:
            try:
                strength = float(parts[2])
            except ValueError:
                # if user swapped order, treat as mode
                mode = parts[2].lower()  # type: ignore[assignment]
        if len(parts) > 3 and parts[3]:
            probe_version = int(parts[3]) if parts[3].isdigit() else parts[3]
        specs.append(
            FeatureSpec(
                label_column=label_column,
                layer=layer_val,
                mode=mode,
                strength=strength,
                probe_version=probe_version,
            )
        )
    return specs


def load_standardizer(metadata: dict | None, device: torch.device) -> ActivationStandardizer | None:
    if metadata is None:
        return None

    strategy = metadata.get("strategy", "identity")
    if strategy == "identity":
        return ActivationStandardizer(strategy="identity", device=device)

    if strategy == "standard":
        scaler_state = metadata.get("scaler_state")
        standardizer = ActivationStandardizer(strategy="standard", device=device)
        if scaler_state is not None:
            # Re-attach stored scaler state directly.
            standardizer._scaler = scaler_state  # type: ignore[attr-defined]
        return standardizer

    if strategy == "autoencoder":
        auto_path = metadata.get("autoencoder_artifact")
        if not auto_path:
            raise RuntimeError("Autoencoder strategy specified but no autoencoder_artifact provided in metadata.")
        standardizer = ActivationStandardizer(strategy="autoencoder", device=device)
        standardizer.load_autoencoder_artifact(auto_path, map_location=device)
        return standardizer

    raise ValueError(f"Unknown standardizer strategy '{strategy}'.")


def load_probe_artifact(
    label: str,
    args: argparse.Namespace,
    *,
    device: torch.device,
    layer: int,
    mode: InterventionMode,
    strength: float,
    strength_unit: StrengthUnit,
    intervene: bool = True,
    probe_version_override: str | int | None = None,
) -> FeatureProbeContext:
    candidates: list[Path] = []
    probe_version: str | int | None = probe_version_override if probe_version_override is not None else args.probe_version
    if isinstance(probe_version, str) and probe_version.isdigit():
        probe_version = int(probe_version)
    version_dir = artifact_version_dir(
        kind="probes",
        model_name=args.model_name,
        layer=layer,
        label=label,
        base_dir=args.probe_dir,
        version=probe_version,
    )
    if version_dir:
        candidates.append(version_dir / "probe.pkl")
    candidates.extend(
        [
            Path(args.probe_dir)
            / f"{sanitize_model_name(args.model_name)}_layer{layer}_{args.probe_type}_{label}.pkl",
            Path(args.probe_dir) / f"pythia70m_L{layer}_{label}.pkl",  # legacy naming fallback
        ]
    )
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Missing probe artifact for '{label}'. Tried: {', '.join(str(p) for p in candidates)}")

    artifact = joblib.load(path)
    probe = artifact.get("probe")
    if probe is None:
        raise ValueError(f"Probe artifact at {path} missing 'probe' entry.")

    std_metadata = artifact.get("standardizer")
    standardizer = load_standardizer(std_metadata, device=device)
    if standardizer is not None:
        probe.standardizer = standardizer

    task = artifact.get("task", "classification")
    training_metrics = artifact.get("training_metrics", {})
    classes = artifact.get("classes")
    class_names: list[str] = []
    if classes is not None:
        if isinstance(classes, (list, tuple)):
            class_names = [str(c) for c in classes]
        else:
            class_names = [str(c) for c in list(classes)]

    label_encoder = artifact.get("label_encoder")
    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        class_names = [str(c) for c in label_encoder.classes_]

    return FeatureProbeContext(
        name=label,
        probe=probe,
        layer=layer,
        mode=mode,
        strength=strength,
        class_names=class_names if class_names else (["value"] if task == "regression" else []),
        task=task,
        training_metrics=training_metrics,
        strength_unit=strength_unit,
        intervene=intervene,
    )


def resolve_layer_module(model: AutoModelForCausalLM, layer_idx: int):
    candidates: list[Iterable[torch.nn.Module]] = []
    for root_name in ("transformer", "model", "encoder", "decoder", "gpt_neox", "backbone"):
        root = getattr(model, root_name, None)
        if root is None:
            continue
        for stack_name in ("h", "layers", "block"):
            stack = getattr(root, stack_name, None)
            if stack is not None:
                candidates.append(stack)

        decoder = getattr(root, "decoder", None)
        if decoder is not None and hasattr(decoder, "layers"):
            candidates.append(decoder.layers)

    for stack in candidates:
        modules = list(stack)
        if 0 <= layer_idx < len(modules):
            return modules[layer_idx]
    raise ValueError(f"Unable to locate layer index {layer_idx} in model '{model.config.model_type}'.")


def num_transformer_layers(model: AutoModelForCausalLM) -> int:
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(model.config, attr, None)
        if value is not None:
            return int(value)
    idx = 0
    while True:
        try:
            resolve_layer_module(model, idx)
        except ValueError:
            break
        idx += 1
    if idx == 0:
        raise ValueError(f"Unable to infer layer count for model '{model.config.model_type}'.")
    return idx


def parse_layer_selection(selection: str | None, *, model: AutoModelForCausalLM) -> list[int]:
    if not selection:
        return []
    total_layers = num_transformer_layers(model)
    raw = selection.strip().lower()
    if raw == "all":
        return list(range(total_layers))

    layers: set[int] = set()
    for part in raw.replace(",", " ").split():
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            step = 1 if end >= start else -1
            layers.update(range(start, end + step, step))
        else:
            layers.add(int(part))

    invalid = [layer for layer in layers if layer < 0 or layer >= total_layers]
    if invalid:
        raise ValueError(f"Requested collect layers {invalid} outside valid range 0..{total_layers - 1}.")
    return sorted(layers)


class ProbeInterventionHook:
    def __init__(self, contexts: list[FeatureProbeContext], *, layer_idx: int, output_dir: Path) -> None:
        self.contexts = contexts
        self.step = 0
        self.active = True  # when False, no adjustments applied
        self.layer_idx = layer_idx
        self.output_dir = output_dir

    def __call__(self, module, inputs, outputs):
        hidden, payload_type = self._extract_hidden(outputs)
        batch, seq_len, hidden_dim = hidden.shape
        flat = hidden.reshape(-1, hidden_dim)
        flat_np = tensor_to_numpy_float(flat)

        total_adjustment = torch.zeros_like(flat)

        for ctx in self.contexts:
            with torch.enable_grad():
                raw_gradients = ctx.probe.get_gradient(flat_np, normalize=False)

            grad_source = "gradient"
            raw_norms = np.linalg.norm(raw_gradients, axis=1)
            ctx.analytics.append(
                {
                    "step": self.step,
                    "mode": ctx.mode,
                    "strength": ctx.strength,
                    "strength_unit": ctx.strength_unit,
                    "intervene": ctx.intervene,
                    "tag": "gradient_norm",
                    "source": grad_source,
                    "tokens_observed": int(raw_norms.shape[0]),
                    "grad_norm_mean": float(np.mean(raw_norms)),
                    "grad_norm_std": float(np.std(raw_norms)),
                    "grad_norm_min": float(np.min(raw_norms)),
                    "grad_norm_max": float(np.max(raw_norms)),
                }
            )

            grad_tensor = torch.from_numpy(raw_gradients).to(flat.device)
            grad_norm = grad_tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            grad_unit = grad_tensor / grad_norm

            if self.active and ctx.intervene and ctx.strength != 0:
                if ctx.strength_unit == "activation_pct":
                    activation_norm = flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    pct = ctx.strength / 100.0
                    if ctx.mode == "increase":
                        adjustment = pct * activation_norm * grad_unit
                    elif ctx.mode == "decrease":
                        adjustment = -pct * activation_norm * grad_unit
                    else:
                        projection = (flat * grad_unit).sum(dim=-1, keepdim=True)
                        adjustment = -pct * projection * grad_unit
                elif ctx.mode == "increase":
                    adjustment = ctx.strength * grad_tensor
                elif ctx.mode == "decrease":
                    adjustment = -ctx.strength * grad_tensor
                else:
                    projection = (flat * grad_tensor).sum(dim=-1, keepdim=True)
                    adjustment = -ctx.strength * projection * grad_tensor
                total_adjustment += adjustment

            # pre-adjustment stats
            try:
                pre_scores = ctx.probe.predict_proba(flat_np)
            except Exception:
                pre_scores = ctx.probe.predict(flat_np)
            pre_arr = np.asarray(pre_scores)
            if pre_arr.ndim == 1:
                pre_arr = pre_arr.reshape(-1, 1)
            ctx.log(self.step, pre_arr, tag="pre")

        flat = flat + total_adjustment
        # post-adjustment stats
        for ctx in self.contexts:
            adjusted_np = tensor_to_numpy_float(flat)
            try:
                post_scores = ctx.probe.predict_proba(adjusted_np)
            except Exception:
                post_scores = ctx.probe.predict(adjusted_np)
            post_arr = np.asarray(post_scores)
            if post_arr.ndim == 1:
                post_arr = post_arr.reshape(-1, 1)
            ctx.log(self.step, post_arr, tag="post" if self.active else "neutral")

        hidden = flat.reshape(batch, seq_len, hidden_dim)
        self.step += 1

        if payload_type == "tuple":
            return (hidden,) + outputs[1:]
        return hidden

    @staticmethod
    def _extract_hidden(outputs):
        if isinstance(outputs, tuple):
            return outputs[0], "tuple"
        return outputs, "tensor"


def read_prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def compute_logprobs_and_entropy(outputs, generated_ids: torch.Tensor, *, chunk_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute logprobs for generated/chosen tokens and token-wise entropy.
    Entropy is derived from the full distribution at each step.
    """
    scores = getattr(outputs, "scores", None)
    if scores is not None and generated_ids.numel() > 0:
        logprobs: list[np.ndarray] = []
        entropies: list[np.ndarray] = []
        for step, score in enumerate(scores):
            logp = torch.log_softmax(score, dim=-1)
            probs = torch.softmax(score, dim=-1)
            entropy = -(probs * logp).sum(dim=-1)
            token_ids = generated_ids[:, step]
            token_logprobs = logp.gather(1, token_ids.view(-1, 1)).squeeze(1)
            logprobs.append(tensor_to_numpy_float(token_logprobs))
            entropies.append(tensor_to_numpy_float(entropy))
        return np.stack(logprobs, axis=1), np.stack(entropies, axis=1)

    logits = getattr(outputs, "logits", None)
    if logits is None or generated_ids.numel() == 0:
        zeros = np.zeros((generated_ids.size(0), generated_ids.size(1)), dtype=np.float32)
        return zeros, zeros

    target = generated_ids
    max_t = min(logits.size(1), target.size(1))
    if max_t <= 1:
        zeros = np.zeros((target.size(0), target.size(1)), dtype=np.float32)
        return zeros, zeros

    target = target[:, 1:max_t]
    gathered_chunks: list[torch.Tensor] = []
    entropy_chunks: list[torch.Tensor] = []
    for start in range(0, max_t - 1, max(1, chunk_size)):
        end = min(max_t - 1, start + max(1, chunk_size))
        logits_chunk = logits[:, start:end, :]
        target_chunk = target[:, start:end]
        log_probs_chunk = torch.log_softmax(logits_chunk, dim=-1)
        probs_chunk = torch.exp(log_probs_chunk)
        gathered_chunks.append(log_probs_chunk.gather(2, target_chunk.unsqueeze(-1)).squeeze(-1).detach().float().cpu())
        entropy_chunks.append((-(probs_chunk * log_probs_chunk).sum(dim=-1)).detach().float().cpu())
        del logits_chunk, target_chunk, log_probs_chunk, probs_chunk
    gathered = torch.cat(gathered_chunks, dim=1)
    entropy = torch.cat(entropy_chunks, dim=1)

    pad = torch.full((gathered.size(0), 1), float("nan"))
    pad_h = torch.full((entropy.size(0), 1), float("nan"))
    seq_logprobs = torch.cat([pad, gathered], dim=1)
    seq_entropy = torch.cat([pad_h, entropy], dim=1)

    if seq_logprobs.size(1) < generated_ids.size(1):
        tail = generated_ids.size(1) - seq_logprobs.size(1)
        pad_tail_lp = torch.full((seq_logprobs.size(0), tail), float("nan"))
        pad_tail_ent = torch.full((seq_entropy.size(0), tail), float("nan"))
        seq_logprobs = torch.cat([seq_logprobs, pad_tail_lp], dim=1)
        seq_entropy = torch.cat([seq_entropy, pad_tail_ent], dim=1)

    return seq_logprobs.cpu().numpy(), seq_entropy.cpu().numpy()


def run_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    *,
    prompts: list[str],
    contexts_by_layer: dict[int, list[FeatureProbeContext]],
    device: torch.device,
    args: argparse.Namespace,
    out_dir: Path,
    generate_new_tokens: bool = True,
    hook_active: bool = True,
):
    cos_records: list[dict[str, object]] = []
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    if contexts_by_layer:
        for layer_idx, ctxs in contexts_by_layer.items():
            hook_contexts = [ctx for ctx in ctxs if ctx.intervene]
            if not hook_contexts:
                continue
            module = resolve_layer_module(model, layer_idx)
            intervention_hook = ProbeInterventionHook(hook_contexts, layer_idx=layer_idx, output_dir=out_dir)
            intervention_hook.active = hook_active
            hooks.append(module.register_forward_hook(intervention_hook))
    try:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        generations: list[str] = []
        all_logprobs: list[np.ndarray] = []
        all_entropy: list[np.ndarray] = []
        all_token_ids: list[np.ndarray] = []
        per_token_preds: list[dict[str, list[float]]] = []
        batch_size = max(1, args.generation_batch_size)
        run_label = f"{'intervention' if hook_active else 'baseline'}_{'new' if generate_new_tokens else 'old'}"
        batch_starts = range(0, len(prompts), batch_size)
        total_batches = len(batch_starts)
        progress_style = "none" if args.no_progress else args.progress_style
        if progress_style == "tqdm":
            batch_iter = tqdm(
                batch_starts,
                desc=run_label,
                unit="batch",
                dynamic_ncols=True,
            )
        else:
            batch_iter = batch_starts
            if progress_style == "line":
                print(f"[progress] {run_label} batches={total_batches} batch_size={batch_size}", flush=True)

        for start in batch_iter:
            batch_start_time = time.perf_counter()
            batch_num = (start // batch_size) + 1
            batch_prompts = prompts[start : start + batch_size]
            layers_scored = 0
            enc = None
            outputs = None
            sequences = None
            generated_ids = None
            hidden_states = None
            layer_states = None
            enc_for_hidden = None
            try:
                enc = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    if generate_new_tokens:
                        outputs = model.generate(
                            **enc,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                        sequences = outputs.sequences
                        generated_ids = sequences[:, enc.input_ids.shape[1] :]
                    else:
                        outputs = model(**enc, output_hidden_states=True)
                        generated_ids = enc.input_ids  # evaluate plausibility of existing sequence

                # If no new tokens, still capture the prompt (or empty generation)
                generations.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True) if generate_new_tokens else [""] * enc.input_ids.size(0))
                lp, ent = compute_logprobs_and_entropy(outputs, generated_ids, chunk_size=args.logprob_chunk_size)
                all_logprobs.append(lp)
                all_entropy.append(ent)
                all_token_ids.extend(list(generated_ids.cpu().numpy()))
                if contexts_by_layer:
                    # compute gradient cosines on this batch of hidden states
                    with torch.no_grad():
                        attention_mask = enc["attention_mask"]
                        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                            hidden_states = outputs.hidden_states
                        else:
                            enc_for_hidden = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
                            hidden_states = model(**enc_for_hidden, output_hidden_states=True).hidden_states
                            attention_mask = enc_for_hidden["attention_mask"]
                        # align mask and hidden length defensively
                        seq_len = min(attention_mask.shape[1], hidden_states[0].shape[1])
                        attention_mask = attention_mask[:, :seq_len]
                        total_layers = len(hidden_states)
                        layer_items = sorted(contexts_by_layer.items(), key=lambda kv: kv[0])
                        for layer_idx, ctxs in layer_items:
                            layers_scored += 1
                            # hidden_states includes embeddings at position 0; shift positives by +1
                            idx = layer_idx + 1 if layer_idx >= 0 else total_layers + layer_idx
                            if idx < 0 or idx >= total_layers:
                                raise IndexError(f"Requested layer index {layer_idx} maps to hidden_states[{idx}] out of range 0..{total_layers-1}")
                            layer_states = hidden_states[idx][:, :seq_len, :]
                            valid_mask = attention_mask.bool()
                            flat = tensor_to_numpy_float(layer_states[valid_mask])
                            cos_records.extend(compute_gradient_cosines(ctxs, flat, layer=layer_idx))

                            # per-token probe predictions
                            batch_size_cur = layer_states.size(0)
                            seq_len_cur = layer_states.size(1)
                            for ctx in ctxs:
                                try:
                                    preds = ctx.probe.predict_proba(flat)
                                    if preds.ndim == 2 and preds.shape[1] > 1:
                                        preds_scalar = preds[:, 1]  # prob of class 1
                                    else:
                                        preds_scalar = preds.reshape(-1)
                                except Exception:
                                    try:
                                        preds = ctx.probe.predict(flat)
                                        preds_scalar = np.asarray(preds).reshape(-1)
                                    except Exception:
                                        preds_scalar = np.full(flat.shape[0], np.nan)

                                filled = np.full((batch_size_cur, seq_len_cur), np.nan, dtype=float)
                                flat_idx = 0
                                vm = valid_mask.cpu().numpy()
                                for bi in range(batch_size_cur):
                                    positions = np.nonzero(vm[bi])[0]
                                    count = len(positions)
                                    if count > 0:
                                        filled[bi, positions] = preds_scalar[flat_idx : flat_idx + count]
                                        flat_idx += count

                                # ensure per_token_preds has entries
                                pred_key = f"{ctx.name}_L{layer_idx}"
                                while len(per_token_preds) < start + batch_size_cur:
                                    per_token_preds.append({})
                                for bi in range(batch_size_cur):
                                    idx_global = start + bi
                                    if idx_global >= len(per_token_preds):
                                        per_token_preds.append({})
                                    entry = per_token_preds[idx_global]
                                    entry[pred_key] = filled[bi].tolist()

            finally:
                del outputs, sequences, generated_ids, hidden_states, layer_states, enc_for_hidden, enc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if progress_style == "line":
                elapsed = time.perf_counter() - batch_start_time
                print(
                    f"[progress] {run_label} batch={batch_num}/{total_batches} "
                    f"rows={len(batch_prompts)} probe_layers={layers_scored} elapsed={elapsed:.1f}s",
                    flush=True,
                )

        def pad_and_concat(arrs: list[np.ndarray]) -> np.ndarray | None:
            if not arrs:
                return None
            max_cols = max(arr.shape[1] for arr in arrs)
            padded = []
            for arr in arrs:
                if arr.shape[1] < max_cols:
                    pad_width = ((0, 0), (0, max_cols - arr.shape[1]))
                    arr = np.pad(arr, pad_width, constant_values=np.nan)
                padded.append(arr)
            return np.concatenate(padded, axis=0)

        logprobs_concat = pad_and_concat(all_logprobs)
        entropy_concat = pad_and_concat(all_entropy)
        # ensure per_token_preds aligns with prompt count
        while len(per_token_preds) < len(prompts):
            per_token_preds.append({})

        return (
            generations,
            None,
            logprobs_concat,
            entropy_concat,
            all_token_ids,
            cos_records,
            per_token_preds,
        )
    finally:
        for h in hooks:
            h.remove()


def save_token_stats(
    output_dir: Path,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    token_ids: list[np.ndarray],
    logprobs: Optional[np.ndarray],
    entropy: Optional[np.ndarray],
    *,
    contexts: Optional[list[FeatureProbeContext]] = None,
    token_preds: Optional[list[dict[str, list[float]]]] = None,
    tag: str = "token_stats",
    file_format: str = "parquet",   # "parquet" or "csv"
    include_token_text: bool = False,
    skip_special_tokens: bool = False,
    chunk_size: int = 200_000,
) -> Path:
    """
    Wide-format token stats writer.
    One row per (prompt_index, token_index) with one prediction column per feature/layer key.
    """
    import math
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize format
    fmt = file_format.lower().strip()
    if fmt not in {"csv", "parquet"}:
        raise ValueError("file_format must be 'csv' or 'parquet'.")

    feature_names: set[str] = set()
    if contexts:
        for ctx in contexts:
            feature_names.add(f"{ctx.name}_L{ctx.layer}")
    if token_preds:
        for entry in token_preds:
            feature_names.update(entry.keys())
    feature_columns = sorted(feature_names)

    # Try parquet writer (chunk-append)
    parquet_writer = None
    pa = None
    pq = None
    if fmt == "parquet":
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except Exception:
            fmt = "csv"  # fallback

    out_path = output_dir / f"{tag}.{fmt}"
    if out_path.exists():
        out_path.unlink()

    token_cache: dict[int, str] = {}
    all_special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if eos_id is not None:
        all_special_ids.add(int(eos_id))
    if pad_id is not None:
        all_special_ids.add(int(pad_id))

    columns = [
        "tag",
        "prompt_index",
        "token_index",
        "token_id",
        "token_text",
        "is_special",
        "logprob",
        "entropy",
    ] + feature_columns

    rows: list[tuple] = []
    csv_header_written = False

    def flush_rows() -> None:
        nonlocal rows, parquet_writer, csv_header_written
        if not rows:
            return
        df = pd.DataFrame.from_records(rows, columns=columns)

        # keep storage compact
        for c in ("logprob", "entropy", *feature_columns):
            df[c] = df[c].astype("float32")
        df["is_special"] = df["is_special"].astype("bool")
        df["token_id"] = df["token_id"].astype("int32")
        df["token_index"] = df["token_index"].astype("int32")
        df["prompt_index"] = df["prompt_index"].astype("int32")
        if not include_token_text:
            df = df.drop(columns=["token_text"])

        if fmt == "csv":
            df.to_csv(out_path, mode="a", index=False, header=not csv_header_written)
            csv_header_written = True
        else:
            table = pa.Table.from_pandas(df, preserve_index=False)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(str(out_path), table.schema, compression="zstd")
            parquet_writer.write_table(table)

        rows = []

    n_prompts = min(len(prompts), len(token_ids))
    for i in range(n_prompts):
        ids = token_ids[i]
        lp_row = logprobs[i] if (logprobs is not None and i < logprobs.shape[0]) else None
        ent_row = entropy[i] if (entropy is not None and i < entropy.shape[0]) else None
        preds_map = token_preds[i] if (token_preds is not None and i < len(token_preds)) else {}

        max_len = len(ids)
        if lp_row is not None:
            max_len = min(max_len, int(lp_row.shape[0]))
        if ent_row is not None:
            max_len = min(max_len, int(ent_row.shape[0]))

        for j in range(max_len):
            tid = int(ids[j])
            is_special = tid in all_special_ids
            if skip_special_tokens and is_special:
                continue

            token_text = None
            if include_token_text:
                token_text = token_cache.get(tid)
                if token_text is None:
                    token_text = tokenizer.convert_ids_to_tokens([tid])[0]
                    token_cache[tid] = token_text

            lp_val = float(lp_row[j]) if lp_row is not None else math.nan
            ent_val = float(ent_row[j]) if ent_row is not None else math.nan

            row = [
                tag,
                i,
                j,
                tid,
                token_text,
                is_special,
                lp_val,
                ent_val,
            ]
            for fk in feature_columns:
                vals = preds_map.get(fk)
                row.append(float(vals[j]) if (vals is not None and j < len(vals)) else math.nan)
            rows.append(tuple(row))

            if len(rows) >= chunk_size:
                flush_rows()

    flush_rows()
    if parquet_writer is not None:
        parquet_writer.close()
    return out_path


def save_generations(
    output_dir: Path,
    prompts: list[str],
    baseline_generations: list[str],
    intervened_generations: list[str],
    baseline_logprobs: Optional[np.ndarray],
    intervened_logprobs: Optional[np.ndarray],
    tag: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = output_dir / f"generations_{tag}.json"
    payload = []
    if baseline_logprobs is not None and intervened_logprobs is not None:
        min_len = min(baseline_logprobs.shape[1], intervened_logprobs.shape[1])
        clipped_base = baseline_logprobs[:, :min_len]
        clipped_int = intervened_logprobs[:, :min_len]
        deltas = (clipped_int - clipped_base).mean(axis=1)
    else:
        deltas = np.full(len(prompts), np.nan)

    for prompt, base_gen, int_gen, delta in zip(prompts, baseline_generations, intervened_generations, deltas):
        payload.append(
            {
                "prompt": prompt,
                "baseline_generation": base_gen,
                "intervened_generation": int_gen,
                "tag": sanitize_model_name(prompt[:32]),
                "avg_logprob_delta": float(delta) if not np.isnan(delta) else None,
            }
        )
    generations_path.write_text(json.dumps(payload, indent=2))


def compute_gradient_cosines(
    contexts: list[FeatureProbeContext],
    flat_np: np.ndarray,
    *,
    layer: int,
) -> list[dict[str, object]]:
    """
    Compute cosine similarity between probe gradients on a shared activation batch.
    Returns list of records with pair, cosine, and token count used.
    """
    records: list[dict[str, object]] = []
    if len(contexts) < 2:
        return records
    gradients: dict[str, np.ndarray] = {}
    for ctx in contexts:
        try:
            grad = ctx.probe.get_gradient(flat_np)
        except Exception as exc:  # pragma: no cover - diagnostics only
            records.append({"pair": (ctx.name, None), "error": str(exc), "tokens": flat_np.shape[0], "layer": layer})
            continue
        gradients[ctx.name] = grad

    names = list(gradients)
    for i, name_i in enumerate(names):
        grad_i = gradients[name_i]
        norm_i = np.linalg.norm(grad_i, axis=1, keepdims=True) + 1e-12
        grad_i_norm = grad_i / norm_i
        for name_j in names[i + 1 :]:
            grad_j = gradients[name_j]
            norm_j = np.linalg.norm(grad_j, axis=1, keepdims=True) + 1e-12
            grad_j_norm = grad_j / norm_j
            cos = float(np.mean(np.sum(grad_i_norm * grad_j_norm, axis=1)))
            records.append({"pair": (name_i, name_j), "cosine": cos, "tokens": flat_np.shape[0], "layer": layer})
    return records


def save_gradient_cosines(
    output_dir: Path,
    records: list[dict[str, object]],
) -> None:
    if not records:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "gradient_cosines.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_a", "feature_b", "cosine", "tokens", "layer", "error"])
        for rec in records:
            pair = rec.get("pair", ("", ""))
            error = rec.get("error")
            writer.writerow(
                [
                    pair[0],
                    pair[1] if len(pair) > 1 else "",
                    rec.get("cosine", ""),
                    rec.get("tokens", ""),
                    rec.get("layer", ""),
                    error if error else "",
                ]
            )


def save_probe_analytics(output_dir: Path, contexts: list[FeatureProbeContext]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = []
    for ctx in contexts:
        payload.append(
            {
                "feature": ctx.name,
                "layer": ctx.layer,
                "mode": ctx.mode,
                "strength": ctx.strength,
                "strength_unit": ctx.strength_unit,
                "intervene": ctx.intervene,
                "task": ctx.task,
                "class_names": ctx.class_names,
                "training_metrics": ctx.training_metrics,
                "analytics": ctx.analytics,
            }
        )
    save_json({"contexts": payload}, output_dir / "probe_analytics.json")


def save_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    feature_specs: list[FeatureSpec],
    layers: list[int],
    *,
    data_hash: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "artifact_type": "experiment",
        "experiment_name": args.experiment_name,
        "model_name": args.model_name,
        "layers": layers,
        "probe_dir": args.probe_dir,
        "probe_version": args.probe_version,
        "probe_type": args.probe_type,
        "data_path": args.data_path,
        "data_hash": data_hash,
        "sheet": args.sheet,
        "features": [spec.__dict__ for spec in feature_specs],
        "strength_unit": args.strength_unit,
        "collect_probe_layers": args.collect_probe_layers,
        "dtype": args.dtype,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": args.do_sample,
            "generation_batch_size": args.generation_batch_size,
        },
        "args": vars(args),
    }
    save_json(meta, output_dir / "metadata.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature interaction experiment with probe interventions.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sheet", default=None, help="Optional worksheet name (xlsx only).")
    parser.add_argument("--model-filter-column", default="model_name",
                        help="If present in the dataset, keep only rows whose model name matches --model-name.")
    parser.add_argument("--model-filter-value", default=None,
                        help="Explicit dataset model_name value to filter for, overriding --model-name matching.")
    parser.add_argument("--no-model-filter", action="store_true",
                        help="Disable filtering dataset rows by --model-name.")
    parser.add_argument("--layer", type=int, default=None, help="Optional default layer applied to any feature without an explicit layer.")
    parser.add_argument("--probe-dir", default="artifacts/probes", help="Directory containing pre-trained probe artifacts.")
    parser.add_argument("--probe-version", default="latest", help="Probe artifact version to load, e.g. 'latest' or '73'.")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--prompt-template", default=PROMPT_TEMPLATE)
    parser.add_argument("--template-fields", nargs="+", default=["gender", "level", "trait", "belief", "question", "type", "pronoun"])
    parser.add_argument("--response-column", default="response")
    parser.add_argument("--feature", dest="feature_specs", action="append", required=True,
                        help="Format: label_column[:mode[:strength[:probe_version]]]. Mode ∈ {increase,decrease,project}.")
    parser.add_argument("--collect-feature", dest="collect_feature_specs", action="append", default=[],
                        help="Additional label_column[:mode[:strength[:probe_version]]] probes to score without intervening. Layers come from --collect-probe-layers unless annotated as label@layer.")
    parser.add_argument("--strength-unit", choices=["raw", "activation_pct"], default="raw",
                        help="raw: multiply raw probe gradient by strength. activation_pct: strength is percent of activation norm along normalized probe direction.")
    parser.add_argument("--collect-probe-layers", default=None,
                        help="Additional layers to observe for every requested feature without intervening. Use 'all', '0,4,8', or ranges like '0-31'.")
    parser.add_argument("--probe-type", choices=["linear", "decision_tree", "shallow_nn"], default="linear",
                        help="Probe type used in artifact naming (no training happens here).")
    parser.add_argument("--output-dir", default="artifacts/experiments", help="Base directory for experiment outputs.")
    parser.add_argument("--experiment-name", default="feature_interaction", help="Name used for versioned experiment directory.")
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Optional explicit version number; otherwise next version is created.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--generation-batch-size", type=int, default=4, help="Batch size for generation to manage memory.")
    parser.add_argument("--logprob-chunk-size", type=int, default=16,
                        help="Sequence chunk size for full-vocabulary logprob/entropy computation; lower values reduce GPU memory.")
    parser.add_argument("--token-stats-format", choices=["csv", "parquet"], default="parquet",
                        help="Storage format for token-level statistics.")
    parser.add_argument("--token-stats-include-text", action="store_true",
                        help="Include decoded token text in token-level outputs (larger files).")
    parser.add_argument("--token-stats-skip-special", action="store_true",
                        help="Skip special tokens when exporting token-level stats.")
    parser.add_argument("--progress-style", choices=["line", "tqdm", "none"], default="line",
                        help="Progress reporting style. 'line' is Slurm-friendly; 'tqdm' is better for interactive terminals.")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress reporting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    device = resolve_device(args.device)
    out_dir = next_version_dir(
        kind="experiments",
        model_name=args.model_name,
        label=args.experiment_name,
        base_dir=args.output_dir,
        version_override=args.version,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype_map[args.dtype])
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype_map[args.dtype])
    model.to(device)
    model.eval()

    df = load_table(args.data_path, sheet=args.sheet)
    if not args.no_model_filter:
        df = filter_rows_for_model(
            df,
            model_name=args.model_name,
            column=args.model_filter_column,
            filter_value=args.model_filter_value,
        )
    data_hash = sha256_file(args.data_path)
    text_series, filtered_df = build_text_series(
        df,
        text_column=args.text_column,
        prompt_template=args.prompt_template,
        template_fields=args.template_fields,
        response_column=args.response_column,
        prompt_response_sep="",
    )  # includes responses when available

    feature_specs = parse_feature_specs(args.feature_specs)
    contexts_by_layer: dict[int, list[FeatureProbeContext]] = {}
    layers: set[int] = set()
    context_keys: set[tuple[str, int]] = set()

    for spec in feature_specs:
        resolved_layer = spec.layer if spec.layer is not None else args.layer
        if resolved_layer is None:
            raise ValueError("No layer specified. Provide --layer or annotate each --feature with @<layer>.")
        layers.add(resolved_layer)
        print(f"[load] feature={spec.label_column} layer={resolved_layer} mode={spec.mode} strength={spec.strength} unit={args.strength_unit} intervene=1")
        context = load_probe_artifact(
            spec.label_column,
            args,
            device=device,
            layer=resolved_layer,
            mode=spec.mode,
            strength=spec.strength,
            strength_unit=args.strength_unit,
            intervene=True,
            probe_version_override=spec.probe_version,
        )
        contexts_by_layer.setdefault(resolved_layer, []).append(context)
        context_keys.add((spec.label_column, resolved_layer))

    collect_layers = parse_layer_selection(args.collect_probe_layers, model=model)
    if collect_layers:
        for spec in feature_specs:
            for collect_layer in collect_layers:
                key = (spec.label_column, collect_layer)
                if key in context_keys:
                    continue
                print(f"[load] feature={spec.label_column} layer={collect_layer} mode={spec.mode} strength=0.0 unit={args.strength_unit} intervene=0")
                try:
                    context = load_probe_artifact(
                        spec.label_column,
                        args,
                        device=device,
                        layer=collect_layer,
                        mode=spec.mode,
                        strength=0.0,
                        strength_unit=args.strength_unit,
                        intervene=False,
                        probe_version_override=spec.probe_version,
                    )
                except FileNotFoundError as exc:
                    print(f"[warn] skipping missing collection probe: {exc}")
                    continue
                contexts_by_layer.setdefault(collect_layer, []).append(context)
                context_keys.add(key)
                layers.add(collect_layer)

    if args.collect_feature_specs:
        collect_feature_specs = parse_feature_specs(args.collect_feature_specs)
        fallback_layers = collect_layers if collect_layers else sorted(layers)
        for spec in collect_feature_specs:
            target_layers = [spec.layer] if spec.layer is not None else fallback_layers
            if not target_layers:
                raise ValueError("No collection layers available. Set --collect-probe-layers, --layer, or use collect-feature@layer.")
            for collect_layer in target_layers:
                key = (spec.label_column, collect_layer)
                if key in context_keys:
                    continue
                print(f"[load] collect_feature={spec.label_column} layer={collect_layer} mode={spec.mode} strength=0.0 unit={args.strength_unit} intervene=0")
                try:
                    context = load_probe_artifact(
                        spec.label_column,
                        args,
                        device=device,
                        layer=collect_layer,
                        mode=spec.mode,
                        strength=0.0,
                        strength_unit=args.strength_unit,
                        intervene=False,
                        probe_version_override=spec.probe_version,
                    )
                except FileNotFoundError as exc:
                    print(f"[warn] skipping missing collection probe: {exc}")
                    continue
                contexts_by_layer.setdefault(collect_layer, []).append(context)
                context_keys.add(key)
                layers.add(collect_layer)

    # Prompts for new generation (no responses), and full texts (with responses) for old-answer passes.
    gen_prompts = build_generation_prompts(filtered_df, args)
    full_texts = text_series.tolist()
    if args.max_samples is not None:
        gen_prompts = gen_prompts[: args.max_samples]
        full_texts = full_texts[: args.max_samples]

    # Use baseline answers from the sheet when available; otherwise, run baseline generation.
    baseline_logprobs = None
    # Pass 1: neutral interventions, no new tokens (old answers)
    neutral_generations, _, neutral_logprobs, neutral_entropy, neutral_token_ids, neutral_cos, neutral_token_preds = run_generation(
        model,
        tokenizer,
        prompts=full_texts,
        contexts_by_layer=contexts_by_layer,
        device=device,
        args=args,
        out_dir=out_dir,
        generate_new_tokens=False,
        hook_active=False,
    )

    # Pass 2: active interventions, no new tokens (old answers)
    active_generations, _, active_logprobs, active_entropy, active_token_ids, active_cos, active_token_preds = run_generation(
        model,
        tokenizer,
        prompts=full_texts,
        contexts_by_layer=contexts_by_layer,
        device=device,
        args=args,
        out_dir=out_dir,
        generate_new_tokens=False,
        hook_active=True,
    )

    # Pass 3: new generation without interventions
    baseline_generations, _, baseline_logprobs, baseline_entropy, baseline_token_ids, baseline_cos, baseline_token_preds = run_generation(
        model,
        tokenizer,
        prompts=gen_prompts,
        contexts_by_layer=contexts_by_layer,
        device=device,
        args=args,
        out_dir=out_dir,
        generate_new_tokens=True,
        hook_active=False,
    )

    # Pass 4: new generation with interventions
    generations, _, logprobs, entropy, token_ids, cos_records, intervened_token_preds = run_generation(
        model,
        tokenizer,
        prompts=gen_prompts,
        contexts_by_layer=contexts_by_layer,
        device=device,
        args=args,
        out_dir=out_dir,
        generate_new_tokens=True,
        hook_active=True,
    )

    all_contexts = [ctx for _, layer_contexts in sorted(contexts_by_layer.items(), key=lambda kv: kv[0]) for ctx in layer_contexts]

    save_token_stats(
        out_dir,
        tokenizer,
        prompts=full_texts,
        token_ids=neutral_token_ids,
        logprobs=neutral_logprobs,
        entropy=neutral_entropy,
        contexts=all_contexts,
        token_preds=neutral_token_preds,
        tag="token_stats_baseline_old",
        file_format=args.token_stats_format,
        include_token_text=args.token_stats_include_text,
        skip_special_tokens=args.token_stats_skip_special,
    )

    save_token_stats(
        out_dir,
        tokenizer,
        prompts=full_texts,
        token_ids=active_token_ids,
        logprobs=active_logprobs,
        entropy=active_entropy,
        contexts=all_contexts,
        token_preds=active_token_preds,
        tag="token_stats_intervention_old",
        file_format=args.token_stats_format,
        include_token_text=args.token_stats_include_text,
        skip_special_tokens=args.token_stats_skip_special,
    )

    save_token_stats(
        out_dir,
        tokenizer,
        prompts=gen_prompts,
        token_ids=baseline_token_ids,
        logprobs=baseline_logprobs,
        entropy=baseline_entropy,
        contexts=all_contexts,
        token_preds=baseline_token_preds,
        tag="token_stats_baseline_new",
        file_format=args.token_stats_format,
        include_token_text=args.token_stats_include_text,
        skip_special_tokens=args.token_stats_skip_special,
    )

    save_token_stats(
        out_dir,
        tokenizer,
        prompts=gen_prompts,
        token_ids=token_ids,
        logprobs=logprobs,
        entropy=entropy,
        contexts=all_contexts,
        token_preds=intervened_token_preds,
        tag="token_stats_intervention_new",
        file_format=args.token_stats_format,
        include_token_text=args.token_stats_include_text,
        skip_special_tokens=args.token_stats_skip_special,
    )

    save_generations(
        out_dir,
        gen_prompts,
        baseline_generations,
        generations,
        baseline_logprobs,
        logprobs,
        tag="intervention_new",
    )

    save_generations(
        out_dir,
        full_texts,
        [""] * len(full_texts),  # neutral has no new tokens
        neutral_generations,
        None,
        neutral_logprobs,
        tag="baseline_old",
    )

    save_generations(
        out_dir,
        full_texts,
        [""] * len(full_texts),  # active old answers
        active_generations,
        None,
        active_logprobs,
        tag="intervention_old",
    )

    save_generations(
        out_dir,
        gen_prompts,
        baseline_generations,
        baseline_generations,
        baseline_logprobs,
        baseline_logprobs,
        tag="baseline_new",
    )

    all_cos = cos_records + neutral_cos + active_cos + baseline_cos
    save_gradient_cosines(out_dir, all_cos)
    save_probe_analytics(out_dir, all_contexts)

    save_metadata(out_dir, args, feature_specs, sorted(layers), data_hash=data_hash)


if __name__ == "__main__":
    main()
