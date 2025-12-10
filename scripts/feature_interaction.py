#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_standardizer import ActivationStandardizer
from probes import DecisionTreeProbe, LinearProbe, ShallowNNProbe
from scripts.train_probes import (
    PROMPT_TEMPLATE,
    build_text_series,
    resolve_device,
    sanitize_model_name,
    set_seeds,
)

InterventionMode = Literal["increase", "decrease", "project"]


@dataclass
class FeatureSpec:
    label_column: str
    layer: Optional[int] = None
    mode: InterventionMode = "increase"
    strength: float = 1.0


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
    if args.text_column and args.text_column in df.columns:
        series = df[args.text_column]
    else:
        if not args.prompt_template:
            raise ValueError("Provide --text-column or --prompt-template")

        def render(row: pd.Series) -> str:
            values = {field: row.get(field, "") for field in args.template_fields}
            return args.prompt_template.format_map(values)

        series = df.apply(render, axis=1)

    series = series.fillna("").astype(str).str.strip()
    mask = series != ""
    prompts = series.loc[mask].tolist()
    return prompts


@dataclass
class FeatureProbeContext:
    name: str
    probe: LinearProbe | DecisionTreeProbe | ShallowNNProbe
    mode: InterventionMode
    strength: float
    class_names: list[str]
    task: str
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
        specs.append(FeatureSpec(label_column=label_column, layer=layer_val, mode=mode, strength=strength))
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
) -> FeatureProbeContext:
    candidates = [
        Path(args.probe_dir)
        / f"{sanitize_model_name(args.model_name)}_layer{layer}_{args.probe_type}_{label}.pkl",
        Path(args.probe_dir) / f"pythia70m_L{layer}_{label}.pkl",  # legacy naming fallback
    ]
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
        mode=mode,
        strength=strength,
        class_names=class_names if class_names else (["value"] if task == "regression" else []),
        task=task,
        training_metrics=training_metrics,
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
        flat_np = flat.detach().cpu().numpy()

        total_adjustment = torch.zeros_like(flat)

        for ctx in self.contexts:
            raw_gradients = ctx.probe.get_gradient(flat_np, normalize=False)
            grad_source = "gradient"
            raw_norms = np.linalg.norm(raw_gradients, axis=1)
            ctx.analytics.append(
                {
                    "step": self.step,
                    "mode": ctx.mode,
                    "strength": ctx.strength,
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

            # Check that the raw gradient moves the positive class in the intended direction (diagnostics only).
            pre_scores = None
            try:
                pre_scores = ctx.probe.predict_proba(flat_np)
                pos_idx = 1 if pre_scores.ndim > 1 and pre_scores.shape[1] > 1 else 0
                step_flat = flat_np + 10*raw_gradients
                step_scores = ctx.probe.predict_proba(step_flat)
                delta_pos = float(step_scores[:, pos_idx].mean() - pre_scores[:, pos_idx].mean())
                desired_sign = 1.0 if ctx.mode == "increase" else -1.0
                tol = 1e-5  # treat tiny negatives as noise
                flipped = bool(ctx.mode in {"increase", "decrease"} and delta_pos * desired_sign < -tol)
                flip_snapshot = None
                if flipped:
                    snapshot_path = Path(self.output_dir) / f"direction_flip_layer{self.layer_idx}_step{self.step}_{ctx.name}.npy"
                    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(snapshot_path, flat_np)

                ctx.analytics.append(
                    {
                        "step": self.step,
                        "mode": ctx.mode,
                        "strength": ctx.strength,
                        "tag": "direction_check",
                        "source": grad_source,
                        "tokens_observed": int(raw_norms.shape[0]),
                        "delta_pos": delta_pos,
                        "flipped": flipped,
                    }
                )
            except Exception as e:
                # print(e)
                pre_scores = None

            if self.active:
                if ctx.mode == "increase":
                    adjustment = ctx.strength * grad_tensor
                elif ctx.mode == "decrease":
                    adjustment = -ctx.strength * grad_tensor
                else:
                    projection = (flat * grad_tensor).sum(dim=-1, keepdim=True)
                    adjustment = -ctx.strength * projection * grad_tensor
                total_adjustment += adjustment

            # pre-adjustment stats
            if pre_scores is None:
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
            adjusted_np = flat.detach().cpu().numpy()
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


def compute_logprobs_and_entropy(outputs, generated_ids: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
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
            logprobs.append(token_logprobs.cpu().numpy())
            entropies.append(entropy.cpu().numpy())
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

    logits = logits[:, : max_t - 1, :]
    target = target[:, 1:max_t]
    log_probs_full = torch.log_softmax(logits, dim=-1)
    probs_full = torch.softmax(logits, dim=-1)
    gathered = log_probs_full.gather(2, target.unsqueeze(-1)).squeeze(-1)  # [batch, max_t-1]
    entropy = -(probs_full * log_probs_full).sum(dim=-1)  # [batch, max_t-1]

    pad = torch.full((gathered.size(0), 1), float("nan"), device=gathered.device)
    pad_h = torch.full((entropy.size(0), 1), float("nan"), device=entropy.device)
    seq_logprobs = torch.cat([pad, gathered], dim=1)
    seq_entropy = torch.cat([pad_h, entropy], dim=1)

    if seq_logprobs.size(1) < generated_ids.size(1):
        tail = generated_ids.size(1) - seq_logprobs.size(1)
        pad_tail_lp = torch.full((seq_logprobs.size(0), tail), float("nan"), device=seq_logprobs.device)
        pad_tail_ent = torch.full((seq_entropy.size(0), tail), float("nan"), device=seq_entropy.device)
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
    generate_new_tokens: bool = True,
    hook_active: bool = True,
):
    hook = None
    cos_records: list[dict[str, object]] = []
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    if contexts_by_layer:
        for layer_idx, ctxs in contexts_by_layer.items():
            module = resolve_layer_module(model, layer_idx)
            intervention_hook = ProbeInterventionHook(ctxs, layer_idx=layer_idx, output_dir=Path(args.output_dir))
            intervention_hook.active = hook_active
            hooks.append(module.register_forward_hook(intervention_hook))

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generations: list[str] = []
    all_logprobs: list[np.ndarray] = []
    all_entropy: list[np.ndarray] = []
    all_token_ids: list[np.ndarray] = []
    last_outputs = None
    prompt_stats: list[dict[str, object]] = []
    per_token_preds: list[dict[str, list[float]]] = []
    batch_size = max(1, args.generation_batch_size)
    prompt_idx = 0
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
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
                last_outputs = outputs
                sequences = outputs.sequences
                generated_ids = sequences[:, enc.input_ids.shape[1] :]
            else:
                outputs = model(**enc, output_hidden_states=True)
                last_outputs = outputs
                generated_ids = enc.input_ids  # evaluate plausibility of existing sequence

        # If no new tokens, still capture the prompt (or empty generation)
        generations.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True) if generate_new_tokens else [""] * enc.input_ids.size(0))
        lp, ent = compute_logprobs_and_entropy(outputs, generated_ids)
        all_logprobs.append(lp)
        all_entropy.append(ent)
        # per-prompt stats for this batch
        for b in range(lp.shape[0]):
            mask = ~np.isnan(lp[b])
            tokens = int(mask.sum())
            mean_lp = float(np.nanmean(lp[b])) if tokens else ""
            mean_prob = float(np.nanmean(np.exp(lp[b][mask]))) if tokens else ""
            mean_ent = float(np.nanmean(ent[b])) if tokens else ""
            prompt_stats.append(
                {
                    "prompt_index": prompt_idx,
                    "mean_logprob": mean_lp,
                    "mean_prob": mean_prob,
                    "mean_entropy": mean_ent,
                    "tokens": tokens,
                }
            )
            prompt_idx += 1
        all_token_ids.extend(list(generated_ids.cpu().numpy()))
        if contexts_by_layer:
            # compute gradient cosines on this batch of hidden states
            with torch.no_grad():
                if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                    hidden_states = outputs.hidden_states
                else:
                    enc_for_hidden = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
                    hidden_states = model(**enc_for_hidden, output_hidden_states=True).hidden_states
                    attention_mask = enc_for_hidden["attention_mask"]
                if 'attention_mask' not in locals():
                    attention_mask = enc.get("attention_mask")
                # align mask and hidden length defensively
                seq_len = min(attention_mask.shape[1], hidden_states[0].shape[1])
                attention_mask = attention_mask[:, :seq_len]
                total_layers = len(hidden_states)
                for layer_idx, ctxs in contexts_by_layer.items():
                    # hidden_states includes embeddings at position 0; shift positives by +1
                    idx = layer_idx + 1 if layer_idx >= 0 else total_layers + layer_idx
                    if idx < 0 or idx >= total_layers:
                        raise IndexError(f"Requested layer index {layer_idx} maps to hidden_states[{idx}] out of range 0..{total_layers-1}")
                    layer_states = hidden_states[idx][:, :seq_len, :]
                    valid_mask = attention_mask.bool()
                    flat = layer_states[valid_mask].detach().cpu().numpy()
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
                        while len(per_token_preds) < prompt_idx:
                            per_token_preds.append({})
                        for bi in range(batch_size_cur):
                            idx_global = start + bi
                            if idx_global >= len(per_token_preds):
                                per_token_preds.append({})
                            entry = per_token_preds[idx_global]
                            entry[pred_key] = filled[bi].tolist()

    for h in hooks:
        h.remove()

    def pad_and_concat(arrs: list[np.ndarray]) -> tuple[np.ndarray | None, list[np.ndarray]]:
        if not arrs:
            return None, []
        max_cols = max(arr.shape[1] for arr in arrs)
        padded = []
        for arr in arrs:
            if arr.shape[1] < max_cols:
                pad_width = ((0, 0), (0, max_cols - arr.shape[1]))
                arr = np.pad(arr, pad_width, constant_values=np.nan)
            padded.append(arr)
        return np.concatenate(padded, axis=0), padded

    logprobs_concat, logprobs_padded = pad_and_concat(all_logprobs)
    entropy_concat, entropy_padded = pad_and_concat(all_entropy)
    # ensure per_token_preds aligns with prompt count
    while len(per_token_preds) < len(prompts):
        per_token_preds.append({})

    return (
        generations,
        last_outputs,
        logprobs_concat,
        entropy_concat,
        all_token_ids,
        cos_records,
        logprobs_padded,
        entropy_padded,
        prompt_stats,
        per_token_preds,
    )


def save_token_stats(
    output_dir: Path,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    intervened_token_ids: Optional[list[np.ndarray]] = None,
    intervened_logprobs: Optional[np.ndarray] = None,
    intervened_entropy: Optional[np.ndarray] = None,
    baseline_token_ids: Optional[list[np.ndarray]] = None,
    baseline_logprobs: Optional[np.ndarray] = None,
    baseline_entropy: Optional[np.ndarray] = None,
    contexts: Optional[list[FeatureProbeContext]] = None,
    token_preds: Optional[list[dict[str, list[float]]]] = None,
    tag: str = "token_stats",
    column_suffix: str = "",
) -> None:
    token_ids = intervened_token_ids or baseline_token_ids
    logprob_arr = intervened_logprobs if intervened_logprobs is not None else baseline_logprobs
    entropy_arr = intervened_entropy if intervened_entropy is not None else baseline_entropy

    if token_ids is None or logprob_arr is None:
        return
    class_stats_summary = {}
    if contexts:
        class_stats_summary = {
            ctx.name: {
                "mode": ctx.mode,
                "strength": ctx.strength,
                "training_metrics": ctx.training_metrics,
                "records": ctx.analytics,
                "task": ctx.task,
            }
            for ctx in contexts
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    token_stats_path = output_dir / f"{tag}.csv"
    with token_stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        feature_names = sorted({f"{ctx.name}{column_suffix}" for ctx in contexts}) if contexts else []
        if not feature_names and token_preds:
            for entry in token_preds:
                feature_names.extend(entry.keys())
            feature_names = sorted(set(feature_names))
        header = ["prompt_index", "token_index", "token", "logprob", "entropy"] + [f"pred_{name}" for name in feature_names]
        writer.writerow(header)
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        special_tokens = set(tokenizer.all_special_tokens)
        for i, (prompt, ids) in enumerate(zip(prompts, token_ids)):
            lp_row = logprob_arr[i] if logprob_arr is not None and i < logprob_arr.shape[0] else None
            ent_row = entropy_arr[i] if entropy_arr is not None and i < entropy_arr.shape[0] else None

            max_len = len(ids)
            if lp_row is not None:
                max_len = min(max_len, lp_row.shape[0])
            if ent_row is not None:
                max_len = min(max_len, ent_row.shape[0])

            for j in range(max_len):
                token_id = int(ids[j])
                token = tokenizer.convert_ids_to_tokens([token_id])[0]
                if (eos_id is not None and token_id == eos_id) or (pad_id is not None and token_id == pad_id) or token in special_tokens:
                    continue
                lp_val = float(lp_row[j]) if lp_row is not None and j < len(lp_row) else ""
                ent_val = float(ent_row[j]) if ent_row is not None and j < len(ent_row) else ""
                row = [i, j, token, lp_val, ent_val]
                if feature_names:
                    preds_map = token_preds[i] if token_preds and i < len(token_preds) else {}
                    for name in feature_names:
                        vals = preds_map.get(name)
                        val = vals[j] if vals and j < len(vals) else ""
                        row.append(val)
                writer.writerow(row)


def save_probe_analytics(
    output_dir: Path,
    contexts_by_layer: dict[int, list[FeatureProbeContext]],
) -> None:
    if not contexts_by_layer:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, object]] = []
    for layer_idx, ctxs in sorted(contexts_by_layer.items(), key=lambda kv: kv[0]):
        for ctx in ctxs:
            payload.append(
                {
                    "layer": layer_idx,
                    "name": ctx.name,
                    "mode": ctx.mode,
                    "strength": ctx.strength,
                    "task": ctx.task,
                    "class_names": ctx.class_names,
                    "training_metrics": ctx.training_metrics,
                    "records": ctx.analytics,
                }
            )
    path = output_dir / "probe_analytics.json"
    path.write_text(json.dumps(payload, indent=2))


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


def save_logprob_batches(
    output_dir: Path,
    stats: list[dict[str, object]],
    tag: str,
) -> None:
    if not stats:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"logprob_stats_{tag}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_index", "mean_logprob", "mean_prob", "mean_entropy", "tokens"])
        for row in stats:
            writer.writerow(
                [
                    row.get("prompt_index", ""),
                    row.get("mean_logprob", ""),
                    row.get("mean_prob", ""),
                    row.get("mean_entropy", ""),
                    row.get("tokens", ""),
                ]
            )


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


def save_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    feature_specs: list[FeatureSpec],
    layers: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_name": args.model_name,
        "layers": layers,
        "probe_dir": args.probe_dir,
        "probe_type": args.probe_type,
        "data_path": args.data_path,
        "sheet": args.sheet,
        "features": [spec.__dict__ for spec in feature_specs],
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature interaction experiment with probe interventions.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sheet", required=True)
    parser.add_argument("--layer", type=int, default=None, help="Optional default layer applied to any feature without an explicit layer.")
    parser.add_argument("--probe-dir", default="artifacts/probes", help="Directory containing pre-trained probe artifacts.")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--prompt-template", default=PROMPT_TEMPLATE)
    parser.add_argument("--template-fields", nargs="+", default=["gender", "level", "trait", "belief", "question", "type", "pronoun"])
    parser.add_argument("--response-column", default="response")
    parser.add_argument("--feature", dest="feature_specs", action="append", required=True,
                        help="Format: label_column[:mode[:strength]]. Mode ∈ {increase,decrease,project}.")
    parser.add_argument("--probe-type", choices=["linear", "decision_tree", "shallow_nn"], default="linear",
                        help="Probe type used in artifact naming (no training happens here).")
    parser.add_argument("--output-dir", default="artifacts/experiments")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    df = pd.read_excel(args.data_path, sheet_name=args.sheet)
    text_series, filtered_df = build_text_series(df, args)  # includes responses when available

    feature_specs = parse_feature_specs(args.feature_specs)
    contexts_by_layer: dict[int, list[FeatureProbeContext]] = {}
    layers: set[int] = set()

    for spec in feature_specs:
        resolved_layer = spec.layer if spec.layer is not None else args.layer
        if resolved_layer is None:
            raise ValueError("No layer specified. Provide --layer or annotate each --feature with @<layer>.")
        layers.add(resolved_layer)
        print(f"[load] feature={spec.label_column} mode={spec.mode} strength={spec.strength}")
        context = load_probe_artifact(
            spec.label_column,
            args,
            device=device,
            layer=resolved_layer,
            mode=spec.mode,
            strength=spec.strength,
        )
        contexts_by_layer.setdefault(resolved_layer, []).append(context)

    # Prompts for new generation (no responses), and full texts (with responses) for old-answer passes.
    gen_prompts = build_generation_prompts(filtered_df, args)
    full_texts = text_series.tolist()
    if args.max_samples is not None:
        gen_prompts = gen_prompts[: args.max_samples]
        full_texts = full_texts[: args.max_samples]

    # Use baseline answers from the sheet when available; otherwise, run baseline generation.
    baseline_logprobs = None
    # Pass 1: neutral interventions, no new tokens (old answers)
    neutral_generations, _, neutral_logprobs, neutral_entropy, neutral_token_ids, neutral_cos, neutral_lp_batches, neutral_ent_batches, neutral_stats, neutral_token_preds = run_generation(
        model,
        tokenizer,
        prompts=full_texts,
        contexts_by_layer=contexts_by_layer,
        device=device,
        args=args,
        generate_new_tokens=False,
        hook_active=False,
    )

    # Pass 2: active interventions, no new tokens (old answers)
    active_generations, _, active_logprobs, active_entropy, active_token_ids, active_cos, active_lp_batches, active_ent_batches, active_stats, active_token_preds = run_generation(
        model,
        tokenizer,
        prompts=full_texts,
        contexts_by_layer=contexts_by_layer,
        device=device,
        args=args,
        generate_new_tokens=False,
        hook_active=True,
    )

    # Pass 3: new generation without interventions
    baseline_generations, _, baseline_logprobs, baseline_entropy, baseline_token_ids, baseline_cos, baseline_lp_batches, baseline_ent_batches, baseline_stats, baseline_token_preds = run_generation(
        model,
        tokenizer,
        prompts=gen_prompts,
        contexts_by_layer=contexts_by_layer,
        device=device,
        args=args,
        generate_new_tokens=True,
        hook_active=False,
    )

    # Pass 4: new generation with interventions
    generations, _, logprobs, entropy, token_ids, cos_records, intervened_lp_batches, intervened_ent_batches, intervened_stats, intervened_token_preds = run_generation(
        model,
        tokenizer,
        prompts=gen_prompts,
        contexts_by_layer=contexts_by_layer,
        device=device,
        args=args,
        generate_new_tokens=True,
        hook_active=True,
    )

    out_dir = Path(args.output_dir)
    layer_items = sorted(contexts_by_layer.items(), key=lambda kv: kv[0])
    for layer_idx, layer_contexts in layer_items:
        suffix = f"_L{layer_idx}"
        save_token_stats(
            out_dir,
            tokenizer,
            prompts=full_texts,
            intervened_token_ids=neutral_token_ids,
            intervened_logprobs=neutral_logprobs,
            intervened_entropy=neutral_entropy,
            baseline_token_ids=None,
            baseline_logprobs=None,
            baseline_entropy=None,
            contexts=layer_contexts,
            token_preds=neutral_token_preds,
            tag=f"token_stats_baseline_old{suffix}",
            column_suffix=suffix,
        )

        save_token_stats(
            out_dir,
            tokenizer,
            prompts=full_texts,
            intervened_token_ids=active_token_ids,
            intervened_logprobs=active_logprobs,
            intervened_entropy=active_entropy,
            baseline_token_ids=None,
            baseline_logprobs=None,
            baseline_entropy=None,
            contexts=layer_contexts,
            token_preds=active_token_preds,
            tag=f"token_stats_intervention_old{suffix}",
            column_suffix=suffix,
        )

        save_token_stats(
            out_dir,
            tokenizer,
            prompts=gen_prompts,
            intervened_token_ids=baseline_token_ids,
            intervened_logprobs=baseline_logprobs,
            intervened_entropy=baseline_entropy,
            baseline_token_ids=None,
            baseline_logprobs=None,
            baseline_entropy=None,
            contexts=layer_contexts,
            token_preds=baseline_token_preds,
            tag=f"token_stats_baseline_new{suffix}",
            column_suffix=suffix,
        )

        save_token_stats(
            out_dir,
            tokenizer,
            prompts=gen_prompts,
            intervened_token_ids=token_ids,
            intervened_logprobs=logprobs,
            intervened_entropy=entropy,
            baseline_token_ids=baseline_token_ids,
            baseline_logprobs=baseline_logprobs,
            baseline_entropy=baseline_entropy,
            contexts=layer_contexts,
            token_preds=intervened_token_preds,
            tag=f"token_stats_intervention_new{suffix}",
            column_suffix=suffix,
        )

    save_generations(
        Path(args.output_dir),
        gen_prompts,
        baseline_generations,
        generations,
        baseline_logprobs,
        logprobs,
        tag="intervention_new",
    )
    save_logprob_batches(Path(args.output_dir), intervened_stats, tag="intervention_new")
    save_generations(
        Path(args.output_dir),
        full_texts,
        [""] * len(full_texts),  # neutral has no new tokens
        neutral_generations,
        None,
        neutral_logprobs,
        tag="baseline_old",
    )
    save_logprob_batches(Path(args.output_dir), neutral_stats, tag="baseline_old")
    save_generations(
        Path(args.output_dir),
        full_texts,
        [""] * len(full_texts),  # active old answers
        active_generations,
        None,
        active_logprobs,
        tag="intervention_old",
    )
    save_logprob_batches(Path(args.output_dir), active_stats, tag="intervention_old")
    save_generations(
        Path(args.output_dir),
        gen_prompts,
        baseline_generations,
        baseline_generations,
        baseline_logprobs,
        baseline_logprobs,
        tag="baseline_new",
    )
    save_logprob_batches(Path(args.output_dir), baseline_stats, tag="baseline_new")
    all_cos = cos_records + neutral_cos + active_cos + baseline_cos
    save_gradient_cosines(Path(args.output_dir), all_cos)
    save_probe_analytics(Path(args.output_dir), contexts_by_layer)
    save_metadata(Path(args.output_dir), args, feature_specs, sorted(layers))
    for prompt, generation in zip(gen_prompts, generations):
        print("=" * 80)
        print(prompt)
        print("-" * 80)
        print(generation)


if __name__ == "__main__":
    main()
