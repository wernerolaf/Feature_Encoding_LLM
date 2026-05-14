from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch


def transformer_layers(model: torch.nn.Module) -> torch.nn.ModuleList | list[torch.nn.Module]:
    """Return the model's decoder/block stack for common causal LM families."""
    candidates = []
    base = getattr(model, "base_model", model)
    inner = getattr(base, "model", base)
    for obj in (model, base, inner, getattr(inner, "model", None), getattr(inner, "transformer", None)):
        if obj is None:
            continue
        for attr in ("layers", "h", "block"):
            stack = getattr(obj, attr, None)
            if stack is not None and hasattr(stack, "__len__"):
                candidates.append(stack)
        decoder = getattr(obj, "decoder", None)
        if decoder is not None:
            stack = getattr(decoder, "layers", None)
            if stack is not None and hasattr(stack, "__len__"):
                candidates.append(stack)
    if candidates:
        return candidates[0]
    raise ValueError(f"Unable to locate transformer layers for model type {getattr(model.config, 'model_type', '<unknown>')!r}.")


def resolve_layer_index(model: torch.nn.Module, layer: int) -> int:
    layers = transformer_layers(model)
    resolved = layer if layer >= 0 else len(layers) + layer
    if resolved < 0 or resolved >= len(layers):
        raise ValueError(f"Layer {layer} resolves to {resolved}, outside valid range 0..{len(layers) - 1}.")
    return resolved


def freeze_except_transformer_layer(model: torch.nn.Module, layer: int) -> int:
    """Freeze all parameters except one transformer block."""
    resolved = resolve_layer_index(model, layer)
    for param in model.parameters():
        param.requires_grad = False
    for param in transformer_layers(model)[resolved].parameters():
        param.requires_grad = True
    return resolved


def freeze_peft_adapters_except_layer(model: torch.nn.Module, layer: int) -> int:
    """Keep only adapter parameters attached to one transformer block trainable."""
    resolved = resolve_layer_index(model, layer)
    markers = tuple(_layer_name_markers(resolved))
    for name, param in model.named_parameters():
        if "lora_" in name or "adapter" in name:
            param.requires_grad = any(marker in name for marker in markers)
        else:
            param.requires_grad = False
    return resolved


def load_peft_adapter_if_requested(model: torch.nn.Module, adapter_path: str | None, *, is_trainable: bool = False) -> torch.nn.Module:
    if not adapter_path:
        return model
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(Path(adapter_path)), is_trainable=is_trainable)


def trainable_parameter_summary(model: torch.nn.Module) -> dict[str, int | float]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    pct = (100.0 * trainable / total) if total else 0.0
    return {"trainable": trainable, "total": total, "pct": pct}


def _layer_name_markers(layer: int) -> Iterable[str]:
    yield f".layers.{layer}."
    yield f".h.{layer}."
    yield f".block.{layer}."
    yield f".decoder.layers.{layer}."
