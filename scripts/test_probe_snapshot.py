#!/usr/bin/env python3
"""
Quick diagnostic: load a probe artifact and evaluate predictions/gradient direction
on a provided activation snapshot.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def load_artifact(path: Path) -> Dict[str, Any]:
    try:
        import joblib  # type: ignore
    except ImportError:
        joblib = None

    if joblib is not None:
        return joblib.load(path)

    with path.open("rb") as f:
        return pickle.load(f)


def resolve_class_names(artifact: Dict[str, Any]) -> List[str]:
    classes = artifact.get("classes")
    if classes is None:
        le = artifact.get("label_encoder")
        if le is not None and hasattr(le, "classes_"):
            classes = le.classes_
    if classes is None:
        return []
    return [str(c) for c in classes]


def parse_snapshot(arg: str) -> np.ndarray:
    """
    Accept either a path to JSON/NPY or an inline JSON list string.
    """
    path = Path(arg)
    if path.exists():
        if path.suffix.lower() == ".npy":
            return np.load(path)
        payload = json.loads(path.read_text())
    else:
        payload = json.loads(arg)
    arr = np.asarray(payload, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser(description="Test a probe on a captured activation snapshot.")
    ap.add_argument("--artifact", required=True, help="Path to probe artifact (.pkl).")
    ap.add_argument("--snapshot", required=True, help="Path to JSON/NPY or inline JSON of activations.")
    ap.add_argument("--epsilon", type=float, default=1e-3, help="Step size along raw gradient for the base step.")
    ap.add_argument("--max-step", type=float, default=1.0, help="Maximum multiplier for the gradient sweep.")
    ap.add_argument("--num-points", type=int, default=21, help="Number of points in the gradient sweep plot.")
    ap.add_argument("--plot-out", default=None, help="Optional path to save a probability vs step plot (png).")
    args = ap.parse_args()

    artifact = load_artifact(Path(args.artifact))
    probe = artifact.get("probe")
    if probe is None:
        raise SystemExit("Artifact missing probe.")
    # restore standardizer (including autoencoder) if present
    std_meta = artifact.get("standardizer")
    if std_meta:
        try:
            from activation_standardizer import ActivationStandardizer
        except Exception as exc:
            raise SystemExit(f"Could not import ActivationStandardizer: {exc}")
        strategy = std_meta.get("strategy", "identity")
        std = ActivationStandardizer(strategy=strategy)
        if strategy == "standard":
            scaler_state = std_meta.get("scaler_state")
            if scaler_state is not None:
                std._scaler = scaler_state  # type: ignore[attr-defined]
        elif strategy == "autoencoder":
            auto_path = std_meta.get("autoencoder_artifact")
            if auto_path:
                std.load_autoencoder_artifact(auto_path)
        probe.standardizer = std

    classes = resolve_class_names(artifact)
    snapshot = parse_snapshot(args.snapshot)

    try:
        base_probs = probe.predict_proba(snapshot)
    except Exception as exc:
        raise SystemExit(f"predict_proba failed: {exc}")

    # use raw (unnormalized) gradient for direction check
    grad = probe.get_gradient(snapshot, normalize=False)
    step = snapshot + args.epsilon * grad
    try:
        step_probs = probe.predict_proba(step)
    except Exception as exc:
        raise SystemExit(f"predict_proba on stepped activations failed: {exc}")

    pos_idx = 1 if base_probs.ndim > 1 and base_probs.shape[1] > 1 else 0
    delta_pos = float(np.mean(step_probs[:, pos_idx] - base_probs[:, pos_idx]))

    print("Classes:", classes or "<unknown order>")
    print("Pos class index:", pos_idx)
    print("Base probs (mean over batch):", np.mean(base_probs, axis=0))
    print("Step probs (mean over batch):", np.mean(step_probs, axis=0))
    print("Delta pos (mean):", delta_pos)
    grad_norms = np.linalg.norm(grad, axis=1)
    base_norms = np.linalg.norm(snapshot, axis=1)
    print("Grad norm stats:", {
        "mean": float(grad_norms.mean()),
        "min": float(grad_norms.min()),
        "max": float(grad_norms.max()),
    })
    print("Base norm stats:", {
        "mean": float(base_norms.mean()),
        "min": float(base_norms.min()),
        "max": float(base_norms.max()),
    })

    # Sweep along the gradient direction to visualize monotonicity.
    multipliers = np.linspace(-args.max_step, args.max_step, args.num_points)
    pos_curve = []
    for m in multipliers:
        swept = snapshot + m * args.epsilon * grad
        probs = probe.predict_proba(swept)
        pos_curve.append(float(np.mean(probs[:, pos_idx])))

    print("Sweep multipliers:", multipliers.tolist())
    print("Pos probs along sweep:", pos_curve)

    if args.plot_out:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(multipliers, pos_curve, marker="o")
        ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel("gradient multiplier (epsilon-scaled)")
        ax.set_ylabel(f"P(class {pos_idx}) mean")
        ax.set_title("Probe probability along gradient direction")
        ax.grid(alpha=0.3)
        out_path = Path(args.plot_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
