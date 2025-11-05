from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np

from activation_standardizer import ActivationStandardizer


ArrayLike = Union[np.ndarray, list]


class BaseProbe(ABC):
    def __init__(
        self,
        *,
        standardizer: Optional[ActivationStandardizer] = None,
    ) -> None:
        self.standardizer = standardizer or ActivationStandardizer(strategy="identity")
        self._fitted = False

    # ------------------------------------------------------------------ #
    def fit(self, X: ArrayLike, y: ArrayLike) -> "BaseProbe":
        X_arr = self._to_numpy(X)
        y_arr = np.asarray(y)

        X_prepped = self.standardizer.fit_transform(X_arr)
        self._fit_model(X_prepped, y_arr)
        self._fitted = True
        return self

    def predict(self, X: ArrayLike):
        self._ensure_fitted()
        X_prepped, squeeze = self._prepare_inference_input(X)
        preds = self._predict_model(X_prepped)
        return preds[0] if squeeze else preds

    def predict_proba(self, X: ArrayLike):
        self._ensure_fitted()
        X_prepped, squeeze = self._prepare_inference_input(X)
        probs = self._predict_proba_model(X_prepped)
        return probs[0] if squeeze else probs

    def get_gradient(
        self,
        X: Optional[ArrayLike] = None,
        *,
        normalize: bool = True,
    ):
        self._ensure_fitted()
        X_prepped: Optional[np.ndarray]
        squeeze = False
        if X is None:
            X_prepped = None
        else:
            X_prepped, squeeze = self._prepare_inference_input(X)

        grad = self._compute_gradient(X_prepped)
        grad = self.standardizer.direction_to_input_space(grad)

        grad = np.asarray(grad)
        if normalize:
            if grad.ndim == 1:
                norm = np.linalg.norm(grad) + 1e-12
                grad = grad / norm
            else:
                norms = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
                grad = grad / norms

        return grad[0] if squeeze and grad.ndim > 1 else grad

    def intervine(
        self,
        X: ArrayLike,
        intervention_type: str,
        *,
        strength: float = 1.0,
    ) -> np.ndarray:
        self._ensure_fitted()
        base, squeeze = self._prepare_inference_input(X, skip_standardizer=True)
        direction = self.get_gradient(X, normalize=True)

        proj = np.sum(base * direction, axis=-1, keepdims=True)
        if intervention_type == "neutralize":
            adjusted = base - strength * proj * direction
        elif intervention_type == "amplify":
            adjusted = base + strength * direction
        elif intervention_type == "dampen":
            adjusted = base - strength * direction
        else:
            raise ValueError(f"Unknown intervention_type: {intervention_type}")

        return adjusted[0] if squeeze else adjusted

    # ------------------------------------------------------------------ #
    @abstractmethod
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def _predict_model(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def _predict_proba_model(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def _compute_gradient(self, X: Optional[np.ndarray]) -> np.ndarray: ...

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_numpy(X: ArrayLike) -> np.ndarray:
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("Expected activations to be 2D.")
        return arr

    def _prepare_inference_input(
        self,
        X: ArrayLike,
        *,
        skip_standardizer: bool = False,
    ) -> Tuple[np.ndarray, bool]:
        arr = np.asarray(X)
        squeeze = False
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
            squeeze = True
        if arr.ndim != 2:
            raise ValueError("Expected activations to be 2D.")

        if skip_standardizer:
            return arr, squeeze

        return self.standardizer.transform(arr), squeeze

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Probe not fitted yet.")
