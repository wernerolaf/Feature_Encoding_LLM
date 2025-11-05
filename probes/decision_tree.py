from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from activation_standardizer import ActivationStandardizer
from probes.base import BaseProbe


class DecisionTreeProbe(BaseProbe):
    def __init__(
        self,
        *,
        standardizer: ActivationStandardizer | None = None,
        tree_kwargs: dict | None = None,
    ) -> None:
        super().__init__(standardizer=standardizer)
        cfg = dict(max_depth=None, random_state=0)
        if tree_kwargs:
            cfg.update(tree_kwargs)
        self.model = DecisionTreeClassifier(**cfg)
        self._explainer = None
        self._train_X: np.ndarray | None = None

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self._train_X = X
        try:
            import shap  # noqa: F401

            self._explainer = shap.TreeExplainer(self.model)
        except Exception as exc:
            self._explainer = exc  # remember the failure so we can surface it later

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _predict_proba_model(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def _compute_gradient(self, X: np.ndarray | None) -> np.ndarray:
        if self._explainer is None:
            raise RuntimeError(
                "SHAP TreeExplainer unavailable. Ensure the `shap` package is installed."
            )
        if isinstance(self._explainer, Exception):
            raise RuntimeError(
                f"Failed to initialize SHAP explainer: {self._explainer}"
            ) from self._explainer

        if X is None:
            if self._train_X is None:
                raise RuntimeError("No reference activations available for SHAP computation.")
            data = self._train_X
        else:
            data = np.asarray(X)

        shap_vals = self._explainer.shap_values(data)
        values = self._select_positive_class_shap(shap_vals)

        if X is None:
            return values.mean(axis=0)
        return values

        # ------------------------------------------------------------------ #
    def _select_positive_class_shap(self, shap_vals) -> np.ndarray:
        n_classes = getattr(self.model, "n_classes_", getattr(self.model, "classes_", [None]))
        if isinstance(n_classes, int):
            total_classes = n_classes
        elif isinstance(n_classes, np.ndarray):
            total_classes = n_classes.size
        else:
            total_classes = len(self.model.classes_)

        target_class = 1 if total_classes > 1 else 0
        values = shap_vals

        if isinstance(values, list):
            values = values[target_class]
        else:
            values = np.asarray(values)

        if values.ndim == 3:
            if values.shape[0] == total_classes:
                values = values[target_class]
            elif values.shape[1] == total_classes:
                values = values[:, target_class, :]
            elif values.shape[2] == total_classes:
                values = values[:, :, target_class]
            else:
                raise ValueError(
                    f"Unexpected SHAP value shape {values.shape}; "
                    f"cannot locate class axis of length {total_classes}."
                )

        if values.ndim == 1:
            values = values.reshape(1, -1)

        if values.ndim != 2:
            raise ValueError(f"SHAP values must be 2D after class selection; got {values.shape}.")

        expected_features = self._train_X.shape[1] if self._train_X is not None else values.shape[1]
        if values.shape[1] != expected_features:
            raise ValueError(
                f"SHAP values feature dimension {values.shape[1]} does not "
                f"match expected size {expected_features}."
            )
        return values
