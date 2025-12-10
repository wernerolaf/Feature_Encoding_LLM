from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from activation_standardizer import ActivationStandardizer
from probes.base import BaseProbe


class LinearProbe(BaseProbe):
    def __init__(
        self,
        *,
        standardizer: ActivationStandardizer | None = None,
        logistic_kwargs: dict | None = None,
        regression_kwargs: dict | None = None,
        task: str = "classification",
    ) -> None:
        super().__init__(standardizer=standardizer)
        task_normalized = task.lower()
        if task_normalized not in {"classification", "regression"}:
            raise ValueError("LinearProbe task must be 'classification' or 'regression'.")

        self.task = task_normalized

        if self.task == "classification":
            cfg = dict(
                penalty="l1",
                C=1.0,
                solver="saga",
                max_iter=2000,
                random_state=0,
            )
            if logistic_kwargs:
                cfg.update(logistic_kwargs)
            self.model = LogisticRegression(**cfg)
        else:
            reg_cfg = dict(alpha=1.0, random_state=0)
            if regression_kwargs:
                reg_cfg.update(regression_kwargs)
            self.model = Ridge(**reg_cfg)

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _predict_proba_model(self, X: np.ndarray) -> np.ndarray:
        if self.task == "regression":
            raise RuntimeError("predict_proba is not available for regression probes.")
        return self.model.predict_proba(X)

    def _compute_gradient(self, X: np.ndarray | None) -> np.ndarray:
        coef = self.model.coef_.ravel()
        if self.task == "regression":
            if X is None:
                return coef
            return np.tile(coef, (X.shape[0], 1))

        # classification: gradient depends on the probability mass at the positive class
        probs = self.model.predict_proba(X)[:, 1]
        factors = probs * (1.0 - probs)
        return (factors[:, None] * coef[None, :])
