from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from activation_standardizer import ActivationStandardizer
from probes.base import BaseProbe


class LinearProbe(BaseProbe):
    def __init__(
        self,
        *,
        standardizer: ActivationStandardizer | None = None,
        logistic_kwargs: dict | None = None,
    ) -> None:
        super().__init__(standardizer=standardizer)
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

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _predict_proba_model(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def _compute_gradient(self, X: np.ndarray | None) -> np.ndarray:
        coef = self.model.coef_[0]
        if X is None:
            return coef

        probs = self.model.predict_proba(X)[:, 1]
        factors = probs * (1.0 - probs)
        return (factors[:, None] * coef[None, :])
