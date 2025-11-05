from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from activation_standardizer import ActivationStandardizer
from probes.base import BaseProbe


class ShallowNNProbe(BaseProbe):
    def __init__(
        self,
        *,
        standardizer: ActivationStandardizer | None = None,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(standardizer=standardizer)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[nn.Module] = None
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _build_model(self, input_dim: int) -> None:
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        ).to(self.device)

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y.astype(np.float32)).float().unsqueeze(-1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        if self.model is None:
            self._build_model(X_tensor.shape[1])

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.model.train()

        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = self.loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

        self.model.eval()

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        probs = self._predict_proba_model(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def _predict_proba_model(self, X: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.concatenate([1 - probs, probs], axis=1)

    def _compute_gradient(self, X: np.ndarray | None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not built.")
        self.model.eval()

        if X is None:
            return self._gradient_at_point(self.model, np.zeros(self.model[0].in_features))

        return np.vstack([self._gradient_at_point(self.model, x) for x in X])

    def _gradient_at_point(self, model: nn.Module, x: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(x.astype(np.float32)).to(self.device)
        tensor.requires_grad_(True)

        logits = model(tensor)
        score = torch.sigmoid(logits)
        score.backward()
        grad = tensor.grad.detach().cpu().numpy()
        return grad
