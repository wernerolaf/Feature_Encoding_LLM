from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from activation_standardizer import ActivationStandardizer
from probes.base import BaseProbe

from torch.utils.tensorboard import SummaryWriter

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
        log_dir: Optional[str] = None,
        log_interval: int = 10,
        track_history: bool = False,
    ) -> None:
        super().__init__(standardizer=standardizer)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_interval = max(1, log_interval)
        self.track_history = track_history

        self.model: Optional[nn.Module] = None
        self.loss_fn = nn.BCEWithLogitsLoss()
        self._history: list[dict[str, float]] = []

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
        writer = None
        if self.log_dir:
            if SummaryWriter is None:
                raise RuntimeError(
                    "TensorBoard logging requested for ShallowNNProbe, but torch.utils.tensorboard is unavailable. "
                    "Install the 'tensorboard' package to enable logging."
                )
            run_dir = self.log_dir / f"run_{int(time.time())}"
            run_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(run_dir))

        track_history = self.track_history
        history_records: list[dict[str, float]] | None = [] if track_history else None
        global_step = 0
        total_samples = len(dataset)

        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_index, (batch_X, batch_y) in enumerate(loader):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = self.loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

                loss_value = float(loss.item())
                epoch_loss += loss_value * batch_X.size(0)

                if history_records is not None:
                    history_records.append(
                        {"epoch": epoch, "batch": batch_index, "loss": loss_value}
                    )
                if writer and global_step % self.log_interval == 0:
                    writer.add_scalar("probe/batch_loss", loss_value, global_step)
                global_step += 1

            avg_loss = epoch_loss / total_samples if total_samples else float("nan")
            if history_records is not None:
                history_records.append({"epoch": epoch, "batch": -1, "loss": float(avg_loss)})
            if writer:
                writer.add_scalar("probe/epoch_loss", avg_loss, epoch)

        if writer:
            writer.flush()
            writer.close()

        self.model.eval()
        self._history = history_records or []

    def get_history(self) -> list[dict[str, float]]:
        return list(self._history)

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
