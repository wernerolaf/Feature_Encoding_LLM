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
        task: str = "classification",
    ) -> None:
        super().__init__(standardizer=standardizer)
        task_normalized = task.lower()
        if task_normalized not in {"classification", "regression"}:
            raise ValueError("ShallowNNProbe task must be 'classification' or 'regression'.")
        self.task = task_normalized
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
        self._pin_memory = self.device.type == "cuda"

        self.model: Optional[nn.Module] = None
        if self.task == "classification":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()
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
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, pin_memory=self._pin_memory)

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
                batch_X = batch_X.to(self.device, non_blocking=self._pin_memory)
                batch_y = batch_y.to(self.device, non_blocking=self._pin_memory)
                optimizer.zero_grad()
                logits = self.model(batch_X)
                if self.task == "classification":
                    loss = self.loss_fn(logits, batch_y)
                else:
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
        if self.task == "classification":
            probs = self._predict_proba_model(X)[:, 1]
            return (probs >= 0.5).astype(int)

        tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor).cpu().numpy().reshape(-1)
        return outputs

    def _predict_proba_model(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise RuntimeError("predict_proba is not available for regression probes.")
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
            X_work = np.zeros((1, self.model[0].in_features), dtype=np.float32)
            squeeze = True
        else:
            X_work = np.asarray(X, dtype=np.float32)
            if X_work.ndim == 1:
                X_work = X_work.reshape(1, -1)
                squeeze = True
            elif X_work.ndim == 2:
                squeeze = False
            else:
                raise ValueError("Expected activations to be 1D or 2D.")

        grads: list[np.ndarray] = []
        for start in range(0, X_work.shape[0], self.batch_size):
            chunk = torch.from_numpy(X_work[start : start + self.batch_size]).to(
                self.device,
                non_blocking=self._pin_memory,
            )
            chunk.requires_grad_(True)
            logits = self.model(chunk).reshape(-1)
            if self.task == "classification":
                score = torch.sigmoid(logits)
            else:
                score = logits
            grad = torch.autograd.grad(score.sum(), chunk, retain_graph=False, create_graph=False)[0]
            grads.append(grad.detach().cpu().numpy())

        grad_arr = (
            np.concatenate(grads, axis=0)
            if grads
            else np.empty((0, X_work.shape[1]), dtype=np.float32)
        )
        return grad_arr[0] if squeeze else grad_arr
