from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Optional, Union

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.tensorboard import SummaryWriter


class ActivationStandardizer:
    """
    Strategy object that normalizes activations before probing and
    can project probe directions back into the original space.
    Supported strategies:
      - "identity": passthrough
      - "standard": sklearn StandardScaler
      - "autoencoder": sparse autoencoder (produces latent features)
    """

    def __init__(
        self,
        *,
        strategy: Literal["identity", "standard", "autoencoder"] = "standard",
        scaler_kwargs: Optional[dict] = None,
        autoencoder_config: Optional["AutoEncoderConfig"] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.strategy = strategy
        self.scaler_kwargs = scaler_kwargs or dict(with_mean=True, with_std=True)
        self.autoencoder_config = autoencoder_config or AutoEncoderConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._scaler: Optional[StandardScaler] = None
        self._autoencoder: Optional[SparseAutoencoder] = None
        self._input_dim: Optional[int] = None
        self._latent_dim: Optional[int] = None
        self._autoencoder_history: list[dict[str, float]] = []

    def fit(self, X: np.ndarray) -> "ActivationStandardizer":
        X = self._require_2d(X)
        self._input_dim = X.shape[1]

        if self.strategy == "identity":
            return self

        if self.strategy == "standard":
            self._scaler = StandardScaler(**self.scaler_kwargs).fit(X)
            return self

        if self.strategy == "autoencoder":
            if self._autoencoder is not None:
                if self._input_dim != X.shape[1]:
                    raise ValueError(
                        f"Expected activations with dimension {self._input_dim}, "
                        f"but received {X.shape[1]}."
                    )
                return self
            cfg = self.autoencoder_config
            self._latent_dim = cfg.hidden_dim
            self._autoencoder = SparseAutoencoder(
                input_dim=self._input_dim,
                hidden_dim=cfg.hidden_dim,
                activation=cfg.activation,
                beta=cfg.beta,
            ).to(self.device)
            self._train_autoencoder(X, cfg)
            return self

        raise ValueError(f"Unknown strategy '{self.strategy}'")

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._require_2d(X)

        if self.strategy == "identity":
            return X

        if self.strategy == "standard":
            if self._scaler is None:
                raise RuntimeError("StandardScaler not fitted.")
            return self._scaler.transform(X)

        if self.strategy == "autoencoder":
            if self._autoencoder is None:
                raise RuntimeError("Autoencoder not fitted.")
            tensor = torch.from_numpy(X).float().to(self.device)
            with torch.no_grad():
                latent = self._autoencoder.encode(tensor)
            return latent.cpu().numpy()

        raise ValueError(f"Unknown strategy '{self.strategy}'")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.strategy == "identity":
            return X

        if self.strategy == "standard":
            if self._scaler is None:
                raise RuntimeError("StandardScaler not fitted.")
            return self._scaler.inverse_transform(X)

        if self.strategy == "autoencoder":
            if self._autoencoder is None:
                raise RuntimeError("Autoencoder not fitted.")
            tensor = torch.from_numpy(self._require_2d(X)).float().to(self.device)
            with torch.no_grad():
                decoded = self._autoencoder.decode(tensor)
            return decoded.cpu().numpy()

        raise ValueError(f"Unknown strategy '{self.strategy}'")

    def direction_to_input_space(self, direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction)
        original_was_1d = direction.ndim == 1
        if original_was_1d:
            direction = direction.reshape(1, -1)

        if self.strategy == "identity":
            return direction[0] if original_was_1d else direction

        if self.strategy == "standard":
            if self._scaler is None:
                raise RuntimeError("StandardScaler not fitted.")
            scale = getattr(self._scaler, "scale_", None)
            if scale is None:
                raise RuntimeError("Scaler missing scale_ attribute.")

            safe_scale = np.where(scale == 0, 1.0, scale)

            # SHAP sometimes returns (n_features, n_samples); fix the orientation.
            if (
                direction.ndim == 2
                and direction.shape[1] != safe_scale.size
                and direction.shape[0] == safe_scale.size
            ):
                direction = direction.T

            scaled = direction / safe_scale

            return scaled[0] if original_was_1d else scaled

        if self.strategy == "autoencoder":
            if self._autoencoder is None:
                raise RuntimeError("Autoencoder not fitted.")
            tensor = torch.from_numpy(direction).float().to(self.device)
            with torch.no_grad():
                mapped = self._autoencoder.decode(tensor)
            mapped_np = mapped.cpu().numpy()
            return mapped_np[0] if original_was_1d else mapped_np

        raise ValueError(f"Unknown strategy '{self.strategy}'")

    def load_autoencoder_state(
        self,
        *,
        input_dim: int,
        state_dict: Mapping[str, torch.Tensor],
        hidden_dim: Optional[int] = None,
        beta: Optional[float] = None,
        activation: Optional[nn.Module] = None,
    ) -> None:
        cfg = self.autoencoder_config
        hidden = hidden_dim or cfg.hidden_dim
        beta_value = beta if beta is not None else cfg.beta
        activation_module = activation or cfg.activation

        self._input_dim = input_dim
        self._latent_dim = hidden
        self._autoencoder = SparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden,
            activation=activation_module,
            beta=beta_value,
        ).to(self.device)
        self._autoencoder.load_state_dict(dict(state_dict))
        self._autoencoder.eval()
        self._autoencoder_history = []

    def load_autoencoder_artifact(
        self,
        artifact: Union[str, Path, Mapping[str, object]],
        *,
        map_location: Optional[torch.device] = None,
    ) -> None:
        if isinstance(artifact, (str, Path)):
            loaded = torch.load(artifact, map_location=map_location or self.device)
        elif isinstance(artifact, Mapping):
            loaded = artifact
        else:
            raise TypeError("artifact must be a path or mapping.")

        config_dict = loaded.get("config")
        if config_dict:
            self.autoencoder_config = AutoEncoderConfig.from_dict(config_dict)

        state_dict = loaded.get("state_dict")
        if state_dict is None:
            raise KeyError("Autoencoder artifact missing 'state_dict'.")

        input_dim_obj = loaded.get("input_dim")
        if input_dim_obj is None:
            raise KeyError("Autoencoder artifact missing 'input_dim'.")
        input_dim = int(input_dim_obj)

        latent_dim_obj = loaded.get("latent_dim")
        latent_dim = int(latent_dim_obj) if latent_dim_obj is not None else self.autoencoder_config.hidden_dim

        self.strategy = "autoencoder"
        self.load_autoencoder_state(
            input_dim=input_dim,
            hidden_dim=latent_dim,
            state_dict=state_dict,
            beta=self.autoencoder_config.beta,
            activation=self.autoencoder_config.activation,
        )

    # ------------------------------------------------------------------ #
    def _train_autoencoder(self, X: np.ndarray, cfg: "AutoEncoderConfig") -> None:
        if self._autoencoder is None:
            raise RuntimeError("Autoencoder not initialized.")
        tensor = torch.from_numpy(X).float()
        dataset = TensorDataset(tensor)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.Adam(self._autoencoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.MSELoss()

        writer = None
        log_dir = getattr(cfg, "log_dir", None)
        if log_dir:
            if SummaryWriter is None:
                raise RuntimeError(
                    "TensorBoard logging requested, but torch.utils.tensorboard is unavailable. "
                    "Install the 'tensorboard' package to enable logging."
                )
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(log_path))

        track_history = bool(getattr(cfg, "track_history", True))
        log_interval = max(1, int(getattr(cfg, "log_interval", 1)))
        history_records: list[dict[str, float]] | None = [] if track_history else None
        global_step = 0
        total_samples = len(dataset)

        self._autoencoder.train()
        for epoch in range(cfg.epochs):
            epoch_loss = 0.0
            for batch_index, (batch,) in enumerate(loader):
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon, latent = self._autoencoder(batch)
                mse = loss_fn(recon, batch)
                sparsity = cfg.beta * latent.abs().mean()
                loss = mse + sparsity
                loss.backward()
                optimizer.step()

                loss_value = float(loss.item())
                epoch_loss += loss_value * batch.size(0)

                if history_records is not None:
                    history_records.append(
                        {"epoch": epoch, "batch": batch_index, "loss": loss_value}
                    )
                if writer and global_step % log_interval == 0:
                    writer.add_scalar("autoencoder/batch_loss", loss_value, global_step)
                global_step += 1

            avg_loss = epoch_loss / total_samples if total_samples else float("nan")
            if history_records is not None:
                history_records.append(
                    {"epoch": epoch, "batch": -1, "loss": float(avg_loss)}
                )
            if writer:
                writer.add_scalar("autoencoder/epoch_loss", avg_loss, epoch)
            if cfg.verbose:
                print(
                    f"[Autoencoder] epoch={epoch+1}/{cfg.epochs} loss={avg_loss:.6f}"
                )

        if writer:
            writer.flush()
            writer.close()

        self._autoencoder.eval()
        self._autoencoder_history = history_records or []

    def get_autoencoder_history(self) -> list[dict[str, float]]:
        return list(self._autoencoder_history)

    @staticmethod
    def _require_2d(X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("Expected 2D activations.")
        return arr


@dataclass
class AutoEncoderConfig:
    hidden_dim: int = 256
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 30
    beta: float = 1e-3
    weight_decay: float = 1e-5
    activation: nn.Module = nn.ReLU()
    verbose: bool = False
    log_dir: Optional[str] = None
    log_interval: int = 10
    track_history: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AutoEncoderConfig":
        defaults = cls()

        activation_name = data.get("activation")
        if isinstance(activation_name, str):
            activation_cls = getattr(nn, activation_name, None)
            if isinstance(activation_cls, type) and issubclass(activation_cls, nn.Module):
                try:
                    activation = activation_cls()
                except TypeError:
                    activation = defaults.activation.__class__()
            else:
                activation = defaults.activation.__class__()
        else:
            activation = defaults.activation.__class__()

        return cls(
            hidden_dim=int(data.get("hidden_dim", defaults.hidden_dim)),
            lr=float(data.get("lr", defaults.lr)),
            batch_size=int(data.get("batch_size", defaults.batch_size)),
            epochs=int(data.get("epochs", defaults.epochs)),
            beta=float(data.get("beta", defaults.beta)),
            weight_decay=float(data.get("weight_decay", defaults.weight_decay)),
            activation=activation,
            verbose=cls._coerce_bool(data.get("verbose", defaults.verbose)),
            log_dir=(
                str(data["log_dir"])
                if data.get("log_dir", defaults.log_dir) is not None
                else None
            ),
            log_interval=int(data.get("log_interval", defaults.log_interval)),
            track_history=cls._coerce_bool(
                data.get("track_history", defaults.track_history)
            ),
        )

    @staticmethod
    def _coerce_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off"}
        return bool(value)


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, activation: nn.Module, beta: float) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.activation = activation
        self.beta = beta

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.encoder(x))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)
