from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Optional, Union

import numpy as np
import torch
from tqdm import tqdm
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
        self._autoencoder_metrics: dict[str, float] = {}
        self._input_norm: str = "none"
        self._norm_eps: float = 1e-5
        self._l0_threshold: float = 1e-6

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
            self._input_norm = cfg.input_norm
            self._norm_eps = cfg.norm_eps
            self._l0_threshold = cfg.l0_threshold
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
            tensor = self._apply_input_norm_tensor(tensor, self._input_norm, self._norm_eps)
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
        # gradients do not need bias
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
                mapped = tensor @ self._autoencoder.decoder.weight.T
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
        self._input_norm = cfg.input_norm
        self._norm_eps = cfg.norm_eps
        self._l0_threshold = cfg.l0_threshold
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
        self._autoencoder_metrics = {}

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
        metrics_obj = loaded.get("metrics")
        if isinstance(metrics_obj, Mapping):
            self._autoencoder_metrics = {str(k): float(v) for k, v in metrics_obj.items() if v is not None}

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
        total_seen = 0
        total_elements = 0
        mse_sum = 0.0
        l1_sum = 0.0
        l0_sum = 0.0
        latent_sum = 0.0
        latent_sumsq = 0.0
        latent_zero_count = 0.0
        active_mask: torch.Tensor | None = None

        self._autoencoder.train()
        epoch_iter = tqdm(range(cfg.epochs), desc="SAE epochs", disable=not cfg.tqdm)
        for epoch in epoch_iter:
            batch_iter = tqdm(
                loader,
                desc=f"Epoch {epoch+1}/{cfg.epochs}",
                leave=False,
                disable=not cfg.tqdm,
            )
            for batch_index, (batch,) in enumerate(batch_iter):
                batch = batch.to(self.device)
                batch = self._apply_input_norm_tensor(batch, cfg.input_norm, cfg.norm_eps)
                optimizer.zero_grad()
                recon, latent = self._autoencoder(batch)
                mse = loss_fn(recon, batch)
                l1_mean = latent.abs().mean()
                loss = mse + cfg.beta * l1_mean
                loss.backward()
                optimizer.step()

                batch_size = batch.size(0)
                loss_value = float(loss.item())
                mse_value = float(mse.item())
                l1_value = float(l1_mean.item())
                l0_counts = (latent.abs() > cfg.l0_threshold).sum(dim=1).float()
                l0_mean = float(l0_counts.mean().item())
                sparsity = float((latent.abs() <= cfg.l0_threshold).float().mean().item())

                mse_sum += mse_value * batch_size
                l1_sum += l1_value * batch_size
                l0_sum += l0_mean * batch_size
                latent_sum += float(latent.sum().item())
                latent_sumsq += float((latent ** 2).sum().item())
                latent_zero_count += float((latent.abs() <= cfg.l0_threshold).sum().item())
                total_elements += latent.numel()
                total_seen += batch_size

                if active_mask is None:
                    active_mask = (latent.abs() > cfg.l0_threshold).any(dim=0)
                else:
                    active_mask |= (latent.abs() > cfg.l0_threshold).any(dim=0)

                if history_records is not None:
                    history_records.append(
                        {
                            "epoch": epoch,
                            "batch": batch_index,
                            "loss": loss_value,
                            "mse": mse_value,
                            "l1": l1_value,
                            "avg_l0": l0_mean,
                            "sparsity": sparsity,
                        }
                    )
                if writer and global_step % log_interval == 0:
                    writer.add_scalar("autoencoder/batch_loss", loss_value, global_step)
                    writer.add_scalar("autoencoder/batch_mse", mse_value, global_step)
                    writer.add_scalar("autoencoder/batch_l1", l1_value, global_step)
                    writer.add_scalar("autoencoder/batch_avg_l0", l0_mean, global_step)
                    writer.add_scalar("autoencoder/batch_sparsity", sparsity, global_step)
                global_step += 1

            if cfg.verbose:
                avg_loss = (mse_sum + cfg.beta * l1_sum) / max(1, total_seen)
                print(f"[Autoencoder] epoch={epoch+1}/{cfg.epochs} loss={avg_loss:.6f}")

        if writer:
            writer.flush()
            writer.close()

        total_samples_safe = max(1, total_seen)
        avg_mse = mse_sum / total_samples_safe
        avg_l1 = l1_sum / total_samples_safe
        avg_l0 = l0_sum / total_samples_safe
        latent_mean = latent_sum / max(1, total_elements)
        latent_var = (latent_sumsq / max(1, total_elements)) - latent_mean ** 2
        latent_std = float(np.sqrt(max(latent_var, 0.0)))
        latent_sparsity = latent_zero_count / max(1.0, total_elements)
        dead_fraction = 0.0
        if active_mask is not None:
            dead_fraction = float(1.0 - active_mask.float().mean().item())

        self._autoencoder.eval()
        self._autoencoder_history = history_records or []
        self._autoencoder_metrics = {
            "reconstruction_mse": float(avg_mse),
            "avg_l1": float(avg_l1),
            "avg_l0": float(avg_l0),
            "dead_latent_fraction": float(dead_fraction),
            "latent_mean": float(latent_mean),
            "latent_std": float(latent_std),
            "latent_sparsity": float(latent_sparsity),
            "total_steps": float(global_step),
            "steps_per_epoch": float(len(loader)),
        }

    def get_autoencoder_history(self) -> list[dict[str, float]]:
        return list(self._autoencoder_history)

    def get_autoencoder_metrics(self) -> dict[str, float]:
        return dict(self._autoencoder_metrics)

    @staticmethod
    def _apply_input_norm_array(x: np.ndarray, mode: str, eps: float) -> np.ndarray:
        if mode == "none":
            return x
        if mode == "layernorm":
            mean = x.mean(axis=1, keepdims=True)
            var = x.var(axis=1, keepdims=True)
            return (x - mean) / np.sqrt(var + eps)
        if mode == "rmsnorm":
            rms = np.sqrt((x ** 2).mean(axis=1, keepdims=True) + eps)
            return x / rms
        raise ValueError(f"Unknown input norm '{mode}'")

    @staticmethod
    def _apply_input_norm_tensor(x: torch.Tensor, mode: str, eps: float) -> torch.Tensor:
        if mode == "none":
            return x
        if mode == "layernorm":
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False)
            return (x - mean) / torch.sqrt(var + eps)
        if mode == "rmsnorm":
            rms = torch.sqrt((x ** 2).mean(dim=1, keepdim=True) + eps)
            return x / rms
        raise ValueError(f"Unknown input norm '{mode}'")

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
    training_steps: int | None = None
    beta: float = 1e-3
    weight_decay: float = 1e-5
    activation: nn.Module = nn.ReLU()
    verbose: bool = False
    log_dir: Optional[str] = None
    log_interval: int = 10
    track_history: bool = True
    input_norm: str = "none"
    norm_eps: float = 1e-5
    l0_threshold: float = 1e-6
    tqdm: bool = True

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
            training_steps=(
                int(data["training_steps"]) if data.get("training_steps") is not None else None
            ),
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
            input_norm=str(data.get("input_norm", defaults.input_norm)),
            norm_eps=float(data.get("norm_eps", defaults.norm_eps)),
            l0_threshold=float(data.get("l0_threshold", defaults.l0_threshold)),
            tqdm=cls._coerce_bool(data.get("tqdm", defaults.tqdm)),
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
