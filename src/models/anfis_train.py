"""Training helpers for the ANFIS model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.config import AnfisConfig
from src.models.anfis_model import ANFIS


@dataclass(slots=True)
class TrainingHistory:
    train_losses: list[float]
    val_losses: list[float]
    best_val_loss: float
    best_epoch: int
    stopped_early: bool


@dataclass(slots=True)
class TrainingArtifacts:
    model: ANFIS
    history: TrainingHistory


class AnfisTrainer:
    """Train ANFIS with optional hybrid learning and early stopping."""

    def __init__(self, config: AnfisConfig) -> None:
        self.config = config
        self.device = torch.device("cpu")
        torch.manual_seed(config.random_seed)

    def fit(
        self,
        model: ANFIS,
        train_features: Tensor,
        train_targets: Tensor,
        val_features: Tensor | None = None,
        val_targets: Tensor | None = None,
    ) -> TrainingArtifacts:
        train_loader = self._build_loader(train_features, train_targets)
        parameters = list(model.membership_layer.parameters())
        if not self.config.use_hybrid_learning:
            parameters.append(model.consequents)

        optimizer = torch.optim.Adam(parameters, lr=self.config.learning_rate, weight_decay=self.config.l2_weight_decay)
        criterion = torch.nn.MSELoss()

        model.to(self.device)
        best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        best_val_loss = float("inf")
        best_epoch = -1
        patience_counter = 0
        train_losses: list[float] = []
        val_losses: list[float] = []
        stopped_early = False

        for epoch in range(self.config.epochs):
            if self.config.use_hybrid_learning:
                model.solve_consequents(train_features, train_targets)

            model.train()
            epoch_losses: list[float] = []
            for batch_features, batch_targets in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_features)
                loss = criterion(predictions, _ensure_2d(batch_targets))
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            train_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
            train_losses.append(train_loss)

            val_loss = train_loss
            if val_features is not None and val_targets is not None:
                if self.config.use_hybrid_learning:
                    model.solve_consequents(train_features, train_targets)
                val_loss = self.evaluate(model, val_features, val_targets)
            val_losses.append(val_loss)

            if val_loss + self.config.min_delta < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                stopped_early = True
                break

        model.load_state_dict(best_state)
        history = TrainingHistory(
            train_losses=train_losses,
            val_losses=val_losses,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            stopped_early=stopped_early,
        )
        return TrainingArtifacts(model=model, history=history)

    def evaluate(self, model: ANFIS, features: Tensor, targets: Tensor) -> float:
        model.eval()
        criterion = torch.nn.MSELoss()
        with torch.no_grad():
            predictions = model(features)
            loss = criterion(predictions, _ensure_2d(targets))
        return float(loss.item())

    def predict(self, model: ANFIS, features: Tensor) -> Tensor:
        model.eval()
        with torch.no_grad():
            return model(features)

    def save_model(self, model: ANFIS, path: str | Path) -> Path:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        return save_path

    def load_model(self, path: str | Path, input_dim: int, num_memberships: int) -> ANFIS:
        model = ANFIS(input_dim=input_dim, num_memberships=num_memberships)
        state_dict = torch.load(Path(path), map_location=self.device)
        model.load_state_dict(state_dict)
        return model

    def _build_loader(self, features: Tensor, targets: Tensor) -> DataLoader:
        dataset = TensorDataset(features, _ensure_2d(targets))
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)


def _ensure_2d(tensor: Tensor) -> Tensor:
    return tensor.unsqueeze(1) if tensor.ndim == 1 else tensor
