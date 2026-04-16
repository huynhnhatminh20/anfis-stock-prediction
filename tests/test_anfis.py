import numpy as np
import torch

from src.config import AnfisConfig
from src.evaluation.compare_models import compare_model_predictions
from src.evaluation.statistical_test import diebold_mariano_test, wilcoxon_signed_rank_test
from src.models.anfis_model import ANFIS
from src.models.anfis_train import AnfisTrainer


def _make_regression_dataset(n: int = 96) -> tuple[torch.Tensor, torch.Tensor]:
    x1 = torch.linspace(0.0, 1.0, n)
    x2 = torch.linspace(1.0, 0.0, n)
    features = torch.stack([x1, x2], dim=1)
    targets = (0.6 * x1 + 0.4 * x2 + 0.1 * torch.sin(3 * x1)).unsqueeze(1)
    return features, targets


def test_anfis_forward_shapes_and_rule_normalization() -> None:
    model = ANFIS(input_dim=2, num_memberships=2)
    features, _ = _make_regression_dataset(12)

    details = model.forward_with_details(features)

    assert details.predictions.shape == (12, 1)
    assert details.membership_values.shape == (12, 2, 2)
    assert details.firing_strengths.shape == (12, 4)
    assert torch.allclose(
        details.normalized_strengths.sum(dim=1),
        torch.ones(12),
        atol=1e-5,
    )


def test_anfis_training_reduces_validation_loss() -> None:
    features, targets = _make_regression_dataset()
    train_features, val_features = features[:72], features[72:]
    train_targets, val_targets = targets[:72], targets[72:]

    config = AnfisConfig(
        input_dim=2,
        num_memberships=2,
        epochs=25,
        batch_size=16,
        patience=6,
        learning_rate=0.03,
        use_hybrid_learning=True,
    )
    model = ANFIS(input_dim=config.input_dim, num_memberships=config.num_memberships)
    trainer = AnfisTrainer(config)

    baseline_loss = trainer.evaluate(model, val_features, val_targets)
    artifacts = trainer.fit(model, train_features, train_targets, val_features, val_targets)
    final_loss = trainer.evaluate(artifacts.model, val_features, val_targets)

    assert artifacts.history.train_losses
    assert artifacts.history.val_losses
    assert final_loss < baseline_loss


def test_compare_models_returns_sorted_best_model() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    result = compare_model_predictions(
        y_true=actual,
        predictions_by_model={
            "anfis": np.array([1.1, 1.9, 2.9, 4.1]),
            "mlp": np.array([0.8, 2.3, 3.4, 4.5]),
        },
    )

    assert list(result.metrics_table["model"]) == ["anfis", "mlp"]
    assert result.best_model == "anfis"
    assert set(result.predictions_table.columns) == {"actual", "anfis", "mlp"}


def test_statistical_tests_return_valid_probabilities() -> None:
    actual = np.array([10, 11, 12, 11, 13, 12, 14, 15], dtype=float)
    pred_a = np.array([10.1, 10.9, 12.2, 10.8, 12.9, 12.1, 14.1, 15.2])
    pred_b = np.array([9.5, 11.8, 11.3, 11.7, 13.6, 11.5, 13.1, 14.1])

    dm_result = diebold_mariano_test(actual, pred_a, pred_b)
    wilcoxon_result = wilcoxon_signed_rank_test(actual, pred_a, pred_b)

    assert 0.0 <= dm_result.p_value <= 1.0
    assert 0.0 <= wilcoxon_result.p_value <= 1.0
