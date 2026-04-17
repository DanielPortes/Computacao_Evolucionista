"""Avaliacao do baseline temporal do CE4."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import torch
from torch.utils.data import DataLoader

from ce.ex04_forecasting.train import ForecastConfig, TrainingArtifacts, train_model


@dataclass(frozen=True)
class RegressionMetrics:
    """Metricas de regressao no dominio original do alvo."""

    rmse: float
    mae: float


@dataclass(frozen=True)
class BaselineResult:
    """Resultado completo do baseline com treino e metricas."""

    training: TrainingArtifacts
    validation_metrics: RegressionMetrics
    test_metrics: RegressionMetrics


def run_baseline(config: ForecastConfig) -> BaselineResult:
    """Treina o baseline e retorna metricas para validacao e teste."""

    training = train_model(config)
    validation_metrics = evaluate_split(training, split_name="validation")
    test_metrics = evaluate_split(training, split_name="test")
    return BaselineResult(
        training=training,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )


def evaluate_split(training: TrainingArtifacts, split_name: str) -> RegressionMetrics:
    """Avalia um split no dominio original do `GLOBAL`."""

    dataset = {
        "validation": training.prepared_data.validation_dataset,
        "test": training.prepared_data.test_dataset,
    }[split_name]
    loader = DataLoader(dataset, batch_size=training.config.batch_size, shuffle=False)
    device = torch.device(training.config.device)

    predictions: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    training.model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = training.model(inputs.to(device)).cpu().numpy()
            predictions.append(outputs)
            actuals.append(targets.cpu().numpy())

    prediction_array = np.vstack(predictions)
    actual_array = np.vstack(actuals)
    target_scaler = training.prepared_data.scalers.target_scaler
    predictions_original = target_scaler.inverse_transform(prediction_array)
    actuals_original = target_scaler.inverse_transform(actual_array)

    return compute_regression_metrics(predictions_original, actuals_original)


def compute_regression_metrics(
    predictions_original: np.ndarray,
    actuals_original: np.ndarray,
) -> RegressionMetrics:
    """Calcula metricas de regressao a partir de arrays ja no dominio original."""

    residuals = predictions_original - actuals_original
    rmse = sqrt(float(np.mean(np.square(residuals))))
    mae = float(np.mean(np.abs(residuals)))
    return RegressionMetrics(rmse=rmse, mae=mae)
