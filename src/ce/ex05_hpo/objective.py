"""Objetivo de HPO com validacao cruzada temporal."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field
from pymoo.core.problem import ElementwiseProblem
from sklearn.model_selection import TimeSeriesSplit
from torch import nn
from torch.utils.data import DataLoader

from ce.common.seeds import set_global_seed
from ce.ex04_forecasting.data import load_station_frame
from ce.ex04_forecasting.dataset import SlidingWindowDataset
from ce.ex04_forecasting.features import build_window_arrays, fit_scalers, transform_frame
from ce.ex04_forecasting.model import ForecastLSTM
from ce.ex05_hpo.search_space import ForecastSearchParams, ForecastSearchSpace


@dataclass(frozen=True)
class TemporalFold:
    """Representa um fold temporal contiguo."""

    train_end: int
    validation_end: int


class TemporalCVConfig(BaseModel):
    """Configuracao do objetivo temporal usado no HPO."""

    station: str = "rola_moca"
    n_splits: int = Field(default=3, ge=2)
    max_epochs: int = Field(default=5, ge=1)
    patience: int = Field(default=2, ge=1)
    max_rows: int | None = Field(default=240, ge=64)
    device: str = "cpu"
    seed: int = Field(default=42, ge=0)


def build_time_series_folds(length: int, n_splits: int) -> tuple[TemporalFold, ...]:
    """Gera folds temporais ordenados para avaliacao do HPO."""

    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds: list[TemporalFold] = []
    for train_indices, validation_indices in splitter.split(np.arange(length)):
        folds.append(
            TemporalFold(
                train_end=int(train_indices[-1]) + 1,
                validation_end=int(validation_indices[-1]) + 1,
            )
        )
    return tuple(folds)


class ForecastHPOProblem(ElementwiseProblem):  # type: ignore[misc]
    """Problema mono-objetivo de HPO sobre o baseline temporal."""

    def __init__(self, search_space: ForecastSearchSpace, objective_config: TemporalCVConfig):
        self.search_space = search_space
        self.objective_config = objective_config
        self.frame = load_station_frame(
            objective_config.station,
            max_rows=objective_config.max_rows,
        )
        self.folds = build_time_series_folds(len(self.frame), objective_config.n_splits)
        super().__init__(
            n_var=search_space.dimension,
            n_obj=1,
            xl=search_space.lower_bounds(),
            xu=search_space.upper_bounds(),
        )

    def _evaluate(
        self,
        x: np.ndarray,
        out: dict[str, float],
        *args: object,
        **kwargs: object,
    ) -> None:
        params = self.search_space.decode(np.asarray(x, dtype=float))
        out["F"] = float(
            evaluate_search_params(
                params,
                self.frame,
                self.folds,
                self.objective_config,
            )
        )


def evaluate_search_params(
    params: ForecastSearchParams,
    frame: pd.DataFrame,
    folds: tuple[TemporalFold, ...],
    objective_config: TemporalCVConfig,
) -> float:
    """Calcula o RMSE medio em validacao cruzada temporal."""

    fold_scores = [
        _evaluate_fold(
            params=params,
            frame=frame,
            fold=fold,
            objective_config=objective_config,
            fold_seed=objective_config.seed + index,
        )
        for index, fold in enumerate(folds)
    ]
    return float(np.mean(fold_scores))


def _evaluate_fold(
    params: ForecastSearchParams,
    frame: pd.DataFrame,
    fold: TemporalFold,
    objective_config: TemporalCVConfig,
    fold_seed: int,
) -> float:
    set_global_seed(fold_seed)
    torch.set_num_threads(1)
    scalers = fit_scalers(frame.iloc[: fold.train_end])
    transformed = transform_frame(frame, scalers)

    train_inputs, train_targets = build_window_arrays(
        transformed.inputs,
        transformed.targets,
        lookback=params.lookback,
        start_index=params.lookback,
        end_index=fold.train_end,
    )
    validation_inputs, validation_targets = build_window_arrays(
        transformed.inputs,
        transformed.targets,
        lookback=params.lookback,
        start_index=fold.train_end,
        end_index=fold.validation_end,
    )

    train_dataset = SlidingWindowDataset(train_inputs, train_targets)
    validation_dataset = SlidingWindowDataset(validation_inputs, validation_targets)
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=params.batch_size, shuffle=False)

    device = torch.device(objective_config.device)
    model = ForecastLSTM(
        input_size=train_inputs.shape[-1],
        hidden_size=params.hidden_size,
        second_hidden_size=params.second_hidden_size,
        dropout=params.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    criterion = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale_epochs = 0

    for _ in range(objective_config.max_epochs):
        _run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
        validation_loss = _run_epoch(
            model,
            validation_loader,
            criterion,
            optimizer=None,
            device=device,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= objective_config.patience:
                break

    model.load_state_dict(best_state)
    predictions, actuals = _predict(model, validation_loader, device)
    predictions_original = scalers.target_scaler.inverse_transform(predictions)
    actuals_original = scalers.target_scaler.inverse_transform(actuals)
    return sqrt(float(np.mean(np.square(predictions_original - actuals_original))))


def _run_epoch(
    model: ForecastLSTM,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_items = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_size = int(inputs.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    if total_items == 0:
        raise ValueError("Fold temporal sem amostras suficientes para o lookback selecionado.")
    return total_loss / total_items


def _predict(
    model: ForecastLSTM,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            predictions.append(model(inputs.to(device)).cpu().numpy())
            actuals.append(targets.cpu().numpy())
    return np.vstack(predictions), np.vstack(actuals)
