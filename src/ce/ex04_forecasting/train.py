"""Treino reproduzivel do baseline temporal do CE4."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pandas as pd
import torch
from pydantic import BaseModel, Field
from torch import nn
from torch.utils.data import DataLoader

from ce.common.seeds import set_global_seed
from ce.ex04_forecasting.data import load_station_frame
from ce.ex04_forecasting.dataset import SlidingWindowDataset
from ce.ex04_forecasting.features import (
    ScalerBundle,
    TemporalSplit,
    build_window_arrays,
    fit_scalers,
    temporal_split,
    transform_frame,
)
from ce.ex04_forecasting.model import ForecastLSTM


class ForecastConfig(BaseModel):
    """Configuracao declarativa do baseline LSTM."""

    station: str = "rola_moca"
    lookback: int = Field(default=3, ge=1)
    train_fraction: float = Field(default=0.7, gt=0.0, lt=1.0)
    validation_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    batch_size: int = Field(default=32, ge=1)
    max_epochs: int = Field(default=100, ge=1)
    patience: int = Field(default=10, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    hidden_size: int = Field(default=64, ge=1)
    second_hidden_size: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.2, ge=0.0, le=1.0)
    seed: int = Field(default=42, ge=0)
    device: str = "cpu"
    max_rows: int | None = Field(default=None, ge=32)


@dataclass(frozen=True)
class PreparedData:
    """Dados preparados para treino/avaliacao."""

    frame: pd.DataFrame
    split: TemporalSplit
    scalers: ScalerBundle
    train_dataset: SlidingWindowDataset
    validation_dataset: SlidingWindowDataset
    test_dataset: SlidingWindowDataset


@dataclass(frozen=True)
class TrainingArtifacts:
    """Artefatos resultantes do treino baseline."""

    config: ForecastConfig
    model: ForecastLSTM
    prepared_data: PreparedData
    best_epoch: int
    best_validation_loss: float
    train_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]


def prepare_data(config: ForecastConfig) -> PreparedData:
    """Prepara splits temporais, scalers e datasets PyTorch."""

    frame = load_station_frame(config.station, max_rows=config.max_rows)
    split = temporal_split(
        len(frame),
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    scalers = fit_scalers(frame.iloc[: split.train_end])
    transformed = transform_frame(frame, scalers)

    train_inputs, train_targets = build_window_arrays(
        transformed.inputs,
        transformed.targets,
        lookback=config.lookback,
        start_index=config.lookback,
        end_index=split.train_end,
    )
    validation_inputs, validation_targets = build_window_arrays(
        transformed.inputs,
        transformed.targets,
        lookback=config.lookback,
        start_index=split.train_end,
        end_index=split.validation_end,
    )
    test_inputs, test_targets = build_window_arrays(
        transformed.inputs,
        transformed.targets,
        lookback=config.lookback,
        start_index=split.validation_end,
        end_index=len(frame),
    )

    return PreparedData(
        frame=frame,
        split=split,
        scalers=scalers,
        train_dataset=SlidingWindowDataset(train_inputs, train_targets),
        validation_dataset=SlidingWindowDataset(validation_inputs, validation_targets),
        test_dataset=SlidingWindowDataset(test_inputs, test_targets),
    )


def train_model(config: ForecastConfig) -> TrainingArtifacts:
    """Treina o baseline em CPU ou GPU, com early stopping e restaure do melhor estado."""

    set_global_seed(config.seed)
    torch.set_num_threads(1)
    prepared = prepare_data(config)
    device = torch.device(config.device)

    model = ForecastLSTM(
        input_size=prepared.train_dataset.inputs.shape[-1],
        hidden_size=config.hidden_size,
        second_hidden_size=config.second_hidden_size,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        prepared.train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    validation_loader = DataLoader(
        prepared.validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(config.max_epochs):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
        validation_loss = _run_epoch(
            model,
            validation_loader,
            criterion,
            optimizer=None,
            device=device,
        )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    return TrainingArtifacts(
        config=config,
        model=model,
        prepared_data=prepared,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        train_losses=tuple(train_losses),
        validation_losses=tuple(validation_losses),
    )


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
        raise ValueError(
            "O DataLoader recebeu zero amostras; "
            "ajuste o split temporal ou o lookback."
        )
    return total_loss / total_items
