"""Split temporal, escalonamento e janelamento para CE4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ce.ex04_forecasting.data import INPUT_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True)
class TemporalSplit:
    """Indices de corte para treino, validacao e teste."""

    train_end: int
    validation_end: int


@dataclass(frozen=True)
class ScalerBundle:
    """Scalers ajustados apenas no trecho de treino."""

    input_scaler: MinMaxScaler
    target_scaler: MinMaxScaler


@dataclass(frozen=True)
class TransformedSeries:
    """Serie escalonada pronta para virar janelas."""

    inputs: np.ndarray
    targets: np.ndarray


def temporal_split(
    length: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> TemporalSplit:
    """Calcula um split temporal sem embaralhamento."""

    if length < 10:
        raise ValueError("O dataset precisa ter ao menos 10 amostras para o split temporal.")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train_fraction e validation_fraction precisam ser positivos.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction precisa ser menor que 1.")

    train_end = int(length * train_fraction)
    validation_end = train_end + int(length * validation_fraction)
    if train_end < 2 or validation_end >= length:
        raise ValueError("Split temporal invalido para o tamanho informado.")
    return TemporalSplit(train_end=train_end, validation_end=validation_end)


def fit_scalers(train_frame: pd.DataFrame) -> ScalerBundle:
    """Ajusta scalers apenas nos dados de treino."""

    input_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    target_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    input_scaler.fit(train_frame.loc[:, INPUT_COLUMNS])
    target_scaler.fit(train_frame.loc[:, [TARGET_COLUMN]])
    return ScalerBundle(input_scaler=input_scaler, target_scaler=target_scaler)


def transform_frame(frame: pd.DataFrame, scalers: ScalerBundle) -> TransformedSeries:
    """Transforma a serie inteira usando scalers ajustados no treino."""

    inputs = scalers.input_scaler.transform(frame.loc[:, INPUT_COLUMNS]).astype(np.float32)
    targets = scalers.target_scaler.transform(frame.loc[:, [TARGET_COLUMN]]).astype(np.float32)
    return TransformedSeries(inputs=inputs, targets=targets)


def build_window_arrays(
    inputs: np.ndarray,
    targets: np.ndarray,
    lookback: int,
    start_index: int,
    end_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Constroi janelas deslizantes com alvo explicitamente igual a `GLOBAL`."""

    if lookback < 1:
        raise ValueError("lookback precisa ser pelo menos 1.")
    if start_index < lookback:
        raise ValueError("start_index precisa ser maior ou igual a lookback.")
    if end_index <= start_index:
        raise ValueError("end_index precisa ser maior que start_index.")

    window_inputs: list[np.ndarray] = []
    window_targets: list[np.ndarray] = []
    for target_index in range(start_index, end_index):
        window_inputs.append(inputs[target_index - lookback : target_index])
        window_targets.append(targets[target_index])
    return np.asarray(window_inputs, dtype=np.float32), np.asarray(window_targets, dtype=np.float32)
