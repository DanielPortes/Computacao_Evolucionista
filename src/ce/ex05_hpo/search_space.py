"""Espaco de busca coerente para o HPO do CE5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SearchAxis:
    """Eixo discreto do espaco de busca."""

    name: str
    values: tuple[float | int, ...]


@dataclass(frozen=True)
class ForecastSearchParams:
    """Parametros decodificados para o baseline LSTM."""

    lookback: int
    hidden_size: int
    second_hidden_size: int
    dropout: float
    learning_rate: float
    batch_size: int


@dataclass(frozen=True)
class ForecastSearchSpace:
    """Espaco de busca discreto e coerente com o modelo PyTorch."""

    axes: tuple[SearchAxis, ...]

    @property
    def dimension(self) -> int:
        return len(self.axes)

    def lower_bounds(self) -> np.ndarray:
        return np.zeros(self.dimension, dtype=float)

    def upper_bounds(self) -> np.ndarray:
        return np.asarray([len(axis.values) - 1 for axis in self.axes], dtype=float)

    def decode(self, vector: np.ndarray) -> ForecastSearchParams:
        indices = [
            int(np.clip(round(value), 0, len(axis.values) - 1))
            for value, axis in zip(vector, self.axes, strict=True)
        ]
        lookback = int(self.axes[0].values[indices[0]])
        hidden_size = int(self.axes[1].values[indices[1]])
        ratio = float(self.axes[2].values[indices[2]])
        second_hidden_size = max(8, int(hidden_size * ratio))
        dropout = float(self.axes[3].values[indices[3]])
        learning_rate = float(self.axes[4].values[indices[4]])
        batch_size = int(self.axes[5].values[indices[5]])
        return ForecastSearchParams(
            lookback=lookback,
            hidden_size=hidden_size,
            second_hidden_size=second_hidden_size,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

    def default_params(self) -> ForecastSearchParams:
        return ForecastSearchParams(
            lookback=3,
            hidden_size=64,
            second_hidden_size=32,
            dropout=0.2,
            learning_rate=1e-3,
            batch_size=32,
        )


def default_search_space() -> ForecastSearchSpace:
    """Retorna o espaco de busca canonico do CE5."""

    return ForecastSearchSpace(
        axes=(
            SearchAxis("lookback", (3, 6, 12)),
            SearchAxis("hidden_size", (16, 32, 64)),
            SearchAxis("decoder_ratio", (0.5, 0.75, 1.0)),
            SearchAxis("dropout", (0.0, 0.1, 0.2, 0.3)),
            SearchAxis("learning_rate", (1e-3, 5e-4, 1e-2)),
            SearchAxis("batch_size", (16, 32, 64)),
        )
    )
