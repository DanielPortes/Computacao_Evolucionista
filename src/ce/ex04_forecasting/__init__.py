"""Modulo canonico do exercicio CE4."""

from ce.ex04_forecasting.data import INPUT_COLUMNS, TARGET_COLUMN, load_station_frame
from ce.ex04_forecasting.evaluate import run_baseline
from ce.ex04_forecasting.train import ForecastConfig, train_model

__all__ = [
    "ForecastConfig",
    "INPUT_COLUMNS",
    "TARGET_COLUMN",
    "load_station_frame",
    "run_baseline",
    "train_model",
]

