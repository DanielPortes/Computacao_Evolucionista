"""Leitura e limpeza canonica dos dados de previsao temporal."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ce.common.paths import weather_dataset

TARGET_COLUMN = "GLOBAL"
INPUT_COLUMNS = (
    "$Temperatura_{Inst}$",
    "$Temperatura_{Max}$",
    "$Temperatura_{Min}$",
    "$Umidade_{Max}$",
    "$Umidade_{Min}$",
    "$Umidade_{Inst}$",
    TARGET_COLUMN,
)


def load_station_frame(station: str, max_rows: int | None = None) -> pd.DataFrame:
    """Carrega e limpa o dataset canonico de uma estacao."""

    path = weather_dataset(station)
    frame = _read_csv(path)
    frame = _normalize_columns(frame)
    frame = _select_model_columns(frame)
    frame = _drop_inconsistent_rows(frame)
    frame = frame.dropna().sort_index()
    if max_rows is not None:
        frame = frame.iloc[:max_rows].copy()
    return frame


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";")
    frame["DATE"] = pd.to_datetime(frame["DATE"], errors="raise")
    frame = frame.set_index("DATE")
    frame = frame.replace(",", ".", regex=True)
    return frame


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "H" in normalized.columns and TARGET_COLUMN not in normalized.columns:
        normalized = normalized.rename(columns={"H": TARGET_COLUMN})
    if "H" in normalized.columns and TARGET_COLUMN in normalized.columns:
        normalized[TARGET_COLUMN] = normalized[TARGET_COLUMN].fillna(normalized["H"])
        normalized = normalized.drop(columns=["H"])
    return normalized


def _select_model_columns(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset canonico: {missing}.")
    selected = frame.loc[:, INPUT_COLUMNS].copy()
    return selected.astype(float)


def _drop_inconsistent_rows(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame.copy()
    invalid_index = filtered.loc[
        (filtered["$Temperatura_{Max}$"] < filtered["$Temperatura_{Min}$"])
        | (filtered["$Umidade_{Max}$"] < filtered["$Umidade_{Min}$"])
        | (filtered["$Temperatura_{Max}$"] < filtered["$Temperatura_{Inst}$"])
        | (filtered["$Umidade_{Max}$"] < filtered["$Umidade_{Inst}$"])
        | (filtered["$Temperatura_{Inst}$"] < filtered["$Temperatura_{Min}$"])
        | (filtered["$Umidade_{Inst}$"] < filtered["$Umidade_{Min}$"])
    ].index
    if len(invalid_index) > 0:
        filtered = filtered.drop(index=invalid_index)
    return filtered

