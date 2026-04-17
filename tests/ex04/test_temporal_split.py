from __future__ import annotations

import pandas as pd

from ce.ex04_forecasting.data import INPUT_COLUMNS
from ce.ex04_forecasting.features import fit_scalers, temporal_split, transform_frame


def test_temporal_split_is_ordered_and_scaler_is_fit_only_on_train() -> None:
    frame = pd.DataFrame({column: range(10) for column in INPUT_COLUMNS})
    split = temporal_split(len(frame), train_fraction=0.6, validation_fraction=0.2)

    assert split.train_end == 6
    assert split.validation_end == 8

    scalers = fit_scalers(frame.iloc[: split.train_end])
    transformed = transform_frame(frame, scalers)

    assert transformed.inputs[0, 0] == 0.0
    assert transformed.inputs[5, 0] == 1.0
    assert transformed.inputs[7, 0] > 1.0

