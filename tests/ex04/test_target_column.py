from __future__ import annotations

import numpy as np

from ce.ex04_forecasting.features import build_window_arrays


def test_window_builder_uses_global_series_as_target() -> None:
    inputs = np.asarray(
        [
            [10.0, 100.0],
            [11.0, 101.0],
            [12.0, 102.0],
            [13.0, 103.0],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)

    window_inputs, window_targets = build_window_arrays(
        inputs,
        targets,
        lookback=2,
        start_index=2,
        end_index=4,
    )

    assert window_inputs.shape == (2, 2, 2)
    assert window_targets[:, 0].tolist() == [3.0, 4.0]
    assert window_targets[:, 0].tolist() != window_inputs[:, -1, 0].tolist()

