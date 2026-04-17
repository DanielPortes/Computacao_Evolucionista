from __future__ import annotations

import math

from ce.ex04_forecasting.evaluate import run_baseline
from ce.ex04_forecasting.train import ForecastConfig


def test_cpu_smoke_train_runs_without_notebook() -> None:
    result = run_baseline(
        ForecastConfig(
            station="rola_moca",
            lookback=3,
            batch_size=16,
            max_epochs=3,
            patience=2,
            learning_rate=1e-2,
            hidden_size=16,
            second_hidden_size=8,
            device="cpu",
            max_rows=160,
            seed=7,
        )
    )

    assert result.training.config.device == "cpu"
    assert result.training.best_validation_loss >= 0.0
    assert len(result.training.validation_losses) >= 1
    assert math.isfinite(result.validation_metrics.rmse)
    assert math.isfinite(result.test_metrics.rmse)

