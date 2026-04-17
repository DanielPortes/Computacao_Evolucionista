from __future__ import annotations

import math

from ce.ex04_forecasting.gp import GPConfig, run_gp_baseline


def test_gp_smoke_returns_finite_temporal_metrics() -> None:
    result = run_gp_baseline(
        GPConfig(
            max_rows=64,
            lookback=3,
            population_size=8,
            generations=3,
            tournament_size=3,
            max_depth=3,
            seed=7,
        )
    )

    assert result.best_expression
    assert result.best_tree_size >= 1
    assert len(result.history) == 4
    assert math.isfinite(result.validation_metrics.rmse)
    assert math.isfinite(result.test_metrics.rmse)
