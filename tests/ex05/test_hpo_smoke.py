from __future__ import annotations

import math

from ce.ex05_hpo.objective import TemporalCVConfig
from ce.ex05_hpo.search import EvolutionSearchConfig, run_search


def test_hpo_smoke_runs_with_small_budget() -> None:
    result = run_search(
        search_config=EvolutionSearchConfig(population_size=4, generations=1, seed=7),
        objective_config=TemporalCVConfig(
            n_splits=2,
            max_epochs=2,
            patience=1,
            max_rows=160,
            device="cpu",
            seed=7,
        ),
    )

    assert math.isfinite(result.best_score)
    assert math.isfinite(result.baseline_score)
    assert result.best_params.second_hidden_size <= result.best_params.hidden_size

