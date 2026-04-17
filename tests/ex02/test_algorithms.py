from __future__ import annotations

import math

import pytest

from ce.ex02_cec2017.algorithms import available_algorithms
from ce.ex02_cec2017.run import ExperimentConfig, run_suite


@pytest.mark.parametrize("algorithm_name", available_algorithms())
def test_each_continuous_algorithm_runs_once_on_small_budget(algorithm_name: str) -> None:
    config = ExperimentConfig(
        algorithm_names=(algorithm_name,),
        function_ids=(3,),
        dimensions=(2,),
        budget_multiplier=6,
        n_runs=1,
        population_size=6,
        base_seed=7,
    )

    result = run_suite(config)[0]
    run = result.runs[0]

    assert result.algorithm_name == algorithm_name
    assert len(run.best_x) == 2
    assert run.evaluations == 12
    assert len(run.history) >= 1
    assert math.isfinite(run.best_f)
