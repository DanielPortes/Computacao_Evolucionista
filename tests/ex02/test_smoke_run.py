from __future__ import annotations

import math

from ce.ex02_cec2017.run import ExperimentConfig, run_suite


def test_smoke_run_uses_best_value_per_run_without_shape_corruption() -> None:
    config = ExperimentConfig(
        function_ids=(3,),
        dimensions=(2, 10),
        budget_multiplier=5,
        n_runs=1,
        population_size=20,
        base_seed=7,
    )

    results = run_suite(config)
    results_by_dimension = {result.dimension: result for result in results}

    assert set(results_by_dimension) == {2, 10}
    assert all(result.algorithm_name == "ga" for result in results)
    assert results_by_dimension[2].budget == 10
    assert results_by_dimension[10].budget == 50
    assert len(results_by_dimension[2].runs[0].best_x) == 2
    assert len(results_by_dimension[10].runs[0].best_x) == 10
    assert math.isfinite(results_by_dimension[10].runs[0].best_f)
    assert results_by_dimension[10].runs[0].evaluations == 50
    assert len(results_by_dimension[10].runs[0].history) >= 1
    assert results_by_dimension[10].summary.minimum <= results_by_dimension[10].summary.maximum


def test_run_suite_aggregates_results_by_algorithm_function_and_dimension() -> None:
    config = ExperimentConfig(
        algorithm_names=("es", "de"),
        function_ids=(3,),
        dimensions=(2,),
        budget_multiplier=6,
        n_runs=1,
        population_size=6,
        base_seed=5,
    )

    results = run_suite(config)

    observed = [(result.algorithm_name, result.function_id, result.dimension) for result in results]

    assert observed == [
        ("es", 3, 2),
        ("de", 3, 2),
    ]
