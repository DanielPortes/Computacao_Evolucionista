from __future__ import annotations

from ce.ex03_tsp.ga import GeneticAlgorithmConfig, solve_problem
from ce.ex03_tsp.problem import build_problem
from ce.ex03_tsp.run import ExperimentConfig, solve_instances


def test_ga_smoke_preserves_valid_permutations() -> None:
    problem = build_problem("berlin52")
    result = solve_problem(
        problem,
        GeneticAlgorithmConfig(
            population_size=20,
            generations=5,
            crossover_rate=0.7,
            mutation_rate=0.1,
            tournament_size=3,
            elite_size=1,
            seed=7,
        ),
    )

    problem.validate_route(result.best_route)
    assert len(result.best_route) == problem.instance.dimension
    assert len(result.history) == 5


def test_run_supports_ga_and_aco_for_same_instance() -> None:
    results = solve_instances(
        ExperimentConfig(
            algorithm_names=("ga", "aco"),
            instance_names=("berlin52",),
            population_size=12,
            generations=3,
            crossover_rate=0.7,
            mutation_rate=0.1,
            tournament_size=3,
            elite_size=1,
            base_seed=7,
        )
    )

    assert [(result.algorithm_name, result.instance_name) for result in results] == [
        ("ga", "berlin52"),
        ("aco", "berlin52"),
    ]
