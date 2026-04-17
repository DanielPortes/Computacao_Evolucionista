from __future__ import annotations

from ce.ex03_tsp.aco import AntColonyConfig, solve_problem
from ce.ex03_tsp.problem import build_problem


def test_aco_smoke_preserves_valid_permutations() -> None:
    problem = build_problem("berlin52")
    result = solve_problem(
        problem,
        AntColonyConfig(
            ant_count=12,
            iterations=4,
            seed=7,
        ),
    )

    problem.validate_route(result.best_route)
    assert len(result.best_route) == problem.instance.dimension
    assert len(result.history) == 4
