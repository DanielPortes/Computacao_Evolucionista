from __future__ import annotations

from ce.ex03_tsp.problem import build_problem


def test_tsp_cost_uses_instance_weights() -> None:
    problem = build_problem("berlin52")
    route = problem.node_ids

    expected = 0.0
    for start, end in zip(route, route[1:] + route[:1], strict=True):
        expected += problem.instance.weight(start, end)

    assert problem.route_cost(route) == expected


def test_atsp_cost_uses_directed_instance_weights() -> None:
    problem = build_problem("br17")
    route = problem.node_ids

    expected = 0.0
    for start, end in zip(route, route[1:] + route[:1], strict=True):
        expected += problem.instance.weight(start, end)

    assert problem.route_cost(route) == expected

