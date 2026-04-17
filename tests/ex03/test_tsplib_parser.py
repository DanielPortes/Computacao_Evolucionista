from __future__ import annotations

from ce.ex03_tsp.data import available_instances, load_instance


def test_all_instances_load_from_tsplib95() -> None:
    assert set(available_instances()) == {"berlin52", "br17", "ch130", "ftv70"}

    berlin = load_instance("berlin52")
    br17 = load_instance("br17")

    assert berlin.problem_type == "TSP"
    assert berlin.dimension == 52
    assert br17.problem_type == "ATSP"
    assert br17.dimension == 17

