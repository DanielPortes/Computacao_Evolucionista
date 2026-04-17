"""Metricas do exercicio TSP/ATSP."""

from __future__ import annotations

from ce.ex03_tsp.data import KNOWN_OPTIMA


def best_known_value(instance_name: str) -> float | None:
    """Retorna o melhor valor conhecido herdado do legado quando existir."""

    return KNOWN_OPTIMA.get(instance_name)


def relative_error(found_cost: float, reference_cost: float) -> float:
    """Calcula o erro relativo percentual em relacao a uma referencia."""

    if reference_cost <= 0:
        raise ValueError("reference_cost deve ser positivo.")
    return ((found_cost - reference_cost) / reference_cost) * 100.0

