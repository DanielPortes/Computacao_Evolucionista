"""Carregamento canonico das instancias TSPLIB do exercicio."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tsplib95

from ce.common.paths import list_tsplib_instances, tsplib_instance

KNOWN_OPTIMA = {
    "berlin52": 7542.0,
    "br17": 39.0,
    "ch130": 6110.0,
    "ftv70": 1950.0,
}


@dataclass(frozen=True)
class TSPLIBInstance:
    """Representa uma instancia TSP/ATSP carregada de forma canonica."""

    name: str
    source_path: Path
    problem_type: str
    dimension: int
    node_ids: tuple[int, ...]
    problem: Any = field(repr=False, compare=False)

    def weight(self, start: int, end: int) -> float:
        """Retorna o peso canonico definido pela propria instancia."""

        return float(self.problem.get_weight(start, end))


def available_instances() -> dict[str, Path]:
    """Lista as instancias disponiveis por alias canonico."""

    return list_tsplib_instances()


def load_instance(name: str) -> TSPLIBInstance:
    """Carrega uma instancia TSP/ATSP por alias canonico."""

    path = tsplib_instance(name)
    problem = tsplib95.load(path)
    node_ids = tuple(int(node) for node in problem.get_nodes())
    return TSPLIBInstance(
        name=name,
        source_path=path,
        problem_type=str(problem.type),
        dimension=int(problem.dimension),
        node_ids=node_ids,
        problem=problem,
    )

