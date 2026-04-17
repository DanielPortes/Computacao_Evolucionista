"""Modelagem canonica do problema de rota para CE3."""

from __future__ import annotations

from dataclasses import dataclass

from ce.ex03_tsp.data import TSPLIBInstance, load_instance

Route = tuple[int, ...]


@dataclass(frozen=True)
class TSPLIBRouteProblem:
    """Encapsula a validacao e o custo de uma rota sobre uma instancia TSPLIB."""

    instance: TSPLIBInstance

    @property
    def node_ids(self) -> tuple[int, ...]:
        return self.instance.node_ids

    @property
    def problem_type(self) -> str:
        return self.instance.problem_type

    def validate_route(self, route: Route) -> None:
        if len(route) != self.instance.dimension:
            raise ValueError(
                f"Rota invalida: esperado tamanho {self.instance.dimension}, recebido {len(route)}."
            )
        if set(route) != set(self.instance.node_ids):
            raise ValueError("Rota invalida: os nos nao correspondem exatamente aos da instancia.")

    def route_cost(self, route: Route) -> float:
        self.validate_route(route)
        total = 0.0
        for start, end in zip(route, route[1:] + route[:1], strict=True):
            total += self.instance.weight(start, end)
        return total


def build_problem(instance_name: str) -> TSPLIBRouteProblem:
    """Constroi um problema canonico a partir de uma instancia registrada."""

    return TSPLIBRouteProblem(instance=load_instance(instance_name))

