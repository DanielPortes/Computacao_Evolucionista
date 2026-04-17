"""ACO canonico para TSP/ATSP no exercicio CE3."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ce.ex03_tsp.problem import Route, TSPLIBRouteProblem


@dataclass(frozen=True)
class AntColonyConfig:
    """Configuracao declarativa do Ant Colony Optimization."""

    ant_count: int = 100
    iterations: int = 200
    alpha: float = 1.0
    beta: float = 3.0
    evaporation_rate: float = 0.3
    deposit_weight: float = 1.0
    seed: int = 42

    def validate(self) -> None:
        if self.ant_count < 2:
            raise ValueError("ant_count deve ser pelo menos 2.")
        if self.iterations < 1:
            raise ValueError("iterations deve ser pelo menos 1.")
        if self.alpha <= 0.0:
            raise ValueError("alpha deve ser positivo.")
        if self.beta <= 0.0:
            raise ValueError("beta deve ser positivo.")
        if not 0.0 < self.evaporation_rate < 1.0:
            raise ValueError("evaporation_rate deve estar estritamente entre 0 e 1.")
        if self.deposit_weight <= 0.0:
            raise ValueError("deposit_weight deve ser positivo.")


@dataclass(frozen=True)
class ACORunResult:
    """Resultado final do ACO sobre uma instancia."""

    best_route: Route
    best_cost: float
    history: tuple[float, ...]


def solve_problem(problem: TSPLIBRouteProblem, config: AntColonyConfig) -> ACORunResult:
    """Executa um Ant System simples sobre uma instancia TSP/ATSP."""

    config.validate()
    rng = random.Random(config.seed)
    cost_matrix = _build_cost_matrix(problem)
    heuristic_matrix = np.zeros_like(cost_matrix, dtype=float)
    finite_mask = np.isfinite(cost_matrix)
    heuristic_matrix[finite_mask] = 1.0 / cost_matrix[finite_mask]
    pheromone = np.ones_like(cost_matrix, dtype=float)
    node_index = {node_id: index for index, node_id in enumerate(problem.node_ids)}

    best_route: Route | None = None
    best_cost = float("inf")
    history: list[float] = []

    for _ in range(config.iterations):
        iteration_routes: list[tuple[list[int], Route, float]] = []
        for _ in range(config.ant_count):
            route_indices = _construct_route(pheromone, heuristic_matrix, rng, config)
            route = tuple(problem.node_ids[index] for index in route_indices)
            cost = problem.route_cost(route)
            iteration_routes.append((route_indices, route, cost))
            if cost < best_cost:
                best_cost = cost
                best_route = route

        pheromone *= 1.0 - config.evaporation_rate
        for route_indices, _, cost in iteration_routes:
            deposit = config.deposit_weight / max(cost, 1e-12)
            edges = zip(route_indices, route_indices[1:] + route_indices[:1], strict=True)
            for start, end in edges:
                pheromone[start, end] += deposit

        if best_route is not None:
            best_indices = [node_index[node_id] for node_id in best_route]
            best_deposit = config.deposit_weight / max(best_cost, 1e-12)
            for start, end in zip(best_indices, best_indices[1:] + best_indices[:1], strict=True):
                pheromone[start, end] += best_deposit

        pheromone = np.clip(pheromone, 1e-9, 1e9)
        history.append(best_cost)

    if best_route is None:
        raise RuntimeError("ACO terminou sem construir nenhuma rota valida.")
    return ACORunResult(best_route=best_route, best_cost=best_cost, history=tuple(history))


def _build_cost_matrix(problem: TSPLIBRouteProblem) -> np.ndarray:
    dimension = problem.instance.dimension
    matrix = np.full((dimension, dimension), np.inf, dtype=float)
    for row_index, start in enumerate(problem.node_ids):
        for column_index, end in enumerate(problem.node_ids):
            if row_index != column_index:
                matrix[row_index, column_index] = problem.instance.weight(start, end)
    return matrix


def _construct_route(
    pheromone: np.ndarray,
    heuristic_matrix: np.ndarray,
    rng: random.Random,
    config: AntColonyConfig,
) -> list[int]:
    dimension = pheromone.shape[0]
    current = rng.randrange(dimension)
    route = [current]
    unvisited = set(range(dimension))
    unvisited.remove(current)

    while unvisited:
        candidate_list = sorted(unvisited)
        pheromone_values = np.asarray(
            [pheromone[current, index] for index in candidate_list],
            dtype=float,
        )
        heuristic_values = np.asarray(
            [heuristic_matrix[current, index] for index in candidate_list],
            dtype=float,
        )
        desirability = np.power(pheromone_values, config.alpha) * np.power(
            heuristic_values,
            config.beta,
        )

        total = float(np.sum(desirability))
        if total <= 0.0 or not np.isfinite(total):
            next_index = rng.choice(candidate_list)
        else:
            probabilities = desirability / total
            next_index = candidate_list[_roulette(probabilities, rng)]

        route.append(next_index)
        unvisited.remove(next_index)
        current = next_index

    return route


def _roulette(probabilities: np.ndarray, rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, value in enumerate(probabilities):
        cumulative += float(value)
        if threshold <= cumulative:
            return index
    return len(probabilities) - 1
