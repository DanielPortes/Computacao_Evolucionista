"""GA canonico sobre permutacoes para o exercicio CE3."""

from __future__ import annotations

import random
from dataclasses import dataclass

from ce.ex03_tsp.problem import Route, TSPLIBRouteProblem


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    """Configuracao declarativa do GA de rotas."""

    population_size: int = 100
    generations: int = 200
    crossover_rate: float = 0.7
    mutation_rate: float = 0.05
    tournament_size: int = 3
    elite_size: int = 1
    seed: int = 42

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size deve ser pelo menos 2.")
        if self.generations < 1:
            raise ValueError("generations deve ser pelo menos 1.")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate deve estar entre 0 e 1.")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate deve estar entre 0 e 1.")
        if self.tournament_size < 2:
            raise ValueError("tournament_size deve ser pelo menos 2.")
        if self.elite_size < 1 or self.elite_size > self.population_size:
            raise ValueError("elite_size deve estar entre 1 e population_size.")


@dataclass(frozen=True)
class GenerationStats:
    """Resumo de fitness por geracao."""

    minimum: float
    mean: float
    maximum: float


@dataclass(frozen=True)
class GARunResult:
    """Resultado final do GA sobre uma instancia."""

    best_route: Route
    best_cost: float
    history: tuple[GenerationStats, ...]


def solve_problem(problem: TSPLIBRouteProblem, config: GeneticAlgorithmConfig) -> GARunResult:
    """Executa o GA mantendo sempre individuos como permutacoes validas."""

    config.validate()
    rng = random.Random(config.seed)
    population = [_random_route(problem.node_ids, rng) for _ in range(config.population_size)]
    history: list[GenerationStats] = []

    for _ in range(config.generations):
        scored_population = _score_population(problem, population)
        history.append(_summarize_generation(scored_population))

        elites = [route for route, _ in scored_population[: config.elite_size]]
        next_population = elites.copy()
        while len(next_population) < config.population_size:
            parent_a = _tournament_select(scored_population, config.tournament_size, rng)
            parent_b = _tournament_select(scored_population, config.tournament_size, rng)
            if rng.random() < config.crossover_rate:
                child = _ordered_crossover(parent_a, parent_b, rng)
            else:
                child = parent_a
            child = _mutate(child, rng, config.mutation_rate)
            problem.validate_route(child)
            next_population.append(child)
        population = next_population

    final_population = _score_population(problem, population)
    best_route, best_cost = final_population[0]
    return GARunResult(best_route=best_route, best_cost=best_cost, history=tuple(history))


def _random_route(node_ids: tuple[int, ...], rng: random.Random) -> Route:
    route = list(node_ids)
    rng.shuffle(route)
    return tuple(route)


def _score_population(
    problem: TSPLIBRouteProblem,
    population: list[Route],
) -> list[tuple[Route, float]]:
    scored = [(route, problem.route_cost(route)) for route in population]
    return sorted(scored, key=lambda item: item[1])


def _summarize_generation(scored_population: list[tuple[Route, float]]) -> GenerationStats:
    costs = [cost for _, cost in scored_population]
    mean_cost = sum(costs) / len(costs)
    return GenerationStats(minimum=costs[0], mean=mean_cost, maximum=costs[-1])


def _tournament_select(
    scored_population: list[tuple[Route, float]],
    tournament_size: int,
    rng: random.Random,
) -> Route:
    sampled = rng.sample(scored_population, k=tournament_size)
    sampled.sort(key=lambda item: item[1])
    return sampled[0][0]


def _ordered_crossover(parent_a: Route, parent_b: Route, rng: random.Random) -> Route:
    start, end = sorted(rng.sample(range(len(parent_a)), k=2))
    child: list[int | None] = [None] * len(parent_a)
    child[start : end + 1] = parent_a[start : end + 1]

    filler = [gene for gene in parent_b if gene not in child]
    filler_index = 0
    for index, gene in enumerate(child):
        if gene is None:
            child[index] = filler[filler_index]
            filler_index += 1
    assert all(gene is not None for gene in child)
    return tuple(gene for gene in child if gene is not None)


def _mutate(route: Route, rng: random.Random, mutation_rate: float) -> Route:
    mutated = list(route)
    for index in range(len(mutated)):
        if rng.random() < mutation_rate:
            swap_index = rng.randrange(len(mutated))
            mutated[index], mutated[swap_index] = mutated[swap_index], mutated[index]
    return tuple(mutated)
