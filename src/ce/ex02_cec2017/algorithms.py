"""Construtores e execucao de algoritmos evolutivos para CE2."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Literal

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

from ce.ex02_cec2017.benchmarks import CEC2017Benchmark
from ce.ex02_cec2017.problems import build_problem

ContinuousAlgorithmName = Literal["ga", "es", "ep", "de", "pso", "abc"]
CONTINUOUS_ALGORITHMS: tuple[ContinuousAlgorithmName, ...] = ("ga", "es", "ep", "de", "pso", "abc")


@dataclass(frozen=True)
class OptimizerResult:
    """Saida minima compartilhada pelos algoritmos continuos."""

    best_x: tuple[float, ...]
    best_f: float
    evaluations: int
    history: tuple[float, ...]


def available_algorithms() -> tuple[ContinuousAlgorithmName, ...]:
    """Retorna os algoritmos continuos expostos pelo CE2."""

    return CONTINUOUS_ALGORITHMS


def algorithm_label(name: ContinuousAlgorithmName) -> str:
    """Retorna um rotulo amigavel para relatórios."""

    return {
        "ga": "GA",
        "es": "ES",
        "ep": "EP",
        "de": "DE",
        "pso": "PSO",
        "abc": "ABC",
    }[name]


def build_ga(population_size: int = 50) -> GA:
    """Retorna o GA mono-objetivo usado no CE2."""

    if population_size < 2:
        raise ValueError("population_size deve ser pelo menos 2.")

    return GA(
        pop_size=population_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )


def optimize_benchmark(
    algorithm_name: ContinuousAlgorithmName,
    benchmark: CEC2017Benchmark,
    population_size: int,
    budget: int,
    seed: int,
    lower_bound: float,
    upper_bound: float,
) -> OptimizerResult:
    """Executa um algoritmo continuo sobre um benchmark do CE2."""

    if budget < 2:
        raise ValueError("budget deve ser pelo menos 2.")
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound deve ser estritamente menor que upper_bound.")

    if algorithm_name == "ga":
        return _run_ga(
            benchmark=benchmark,
            population_size=population_size,
            budget=budget,
            seed=seed,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
    if algorithm_name == "es":
        return _run_es(benchmark, population_size, budget, seed, lower_bound, upper_bound)
    if algorithm_name == "ep":
        return _run_ep(benchmark, population_size, budget, seed, lower_bound, upper_bound)
    if algorithm_name == "de":
        return _run_de(benchmark, population_size, budget, seed, lower_bound, upper_bound)
    if algorithm_name == "pso":
        return _run_pso(benchmark, population_size, budget, seed, lower_bound, upper_bound)
    if algorithm_name == "abc":
        return _run_abc(benchmark, population_size, budget, seed, lower_bound, upper_bound)
    raise ValueError(f"Algoritmo nao suportado: {algorithm_name}.")


def _run_ga(
    benchmark: CEC2017Benchmark,
    population_size: int,
    budget: int,
    seed: int,
    lower_bound: float,
    upper_bound: float,
) -> OptimizerResult:
    problem = build_problem(
        function_id=benchmark.function_id,
        dimension=benchmark.dimension,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    optimization = minimize(
        problem,
        build_ga(population_size),
        termination=("n_eval", budget),
        seed=seed,
        verbose=False,
    )
    if optimization.X is None or optimization.F is None:
        raise RuntimeError("GA terminou sem solucao valida.")
    best_x = np.asarray(optimization.X, dtype=float).reshape(-1)
    best_f = float(np.asarray(optimization.F, dtype=float).reshape(-1)[0])
    return OptimizerResult(
        best_x=tuple(float(value) for value in best_x),
        best_f=best_f,
        evaluations=budget,
        history=(best_f,),
    )


def _run_es(
    benchmark: CEC2017Benchmark,
    population_size: int,
    budget: int,
    seed: int,
    lower_bound: float,
    upper_bound: float,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    dimension = benchmark.dimension
    size = _effective_population_size(population_size, budget, minimum=2)
    span = upper_bound - lower_bound
    tau = 1.0 / sqrt(2.0 * sqrt(dimension))
    tau_prime = 1.0 / sqrt(2.0 * dimension)

    population = _sample_population(rng, size, dimension, lower_bound, upper_bound)
    sigmas = np.full((size, dimension), 0.15 * span, dtype=float)
    fitness = benchmark.evaluate(population)
    evaluations = size
    history = [float(np.min(fitness))]

    while evaluations < budget:
        offspring_count = min(size, budget - evaluations)
        parent_a = rng.integers(0, size, size=offspring_count)
        parent_b = rng.integers(0, size, size=offspring_count)

        child_x = 0.5 * (population[parent_a] + population[parent_b])
        child_sigma = 0.5 * (sigmas[parent_a] + sigmas[parent_b])
        global_noise = rng.normal(size=(offspring_count, 1))
        local_noise = rng.normal(size=(offspring_count, dimension))
        child_sigma = child_sigma * np.exp(tau_prime * global_noise + tau * local_noise)
        child_sigma = np.clip(child_sigma, 1e-3 * span, span)
        child_x = _clip_to_bounds(
            child_x + child_sigma * rng.normal(size=(offspring_count, dimension)),
            lower_bound,
            upper_bound,
        )

        child_fitness = benchmark.evaluate(child_x)
        evaluations += offspring_count

        population = np.vstack([population, child_x])
        sigmas = np.vstack([sigmas, child_sigma])
        fitness = np.concatenate([fitness, child_fitness])
        selected = np.argsort(fitness)[:size]
        population = population[selected]
        sigmas = sigmas[selected]
        fitness = fitness[selected]
        history.append(float(fitness[0]))

    best_index = int(np.argmin(fitness))
    return OptimizerResult(
        best_x=tuple(float(value) for value in population[best_index]),
        best_f=float(fitness[best_index]),
        evaluations=evaluations,
        history=tuple(history),
    )


def _run_ep(
    benchmark: CEC2017Benchmark,
    population_size: int,
    budget: int,
    seed: int,
    lower_bound: float,
    upper_bound: float,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    dimension = benchmark.dimension
    size = _effective_population_size(population_size, budget, minimum=2)
    span = upper_bound - lower_bound
    tau = 1.0 / sqrt(2.0 * sqrt(dimension))
    tau_prime = 1.0 / sqrt(2.0 * dimension)

    population = _sample_population(rng, size, dimension, lower_bound, upper_bound)
    sigmas = np.full((size, dimension), 0.2 * span, dtype=float)
    fitness = benchmark.evaluate(population)
    evaluations = size
    history = [float(np.min(fitness))]

    while evaluations < budget:
        offspring_count = min(size, budget - evaluations)
        parents = population[:offspring_count]
        parent_sigmas = sigmas[:offspring_count]

        global_noise = rng.normal(size=(offspring_count, 1))
        local_noise = rng.normal(size=(offspring_count, dimension))
        child_sigma = parent_sigmas * np.exp(tau_prime * global_noise + tau * local_noise)
        child_sigma = np.clip(child_sigma, 1e-3 * span, span)
        children = _clip_to_bounds(
            parents + child_sigma * rng.normal(size=(offspring_count, dimension)),
            lower_bound,
            upper_bound,
        )
        child_fitness = benchmark.evaluate(children)
        evaluations += offspring_count

        combined_population = np.vstack([population, children])
        combined_sigmas = np.vstack([sigmas, child_sigma])
        combined_fitness = np.concatenate([fitness, child_fitness])

        wins = _ep_tournament_scores(combined_fitness, rng)
        order = np.lexsort((combined_fitness, -wins))[:size]
        population = combined_population[order]
        sigmas = combined_sigmas[order]
        fitness = combined_fitness[order]
        history.append(float(np.min(fitness)))

    best_index = int(np.argmin(fitness))
    return OptimizerResult(
        best_x=tuple(float(value) for value in population[best_index]),
        best_f=float(fitness[best_index]),
        evaluations=evaluations,
        history=tuple(history),
    )


def _run_de(
    benchmark: CEC2017Benchmark,
    population_size: int,
    budget: int,
    seed: int,
    lower_bound: float,
    upper_bound: float,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    dimension = benchmark.dimension
    size = _effective_population_size(population_size, budget, minimum=4)

    population = _sample_population(rng, size, dimension, lower_bound, upper_bound)
    fitness = benchmark.evaluate(population)
    evaluations = size
    history = [float(np.min(fitness))]
    differential_weight = 0.8
    crossover_rate = 0.9

    while evaluations < budget:
        for target_index in range(size):
            if evaluations >= budget:
                break

            candidates = np.delete(np.arange(size), target_index)
            a, b, c = rng.choice(candidates, size=3, replace=False)
            mutant = population[a] + differential_weight * (population[b] - population[c])
            mutant = _clip_to_bounds(mutant, lower_bound, upper_bound)

            crossover_mask = rng.random(dimension) < crossover_rate
            crossover_mask[rng.integers(0, dimension)] = True
            trial = np.where(crossover_mask, mutant, population[target_index])
            trial_fitness = float(benchmark.evaluate(trial)[0])
            evaluations += 1

            if trial_fitness <= fitness[target_index]:
                population[target_index] = trial
                fitness[target_index] = trial_fitness
        history.append(float(np.min(fitness)))

    best_index = int(np.argmin(fitness))
    return OptimizerResult(
        best_x=tuple(float(value) for value in population[best_index]),
        best_f=float(fitness[best_index]),
        evaluations=evaluations,
        history=tuple(history),
    )


def _run_pso(
    benchmark: CEC2017Benchmark,
    population_size: int,
    budget: int,
    seed: int,
    lower_bound: float,
    upper_bound: float,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    dimension = benchmark.dimension
    size = _effective_population_size(population_size, budget, minimum=2)
    span = upper_bound - lower_bound
    velocity_limit = 0.2 * span

    positions = _sample_population(rng, size, dimension, lower_bound, upper_bound)
    velocities = rng.uniform(-velocity_limit, velocity_limit, size=(size, dimension))
    fitness = benchmark.evaluate(positions)
    evaluations = size

    personal_best_positions = positions.copy()
    personal_best_fitness = fitness.copy()
    global_best_index = int(np.argmin(fitness))
    global_best_position = positions[global_best_index].copy()
    global_best_fitness = float(fitness[global_best_index])

    history = [global_best_fitness]
    total_iterations = max(1, ceil(max(1, budget - size) / size))
    iteration = 0

    while evaluations < budget:
        inertia = 0.9 - 0.5 * min(iteration / total_iterations, 1.0)
        for particle_index in range(size):
            if evaluations >= budget:
                break
            r1 = rng.random(dimension)
            r2 = rng.random(dimension)
            velocities[particle_index] = (
                inertia * velocities[particle_index]
                + 1.7 * r1 * (personal_best_positions[particle_index] - positions[particle_index])
                + 1.7 * r2 * (global_best_position - positions[particle_index])
            )
            velocities[particle_index] = np.clip(
                velocities[particle_index],
                -velocity_limit,
                velocity_limit,
            )
            positions[particle_index] = _clip_to_bounds(
                positions[particle_index] + velocities[particle_index],
                lower_bound,
                upper_bound,
            )
            current_fitness = float(benchmark.evaluate(positions[particle_index])[0])
            evaluations += 1

            if current_fitness <= personal_best_fitness[particle_index]:
                personal_best_positions[particle_index] = positions[particle_index].copy()
                personal_best_fitness[particle_index] = current_fitness
            if current_fitness <= global_best_fitness:
                global_best_position = positions[particle_index].copy()
                global_best_fitness = current_fitness
        history.append(global_best_fitness)
        iteration += 1

    return OptimizerResult(
        best_x=tuple(float(value) for value in global_best_position),
        best_f=global_best_fitness,
        evaluations=evaluations,
        history=tuple(history),
    )


def _run_abc(
    benchmark: CEC2017Benchmark,
    population_size: int,
    budget: int,
    seed: int,
    lower_bound: float,
    upper_bound: float,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    dimension = benchmark.dimension
    size = _effective_population_size(population_size, budget, minimum=2)
    limit = max(5, dimension)

    sources = _sample_population(rng, size, dimension, lower_bound, upper_bound)
    fitness = benchmark.evaluate(sources)
    evaluations = size
    trials = np.zeros(size, dtype=int)
    history = [float(np.min(fitness))]

    while evaluations < budget:
        for source_index in range(size):
            if evaluations >= budget:
                break
            candidate = _abc_neighbor(sources, source_index, rng, lower_bound, upper_bound)
            candidate_fitness = float(benchmark.evaluate(candidate)[0])
            evaluations += 1
            if candidate_fitness <= fitness[source_index]:
                sources[source_index] = candidate
                fitness[source_index] = candidate_fitness
                trials[source_index] = 0
            else:
                trials[source_index] += 1

        probabilities = _abc_selection_probabilities(fitness)
        for _ in range(size):
            if evaluations >= budget:
                break
            source_index = int(rng.choice(size, p=probabilities))
            candidate = _abc_neighbor(sources, source_index, rng, lower_bound, upper_bound)
            candidate_fitness = float(benchmark.evaluate(candidate)[0])
            evaluations += 1
            if candidate_fitness <= fitness[source_index]:
                sources[source_index] = candidate
                fitness[source_index] = candidate_fitness
                trials[source_index] = 0
            else:
                trials[source_index] += 1

        scout_indices = np.where(trials >= limit)[0]
        for source_index in scout_indices:
            if evaluations >= budget:
                break
            sources[source_index] = _sample_population(
                rng,
                1,
                dimension,
                lower_bound,
                upper_bound,
            )[0]
            fitness[source_index] = float(benchmark.evaluate(sources[source_index])[0])
            trials[source_index] = 0
            evaluations += 1

        history.append(float(np.min(fitness)))

    best_index = int(np.argmin(fitness))
    return OptimizerResult(
        best_x=tuple(float(value) for value in sources[best_index]),
        best_f=float(fitness[best_index]),
        evaluations=evaluations,
        history=tuple(history),
    )


def _effective_population_size(population_size: int, budget: int, minimum: int) -> int:
    if population_size < minimum:
        raise ValueError(f"population_size deve ser pelo menos {minimum} para este algoritmo.")
    if budget < minimum:
        raise ValueError(f"budget deve ser pelo menos {minimum} para este algoritmo.")
    return min(population_size, budget)


def _sample_population(
    rng: np.random.Generator,
    size: int,
    dimension: int,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    return np.asarray(rng.uniform(lower_bound, upper_bound, size=(size, dimension)), dtype=float)


def _clip_to_bounds(values: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    return np.asarray(np.clip(values, lower_bound, upper_bound), dtype=float)


def _ep_tournament_scores(fitness: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = len(fitness)
    opponents = min(10, size - 1)
    wins = np.zeros(size, dtype=int)
    for index in range(size):
        candidates = np.delete(np.arange(size), index)
        sampled = rng.choice(candidates, size=opponents, replace=False)
        wins[index] = int(np.sum(fitness[index] <= fitness[sampled]))
    return wins


def _abc_neighbor(
    sources: np.ndarray,
    source_index: int,
    rng: np.random.Generator,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    size, dimension = sources.shape
    other_indices = np.delete(np.arange(size), source_index)
    other_index = int(rng.choice(other_indices))
    phi = rng.uniform(-1.0, 1.0, size=dimension)
    candidate = sources[source_index] + phi * (sources[source_index] - sources[other_index])
    return _clip_to_bounds(candidate, lower_bound, upper_bound)


def _abc_selection_probabilities(fitness: np.ndarray) -> np.ndarray:
    shifted = fitness - np.min(fitness)
    weights = 1.0 / (1.0 + shifted)
    return np.asarray(weights / np.sum(weights), dtype=float)
