"""Execucao canonica do exercicio CE3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import typer
from pydantic import BaseModel, Field, field_validator

from ce.ex03_tsp.aco import ACORunResult, AntColonyConfig
from ce.ex03_tsp.aco import solve_problem as solve_aco
from ce.ex03_tsp.data import available_instances
from ce.ex03_tsp.ga import GARunResult, GeneticAlgorithmConfig, solve_problem
from ce.ex03_tsp.metrics import best_known_value, relative_error
from ce.ex03_tsp.problem import build_problem

DEFAULT_INSTANCES = tuple(sorted(available_instances()))
app = typer.Typer(add_completion=False, help="Executa o modulo canonico TSP/ATSP do exercicio CE3.")
ALGORITHM_OPTION = typer.Option(
    None,
    "--algorithm",
    help="Repita a opcao para selecionar algoritmos especificos; omitido usa apenas GA.",
)
INSTANCE_OPTION = typer.Option(
    None,
    "--instance",
    help="Repita a opcao para selecionar instancias especificas; omitido usa todas.",
)
POPULATION_SIZE_OPTION = typer.Option(100, min=2)
GENERATIONS_OPTION = typer.Option(200, min=1)
CROSSOVER_RATE_OPTION = typer.Option(0.7, min=0.0, max=1.0)
MUTATION_RATE_OPTION = typer.Option(0.05, min=0.0, max=1.0)
TOURNAMENT_SIZE_OPTION = typer.Option(3, min=2)
ELITE_SIZE_OPTION = typer.Option(1, min=1)
BASE_SEED_OPTION = typer.Option(42, min=0)


class ExperimentConfig(BaseModel):
    """Configuracao declarativa da execucao TSP/ATSP."""

    algorithm_names: tuple[str, ...] = ("ga",)
    instance_names: tuple[str, ...] = DEFAULT_INSTANCES
    population_size: int = Field(default=100, ge=2)
    generations: int = Field(default=200, ge=1)
    crossover_rate: float = Field(default=0.7, ge=0.0, le=1.0)
    mutation_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    tournament_size: int = Field(default=3, ge=2)
    elite_size: int = Field(default=1, ge=1)
    base_seed: int = Field(default=42, ge=0)

    @field_validator("algorithm_names")
    @classmethod
    def _validate_algorithm_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        invalid = sorted(set(value) - {"ga", "aco"})
        if invalid:
            raise ValueError(f"Algoritmos nao suportados: {invalid}.")
        return value

    @field_validator("instance_names")
    @classmethod
    def _validate_instance_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        invalid = sorted(set(value) - set(available_instances()))
        if invalid:
            raise ValueError(f"Instancias nao suportadas: {invalid}.")
        return value


@dataclass(frozen=True)
class InstanceRunResult:
    """Resultado agregado por instancia."""

    algorithm_name: str
    instance_name: str
    problem_type: str
    best_cost: float
    best_route: tuple[int, ...]
    relative_error_percent: float | None
    history_length: int


def solve_instances(config: ExperimentConfig) -> tuple[InstanceRunResult, ...]:
    """Executa o GA para o conjunto de instancias configurado."""

    results: list[InstanceRunResult] = []
    for algorithm_offset, algorithm_name in enumerate(config.algorithm_names):
        for instance_offset, instance_name in enumerate(config.instance_names):
            results.append(
                _solve_single_instance(
                    config,
                    algorithm_name=algorithm_name,
                    instance_name=instance_name,
                    seed=config.base_seed + algorithm_offset * 1000 + instance_offset,
                )
            )
    return tuple(results)


def _solve_single_instance(
    config: ExperimentConfig,
    algorithm_name: str,
    instance_name: str,
    seed: int,
) -> InstanceRunResult:
    problem = build_problem(instance_name)
    if algorithm_name == "ga":
        ga_config = GeneticAlgorithmConfig(
            population_size=config.population_size,
            generations=config.generations,
            crossover_rate=config.crossover_rate,
            mutation_rate=config.mutation_rate,
            tournament_size=config.tournament_size,
            elite_size=config.elite_size,
            seed=seed,
        )
        run_result: GARunResult | ACORunResult = solve_problem(problem, ga_config)
    else:
        aco_config = AntColonyConfig(
            ant_count=config.population_size,
            iterations=config.generations,
            seed=seed,
        )
        run_result = solve_aco(problem, aco_config)
    reference = best_known_value(instance_name)
    relative = relative_error(run_result.best_cost, reference) if reference is not None else None

    return InstanceRunResult(
        algorithm_name=algorithm_name,
        instance_name=instance_name,
        problem_type=problem.problem_type,
        best_cost=run_result.best_cost,
        best_route=run_result.best_route,
        relative_error_percent=relative,
        history_length=len(run_result.history),
    )


@app.command()
def main(
    algorithm: list[str] | None = ALGORITHM_OPTION,
    instance: list[str] | None = INSTANCE_OPTION,
    population_size: int = POPULATION_SIZE_OPTION,
    generations: int = GENERATIONS_OPTION,
    crossover_rate: float = CROSSOVER_RATE_OPTION,
    mutation_rate: float = MUTATION_RATE_OPTION,
    tournament_size: int = TOURNAMENT_SIZE_OPTION,
    elite_size: int = ELITE_SIZE_OPTION,
    base_seed: int = BASE_SEED_OPTION,
) -> None:
    """Executa as instancias selecionadas e imprime um resumo por arquivo."""

    config = ExperimentConfig(
        algorithm_names=cast(tuple[str, ...], tuple(algorithm or ("ga",))),
        instance_names=tuple(instance or DEFAULT_INSTANCES),
        population_size=population_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        elite_size=elite_size,
        base_seed=base_seed,
    )

    for result in solve_instances(config):
        fields = [
            result.algorithm_name.upper(),
            result.instance_name,
            result.problem_type,
            f"best={result.best_cost:.2f}",
            f"history={result.history_length}",
        ]
        if result.relative_error_percent is not None:
            fields.append(f"relative_error={result.relative_error_percent:.4f}%")
        typer.echo(" ".join(fields))


if __name__ == "__main__":
    app()
