"""Execucao canonica do exercicio CE2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import typer
from pydantic import BaseModel, Field, field_validator, model_validator

from ce.common.seeds import set_global_seed
from ce.ex02_cec2017.algorithms import (
    ContinuousAlgorithmName,
    algorithm_label,
    available_algorithms,
    optimize_benchmark,
)
from ce.ex02_cec2017.benchmarks import get_benchmark, supported_dimensions, supported_function_ids

DEFAULT_FUNCTION_IDS = supported_function_ids()
DEFAULT_DIMENSIONS = (2, 10)
app = typer.Typer(add_completion=False, help="Executa o benchmark canonico do exercicio CE2.")
ALGORITHM_OPTION = typer.Option(
    None,
    "--algorithm",
    help="Repita a opcao para selecionar algoritmos especificos; omitido usa apenas GA.",
)
FUNCTION_ID_OPTION = typer.Option(
    None,
    "--function-id",
    help="Repita a opcao para selecionar funcoes especificas; omitido usa o conjunto padrao.",
)
DIMENSION_OPTION = typer.Option(
    None,
    "--dimension",
    help="Repita a opcao para selecionar dimensoes especificas; omitido usa o conjunto padrao.",
)
BUDGET_MULTIPLIER_OPTION = typer.Option(10000, min=1)
N_RUNS_OPTION = typer.Option(51, min=1)
POPULATION_SIZE_OPTION = typer.Option(50, min=2)
BASE_SEED_OPTION = typer.Option(42, min=0)


class ExperimentConfig(BaseModel):
    """Configuracao declarativa para execucoes reproduziveis de CE2."""

    algorithm_names: tuple[ContinuousAlgorithmName, ...] = ("ga",)
    function_ids: tuple[int, ...] = DEFAULT_FUNCTION_IDS
    dimensions: tuple[int, ...] = DEFAULT_DIMENSIONS
    budget_multiplier: int = Field(default=10000, ge=1)
    n_runs: int = Field(default=51, ge=1)
    population_size: int = Field(default=50, ge=2)
    base_seed: int = Field(default=42, ge=0)
    lower_bound: float = -100.0
    upper_bound: float = 100.0

    @field_validator("function_ids")
    @classmethod
    def _validate_function_ids(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        invalid = sorted(set(value) - set(supported_function_ids()))
        if invalid:
            raise ValueError(f"Funcoes nao suportadas: {invalid}.")
        return value

    @field_validator("algorithm_names")
    @classmethod
    def _validate_algorithm_names(
        cls,
        value: tuple[ContinuousAlgorithmName, ...],
    ) -> tuple[ContinuousAlgorithmName, ...]:
        invalid = sorted(set(value) - set(available_algorithms()))
        if invalid:
            raise ValueError(f"Algoritmos nao suportados: {invalid}.")
        return value

    @field_validator("dimensions")
    @classmethod
    def _validate_dimensions(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        invalid = sorted(set(value) - set(supported_dimensions()))
        if invalid:
            raise ValueError(f"Dimensoes nao suportadas: {invalid}.")
        return value

    @model_validator(mode="after")
    def _validate_bounds(self) -> ExperimentConfig:
        if self.lower_bound >= self.upper_bound:
            raise ValueError("lower_bound deve ser estritamente menor que upper_bound.")
        return self

    def budget_for_dimension(self, dimension: int) -> int:
        return dimension * self.budget_multiplier


@dataclass(frozen=True)
class RunRecord:
    """Resultado do melhor individuo em uma execucao."""

    seed: int
    best_x: tuple[float, ...]
    best_f: float
    evaluations: int
    history: tuple[float, ...]


@dataclass(frozen=True)
class SummaryStats:
    """Estatisticas agregadas a partir do melhor valor de cada execucao."""

    minimum: float
    maximum: float
    mean: float
    median: float
    std: float


@dataclass(frozen=True)
class ExperimentResult:
    """Resultado agregado por funcao e dimensao."""

    algorithm_name: ContinuousAlgorithmName
    function_id: int
    function_name: str
    dimension: int
    budget: int
    optimum: float
    runs: tuple[RunRecord, ...]
    summary: SummaryStats


def run_suite(config: ExperimentConfig) -> tuple[ExperimentResult, ...]:
    """Executa o conjunto configurado de funcoes e dimensoes."""

    results: list[ExperimentResult] = []
    for algorithm_name in config.algorithm_names:
        for function_id in config.function_ids:
            for dimension in config.dimensions:
                results.append(
                    _run_single_experiment(
                        config,
                        algorithm_name=algorithm_name,
                        function_id=function_id,
                        dimension=dimension,
                    )
                )
    return tuple(results)


def _run_single_experiment(
    config: ExperimentConfig,
    algorithm_name: ContinuousAlgorithmName,
    function_id: int,
    dimension: int,
) -> ExperimentResult:
    benchmark = get_benchmark(function_id=function_id, dimension=dimension)
    budget = config.budget_for_dimension(dimension)

    run_records: list[RunRecord] = []
    for offset in range(config.n_runs):
        seed = config.base_seed + offset
        set_global_seed(seed)
        optimization = optimize_benchmark(
            algorithm_name=algorithm_name,
            benchmark=benchmark,
            population_size=config.population_size,
            budget=budget,
            seed=seed,
            lower_bound=config.lower_bound,
            upper_bound=config.upper_bound,
        )
        run_records.append(
            RunRecord(
                seed=seed,
                best_x=optimization.best_x,
                best_f=optimization.best_f,
                evaluations=optimization.evaluations,
                history=optimization.history,
            )
        )

    summary = _summarize_runs(run_records)
    return ExperimentResult(
        algorithm_name=algorithm_name,
        function_id=function_id,
        function_name=benchmark.name,
        dimension=dimension,
        budget=budget,
        optimum=benchmark.optimum,
        runs=tuple(run_records),
        summary=summary,
    )


def _summarize_runs(run_records: list[RunRecord]) -> SummaryStats:
    best_values = np.asarray([record.best_f for record in run_records], dtype=float)
    return SummaryStats(
        minimum=float(np.min(best_values)),
        maximum=float(np.max(best_values)),
        mean=float(np.mean(best_values)),
        median=float(np.median(best_values)),
        std=float(np.std(best_values)),
    )


@app.command()
def main(
    algorithm: list[str] | None = ALGORITHM_OPTION,
    function_id: list[int] | None = FUNCTION_ID_OPTION,
    dimension: list[int] | None = DIMENSION_OPTION,
    budget_multiplier: int = BUDGET_MULTIPLIER_OPTION,
    n_runs: int = N_RUNS_OPTION,
    population_size: int = POPULATION_SIZE_OPTION,
    base_seed: int = BASE_SEED_OPTION,
) -> None:
    """Executa o benchmark e imprime um resumo por funcao/dimensao."""

    config = ExperimentConfig(
        algorithm_names=cast(tuple[ContinuousAlgorithmName, ...], tuple(algorithm or ("ga",))),
        function_ids=tuple(function_id or DEFAULT_FUNCTION_IDS),
        dimensions=tuple(dimension or DEFAULT_DIMENSIONS),
        budget_multiplier=budget_multiplier,
        n_runs=n_runs,
        population_size=population_size,
        base_seed=base_seed,
    )

    for result in run_suite(config):
        typer.echo(
            " ".join(
                [
                    algorithm_label(result.algorithm_name),
                    f"f{result.function_id}",
                    f"D={result.dimension}",
                    f"budget={result.budget}",
                    f"min={result.summary.minimum:.6f}",
                    f"max={result.summary.maximum:.6f}",
                    f"mean={result.summary.mean:.6f}",
                    f"median={result.summary.median:.6f}",
                    f"std={result.summary.std:.6f}",
                ]
            )
        )


if __name__ == "__main__":
    app()
