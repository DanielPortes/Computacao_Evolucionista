"""CLI principal do repositorio consolidado."""

from __future__ import annotations

from typing import cast

import typer

from ce.ex02_cec2017.algorithms import ContinuousAlgorithmName

app = typer.Typer(
    add_completion=False,
    help="Interface de linha de comando do repositorio consolidado.",
    no_args_is_help=True,
)
EX02_ALGORITHM_OPTION = typer.Option(None, "--algorithm")
EX02_FUNCTION_ID_OPTION = typer.Option(None, "--function-id")
EX02_DIMENSION_OPTION = typer.Option(None, "--dimension")
EX02_BUDGET_OPTION = typer.Option(10000, min=1)
EX02_RUNS_OPTION = typer.Option(51, min=1)
EX02_POPULATION_OPTION = typer.Option(50, min=2)
EX02_SEED_OPTION = typer.Option(42, min=0)
EX03_INSTANCE_OPTION = typer.Option(None, "--instance")
EX03_ALGORITHM_OPTION = typer.Option(None, "--algorithm")
EX03_POPULATION_OPTION = typer.Option(100, min=2)
EX03_GENERATIONS_OPTION = typer.Option(200, min=1)
EX03_CROSSOVER_OPTION = typer.Option(0.7, min=0.0, max=1.0)
EX03_MUTATION_OPTION = typer.Option(0.05, min=0.0, max=1.0)
EX03_TOURNAMENT_OPTION = typer.Option(3, min=2)
EX03_ELITE_OPTION = typer.Option(1, min=1)
EX03_SEED_OPTION = typer.Option(42, min=0)
EX04_STATION_OPTION = typer.Option("rola_moca", "--station")
EX04_EPOCHS_OPTION = typer.Option(10, min=1)
EX04_PATIENCE_OPTION = typer.Option(3, min=1)
EX04_BATCH_OPTION = typer.Option(32, min=1)
EX04_MAX_ROWS_OPTION = typer.Option(240, min=64)
EX04_SEED_OPTION = typer.Option(42, min=0)
EX05_POPULATION_OPTION = typer.Option(8, min=2)
EX05_GENERATIONS_OPTION = typer.Option(2, min=1)
EX05_SPLITS_OPTION = typer.Option(3, min=2)
EX05_EPOCHS_OPTION = typer.Option(5, min=1)
EX05_PATIENCE_OPTION = typer.Option(2, min=1)
EX05_MAX_ROWS_OPTION = typer.Option(240, min=64)
EX05_SEED_OPTION = typer.Option(42, min=0)


@app.callback()
def main() -> None:
    """Agrupa os subcomandos do repositorio consolidado."""


@app.command()
def info() -> None:
    """Exibe o estado atual do repositorio consolidado."""
    typer.echo("Repositorio consolidado e operacional para CE2, CE3, CE4 e CE5.")


@app.command("ex02")
def run_ex02(
    algorithm: list[str] | None = EX02_ALGORITHM_OPTION,
    function_id: list[int] | None = EX02_FUNCTION_ID_OPTION,
    dimension: list[int] | None = EX02_DIMENSION_OPTION,
    budget_multiplier: int = EX02_BUDGET_OPTION,
    n_runs: int = EX02_RUNS_OPTION,
    population_size: int = EX02_POPULATION_OPTION,
    base_seed: int = EX02_SEED_OPTION,
) -> None:
    """Executa o modulo canonico do CE2."""

    from ce.ex02_cec2017.run import (
        DEFAULT_DIMENSIONS,
        DEFAULT_FUNCTION_IDS,
        ExperimentConfig,
        run_suite,
    )

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
                    result.algorithm_name.upper(),
                    f"f{result.function_id}",
                    f"D={result.dimension}",
                    f"budget={result.budget}",
                    f"min={result.summary.minimum:.6f}",
                    f"max={result.summary.maximum:.6f}",
                    f"mean={result.summary.mean:.6f}",
                ]
            )
        )


@app.command("ex03")
def run_ex03(
    algorithm: list[str] | None = EX03_ALGORITHM_OPTION,
    instance: list[str] | None = EX03_INSTANCE_OPTION,
    population_size: int = EX03_POPULATION_OPTION,
    generations: int = EX03_GENERATIONS_OPTION,
    crossover_rate: float = EX03_CROSSOVER_OPTION,
    mutation_rate: float = EX03_MUTATION_OPTION,
    tournament_size: int = EX03_TOURNAMENT_OPTION,
    elite_size: int = EX03_ELITE_OPTION,
    base_seed: int = EX03_SEED_OPTION,
) -> None:
    """Executa o modulo canonico do CE3."""

    from ce.ex03_tsp.run import DEFAULT_INSTANCES, ExperimentConfig, solve_instances

    config = ExperimentConfig(
        algorithm_names=tuple(algorithm or ("ga",)),
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
        ]
        if result.relative_error_percent is not None:
            fields.append(f"relative_error={result.relative_error_percent:.4f}%")
        typer.echo(" ".join(fields))


@app.command("ex04")
def run_ex04(
    station: str = EX04_STATION_OPTION,
    max_epochs: int = EX04_EPOCHS_OPTION,
    patience: int = EX04_PATIENCE_OPTION,
    batch_size: int = EX04_BATCH_OPTION,
    max_rows: int = EX04_MAX_ROWS_OPTION,
    seed: int = EX04_SEED_OPTION,
) -> None:
    """Executa o baseline temporal do CE4."""

    from ce.ex04_forecasting.evaluate import run_baseline
    from ce.ex04_forecasting.train import ForecastConfig

    result = run_baseline(
        ForecastConfig(
            station=station,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            max_rows=max_rows,
            seed=seed,
        )
    )
    typer.echo(
        " ".join(
            [
                f"station={station}",
                f"val_rmse={result.validation_metrics.rmse:.6f}",
                f"test_rmse={result.test_metrics.rmse:.6f}",
                f"best_epoch={result.training.best_epoch}",
            ]
        )
    )


@app.command("ex05")
def run_ex05(
    population_size: int = EX05_POPULATION_OPTION,
    generations: int = EX05_GENERATIONS_OPTION,
    n_splits: int = EX05_SPLITS_OPTION,
    max_epochs: int = EX05_EPOCHS_OPTION,
    patience: int = EX05_PATIENCE_OPTION,
    max_rows: int = EX05_MAX_ROWS_OPTION,
    seed: int = EX05_SEED_OPTION,
) -> None:
    """Executa o HPO temporal do CE5."""

    from ce.ex05_hpo.objective import TemporalCVConfig
    from ce.ex05_hpo.search import EvolutionSearchConfig, run_search

    result = run_search(
        search_config=EvolutionSearchConfig(
            population_size=population_size,
            generations=generations,
            seed=seed,
        ),
        objective_config=TemporalCVConfig(
            n_splits=n_splits,
            max_epochs=max_epochs,
            patience=patience,
            max_rows=max_rows,
            seed=seed,
        ),
    )
    typer.echo(
        " ".join(
            [
                f"best_score={result.best_score:.6f}",
                f"baseline_score={result.baseline_score:.6f}",
                f"lookback={result.best_params.lookback}",
                f"hidden={result.best_params.hidden_size}",
                f"decoder={result.best_params.second_hidden_size}",
                f"dropout={result.best_params.dropout:.2f}",
                f"lr={result.best_params.learning_rate}",
                f"batch={result.best_params.batch_size}",
            ]
        )
    )


if __name__ == "__main__":
    app()
