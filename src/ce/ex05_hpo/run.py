"""CLI do HPO temporal do CE5."""

from __future__ import annotations

import typer

from ce.ex05_hpo.objective import TemporalCVConfig
from ce.ex05_hpo.search import EvolutionSearchConfig, HPOResult, run_search

app = typer.Typer(add_completion=False, help="Executa o HPO evolucionario do exercicio CE5.")
POPULATION_SIZE_OPTION = typer.Option(8, min=2)
GENERATIONS_OPTION = typer.Option(2, min=1)
N_SPLITS_OPTION = typer.Option(3, min=2)
MAX_EPOCHS_OPTION = typer.Option(5, min=1)
PATIENCE_OPTION = typer.Option(2, min=1)
MAX_ROWS_OPTION = typer.Option(240, min=64)
BASE_SEED_OPTION = typer.Option(42, min=0)


def run_hpo(
    search_config: EvolutionSearchConfig,
    objective_config: TemporalCVConfig,
) -> HPOResult:
    """Executa o HPO e retorna o resultado agregado."""

    return run_search(search_config=search_config, objective_config=objective_config)


@app.command()
def main(
    population_size: int = POPULATION_SIZE_OPTION,
    generations: int = GENERATIONS_OPTION,
    n_splits: int = N_SPLITS_OPTION,
    max_epochs: int = MAX_EPOCHS_OPTION,
    patience: int = PATIENCE_OPTION,
    max_rows: int = MAX_ROWS_OPTION,
    seed: int = BASE_SEED_OPTION,
) -> None:
    """Executa uma busca evolucionaria barata e imprime o melhor resultado."""

    search_config = EvolutionSearchConfig(
        population_size=population_size,
        generations=generations,
        seed=seed,
    )
    objective_config = TemporalCVConfig(
        n_splits=n_splits,
        max_epochs=max_epochs,
        patience=patience,
        max_rows=max_rows,
        seed=seed,
    )
    result = run_hpo(search_config, objective_config)
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
