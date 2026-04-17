"""Orquestracao consolidada dos exercicios para notebook e analise."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, Field

from ce.ex02_cec2017.algorithms import ContinuousAlgorithmName
from ce.ex02_cec2017.run import ExperimentConfig as Ex02Config
from ce.ex02_cec2017.run import run_suite as run_ex02_suite
from ce.ex03_tsp.run import ExperimentConfig as Ex03Config
from ce.ex03_tsp.run import solve_instances
from ce.ex04_forecasting.evaluate import run_baseline
from ce.ex04_forecasting.gp import GPConfig, run_gp_baseline
from ce.ex04_forecasting.train import ForecastConfig
from ce.ex05_hpo.objective import TemporalCVConfig
from ce.ex05_hpo.search import EvolutionSearchConfig, run_search


class NotebookRunConfig(BaseModel):
    """Configuracao barata e reproduzivel para execucao consolidada."""

    ex02_algorithms: tuple[ContinuousAlgorithmName, ...] = ("ga", "es", "ep", "de", "pso", "abc")
    ex02_function_ids: tuple[int, ...] = (3,)
    ex02_dimensions: tuple[int, ...] = (2,)
    ex02_budget_multiplier: int = Field(default=6, ge=1)
    ex02_n_runs: int = Field(default=1, ge=1)
    ex02_population_size: int = Field(default=6, ge=2)
    ex02_seed: int = Field(default=7, ge=0)

    ex03_algorithms: tuple[str, ...] = ("ga", "aco")
    ex03_instances: tuple[str, ...] = ("berlin52",)
    ex03_population_size: int = Field(default=12, ge=2)
    ex03_generations: int = Field(default=3, ge=1)
    ex03_seed: int = Field(default=7, ge=0)

    ex04_station: str = "rola_moca"
    ex04_max_rows: int = Field(default=64, ge=64)
    ex04_lstm_epochs: int = Field(default=1, ge=1)
    ex04_lstm_patience: int = Field(default=1, ge=1)
    ex04_lstm_batch_size: int = Field(default=8, ge=1)
    ex04_gp_population_size: int = Field(default=8, ge=4)
    ex04_gp_generations: int = Field(default=3, ge=1)
    ex04_gp_max_depth: int = Field(default=3, ge=1)
    ex04_seed: int = Field(default=7, ge=0)

    ex05_population_size: int = Field(default=2, ge=2)
    ex05_generations: int = Field(default=1, ge=1)
    ex05_n_splits: int = Field(default=2, ge=2)
    ex05_max_epochs: int = Field(default=1, ge=1)
    ex05_patience: int = Field(default=1, ge=1)
    ex05_max_rows: int = Field(default=64, ge=64)
    ex05_seed: int = Field(default=7, ge=0)


@dataclass(frozen=True)
class RunAllResults:
    """Colecao de DataFrames prontos para notebook e relatórios."""

    ex02: pd.DataFrame
    ex03: pd.DataFrame
    ex04: pd.DataFrame
    ex05: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "ce2": self.ex02,
            "ce3": self.ex03,
            "ce4": self.ex04,
            "ce5": self.ex05,
        }


def run_all_exercises(config: NotebookRunConfig | None = None) -> RunAllResults:
    """Executa os quatro exercicios canônicos com budgets reduzidos."""

    effective_config = config or NotebookRunConfig()

    ex02_results = run_ex02_suite(
        Ex02Config(
            algorithm_names=effective_config.ex02_algorithms,
            function_ids=effective_config.ex02_function_ids,
            dimensions=effective_config.ex02_dimensions,
            budget_multiplier=effective_config.ex02_budget_multiplier,
            n_runs=effective_config.ex02_n_runs,
            population_size=effective_config.ex02_population_size,
            base_seed=effective_config.ex02_seed,
        )
    )
    ex02_frame = pd.DataFrame(
        [
            {
                "algorithm": result.algorithm_name.upper(),
                "function_id": result.function_id,
                "dimension": result.dimension,
                "budget": result.budget,
                "minimum": result.summary.minimum,
                "maximum": result.summary.maximum,
                "mean": result.summary.mean,
                "median": result.summary.median,
                "std": result.summary.std,
            }
            for result in ex02_results
        ]
    )

    ex03_results = solve_instances(
        Ex03Config(
            algorithm_names=effective_config.ex03_algorithms,
            instance_names=effective_config.ex03_instances,
            population_size=effective_config.ex03_population_size,
            generations=effective_config.ex03_generations,
            base_seed=effective_config.ex03_seed,
        )
    )
    ex03_frame = pd.DataFrame(
        [
            {
                "algorithm": result.algorithm_name.upper(),
                "instance": result.instance_name,
                "problem_type": result.problem_type,
                "best_cost": result.best_cost,
                "relative_error_percent": result.relative_error_percent,
                "history_length": result.history_length,
            }
            for result in ex03_results
        ]
    )

    lstm_result = run_baseline(
        ForecastConfig(
            station=effective_config.ex04_station,
            max_rows=effective_config.ex04_max_rows,
            max_epochs=effective_config.ex04_lstm_epochs,
            patience=effective_config.ex04_lstm_patience,
            batch_size=effective_config.ex04_lstm_batch_size,
            seed=effective_config.ex04_seed,
        )
    )
    gp_result = run_gp_baseline(
        GPConfig(
            station=effective_config.ex04_station,
            max_rows=effective_config.ex04_max_rows,
            population_size=effective_config.ex04_gp_population_size,
            generations=effective_config.ex04_gp_generations,
            max_depth=effective_config.ex04_gp_max_depth,
            seed=effective_config.ex04_seed,
        )
    )
    ex04_frame = pd.DataFrame(
        [
            {
                "model": "LSTM",
                "station": effective_config.ex04_station,
                "validation_rmse": lstm_result.validation_metrics.rmse,
                "validation_mae": lstm_result.validation_metrics.mae,
                "test_rmse": lstm_result.test_metrics.rmse,
                "test_mae": lstm_result.test_metrics.mae,
                "detail": f"best_epoch={lstm_result.training.best_epoch}",
            },
            {
                "model": "GP",
                "station": effective_config.ex04_station,
                "validation_rmse": gp_result.validation_metrics.rmse,
                "validation_mae": gp_result.validation_metrics.mae,
                "test_rmse": gp_result.test_metrics.rmse,
                "test_mae": gp_result.test_metrics.mae,
                "detail": gp_result.best_expression,
            },
        ]
    )

    ex05_result = run_search(
        search_config=EvolutionSearchConfig(
            population_size=effective_config.ex05_population_size,
            generations=effective_config.ex05_generations,
            seed=effective_config.ex05_seed,
        ),
        objective_config=TemporalCVConfig(
            n_splits=effective_config.ex05_n_splits,
            max_epochs=effective_config.ex05_max_epochs,
            patience=effective_config.ex05_patience,
            max_rows=effective_config.ex05_max_rows,
            seed=effective_config.ex05_seed,
        ),
    )
    ex05_frame = pd.DataFrame(
        [
            {
                "best_score": ex05_result.best_score,
                "baseline_score": ex05_result.baseline_score,
                "lookback": ex05_result.best_params.lookback,
                "hidden_size": ex05_result.best_params.hidden_size,
                "second_hidden_size": ex05_result.best_params.second_hidden_size,
                "dropout": ex05_result.best_params.dropout,
                "learning_rate": ex05_result.best_params.learning_rate,
                "batch_size": ex05_result.best_params.batch_size,
            }
        ]
    )

    return RunAllResults(
        ex02=ex02_frame,
        ex03=ex03_frame,
        ex04=ex04_frame,
        ex05=ex05_frame,
    )


def build_plotly_figures(results: RunAllResults) -> dict[str, go.Figure]:
    """Constroi figuras `plotly` a partir dos resultados consolidados."""

    ex02_figure = px.bar(
        results.ex02,
        x="algorithm",
        y="mean",
        color="algorithm",
        title="CE2 - Benchmark continuo por algoritmo",
    )
    ex02_figure.update_layout(
        showlegend=False,
        xaxis_title="Algoritmo",
        yaxis_title="Fitness medio",
    )

    ex03_figure = px.bar(
        results.ex03,
        x="algorithm",
        y="best_cost",
        color="algorithm",
        title="CE3 - Melhor custo por algoritmo",
    )
    ex03_figure.update_layout(showlegend=False, xaxis_title="Algoritmo", yaxis_title="Custo")

    ex04_plot_frame = results.ex04.melt(
        id_vars=["model"],
        value_vars=["validation_rmse", "test_rmse"],
        var_name="split",
        value_name="rmse",
    )
    ex04_figure = px.bar(
        ex04_plot_frame,
        x="model",
        y="rmse",
        color="split",
        barmode="group",
        title="CE4 - RMSE de validacao e teste",
    )
    ex04_figure.update_layout(xaxis_title="Modelo", yaxis_title="RMSE")

    ex05_plot_frame = pd.DataFrame(
        [
            {"metric": "baseline_score", "value": float(results.ex05.iloc[0]["baseline_score"])},
            {"metric": "best_score", "value": float(results.ex05.iloc[0]["best_score"])},
        ]
    )
    ex05_figure = px.bar(
        ex05_plot_frame,
        x="metric",
        y="value",
        color="metric",
        title="CE5 - HPO versus baseline",
    )
    ex05_figure.update_layout(showlegend=False, xaxis_title="Metrica", yaxis_title="Score")

    return {
        "ce2": ex02_figure,
        "ce3": ex03_figure,
        "ce4": ex04_figure,
        "ce5": ex05_figure,
    }
