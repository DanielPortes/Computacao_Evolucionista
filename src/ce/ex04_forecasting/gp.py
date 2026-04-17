"""Baseline de Programacao Genetica para o CE4."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, Field

from ce.common.seeds import set_global_seed
from ce.ex04_forecasting.evaluate import RegressionMetrics, compute_regression_metrics
from ce.ex04_forecasting.train import ForecastConfig, PreparedData, prepare_data

OperatorName = Literal["add", "sub", "mul", "div", "sin", "cos", "neg"]


@dataclass(frozen=True)
class PrimitiveSpec:
    """Descricao de um operador do conjunto funcional."""

    name: OperatorName
    arity: int
    function: Callable[..., np.ndarray]
    label: str


@dataclass(frozen=True)
class GPNode:
    """No de uma arvore de regressao simbolica."""

    kind: Literal["feature", "constant", "operator"]
    value: int | float | OperatorName
    children: tuple[GPNode, ...] = ()


class GPConfig(BaseModel):
    """Configuracao declarativa do baseline de GP."""

    station: str = "rola_moca"
    lookback: int = Field(default=3, ge=1)
    train_fraction: float = Field(default=0.7, gt=0.0, lt=1.0)
    validation_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    population_size: int = Field(default=40, ge=4)
    generations: int = Field(default=20, ge=1)
    tournament_size: int = Field(default=3, ge=2)
    max_depth: int = Field(default=4, ge=1)
    crossover_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    mutation_rate: float = Field(default=0.2, ge=0.0, le=1.0)
    parsimony_penalty: float = Field(default=1e-3, ge=0.0)
    seed: int = Field(default=42, ge=0)
    max_rows: int | None = Field(default=None, ge=32)


@dataclass(frozen=True)
class GPForecastResult:
    """Resultado final do baseline de GP."""

    best_expression: str
    best_tree_size: int
    validation_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    history: tuple[float, ...]


@dataclass(frozen=True)
class DatasetMatrices:
    """Matrizes achatadas derivadas das janelas temporais."""

    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


@dataclass(frozen=True)
class IndividualScore:
    """Fitness de um individuo sobre treino e validacao."""

    tree: GPNode
    train_mse: float
    validation_mse: float
    adjusted_fitness: float


PRIMITIVES: tuple[PrimitiveSpec, ...] = (
    PrimitiveSpec("add", 2, lambda left, right: left + right, "+"),
    PrimitiveSpec("sub", 2, lambda left, right: left - right, "-"),
    PrimitiveSpec("mul", 2, lambda left, right: left * right, "*"),
    PrimitiveSpec(
        "div",
        2,
        lambda left, right: np.divide(
            left,
            right,
            out=np.zeros_like(left),
            where=np.abs(right) > 1e-6,
        ),
        "/",
    ),
    PrimitiveSpec("sin", 1, np.sin, "sin"),
    PrimitiveSpec("cos", 1, np.cos, "cos"),
    PrimitiveSpec("neg", 1, np.negative, "neg"),
)
PRIMITIVES_BY_NAME = {primitive.name: primitive for primitive in PRIMITIVES}


def run_gp_baseline(config: GPConfig) -> GPForecastResult:
    """Executa um baseline de regressao simbolica temporal."""

    set_global_seed(config.seed)
    rng = random.Random(config.seed)
    prepared_data = prepare_data(
        ForecastConfig(
            station=config.station,
            lookback=config.lookback,
            train_fraction=config.train_fraction,
            validation_fraction=config.validation_fraction,
            seed=config.seed,
            max_rows=config.max_rows,
        )
    )
    matrices = _flatten_prepared_data(prepared_data)
    feature_count = matrices.train_x.shape[1]

    population = [
        _random_tree(rng, config.max_depth, feature_count, force_function=True)
        for _ in range(config.population_size)
    ]
    scored_population = [
        _score_individual(tree, matrices, parsimony_penalty=config.parsimony_penalty)
        for tree in population
    ]
    best_by_validation = min(scored_population, key=lambda item: item.validation_mse)
    history = [math.sqrt(best_by_validation.validation_mse)]

    for _ in range(config.generations):
        next_population = [best_by_validation.tree]
        while len(next_population) < config.population_size:
            parent_a = _tournament_select(scored_population, config.tournament_size, rng)
            child = parent_a.tree
            if rng.random() < config.crossover_rate:
                parent_b = _tournament_select(scored_population, config.tournament_size, rng)
                child = _subtree_crossover(parent_a.tree, parent_b.tree, rng)
            if rng.random() < config.mutation_rate:
                child = _subtree_mutation(child, rng, config.max_depth, feature_count)
            next_population.append(child)

        scored_population = [
            _score_individual(tree, matrices, parsimony_penalty=config.parsimony_penalty)
            for tree in next_population
        ]
        generation_best = min(scored_population, key=lambda item: item.validation_mse)
        if generation_best.validation_mse < best_by_validation.validation_mse:
            best_by_validation = generation_best
        history.append(math.sqrt(best_by_validation.validation_mse))

    validation_metrics, test_metrics = _final_metrics(
        best_by_validation.tree,
        matrices,
        prepared_data.scalers.target_scaler,
    )
    return GPForecastResult(
        best_expression=_tree_to_string(best_by_validation.tree),
        best_tree_size=_tree_size(best_by_validation.tree),
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        history=tuple(history),
    )


def _flatten_prepared_data(prepared_data: PreparedData) -> DatasetMatrices:
    train_dataset = prepared_data.train_dataset
    validation_dataset = prepared_data.validation_dataset
    test_dataset = prepared_data.test_dataset
    return DatasetMatrices(
        train_x=_flatten_inputs(train_dataset.inputs.numpy()),
        train_y=train_dataset.targets.numpy().reshape(-1, 1),
        validation_x=_flatten_inputs(validation_dataset.inputs.numpy()),
        validation_y=validation_dataset.targets.numpy().reshape(-1, 1),
        test_x=_flatten_inputs(test_dataset.inputs.numpy()),
        test_y=test_dataset.targets.numpy().reshape(-1, 1),
    )


def _flatten_inputs(values: np.ndarray) -> np.ndarray:
    return np.asarray(values.reshape(values.shape[0], -1), dtype=np.float64)


def _score_individual(
    tree: GPNode,
    matrices: DatasetMatrices,
    parsimony_penalty: float,
) -> IndividualScore:
    train_predictions = _evaluate_tree(tree, matrices.train_x).reshape(-1, 1)
    validation_predictions = _evaluate_tree(tree, matrices.validation_x).reshape(-1, 1)
    train_mse = float(np.mean(np.square(train_predictions - matrices.train_y)))
    validation_mse = float(np.mean(np.square(validation_predictions - matrices.validation_y)))
    adjusted_fitness = train_mse + parsimony_penalty * _tree_size(tree)
    return IndividualScore(
        tree=tree,
        train_mse=train_mse,
        validation_mse=validation_mse,
        adjusted_fitness=adjusted_fitness,
    )


def _final_metrics(
    tree: GPNode,
    matrices: DatasetMatrices,
    target_scaler: Any,
) -> tuple[RegressionMetrics, RegressionMetrics]:
    validation_predictions = _evaluate_tree(tree, matrices.validation_x).reshape(-1, 1)
    test_predictions = _evaluate_tree(tree, matrices.test_x).reshape(-1, 1)

    validation_predictions_original = target_scaler.inverse_transform(validation_predictions)
    validation_targets_original = target_scaler.inverse_transform(matrices.validation_y)
    test_predictions_original = target_scaler.inverse_transform(test_predictions)
    test_targets_original = target_scaler.inverse_transform(matrices.test_y)
    return (
        compute_regression_metrics(validation_predictions_original, validation_targets_original),
        compute_regression_metrics(test_predictions_original, test_targets_original),
    )


def _tournament_select(
    scored_population: list[IndividualScore],
    tournament_size: int,
    rng: random.Random,
) -> IndividualScore:
    sampled = rng.sample(scored_population, k=tournament_size)
    return min(sampled, key=lambda item: item.adjusted_fitness)


def _evaluate_tree(tree: GPNode, features: np.ndarray) -> np.ndarray:
    if tree.kind == "feature":
        column = int(tree.value)
        return np.asarray(features[:, column], dtype=np.float64)
    if tree.kind == "constant":
        return np.full(features.shape[0], float(tree.value), dtype=np.float64)

    primitive_name = cast(OperatorName, tree.value)
    primitive = PRIMITIVES_BY_NAME[primitive_name]
    child_values = [_evaluate_tree(child, features) for child in tree.children]
    values = primitive.function(*child_values)
    return np.asarray(
        np.clip(np.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0),
        dtype=np.float64,
    )


def _random_tree(
    rng: random.Random,
    max_depth: int,
    feature_count: int,
    force_function: bool = False,
) -> GPNode:
    if max_depth == 0 or (not force_function and rng.random() < 0.35):
        return _random_terminal(rng, feature_count)

    primitive = rng.choice(PRIMITIVES)
    children = tuple(
        _random_tree(rng, max_depth - 1, feature_count, force_function=False)
        for _ in range(primitive.arity)
    )
    return GPNode(kind="operator", value=primitive.name, children=children)


def _random_terminal(rng: random.Random, feature_count: int) -> GPNode:
    if rng.random() < 0.7:
        return GPNode(kind="feature", value=rng.randrange(feature_count))
    return GPNode(kind="constant", value=rng.uniform(-1.0, 1.0))


def _tree_size(tree: GPNode) -> int:
    return 1 + sum(_tree_size(child) for child in tree.children)


def _all_paths(tree: GPNode, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    paths = [prefix]
    for index, child in enumerate(tree.children):
        paths.extend(_all_paths(child, prefix + (index,)))
    return paths


def _subtree_at(tree: GPNode, path: tuple[int, ...]) -> GPNode:
    node = tree
    for index in path:
        node = node.children[index]
    return node


def _replace_subtree(tree: GPNode, path: tuple[int, ...], new_subtree: GPNode) -> GPNode:
    if not path:
        return new_subtree
    index = path[0]
    replaced_children = list(tree.children)
    replaced_children[index] = _replace_subtree(replaced_children[index], path[1:], new_subtree)
    return GPNode(kind=tree.kind, value=tree.value, children=tuple(replaced_children))


def _subtree_crossover(parent_a: GPNode, parent_b: GPNode, rng: random.Random) -> GPNode:
    path_a = rng.choice(_all_paths(parent_a))
    path_b = rng.choice(_all_paths(parent_b))
    return _replace_subtree(parent_a, path_a, _subtree_at(parent_b, path_b))


def _subtree_mutation(
    tree: GPNode,
    rng: random.Random,
    max_depth: int,
    feature_count: int,
) -> GPNode:
    path = rng.choice(_all_paths(tree))
    new_subtree = _random_tree(rng, max(1, max_depth - 1), feature_count, force_function=False)
    return _replace_subtree(tree, path, new_subtree)


def _tree_to_string(tree: GPNode) -> str:
    if tree.kind == "feature":
        return f"x{int(tree.value)}"
    if tree.kind == "constant":
        return f"{float(tree.value):.3f}"

    primitive_name = cast(OperatorName, tree.value)
    primitive = PRIMITIVES_BY_NAME[primitive_name]
    rendered_children = tuple(_tree_to_string(child) for child in tree.children)
    if primitive.arity == 1:
        return f"{primitive.label}({rendered_children[0]})"
    return f"({rendered_children[0]} {primitive.label} {rendered_children[1]})"
