"""Busca evolucionaria para o CE5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

from ce.ex05_hpo.objective import (
    ForecastHPOProblem,
    TemporalCVConfig,
    evaluate_search_params,
)
from ce.ex05_hpo.search_space import (
    ForecastSearchParams,
    ForecastSearchSpace,
    default_search_space,
)


class EvolutionSearchConfig(BaseModel):
    """Configuracao da busca evolucionaria."""

    population_size: int = Field(default=8, ge=2)
    generations: int = Field(default=2, ge=1)
    seed: int = Field(default=42, ge=0)


@dataclass(frozen=True)
class HPOResult:
    """Resultado agregado da busca."""

    best_params: ForecastSearchParams
    best_score: float
    baseline_score: float


def run_search(
    search_config: EvolutionSearchConfig,
    objective_config: TemporalCVConfig,
    search_space: ForecastSearchSpace | None = None,
) -> HPOResult:
    """Executa a busca evolucionaria com custo temporal barato."""

    resolved_space = search_space or default_search_space()
    problem = ForecastHPOProblem(resolved_space, objective_config)
    algorithm = GA(
        pop_size=search_config.population_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )
    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", search_config.generations),
        seed=search_config.seed,
        verbose=False,
    )

    best_vector = np.asarray(result.X, dtype=float).reshape(-1)
    best_params = resolved_space.decode(best_vector)
    best_score = float(np.asarray(result.F, dtype=float).reshape(-1)[0])
    baseline_params = resolved_space.default_params()
    baseline_score = evaluate_search_params(
        baseline_params,
        problem.frame,
        problem.folds,
        objective_config,
    )
    return HPOResult(best_params=best_params, best_score=best_score, baseline_score=baseline_score)
