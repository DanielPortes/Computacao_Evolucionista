"""Problemas mono-objetivo para o exercicio CE2."""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from ce.ex02_cec2017.benchmarks import CEC2017Benchmark, get_benchmark


class CEC2017Problem(ElementwiseProblem):  # type: ignore[misc]
    """Problema mono-objetivo encapsulando uma funcao CEC2017."""

    def __init__(
        self,
        benchmark: CEC2017Benchmark,
        lower_bound: float = -100.0,
        upper_bound: float = 100.0,
    ):
        if lower_bound >= upper_bound:
            raise ValueError("lower_bound deve ser estritamente menor que upper_bound.")

        super().__init__(
            n_var=benchmark.dimension,
            n_obj=1,
            xl=np.full(benchmark.dimension, lower_bound, dtype=float),
            xu=np.full(benchmark.dimension, upper_bound, dtype=float),
        )
        self.benchmark = benchmark

    def _evaluate(
        self,
        x: np.ndarray,
        out: dict[str, float],
        *args: object,
        **kwargs: object,
    ) -> None:
        out["F"] = float(self.benchmark.evaluate(np.asarray(x, dtype=float))[0])


def build_problem(
    function_id: int,
    dimension: int,
    lower_bound: float = -100.0,
    upper_bound: float = 100.0,
) -> CEC2017Problem:
    """Constroi um problema pronto para a otimizacao canonica."""

    benchmark = get_benchmark(function_id=function_id, dimension=dimension)
    return CEC2017Problem(benchmark=benchmark, lower_bound=lower_bound, upper_bound=upper_bound)
