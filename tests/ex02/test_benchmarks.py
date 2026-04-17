from __future__ import annotations

import numpy as np
import pytest

from ce.ex02_cec2017.benchmarks import get_benchmark, supported_dimensions, supported_function_ids


def test_required_dimensions_are_supported() -> None:
    assert {2, 10}.issubset(set(supported_dimensions()))


def test_supported_benchmarks_return_finite_values() -> None:
    for function_id in supported_function_ids():
        for dimension in (2, 10):
            benchmark = get_benchmark(function_id=function_id, dimension=dimension)
            points = np.zeros((3, dimension), dtype=float)
            values = benchmark.evaluate(points)
            assert values.shape == (3,)
            assert np.all(np.isfinite(values))


def test_invalid_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="Dimensao nao suportada"):
        get_benchmark(function_id=3, dimension=3)

