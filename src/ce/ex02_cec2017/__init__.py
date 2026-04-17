"""Modulo canonico do exercicio CE2."""

from ce.ex02_cec2017.benchmarks import get_benchmark, supported_dimensions, supported_function_ids
from ce.ex02_cec2017.run import ExperimentConfig, run_suite

__all__ = [
    "ExperimentConfig",
    "get_benchmark",
    "run_suite",
    "supported_dimensions",
    "supported_function_ids",
]

