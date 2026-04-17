"""Modulo canonico do exercicio CE3."""

from ce.ex03_tsp.data import KNOWN_OPTIMA, available_instances, load_instance
from ce.ex03_tsp.ga import GeneticAlgorithmConfig, solve_problem
from ce.ex03_tsp.problem import TSPLIBRouteProblem, build_problem

__all__ = [
    "GeneticAlgorithmConfig",
    "KNOWN_OPTIMA",
    "TSPLIBRouteProblem",
    "available_instances",
    "build_problem",
    "load_instance",
    "solve_problem",
]

