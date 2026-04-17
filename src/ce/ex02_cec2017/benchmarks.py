"""Adaptador canonico para o subconjunto CEC2017 usado em CE2."""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from numpy.exceptions import VisibleDeprecationWarning

from ce.common.paths import cec2017_data_file

FloatArray = npt.NDArray[np.float64]
SUPPORTED_FUNCTION_IDS = (3, 4, 5, 6, 7, 8, 9)
SUPPORTED_DIMENSIONS = (2, 10, 20, 30, 50, 100)


@dataclass(frozen=True)
class TransformCatalog:
    """Catalogo de shifts e rotacoes carregado do `data.pkl` legado."""

    rotations: dict[int, FloatArray]
    shifts: FloatArray


@dataclass(frozen=True)
class BenchmarkSpec:
    """Metadados minimos para uma funcao de benchmark canonica."""

    function_id: int
    name: str
    offset: float
    transform_index: int


@dataclass(frozen=True)
class CEC2017Benchmark:
    """Representa uma funcao do benchmark pronta para avaliacao."""

    spec: BenchmarkSpec
    dimension: int

    @property
    def function_id(self) -> int:
        return self.spec.function_id

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def optimum(self) -> float:
        return self.spec.offset

    def evaluate(self, points: npt.ArrayLike) -> FloatArray:
        """Avalia um lote de pontos sem depender de estado global."""

        x = _as_2d_array(points, expected_dimension=self.dimension)
        rotation, shift = _transform_for(self.spec.transform_index, self.dimension)

        if self.function_id == 3:
            transformed = _shift_rotate(x, shift, rotation)
            return np.asarray(_zakharov(transformed) + self.optimum, dtype=float)
        if self.function_id == 4:
            transformed = _shift_rotate(x, shift, rotation)
            return np.asarray(_rosenbrock(transformed) + self.optimum, dtype=float)
        if self.function_id == 5:
            transformed = _shift_rotate(x, shift, rotation)
            return np.asarray(_rastrigin(transformed) + self.optimum, dtype=float)
        if self.function_id == 6:
            transformed = _shift_rotate(x, shift, rotation)
            return np.asarray(_schaffers_f7(transformed) + self.optimum, dtype=float)
        if self.function_id == 7:
            return np.asarray(_lunacek_bi_rastrigin(x, shift, rotation) + self.optimum, dtype=float)
        if self.function_id == 8:
            return np.asarray(_non_cont_rastrigin(x, shift, rotation) + self.optimum, dtype=float)
        if self.function_id == 9:
            transformed = _shift_rotate(x, shift, rotation)
            return np.asarray(_levy(transformed) + self.optimum, dtype=float)
        raise ValueError(f"Funcao nao suportada: f{self.function_id}.")


FUNCTION_SPECS = {
    3: BenchmarkSpec(
        function_id=3,
        name="Shifted Rotated Zakharov",
        offset=300.0,
        transform_index=2,
    ),
    4: BenchmarkSpec(
        function_id=4,
        name="Shifted Rotated Rosenbrock",
        offset=400.0,
        transform_index=3,
    ),
    5: BenchmarkSpec(
        function_id=5,
        name="Shifted Rotated Rastrigin",
        offset=500.0,
        transform_index=4,
    ),
    6: BenchmarkSpec(
        function_id=6,
        name="Shifted Rotated Schaffers F7",
        offset=600.0,
        transform_index=5,
    ),
    7: BenchmarkSpec(
        function_id=7,
        name="Shifted Rotated Lunacek Bi-Rastrigin",
        offset=700.0,
        transform_index=6,
    ),
    8: BenchmarkSpec(
        function_id=8,
        name="Shifted Rotated Non-Continuous Rastrigin",
        offset=800.0,
        transform_index=7,
    ),
    9: BenchmarkSpec(function_id=9, name="Shifted Rotated Levy", offset=900.0, transform_index=8),
}


def supported_function_ids() -> tuple[int, ...]:
    """Retorna o subconjunto de funcoes usado pelo exercicio."""

    return SUPPORTED_FUNCTION_IDS


def supported_dimensions() -> tuple[int, ...]:
    """Retorna as dimensoes suportadas pelo dataset oficial carregado."""

    return SUPPORTED_DIMENSIONS


def get_benchmark(function_id: int, dimension: int) -> CEC2017Benchmark:
    """Constroi uma funcao de benchmark valida para o exercicio."""

    if function_id not in FUNCTION_SPECS:
        supported = ", ".join(f"f{value}" for value in SUPPORTED_FUNCTION_IDS)
        raise ValueError(f"Funcao nao suportada: f{function_id}. Suportadas: {supported}.")
    if dimension not in SUPPORTED_DIMENSIONS:
        supported = ", ".join(str(value) for value in SUPPORTED_DIMENSIONS)
        raise ValueError(f"Dimensao nao suportada: {dimension}. Suportadas: {supported}.")
    return CEC2017Benchmark(spec=FUNCTION_SPECS[function_id], dimension=dimension)


@lru_cache(maxsize=1)
def _load_transform_catalog() -> TransformCatalog:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
            category=VisibleDeprecationWarning,
        )
        with cec2017_data_file().open("rb") as handle:
            raw_catalog = pickle.load(handle)

    rotations = {
        2: np.asarray(raw_catalog["M_D2"], dtype=float),
        10: np.asarray(raw_catalog["M_D10"], dtype=float),
        20: np.asarray(raw_catalog["M_D20"], dtype=float),
        30: np.asarray(raw_catalog["M_D30"], dtype=float),
        50: np.asarray(raw_catalog["M_D50"], dtype=float),
        100: np.asarray(raw_catalog["M_D100"], dtype=float),
    }
    shifts = np.asarray(raw_catalog["shift"], dtype=float)
    return TransformCatalog(rotations=rotations, shifts=shifts)


def _transform_for(function_index: int, dimension: int) -> tuple[FloatArray, FloatArray]:
    catalog = _load_transform_catalog()
    return catalog.rotations[dimension][function_index], catalog.shifts[function_index][:dimension]


def _as_2d_array(points: npt.ArrayLike, expected_dimension: int) -> FloatArray:
    array = np.asarray(points, dtype=float)
    if array.ndim == 1:
        array = np.expand_dims(array, axis=0)
    if array.ndim != 2 or array.shape[1] != expected_dimension:
        message = (
            f"Esperado array bidimensional com {expected_dimension} colunas; "
            f"recebido shape={array.shape}."
        )
        raise ValueError(
            message
        )
    return np.asarray(array, dtype=float)


def _shift_rotate(x: FloatArray, shift: FloatArray, rotation: FloatArray) -> FloatArray:
    shifted = np.expand_dims(x - np.expand_dims(shift, axis=0), axis=-1)
    transformed = np.matmul(np.expand_dims(rotation, axis=0), shifted)
    return np.asarray(transformed[:, :, 0], dtype=float)


def _zakharov(x: FloatArray) -> FloatArray:
    indices = np.expand_dims(np.arange(x.shape[1], dtype=float) + 1.0, axis=0)
    linear = np.sum(indices * x, axis=1)
    squared = np.sum(x * x, axis=1)
    half_linear_sq = np.square(0.5 * linear)
    return np.asarray(squared + half_linear_sq + np.square(half_linear_sq), dtype=float)


def _rosenbrock(x: FloatArray) -> FloatArray:
    shifted = 0.02048 * x + 1.0
    term_1 = 100.0 * np.square(np.square(shifted[:, :-1]) - shifted[:, 1:])
    term_2 = np.square(shifted[:, :-1] - 1.0)
    return np.asarray(np.sum(term_1 + term_2, axis=1), dtype=float)


def _rastrigin(x: FloatArray) -> FloatArray:
    scaled = 0.0512 * x
    values = np.square(scaled) - 10.0 * np.cos(2.0 * np.pi * scaled) + 10.0
    return np.asarray(np.sum(values, axis=1), dtype=float)


def _schaffers_f7(x: FloatArray) -> FloatArray:
    nx = x.shape[1]
    radial = np.sqrt(np.square(x[:, :-1]) + np.square(x[:, 1:]))
    oscillation = np.sin(50.0 * np.power(radial, 0.2))
    accumulated = np.sqrt(radial) * (np.square(oscillation) + 1.0)
    summed = np.sum(accumulated, axis=1)
    denominator = nx * nx - 2 * nx + 1
    return np.asarray(np.square(summed) / denominator, dtype=float)


def _lunacek_bi_rastrigin(x: FloatArray, shift: FloatArray, rotation: FloatArray) -> FloatArray:
    nx = x.shape[1]
    shift_row = np.expand_dims(shift, axis=0)
    mu_0 = 2.5
    s_value = 1 - 1 / (2 * np.sqrt(nx + 20) - 8.2)
    mu_1 = -np.sqrt((mu_0 * mu_0 - 1) / s_value)

    y = 0.1 * (x - shift_row)
    tmp_x = 2.0 * y
    tmp_x[:, shift_row[0] < 0] *= -1

    z = tmp_x.copy()
    shifted_tmp = tmp_x + mu_0
    term_1 = np.sum(np.square(shifted_tmp - mu_0), axis=1)
    term_2 = s_value * np.square(shifted_tmp - mu_1)
    term_2 = np.sum(term_2, axis=1) + nx

    rotated = np.matmul(np.expand_dims(rotation, axis=0), np.expand_dims(z, axis=-1))[:, :, 0]
    oscillation = np.sum(np.cos(2.0 * np.pi * rotated), axis=1)

    minima = term_1.copy()
    mask = term_1 >= term_2
    minima[mask] = term_2[mask]
    return np.asarray(minima + 10.0 * (nx - oscillation), dtype=float)


def _non_cont_rastrigin(x: FloatArray, shift: FloatArray, rotation: FloatArray) -> FloatArray:
    shift_row = np.expand_dims(shift, axis=0)
    shifted = x - shift_row

    # A versao legada tentava discretizar `x`, mas a saida final dependia apenas
    # de `shifted`. Mantemos a mesma semantica numerica do legado.
    z = 0.0512 * shifted
    rotated = np.matmul(np.expand_dims(rotation, axis=0), np.expand_dims(z, axis=-1))[:, :, 0]
    values = np.square(rotated) - 10.0 * np.cos(2.0 * np.pi * rotated) + 10.0
    return np.asarray(np.sum(values, axis=1), dtype=float)


def _levy(x: FloatArray) -> FloatArray:
    w = 1.0 + 0.25 * (x - 1.0)
    term_1 = np.square(np.sin(np.pi * w[:, 0]))
    term_3 = np.square(w[:, -1] - 1.0) * (1.0 + np.square(np.sin(2.0 * np.pi * w[:, -1])))
    interior = np.square(w[:, :-1] - 1.0) * (
        1.0 + 10.0 * np.square(np.sin(np.pi * w[:, :-1] + 1.0))
    )
    return np.asarray(term_1 + np.sum(interior, axis=1) + term_3, dtype=float)
