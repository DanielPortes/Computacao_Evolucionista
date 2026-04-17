"""Resolucao canonica de caminhos do repositorio consolidado."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CEC2017_DIR = RAW_DATA_DIR / "cec2017"
TSP_DIR = RAW_DATA_DIR / "tsp"
WEATHER_DIR = RAW_DATA_DIR / "weather"

TSPLIB_INSTANCES = {
    "berlin52": "berlin52.tsp",
    "br17": "br17.atsp",
    "ch130": "ch130.tsp",
    "ftv70": "ftv70.atsp",
}

WEATHER_DATASETS = {
    "pampulha": "BELO HORIZONTE (PAMPULHA)_MG.csv",
    "cercadinho": "BELO HORIZONTE - CERCADINHO_MG.csv",
    "rola_moca": "IBIRITE (ROLA MOCA)_MG.csv",
}


def project_root() -> Path:
    """Retorna a raiz do repositorio consolidado."""

    return PROJECT_ROOT


def raw_data_dir() -> Path:
    """Retorna o diretorio canonico de dados brutos."""

    return RAW_DATA_DIR


def cec2017_data_file() -> Path:
    """Retorna o arquivo de apoio do benchmark CEC2017."""

    return _must_exist(CEC2017_DIR / "data.pkl")


def tsplib_instance(name: str) -> Path:
    """Retorna o caminho de uma instancia TSPLIB suportada."""

    try:
        filename = TSPLIB_INSTANCES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(TSPLIB_INSTANCES))
        message = f"Instancia TSPLIB desconhecida: {name!r}. Suportadas: {supported}."
        raise ValueError(message) from exc
    return _must_exist(TSP_DIR / filename)


def list_tsplib_instances() -> dict[str, Path]:
    """Lista as instancias TSPLIB disponiveis por alias canonico."""

    return {name: _must_exist(TSP_DIR / filename) for name, filename in TSPLIB_INSTANCES.items()}


def weather_dataset(name: str) -> Path:
    """Retorna o caminho de um dataset meteorologico suportado."""

    try:
        filename = WEATHER_DATASETS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(WEATHER_DATASETS))
        message = f"Dataset meteorologico desconhecido: {name!r}. Suportados: {supported}."
        raise ValueError(message) from exc
    return _must_exist(WEATHER_DIR / filename)


def list_weather_datasets() -> dict[str, Path]:
    """Lista os datasets meteorologicos disponiveis por alias canonico."""

    return {
        name: _must_exist(WEATHER_DIR / filename)
        for name, filename in WEATHER_DATASETS.items()
    }


def _must_exist(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path
