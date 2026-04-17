from __future__ import annotations

from ce.common import paths


def test_project_root_contains_data_directory() -> None:
    assert paths.project_root().name == "Computacao_Evolucionista"
    assert paths.raw_data_dir().is_dir()


def test_cec2017_data_file_is_available() -> None:
    assert paths.cec2017_data_file().name == "data.pkl"


def test_tsplib_instances_resolve_from_canonical_catalog() -> None:
    instances = paths.list_tsplib_instances()
    assert set(instances) == {"berlin52", "br17", "ch130", "ftv70"}
    assert instances["berlin52"].suffix == ".tsp"
    assert instances["br17"].suffix == ".atsp"


def test_weather_datasets_resolve_from_single_catalog() -> None:
    datasets = paths.list_weather_datasets()
    assert set(datasets) == {"pampulha", "cercadinho", "rola_moca"}
    assert all(path.suffix == ".csv" for path in datasets.values())

