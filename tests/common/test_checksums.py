from __future__ import annotations

from ce.common.io import sha256_file
from ce.common.paths import cec2017_data_file, list_tsplib_instances, list_weather_datasets

EXPECTED_HASHES = {
    "cec2017": "5f978ec6620e659ea933e21c4c4398b47b811467d3eed3dcb204566c3965924b",
    "berlin52": "1e2815c9b1f3a507e06d74e03b3a1bf3f66ae823d94d073aba9be333ffb44993",
    "br17": "08fe522e7198675518c023246acdba6a37bb1f5aa4f5205965dd78e00191319c",
    "ch130": "e816e347c462ee2d9e9e3d1536ea300fea986df293276419ffa7a202106b7502",
    "ftv70": "f078b66a9722e3ff7fe15b91e47a41d0594c63302b23958d1906336c83ddd0c3",
    "pampulha": "9c07741014fca8d56b2106de7ae2b9f3d47b5e45b78ea51b8b60de972c0baf63",
    "cercadinho": "8d894ca4cbb965625c10c9e65225142b814a51e7657527d7af84fb1591227e15",
    "rola_moca": "8a30a6130fa03e2d9096a14fe31efe12b8e0d4a392e78e4413eddd723d10d0e8",
}


def test_canonical_files_preserve_legacy_checksums() -> None:
    assert sha256_file(cec2017_data_file()) == EXPECTED_HASHES["cec2017"]
    for name, path in list_tsplib_instances().items():
        assert sha256_file(path) == EXPECTED_HASHES[name]
    for name, path in list_weather_datasets().items():
        assert sha256_file(path) == EXPECTED_HASHES[name]


def test_weather_catalog_is_deduplicated() -> None:
    hashes = {sha256_file(path) for path in list_weather_datasets().values()}
    assert len(list_weather_datasets()) == 3
    assert len(hashes) == 3
