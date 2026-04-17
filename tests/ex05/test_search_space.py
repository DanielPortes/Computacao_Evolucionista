from __future__ import annotations

import numpy as np

from ce.ex05_hpo.search_space import default_search_space


def test_search_space_decodes_to_coherent_lstm_params() -> None:
    space = default_search_space()
    params = space.decode(np.asarray([0, 2, 0, 3, 1, 2], dtype=float))

    assert params.lookback in {3, 6, 12}
    assert params.hidden_size in {16, 32, 64}
    assert params.second_hidden_size <= params.hidden_size
    assert params.dropout in {0.0, 0.1, 0.2, 0.3}
    assert params.batch_size in {16, 32, 64}

