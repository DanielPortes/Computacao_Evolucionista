from __future__ import annotations

from ce.ex05_hpo.objective import build_time_series_folds


def test_time_series_folds_are_strictly_temporal() -> None:
    folds = build_time_series_folds(length=30, n_splits=3)

    assert len(folds) == 3
    assert all(fold.train_end < fold.validation_end for fold in folds)
    assert folds[0].train_end < folds[1].train_end < folds[2].train_end

