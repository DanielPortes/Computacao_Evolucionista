from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go

from ce.analysis.run_all import NotebookRunConfig, build_plotly_figures, run_all_exercises


def test_run_all_exercises_returns_frames_and_figures() -> None:
    results = run_all_exercises(NotebookRunConfig())
    figures = build_plotly_figures(results)

    assert set(results.as_dict()) == {"ce2", "ce3", "ce4", "ce5"}
    assert set(figures) == {"ce2", "ce3", "ce4", "ce5"}
    assert set(results.ex02["algorithm"]) == {"GA", "ES", "EP", "DE", "PSO", "ABC"}
    assert set(results.ex03["algorithm"]) == {"GA", "ACO"}
    assert set(results.ex04["model"]) == {"LSTM", "GP"}
    assert float(results.ex05.iloc[0]["baseline_score"]) > 0.0
    assert all(isinstance(figure, go.Figure) for figure in figures.values())


def test_main_execution_notebook_references_canonical_run_layer() -> None:
    notebook_path = Path("notebooks/main_execution.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    joined_sources = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "run_all_exercises" in joined_sources
    assert "build_plotly_figures" in joined_sources
