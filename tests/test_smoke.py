from __future__ import annotations

from typer.testing import CliRunner

from ce import __version__
from ce.cli import app


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_help_runs() -> None:
    result = CliRunner().invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Repositorio consolidado" in result.stdout
