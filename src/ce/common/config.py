"""Modelos declarativos basicos compartilhados entre os exercicios."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from . import paths


class SeedConfig(BaseModel):
    """Configuracao minima de semente para execucoes reproduziveis."""

    value: int = Field(default=42, ge=0)


class RepositoryConfig(BaseModel):
    """Configura caminhos e defaults compartilhados do repositorio."""

    project_root: Path = Field(default_factory=paths.project_root)
    raw_data_dir: Path = Field(default_factory=paths.raw_data_dir)
    seed: SeedConfig = Field(default_factory=SeedConfig)
