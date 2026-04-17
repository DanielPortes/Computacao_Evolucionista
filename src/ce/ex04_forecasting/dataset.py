"""Datasets PyTorch para CE4."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset simples de janelas temporais para previsao univariada do alvo."""

    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.as_tensor(inputs, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.inputs[index], self.targets[index]

