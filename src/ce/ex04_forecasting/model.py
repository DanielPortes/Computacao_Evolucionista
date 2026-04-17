"""Modelo LSTM baseline para CE4."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn


class ForecastLSTM(nn.Module):
    """LSTM empilhada pequena para previsao do `GLOBAL`."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        second_hidden_size: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=second_hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(second_hidden_size, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded, _ = self.encoder(inputs)
        decoded, _ = self.decoder(encoded)
        last_hidden = decoded[:, -1, :]
        prediction = self.head(self.dropout(last_hidden))
        return cast(Tensor, prediction)
