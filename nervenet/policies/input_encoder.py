import torch
from torch import nn


class NodeInputEncoder(nn.Module):
    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(
            in_features=input_size,
            out_features=hidden_size,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(observations))
