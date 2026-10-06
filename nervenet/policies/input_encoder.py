import torch
from torch import nn

from nervenet.policies.defaults import DEFAULT_HIDDEN_SIZE


class NodeInputEncoder(nn.Module):
    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(
            in_features=input_size,
            out_features=hidden_size,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(observations))
