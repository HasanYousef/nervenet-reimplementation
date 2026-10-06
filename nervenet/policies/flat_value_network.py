from math import prod

from torch import Tensor, nn

from nervenet.policies.defaults import DEFAULT_HIDDEN_SIZE


class FlatValueNetwork(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
    ) -> None:
        super().__init__()

        input_size = prod(observation_shape)

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, observations: Tensor) -> Tensor:
        flat_observations = observations.flatten(start_dim=-2)

        return self.network(flat_observations)
