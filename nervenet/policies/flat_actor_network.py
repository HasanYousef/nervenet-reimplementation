from math import prod

from torch import Tensor, nn


class FlatActorNetwork(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()

        input_size = prod(observation_shape)

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, observations: Tensor) -> Tensor:
        flat_observations = observations.flatten(start_dim=-2)

        return self.network(flat_observations)
