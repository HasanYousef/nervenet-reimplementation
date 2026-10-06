from collections.abc import Sequence

from torch import Tensor, nn

from nervenet.policies.flat_value_network import FlatValueNetwork
from nervenet.policies.graph_actor import GraphActor


class GraphActorCriticExtractor(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        routes: Sequence[tuple[int, int]],
        actuator_node_indices: Sequence[int],
        hidden_size: int = 64,
        message_passing_steps: int = 2,
    ) -> None:
        super().__init__()

        self.routes = tuple(routes)
        self.actuator_node_indices = tuple(actuator_node_indices)

        self.latent_dim_pi = len(self.actuator_node_indices)
        self.latent_dim_vf = 1

        self.actor = GraphActor(
            input_size=observation_shape[-1],
            hidden_size=hidden_size,
            message_passing_steps=message_passing_steps,
        )
        self.critic = FlatValueNetwork(
            observation_shape=observation_shape,
            hidden_size=hidden_size,
        )

    def forward(
        self,
        observations: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return (
            self.forward_actor(observations),
            self.forward_critic(observations),
        )

    def forward_actor(self, observations: Tensor) -> Tensor:
        return self.actor(
            observations,
            self.routes,
            self.actuator_node_indices,
        )

    def forward_critic(self, observations: Tensor) -> Tensor:
        return self.critic(observations)
