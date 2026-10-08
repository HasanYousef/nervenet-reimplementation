from torch import Tensor, nn

from nervenet.policies.defaults import DEFAULT_HIDDEN_SIZE
from nervenet.policies.flat_actor_network import FlatActorNetwork
from nervenet.policies.flat_value_network import FlatValueNetwork


class MatchedActorCriticExtractor(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_size: int,
        actor_hidden_size: int,
        critic_hidden_size: int = DEFAULT_HIDDEN_SIZE,
    ) -> None:
        super().__init__()

        self.latent_dim_pi = action_size
        self.latent_dim_vf = 1

        self.actor = FlatActorNetwork(
            observation_shape=observation_shape,
            action_size=action_size,
            hidden_size=actor_hidden_size,
        )
        self.critic = FlatValueNetwork(
            observation_shape=observation_shape,
            hidden_size=critic_hidden_size,
        )

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self.forward_actor(observations),
            self.forward_critic(observations),
        )

    def forward_actor(self, observations: Tensor) -> Tensor:
        return self.actor(observations)

    def forward_critic(self, observations: Tensor) -> Tensor:
        return self.critic(observations)
