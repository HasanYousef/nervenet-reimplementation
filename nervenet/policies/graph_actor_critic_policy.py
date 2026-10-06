from collections.abc import Sequence
from typing import Any

from gymnasium import spaces
import torch
from torch import nn
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

from nervenet.policies.graph_actor_critic_extractor import (
    GraphActorCriticExtractor,
)
from nervenet.policies.graph_features_extractor import (
    GraphFeaturesExtractor,
)


class GraphActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        routes: Sequence[tuple[int, int]],
        actuator_node_indices: Sequence[int],
        hidden_size: int = 64,
        message_passing_steps: int = 2,
        **kwargs: Any,
    ) -> None:
        self.routes = tuple(routes)
        self.actuator_node_indices = tuple(actuator_node_indices)
        self.hidden_size = hidden_size
        self.message_passing_steps = message_passing_steps

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=GraphFeaturesExtractor,
            net_arch=[],
            ortho_init=False,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GraphActorCriticExtractor(
            observation_shape=self.observation_space.shape,
            routes=self.routes,
            actuator_node_indices=self.actuator_node_indices,
            hidden_size=self.hidden_size,
            message_passing_steps=self.message_passing_steps,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise TypeError("Graph policy requires a continuous Box action space")

        if self.mlp_extractor.latent_dim_pi != self.action_dist.action_dim:
            raise ValueError(
                "Actuator-node count must match the action-space size"
            )

        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        self.log_std = nn.Parameter(
            torch.full(
                (self.action_dist.action_dim,),
                self.log_std_init,
            )
        )

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
