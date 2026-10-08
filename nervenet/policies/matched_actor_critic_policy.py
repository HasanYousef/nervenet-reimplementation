from typing import Any

from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from nervenet.policies.defaults import DEFAULT_HIDDEN_SIZE
from nervenet.policies.direct_actor_critic_policy import (
    DirectActorCriticPolicy,
)
from nervenet.policies.graph_features_extractor import GraphFeaturesExtractor
from nervenet.policies.matched_actor_critic_extractor import (
    MatchedActorCriticExtractor,
)


class MatchedActorCriticPolicy(DirectActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        actor_hidden_size: int,
        critic_hidden_size: int = DEFAULT_HIDDEN_SIZE,
        **kwargs: Any,
    ) -> None:
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size

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
        self.mlp_extractor = MatchedActorCriticExtractor(
            observation_shape=self.observation_space.shape,
            action_size=self.action_dist.action_dim,
            actor_hidden_size=self.actor_hidden_size,
            critic_hidden_size=self.critic_hidden_size,
        )
