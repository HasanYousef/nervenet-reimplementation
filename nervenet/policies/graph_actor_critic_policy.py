from collections.abc import Sequence
from typing import Any

from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from nervenet.graphs import BodyNodeType
from nervenet.policies.defaults import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MESSAGE_PASSING_STEPS,
)
from nervenet.policies.direct_actor_critic_policy import (
    DirectActorCriticPolicy,
)
from nervenet.policies.graph_actor_critic_extractor import (
    GraphActorCriticExtractor,
)
from nervenet.policies.graph_features_extractor import (
    GraphFeaturesExtractor,
)


class GraphActorCriticPolicy(DirectActorCriticPolicy):
    action_dimension_error = "Actuator-node count must match the action-space size"

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        routes: Sequence[tuple[int, int]],
        actuator_node_indices: Sequence[int],
        node_types: Sequence[BodyNodeType],
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        message_passing_steps: int = DEFAULT_MESSAGE_PASSING_STEPS,
        **kwargs: Any,
    ) -> None:
        self.routes = tuple(routes)
        self.actuator_node_indices = tuple(actuator_node_indices)
        self.node_types = tuple(node_types)
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
            node_types=self.node_types,
            hidden_size=self.hidden_size,
            message_passing_steps=self.message_passing_steps,
        )
