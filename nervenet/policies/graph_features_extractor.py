from math import prod

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import Tensor


class GraphFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
    ) -> None:
        super().__init__(
            observation_space,
            features_dim=prod(observation_space.shape),
        )

    def forward(self, observations: Tensor) -> Tensor:
        return observations
