import gymnasium as gym
from gymnasium import spaces
import numpy as np

from nervenet.envs.crawler_env import CrawlerEnv
from nervenet.graphs import (
    build_body_graph,
    get_graph_observations,
    pad_graph_observations,
)


class GraphObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: CrawlerEnv) -> None:
        super().__init__(env)

        self.crawler_env = env
        self.graph = build_body_graph(env.model)

        initial_observation = self._get_graph_observation()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=initial_observation.shape,
            dtype=np.float64,
        )

    def observation(
        self,
        _observation: np.ndarray,
    ) -> np.ndarray:
        return self._get_graph_observation()

    def _get_graph_observation(self) -> np.ndarray:
        return pad_graph_observations(
            get_graph_observations(
                self.crawler_env.model,
                self.crawler_env.data,
                self.graph,
            )
        )
