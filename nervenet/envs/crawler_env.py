import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from nervenet.models import build_crawler_model


class CrawlerEnv(gym.Env):
    def __init__(self, module_count: int = 2) -> None:
        super().__init__()

        self.model = build_crawler_model(module_count)
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float64,
        )

        observation_size = self.model.nq - 2 + self.model.nv

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float64,
        )

    def _get_observation(self) -> np.ndarray:
        return np.concatenate(
            [
                self.data.qpos[2:],
                self.data.qvel,
            ]
        ).astype(np.float64, copy=True)
