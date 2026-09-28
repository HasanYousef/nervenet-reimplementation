import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from nervenet.models import build_crawler_model


class CrawlerEnv(gym.Env):
    def __init__(self, module_count: int = 2) -> None:
        super().__init__()

        self.frame_skip = 10
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

    @property
    def control_timestep(self) -> float:
        return self.model.opt.timestep * self.frame_skip

    def _get_observation(self) -> np.ndarray:
        return np.concatenate(
            [
                self.data.qpos[2:],
                self.data.qvel,
            ]
        ).astype(np.float64, copy=True)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float64)

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.data.ctrl[:] = action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        observation = self._get_observation()

        reward = 0.0
        terminated = not np.isfinite(observation).all()
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info
