import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from nervenet.models import build_crawler_model


class CrawlerEnv(gym.Env):

    def __init__(
        self,
        module_count: int = 2,
        max_episode_seconds: float = 10.0,
        reset_position_noise: float = 0.01,
        reset_velocity_noise: float = 0.01,
    ) -> None:
        super().__init__()

        if max_episode_seconds <= 0:
            raise ValueError("max_episode_seconds must be positive")

        if reset_position_noise < 0:
            raise ValueError("reset_position_noise must be non-negative")

        if reset_velocity_noise < 0:
            raise ValueError("reset_velocity_noise must be non-negative")

        self.frame_skip = 10
        self.episode_steps = 0
        self.model = build_crawler_model(module_count)
        self.data = mujoco.MjData(self.model)

        self.reset_position_noise = reset_position_noise
        self.reset_velocity_noise = reset_velocity_noise

        self.max_episode_steps = round(max_episode_seconds / self.control_timestep)

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
        self.data.qpos[7:] += self.np_random.uniform(
            low=-self.reset_position_noise,
            high=self.reset_position_noise,
            size=self.model.nq - 7,
        )
        self.data.qvel[6:] += self.np_random.uniform(
            low=-self.reset_velocity_noise,
            high=self.reset_velocity_noise,
            size=self.model.nv - 6,
        )
        self.episode_steps = 0
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        x_before = float(self.data.qpos[0])

        action = np.asarray(action, dtype=np.float64)

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.data.ctrl[:] = action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.episode_steps += 1
        x_after = float(self.data.qpos[0])
        forward_velocity = (x_after - x_before) / self.control_timestep

        observation = self._get_observation()

        reward = forward_velocity
        terminated = not np.isfinite(observation).all()
        truncated = self.episode_steps >= self.max_episode_steps
        info = {
            "x_position": x_after,
            "x_velocity": forward_velocity,
        }

        return observation, reward, terminated, truncated, info
