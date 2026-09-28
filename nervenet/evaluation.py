from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nervenet.envs import CrawlerEnv


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    steps: int
    total_reward: float
    distance: float
    terminated: bool
    truncated: bool


ActionSelector = Callable[[np.ndarray], np.ndarray]


def run_episode(
    env: CrawlerEnv,
    action_selector: ActionSelector,
    seed: int,
) -> EpisodeResult:
    observation, _ = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    info = {"x_position": float(env.data.qpos[0])}

    while not (terminated or truncated):
        action = action_selector(observation)

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

    return EpisodeResult(
        seed=seed,
        steps=steps,
        total_reward=total_reward,
        distance=info["x_position"],
        terminated=terminated,
        truncated=truncated,
    )
