from collections.abc import Callable
from dataclasses import dataclass
from math import atan2, cos, sin
from statistics import fmean

import numpy as np

from nervenet.envs import CrawlerEnv, GraphObservationWrapper


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    steps: int
    total_reward: float
    distance: float
    lateral_distance: float
    mean_abs_lateral_velocity: float
    final_heading_error_radians: float
    mean_abs_action: float
    action_saturation_fraction: float
    mean_abs_action_change: float
    mean_abs_actuated_joint_velocity: float
    terminated: bool
    truncated: bool


ActionSelector = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class EvaluationSummary:
    mean_forward_distance: float
    mean_lateral_distance: float
    mean_absolute_lateral_distance: float
    mean_absolute_lateral_velocity: float
    mean_final_heading_error_radians: float
    mean_total_reward: float
    mean_absolute_action: float
    action_saturation_fraction: float
    mean_absolute_action_change: float
    mean_actuated_joint_speed: float
    unstable_episodes: int


def _yaw_from_quaternion(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _wrapped_angle_difference(angle: float, reference: float) -> float:
    difference = angle - reference
    return atan2(sin(difference), cos(difference))


def summarize_episode_results(
    results: list[EpisodeResult],
) -> EvaluationSummary:
    if not results:
        raise ValueError("results must not be empty")

    return EvaluationSummary(
        mean_forward_distance=fmean(result.distance for result in results),
        mean_lateral_distance=fmean(
            result.lateral_distance for result in results
        ),
        mean_absolute_lateral_distance=fmean(
            abs(result.lateral_distance) for result in results
        ),
        mean_absolute_lateral_velocity=fmean(
            result.mean_abs_lateral_velocity for result in results
        ),
        mean_final_heading_error_radians=fmean(
            result.final_heading_error_radians for result in results
        ),
        mean_total_reward=fmean(result.total_reward for result in results),
        mean_absolute_action=fmean(
            result.mean_abs_action for result in results
        ),
        action_saturation_fraction=fmean(
            result.action_saturation_fraction for result in results
        ),
        mean_absolute_action_change=fmean(
            result.mean_abs_action_change for result in results
        ),
        mean_actuated_joint_speed=fmean(
            result.mean_abs_actuated_joint_velocity for result in results
        ),
        unstable_episodes=sum(result.terminated for result in results),
    )


def run_episode(
    env: CrawlerEnv | GraphObservationWrapper,
    action_selector: ActionSelector,
    seed: int,
) -> EpisodeResult:
    observation, _ = env.reset(seed=seed)
    crawler_env = env.unwrapped
    x_start = float(crawler_env.data.qpos[0])
    y_start = float(crawler_env.data.qpos[1])
    initial_heading = _yaw_from_quaternion(crawler_env.data.qpos[3:7])

    actuated_dof_addresses = crawler_env.model.jnt_dofadr[
        crawler_env.model.actuator_trnid[:, 0]
    ]

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    action_magnitude_sum = 0.0
    action_value_count = 0
    saturated_action_count = 0
    action_change_sum = 0.0
    action_change_count = 0
    joint_velocity_sum = 0.0
    joint_velocity_count = 0
    absolute_lateral_velocity_sum = 0.0
    previous_action = None

    while not (terminated or truncated):
        y_before = float(crawler_env.data.qpos[1])
        action = np.asarray(action_selector(observation), dtype=np.float64)

        absolute_action = np.abs(action)
        action_magnitude_sum += float(absolute_action.sum())
        action_value_count += action.size
        saturated_action_count += int((absolute_action >= 0.95).sum())

        if previous_action is not None:
            absolute_change = np.abs(action - previous_action)
            action_change_sum += float(absolute_change.sum())
            action_change_count += action.size

        observation, reward, terminated, truncated, _ = env.step(action)
        y_after = float(crawler_env.data.qpos[1])
        absolute_lateral_velocity_sum += abs(
            (y_after - y_before) / crawler_env.control_timestep
        )

        joint_velocities = np.abs(
            crawler_env.data.qvel[actuated_dof_addresses]
        )
        joint_velocity_sum += float(joint_velocities.sum())
        joint_velocity_count += joint_velocities.size

        total_reward += reward
        steps += 1
        previous_action = action.copy()

    return EpisodeResult(
        seed=seed,
        steps=steps,
        total_reward=total_reward,
        distance=float(crawler_env.data.qpos[0]) - x_start,
        lateral_distance=float(crawler_env.data.qpos[1]) - y_start,
        mean_abs_lateral_velocity=(absolute_lateral_velocity_sum / steps),
        final_heading_error_radians=abs(
            _wrapped_angle_difference(
                _yaw_from_quaternion(crawler_env.data.qpos[3:7]),
                initial_heading,
            )
        ),
        mean_abs_action=action_magnitude_sum / action_value_count,
        action_saturation_fraction=saturated_action_count / action_value_count,
        mean_abs_action_change=(
            action_change_sum / action_change_count
            if action_change_count
            else 0.0
        ),
        mean_abs_actuated_joint_velocity=(
            joint_velocity_sum / joint_velocity_count
        ),
        terminated=terminated,
        truncated=truncated,
    )
