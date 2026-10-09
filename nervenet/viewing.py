from pathlib import Path
import time

import mujoco
import mujoco.viewer

from nervenet.models.crawler import TORSO_HEIGHT, TORSO_SPACING
from nervenet.policy_loading import PolicyType, load_crawler_policy


def view_crawler_policy(
    model_path: Path,
    policy_type: PolicyType,
    module_count: int,
    seed: int = 0,
    control_cost_weight: float = 0.05,
) -> None:
    policy, env = load_crawler_policy(
        model_path=model_path,
        policy_type=policy_type,
        module_count=module_count,
        control_cost_weight=control_cost_weight,
    )
    observation, _ = env.reset(seed=seed)
    crawler_env = env.unwrapped

    with mujoco.viewer.launch_passive(
        crawler_env.model,
        crawler_env.data,
    ) as viewer:
        tracked_body = mujoco.mj_name2id(
            crawler_env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "module_1_torso",
        )
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = tracked_body
        viewer.cam.lookat[:] = [
            -TORSO_SPACING * (module_count - 1) / 2.0,
            0.0,
            TORSO_HEIGHT,
        ]
        viewer.cam.distance = max(5.0, 1.5 * module_count)
        viewer.cam.azimuth = 100.0
        viewer.cam.elevation = -25.0

        while viewer.is_running():
            step_started_at = time.monotonic()
            action, _ = policy.predict(
                observation,
                deterministic=True,
            )
            observation, _, terminated, truncated, _ = env.step(action)
            viewer.sync()

            if terminated or truncated:
                observation, _ = env.reset(seed=seed)

            elapsed = time.monotonic() - step_started_at
            time.sleep(max(0.0, crawler_env.control_timestep - elapsed))
