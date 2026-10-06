import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer

from nervenet.models.crawler import TORSO_HEIGHT, TORSO_SPACING
from nervenet.policy_loading import default_policy_path, load_crawler_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="View a trained crawler policy.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
    )
    parser.add_argument("--modules", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy-type",
        choices=("flat", "graph"),
        default="flat",
    )
    args = parser.parse_args()

    model_path = args.model or default_policy_path(
        args.policy_type,
        args.modules,
    )
    policy, env = load_crawler_policy(
        model_path=model_path,
        policy_type=args.policy_type,
        module_count=args.modules,
    )
    observation, _ = env.reset(seed=args.seed)
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
            -TORSO_SPACING * (args.modules - 1) / 2.0,
            0.0,
            TORSO_HEIGHT,
        ]
        viewer.cam.distance = max(5.0, 1.5 * args.modules)
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
                observation, _ = env.reset(seed=args.seed)

            elapsed = time.monotonic() - step_started_at
            time.sleep(max(0.0, crawler_env.control_timestep - elapsed))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
