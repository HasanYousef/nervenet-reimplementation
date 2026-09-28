import argparse
from pathlib import Path
import time

import mujoco.viewer
from stable_baselines3 import PPO

from nervenet.envs import CrawlerEnv
from nervenet.models.crawler import TORSO_HEIGHT, TORSO_SPACING


def main() -> None:
    parser = argparse.ArgumentParser(description="View a trained crawler policy.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
    )
    parser.add_argument("--modules", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_path = args.model or Path(
        f"artifacts/flat_policy_{args.modules}_modules.zip"
    )
    env = CrawlerEnv(module_count=args.modules)
    policy = PPO.load(model_path, device="cpu")
    observation, _ = env.reset(seed=args.seed)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [
            -TORSO_SPACING * (args.modules - 1) / 2.0,
            0.0,
            TORSO_HEIGHT,
        ]
        viewer.cam.distance = max(3.8, 1.2 * args.modules)
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
            time.sleep(max(0.0, env.control_timestep - elapsed))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
