import argparse

from nervenet.envs import CrawlerEnv


def run_random_episode(
    env: CrawlerEnv,
    seed: int,
) -> tuple[int, float, bool, bool, dict]:
    env.reset(seed=seed)
    env.action_space.seed(seed)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    info = {}

    while not (terminated or truncated):
        action = env.action_space.sample()
        _, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

    return steps, total_reward, terminated, truncated, info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one episode with random crawler actions."
    )
    parser.add_argument("--modules", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = CrawlerEnv(module_count=args.modules)
    steps, total_reward, terminated, truncated, info = run_random_episode(
        env,
        seed=args.seed,
    )

    end_reason = "unstable state" if terminated else "time limit"
    duration = steps * env.control_timestep

    print(f"Seed: {args.seed}")
    print(f"Steps: {steps}")
    print(f"Duration: {duration:.2f} s")
    print(f"Distance: {info['x_position']:.3f} m")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Ended by: {end_reason}")


if __name__ == "__main__":
    main()
