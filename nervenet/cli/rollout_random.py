import argparse

from nervenet.envs import CrawlerEnv
from nervenet.evaluation import run_episode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one episode with random crawler actions."
    )
    parser.add_argument("--modules", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = CrawlerEnv(module_count=args.modules)
    env.action_space.seed(args.seed)

    result = run_episode(
        env=env,
        action_selector=lambda _observation: env.action_space.sample(),
        seed=args.seed,
    )

    end_reason = "unstable state" if result.terminated else "time limit"
    duration = result.steps * env.control_timestep

    print(f"Seed: {result.seed}")
    print(f"Steps: {result.steps}")
    print(f"Duration: {duration:.2f} s")
    print(f"Distance: {result.distance:.3f} m")
    print(f"Total reward: {result.total_reward:.3f}")
    print(f"Ended by: {end_reason}")


if __name__ == "__main__":
    main()
