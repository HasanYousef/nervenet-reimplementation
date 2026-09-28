import argparse
from pathlib import Path
from statistics import fmean

from stable_baselines3 import PPO

from nervenet.envs import CrawlerEnv
from nervenet.evaluation import EpisodeResult, run_episode


def print_summary(
    name: str,
    results: list[EpisodeResult],
) -> None:
    mean_distance = fmean(result.distance for result in results)
    mean_reward = fmean(result.total_reward for result in results)
    unstable_episodes = sum(result.terminated for result in results)

    print(name)
    print(f"  Mean distance: {mean_distance:.3f} m")
    print(f"  Mean total reward: {mean_reward:.3f}")
    print(f"  Unstable episodes: {unstable_episodes}/{len(results)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare random and flat PPO crawler policies."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
    )
    parser.add_argument("--modules", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    model_path = args.model or Path(
        f"artifacts/flat_policy_{args.modules}_modules.zip"
    )
    model = PPO.load(model_path, device="cpu")

    random_results = []
    flat_results = []

    for seed in range(args.episodes):
        random_env = CrawlerEnv(module_count=args.modules)
        random_env.action_space.seed(seed)

        random_results.append(
            run_episode(
                env=random_env,
                action_selector=lambda _observation: random_env.action_space.sample(),
                seed=seed,
            )
        )

        flat_env = CrawlerEnv(module_count=args.modules)

        flat_results.append(
            run_episode(
                env=flat_env,
                action_selector=lambda observation: model.predict(
                    observation,
                    deterministic=True,
                )[0],
                seed=seed,
            )
        )

    print_summary("Random policy", random_results)
    print_summary("Flat PPO policy", flat_results)


if __name__ == "__main__":
    main()
