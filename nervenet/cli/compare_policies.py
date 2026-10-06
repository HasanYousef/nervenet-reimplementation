import argparse
from pathlib import Path
from statistics import fmean

from nervenet.envs import CrawlerEnv
from nervenet.evaluation import EpisodeResult, run_episode
from nervenet.policy_loading import default_policy_path, load_crawler_policy


def print_summary(
    name: str,
    results: list[EpisodeResult],
) -> None:
    mean_distance = fmean(result.distance for result in results)
    mean_lateral_distance = fmean(result.lateral_distance for result in results)
    mean_reward = fmean(result.total_reward for result in results)
    mean_abs_action = fmean(result.mean_abs_action for result in results)
    mean_saturation = fmean(
        result.action_saturation_fraction for result in results
    )
    mean_action_change = fmean(
        result.mean_abs_action_change for result in results
    )
    mean_joint_velocity = fmean(
        result.mean_abs_actuated_joint_velocity for result in results
    )
    unstable_episodes = sum(result.terminated for result in results)

    print(name)
    print(f"  Mean forward distance: {mean_distance:.3f} m")
    print(f"  Mean lateral distance: {mean_lateral_distance:.3f} m")
    print(f"  Mean total reward: {mean_reward:.3f}")
    print(f"  Mean absolute action: {mean_abs_action:.3f}")
    print(f"  Action saturation: {mean_saturation:.1%}")
    print(f"  Mean absolute action change: {mean_action_change:.3f}")
    print(f"  Mean actuated joint speed: {mean_joint_velocity:.3f} rad/s")
    print(f"  Unstable episodes: {unstable_episodes}/{len(results)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare random and trained PPO crawler policies."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
    )
    parser.add_argument("--modules", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--policy-type",
        choices=("flat", "graph"),
        default="flat",
    )
    parser.add_argument(
        "--control-cost-weight",
        type=float,
        default=0.05,
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    model_path = args.model or default_policy_path(
        args.policy_type,
        args.modules,
    )
    model, policy_env = load_crawler_policy(
        model_path=model_path,
        policy_type=args.policy_type,
        module_count=args.modules,
        control_cost_weight=args.control_cost_weight,
    )

    random_results = []
    policy_results = []

    for seed in range(args.episodes):
        random_env = CrawlerEnv(
            module_count=args.modules,
            control_cost_weight=args.control_cost_weight,
        )
        random_env.action_space.seed(seed)

        random_results.append(
            run_episode(
                env=random_env,
                action_selector=lambda _observation: random_env.action_space.sample(),
                seed=seed,
            )
        )

        policy_results.append(
            run_episode(
                env=policy_env,
                action_selector=lambda observation: model.predict(
                    observation,
                    deterministic=True,
                )[0],
                seed=seed,
            )
        )

    print_summary("Random policy", random_results)
    print_summary(f"{args.policy_type.title()} PPO policy", policy_results)


if __name__ == "__main__":
    main()
