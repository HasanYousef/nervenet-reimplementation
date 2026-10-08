import argparse
from pathlib import Path
from statistics import fmean

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
        description="Evaluate one trained PPO crawler policy."
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
        choices=("flat", "matched", "graph"),
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
    model, env = load_crawler_policy(
        model_path=model_path,
        policy_type=args.policy_type,
        module_count=args.modules,
        control_cost_weight=args.control_cost_weight,
    )

    results = [
        run_episode(
            env=env,
            action_selector=lambda observation: model.predict(
                observation,
                deterministic=True,
            )[0],
            seed=seed,
        )
        for seed in range(args.episodes)
    ]

    policy_names = {
        "flat": "Flat PPO policy",
        "matched": "Matched MLP PPO policy",
        "graph": "Graph PPO policy",
    }
    print_summary(policy_names[args.policy_type], results)


if __name__ == "__main__":
    main()
