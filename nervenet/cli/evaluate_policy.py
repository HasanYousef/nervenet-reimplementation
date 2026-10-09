import argparse
from math import degrees
from pathlib import Path

from nervenet.evaluation import (
    EpisodeResult,
    run_episode,
    summarize_episode_results,
)
from nervenet.policy_loading import default_policy_path, load_crawler_policy


def print_summary(
    name: str,
    results: list[EpisodeResult],
) -> None:
    summary = summarize_episode_results(results)

    print(name)
    print(f"  Mean forward distance: {summary.mean_forward_distance:.3f} m")
    print(f"  Mean lateral distance: {summary.mean_lateral_distance:.3f} m")
    print(
        "  Mean absolute lateral distance: "
        f"{summary.mean_absolute_lateral_distance:.3f} m"
    )
    print(
        "  Mean absolute lateral speed: "
        f"{summary.mean_absolute_lateral_velocity:.3f} m/s"
    )
    print(
        "  Mean final heading error: "
        f"{degrees(summary.mean_final_heading_error_radians):.1f} deg"
    )
    print(f"  Mean total reward: {summary.mean_total_reward:.3f}")
    print(f"  Mean absolute action: {summary.mean_absolute_action:.3f}")
    print(f"  Action saturation: {summary.action_saturation_fraction:.1%}")
    print(
        "  Mean absolute action change: "
        f"{summary.mean_absolute_action_change:.3f}"
    )
    print(
        "  Mean actuated joint speed: "
        f"{summary.mean_actuated_joint_speed:.3f} rad/s"
    )
    print(f"  Unstable episodes: {summary.unstable_episodes}/{len(results)}")


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
