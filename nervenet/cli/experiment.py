import argparse
from math import degrees
from pathlib import Path
from typing import Any

from nervenet.experiments import (
    ExperimentConfig,
    ExperimentRunner,
    ExperimentStore,
)
from nervenet.experiments.manifest import default_ppo_config
from nervenet.policies.defaults import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MESSAGE_PASSING_STEPS,
)
from nervenet.viewing import view_crawler_policy


def _add_location_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
    )


def _print_evaluation(evaluation: dict[str, Any]) -> None:
    summary = evaluation["summary"]
    print(f"Episodes: {evaluation['episode_count']}")
    print(
        "  Mean forward distance: "
        f"{summary['mean_forward_distance']:.3f} m"
    )
    print(
        "  Mean lateral distance: "
        f"{summary['mean_lateral_distance']:.3f} m"
    )
    print(
        "  Mean absolute lateral distance: "
        f"{summary['mean_absolute_lateral_distance']:.3f} m"
    )
    print(
        "  Mean absolute lateral speed: "
        f"{summary['mean_absolute_lateral_velocity']:.3f} m/s"
    )
    print(
        "  Mean final heading error: "
        f"{degrees(summary['mean_final_heading_error_radians']):.1f} deg"
    )
    print(f"  Mean total reward: {summary['mean_total_reward']:.3f}")
    print(
        "  Mean absolute action: "
        f"{summary['mean_absolute_action']:.3f}"
    )
    print(
        "  Action saturation: "
        f"{summary['action_saturation_fraction']:.1%}"
    )
    print(
        "  Mean absolute action change: "
        f"{summary['mean_absolute_action_change']:.3f}"
    )
    print(
        "  Mean actuated joint speed: "
        f"{summary['mean_actuated_joint_speed']:.3f} rad/s"
    )
    print(f"  Unstable episodes: {summary['unstable_episodes']}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage reproducible PPO training experiments."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start", help="Create and train an experiment.")
    start.add_argument("--name", required=True)
    start.add_argument(
        "--policy-type",
        choices=("matched", "graph"),
        required=True,
    )
    start.add_argument("--modules", type=int, default=3)
    start.add_argument("--timesteps", type=int, required=True)
    start.add_argument("--seed", type=int, default=0)
    start.add_argument("--control-cost-weight", type=float, default=0.05)
    start.add_argument("--checkpoint-every", type=int, default=100_000)
    start.add_argument("--evaluation-episodes", type=int, default=20)
    start.add_argument("--evaluation-seed-start", type=int, default=0)
    start.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Stop a PPO update early when its approximate KL grows too large.",
    )
    start.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    start.add_argument(
        "--message-passing-steps",
        type=int,
        default=DEFAULT_MESSAGE_PASSING_STEPS,
    )
    _add_location_argument(start)

    continue_parser = commands.add_parser(
        "continue",
        help="Continue an experiment from its latest checkpoint.",
    )
    continue_parser.add_argument("--name", required=True)
    continue_parser.add_argument("--timesteps", type=int, required=True)
    _add_location_argument(continue_parser)

    evaluate = commands.add_parser(
        "evaluate",
        help="Evaluate an experiment checkpoint.",
    )
    evaluate.add_argument("--name", required=True)
    evaluate.add_argument("--step", type=int, default=None)
    evaluate.add_argument("--best", action="store_true")
    _add_location_argument(evaluate)

    view = commands.add_parser("view", help="View an experiment checkpoint.")
    view.add_argument("--name", required=True)
    view.add_argument("--step", type=int, default=None)
    view.add_argument("--best", action="store_true")
    view.add_argument("--seed", type=int, default=0)
    _add_location_argument(view)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    runner = ExperimentRunner(args.experiments_dir)

    if args.command == "start":
        ppo = default_ppo_config()
        ppo["target_kl"] = args.target_kl
        config = ExperimentConfig(
            policy_type=args.policy_type,
            module_count=args.modules,
            seed=args.seed,
            control_cost_weight=args.control_cost_weight,
            checkpoint_interval=args.checkpoint_every,
            evaluation_episodes=args.evaluation_episodes,
            evaluation_seed_start=args.evaluation_seed_start,
            graph_hidden_size=args.hidden_size,
            message_passing_steps=args.message_passing_steps,
            ppo=ppo,
        )
        manifest = runner.start(args.name, config, args.timesteps)
        print(
            f"Experiment {args.name} completed at "
            f"{manifest['current_timesteps']} timesteps."
        )
        return

    if args.command == "continue":
        manifest = runner.continue_experiment(args.name, args.timesteps)
        print(
            f"Experiment {args.name} completed at "
            f"{manifest['current_timesteps']} timesteps."
        )
        return

    store = ExperimentStore(args.experiments_dir, args.name)
    config = store.config()

    if args.command == "evaluate":
        _print_evaluation(
            runner.evaluate(args.name, args.step, best=args.best)
        )
        return

    if args.best and args.step is not None:
        parser.error("--best and --step cannot be used together")
    checkpoint_path = (
        store.resolve_best_checkpoint()
        if args.best
        else store.resolve_checkpoint(args.step)
    )
    view_crawler_policy(
        model_path=checkpoint_path,
        policy_type=config.policy_type,
        module_count=config.module_count,
        seed=args.seed,
        control_cost_weight=config.control_cost_weight,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Experiment interrupted. Completed checkpoints were preserved.")
        raise SystemExit(130)
