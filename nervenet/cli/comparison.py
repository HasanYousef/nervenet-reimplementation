import argparse
from pathlib import Path

from nervenet.comparisons import ComparisonConfig, ComparisonRunner


def _add_location_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and report controlled multi-seed policy comparisons."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create")
    create.add_argument("--name", required=True)
    create.add_argument("--timesteps", type=int, default=1_000_000)
    create.add_argument("--modules", type=int, default=3)
    create.add_argument("--checkpoint-every", type=int, default=50_000)
    create.add_argument("--validation-episodes", type=int, default=10)
    create.add_argument("--test-episodes", type=int, default=50)
    create.add_argument("--control-cost-weight", type=float, default=0.05)
    _add_location_argument(create)

    run = commands.add_parser("run")
    run.add_argument("--name", required=True)
    run.add_argument("--policy-type", choices=("matched", "graph"))
    run.add_argument("--seed", type=int, choices=range(5))
    _add_location_argument(run)

    extend = commands.add_parser("extend")
    extend.add_argument("--name", required=True)
    extend.add_argument("--timesteps", type=int, required=True)
    _add_location_argument(extend)

    status = commands.add_parser("status")
    status.add_argument("--name", required=True)
    _add_location_argument(status)

    report = commands.add_parser("report")
    report.add_argument("--name", required=True)
    _add_location_argument(report)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    runner = ComparisonRunner(args.experiments_dir)

    if args.command == "create":
        config = ComparisonConfig(
            total_timesteps=args.timesteps,
            module_count=args.modules,
            control_cost_weight=args.control_cost_weight,
            checkpoint_interval=args.checkpoint_every,
            validation_episode_count=args.validation_episodes,
            test_episode_count=args.test_episodes,
        )
        runner.create(args.name, config)
        print(f"Created comparison: {args.name}")
        return

    if args.command == "run":
        runner.run(args.name, args.policy_type, args.seed)
        print(f"Selected comparison runs are complete: {args.name}")
        return

    if args.command == "extend":
        runner.extend(args.name, args.timesteps)
        print(f"Extended comparison runs are complete: {args.name}")
        return

    if args.command == "status":
        for row in runner.status(args.name):
            print(
                f"{row['policy_type']:7} seed={row['seed']} "
                f"{row['status']:12} {row['timesteps']:,} / "
                f"{row['target_timesteps']:,} steps"
            )
        return

    report_path = runner.report(args.name)
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Comparison interrupted. Completed checkpoints were preserved.")
        raise SystemExit(130)
