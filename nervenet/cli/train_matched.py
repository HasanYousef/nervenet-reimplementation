import argparse
from pathlib import Path

from nervenet.training import train_matched_policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the parameter-matched MLP crawler baseline."
    )
    parser.add_argument("--modules", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--control-cost-weight",
        type=float,
        default=0.05,
    )
    args = parser.parse_args()
    output = args.output or Path(
        f"artifacts/matched_policy_{args.modules}_modules"
    )

    model = train_matched_policy(
        total_timesteps=args.timesteps,
        seed=args.seed,
        module_count=args.modules,
        control_cost_weight=args.control_cost_weight,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)

    print(f"Saved model to {output}.zip")


if __name__ == "__main__":
    main()
