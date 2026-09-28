import argparse
from pathlib import Path

from nervenet.training.flat_policy import train_flat_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the flat PPO crawler baseline.")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/flat_policy"),
    )
    args = parser.parse_args()

    model = train_flat_policy(
        total_timesteps=args.timesteps,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)

    print(f"Saved model to {args.output}.zip")


if __name__ == "__main__":
    main()
