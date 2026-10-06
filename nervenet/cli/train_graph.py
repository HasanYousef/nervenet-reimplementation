import argparse
from pathlib import Path

from nervenet.policies.defaults import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MESSAGE_PASSING_STEPS,
)
from nervenet.training import train_graph_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the graph PPO crawler policy.")
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
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
    )
    parser.add_argument(
        "--message-passing-steps",
        type=int,
        default=DEFAULT_MESSAGE_PASSING_STEPS,
    )
    args = parser.parse_args()
    output = args.output or Path(f"artifacts/graph_policy_{args.modules}_modules")

    model = train_graph_policy(
        total_timesteps=args.timesteps,
        seed=args.seed,
        module_count=args.modules,
        control_cost_weight=args.control_cost_weight,
        hidden_size=args.hidden_size,
        message_passing_steps=args.message_passing_steps,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)

    print(f"Saved model to {output}.zip")


if __name__ == "__main__":
    main()
