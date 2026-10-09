import argparse
from pathlib import Path

from nervenet.policy_loading import default_policy_path
from nervenet.viewing import view_crawler_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="View a trained crawler policy.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
    )
    parser.add_argument("--modules", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy-type",
        choices=("flat", "matched", "graph"),
        default="flat",
    )
    args = parser.parse_args()

    model_path = args.model or default_policy_path(
        args.policy_type,
        args.modules,
    )
    view_crawler_policy(
        model_path=model_path,
        policy_type=args.policy_type,
        module_count=args.modules,
        seed=args.seed,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
