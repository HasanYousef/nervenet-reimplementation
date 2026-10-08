from pathlib import Path
from typing import Literal

from stable_baselines3 import PPO

from nervenet.envs import CrawlerEnv, GraphObservationWrapper


PolicyType = Literal["flat", "matched", "graph"]
PolicyEnvironment = CrawlerEnv | GraphObservationWrapper


def default_policy_path(
    policy_type: PolicyType,
    module_count: int,
) -> Path:
    return Path(f"artifacts/{policy_type}_policy_{module_count}_modules.zip")


def create_policy_environment(
    policy_type: PolicyType,
    module_count: int,
    control_cost_weight: float = 0.05,
) -> PolicyEnvironment:
    env = CrawlerEnv(
        module_count=module_count,
        control_cost_weight=control_cost_weight,
    )

    if policy_type == "flat":
        return env
    if policy_type in ("matched", "graph"):
        return GraphObservationWrapper(env)

    raise ValueError(f"unknown policy type: {policy_type}")


def load_crawler_policy(
    model_path: Path,
    policy_type: PolicyType,
    module_count: int,
    control_cost_weight: float = 0.05,
) -> tuple[PPO, PolicyEnvironment]:
    env = create_policy_environment(
        policy_type,
        module_count,
        control_cost_weight,
    )
    model = PPO.load(
        model_path,
        device="cpu",
        verbose=0,
    )

    return model, env
