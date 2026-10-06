from pathlib import Path
from typing import Literal

from stable_baselines3 import PPO

from nervenet.envs import CrawlerEnv, GraphObservationWrapper


PolicyType = Literal["flat", "graph"]
PolicyEnvironment = CrawlerEnv | GraphObservationWrapper


def default_policy_path(
    policy_type: PolicyType,
    module_count: int,
) -> Path:
    return Path(f"artifacts/{policy_type}_policy_{module_count}_modules.zip")


def create_policy_environment(
    policy_type: PolicyType,
    module_count: int,
) -> PolicyEnvironment:
    env = CrawlerEnv(module_count=module_count)

    if policy_type == "flat":
        return env
    if policy_type == "graph":
        return GraphObservationWrapper(env)

    raise ValueError(f"unknown policy type: {policy_type}")


def load_crawler_policy(
    model_path: Path,
    policy_type: PolicyType,
    module_count: int,
) -> tuple[PPO, PolicyEnvironment]:
    env = create_policy_environment(policy_type, module_count)
    model = PPO.load(model_path, env=env, device="cpu")

    return model, env
