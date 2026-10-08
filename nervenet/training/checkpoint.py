from pathlib import Path

from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy


def load_ppo_checkpoint(
    checkpoint: Path,
    env: Env,
    expected_policy_type: type[BasePolicy],
) -> PPO:
    model = PPO.load(
        checkpoint,
        env=env,
        device="cpu",
        verbose=1,
    )

    if not isinstance(model.policy, expected_policy_type):
        raise ValueError(
            f"checkpoint contains {type(model.policy).__name__}, expected "
            f"{expected_policy_type.__name__}"
        )

    return model
