from stable_baselines3 import PPO

from nervenet.envs import CrawlerEnv, GraphObservationWrapper
from nervenet.policies import MatchedActorCriticPolicy


MATCHED_ACTOR_HIDDEN_SIZE = 151


def create_matched_policy(
    seed: int,
    module_count: int = 3,
    control_cost_weight: float = 0.05,
    verbose: int = 0,
) -> PPO:
    env = GraphObservationWrapper(
        CrawlerEnv(
            module_count=module_count,
            control_cost_weight=control_cost_weight,
        )
    )

    return PPO(
        policy=MatchedActorCriticPolicy,
        env=env,
        policy_kwargs={
            "actor_hidden_size": MATCHED_ACTOR_HIDDEN_SIZE,
        },
        seed=seed,
        device="cpu",
        verbose=verbose,
    )


def train_matched_policy(
    total_timesteps: int,
    seed: int,
    module_count: int = 3,
    control_cost_weight: float = 0.05,
) -> PPO:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")

    model = create_matched_policy(
        seed=seed,
        module_count=module_count,
        control_cost_weight=control_cost_weight,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)
    return model
