from pathlib import Path

from stable_baselines3 import PPO

from nervenet.envs import CrawlerEnv, GraphObservationWrapper
from nervenet.policies import MatchedActorCriticPolicy
from nervenet.training.checkpoint import load_ppo_checkpoint


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
    resume_from: Path | None = None,
) -> PPO:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")

    if resume_from is None:
        model = create_matched_policy(
            seed=seed,
            module_count=module_count,
            control_cost_weight=control_cost_weight,
            verbose=1,
        )
    else:
        env = GraphObservationWrapper(
            CrawlerEnv(
                module_count=module_count,
                control_cost_weight=control_cost_weight,
            )
        )
        model = load_ppo_checkpoint(
            checkpoint=resume_from,
            env=env,
            expected_policy_type=MatchedActorCriticPolicy,
        )

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=resume_from is None,
    )
    return model
