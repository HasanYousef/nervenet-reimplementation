from pathlib import Path

from stable_baselines3 import PPO

from nervenet.envs import CrawlerEnv, GraphObservationWrapper
from nervenet.graphs import (
    get_actuator_node_indices,
    get_message_routes,
)
from nervenet.policies import GraphActorCriticPolicy
from nervenet.policies.defaults import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MESSAGE_PASSING_STEPS,
)
from nervenet.training.checkpoint import load_ppo_checkpoint


def train_graph_policy(
    total_timesteps: int,
    seed: int,
    module_count: int = 3,
    control_cost_weight: float = 0.05,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    message_passing_steps: int = DEFAULT_MESSAGE_PASSING_STEPS,
    resume_from: Path | None = None,
) -> PPO:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")

    env = GraphObservationWrapper(
        CrawlerEnv(
            module_count=module_count,
            control_cost_weight=control_cost_weight,
        )
    )

    if resume_from is None:
        model = PPO(
            policy=GraphActorCriticPolicy,
            env=env,
            policy_kwargs={
                "routes": get_message_routes(env.graph),
                "actuator_node_indices": get_actuator_node_indices(env.graph),
                "node_types": tuple(node.node_type for node in env.graph.nodes),
                "hidden_size": hidden_size,
                "message_passing_steps": message_passing_steps,
            },
            seed=seed,
            device="cpu",
            verbose=1,
        )
    else:
        model = load_ppo_checkpoint(
            checkpoint=resume_from,
            env=env,
            expected_policy_type=GraphActorCriticPolicy,
        )

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=resume_from is None,
    )

    return model
