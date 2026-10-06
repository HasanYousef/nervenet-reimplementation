from stable_baselines3 import PPO

from nervenet.envs import CrawlerEnv, GraphObservationWrapper
from nervenet.graphs import (
    get_actuator_node_indices,
    get_message_routes,
)
from nervenet.policies import GraphActorCriticPolicy


def train_graph_policy(
    total_timesteps: int,
    seed: int,
    module_count: int = 3,
    control_cost_weight: float = 0.05,
    hidden_size: int = 64,
    message_passing_steps: int = 2,
) -> PPO:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")

    env = GraphObservationWrapper(
        CrawlerEnv(
            module_count=module_count,
            control_cost_weight=control_cost_weight,
        )
    )

    model = PPO(
        policy=GraphActorCriticPolicy,
        env=env,
        policy_kwargs={
            "routes": get_message_routes(env.graph),
            "actuator_node_indices": get_actuator_node_indices(env.graph),
            "hidden_size": hidden_size,
            "message_passing_steps": message_passing_steps,
        },
        seed=seed,
        device="cpu",
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)

    return model
