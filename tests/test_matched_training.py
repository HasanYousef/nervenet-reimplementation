import unittest

from stable_baselines3 import PPO

from nervenet.envs import CrawlerEnv, GraphObservationWrapper
from nervenet.graphs import get_actuator_node_indices, get_message_routes
from nervenet.policies import GraphActorCriticPolicy
from nervenet.training.matched_policy import (
    create_matched_policy,
    train_matched_policy,
)


class MatchedPolicyTrainingTest(unittest.TestCase):
    def test_total_timesteps_must_be_positive(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "total_timesteps must be positive",
        ):
            train_matched_policy(total_timesteps=0, seed=0)

    def test_three_module_policy_matches_graph_parameter_count(self) -> None:
        matched = create_matched_policy(seed=0, module_count=3)

        graph_env = GraphObservationWrapper(CrawlerEnv(module_count=3))
        graph = PPO(
            policy=GraphActorCriticPolicy,
            env=graph_env,
            policy_kwargs={
                "routes": get_message_routes(graph_env.graph),
                "actuator_node_indices": get_actuator_node_indices(
                    graph_env.graph
                ),
                "node_types": tuple(
                    node.node_type for node in graph_env.graph.nodes
                ),
            },
            seed=0,
            device="cpu",
            verbose=0,
        )

        matched_parameters = sum(
            parameter.numel() for parameter in matched.policy.parameters()
        )
        graph_parameters = sum(
            parameter.numel() for parameter in graph.policy.parameters()
        )
        relative_difference = abs(
            matched_parameters - graph_parameters
        ) / graph_parameters

        self.assertLess(relative_difference, 0.01)


if __name__ == "__main__":
    unittest.main()
