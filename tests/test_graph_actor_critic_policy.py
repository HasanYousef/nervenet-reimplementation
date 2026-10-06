import unittest

import numpy as np
import torch

from nervenet.envs import CrawlerEnv, GraphObservationWrapper
from nervenet.graphs import (
    get_actuator_node_indices,
    get_message_routes,
)
from nervenet.policies import GraphActorCriticPolicy


class GraphActorCriticPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = GraphObservationWrapper(CrawlerEnv(module_count=1))
        self.routes = get_message_routes(self.env.graph)
        self.actuator_node_indices = get_actuator_node_indices(
            self.env.graph
        )
        self.policy = GraphActorCriticPolicy(
            self.env.observation_space,
            self.env.action_space,
            lambda _: 3e-4,
            routes=self.routes,
            actuator_node_indices=self.actuator_node_indices,
        )

    def test_forward_returns_actions_values_and_log_probabilities(self) -> None:
        observation, _ = self.env.reset(seed=0)
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
        ).unsqueeze(0)

        actions, values, log_prob = self.policy(
            observation_tensor,
            deterministic=False,
        )

        self.assertEqual(actions.shape, (1, 4))
        self.assertEqual(values.shape, (1, 1))
        self.assertEqual(log_prob.shape, (1,))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(torch.isfinite(log_prob).all())

    def test_evaluate_actions_returns_ppo_training_values(self) -> None:
        observation, _ = self.env.reset(seed=0)
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
        ).unsqueeze(0)
        actions, _, _ = self.policy(
            observation_tensor,
            deterministic=True,
        )

        values, log_prob, entropy = self.policy.evaluate_actions(
            observation_tensor,
            actions,
        )

        self.assertEqual(values.shape, (1, 1))
        self.assertEqual(log_prob.shape, (1,))
        self.assertIsNotNone(entropy)
        self.assertEqual(entropy.shape, (1,))

    def test_predict_returns_one_action_per_motor(self) -> None:
        observation, _ = self.env.reset(seed=0)

        action, _ = self.policy.predict(
            observation,
            deterministic=True,
        )

        self.assertEqual(action.shape, (4,))
        self.assertTrue(np.isfinite(action).all())

    def test_actuator_mapping_must_match_action_space(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Actuator-node count must match",
        ):
            GraphActorCriticPolicy(
                self.env.observation_space,
                self.env.action_space,
                lambda _: 3e-4,
                routes=self.routes,
                actuator_node_indices=(1, 2, 3),
            )


if __name__ == "__main__":
    unittest.main()
