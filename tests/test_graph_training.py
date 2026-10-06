import unittest

from stable_baselines3 import PPO

from nervenet.envs import CrawlerEnv, GraphObservationWrapper
from nervenet.graphs import get_actuator_node_indices, get_message_routes
from nervenet.policies import GraphActorCriticPolicy
from nervenet.training import train_graph_policy


class GraphPolicyTrainingTest(unittest.TestCase):
    def test_total_timesteps_must_be_positive(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "total_timesteps must be positive",
        ):
            train_graph_policy(total_timesteps=0, seed=0)

    def test_graph_policy_can_complete_one_ppo_update(self) -> None:
        env = GraphObservationWrapper(
            CrawlerEnv(
                module_count=1,
                max_episode_seconds=0.04,
            )
        )

        model = PPO(
            policy=GraphActorCriticPolicy,
            env=env,
            policy_kwargs={
                "routes": get_message_routes(env.graph),
                "actuator_node_indices": get_actuator_node_indices(env.graph),
            },
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            seed=0,
            device="cpu",
            verbose=0,
        )

        model.learn(total_timesteps=8)

        self.assertEqual(model.num_timesteps, 8)


if __name__ == "__main__":
    unittest.main()
