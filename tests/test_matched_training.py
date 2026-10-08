from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from stable_baselines3 import PPO

from nervenet.envs import CrawlerEnv, GraphObservationWrapper
from nervenet.graphs import get_actuator_node_indices, get_message_routes
from nervenet.policies import (
    GraphActorCriticPolicy,
    MatchedActorCriticPolicy,
)
from nervenet.training.matched_policy import (
    MATCHED_ACTOR_HIDDEN_SIZE,
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

        matched_actor_parameters = sum(
            parameter.numel()
            for parameter in matched.policy.mlp_extractor.actor.parameters()
        )
        graph_actor_parameters = sum(
            parameter.numel()
            for parameter in graph.policy.mlp_extractor.actor.parameters()
        )
        active_node_types = {
            node_type.value
            for node_type in graph.policy.mlp_extractor.node_types
        }
        state_updater = (
            graph.policy.mlp_extractor.actor.processor.layer.state_updater
        )
        inactive_graph_actor_parameters = sum(
            parameter.numel()
            for node_type, gru in state_updater.grus.items()
            if node_type not in active_node_types
            for parameter in gru.parameters()
        )
        active_graph_actor_parameters = (
            graph_actor_parameters - inactive_graph_actor_parameters
        )
        matched_critic_parameters = sum(
            parameter.numel()
            for parameter in matched.policy.mlp_extractor.critic.parameters()
        )
        graph_critic_parameters = sum(
            parameter.numel()
            for parameter in graph.policy.mlp_extractor.critic.parameters()
        )
        matched_total_parameters = sum(
            parameter.numel() for parameter in matched.policy.parameters()
        )
        graph_total_parameters = sum(
            parameter.numel() for parameter in graph.policy.parameters()
        )
        active_graph_total_parameters = (
            graph_total_parameters - inactive_graph_actor_parameters
        )
        actor_relative_difference = abs(
            matched_actor_parameters - active_graph_actor_parameters
        ) / active_graph_actor_parameters
        total_relative_difference = abs(
            matched_total_parameters - active_graph_total_parameters
        ) / active_graph_total_parameters

        self.assertLess(actor_relative_difference, 0.01)
        self.assertEqual(matched_critic_parameters, graph_critic_parameters)
        self.assertLess(total_relative_difference, 0.01)
        self.assertFalse(matched.policy.ortho_init)
        self.assertFalse(graph.policy.ortho_init)

    def test_resume_adds_timesteps_to_saved_matched_policy(self) -> None:
        env = GraphObservationWrapper(CrawlerEnv(module_count=1))
        model = PPO(
            policy=MatchedActorCriticPolicy,
            env=env,
            policy_kwargs={
                "actor_hidden_size": MATCHED_ACTOR_HIDDEN_SIZE,
            },
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            seed=0,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=8)

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "matched_checkpoint"
            model.save(checkpoint)
            resumed = train_matched_policy(
                total_timesteps=8,
                seed=0,
                module_count=1,
                resume_from=checkpoint.with_suffix(".zip"),
            )

        self.assertEqual(resumed.num_timesteps, 16)


if __name__ == "__main__":
    unittest.main()
