import unittest

import mujoco
import torch

from nervenet.graphs import (
    build_body_graph,
    get_actuator_node_indices,
    get_graph_observations,
    get_message_routes,
    pad_graph_observations,
)
from nervenet.models import build_crawler_model
from nervenet.policies import GraphActor


class GraphActorTest(unittest.TestCase):
    def test_same_actor_handles_different_crawler_sizes(self) -> None:
        actor = GraphActor()

        one_module_actions = self._run_actor(actor, module_count=1)
        three_module_actions = self._run_actor(actor, module_count=3)

        self.assertEqual(one_module_actions.shape, (4,))
        self.assertEqual(three_module_actions.shape, (12,))
        self.assertTrue(torch.isfinite(one_module_actions).all())
        self.assertTrue(torch.isfinite(three_module_actions).all())

    def _run_actor(
        self,
        actor: GraphActor,
        module_count: int,
    ) -> torch.Tensor:
        model = build_crawler_model(module_count)
        data = mujoco.MjData(model)
        graph = build_body_graph(model)

        observations = pad_graph_observations(
            get_graph_observations(model, data, graph)
        )
        observation_tensor = torch.tensor(
            observations,
            dtype=torch.float32,
        )

        return actor(
            observation_tensor,
            get_message_routes(graph),
            get_actuator_node_indices(graph),
        )

    def test_actor_processes_a_batch_of_graph_observations(self) -> None:
        actor = GraphActor()

        observations = torch.randn(3, 5, 11)
        routes = [
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (0, 3),
            (3, 0),
            (3, 4),
            (4, 3),
        ]

        actions = actor(
            observations,
            routes,
            actuator_node_indices=(1, 2, 3, 4),
        )

        self.assertEqual(actions.shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
