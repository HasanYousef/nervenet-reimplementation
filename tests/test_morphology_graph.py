import unittest

from nervenet.graphs import build_morphology_graph
from nervenet.models import build_crawler_model


class MorphologyGraphTest(unittest.TestCase):
    def test_three_module_crawler_topology(self) -> None:
        graph = build_morphology_graph(build_crawler_model(3))

        self.assertEqual(len(graph.nodes), 15)
        self.assertEqual(len(graph.edges), 14)
        self.assertEqual(graph.action_size, 12)
        self.assertEqual(graph.nodes[0].name, "module_1_torso")
        self.assertIsNone(graph.nodes[0].parent_index)

    def test_actuators_belong_to_their_child_bodies(self) -> None:
        graph = build_morphology_graph(build_crawler_model(2))
        nodes = {node.name: node for node in graph.nodes}

        self.assertEqual(len(nodes["module_1_left_upper_leg"].actuator_ids), 1)
        self.assertEqual(len(nodes["module_1_left_lower_leg"].actuator_ids), 1)
        self.assertEqual(len(nodes["module_1_torso"].actuator_ids), 0)
        self.assertEqual(len(nodes["module_2_torso"].actuator_ids), 0)

    def test_graph_size_scales_without_changing_action_ownership(self) -> None:
        one_module = build_morphology_graph(build_crawler_model(1))
        three_modules = build_morphology_graph(build_crawler_model(3))

        self.assertEqual((len(one_module.nodes), one_module.action_size), (5, 4))
        self.assertEqual((len(three_modules.nodes), three_modules.action_size), (15, 12))


if __name__ == "__main__":
    unittest.main()
