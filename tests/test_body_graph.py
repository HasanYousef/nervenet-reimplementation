import unittest

from nervenet.graphs import (
    BodyNodeType,
    build_body_graph,
    get_actuator_node_indices,
    get_message_routes,
)
from nervenet.models import build_crawler_model


class BodyGraphTest(unittest.TestCase):
    def test_one_module_body_hierarchy_and_ownership(self) -> None:
        graph = build_body_graph(build_crawler_model(1))
        nodes = {node.name: node for node in graph.nodes}

        self.assertEqual(len(graph.nodes), 5)

        torso = nodes["module_1_torso"]
        self.assertIsNone(torso.parent_body_id)
        self.assertEqual(torso.joint_ids, (0,))
        self.assertEqual(torso.actuator_ids, ())

        upper_leg = nodes["module_1_left_upper_leg"]
        self.assertEqual(upper_leg.parent_body_id, torso.body_id)
        self.assertEqual(upper_leg.joint_ids, (1,))
        self.assertEqual(upper_leg.actuator_ids, (0,))

        lower_leg = nodes["module_1_left_lower_leg"]
        self.assertEqual(lower_leg.parent_body_id, upper_leg.body_id)
        self.assertEqual(lower_leg.joint_ids, (2,))
        self.assertEqual(lower_leg.actuator_ids, (1,))

    def test_passive_spine_belongs_to_second_torso(self) -> None:
        graph = build_body_graph(build_crawler_model(2))
        nodes = {node.name: node for node in graph.nodes}

        first_torso = nodes["module_1_torso"]
        second_torso = nodes["module_2_torso"]

        self.assertEqual(second_torso.parent_body_id, first_torso.body_id)
        self.assertEqual(len(second_torso.joint_ids), 1)
        self.assertEqual(second_torso.actuator_ids, ())
        self.assertEqual(
            second_torso.node_type,
            BodyNodeType.JOINT,
        )

    def test_graph_scales_with_module_count(self) -> None:
        one_module = build_body_graph(build_crawler_model(1))
        three_modules = build_body_graph(build_crawler_model(3))

        self.assertEqual(len(one_module.nodes), 5)
        self.assertEqual(len(three_modules.nodes), 15)
        self.assertEqual(
            sum(len(node.actuator_ids) for node in three_modules.nodes),
            12,
        )

    def test_one_module_message_routes_follow_body_hierarchy(self) -> None:
        graph = build_body_graph(build_crawler_model(1))

        routes = get_message_routes(graph)

        self.assertEqual(
            routes,
            [
                (0, 1),
                (1, 0),
                (1, 2),
                (2, 1),
                (0, 3),
                (3, 0),
                (3, 4),
                (4, 3),
            ],
        )

    def test_actuator_nodes_follow_mujoco_actuator_order(self) -> None:
        one_module = build_body_graph(build_crawler_model(1))
        two_modules = build_body_graph(build_crawler_model(2))

        self.assertEqual(
            get_actuator_node_indices(one_module),
            (1, 2, 3, 4),
        )
        self.assertEqual(
            get_actuator_node_indices(two_modules),
            (1, 2, 3, 4, 6, 7, 8, 9),
        )

    def test_nodes_are_classified_by_physical_role(self) -> None:
        model = build_crawler_model(1)
        graph = build_body_graph(model)

        node_types = {node.name: node.node_type for node in graph.nodes}

        self.assertEqual(
            node_types["module_1_torso"],
            BodyNodeType.ROOT,
        )
        self.assertEqual(
            node_types["module_1_left_upper_leg"],
            BodyNodeType.JOINT,
        )
        self.assertEqual(
            node_types["module_1_left_lower_leg"],
            BodyNodeType.JOINT,
        )


if __name__ == "__main__":
    unittest.main()
