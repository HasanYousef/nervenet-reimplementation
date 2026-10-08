import unittest

import torch

from nervenet.graphs import BodyNodeType
from nervenet.policies import NodeStateUpdater


class NodeStateUpdaterTest(unittest.TestCase):
    def test_updater_preserves_node_count_and_hidden_size(self) -> None:
        updater = NodeStateUpdater(hidden_size=64)

        hidden_states = torch.randn(5, 64)
        messages = torch.randn(5, 64)

        node_types = (
            BodyNodeType.ROOT,
            BodyNodeType.JOINT,
            BodyNodeType.JOINT,
            BodyNodeType.JOINT,
            BodyNodeType.BODY,
        )

        updated_states = updater(hidden_states, messages, node_types)

        self.assertEqual(updated_states.shape, (5, 64))

    def test_same_updater_handles_different_graph_sizes(self) -> None:
        updater = NodeStateUpdater(hidden_size=64)

        five_node_result = updater(
            torch.randn(5, 64),
            torch.randn(5, 64),
            (BodyNodeType.ROOT,) + (BodyNodeType.JOINT,) * 4,
        )
        fifteen_node_result = updater(
            torch.randn(15, 64),
            torch.randn(15, 64),
            (BodyNodeType.ROOT,) + (BodyNodeType.JOINT,) * 14,
        )

        self.assertEqual(five_node_result.shape, (5, 64))
        self.assertEqual(fifteen_node_result.shape, (15, 64))

    def test_node_types_use_separate_gru_parameters(self) -> None:
        updater = NodeStateUpdater(hidden_size=64)

        self.assertIsNot(
            updater.grus[BodyNodeType.ROOT.value],
            updater.grus[BodyNodeType.JOINT.value],
        )
        self.assertIsNot(
            updater.grus[BodyNodeType.JOINT.value],
            updater.grus[BodyNodeType.BODY.value],
        )

    def test_node_type_count_must_match_node_count(self) -> None:
        updater = NodeStateUpdater(hidden_size=64)

        with self.assertRaisesRegex(ValueError, "one type per node"):
            updater(
                torch.randn(5, 64),
                torch.randn(5, 64),
                (BodyNodeType.ROOT, BodyNodeType.JOINT),
            )

    def test_every_node_type_must_be_valid(self) -> None:
        updater = NodeStateUpdater(hidden_size=64)

        with self.assertRaisesRegex(TypeError, "must be a BodyNodeType"):
            updater(
                torch.randn(2, 64),
                torch.randn(2, 64),
                (BodyNodeType.ROOT, "joint"),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
