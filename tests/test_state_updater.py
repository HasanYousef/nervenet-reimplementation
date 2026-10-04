import unittest

import torch

from nervenet.policies import NodeStateUpdater


class NodeStateUpdaterTest(unittest.TestCase):
    def test_updater_preserves_node_count_and_hidden_size(self) -> None:
        updater = NodeStateUpdater(hidden_size=64)

        hidden_states = torch.randn(5, 64)
        messages = torch.randn(5, 64)

        updated_states = updater(hidden_states, messages)

        self.assertEqual(updated_states.shape, (5, 64))

    def test_same_updater_handles_different_graph_sizes(self) -> None:
        updater = NodeStateUpdater(hidden_size=64)

        five_node_result = updater(
            torch.randn(5, 64),
            torch.randn(5, 64),
        )
        fifteen_node_result = updater(
            torch.randn(15, 64),
            torch.randn(15, 64),
        )

        self.assertEqual(five_node_result.shape, (5, 64))
        self.assertEqual(fifteen_node_result.shape, (15, 64))


if __name__ == "__main__":
    unittest.main()
