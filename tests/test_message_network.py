import unittest

import torch

from nervenet.policies import MessageNetwork


class MessageNetworkTest(unittest.TestCase):
    def test_preserves_node_and_hidden_dimensions(self) -> None:
        network = MessageNetwork(hidden_size=64)
        hidden_states = torch.randn(5, 64)

        messages = network(hidden_states)

        self.assertEqual(messages.shape, (5, 64))

    def test_processes_batched_graphs(self) -> None:
        network = MessageNetwork(hidden_size=64)
        hidden_states = torch.randn(3, 5, 64)

        messages = network(hidden_states)

        self.assertEqual(messages.shape, (3, 5, 64))


if __name__ == "__main__":
    unittest.main()
