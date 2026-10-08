import unittest

import torch

from nervenet.graphs import BodyNodeType
from nervenet.policies import MessagePassingLayer


class MessagePassingLayerTest(unittest.TestCase):
    def test_one_round_preserves_hidden_state_shape(self) -> None:
        layer = MessagePassingLayer(hidden_size=64)
        hidden_states = torch.randn(5, 64)

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

        node_types = (BodyNodeType.ROOT,) + (BodyNodeType.JOINT,) * 4

        updated_states = layer(hidden_states, routes, node_types)

        self.assertEqual(updated_states.shape, (5, 64))

    def test_training_gradients_flow_through_the_round(self) -> None:
        layer = MessagePassingLayer(hidden_size=64)
        hidden_states = torch.randn(3, 64, requires_grad=True)
        routes = [(0, 1), (1, 0), (1, 2), (2, 1)]

        node_types = (
            BodyNodeType.ROOT,
            BodyNodeType.JOINT,
            BodyNodeType.BODY,
        )

        updated_states = layer(hidden_states, routes, node_types)
        updated_states.sum().backward()

        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(torch.isfinite(hidden_states.grad).all())
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in layer.message_network.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
