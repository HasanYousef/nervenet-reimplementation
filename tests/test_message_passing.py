import unittest

import torch

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

        updated_states = layer(hidden_states, routes)

        self.assertEqual(updated_states.shape, (5, 64))

    def test_training_gradients_flow_through_the_round(self) -> None:
        layer = MessagePassingLayer(hidden_size=64)
        hidden_states = torch.randn(3, 64, requires_grad=True)
        routes = [(0, 1), (1, 0), (1, 2), (2, 1)]

        updated_states = layer(hidden_states, routes)
        updated_states.sum().backward()

        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(torch.isfinite(hidden_states.grad).all())


if __name__ == "__main__":
    unittest.main()
