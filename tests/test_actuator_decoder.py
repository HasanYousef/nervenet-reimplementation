import unittest

import torch

from nervenet.policies import ActuatorDecoder


class ActuatorDecoderTest(unittest.TestCase):
    def test_decoder_selects_nodes_in_actuator_order(self) -> None:
        decoder = ActuatorDecoder(hidden_size=2)

        with torch.no_grad():
            decoder.linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
            decoder.linear.bias.zero_()

        hidden_states = torch.tensor(
            [
                [10.0, 1.0],
                [20.0, 2.0],
                [30.0, 3.0],
            ]
        )

        result = decoder(
            hidden_states,
            actuator_node_indices=(2, 0),
        )

        torch.testing.assert_close(
            result,
            torch.tensor([30.0, 10.0]),
        )

    def test_same_decoder_handles_different_actuator_counts(self) -> None:
        decoder = ActuatorDecoder(hidden_size=64)

        four_actions = decoder(
            torch.randn(5, 64),
            actuator_node_indices=(1, 2, 3, 4),
        )
        eight_actions = decoder(
            torch.randn(10, 64),
            actuator_node_indices=(1, 2, 3, 4, 6, 7, 8, 9),
        )

        self.assertEqual(four_actions.shape, (4,))
        self.assertEqual(eight_actions.shape, (8,))


if __name__ == "__main__":
    unittest.main()
