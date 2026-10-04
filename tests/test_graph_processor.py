import unittest

import torch

from nervenet.policies import GraphProcessor


class GraphProcessorTest(unittest.TestCase):
    def test_multiple_rounds_preserve_hidden_state_shape(self) -> None:
        processor = GraphProcessor(
            hidden_size=64,
            message_passing_steps=2,
        )

        hidden_states = torch.randn(5, 64)
        routes = [(0, 1), (1, 0), (1, 2), (2, 1)]

        result = processor(hidden_states, routes)

        self.assertEqual(result.shape, (5, 64))

    def test_round_count_does_not_add_parameters(self) -> None:
        one_round = GraphProcessor(message_passing_steps=1)
        three_rounds = GraphProcessor(message_passing_steps=3)

        one_round_parameters = sum(
            parameter.numel() for parameter in one_round.parameters()
        )
        three_round_parameters = sum(
            parameter.numel() for parameter in three_rounds.parameters()
        )

        self.assertEqual(one_round_parameters, three_round_parameters)

    def test_round_count_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            GraphProcessor(message_passing_steps=0)


if __name__ == "__main__":
    unittest.main()
