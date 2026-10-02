import unittest

import torch

from nervenet.policies import NodeInputEncoder


class NodeInputEncoderTest(unittest.TestCase):
    def test_encoder_preserves_node_count_and_changes_feature_width(self) -> None:
        encoder = NodeInputEncoder(
            input_size=11,
            hidden_size=64,
        )

        observations = torch.zeros((5, 11))

        hidden_states = encoder(observations)

        self.assertEqual(hidden_states.shape, (5, 64))

    def test_same_encoder_handles_different_graph_sizes(self) -> None:
        encoder = NodeInputEncoder(
            input_size=11,
            hidden_size=64,
        )

        one_module = torch.zeros((5, 11))
        three_modules = torch.zeros((15, 11))

        one_module_hidden = encoder(one_module)
        three_module_hidden = encoder(three_modules)

        self.assertEqual(one_module_hidden.shape, (5, 64))
        self.assertEqual(three_module_hidden.shape, (15, 64))

    def test_parameter_count_does_not_depend_on_node_count(self) -> None:
        encoder = NodeInputEncoder(
            input_size=11,
            hidden_size=64,
        )

        parameter_count = sum(
            parameter.numel()
            for parameter in encoder.parameters()
        )

        self.assertEqual(parameter_count, 768)

    def test_identical_node_observations_produce_identical_hidden_states(
        self,
    ) -> None:
        encoder = NodeInputEncoder(
            input_size=11,
            hidden_size=64,
        )

        node_observation = torch.tensor(
            [0.4, -1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        observations = torch.stack([node_observation, node_observation])

        hidden_states = encoder(observations)

        torch.testing.assert_close(
            hidden_states[0],
            hidden_states[1],
        )


if __name__ == "__main__":
    unittest.main()
