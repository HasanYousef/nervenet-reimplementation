import unittest

import torch

from nervenet.policies import FlatValueNetwork


class FlatValueNetworkTest(unittest.TestCase):
    def test_network_returns_one_value_per_observation(self) -> None:
        network = FlatValueNetwork(
            observation_shape=(15, 11),
        )

        single_value = network(torch.randn(15, 11))
        batch_values = network(torch.randn(32, 15, 11))

        self.assertEqual(single_value.shape, (1,))
        self.assertEqual(batch_values.shape, (32, 1))

    def test_training_gradients_reach_the_observation(self) -> None:
        network = FlatValueNetwork(
            observation_shape=(15, 11),
        )
        observations = torch.randn(
            4,
            15,
            11,
            requires_grad=True,
        )

        values = network(observations)
        values.sum().backward()

        self.assertIsNotNone(observations.grad)
        self.assertTrue(torch.isfinite(observations.grad).all())


if __name__ == "__main__":
    unittest.main()
