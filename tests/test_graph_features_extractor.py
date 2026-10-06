import unittest

from gymnasium import spaces
import numpy as np
import torch

from nervenet.policies import GraphFeaturesExtractor


class GraphFeaturesExtractorTest(unittest.TestCase):
    def test_extractor_preserves_graph_shape_and_values(self) -> None:
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15, 11),
            dtype=np.float64,
        )
        extractor = GraphFeaturesExtractor(observation_space)
        observations = torch.randn(4, 15, 11)

        result = extractor(observations)

        self.assertEqual(extractor.features_dim, 165)
        self.assertEqual(result.shape, (4, 15, 11))
        torch.testing.assert_close(result, observations)


if __name__ == "__main__":
    unittest.main()
