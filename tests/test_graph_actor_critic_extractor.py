import unittest

import torch

from nervenet.graphs import BodyNodeType
from nervenet.policies import GraphActorCriticExtractor


class GraphActorCriticExtractorTest(unittest.TestCase):
    def test_actor_and_critic_return_expected_shapes(self) -> None:
        extractor = GraphActorCriticExtractor(
            observation_shape=(5, 17),
            routes=[
                (0, 1),
                (1, 0),
                (1, 2),
                (2, 1),
                (0, 3),
                (3, 0),
                (3, 4),
                (4, 3),
            ],
            actuator_node_indices=(1, 2, 3, 4),
            node_types=(BodyNodeType.ROOT,) + (BodyNodeType.JOINT,) * 4,
        )

        observations = torch.randn(3, 5, 17)

        action_means, state_values = extractor(observations)

        self.assertEqual(extractor.latent_dim_pi, 4)
        self.assertEqual(extractor.latent_dim_vf, 1)
        self.assertEqual(action_means.shape, (3, 4))
        self.assertEqual(state_values.shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
