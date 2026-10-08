import unittest

import torch

from nervenet.policies import MatchedActorCriticExtractor


class MatchedActorCriticExtractorTest(unittest.TestCase):
    def test_actor_and_critic_return_expected_batch_shapes(self) -> None:
        extractor = MatchedActorCriticExtractor(
            observation_shape=(5, 17),
            action_size=4,
            actor_hidden_size=192,
        )
        observations = torch.randn(3, 5, 17)

        action_means, state_values = extractor(observations)

        self.assertEqual(action_means.shape, (3, 4))
        self.assertEqual(state_values.shape, (3, 1))

    def test_actor_and_critic_share_no_parameters(self) -> None:
        extractor = MatchedActorCriticExtractor(
            observation_shape=(5, 17),
            action_size=4,
            actor_hidden_size=192,
        )

        actor_parameters = {
            id(parameter) for parameter in extractor.actor.parameters()
        }
        critic_parameters = {
            id(parameter) for parameter in extractor.critic.parameters()
        }

        self.assertTrue(actor_parameters.isdisjoint(critic_parameters))


if __name__ == "__main__":
    unittest.main()
