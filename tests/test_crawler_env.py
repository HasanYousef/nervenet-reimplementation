import unittest

import numpy as np

from nervenet.envs import CrawlerEnv


class CrawlerEnvTest(unittest.TestCase):
    def test_two_module_spaces_match_model_dimensions(self) -> None:
        env = CrawlerEnv(module_count=2)

        self.assertEqual(env.action_space.shape, (8,))
        self.assertEqual(env.observation_space.shape, (29,))

        observation = env._get_observation()
        self.assertEqual(observation.shape, (29,))
        self.assertEqual(observation.dtype, np.float64)
        self.assertTrue(env.observation_space.contains(observation))

    def test_action_space_scales_with_module_count(self) -> None:
        self.assertEqual(CrawlerEnv(module_count=1).action_space.shape, (4,))
        self.assertEqual(CrawlerEnv(module_count=3).action_space.shape, (12,))


if __name__ == "__main__":
    unittest.main()
