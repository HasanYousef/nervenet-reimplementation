import unittest

import numpy as np

from nervenet.envs import CrawlerEnv, GraphObservationWrapper


class GraphObservationWrapperTest(unittest.TestCase):
    def test_observation_shape_scales_with_module_count(self) -> None:
        one_module = GraphObservationWrapper(CrawlerEnv(module_count=1))
        three_modules = GraphObservationWrapper(CrawlerEnv(module_count=3))

        self.assertEqual(one_module.observation_space.shape, (5, 11))
        self.assertEqual(three_modules.observation_space.shape, (15, 11))

    def test_reset_returns_graph_observation(self) -> None:
        env = GraphObservationWrapper(CrawlerEnv(module_count=2))

        observation, info = env.reset(seed=7)

        self.assertEqual(observation.shape, (10, 11))
        self.assertEqual(observation.dtype, np.float64)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info, {})

    def test_step_preserves_graph_observation_shape(self) -> None:
        env = GraphObservationWrapper(CrawlerEnv(module_count=2))
        env.reset(seed=7)

        observation, _, _, _, _ = env.step(
            np.zeros(env.action_space.shape, dtype=np.float64)
        )

        self.assertEqual(observation.shape, (10, 11))
        self.assertTrue(env.observation_space.contains(observation))


if __name__ == "__main__":
    unittest.main()
