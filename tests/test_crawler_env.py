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

    def test_reset_restores_initial_state(self) -> None:
        env = CrawlerEnv(module_count=2)
        env.data.time = 1.0
        env.data.qvel[:] = 1.0

        observation, info = env.reset(seed=7)

        self.assertEqual(env.data.time, 0.0)
        self.assertTrue(np.all(env.data.qvel == 0.0))
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info, {})

    def test_step_advances_one_control_timestep(self) -> None:
        env = CrawlerEnv(module_count=2)
        env.reset()

        observation, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float64)
        )

        self.assertAlmostEqual(env.data.time, env.control_timestep)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info, {})

    def test_step_rejects_action_with_wrong_shape(self) -> None:
        env = CrawlerEnv(module_count=2)
        env.reset()

        with self.assertRaisesRegex(ValueError, "Invalid action"):
            env.step(np.zeros(7, dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
