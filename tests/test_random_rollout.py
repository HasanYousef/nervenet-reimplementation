import unittest

from nervenet.evaluation import run_episode
import numpy as np

from nervenet.envs import CrawlerEnv, GraphObservationWrapper


class RandomRolloutTest(unittest.TestCase):
    def test_seeded_rollout_reaches_time_limit_reproducibly(self) -> None:
        def run_seeded_episode():
            env = CrawlerEnv(module_count=1, max_episode_seconds=0.04)
            env.action_space.seed(7)

            return run_episode(
                env=env,
                action_selector=lambda observation: env.action_space.sample(),
                seed=7,
            )

        first = run_seeded_episode()
        second = run_seeded_episode()

        self.assertEqual(first, second)
        self.assertEqual(first.steps, 2)
        self.assertFalse(first.terminated)
        self.assertTrue(first.truncated)
        self.assertGreaterEqual(first.mean_abs_action, 0.0)
        self.assertLessEqual(first.mean_abs_action, 1.0)
        self.assertGreaterEqual(first.action_saturation_fraction, 0.0)
        self.assertLessEqual(first.action_saturation_fraction, 1.0)
        self.assertGreaterEqual(first.mean_abs_lateral_velocity, 0.0)
        self.assertGreaterEqual(first.final_heading_error_radians, 0.0)
        self.assertLessEqual(first.final_heading_error_radians, np.pi)

    def test_graph_observation_wrapper_can_be_evaluated(self) -> None:
        env = GraphObservationWrapper(
            CrawlerEnv(module_count=1, max_episode_seconds=0.04)
        )

        result = run_episode(
            env=env,
            action_selector=lambda _observation: np.zeros(4),
            seed=0,
        )

        self.assertEqual(result.steps, 2)
        self.assertTrue(result.truncated)


if __name__ == "__main__":
    unittest.main()
