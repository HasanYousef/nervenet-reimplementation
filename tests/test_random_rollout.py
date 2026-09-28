import unittest

from nervenet.cli.rollout_random import run_random_episode
from nervenet.envs import CrawlerEnv


class RandomRolloutTest(unittest.TestCase):
    def test_seeded_rollout_reaches_time_limit_reproducibly(self) -> None:
        first = run_random_episode(
            CrawlerEnv(module_count=1, max_episode_seconds=0.04),
            seed=7,
        )
        second = run_random_episode(
            CrawlerEnv(module_count=1, max_episode_seconds=0.04),
            seed=7,
        )

        self.assertEqual(first, second)

        steps, _, terminated, truncated, _ = first
        self.assertEqual(steps, 2)
        self.assertFalse(terminated)
        self.assertTrue(truncated)


if __name__ == "__main__":
    unittest.main()
