import unittest
from pathlib import Path

from nervenet.envs import CrawlerEnv, GraphObservationWrapper
from nervenet.policy_loading import (
    create_policy_environment,
    default_policy_path,
)


class PolicyLoadingTest(unittest.TestCase):
    def test_default_path_reflects_policy_type_and_module_count(self) -> None:
        self.assertEqual(
            default_policy_path("graph", 3),
            Path("artifacts/graph_policy_3_modules.zip"),
        )

    def test_flat_policy_uses_flat_environment(self) -> None:
        env = create_policy_environment("flat", 1)

        self.assertIsInstance(env, CrawlerEnv)
        self.assertEqual(env.observation_space.shape, (19,))

    def test_graph_policy_uses_graph_environment(self) -> None:
        env = create_policy_environment("graph", 1)

        self.assertIsInstance(env, GraphObservationWrapper)
        self.assertEqual(env.observation_space.shape, (5, 11))

    def test_unknown_policy_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown policy type"):
            create_policy_environment("unknown", 1)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
