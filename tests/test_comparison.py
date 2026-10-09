import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nervenet.comparisons import (
    ComparisonConfig,
    ComparisonRunner,
    ComparisonStore,
)


def small_ppo_config() -> dict:
    return {
        "learning_rate": 0.0003,
        "n_steps": 8,
        "batch_size": 8,
        "n_epochs": 1,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "normalize_advantage": True,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
    }


class ComparisonTest(unittest.TestCase):
    def test_validation_and_test_seeds_must_not_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            ComparisonConfig(
                validation_episode_count=10,
                validation_seed_start=100,
                test_episode_count=10,
                test_seed_start=105,
            )

    def test_small_comparison_runs_and_writes_report(self) -> None:
        config = ComparisonConfig(
            policy_types=("matched",),
            training_seeds=(0, 1),
            total_timesteps=8,
            module_count=1,
            checkpoint_interval=8,
            validation_episode_count=1,
            validation_seed_start=100,
            test_episode_count=1,
            test_seed_start=200,
            ppo=small_ppo_config(),
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            runner = ComparisonRunner(root)
            runner.create("small-comparison", config)
            runner.run("small-comparison")
            status = runner.status("small-comparison")
            report_path = runner.report("small-comparison")
            store = ComparisonStore(root, "small-comparison")
            summary = json.loads(
                (store.directory / "summary.json").read_text()
            )

            self.assertTrue(report_path.exists())
            self.assertTrue((store.directory / "results.csv").exists())
            self.assertTrue(
                (store.directory / "learning_curves.csv").exists()
            )
            self.assertTrue(
                (store.figures_directory / "reward.png").exists()
            )
            self.assertTrue(
                (store.figures_directory / "distance.png").exists()
            )

        self.assertEqual(len(status), 2)
        self.assertTrue(all(row["timesteps"] == 8 for row in status))
        self.assertEqual(len(summary["runs"]), 2)
        self.assertEqual(summary["config"]["test_seed_start"], 200)


if __name__ == "__main__":
    unittest.main()
