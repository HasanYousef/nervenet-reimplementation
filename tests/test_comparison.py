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
            first_archive = store.report_archive_directory(8)
            self.assertTrue((first_archive / "report.md").exists())
            self.assertTrue((first_archive / "figures" / "reward.png").exists())

            runner.extend("small-comparison", 8)
            extended_status = runner.status("small-comparison")
            extended_manifest = store.load()
            runner.report("small-comparison")
            extended_summary = json.loads(
                (store.directory / "summary.json").read_text()
            )
            extended_archive_exists = (
                store.report_archive_directory(16) / "summary.json"
            ).exists()
            export_path = runner.export_results(
                "small-comparison",
                root / "published-results",
            )
            export_readme = (export_path / "README.md").read_text()
            export_manifest_exists = (
                export_path / "comparison.json"
            ).exists()
            training_sessions_exists = (
                export_path / "training_sessions.csv"
            ).exists()
            first_export_exists = (
                export_path
                / "reports"
                / "target_000000008"
                / "summary.json"
            ).exists()
            extended_export_exists = (
                export_path
                / "reports"
                / "target_000000016"
                / "figures"
                / "distance.png"
            ).exists()
            export_contains_model = any(
                path.suffix == ".zip"
                for path in export_path.rglob("*")
                if path.is_file()
            )

        self.assertEqual(len(status), 2)
        self.assertTrue(all(row["timesteps"] == 8 for row in status))
        self.assertTrue(all(row["target_timesteps"] == 8 for row in status))
        self.assertEqual(len(summary["runs"]), 2)
        self.assertEqual(summary["config"]["test_seed_start"], 200)
        self.assertEqual(summary["target_timesteps"], 8)
        self.assertEqual(summary["test_seed_start"], 200)
        self.assertTrue(
            all(row["timesteps"] >= 16 for row in extended_status)
        )
        self.assertTrue(
            all(row["target_timesteps"] == 16 for row in extended_status)
        )
        self.assertEqual(extended_manifest["current_target_timesteps"], 16)
        self.assertEqual(extended_manifest["current_test_seed_start"], 10_200)
        self.assertEqual(len(extended_manifest["budget_history"]), 2)
        self.assertEqual(extended_summary["target_timesteps"], 16)
        self.assertEqual(extended_summary["test_seed_start"], 10_200)
        self.assertTrue(extended_archive_exists)
        self.assertTrue(export_manifest_exists)
        self.assertTrue(training_sessions_exists)
        self.assertTrue(first_export_exists)
        self.assertTrue(extended_export_exists)
        self.assertFalse(export_contains_model)
        self.assertIn("8 | 200–200 | Matched", export_readme)
        self.assertIn("16 | 10,200–10,200 | Matched", export_readme)
        self.assertIn("Model checkpoints", export_readme)
        self.assertIn("Recorded training time", export_readme)

    def test_extension_requires_completed_report(self) -> None:
        config = ComparisonConfig(
            policy_types=("matched",),
            training_seeds=(0,),
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
            runner = ComparisonRunner(Path(directory))
            runner.create("unfinished-comparison", config)
            with self.assertRaisesRegex(ValueError, "must be completed"):
                runner.extend("unfinished-comparison", 8)

            runner.run("unfinished-comparison")
            with self.assertRaisesRegex(ValueError, "generate.*report"):
                runner.extend("unfinished-comparison", 8)

    def test_extension_budget_must_be_positive(self) -> None:
        config = ComparisonConfig(
            policy_types=("matched",),
            training_seeds=(0,),
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
            runner = ComparisonRunner(Path(directory))
            runner.create("invalid-extension", config)
            with self.assertRaisesRegex(ValueError, "must be positive"):
                runner.extend("invalid-extension", 0)


if __name__ == "__main__":
    unittest.main()
