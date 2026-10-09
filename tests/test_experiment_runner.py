import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nervenet.experiments import ExperimentConfig, ExperimentRunner
from nervenet.experiments.runner import create_experiment_policy


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


class ExperimentRunnerTest(unittest.TestCase):
    def test_target_kl_is_applied_to_ppo(self) -> None:
        ppo = small_ppo_config()
        ppo["target_kl"] = 0.03
        model = create_experiment_policy(
            ExperimentConfig(
                policy_type="matched",
                module_count=1,
                ppo=ppo,
            )
        )

        self.assertEqual(model.target_kl, 0.03)

    def test_experiment_can_checkpoint_evaluate_and_continue(self) -> None:
        config = ExperimentConfig(
            policy_type="matched",
            module_count=1,
            checkpoint_interval=8,
            evaluation_episodes=1,
            ppo=small_ppo_config(),
        )

        with TemporaryDirectory() as directory:
            runner = ExperimentRunner(Path(directory))
            first = runner.start("matched-test", config, total_timesteps=8)
            continued = runner.continue_experiment(
                "matched-test",
                additional_timesteps=8,
            )
            evaluation = runner.evaluate("matched-test")
            evaluated_manifest = json.loads(
                (
                    Path(directory) / "matched-test/experiment.json"
                ).read_text()
            )

            experiment_directory = Path(directory) / "matched-test"
            first_checkpoint = (
                experiment_directory / first["checkpoints"][0]["path"]
            )
            latest_checkpoint = (
                experiment_directory / continued["latest_checkpoint"]
            )
            first_log = experiment_directory / "logs/session_001/progress.csv"
            second_log = experiment_directory / "logs/session_002/progress.csv"

            self.assertTrue(first_checkpoint.exists())
            self.assertTrue(latest_checkpoint.exists())
            self.assertTrue(first_log.exists())
            self.assertTrue(second_log.exists())

        self.assertEqual(first["current_timesteps"], 8)
        self.assertEqual(continued["current_timesteps"], 16)
        self.assertEqual(len(continued["sessions"]), 2)
        self.assertEqual(len(continued["checkpoints"]), 2)
        self.assertIsNotNone(continued["best_checkpoint"])
        self.assertEqual(continued["sessions"][1]["starting_timesteps"], 8)
        self.assertEqual(continued["sessions"][1]["ending_timesteps"], 16)
        self.assertEqual(len(continued["sessions"][0]["training_history"]), 1)
        self.assertEqual(
            continued["sessions"][0]["training_history"][0][
                "time/total_timesteps"
            ],
            8,
        )
        self.assertEqual(evaluation["episode_count"], 1)
        self.assertEqual(len(evaluated_manifest["manual_evaluations"]), 1)
        self.assertEqual(
            evaluated_manifest["manual_evaluations"][0][
                "checkpoint_timesteps"
            ],
            16,
        )
        self.assertIn(
            "mean_forward_distance",
            continued["checkpoints"][-1]["evaluation"]["summary"],
        )
        self.assertIn(
            "mean_absolute_lateral_distance",
            continued["checkpoints"][-1]["evaluation"]["summary"],
        )

    def test_best_checkpoint_can_be_selected(self) -> None:
        config = ExperimentConfig(
            policy_type="matched",
            module_count=1,
            checkpoint_interval=8,
            evaluation_episodes=1,
            evaluation_seed_start=50,
            ppo=small_ppo_config(),
        )

        with TemporaryDirectory() as directory:
            runner = ExperimentRunner(Path(directory))
            manifest = runner.start("best-test", config, total_timesteps=16)
            store_path = Path(directory) / "best-test"
            best_path = store_path / manifest["best_checkpoint"]["path"]
            evaluation = runner.evaluate(
                "best-test",
                best=True,
                episode_count=2,
                seed_start=100,
            )
            self.assertTrue(best_path.exists())

        self.assertEqual(evaluation["seeds"], [100, 101])

    def test_graph_experiment_uses_the_same_lifecycle(self) -> None:
        config = ExperimentConfig(
            policy_type="graph",
            module_count=1,
            checkpoint_interval=8,
            evaluation_episodes=1,
            ppo=small_ppo_config(),
        )

        with TemporaryDirectory() as directory:
            runner = ExperimentRunner(Path(directory))
            manifest = runner.start(
                "graph-test",
                config,
                total_timesteps=16,
            )

        self.assertEqual(manifest["current_timesteps"], 16)
        self.assertEqual(len(manifest["checkpoints"]), 2)
        self.assertEqual(
            [
                checkpoint["timesteps"]
                for checkpoint in manifest["checkpoints"]
            ],
            [8, 16],
        )
        self.assertEqual(
            manifest["checkpoints"][0]["training_metrics"][
                "train/n_updates"
            ],
            1,
        )
        self.assertEqual(manifest["status"], "completed")


if __name__ == "__main__":
    unittest.main()
