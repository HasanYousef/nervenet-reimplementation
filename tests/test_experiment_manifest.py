from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nervenet.experiments import ExperimentConfig, ExperimentStore


class ExperimentManifestTest(unittest.TestCase):
    def test_name_rejects_paths_and_uppercase_characters(self) -> None:
        with self.assertRaisesRegex(ValueError, "experiment name"):
            ExperimentStore(Path("experiments"), "../Other")

    def test_create_persists_immutable_configuration(self) -> None:
        config = ExperimentConfig(
            policy_type="graph",
            module_count=3,
            seed=7,
        )

        with TemporaryDirectory() as directory:
            store = ExperimentStore(Path(directory), "graph-seed7")
            store.create(config)
            loaded = store.load()

        self.assertEqual(loaded["name"], "graph-seed7")
        self.assertEqual(loaded["config"], config.to_dict())
        self.assertEqual(loaded["current_timesteps"], 0)
        self.assertEqual(loaded["sessions"], [])
        self.assertEqual(loaded["checkpoints"], [])
        self.assertEqual(loaded["manual_evaluations"], [])
        self.assertIsNone(loaded["best_checkpoint"])

    def test_existing_experiment_cannot_be_recreated(self) -> None:
        with TemporaryDirectory() as directory:
            store = ExperimentStore(Path(directory), "matched-seed0")
            store.create(ExperimentConfig(policy_type="matched"))

            with self.assertRaises(FileExistsError):
                store.create(ExperimentConfig(policy_type="matched"))

    def test_target_kl_must_be_positive_when_enabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_kl must be positive"):
            ExperimentConfig(
                policy_type="graph",
                ppo={"target_kl": 0.0},
            )

    def test_evaluation_seed_start_must_be_non_negative(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "evaluation_seed_start must be non-negative",
        ):
            ExperimentConfig(
                policy_type="graph",
                evaluation_seed_start=-1,
            )


if __name__ == "__main__":
    unittest.main()
