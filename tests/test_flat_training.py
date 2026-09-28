import unittest

from nervenet.training.flat_policy import train_flat_policy


class FlatPolicyTrainingTest(unittest.TestCase):
    def test_total_timesteps_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "total_timesteps must be positive"):
            train_flat_policy(total_timesteps=0, seed=0)


if __name__ == "__main__":
    unittest.main()
