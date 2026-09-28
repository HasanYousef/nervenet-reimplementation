from stable_baselines3 import PPO

from nervenet.envs import CrawlerEnv


def train_flat_policy(
    total_timesteps: int,
    seed: int,
) -> PPO:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")

    env = CrawlerEnv(module_count=2)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        device="cpu",
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)
    return model
