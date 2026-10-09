from dataclasses import asdict
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.policies import BasePolicy

from nervenet.evaluation import run_episode, summarize_episode_results
from nervenet.graphs import get_actuator_node_indices, get_message_routes
from nervenet.policies import (
    GraphActorCriticPolicy,
    MatchedActorCriticPolicy,
)
from nervenet.policy_loading import create_policy_environment
from nervenet.training.checkpoint import load_ppo_checkpoint
from nervenet.experiments.manifest import (
    ExperimentConfig,
    ExperimentStore,
    software_metadata,
    utc_now,
)


def _expected_policy_type(config: ExperimentConfig) -> type[BasePolicy]:
    if config.policy_type == "graph":
        return GraphActorCriticPolicy
    return MatchedActorCriticPolicy


def create_experiment_policy(config: ExperimentConfig) -> PPO:
    env = create_policy_environment(
        config.policy_type,
        config.module_count,
        config.control_cost_weight,
    )

    if config.policy_type == "graph":
        policy = GraphActorCriticPolicy
        policy_kwargs = {
            "routes": get_message_routes(env.graph),
            "actuator_node_indices": get_actuator_node_indices(env.graph),
            "node_types": tuple(node.node_type for node in env.graph.nodes),
            "hidden_size": config.graph_hidden_size,
            "message_passing_steps": config.message_passing_steps,
        }
    else:
        policy = MatchedActorCriticPolicy
        policy_kwargs = {
            "actor_hidden_size": config.matched_actor_hidden_size,
        }

    return PPO(
        policy=policy,
        env=env,
        policy_kwargs=policy_kwargs,
        seed=config.seed,
        device="cpu",
        verbose=1,
        **config.ppo,
    )


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "numel") and value.numel() == 1:
        return value.item()
    return str(value)


def evaluate_model(
    model: PPO,
    config: ExperimentConfig,
) -> dict[str, Any]:
    env = create_policy_environment(
        config.policy_type,
        config.module_count,
        config.control_cost_weight,
    )
    results = [
        run_episode(
            env=env,
            action_selector=lambda observation: model.predict(
                observation,
                deterministic=True,
            )[0],
            seed=seed,
        )
        for seed in range(config.evaluation_episodes)
    ]
    env.close()

    return {
        "episode_count": len(results),
        "seeds": [result.seed for result in results],
        "summary": asdict(summarize_episode_results(results)),
        "episodes": [asdict(result) for result in results],
    }


class ExperimentCheckpointCallback(BaseCallback):
    def __init__(
        self,
        store: ExperimentStore,
        config: ExperimentConfig,
        session_index: int,
        starting_timesteps: int,
    ) -> None:
        super().__init__(verbose=0)
        self.store = store
        self.config = config
        self.session_index = session_index
        self.starting_timesteps = starting_timesteps
        self.next_checkpoint_at = (
            starting_timesteps // config.checkpoint_interval + 1
        ) * config.checkpoint_interval
        self.saved_timesteps: set[int] = set()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        if (
            self.model.num_timesteps > self.starting_timesteps
            and self.model.num_timesteps >= self.next_checkpoint_at
        ):
            self._save_checkpoint()
            while self.next_checkpoint_at <= self.model.num_timesteps:
                self.next_checkpoint_at += self.config.checkpoint_interval

    def _on_training_end(self) -> None:
        self._save_checkpoint()

    def _training_metrics(self) -> dict[str, Any]:
        metrics = {
            key: _json_scalar(value)
            for key, value in self.model.logger.name_to_value.items()
            if key.startswith("train/")
        }

        if self.model.ep_info_buffer:
            metrics["rollout/ep_rew_mean"] = fmean(
                episode["r"] for episode in self.model.ep_info_buffer
            )
            metrics["rollout/ep_len_mean"] = fmean(
                episode["l"] for episode in self.model.ep_info_buffer
            )

        return metrics

    def _save_checkpoint(self) -> None:
        timesteps = self.model.num_timesteps
        if timesteps in self.saved_timesteps:
            return

        checkpoint_path = self.store.checkpoint_path(timesteps)
        self.model.save(checkpoint_path)
        evaluation = evaluate_model(self.model, self.config)
        manifest = self.store.load()
        relative_path = checkpoint_path.relative_to(self.store.directory)
        manifest["checkpoints"].append(
            {
                "timesteps": timesteps,
                "path": str(relative_path),
                "created_at": utc_now(),
                "session_index": self.session_index,
                "training_metrics": self._training_metrics(),
                "evaluation": evaluation,
            }
        )
        manifest["current_timesteps"] = timesteps
        manifest["latest_checkpoint"] = str(relative_path)
        self.store.write(manifest)
        self.saved_timesteps.add(timesteps)
        summary = evaluation["summary"]
        print(f"Saved experiment checkpoint: {checkpoint_path}")
        print(
            "  Evaluation: "
            f"distance={summary['mean_forward_distance']:.3f} m, "
            "absolute_lateral_distance="
            f"{summary['mean_absolute_lateral_distance']:.3f} m, "
            f"reward={summary['mean_total_reward']:.3f}"
        )


class ExperimentRunner:
    def __init__(self, experiments_root: Path = Path("experiments")) -> None:
        self.experiments_root = experiments_root

    def start(
        self,
        name: str,
        config: ExperimentConfig,
        total_timesteps: int,
    ) -> dict[str, Any]:
        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")

        store = ExperimentStore(self.experiments_root, name)
        store.create(config)
        model = create_experiment_policy(config)
        return self._train(
            store=store,
            config=config,
            model=model,
            additional_timesteps=total_timesteps,
            reset_num_timesteps=True,
        )

    def continue_experiment(
        self,
        name: str,
        additional_timesteps: int,
    ) -> dict[str, Any]:
        if additional_timesteps <= 0:
            raise ValueError("additional_timesteps must be positive")

        store = ExperimentStore(self.experiments_root, name)
        manifest = store.load()
        config = ExperimentConfig.from_dict(manifest["config"])
        env = create_policy_environment(
            config.policy_type,
            config.module_count,
            config.control_cost_weight,
        )
        model = load_ppo_checkpoint(
            checkpoint=store.resolve_checkpoint(),
            env=env,
            expected_policy_type=_expected_policy_type(config),
        )

        if model.num_timesteps != manifest["current_timesteps"]:
            raise ValueError(
                "checkpoint timestep count does not match experiment metadata"
            )

        return self._train(
            store=store,
            config=config,
            model=model,
            additional_timesteps=additional_timesteps,
            reset_num_timesteps=False,
        )

    def evaluate(
        self,
        name: str,
        timesteps: int | None = None,
    ) -> dict[str, Any]:
        store = ExperimentStore(self.experiments_root, name)
        config = store.config()
        checkpoint_path = store.resolve_checkpoint(timesteps)
        model = PPO.load(checkpoint_path, device="cpu", verbose=0)

        if not isinstance(model.policy, _expected_policy_type(config)):
            raise ValueError("checkpoint policy type does not match experiment")

        evaluation = evaluate_model(model, config)
        manifest = store.load()
        checkpoint = next(
            checkpoint
            for checkpoint in manifest["checkpoints"]
            if (store.directory / checkpoint["path"]) == checkpoint_path
        )
        manifest["manual_evaluations"].append(
            {
                "created_at": utc_now(),
                "checkpoint_timesteps": checkpoint["timesteps"],
                "checkpoint_path": checkpoint["path"],
                "software": software_metadata(),
                "evaluation": evaluation,
            }
        )
        store.write(manifest)
        return evaluation

    def _train(
        self,
        store: ExperimentStore,
        config: ExperimentConfig,
        model: PPO,
        additional_timesteps: int,
        reset_num_timesteps: bool,
    ) -> dict[str, Any]:
        manifest = store.load()
        session_index = len(manifest["sessions"]) + 1
        session_log_directory = (
            store.logs_directory / f"session_{session_index:03d}"
        )
        session_log_directory.mkdir()
        session = {
            "index": session_index,
            "status": "running",
            "started_at": utc_now(),
            "ended_at": None,
            "starting_timesteps": model.num_timesteps,
            "ending_timesteps": None,
            "requested_additional_timesteps": additional_timesteps,
            "log_directory": str(
                session_log_directory.relative_to(store.directory)
            ),
            "software": software_metadata(),
            "training_history": [],
            "error": None,
        }
        manifest["sessions"].append(session)
        manifest["status"] = "running"
        store.write(manifest)

        model.set_logger(
            configure(
                folder=str(session_log_directory),
                format_strings=["stdout", "csv", "json"],
            )
        )
        callback = ExperimentCheckpointCallback(
            store=store,
            config=config,
            session_index=session_index,
            starting_timesteps=model.num_timesteps,
        )

        try:
            model.learn(
                total_timesteps=additional_timesteps,
                reset_num_timesteps=reset_num_timesteps,
                callback=callback,
            )
        except KeyboardInterrupt:
            self._finish_session(
                store,
                session_index,
                model.num_timesteps,
                "interrupted",
            )
            raise
        except Exception as error:
            self._finish_session(
                store,
                session_index,
                model.num_timesteps,
                "failed",
                repr(error),
            )
            raise
        else:
            self._finish_session(
                store,
                session_index,
                model.num_timesteps,
                "completed",
            )
        finally:
            model.logger.close()

        return store.load()

    def _finish_session(
        self,
        store: ExperimentStore,
        session_index: int,
        ending_timesteps: int,
        status: str,
        error: str | None = None,
    ) -> None:
        manifest = store.load()
        session = manifest["sessions"][session_index - 1]
        session["status"] = status
        session["ended_at"] = utc_now()
        session["ending_timesteps"] = ending_timesteps
        session["error"] = error
        progress_path = (
            store.directory / session["log_directory"] / "progress.json"
        )
        if progress_path.exists():
            session["training_history"] = [
                json.loads(line)
                for line in progress_path.read_text().splitlines()
                if line
            ]
        manifest["status"] = status
        store.write(manifest)
