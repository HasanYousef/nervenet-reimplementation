from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Literal

import mujoco

from nervenet.policies.defaults import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MESSAGE_PASSING_STEPS,
)
from nervenet.training.matched_policy import MATCHED_ACTOR_HIDDEN_SIZE


ExperimentPolicyType = Literal["matched", "graph"]
EXPERIMENT_SCHEMA_VERSION = 1
EXPERIMENT_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_ppo_config() -> dict[str, Any]:
    return {
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "normalize_advantage": True,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
    }


@dataclass(frozen=True)
class ExperimentConfig:
    policy_type: ExperimentPolicyType
    module_count: int = 3
    seed: int = 0
    control_cost_weight: float = 0.05
    checkpoint_interval: int = 100_000
    evaluation_episodes: int = 20
    graph_hidden_size: int = DEFAULT_HIDDEN_SIZE
    message_passing_steps: int = DEFAULT_MESSAGE_PASSING_STEPS
    matched_actor_hidden_size: int = MATCHED_ACTOR_HIDDEN_SIZE
    ppo: dict[str, Any] = field(default_factory=default_ppo_config)

    def __post_init__(self) -> None:
        if self.policy_type not in ("matched", "graph"):
            raise ValueError("policy_type must be 'matched' or 'graph'")
        if self.module_count <= 0:
            raise ValueError("module_count must be positive")
        if self.control_cost_weight < 0:
            raise ValueError("control_cost_weight must be non-negative")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if self.evaluation_episodes <= 0:
            raise ValueError("evaluation_episodes must be positive")
        if self.graph_hidden_size <= 0:
            raise ValueError("graph_hidden_size must be positive")
        if self.message_passing_steps <= 0:
            raise ValueError("message_passing_steps must be positive")
        if self.matched_actor_hidden_size <= 0:
            raise ValueError("matched_actor_hidden_size must be positive")
        target_kl = self.ppo.get("target_kl")
        if target_kl is not None and target_kl <= 0:
            raise ValueError("target_kl must be positive when provided")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git_output(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip()


def software_metadata() -> dict[str, Any]:
    status = _git_output("status", "--porcelain")

    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "gymnasium": importlib.metadata.version("gymnasium"),
        "stable_baselines3": importlib.metadata.version("stable-baselines3"),
        "torch": importlib.metadata.version("torch"),
    }


class ExperimentStore:
    def __init__(self, root: Path, name: str) -> None:
        if not EXPERIMENT_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                "experiment name must use lowercase letters, numbers, "
                "underscores, or hyphens and be at most 64 characters"
            )

        self.root = root
        self.name = name
        self.directory = root / name
        self.checkpoints_directory = self.directory / "checkpoints"
        self.logs_directory = self.directory / "logs"
        self.manifest_path = self.directory / "experiment.json"

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def create(self, config: ExperimentConfig) -> dict[str, Any]:
        if self.directory.exists():
            raise FileExistsError(
                f"experiment directory already exists: {self.directory}"
            )

        self.checkpoints_directory.mkdir(parents=True)
        self.logs_directory.mkdir()
        now = utc_now()
        manifest = {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "name": self.name,
            "status": "created",
            "created_at": now,
            "updated_at": now,
            "config": config.to_dict(),
            "software": software_metadata(),
            "current_timesteps": 0,
            "latest_checkpoint": None,
            "sessions": [],
            "checkpoints": [],
            "manual_evaluations": [],
        }
        self.write(manifest)
        return manifest

    def load(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"experiment does not exist: {self.name}"
            )

        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError("unsupported experiment schema version")
        return manifest

    def write(self, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = utc_now()
        temporary_path = self.manifest_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary_path.replace(self.manifest_path)

    def config(self) -> ExperimentConfig:
        return ExperimentConfig.from_dict(self.load()["config"])

    def checkpoint_path(self, timesteps: int) -> Path:
        return self.checkpoints_directory / f"step_{timesteps:09d}.zip"

    def resolve_checkpoint(self, timesteps: int | None = None) -> Path:
        manifest = self.load()

        if timesteps is None:
            relative_path = manifest["latest_checkpoint"]
            if relative_path is None:
                raise ValueError("experiment has no checkpoints")
        else:
            matching = [
                checkpoint
                for checkpoint in manifest["checkpoints"]
                if checkpoint["timesteps"] == timesteps
            ]
            if not matching:
                raise ValueError(
                    f"experiment has no checkpoint at step {timesteps}"
                )
            relative_path = matching[-1]["path"]

        checkpoint_path = self.directory / relative_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"checkpoint file is missing: {checkpoint_path}"
            )
        return checkpoint_path
