from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import shutil
from typing import Any

from nervenet.experiments.manifest import (
    EXPERIMENT_NAME_PATTERN,
    ExperimentPolicyType,
    default_ppo_config,
    software_metadata,
    utc_now,
)
from nervenet.policies.defaults import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MESSAGE_PASSING_STEPS,
)
from nervenet.training.matched_policy import MATCHED_ACTOR_HIDDEN_SIZE


COMPARISON_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ComparisonConfig:
    policy_types: tuple[ExperimentPolicyType, ...] = ("matched", "graph")
    training_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    total_timesteps: int = 1_000_000
    module_count: int = 3
    control_cost_weight: float = 0.05
    checkpoint_interval: int = 50_000
    validation_episode_count: int = 10
    validation_seed_start: int = 10_000
    test_episode_count: int = 50
    test_seed_start: int = 20_000
    graph_hidden_size: int = DEFAULT_HIDDEN_SIZE
    message_passing_steps: int = DEFAULT_MESSAGE_PASSING_STEPS
    matched_actor_hidden_size: int = MATCHED_ACTOR_HIDDEN_SIZE
    ppo: dict[str, Any] = field(default_factory=default_ppo_config)

    def __post_init__(self) -> None:
        if not self.policy_types:
            raise ValueError("policy_types must not be empty")
        if len(set(self.policy_types)) != len(self.policy_types):
            raise ValueError("policy_types must be unique")
        if any(policy not in ("matched", "graph") for policy in self.policy_types):
            raise ValueError("policy_types may only contain matched and graph")
        if not self.training_seeds:
            raise ValueError("training_seeds must not be empty")
        if len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("training_seeds must be unique")
        if any(seed < 0 for seed in self.training_seeds):
            raise ValueError("training_seeds must be non-negative")
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.module_count <= 0:
            raise ValueError("module_count must be positive")
        if self.control_cost_weight < 0:
            raise ValueError("control_cost_weight must be non-negative")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if self.validation_episode_count <= 0:
            raise ValueError("validation_episode_count must be positive")
        if self.test_episode_count <= 0:
            raise ValueError("test_episode_count must be positive")
        if self.validation_seed_start < 0 or self.test_seed_start < 0:
            raise ValueError("evaluation seed starts must be non-negative")
        validation_seeds = range(
            self.validation_seed_start,
            self.validation_seed_start + self.validation_episode_count,
        )
        test_seeds = range(
            self.test_seed_start,
            self.test_seed_start + self.test_episode_count,
        )
        if set(validation_seeds).intersection(test_seeds):
            raise ValueError("validation and test seeds must not overlap")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ComparisonConfig":
        normalized = dict(values)
        normalized["policy_types"] = tuple(normalized["policy_types"])
        normalized["training_seeds"] = tuple(normalized["training_seeds"])
        return cls(**normalized)


class ComparisonStore:
    def __init__(self, root: Path, name: str) -> None:
        if not EXPERIMENT_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                "comparison name must use lowercase letters, numbers, "
                "underscores, or hyphens and be at most 64 characters"
            )
        self.root = root
        self.name = name
        self.directory = root / name
        self.runs_directory = self.directory / "runs"
        self.figures_directory = self.directory / "figures"
        self.reports_directory = self.directory / "reports"
        self.manifest_path = self.directory / "comparison.json"

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def create(self, config: ComparisonConfig) -> dict[str, Any]:
        if self.directory.exists():
            raise FileExistsError(
                f"comparison directory already exists: {self.directory}"
            )
        self.runs_directory.mkdir(parents=True)
        self.figures_directory.mkdir()
        self.reports_directory.mkdir()
        now = utc_now()
        manifest = {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "name": self.name,
            "status": "created",
            "created_at": now,
            "updated_at": now,
            "config": config.to_dict(),
            "current_target_timesteps": config.total_timesteps,
            "current_test_seed_start": config.test_seed_start,
            "budget_history": [
                {
                    "created_at": now,
                    "target_timesteps": config.total_timesteps,
                    "test_seed_start": config.test_seed_start,
                    "reason": "initial comparison",
                }
            ],
            "software": software_metadata(),
            "latest_report": None,
        }
        self.write(manifest)
        return manifest

    def load(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"comparison does not exist: {self.name}")
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("schema_version") != COMPARISON_SCHEMA_VERSION:
            raise ValueError("unsupported comparison schema version")
        return manifest

    def write(self, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = utc_now()
        temporary_path = self.manifest_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary_path.replace(self.manifest_path)

    def config(self) -> ComparisonConfig:
        return ComparisonConfig.from_dict(self.load()["config"])

    def target_timesteps(self) -> int:
        manifest = self.load()
        return manifest.get(
            "current_target_timesteps",
            self.config().total_timesteps,
        )

    def test_seed_start(self) -> int:
        manifest = self.load()
        return manifest.get(
            "current_test_seed_start",
            self.config().test_seed_start,
        )

    def extend_budget(self, additional_timesteps: int) -> dict[str, Any]:
        if additional_timesteps <= 0:
            raise ValueError("additional_timesteps must be positive")

        manifest = self.load()
        config = ComparisonConfig.from_dict(manifest["config"])
        previous_target = manifest.get(
            "current_target_timesteps",
            config.total_timesteps,
        )
        if manifest.get("status") != "completed":
            raise ValueError(
                "comparison must be completed before extending its budget"
            )
        if manifest.get("latest_report") is None:
            raise ValueError(
                "generate the current comparison report before extending"
            )

        self.archive_current_report(previous_target)
        previous_test_seed_start = manifest.get(
            "current_test_seed_start",
            config.test_seed_start,
        )
        new_test_seed_start = previous_test_seed_start + 10_000
        new_target = previous_target + additional_timesteps
        now = utc_now()
        manifest["current_target_timesteps"] = new_target
        manifest["current_test_seed_start"] = new_test_seed_start
        manifest.setdefault("budget_history", []).append(
            {
                "created_at": now,
                "previous_target_timesteps": previous_target,
                "additional_timesteps": additional_timesteps,
                "target_timesteps": new_target,
                "test_seed_start": new_test_seed_start,
                "reason": "post-report training extension",
            }
        )
        manifest["status"] = "extended"
        manifest["latest_report"] = None
        self.write(manifest)
        return manifest

    def report_archive_directory(self, target_timesteps: int) -> Path:
        return self.reports_directory / f"target_{target_timesteps:09d}"

    def archive_current_report(self, target_timesteps: int) -> Path:
        archive = self.report_archive_directory(target_timesteps)
        if archive.exists():
            return archive
        archive.mkdir(parents=True)
        for filename in (
            "report.md",
            "results.csv",
            "learning_curves.csv",
            "summary.json",
        ):
            source = self.directory / filename
            if not source.exists():
                raise FileNotFoundError(
                    f"comparison report file is missing: {source}"
                )
            shutil.copy2(source, archive / filename)
        shutil.copytree(self.figures_directory, archive / "figures")
        return archive

    def run_name(self, policy_type: str, seed: int) -> str:
        return f"{policy_type}-seed{seed}"
