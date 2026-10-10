from csv import DictWriter
import json
import os
from pathlib import Path
from statistics import fmean, stdev
import tempfile
from typing import Any

_cache_root = Path(tempfile.gettempdir()) / "nervenet-plot-cache"
_cache_root.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from stable_baselines3 import PPO

from nervenet.comparisons.manifest import ComparisonConfig, ComparisonStore
from nervenet.experiments import ExperimentConfig, ExperimentRunner, ExperimentStore
from nervenet.experiments.manifest import software_metadata, utc_now
from nervenet.experiments.runner import evaluate_model


RESULT_METRICS = (
    "mean_total_reward",
    "mean_forward_distance",
    "mean_absolute_lateral_distance",
    "mean_absolute_lateral_velocity",
    "mean_final_heading_error_radians",
    "mean_absolute_action",
    "action_saturation_fraction",
    "mean_absolute_action_change",
    "mean_actuated_joint_speed",
    "unstable_episodes",
)


def _mean_and_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
    }


def _experiment_config(
    comparison: ComparisonConfig,
    policy_type: str,
    seed: int,
) -> ExperimentConfig:
    return ExperimentConfig(
        policy_type=policy_type,
        module_count=comparison.module_count,
        seed=seed,
        control_cost_weight=comparison.control_cost_weight,
        checkpoint_interval=comparison.checkpoint_interval,
        evaluation_episodes=comparison.validation_episode_count,
        evaluation_seed_start=comparison.validation_seed_start,
        graph_hidden_size=comparison.graph_hidden_size,
        message_passing_steps=comparison.message_passing_steps,
        matched_actor_hidden_size=comparison.matched_actor_hidden_size,
        ppo=comparison.ppo,
    )


class ComparisonRunner:
    def __init__(self, comparisons_root: Path = Path("experiments")) -> None:
        self.comparisons_root = comparisons_root

    def create(self, name: str, config: ComparisonConfig) -> dict[str, Any]:
        return ComparisonStore(self.comparisons_root, name).create(config)

    def run(
        self,
        name: str,
        policy_type: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        store = ComparisonStore(self.comparisons_root, name)
        config = store.config()
        target_timesteps = store.target_timesteps()
        policy_types = (
            (policy_type,) if policy_type is not None else config.policy_types
        )
        seeds = (seed,) if seed is not None else config.training_seeds
        if any(policy not in config.policy_types for policy in policy_types):
            raise ValueError("selected policy is not part of this comparison")
        if any(value not in config.training_seeds for value in seeds):
            raise ValueError("selected seed is not part of this comparison")

        manifest = store.load()
        manifest["status"] = "running"
        store.write(manifest)
        experiment_runner = ExperimentRunner(store.runs_directory)

        try:
            for current_policy in policy_types:
                for current_seed in seeds:
                    run_name = store.run_name(current_policy, current_seed)
                    run_store = ExperimentStore(store.runs_directory, run_name)
                    expected_config = _experiment_config(
                        config,
                        current_policy,
                        current_seed,
                    )
                    if not run_store.exists():
                        experiment_runner.start(
                            run_name,
                            expected_config,
                            config.total_timesteps,
                        )
                        continue

                    run_manifest = run_store.load()
                    if run_store.config() != expected_config:
                        raise ValueError(
                            f"run configuration does not match: {run_name}"
                        )
                    remaining = (
                        target_timesteps
                        - run_manifest["current_timesteps"]
                    )
                    if remaining > 0:
                        experiment_runner.continue_experiment(
                            run_name,
                            remaining,
                        )
        except BaseException:
            manifest = store.load()
            manifest["status"] = "interrupted"
            store.write(manifest)
            raise

        manifest = store.load()
        manifest["status"] = (
            "completed"
            if self._all_runs_complete(store, config, target_timesteps)
            else "partial"
        )
        store.write(manifest)
        return manifest

    def extend(self, name: str, additional_timesteps: int) -> dict[str, Any]:
        store = ComparisonStore(self.comparisons_root, name)
        store.extend_budget(additional_timesteps)
        return self.run(name)

    def status(self, name: str) -> list[dict[str, Any]]:
        store = ComparisonStore(self.comparisons_root, name)
        config = store.config()
        target_timesteps = store.target_timesteps()
        rows = []
        for policy_type in config.policy_types:
            for seed in config.training_seeds:
                run_name = store.run_name(policy_type, seed)
                run_store = ExperimentStore(store.runs_directory, run_name)
                if not run_store.exists():
                    rows.append(
                        {
                            "policy_type": policy_type,
                            "seed": seed,
                            "status": "not started",
                            "timesteps": 0,
                            "target_timesteps": target_timesteps,
                        }
                    )
                    continue
                manifest = run_store.load()
                rows.append(
                    {
                        "policy_type": policy_type,
                        "seed": seed,
                        "status": manifest["status"],
                        "timesteps": manifest["current_timesteps"],
                        "target_timesteps": target_timesteps,
                    }
                )
        return rows

    def report(self, name: str) -> Path:
        store = ComparisonStore(self.comparisons_root, name)
        config = store.config()
        target_timesteps = store.target_timesteps()
        test_seed_start = store.test_seed_start()
        if not self._all_runs_complete(store, config, target_timesteps):
            raise ValueError("all comparison runs must finish before reporting")

        run_results = []
        detailed_results = []
        checkpoint_histories: dict[str, list[list[dict[str, Any]]]] = {
            policy: [] for policy in config.policy_types
        }
        for policy_type in config.policy_types:
            for seed in config.training_seeds:
                run_store = ExperimentStore(
                    store.runs_directory,
                    store.run_name(policy_type, seed),
                )
                run_manifest = run_store.load()
                best = run_store.best_checkpoint()
                model = PPO.load(
                    run_store.directory / best["path"],
                    device="cpu",
                    verbose=0,
                )
                evaluation = evaluate_model(
                    model,
                    run_store.config(),
                    episode_count=config.test_episode_count,
                    seed_start=test_seed_start,
                )
                run_results.append(
                    {
                        "policy_type": policy_type,
                        "training_seed": seed,
                        "selected_checkpoint_timesteps": best["timesteps"],
                        "validation_mean_total_reward": best["evaluation"][
                            "summary"
                        ]["mean_total_reward"],
                        **evaluation["summary"],
                    }
                )
                detailed_results.append(
                    {
                        "policy_type": policy_type,
                        "training_seed": seed,
                        "selected_checkpoint_timesteps": best["timesteps"],
                        "validation": best["evaluation"],
                        "test": evaluation,
                    }
                )
                checkpoint_histories[policy_type].append(
                    run_manifest["checkpoints"]
                )

        curves = self._aggregate_learning_curves(checkpoint_histories)
        aggregate = self._aggregate_results(config, run_results)
        self._write_results_csv(store, run_results)
        self._write_curves_csv(store, curves)
        self._write_figures(store, curves)
        summary = {
            "comparison": name,
            "generated_at": utc_now(),
            "software": software_metadata(),
            "config": config.to_dict(),
            "target_timesteps": target_timesteps,
            "test_seed_start": test_seed_start,
            "aggregate": aggregate,
            "runs": detailed_results,
        }
        (store.directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        report_path = store.directory / "report.md"
        report_path.write_text(
            self._render_report(
                name,
                config,
                target_timesteps,
                test_seed_start,
                aggregate,
                run_results,
            )
        )

        archive = store.archive_current_report(target_timesteps)

        manifest = store.load()
        manifest["latest_report"] = {
            "generated_at": summary["generated_at"],
            "path": "report.md",
            "summary_path": "summary.json",
            "results_path": "results.csv",
            "target_timesteps": target_timesteps,
            "test_seed_start": test_seed_start,
            "archive_path": str(archive.relative_to(store.directory)),
        }
        store.write(manifest)
        return report_path

    def _all_runs_complete(
        self,
        store: ComparisonStore,
        config: ComparisonConfig,
        target_timesteps: int,
    ) -> bool:
        for policy_type in config.policy_types:
            for seed in config.training_seeds:
                run_store = ExperimentStore(
                    store.runs_directory,
                    store.run_name(policy_type, seed),
                )
                if not run_store.exists():
                    return False
                if run_store.load()["current_timesteps"] < target_timesteps:
                    return False
        return True

    def _aggregate_results(
        self,
        config: ComparisonConfig,
        run_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        aggregate = {}
        for policy_type in config.policy_types:
            selected = [
                result
                for result in run_results
                if result["policy_type"] == policy_type
            ]
            aggregate[policy_type] = {
                metric: _mean_and_std(
                    [float(result[metric]) for result in selected]
                )
                for metric in RESULT_METRICS
            }
        return aggregate

    def _aggregate_learning_curves(
        self,
        histories: dict[str, list[list[dict[str, Any]]]],
    ) -> list[dict[str, Any]]:
        rows = []
        for policy_type, policy_histories in histories.items():
            by_step: dict[int, dict[str, list[float]]] = {}
            for checkpoints in policy_histories:
                for checkpoint in checkpoints:
                    summary = checkpoint["evaluation"]["summary"]
                    step = checkpoint["timesteps"]
                    metrics = by_step.setdefault(
                        step,
                        {"reward": [], "distance": []},
                    )
                    metrics["reward"].append(summary["mean_total_reward"])
                    metrics["distance"].append(summary["mean_forward_distance"])
            for step, metrics in sorted(by_step.items()):
                expected_count = len(policy_histories)
                if len(metrics["reward"]) != expected_count:
                    continue
                reward = _mean_and_std(metrics["reward"])
                distance = _mean_and_std(metrics["distance"])
                rows.append(
                    {
                        "policy_type": policy_type,
                        "timesteps": step,
                        "reward_mean": reward["mean"],
                        "reward_standard_deviation": reward[
                            "standard_deviation"
                        ],
                        "distance_mean": distance["mean"],
                        "distance_standard_deviation": distance[
                            "standard_deviation"
                        ],
                    }
                )
        return rows

    def _write_results_csv(
        self,
        store: ComparisonStore,
        results: list[dict[str, Any]],
    ) -> None:
        with (store.directory / "results.csv").open("w", newline="") as file:
            writer = DictWriter(file, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)

    def _write_curves_csv(
        self,
        store: ComparisonStore,
        curves: list[dict[str, Any]],
    ) -> None:
        with (store.directory / "learning_curves.csv").open(
            "w", newline=""
        ) as file:
            writer = DictWriter(file, fieldnames=list(curves[0]))
            writer.writeheader()
            writer.writerows(curves)

    def _write_figures(
        self,
        store: ComparisonStore,
        curves: list[dict[str, Any]],
    ) -> None:
        colors = {"matched": "#e76f51", "graph": "#277da1"}
        for metric, title, y_label, filename in (
            ("reward", "Validation reward", "Mean episode reward", "reward.png"),
            (
                "distance",
                "Forward distance",
                "Mean forward distance (m)",
                "distance.png",
            ),
        ):
            figure, axis = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
            for policy_type in ("matched", "graph"):
                selected = [
                    row for row in curves if row["policy_type"] == policy_type
                ]
                if not selected:
                    continue
                x = [row["timesteps"] / 1_000_000 for row in selected]
                mean = [row[f"{metric}_mean"] for row in selected]
                deviation = [
                    row[f"{metric}_standard_deviation"] for row in selected
                ]
                color = colors[policy_type]
                axis.plot(x, mean, label=policy_type.title(), color=color, linewidth=2)
                axis.fill_between(
                    x,
                    [value - spread for value, spread in zip(mean, deviation)],
                    [value + spread for value, spread in zip(mean, deviation)],
                    color=color,
                    alpha=0.18,
                )
            axis.set_title(title)
            axis.set_xlabel("Training timesteps (millions)")
            axis.set_ylabel(y_label)
            axis.grid(alpha=0.25)
            axis.legend(frameon=False)
            figure.savefig(store.figures_directory / filename, dpi=180)
            plt.close(figure)

    def _render_report(
        self,
        name: str,
        config: ComparisonConfig,
        target_timesteps: int,
        test_seed_start: int,
        aggregate: dict[str, Any],
        run_results: list[dict[str, Any]],
    ) -> str:
        def formatted(policy: str, metric: str, digits: int = 2) -> str:
            values = aggregate[policy][metric]
            return (
                f"{values['mean']:.{digits}f} ± "
                f"{values['standard_deviation']:.{digits}f}"
            )

        lines = [
            f"# {name}",
            "",
            "## Experiment",
            "",
            (
                f"Matched MLP and graph PPO policies were trained for "
                f"{target_timesteps:,} timesteps on a "
                f"{config.module_count}-module crawler. Results use "
                f"{len(config.training_seeds)} independent training seeds."
            ),
            "",
            (
                f"Checkpoints were validated every "
                f"{config.checkpoint_interval:,} timesteps on "
                f"{config.validation_episode_count} fixed episodes. The best "
                f"checkpoint per run was selected by validation reward, then "
                f"evaluated on {config.test_episode_count} held-out episodes "
                f"starting at seed {test_seed_start:,}."
            ),
            "",
            "## Learning curves",
            "",
            "![Validation reward](figures/reward.png)",
            "",
            "![Forward distance](figures/distance.png)",
            "",
            "## Held-out results",
            "",
            "Values are mean ± standard deviation across training seeds.",
            "",
            "| Policy | Reward | Forward distance | Absolute lateral distance "
            "| Action saturation | Joint speed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for policy in config.policy_types:
            saturation = aggregate[policy]["action_saturation_fraction"]
            lines.append(
                f"| {policy.title()} | {formatted(policy, 'mean_total_reward')} "
                f"| {formatted(policy, 'mean_forward_distance')} m "
                f"| {formatted(policy, 'mean_absolute_lateral_distance')} m "
                f"| {saturation['mean']:.1%} ± "
                f"{saturation['standard_deviation']:.1%} "
                f"| {formatted(policy, 'mean_actuated_joint_speed')} rad/s |"
            )
        lines.extend(
            [
                "",
                "## Selected checkpoints",
                "",
                "| Policy | Training seed | Timestep | Validation reward |",
                "|---|---:|---:|---:|",
            ]
        )
        for result in run_results:
            lines.append(
                f"| {result['policy_type'].title()} "
                f"| {result['training_seed']} "
                f"| {result['selected_checkpoint_timesteps']:,} "
                f"| {result['validation_mean_total_reward']:.2f} |"
            )
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                (
                    "This report records descriptive results for this controlled "
                    "comparison. Conclusions should be limited to the tested "
                    "crawler, reward, optimizer configuration, and training budget."
                ),
                "",
                "Raw per-run results are in `results.csv`; aggregated values and "
                "the complete configuration are in `summary.json`.",
                "",
            ]
        )
        return "\n".join(lines)
